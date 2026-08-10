#!/usr/bin/env python3
"""Run the JAX reference and dump every stage the .NET port should reproduce.

Nothing here re-implements the model: it calls the unmodified reference in
``.reference/needle`` and records what comes back, so a mismatch on the .NET side
is a real disagreement with upstream rather than with a second transcription.

Outputs (into ``--out``):

  ``weights.safetensors``   the checkpoint, flattened to Flax paths, float32
  ``config.json``           the checkpoint's TransformerConfig
  ``case-<name>.safetensors`` per prompt: tokens, embeddings, engram keys and
                            values, the per-layer residual snapshots, the final
                            hidden state, the logits and the MTP logits, the
                            contrastive embedding, the confidence logit, and the
                            KV-cached decode logits

Usage:
    python3 scripts/parity/dump_reference.py --checkpoint needle2.pkl --out fixtures/
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".reference"))
sys.path.insert(0, os.path.dirname(__file__))

import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict

from needle.model.architecture import (SimpleAttentionNetwork, TransformerConfig,
                                       make_causal_mask, make_padding_mask)
from needle.model import decode as decode_mod

import safetensors_io

HF_REPO = "Cactus-Compute/needle2"
HF_WEIGHTS = "weights/needle2.pkl"

# Short, fixed prompts.  Token ids are chosen directly so the harness does not
# depend on having the tokenizer to hand.
CASES = {
    "short": [2, 100, 200, 300, 400, 500, 600, 700],
    "repeat": [2, 42, 42, 42, 7, 7, 42, 7, 42, 42],
    "long": [2] + [(i * 37 + 11) % 8000 + 1 for i in range(63)],
}


def fetch_checkpoint(path):
    """Return a local checkpoint path, downloading the released weights if needed."""
    if os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download

    print(f"{path} not found; downloading {HF_REPO}/{HF_WEIGHTS}...", flush=True)
    return hf_hub_download(HF_REPO, HF_WEIGHTS, repo_type="model")


def load_reference(checkpoint):
    """Load the checkpoint and build the reference module in float32."""
    with open(checkpoint, "rb") as handle:
        ckpt = pickle.load(handle)
    if ckpt.get("format_version") != 2:
        raise SystemExit(f"{checkpoint} is not a format-v2 checkpoint")

    settings = dict(ckpt["config"])
    # float32 everywhere and the explicit attention path: the .NET port computes
    # in float32, and decode.py casts to float32 too, so this is the numerics we
    # actually want to match rather than a bf16 training artefact.
    settings["dtype"] = "float32"
    settings["flash"] = False
    config = TransformerConfig(**settings)

    params = jax.tree.map(lambda a: jnp.asarray(a, jnp.float32), ckpt["params"])
    return SimpleAttentionNetwork(config), params, config, settings


def dump_weights(params, out_dir):
    flat = {"/".join(path): np.asarray(value, np.float32)
            for path, value in flatten_dict(params).items()}
    path = os.path.join(out_dir, "weights.safetensors")
    safetensors_io.save(flat, path)
    total = sum(v.size for v in flat.values())
    print(f"wrote {path}: {len(flat)} tensors, {total / 1e6:.1f}M parameters")


def dump_config(settings, config, out_dir):
    payload = dict(settings)
    payload["attn_dim"] = config.attn_dim
    payload["engram_orders"] = list(config.engram_orders)
    payload["engram_layers"] = list(config.engram_layers)
    payload["engram_heads"] = config.engram_heads
    payload["contrastive_dim"] = config.contrastive_dim
    payload["rope_theta"] = config.rope_theta
    payload["pad_token_id"] = config.pad_token_id
    with open(os.path.join(out_dir, "config.json"), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def engram_kv(model, params, tokens, mask):
    """Call the reference's own ``_engram_kv`` rather than reconstructing it."""
    def run(module, toks, msk):
        return module._engram_kv(toks, msk, False)

    return model.apply({"params": params}, tokens, mask, method=run)


def dump_case(name, ids, model, params, config, out_dir, max_new_tokens):
    tokens = jnp.asarray([ids], jnp.int32)
    mask = make_causal_mask(tokens.shape[1])
    tensors = {"tokens": np.asarray(ids, np.int32)}

    # Embeddings, straight out of the reference's own embedding table.
    embed = model.apply({"params": params}, tokens,
                        method=lambda m, t: m.embedding(t) * m.embed_scale)
    tensors["embed"] = np.asarray(embed[0], np.float32)

    keys, values = engram_kv(model, params, tokens, mask)
    for site in range(keys.shape[0]):
        tensors[f"engram{site}.k"] = np.asarray(keys[site, 0], np.float32)
        tensors[f"engram{site}.v"] = np.asarray(values[site, 0], np.float32)

    # hidden_cells is [B, T, L+1, D]: the embedding then every layer's output,
    # which is exactly the per-layer comparison the port needs.
    cells = model.apply({"params": params}, tokens,
                        method=SimpleAttentionNetwork.hidden_cells)
    tensors["cells"] = np.asarray(cells[0], np.float32)

    logits, mtp_logits = model.apply({"params": params}, tokens, return_mtp=True)
    tensors["logits"] = np.asarray(logits[0], np.float32)
    tensors["mtp_logits"] = np.asarray(mtp_logits[0], np.float32)

    _, intermediates = model.apply({"params": params}, tokens, capture_intermediates=True)
    hidden = flatten_dict(intermediates["intermediates"])[("stack", "final_norm", "__call__")]
    tensors["hidden"] = np.asarray(hidden[0][0], np.float32)

    if "contrastive_head" in params:
        embedding = model.apply({"params": params}, tokens,
                                method=SimpleAttentionNetwork.encode_contrastive)
        tensors["contrastive"] = np.asarray(embedding[0], np.float32)
    if "confidence_head" in params:
        confidence = model.apply({"params": params}, tokens,
                                 method=SimpleAttentionNetwork.forward_confidence)
        tensors["confidence"] = np.asarray(confidence, np.float32)

    # The KV-cached decoder: the deployment path, and the one the .NET generator
    # mirrors step for step.
    cached = cached_decode_logits(params, config, ids, max_new_tokens)
    tensors["decode_logits"] = cached["logits"]
    tensors["decode_tokens"] = cached["tokens"]

    path = os.path.join(out_dir, f"case-{name}.safetensors")
    safetensors_io.save(tensors, path)
    print(f"wrote {path}: {len(ids)} prompt tokens, "
          f"{len(cached['tokens'])} generated, {len(tensors)} tensors")


def cached_decode_logits(params, config, ids, max_new_tokens):
    """Greedy-decode with the reference KV cache, keeping every step's logits."""
    from needle.model.architecture import precompute_rope_freqs

    prompt = list(ids)
    max_len = min(config.max_seq_len, len(prompt) + max_new_tokens)
    head_dim = (config.attn_dim or config.d_model) // config.num_heads
    cos, sin = precompute_rope_freqs(head_dim, max_len, config.rope_theta)

    dcfg = decode_mod.decode_cfg(config, kv_window=config.kv_window)
    kc, vc = decode_mod.init_kv_cache(config, 1, max_len)
    hist = jnp.zeros((1, max_len), jnp.int32).at[0, : len(prompt)].set(
        jnp.asarray(prompt, jnp.int32))
    valid = jnp.ones((1, max_len), bool)

    logits, kc, vc = decode_mod.forward_cached(
        params, dcfg, jnp.asarray([prompt], jnp.int32), kc, vc,
        jnp.asarray(0, jnp.int32), cos, sin, None, False, hist, valid, None)

    steps = [np.asarray(logits[0, -1], np.float32)]
    generated = []
    position = len(prompt)
    nxt = int(jnp.argmax(logits[0, -1]))

    for _ in range(max_new_tokens):
        if position >= max_len:
            break
        generated.append(nxt)
        hist = hist.at[0, position].set(nxt)
        logits, kc, vc = decode_mod.forward_cached(
            params, dcfg, jnp.asarray([[nxt]], jnp.int32), kc, vc,
            jnp.asarray(position, jnp.int32), cos, sin, None, False, hist, valid, None)
        steps.append(np.asarray(logits[0, -1], np.float32))
        nxt = int(jnp.argmax(logits[0, -1]))
        position += 1

    return {"logits": np.stack(steps), "tokens": np.asarray(generated, np.int32)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="needle2.pkl",
                        help="path to needle2.pkl (downloaded from HF when absent)")
    parser.add_argument("--out", default="fixtures/parity", help="output directory")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--skip-weights", action="store_true",
                        help="only refresh the per-case dumps")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model, params, config, settings = load_reference(fetch_checkpoint(args.checkpoint))

    dump_config(settings, config, args.out)
    if not args.skip_weights:
        dump_weights(params, args.out)

    for name, ids in CASES.items():
        dump_case(name, ids, model, params, config, args.out, args.max_new_tokens)


if __name__ == "__main__":
    main()
