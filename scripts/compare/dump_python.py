"""Dump deterministic outputs from the Python `needle` package for cross-runtime
comparison against the C# .NET port.  Mirrors `needle dump-compare` in
src/Needle.Cli/CliEntry.cs.

Usage:
    python scripts/compare/dump_python.py \
        --checkpoint <pkl_or_safetensors_path> \
        --tokenizer  <sentencepiece.model> \
        --spec       scripts/compare/spec.json \
        --out        outputs/python.json

The output JSON has the same schema the C# side produces, so
`scripts/compare/compare.py` can diff them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _round_list(arr, prec: int):
    """Round a numpy array element-wise to `prec` decimal places and return
    a nested Python list of floats."""
    rounded = arr.round(prec)
    return rounded.tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pkl checkpoint (Python format).")
    parser.add_argument("--tokenizer", required=True,
                        help="Path to SentencePiece .model file.")
    parser.add_argument("--spec", required=True,
                        help="Path to comparison spec JSON.")
    parser.add_argument("--out", required=True,
                        help="Path to write output JSON.")
    parser.add_argument("--float-precision", type=int, default=8,
                        help="Decimal places of precision for embedding dumps "
                             "(default: 8).")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.exists():
        sys.stderr.write(f"error: spec not found: {spec_path}\n")
        return 1

    with open(spec_path) as f:
        spec = json.load(f)

    # ── Lazy imports — only what each section needs ─────────────────────────
    # Tokenizer is always required.
    from needle.dataset.tokenizer import NeedleTokenizer
    tokenizer = NeedleTokenizer(args.tokenizer)

    output: dict = {}

    # ── Tokenization ───────────────────────────────────────────────────────
    if "tokenize" in spec:
        out_tok = []
        for item in spec["tokenize"]:
            text = item["text"]
            ids  = list(tokenizer.encode(text))
            out_tok.append({
                "text":    text,
                "ids":     ids,
                "decoded": tokenizer.decode(ids),
            })
        output["tokenize"] = out_tok

    # ── Tool-name normalization (pure-function check) ──────────────────────
    if "tool_normalize" in spec:
        from needle.model.run import normalize_tools, restore_tool_names
        out_norm = []
        for tools_json in spec["tool_normalize"]:
            norm, name_map = normalize_tools(tools_json)
            out_norm.append({
                "input":     tools_json,
                "tools":     norm,
                "name_map":  name_map,
            })
        output["tool_normalize"] = out_norm

    needs_model = (
        "generate"         in spec or
        "generate_batch"   in spec or
        "encode_retrieval" in spec or
        "forward_loss"     in spec or
        "quantize"         in spec
    )

    if needs_model:
        import jax
        import jax.numpy as jnp
        import numpy as np  # noqa: F401  (used via JAX outputs below)
        from needle.model.architecture import SimpleAttentionNetwork
        from needle.model.run import (
            load_checkpoint, generate, encode_for_retrieval,
        )

        params, config = load_checkpoint(args.checkpoint)
        model = SimpleAttentionNetwork(config)

        # ── Generation ─────────────────────────────────────────────────────
        if "generate" in spec:
            out_gen = []
            for item in spec["generate"]:
                text = generate(
                    model, params, tokenizer,
                    query        = item["query"],
                    tools        = item.get("tools", "[]"),
                    max_gen_len  = item.get("max_gen_len", 128),
                    max_enc_len  = item.get("max_enc_len", 1024),
                    stream       = False,
                    normalize    = item.get("normalize", False),
                    constrained  = item.get("constrained", False),
                )
                # Re-encode the produced text so both sides emit canonical ID lists.
                ids = list(tokenizer.encode(text))
                pieces = [tokenizer.sp.id_to_piece(i) for i in ids]
                out_gen.append({
                    "id":     item.get("id", ""),
                    "query":  item["query"],
                    "text":   text,
                    "ids":    ids,
                    "pieces": pieces,
                })
            output["generate"] = out_gen

        # ── Batched generation ─────────────────────────────────────────────
        if "generate_batch" in spec:
            from needle.model.run import generate_batch
            out_gb = []
            for entry in spec["generate_batch"]:
                items = entry["items"]
                queries = [it["query"] for it in items]
                tools   = [it["tools"] for it in items]
                preds = generate_batch(
                    model, params, tokenizer,
                    queries, tools,
                    max_gen_len = entry.get("max_gen_len", 128),
                    max_enc_len = entry.get("max_enc_len", 1024),
                    normalize   = entry.get("normalize", False),
                    constrained = entry.get("constrained", False),
                )
                results = []
                for q, t, p in zip(queries, tools, preds):
                    results.append({
                        "query": q, "tools": t, "text": p,
                        "ids":   list(tokenizer.encode(p)),
                    })
                out_gb.append({"id": entry.get("id", ""), "results": results})
            output["generate_batch"] = out_gb

        # ── Retrieval embeddings ───────────────────────────────────────────
        if "encode_retrieval" in spec:
            r       = spec["encode_retrieval"]
            texts   = list(r["texts"])
            max_len = r.get("max_len", 256)
            embs    = encode_for_retrieval(
                model, params, tokenizer, texts, max_len=max_len,
            )
            # encode_for_retrieval returns a JAX array; convert to numpy.
            import numpy as np
            embs_np = np.asarray(embs, dtype=np.float64)
            output["encode_retrieval"] = {
                "shape":      list(embs_np.shape),
                "texts":      texts,
                "embeddings": _round_list(embs_np, args.float_precision),
            }

        # ── Forward-loss (training-math parity) ────────────────────────────
        if "forward_loss" in spec:
            import numpy as np
            import optax
            from needle.model.architecture import (
                make_packing_mask, make_causal_packing_mask, make_cross_packing_mask,
            )

            fl = spec["forward_loss"]
            src         = jnp.asarray(fl["src_tokens"],     dtype=jnp.int32)
            tgt_in      = jnp.asarray(fl["tgt_in_tokens"],  dtype=jnp.int32)
            tgt_out     = jnp.asarray(fl["tgt_out_tokens"], dtype=jnp.int32)
            loss_mask   = jnp.asarray(fl["loss_mask"],      dtype=jnp.int32)
            enc_seg_ids = jnp.asarray(fl["enc_seg_ids"],    dtype=jnp.int32)
            dec_seg_ids = jnp.asarray(fl["dec_seg_ids"],    dtype=jnp.int32)

            src_mask   = make_packing_mask(enc_seg_ids)
            tgt_mask   = make_causal_packing_mask(dec_seg_ids)
            cross_mask = make_cross_packing_mask(enc_seg_ids, dec_seg_ids)

            logits = model.apply(
                {"params": params},
                src, tgt_in,
                src_mask=src_mask, tgt_mask=tgt_mask, cross_mask=cross_mask,
            )
            logits_f32 = logits.astype(jnp.float32)

            # Default token-class weights (base, name, value, key).
            weight_map = jnp.asarray([1.0, 3.0, 2.0, 1.5], dtype=jnp.float32)
            token_weights = weight_map[loss_mask.astype(jnp.int32)]
            padding_mask  = (dec_seg_ids > 0).astype(jnp.float32)
            mask          = token_weights * padding_mask
            num_tokens    = jnp.maximum(jnp.sum(padding_mask), 1.0)

            ce_per_pos = optax.softmax_cross_entropy_with_integer_labels(
                logits_f32, tgt_out,
            )
            ce_loss = float(jnp.sum(ce_per_pos * mask) / num_tokens)
            z_loss  = float(1e-4 * jnp.mean(jax.nn.logsumexp(logits_f32, axis=-1) ** 2))

            # Also a small probe of the raw logits so we can numerically diff
            # the forward output, not just the scalar.
            logits_np = np.asarray(logits_f32, dtype=np.float64)
            B, T, V = logits_np.shape
            probe = []
            for b in range(B):
                for t in range(T):
                    row = logits_np[b, t, :8]  # first 8 vocab positions
                    probe.append([round(float(x), args.float_precision) for x in row])

            output["forward_loss"] = {
                "shape":         [B, T, V],
                "ce_loss":       round(ce_loss, args.float_precision),
                "z_loss":        round(z_loss,  args.float_precision),
                "total_loss":    round(ce_loss + z_loss, args.float_precision),
                "logits_probe":  probe,  # [B*T, 8]
            }

        # ── Quantization ───────────────────────────────────────────────────
        if "quantize" in spec:
            import numpy as np
            import pickle as _pkl
            from needle.model.quantize import _fake_quantize_int4, _fake_quantize_int8

            q = spec["quantize"]
            key   = q["tensor_key"]
            r0,r1 = q.get("rows", [0, 32])
            c0,c1 = q.get("cols", [0, 32])
            gs    = q.get("group_size", 32)
            prec  = q.get("precision", "int4")

            # Re-load the raw .pkl directly, bypassing the bf16 cast that
            # load_checkpoint applies.  This way Python and C# both quantize
            # the same fp32-widened source tensor (matching the safetensors
            # the converter writes for the .NET side).  Otherwise bf16's
            # 7-bit mantissa changes the per-group scale just enough that
            # neighbouring weights round to different int4 grid cells,
            # producing a max-abs delta of ~0.3 that obscures real bugs.
            with open(args.checkpoint, "rb") as _f:
                raw_params = _pkl.load(_f)["params"]

            def _resolve(tree, dotted):
                if dotted == "embedding.weight":
                    return tree["embedding"]["embedding"]
                raise KeyError(f"Don't know how to resolve {dotted} in Python tree.")

            w_np = np.asarray(_resolve(raw_params, key), dtype=np.float32)
            sub  = jnp.asarray(w_np[r0:r1, c0:c1], dtype=jnp.float32)

            qfn = _fake_quantize_int8 if prec == "int8" else _fake_quantize_int4
            q_out = np.asarray(qfn(sub, group_size=gs), dtype=np.float64)

            output["quantize"] = {
                "tensor_key": key,
                "rows":       [r0, r1],
                "cols":       [c0, c1],
                "group_size": gs,
                "precision":  prec,
                "shape":      list(q_out.shape),
                "values":     _round_list(q_out, args.float_precision),
            }

    output["meta"] = {
        "side":            "python",
        "checkpoint":      args.checkpoint,
        "tokenizer":       args.tokenizer,
        "float_precision": args.float_precision,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
