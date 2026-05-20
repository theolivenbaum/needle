"""Convert a Python .pkl Flax checkpoint to a .safetensors file that the
.NET runtime (`WeightLoader.LoadSafetensors`) can consume.

Two structural differences between Flax and TorchSharp need to be reconciled:

1. **Naming.**  Flax stores Dense weights as `kernel` and module paths use
   `/` separators (e.g. `encoder/layers/EncoderBlock_0/self_attn/q_proj/kernel`).
   TorchSharp uses `weight` and `.` separators (e.g.
   `encoder.layer_0.self_attn.q_proj.weight`).

2. **Layout.**  Flax `nn.Dense` kernels are stored as `[in_features, out_features]`;
   PyTorch Linear weights are stored as `[out_features, in_features]`.  Every
   converted kernel is therefore transposed.

3. **Scan stacking.**  The encoder and decoder layer stacks are produced by
   `nn.scan`, which stacks the params of one iteration along a leading
   `num_layers` axis.  Each scanned tensor is unstacked here into
   per-layer tensors `layer_0`, `layer_1`, ….

Usage:
    python scripts/compare/convert_pkl_to_safetensors.py \
        --input  checkpoint.pkl \
        --output checkpoint.safetensors

The converter is intentionally strict: any unknown key triggers an error,
since silently dropping a parameter would mask drift.
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Flax key → TorchSharp key rewriting
# ---------------------------------------------------------------------------

# Maps the auto-numbered `ZCRMSNorm_<n>` inside an EncoderBlock / DecoderBlock
# to the explicit C# name.  Ordering inside the block is what matters.
_ENCODER_NORMS  = {0: "attn_norm", 1: "ffn_norm"}
_DECODER_NORMS  = {0: "self_attn_norm", 1: "cross_attn_norm", 2: "ffn_norm"}


def _rewrite_block_norm(stack: str, name: str) -> str:
    """Translate `ZCRMSNorm_<n>` to the C# name based on its position
    inside the encoder or decoder block."""
    m = re.fullmatch(r"ZCRMSNorm_(\d+)", name)
    if not m:
        return name
    idx = int(m.group(1))
    if stack == "encoder":
        return _ENCODER_NORMS.get(idx, name)
    return _DECODER_NORMS.get(idx, name)


def _flax_path_to_cs(stack: str, parts: tuple[str, ...]) -> str:
    """Rewrite the in-block portion of a Flax path to the C# convention.

    `parts` is the path *inside* the encoder/decoder body (so it doesn't
    include `encoder` / `decoder` itself or the `layer_N` prefix).
    """
    # Strip the leading EncoderBlock_<n> / DecoderBlock_<n> auto-naming
    # introduced by nn.compact when wrapping the body in _Encoder/DecoderScanBody.
    if parts and re.fullmatch(r"(Encoder|Decoder)Block_\d+", parts[0]):
        parts = parts[1:]

    parts = tuple(_rewrite_block_norm(stack, p) for p in parts)

    # MultiHeadAttention's q_norm/k_norm contain ZCRMSNorm scale: in Flax it's
    # `q_norm/scale`, in C# `q_norm.scale`; no rename needed.

    # Dense kernels: rename `kernel` to `weight`.  This is the last component.
    if parts and parts[-1] == "kernel":
        parts = (*parts[:-1], "weight")

    return ".".join(parts)


def _flatten(tree, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    """Flatten a nested dict into {path_tuple: leaf}."""
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(_flatten(v, prefix + (str(k),)))
    else:
        out[prefix] = tree
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",  required=True, help="Path to .pkl checkpoint.")
    parser.add_argument("--output", required=True, help="Path to write .safetensors.")
    parser.add_argument("--dtype",  default="float32",
                        choices=["float32", "bfloat16", "float16"],
                        help="Tensor dtype to write (default: float32).")
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        from safetensors.torch import save_file
    except ImportError as e:
        sys.stderr.write(
            f"error: missing dependency ({e}). "
            "Install with: pip install numpy torch safetensors\n")
        return 1

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    params = data["params"] if isinstance(data, dict) and "params" in data else data
    flat   = _flatten(params)

    dtype_map = {
        "float32":  torch.float32,
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
    }
    target_dtype = dtype_map[args.dtype]

    out_tensors: dict[str, "torch.Tensor"] = {}
    skipped: list[tuple[tuple[str, ...], str]] = []

    for path, leaf in flat.items():
        arr = np.asarray(leaf)
        t   = torch.from_numpy(arr.astype(np.float32, copy=False))

        # --- Top-level singletons -------------------------------------------------
        if path == ("embedding", "embedding"):
            out_tensors["embedding.weight"] = t.to(target_dtype)
            continue
        if path == ("log_temp",):
            # Reshape () → (1,) to match the C# Parameter(zeros(1)).
            out_tensors["log_temp"] = t.reshape(1).to(target_dtype)
            continue
        if path[0] in ("contrastive_hidden", "contrastive_proj"):
            tail = ".".join(path[1:])
            if tail == "kernel":
                out_tensors[f"{path[0]}.weight"] = t.t().contiguous().to(target_dtype)
            elif tail == "bias":
                out_tensors[f"{path[0]}.bias"]   = t.to(target_dtype)
            else:
                skipped.append((path, f"unrecognised {path[0]} param"))
            continue

        # --- Encoder / decoder ----------------------------------------------------
        if path[0] not in ("encoder", "decoder"):
            skipped.append((path, "unknown top-level module"))
            continue

        stack    = path[0]
        in_block = path[1:]

        # `final_norm` lives at encoder/decoder top level.
        if in_block[:1] in (("ZCRMSNorm_0",), ("final_norm",)) and len(in_block) == 2 \
                and in_block[1] == "scale" and len(t.shape) == 1:
            out_tensors[f"{stack}.final_norm.scale"] = t.to(target_dtype)
            continue
        if in_block[:1] == ("ZCRMSNorm_0",) and stack == "decoder" \
                and len(in_block) == 2 and in_block[1] == "scale":
            # Decoder final norm in some checkpoints.
            out_tensors[f"{stack}.final_norm.scale"] = t.to(target_dtype)
            continue

        if in_block[:1] != ("layers",):
            skipped.append((path, "not under `layers`"))
            continue

        body_path = in_block[1:]  # path inside one scan body
        # Determine number of stacked layers from the leading axis.
        if not isinstance(leaf, (np.ndarray, list, tuple)):
            skipped.append((path, "scanned leaf without leading axis"))
            continue
        if t.ndim == 0:
            skipped.append((path, "scalar where stack expected"))
            continue
        num_layers = int(t.shape[0])

        cs_tail = _flax_path_to_cs(stack, body_path)
        # Dense kernels are still in [in, out] inside Flax; they live at depth
        # ending in `weight` after rewriting.  Transpose only those that came
        # from `kernel`.
        is_kernel = body_path[-1] == "kernel"

        for li in range(num_layers):
            slice_t = t[li]
            if is_kernel:
                slice_t = slice_t.t().contiguous()
            # Scalar gate params (attn_gate, self_attn_gate, cross_attn_gate)
            # are stored as () in Flax but as Parameter(zeros(1)) — i.e. shape
            # (1,) — in the .NET model, so reshape singletons up to a 1-D
            # tensor here.
            if slice_t.ndim == 0:
                slice_t = slice_t.reshape(1)
            # C# uses `_layers.<i>` (TorchSharp ModuleList registers under the
            # backing-field name plus index), not `layer_<i>`.
            key = f"{stack}._layers.{li}.{cs_tail}"
            out_tensors[key] = slice_t.to(target_dtype)

    if skipped:
        print("[warn] skipped params:", file=sys.stderr)
        for path, reason in skipped:
            print(f"  /{'/'.join(path)} : {reason}", file=sys.stderr)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_file(out_tensors, args.output)
    print(f"wrote {len(out_tensors)} tensor(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
