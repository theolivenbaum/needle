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

    needs_model = "generate" in spec or "encode_retrieval" in spec

    if needs_model:
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
