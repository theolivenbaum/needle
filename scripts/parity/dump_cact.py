#!/usr/bin/env python3
"""Dump every tensor of a .cact blob using the reference's own reader.

``needle.model.export.read_export`` unpacks the Cactus-Quant weights back to
float32; recording that output lets the .NET reader be checked against it
element for element, which is the only way to be sure the bit-packing, the
codebooks and the Walsh rotation are all being undone the same way.

Tensors are keyed by directory position (``t0000``, ``t0001``, …) because the
format is deliberately nameless.  The embedded tokenizer is written alongside as
a plain file.

Usage:
    python3 scripts/parity/dump_cact.py --cact needle2.cact --out fixtures/cact
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".reference"))
sys.path.insert(0, os.path.dirname(__file__))

from needle.model.export import read_export, parse_tokenizer_blob, RefTokenizer

import safetensors_io

HF_REPO = "Cactus-Compute/needle2"
HF_BLOB = "needle2.cact"

# Texts chosen to exercise every branch of the encoder: chat markers, the dummy
# prefix, multi-byte scripts that need byte fallback, JSON punctuation runs, and
# whitespace at both ends.
TOKENIZER_CASES = [
    "",
    " ",
    "hello world",
    "Hello, World!",
    "  leading and trailing  ",
    "what's it like in Lagos right now?",
    "<|im_start|>user\n<tools>[]</tools>\ndim the kitchen to 10<|im_end|>\n<|im_start|>assistant\n",
    "<think>'kitchen' -> room; 'dim to 10' -> brightness 10</think>\n"
    "<tool_call>[{\"name\":\"set_lights\",\"arguments\":{\"room\":\"kitchen\"}}]</tool_call><|im_end|>",
    "<tool_result>{\"temp_c\": 27, \"sky\": \"clear\"}</tool_result>",
    '{"amount": 1200.00, "to": "@ada_l", "memo": ""}',
    "date: 2026-07-21 Tue 14:30; locale: en-US; device: phone; battery: 62%",
    "naïve café — résumé",
    "日本語のテキスト",
    "emoji: 🌵🚀 and a snowman ☃",
    "tabs\tand\nnewlines\r\nmixed",
    "0123456789 3.14159 -42 1e-9",
    "a" * 200,
    "____----====++++",
]


def fetch(path):
    if os.path.exists(path):
        return path
    from huggingface_hub import hf_hub_download

    print(f"{path} not found; downloading {HF_REPO}/{HF_BLOB}...", flush=True)
    return hf_hub_download(HF_REPO, HF_BLOB, repo_type="model")


def dump_tokenizer_cases(tokenizer, out_dir):
    """Record what the reference encoder/decoder does, for the .NET port to match."""
    cases = []
    for text in TOKENIZER_CASES:
        ids = tokenizer.encode(text)
        cases.append({"text": text, "ids": ids, "decoded": tokenizer.decode(ids)})

    # Round-tripping every id on its own also pins down the piece table itself.
    pieces = [{"id": i, "piece": p, "type": t}
              for i, (p, t) in enumerate(zip(tokenizer.pieces, tokenizer.types))]

    path = os.path.join(out_dir, "tokenizer.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"cases": cases, "pieces": pieces}, handle, ensure_ascii=False, indent=1)
    print(f"wrote {path}: {len(cases)} encode/decode cases, {len(pieces)} pieces")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cact", default="needle2.cact")
    parser.add_argument("--out", default="fixtures/cact")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    header, tensors = read_export(fetch(args.cact))

    dumped, raw_count = {}, 0
    for index, tensor in enumerate(tensors):
        if isinstance(tensor, (bytes, bytearray)):
            raw_count += 1
            meta = parse_tokenizer_blob(tensor)
            with open(os.path.join(args.out, "tokenizer.bin"), "wb") as handle:
                handle.write(tensor)
            dump_tokenizer_cases(RefTokenizer(meta), args.out)
            print(f"t{index:04d}: raw tokenizer blob, {len(tensor)} bytes, "
                  f"{len(meta['pieces'])} pieces")
            continue
        dumped[f"t{index:04d}"] = np.ascontiguousarray(tensor, np.float32)

    safetensors_io.save(dumped, os.path.join(args.out, "tensors.safetensors"))

    with open(os.path.join(args.out, "header.json"), "w") as handle:
        json.dump({k: int(v) for k, v in header.items() if k != "codebook"}, handle, indent=2)
        handle.write("\n")

    print(f"wrote {len(dumped)} dequantised tensors (+{raw_count} raw) to {args.out}")
    print(f"header: {[f'{k}={v}' for k, v in header.items() if k != 'codebook']}")


if __name__ == "__main__":
    main()
