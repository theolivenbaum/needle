"""Compare two dump JSON files produced by the Python and C# sides.

Usage:
    python scripts/compare/compare.py \
        --python outputs/python.json \
        --csharp outputs/csharp.json \
        [--token-strict] \
        [--emb-tol 1e-3]

Exit code is 0 if all enabled checks pass under tolerance, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _max_abs(a, b) -> float:
    """Max absolute error between two nested-list arrays of equal shape."""
    if isinstance(a, list):
        return max(_max_abs(x, y) for x, y in zip(a, b))
    return abs(float(a) - float(b))


def _l2_per_row(a, b):
    """Per-row L2 errors for 2-D nested-list arrays."""
    out = []
    for ra, rb in zip(a, b):
        s = 0.0
        for x, y in zip(ra, rb):
            d = float(x) - float(y)
            s += d * d
        out.append(math.sqrt(s))
    return out


def _check_tokenize(py_arr, cs_arr, fail) -> None:
    if py_arr is None and cs_arr is None:
        return
    if py_arr is None or cs_arr is None:
        fail.append("tokenize section present on only one side")
        return
    if len(py_arr) != len(cs_arr):
        fail.append(f"tokenize length mismatch: py={len(py_arr)} cs={len(cs_arr)}")
        return
    mismatches = 0
    for i, (p, c) in enumerate(zip(py_arr, cs_arr)):
        if p["text"] != c["text"]:
            mismatches += 1
            fail.append(f"tokenize[{i}] text mismatch: py={p['text']!r} cs={c['text']!r}")
            continue
        if p["ids"] != c["ids"]:
            mismatches += 1
            fail.append(
                f"tokenize[{i}] ids differ for text={p['text']!r}:\n"
                f"    py={p['ids']}\n"
                f"    cs={c['ids']}")
    print(f"  tokenize: {len(py_arr)} cases, {mismatches} mismatch(es)")


def _first_diff(a, b):
    """Index of first differing element in two lists, or None."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        return min(len(a), len(b)), None, None
    return None


def _check_generate(py_arr, cs_arr, fail, token_strict: bool) -> None:
    if py_arr is None and cs_arr is None:
        return
    if py_arr is None or cs_arr is None:
        fail.append("generate section present on only one side")
        return
    if len(py_arr) != len(cs_arr):
        fail.append(f"generate length mismatch: py={len(py_arr)} cs={len(cs_arr)}")
        return

    for i, (p, c) in enumerate(zip(py_arr, cs_arr)):
        tag = p.get("id") or c.get("id") or f"#{i}"
        py_ids = p["ids"]; cs_ids = c["ids"]
        if py_ids == cs_ids:
            print(f"  generate[{tag}]: tokens match ({len(py_ids)} ids)")
            continue

        diff = _first_diff(py_ids, cs_ids)
        msg = (f"generate[{tag}] ids differ at index {diff[0]} "
               f"(py={diff[1]} cs={diff[2]}); len py={len(py_ids)} cs={len(cs_ids)}")
        print(f"  generate[{tag}]: DIFFERS — {msg}")
        print(f"    py text:  {p['text']!r}")
        print(f"    cs text:  {c['text']!r}")
        if token_strict:
            fail.append(msg)


def _check_tool_normalize(py_arr, cs_arr, fail) -> None:
    if py_arr is None and cs_arr is None:
        return
    if py_arr is None or cs_arr is None:
        fail.append("tool_normalize section present on only one side")
        return
    if len(py_arr) != len(cs_arr):
        fail.append(f"tool_normalize length mismatch: py={len(py_arr)} cs={len(cs_arr)}")
        return
    mismatches = 0
    for i, (p, c) in enumerate(zip(py_arr, cs_arr)):
        if p["tools"] != c["tools"]:
            mismatches += 1
            fail.append(f"tool_normalize[{i}] tools differ:\n    py={p['tools']}\n    cs={c['tools']}")
        if p["name_map"] != c["name_map"]:
            mismatches += 1
            fail.append(f"tool_normalize[{i}] name_map differs:\n    py={p['name_map']}\n    cs={c['name_map']}")
    print(f"  tool_normalize: {len(py_arr)} cases, {mismatches} mismatch(es)")


def _check_generate_batch(py_arr, cs_arr, fail, token_strict: bool) -> None:
    if py_arr is None and cs_arr is None:
        return
    if py_arr is None or cs_arr is None:
        fail.append("generate_batch section present on only one side")
        return
    if len(py_arr) != len(cs_arr):
        fail.append(f"generate_batch length mismatch: py={len(py_arr)} cs={len(cs_arr)}")
        return
    for i, (p, c) in enumerate(zip(py_arr, cs_arr)):
        tag = p.get("id") or c.get("id") or f"#{i}"
        py_r = p["results"]; cs_r = c["results"]
        if len(py_r) != len(cs_r):
            fail.append(f"generate_batch[{tag}] length mismatch: py={len(py_r)} cs={len(cs_r)}")
            continue
        diffs = 0
        for k, (pp, cc) in enumerate(zip(py_r, cs_r)):
            if pp["ids"] != cc["ids"]:
                diffs += 1
                diff = _first_diff(pp["ids"], cc["ids"])
                msg = (f"generate_batch[{tag}][{k}] ids differ at index {diff[0]} "
                       f"(py={diff[1]} cs={diff[2]}); len py={len(pp['ids'])} cs={len(cc['ids'])}")
                if token_strict:
                    fail.append(msg)
                print(f"  generate_batch[{tag}][{k}]: DIFFERS — {msg}")
                print(f"    py text:  {pp['text']!r}")
                print(f"    cs text:  {cc['text']!r}")
        if diffs == 0:
            print(f"  generate_batch[{tag}]: {len(py_r)} items match")


def _check_forward_loss(py_obj, cs_obj, fail, loss_tol: float, logits_tol: float) -> None:
    if py_obj is None and cs_obj is None:
        return
    if py_obj is None or cs_obj is None:
        fail.append("forward_loss section present on only one side")
        return

    # Scalar losses
    for key in ("ce_loss", "z_loss", "total_loss"):
        d = abs(float(py_obj[key]) - float(cs_obj[key]))
        flag = "OK" if d <= loss_tol else "DIFF"
        print(f"  forward_loss.{key}: py={py_obj[key]} cs={cs_obj[key]}  |Δ|={d:.3e}  {flag}")
        if d > loss_tol:
            fail.append(f"forward_loss.{key} |Δ|={d:.3e} exceeds tol {loss_tol:.3e}")

    # Logits probe
    py_p = py_obj.get("logits_probe"); cs_p = cs_obj.get("logits_probe")
    if py_p is not None and cs_p is not None:
        max_abs = _max_abs(py_p, cs_p)
        print(f"  forward_loss.logits_probe: max_abs={max_abs:.3e}")
        if max_abs > logits_tol:
            fail.append(f"forward_loss.logits_probe max_abs {max_abs:.3e} exceeds tol {logits_tol:.3e}")


def _check_quantize(py_obj, cs_obj, fail, tol: float) -> None:
    if py_obj is None and cs_obj is None:
        return
    if py_obj is None or cs_obj is None:
        fail.append("quantize section present on only one side")
        return
    if py_obj["shape"] != cs_obj["shape"]:
        fail.append(f"quantize shape mismatch: py={py_obj['shape']} cs={cs_obj['shape']}")
        return
    max_abs = _max_abs(py_obj["values"], cs_obj["values"])
    print(f"  quantize ({py_obj.get('precision')}): shape={py_obj['shape']} max_abs={max_abs:.3e}")
    if max_abs > tol:
        fail.append(f"quantize max abs error {max_abs:.3e} exceeds tol {tol:.3e}")


def _check_retrieval(py_obj, cs_obj, fail, emb_tol: float) -> None:
    if py_obj is None and cs_obj is None:
        return
    if py_obj is None or cs_obj is None:
        fail.append("encode_retrieval section present on only one side")
        return
    if py_obj["shape"] != cs_obj["shape"]:
        fail.append(f"encode_retrieval shape mismatch: py={py_obj['shape']} cs={cs_obj['shape']}")
        return

    py = py_obj["embeddings"]; cs = cs_obj["embeddings"]
    max_abs = _max_abs(py, cs)
    rows    = _l2_per_row(py, cs)
    max_row = max(rows) if rows else 0.0
    mean_row = (sum(rows) / len(rows)) if rows else 0.0

    print(f"  encode_retrieval: shape={py_obj['shape']} "
          f"max_abs={max_abs:.3e} max_row_l2={max_row:.3e} mean_row_l2={mean_row:.3e}")

    if max_abs > emb_tol:
        fail.append(f"encode_retrieval max abs error {max_abs:.3e} exceeds tol {emb_tol:.3e}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, help="Python-side JSON dump.")
    parser.add_argument("--csharp", required=True, help="C#-side JSON dump.")
    parser.add_argument("--token-strict", action="store_true",
                        help="Treat generate-token mismatches as failures (default: report only).")
    parser.add_argument("--emb-tol", type=float, default=1e-3,
                        help="Max absolute element-wise tolerance for retrieval embeddings (default: 1e-3).")
    parser.add_argument("--loss-tol", type=float, default=2e-1,
                        help="Max absolute tolerance for forward_loss scalars (default: 2e-1, absorbs Python-bf16 vs .NET-fp32 forward-pass drift).")
    parser.add_argument("--logits-tol", type=float, default=1.0,
                        help="Max absolute tolerance for forward_loss.logits_probe (default: 1.0; logits range over many orders, bf16 vs fp32 drift compounds across 20 layers).")
    parser.add_argument("--quant-tol", type=float, default=1e-3,
                        help="Max absolute tolerance for quantize values (default: 1e-3).")
    args = parser.parse_args()

    py = _load(args.python)
    cs = _load(args.csharp)
    print(f"python: {args.python}  (side={py.get('meta', {}).get('side')})")
    print(f"csharp: {args.csharp}  (side={cs.get('meta', {}).get('side')})")

    fail: list[str] = []
    _check_tokenize(py.get("tokenize"),               cs.get("tokenize"),               fail)
    _check_tool_normalize(py.get("tool_normalize"),   cs.get("tool_normalize"),         fail)
    _check_generate(py.get("generate"),               cs.get("generate"),               fail, args.token_strict)
    _check_generate_batch(py.get("generate_batch"),   cs.get("generate_batch"),         fail, args.token_strict)
    _check_retrieval(py.get("encode_retrieval"),      cs.get("encode_retrieval"),       fail, args.emb_tol)
    _check_forward_loss(py.get("forward_loss"),       cs.get("forward_loss"),           fail, args.loss_tol, args.logits_tol)
    _check_quantize(py.get("quantize"),               cs.get("quantize"),               fail, args.quant_tol)

    if fail:
        print()
        print("FAILURES:")
        for f in fail:
            print(f"  - {f}")
        return 1

    print()
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
