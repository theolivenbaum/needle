"""Minimal safetensors writer/reader (numpy only).

The parity harness moves two things between the JAX reference and the .NET port:
the checkpoint itself and every traced intermediate.  safetensors is a JSON
header over a flat data block, so both sides can read it without pulling in a
dependency -- `Needle.Weights.Safetensors` is the C# counterpart of this file.
"""

import json
import struct

import numpy as np

_DTYPES = {
    "float32": "F32",
    "float16": "F16",
    "float64": "F64",
    "int64": "I64",
    "int32": "I32",
}


def save(tensors, path):
    """Write ``{name: ndarray}`` to ``path``. Arrays are stored as-is."""
    items = sorted(tensors.items())
    header, offset = {}, 0
    blobs = []
    for name, array in items:
        array = np.ascontiguousarray(array)
        dtype = _DTYPES.get(str(array.dtype))
        if dtype is None:
            raise TypeError(f"{name}: unsupported dtype {array.dtype}")
        data = array.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        blobs.append(data)

    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        for blob in blobs:
            handle.write(blob)


_READ_DTYPES = {v: k for k, v in _DTYPES.items()}


def load(path):
    """Read ``path`` into ``{name: ndarray}``."""
    with open(path, "rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len).decode("utf-8"))
        blob = handle.read()

    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        dtype = np.dtype(_READ_DTYPES[meta["dtype"]])
        out[name] = np.frombuffer(blob[start:end], dtype=dtype).reshape(meta["shape"]).copy()
    return out
