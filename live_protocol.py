"""Lossless raw-BGR framing for persistent, isolated live model processes."""
from __future__ import annotations

import json
import math
import struct

import numpy as np

PREFIX = "LIVE_RESULT "
MAX_HEADER = 64 * 1024
MAX_IMAGE = 64 * 1024 * 1024


def encode_frame(bgr, *, sequence, intrinsics=None, roi=None, confidence=.35, hand_hints=None):
    if not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected uint8 BGR frame (H,W,3)")
    if min(bgr.shape[:2]) < 1 or bgr.nbytes > MAX_IMAGE:
        raise ValueError("Live frame has invalid or excessive dimensions")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be a finite number between 0 and 1")
    array = np.ascontiguousarray(bgr)
    header = {"sequence": int(sequence), "shape": list(array.shape), "nbytes": array.nbytes,
              "intrinsics": intrinsics, "roi": roi, "confidence": float(confidence),
              "hand_hints": hand_hints}
    encoded = json.dumps(header, ensure_ascii=True, allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_HEADER:
        raise ValueError("Live request metadata is too large")
    return struct.pack("!I", len(encoded)) + encoded, memoryview(array).cast("B")


def read_exact(stream, size, *, allow_eof=False):
    pieces, remaining = [], size
    while remaining:
        block = stream.read(remaining)
        if not block:
            if allow_eof and remaining == size:
                return None
            raise EOFError("Truncated live frame")
        pieces.append(block)
        remaining -= len(block)
    return b"".join(pieces)


def read_frame(stream):
    prefix = read_exact(stream, 4, allow_eof=True)
    if prefix is None:
        return None
    length = struct.unpack("!I", prefix)[0]
    if not 1 <= length <= MAX_HEADER:
        raise ValueError("Invalid live header length")
    header = json.loads(read_exact(stream, length))
    if not isinstance(header, dict):
        raise ValueError("Live header must be an object")
    shape = header.get("shape")
    if (not isinstance(shape, list) or len(shape) != 3 or shape[2] != 3 or
            any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape)):
        raise ValueError("Invalid live frame shape")
    size = math.prod(shape)
    if size > MAX_IMAGE or header.get("nbytes") != size:
        raise ValueError("Invalid live frame byte count")
    image = np.frombuffer(read_exact(stream, size), dtype=np.uint8).reshape(shape)
    return header, image


def write_all(stream, data):
    view = memoryview(data).cast("B")
    while view:
        count = stream.write(view)
        if not count:
            raise BrokenPipeError("Live model input pipe closed")
        view = view[count:]
