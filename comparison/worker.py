"""Line-oriented model worker. Heavy dependencies stay out of the camera server."""
from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PREFIX = "COMPARE_RESULT "
ADAPTERS = {
    "mediapipe": ("comparison.mediapipe_adapter", "MediaPipeAdapter"),
    "rtmpose": ("comparison.rtmpose_adapter", "RTMPoseAdapter"),
    "egoforce": ("comparison.egoforce_adapter", "EgoForceAdapter"),
}


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def emit(value):
    print(PREFIX + json.dumps(value, ensure_ascii=True, allow_nan=False,
                              default=json_default), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=ADAPTERS, required=True)
    args = parser.parse_args()
    adapter = None
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            module, name = ADAPTERS[args.candidate]
            adapter = getattr(importlib.import_module(module), name)()
        emit({"ready": True, "load_ms": (time.perf_counter()-started)*1000})
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit({"ready": False, "error": str(exc)})
        return
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request.get("op") == "close":
                    break
                image_path = Path(request["image"]).resolve()
                if not image_path.is_relative_to((ROOT / "comparisons").resolve()):
                    raise ValueError("Comparison input must be inside comparisons/")
                bgr = cv2.imdecode(np.frombuffer(image_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("Cannot decode comparison input")
                kwargs = {}
                parameters = inspect.signature(adapter.infer).parameters
                for key in ("intrinsics", "roi", "hand_hints"):
                    if key in parameters:
                        kwargs[key] = request.get(key)
                if hasattr(adapter, "roi") and "roi" not in parameters:
                    adapter.roi = request.get("roi")
                started = time.perf_counter()
                with contextlib.redirect_stdout(sys.stderr):
                    result = adapter.infer(bgr, **kwargs)
                elapsed = (time.perf_counter()-started)*1000
                emit({"ok": True, "inference_ms": elapsed, "result": result})
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                emit({"ok": False, "error": str(exc)})
    finally:
        if adapter is not None and hasattr(adapter, "close"):
            with contextlib.redirect_stdout(sys.stderr):
                adapter.close()


if __name__ == "__main__":
    main()
