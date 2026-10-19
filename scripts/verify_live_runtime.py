#!/usr/bin/env python3
"""Exercise real persistent workers on one original RealSense RGB frame."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from live_runtime import LiveEstimator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--candidates", nargs="+", default=["yolo", "rtmpose", "mediapipe", "egoforce"])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "validation/live_runtime_report.json")
    args = parser.parse_args()
    capture = args.capture or sorted((ROOT / "captures").glob("*/metadata.json"))[-1].parent
    metadata = json.loads((capture / "metadata.json").read_text())
    image = cv2.imread(str(capture / "rgb.jpg"))
    if image is None:
        raise ValueError("Cannot read source capture")
    report = {"source": str(capture), "rgb_array_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
              "shape": list(image.shape), "models": {}}
    for candidate in args.candidates:
        began = time.monotonic()
        estimator = LiveEstimator(candidate)
        constructed_ms = (time.monotonic()-began)*1000
        results, durations = [], []
        try:
            for _ in range(args.repeat):
                began = time.monotonic()
                result = estimator.infer(image, intrinsics=metadata["depth_intrinsics"])
                durations.append((time.monotonic()-began)*1000)
                results.append(result)
            summary = {"backend": estimator.backend, "model": estimator.model,
                       "constructor_ms": constructed_ms, "load_ms": estimator.load_ms,
                       "roundtrip_ms": durations,
                       "worker_inference_ms": [v["inference_ms"] for v in results],
                       "instances": [len(v["instances"]) for v in results],
                       "last_result": results[-1]}
        finally:
            began = time.monotonic()
            process = estimator._process
            estimator.close()
            close_ms = (time.monotonic()-began)*1000
        summary.update(close_ms=close_ms, process_exited=process.poll() is not None)
        report["models"][candidate] = summary
        print(json.dumps({k: v for k, v in summary.items() if k != "last_result"}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
