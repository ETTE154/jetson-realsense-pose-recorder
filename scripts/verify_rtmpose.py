#!/usr/bin/env python3
"""Run the actual RTMPose adapter on saved RealSense frames; write evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from comparison.rtmpose_adapter import RTMPoseAdapter, KEYPOINT_NAMES, EDGES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "validation" / "rtmpose")
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    images = args.images or sorted((ROOT / "captures").glob("*/rgb.jpg"))[-2:]
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    adapter = RTMPoseAdapter()
    report = {"model": adapter.model, "backend": adapter.backend,
              "model_path": adapter.model_path,
              "load_ms": (time.perf_counter() - started) * 1000,
              "threshold": adapter.confidence, "results": [],
              "method": "one full-image ROI; no person detector; measured latency excludes load/build and includes preprocess/infer/decode",
              "accuracy": "No ground truth. Visible-keypoint count is a model-specific score threshold, not an accuracy metric."}
    try:
        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot read image: {path}")
            adapter.infer(image)  # separate one warm-up run
            durations = []
            for _ in range(args.iterations):
                started = time.perf_counter()
                result = adapter.infer(image)
                durations.append((time.perf_counter() - started) * 1000)
            instance = result["instances"][0]
            rendered = image.copy()
            points = instance["keypoints"]
            for a, b in EDGES:
                if min(points[a][2], points[b][2]) >= adapter.confidence:
                    cv2.line(rendered, tuple(round(v) for v in points[a][:2]),
                             tuple(round(v) for v in points[b][:2]), (60, 230, 70), 2)
            for index, point in enumerate(points):
                if point[2] >= adapter.confidence:
                    xy = tuple(round(v) for v in point[:2])
                    cv2.circle(rendered, xy, 3, (0, 190, 255), -1)
                    cv2.putText(rendered, KEYPOINT_NAMES[index], xy, cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 190, 255), 1)
            title = f"RTMPose-m ROI | {instance['visible_keypoints']}/17 >= {adapter.confidence:.2f} | {np.median(durations):.1f} ms"
            cv2.rectangle(rendered, (0, 0), (rendered.shape[1], 28), (15, 20, 25), -1)
            cv2.putText(rendered, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
            output = args.output / f"{path.parent.name}_rtmpose.jpg"
            cv2.imwrite(str(output), rendered)
            sample = {"input": str(path), "overlay": str(output),
                      "visible_keypoints": instance["visible_keypoints"],
                      "latency_ms_median": float(np.median(durations)),
                      "latency_ms_p95": float(np.percentile(durations, 95)),
                      "iterations": args.iterations, "prediction": result}
            report["results"].append(sample)
            print(json.dumps({k: v for k, v in sample.items() if k != "prediction"}), flush=True)
    finally:
        adapter.close()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
