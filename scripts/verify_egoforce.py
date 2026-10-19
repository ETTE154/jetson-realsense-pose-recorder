"""Run genuine EgoForce HALO 2D on a saved calibrated D435i RGB image."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cv2
    import numpy as np
    from comparison.egoforce_adapter import EgoForceAdapter
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--hand-hints", type=Path, help="JSON adapter result containing instances; otherwise use --roi")
    parser.add_argument("--roi", help="Pixel x1,y1,x2,y2. Hand ROI without hints; forearm ROI when hints supplied")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    image = cv2.imdecode(np.fromfile(str(args.image), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Cannot decode RGB image")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    intrinsics = metadata.get("rgb_intrinsics") or metadata.get("color_intrinsics")
    if not intrinsics and metadata.get("depth_alignment") == "color":
        intrinsics = metadata.get("depth_intrinsics")
    if not intrinsics:
        raise RuntimeError("Snapshot is missing color-aligned RGB camera intrinsics")
    hints = None
    if args.hand_hints:
        raw = json.loads(args.hand_hints.read_text(encoding="utf-8"))
        hints = raw.get("instances", raw) if isinstance(raw, dict) else raw
    roi = [float(value) for value in args.roi.split(",")] if args.roi else None
    began = time.perf_counter()
    adapter = EgoForceAdapter()
    init_ms = (time.perf_counter() - began) * 1000
    result = adapter.infer(image, intrinsics=intrinsics, roi=roi, hand_hints=hints)
    result["initialization_ms"] = round(init_ms, 2)
    result["image"] = str(args.image)
    result["accuracy_note"] = "No ground truth labels: keypoint count and latency are not an accuracy score."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    canvas = image.copy()
    for instance in result["instances"]:
        points = instance["keypoints"]
        for a, b in instance["edges"]:
            if points[a][2] >= 0.3 and points[b][2] >= 0.3:
                cv2.line(canvas, tuple(map(int, points[a][:2])), tuple(map(int, points[b][:2])), (0, 220, 255), 2)
        for x, y, score in points:
            if score >= 0.3:
                cv2.circle(canvas, (int(x), int(y)), 3, (20, 240, 100), -1)
    ok, encoded = cv2.imencode(".jpg", canvas)
    if not ok:
        raise RuntimeError("Cannot encode verification overlay")
    encoded.tofile(str(args.output.with_suffix(".jpg")))
    print(json.dumps({"model": result["model"], "backend": result["backend"], "instances": len(result["instances"]),
                      "keypoints": [len(item["keypoints"]) for item in result["instances"]],
                      "initialization_ms": result["initialization_ms"], "model_ms": result["metadata"]["model_ms"],
                      "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
