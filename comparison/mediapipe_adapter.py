"""Official MediaPipe Hand Landmarker adapter, isolated from the camera SDK.

The task reports handedness scores, but no per-landmark visibility/confidence.
The third keypoint value is therefore a drawing mask, not measured confidence.
All image coordinates refer to the original unmirrored input image. Model world
coordinates are deliberately excluded: they are not D435i measured depths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


HAND_NAMES = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
]
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models/comparison/hand_landmarker.task"


def convert_result(result, width: int, height: int) -> list[dict]:
    """Convert task output without inventing per-keypoint confidence or depth."""
    instances = []
    for index, landmarks in enumerate(result.hand_landmarks):
        points = [[float(p.x * width), float(p.y * height), 1.0] for p in landmarks]
        categories = result.handedness[index] if index < len(result.handedness) else []
        category = categories[0] if categories else None
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        if not points:
            continue
        instances.append({
            "bbox": [max(0.0, min(xs)), max(0.0, min(ys)),
                     min(float(width - 1), max(xs)), min(float(height - 1), max(ys))],
            "bbox_kind": "landmark_extent",
            "score": float(category.score) if category else 0.0,
            "score_kind": "handedness_classification",
            "label": str(category.category_name) if category else "Hand",
            "keypoints": points,
            "keypoint_names": HAND_NAMES.copy(),
            "edges": [edge.copy() for edge in HAND_EDGES],
            "keypoint_confidence_kind": "not_reported_display_mask",
            "coordinate_kind": "image_2d",
        })
    return instances


class MediaPipeAdapter:
    def __init__(self, model_path=None, min_confidence: float = 0.5):
        import mediapipe as mp

        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        self.model_path = Path(model_path or DEFAULT_MODEL).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MediaPipe hand model missing: {self.model_path}")
        self.mp = mp
        self.version = getattr(mp, "__version__", "unknown")
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(self.model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def infer(self, bgr: np.ndarray) -> dict:
        if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("Expected BGR uint8 image with shape (height, width, 3)")
        if bgr.dtype != np.uint8 or min(bgr.shape[:2]) <= 0:
            raise ValueError("Expected nonempty BGR uint8 image")
        start = time.perf_counter()
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(image)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "instances": convert_result(result, bgr.shape[1], bgr.shape[0]),
            "backend": "MediaPipe CPU",
            "model": "MediaPipe Hand Landmarker (21 points)",
            "model_version": self.version,
            "model_path": str(self.model_path),
            "inference_ms": round(elapsed_ms, 3),
            "input_size": [int(bgr.shape[1]), int(bgr.shape[0])],
            "mode": "IMAGE (independent frame; no temporal tracking)",
            "confidence_note": "No per-joint confidence. Score is handedness classification; keypoint third value is a drawing mask.",
        }

    def close(self):
        if getattr(self, "landmarker", None) is not None:
            self.landmarker.close()
            self.landmarker = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()
    import cv2

    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise ValueError(f"Cannot read image: {args.image}")
    adapter = MediaPipeAdapter(args.model, args.min_confidence)
    try:
        results = [adapter.infer(bgr) for _ in range(max(1, args.repeat))]
        result = results[-1]
        result["sample"] = str(args.image)
        result["repeat_latency_ms"] = [r["inference_ms"] for r in results]
        result["instance_count"] = len(result["instances"])
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded + "\n", encoding="utf-8")
        print(encoded, flush=True)
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
