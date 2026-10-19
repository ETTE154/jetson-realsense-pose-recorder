"""Raw-frame live model worker; imports heavy runtimes only in its own venv."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from live_protocol import PREFIX, read_frame

ROOT = Path(__file__).resolve().parent


class LiveHandDetector:
    """Video tracking for consecutive frames; comparison keeps IMAGE mode."""
    def __init__(self, model_path=None):
        import mediapipe as mp
        from comparison.mediapipe_adapter import DEFAULT_MODEL, convert_result
        self.mp, self.convert_result = mp, convert_result
        self.last_timestamp_ms = -1
        path = Path(model_path or DEFAULT_MODEL).resolve()
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path), delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=.5, min_hand_presence_confidence=.5,
            min_tracking_confidence=.5)
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def infer(self, bgr):
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(self.last_timestamp_ms + 1, int(time.monotonic()*1000))
        self.last_timestamp_ms = timestamp_ms
        result = self.landmarker.detect_for_video(image, timestamp_ms)
        return {"instances": self.convert_result(result, bgr.shape[1], bgr.shape[0]),
                "backend": "MediaPipe CPU · VIDEO tracking",
                "model": "MediaPipe Hand Landmarker (21 points)",
                "metadata": {"mode": "VIDEO with temporal tracking",
                             "confidence_note": "No per-joint confidence; third keypoint value is a display mask. Handedness score is not detection confidence."}}

    def close(self):
        self.landmarker.close()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def emit(value):
    print(PREFIX + json.dumps(value, ensure_ascii=True, allow_nan=False, default=json_default), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("yolo", "mediapipe", "rtmpose", "egoforce"), required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--backend", default="auto")
    args = parser.parse_args()
    adapter = detector = None
    began = time.perf_counter()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if args.candidate == "yolo":
                from pose_runtime import PoseEstimator
                adapter = PoseEstimator(args.model or ROOT / "models/yolov8n-pose.engine", input_size=args.input_size, backend=args.backend)
                backend, model = adapter.backend, "YOLOv8n-pose (COCO 17)"
            elif args.candidate == "mediapipe":
                adapter = LiveHandDetector(args.model)
                backend, model = "MediaPipe CPU · VIDEO tracking", "MediaPipe Hand Landmarker (21 points)"
            elif args.candidate == "rtmpose":
                from comparison.rtmpose_adapter import RTMPoseAdapter
                adapter = RTMPoseAdapter(args.model)
                backend, model = adapter.backend, adapter.model
            else:
                from comparison.egoforce_adapter import EgoForceAdapter
                detector = LiveHandDetector()
                adapter = EgoForceAdapter(args.model)
                if getattr(adapter, "device", None) is None or adapter.device.type != "cuda":
                    raise RuntimeError("Live EgoForce requires CUDA; CPU fallback is disabled")
                backend, model = adapter.backend, "EgoForce HALO · 2D / shared MediaPipe hand ROI"
        emit({"ready": True, "load_ms": (time.perf_counter()-began)*1000,
              "backend": backend, "model": model,
              "metadata": {"candidate": args.candidate, "persistent_worker": True,
                           "device_requirement": "CUDA mandatory" if args.candidate == "egoforce" else "model default"}})
        while True:
            packet = read_frame(sys.stdin.buffer)
            if packet is None:
                break
            request, bgr = packet
            sequence = request.get("sequence")
            try:
                started = time.perf_counter()
                with contextlib.redirect_stdout(sys.stderr):
                    if args.candidate == "yolo":
                        from comparison.rtmpose_adapter import KEYPOINT_NAMES, EDGES
                        adapter.confidence = request["confidence"]
                        instances = adapter.infer(bgr)
                        for instance in instances:
                            instance.update(keypoint_names=KEYPOINT_NAMES, edges=EDGES,
                                            label="person", score_kind="person_detection",
                                            keypoint_confidence_kind="model_score", coordinate_kind="image_2d")
                        result = {"instances": instances, "backend": backend, "model": model,
                                  "metadata": {"confidence_threshold": request["confidence"],
                                               "roi_note": "YOLO uses the full frame"}}
                    elif args.candidate == "rtmpose":
                        adapter.roi = request.get("roi")
                        adapter.confidence = request["confidence"]
                        result = adapter.infer(bgr)
                    elif args.candidate == "mediapipe":
                        result = adapter.infer(bgr)
                    else:
                        hints = request.get("hand_hints")
                        detector_ms = 0.
                        if hints is None:
                            detector_started = time.perf_counter()
                            hints = detector.infer(bgr)["instances"]
                            detector_ms = (time.perf_counter()-detector_started)*1000
                        halo_started = time.perf_counter()
                        result = adapter.infer(bgr, intrinsics=request.get("intrinsics"),
                                               roi=request.get("roi"), hand_hints=hints)
                        result.setdefault("metadata", {}).update(
                            hand_detector="MediaPipe Hand Landmarker CPU VIDEO tracking (same worker)",
                            hand_detector_ms=detector_ms,
                            halo_ms=(time.perf_counter()-halo_started)*1000,
                            hand_detector_scope="full original RGB frame",
                            pipeline="MediaPipe hand detection → official HALO CUDA 2D heads")
                emit({"sequence": sequence, "ok": True,
                      "inference_ms": (time.perf_counter()-started)*1000, "result": result})
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                emit({"sequence": sequence, "ok": False, "error": str(exc)})
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit({"ready": False, "error": str(exc)})
    finally:
        with contextlib.redirect_stdout(sys.stderr):
            if adapter is not None:
                adapter.close()
            if detector is not None:
                detector.close()


if __name__ == "__main__":
    main()
