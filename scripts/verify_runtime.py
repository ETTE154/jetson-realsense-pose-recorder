"""Reproduce model inference checks on the Jetson; optionally test live D435i capture."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from camera import Camera
from pose_runtime import PoseEstimator


def software_info():
    info = {
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "numpy": np.__version__, "opencv": cv2.__version__,
    }
    try:
        import tensorrt
        info["tensorrt"] = tensorrt.__version__
    except ImportError:
        info["tensorrt"] = None
    for key, filename in (("jetson_model", "/proc/device-tree/model"),
                          ("l4t", "/etc/nv_tegra_release")):
        path = Path(filename)
        info[key] = path.read_text(errors="replace").strip("\x00\r\n ") if path.is_file() else None
    for library_name in (ctypes.util.find_library("cudart"), "libcudart.so.12"):
        if not library_name:
            continue
        try:
            library = ctypes.CDLL(library_name)
            for key, function_name in (("cuda_runtime", "cudaRuntimeGetVersion"),
                                       ("cuda_driver_api", "cudaDriverGetVersion")):
                function = getattr(library, function_name)
                function.argtypes = [ctypes.POINTER(ctypes.c_int)]
                function.restype = ctypes.c_int
                version = ctypes.c_int()
                if function(ctypes.byref(version)) == 0:
                    info[key] = f"{version.value // 1000}.{(version.value % 1000) // 10}"
            break
        except (OSError, AttributeError):
            continue
    return info


def check_people(people):
    if not people:
        raise RuntimeError("The reference image produced no people")
    for person in people:
        points = np.asarray(person["keypoints"], dtype=np.float64)
        if points.shape != (17, 3) or not np.isfinite(points).all():
            raise RuntimeError("Pose output must contain 17 finite COCO (x,y,confidence) joints")
        if np.asarray(person["bbox"]).shape != (4,) or not np.isfinite(person["bbox"]).all():
            raise RuntimeError("Pose bounding box contains invalid coordinates")
        if not np.isfinite(person["score"]):
            raise RuntimeError("Pose score is not finite")


def verify_model(model_path, image_path, iterations):
    image = cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot decode reference image: {image_path}")
    milliseconds, counts = [], []
    with PoseEstimator(model_path, input_size=640, backend="auto") as estimator:
        for _ in range(3):
            check_people(estimator.infer(image))
        for _ in range(iterations):
            started = time.perf_counter()
            people = estimator.infer(image)
            milliseconds.append((time.perf_counter() - started) * 1000)
            check_people(people)
            counts.append(len(people))
        actual_model = Path(estimator.model_path)
        return {
            "verified": True, "backend": estimator.backend_name,
            "model": str(actual_model), "model_sha256": hashlib.sha256(actual_model.read_bytes()).hexdigest(),
            "image": str(image_path), "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "image_shape": list(image.shape), "input_size": 640,
            "warmup_iterations": 3, "iterations": iterations,
            "latency_ms": {"median": float(np.median(milliseconds)),
                           "p95": float(np.percentile(milliseconds, 95)),
                           "mean": float(np.mean(milliseconds))},
            "person_counts": counts, "last_people": people,
            "scope": "Reference-image model inference only; no camera frames used in this benchmark.",
        }


def verify_camera():
    camera = Camera(width=640, height=480, fps=30)
    frames, valid_pixels, total_pixels = 0, 0, 0
    first_received = last_received = None
    after_id = -1
    dimensions, depth_scale = None, None
    started = time.monotonic()
    state = {}
    try:
        camera.start()
        deadline = started + 15.0
        while frames < 60 and time.monotonic() < deadline:
            frame = camera.wait_frame(after_id, timeout=min(.5, max(0., deadline - time.monotonic())))
            if frame is None:
                continue
            after_id = frame.frame_id
            received = time.monotonic()
            first_received = received if first_received is None else first_received
            last_received = received
            if frame.depth.dtype != np.uint16 or frame.depth.shape != frame.rgb.shape[:2]:
                raise RuntimeError("Live camera did not return color-aligned uint16 depth")
            if not np.isfinite(frame.depth_scale) or frame.depth_scale <= 0:
                raise RuntimeError("Live camera reported an invalid depth scale")
            depth_m = frame.depth.astype(np.float32) * frame.depth_scale
            valid_pixels += int(np.count_nonzero((frame.depth > 0) & (depth_m >= .1) & (depth_m <= 10.)))
            total_pixels += frame.depth.size
            frames += 1
            dimensions = {"width": int(frame.rgb.shape[1]), "height": int(frame.rgb.shape[0])}
            depth_scale = float(frame.depth_scale)
        state = camera.snapshot()
    finally:
        camera.stop()
    interval = (last_received - first_received) if frames > 1 else 0.
    verified = frames == 60 and valid_pixels > 0
    return {
        "requested": True, "verified": verified, "frames_received": frames, "frames_requested": 60,
        "timeout_seconds": 15, "elapsed_seconds": time.monotonic() - started,
        "actual_dimensions": dimensions, "depth_scale_m_per_unit": depth_scale,
        "valid_metric_depth_percent": 100. * valid_pixels / total_pixels if total_pixels else None,
        "valid_metric_depth_range_m": [.1, 10.],
        "observed_stream_fps": (frames - 1) / interval if interval > 0 else None,
        "capture_status": state,
        "error": None if verified else (state.get("error") or "Fewer than 60 frames or no valid metric depth received"),
        "scope": "Live D435i RGB-D capture only; pose inference was benchmarked separately.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolov8n-pose.engine")
    parser.add_argument("--image", type=Path, default=ROOT / "models/test_bus.jpg")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/gpu_validation.json")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--camera", action="store_true", help="Also require 60 real RGB-D frames within 15 seconds")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    output = args.output.expanduser().resolve()
    if output.suffix.lower() != ".json":
        parser.error("--output must be a JSON report path")
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "software": software_info(),
        "model_validation": {"verified": False},
        "camera_validation": {"requested": args.camera, "verified": False,
                              "scope": ("Live camera validation requested; not yet completed." if args.camera else
                                        "Camera validation not requested; model-only checks do not verify D435i capture.")},
    }
    success = False
    try:
        report["model_validation"] = verify_model(
            args.model.expanduser().resolve(), args.image.expanduser().resolve(), args.iterations,
        )
        if args.camera:
            report["camera_validation"] = verify_camera()
        success = report["model_validation"]["verified"] and (not args.camera or report["camera_validation"]["verified"])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["success"] = success
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"success": success, "report": str(output),
                      "model_verified": report["model_validation"]["verified"],
                      "camera_verified": report["camera_validation"]["verified"],
                      "error": report.get("error")}, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
