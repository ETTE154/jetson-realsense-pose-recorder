"""Verify official HALO CUDA FP16 inference plus shared MediaPipe crops on Jetson."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cv2
    import numpy as np
    from comparison.egoforce_adapter import EgoForceAdapter
    from comparison.mediapipe_adapter import MediaPipeAdapter
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="append", required=True, help="Capture directory with rgb.jpg and metadata.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "validation/egoforce_gpu_verified.json")
    args = parser.parse_args()
    started = time.perf_counter()
    adapter = EgoForceAdapter()
    if adapter.device.type != "cuda":
        raise RuntimeError("CPU inference is not a valid GPU verification")
    init_ms = (time.perf_counter() - started) * 1000
    torch = adapter.torch
    torch.cuda.reset_peak_memory_stats()
    media = MediaPipeAdapter()
    rows = []
    for source in args.capture:
        directory = Path(source)
        bgr = cv2.imdecode(np.frombuffer((directory / "rgb.jpg").read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        calibration = metadata.get("rgb_intrinsics") or metadata.get("color_intrinsics")
        if calibration is None and metadata.get("depth_alignment") == "color":
            calibration = metadata["depth_intrinsics"]
        hint_began = time.perf_counter()
        hints = media.infer(bgr)["instances"]
        hint_ms = (time.perf_counter() - hint_began) * 1000
        if not hints:
            raise RuntimeError(f"MediaPipe found no hand in GPU verification input: {source}")
        adapter.infer(bgr, intrinsics=calibration, hand_hints=hints)
        times, kernels, result = [], [], None
        for _ in range(max(1, min(args.repeats, 30))):
            began = time.perf_counter()
            result = adapter.infer(bgr, intrinsics=calibration, hand_hints=hints)
            times.append((time.perf_counter() - began) * 1000)
            kernels.append(result["metadata"]["gpu_kernel_ms"])
        if any(len(item["keypoints"]) != 21 for item in result["instances"]):
            raise RuntimeError("Unexpected GPU hand output shape")
        for item in result["instances"]:
            if not np.isfinite(np.asarray(item["keypoints"])).all():
                raise RuntimeError("Nonfinite CUDA keypoint output")
        overlay = bgr.copy()
        for item in result["instances"]:
            points = item["keypoints"]
            for a,b in item["edges"]:
                cv2.line(overlay, tuple(map(int,points[a][:2])),tuple(map(int,points[b][:2])),(50,220,255),2)
            for x,y,_ in points:
                cv2.circle(overlay,(int(x),int(y)),3,(40,255,90),-1)
        row = {"capture": str(directory), "shared_detector": "MediaPipe Hand Landmarker CPU", "detector_ms": hint_ms,
               "repeat_total_ms": times, "repeat_gpu_kernel_ms": kernels,
               "mean_total_ms": statistics.mean(times), "mean_gpu_kernel_ms": statistics.mean(kernels), "result": result}
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".jpg", overlay)
        if not ok:
            raise RuntimeError("GPU verification overlay JPEG failed")
        encoded.tofile(str(args.output.with_name(args.output.stem+f"_{len(rows)}.jpg")))
        print(json.dumps({"capture": str(directory), "hands":len(result["instances"]),
                          "mean_total_ms":row["mean_total_ms"],"mean_gpu_kernel_ms":row["mean_gpu_kernel_ms"]}),flush=True)
    report = {"backend": adapter.backend, "device": str(adapter.device), "cuda": torch.version.cuda,
              "precision": str(next(adapter.model.parameters()).dtype), "parameter_device":str(next(adapter.model.parameters()).device),
              "initialization_ms":init_ms,"peak_cuda_allocated_mib":torch.cuda.max_memory_allocated()/1024**2,
              "cpu_fallback":False,"samples": rows,
              "accuracy_note":"No ground truth labels; GPU placement, finite output, latency, and visual overlays were verified."}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    media.close()
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
