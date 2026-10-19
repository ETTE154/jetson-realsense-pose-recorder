#!/usr/bin/env python3
"""Read-only graph benchmark using recorded native samples; no camera control."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_id")
    parser.add_argument("--module", type=Path, default=ROOT / "imu_video.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = ROOT / "recordings" / args.recording_id
    if directory.resolve().parent != (ROOT / "recordings").resolve():
        parser.error("Invalid recording ID")
    spec = importlib.util.spec_from_file_location("imu_benchmark_target", args.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = json.loads((directory / "manifest.json").read_text())
    timeline = module.IMUTimeline(manifest["start_monotonic"])
    with (directory / "dataset.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not row["record_type"].startswith("imu_"):
                continue
            kind = row["imu_type"]
            suffix = "_m_s2" if kind == "accel" else "_rad_s"
            timeline.add({"type": kind, "units": row["imu_units"], "sample_id": int(row["imu_sample_id"]),
                "frame_number": int(row["source_frame_number"]), "capture_generation": int(row["source_generation"]),
                "received_monotonic": float(row["source_received_monotonic_s"]),
                "timestamp_ms": float(row["source_timestamp_ms"]), "timestamp_domain": row["source_timestamp_domain"],
                **{axis: float(row[kind + "_" + axis + suffix]) for axis in "xyz"}})
    report = {"module": str(args.module), "recording_id": args.recording_id, "benchmarks": []}
    for elapsed in (3., 8., 15., 25.):
        if elapsed > manifest["duration_s"]:
            continue
        views, panels = [], []
        for _ in range(12):
            start = time.perf_counter()
            view = timeline.view(manifest["start_monotonic"] + elapsed)
            views.append((time.perf_counter()-start)*1000.)
            start = time.perf_counter()
            module.render_panel(view, 1280, 280, elapsed)
            panels.append((time.perf_counter()-start)*1000.)
        report["benchmarks"].append({"elapsed_s": elapsed, "view_ms": statistics.median(views),
            "panel_ms": statistics.median(panels), "points": {k:len(s["points"]) for k,s in view["streams"].items()}})
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
