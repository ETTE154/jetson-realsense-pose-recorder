#!/usr/bin/env python3
"""Record real RGB-D/IMU, then check movie pixels against the saved samples."""
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imu_video import IMUTimeline, metadata, output_size, render_panel
import verify_dataset

ROOT = Path(__file__).resolve().parents[1]


def main(recording_seconds=8.):
    report = {"ok": False, "started_unix": time.time()}
    try:
        if not verify_dataset.api("/api/status").get("recording", {}).get("imu_visualization", {}).get("enabled"):
            raise RuntimeError("IMU graph movie recording is disabled. Use verify_video_continuity.py for the current recorder.")
        result = verify_dataset.main(recording_seconds=recording_seconds)
        base = json.loads((ROOT / "validation/imu_dataset_live_report.json").read_text())
        if result not in (0, 2) or not base.get("core_ok"):
            raise RuntimeError("Sensor/CSV/video preflight failed: " + str(base.get("error")))
        state = base["final_recording"]
        directory = ROOT / "recordings" / state["id"]
        counts = state["dataset"]["rows_written"]
        # Offline reconstruction keeps every saved sample, including frames
        # preceding the live rolling buffer. The displayed ten-second window
        # remains identical, so long recordings can be checked from start to end.
        timeline = IMUTimeline(state["start_monotonic"], capacity=max(16384, counts["imu_accel"] + counts["imu_gyro"] + 1))
        by_id = {}
        with (directory / "dataset.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if not row["record_type"].startswith("imu_"):
                    continue
                kind = row["imu_type"]
                prefix = "accel_" if kind == "accel" else "gyro_"
                suffix = "_m_s2" if kind == "accel" else "_rad_s"
                sample = {"type": kind, "units": row["imu_units"], "sample_id": int(row["imu_sample_id"]),
                          "frame_number": int(row["source_frame_number"]),
                          "received_monotonic": float(row["source_received_monotonic_s"]),
                          "timestamp_ms": float(row["source_timestamp_ms"]),
                          "timestamp_domain": row["source_timestamp_domain"],
                          "capture_generation": int(row["source_generation"]),
                          **{axis: float(row[prefix + axis + suffix]) for axis in ("x", "y", "z")}}
                timeline.add(sample)
                by_id[sample["sample_id"]] = sample
        rows = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
        freshness = {}
        for key, limit in (("sensor_source", .75), ("combined_source", 2.1)):
            last, run, longest, maximum_age = None, 0, 0, 0.
            for row in rows:
                source = row.get(key)
                frame_id = source.get("frame_id") if source else None
                run = run + 1 if frame_id == last else 1
                longest = max(longest, run)
                last = frame_id
                if source:
                    maximum_age = max(maximum_age, state["start_monotonic"] + row["t_s"] - source["received_monotonic"])
            freshness[key] = {"max_source_age_s": maximum_age, "longest_same_source_s": longest/state["fps"],
                              "unique_sources": len({r[key]["frame_id"] for r in rows if r.get(key)}), "limit_s": limit}
            assert maximum_age < limit and longest/state["fps"] < limit, (key, freshness[key])
        matched = 0
        for row in rows:
            tick = state["start_monotonic"] + row["t_s"]
            for key in ("imu_video", "combined_imu"):
                saved = row[key]
                reference = saved["reference_monotonic"]
                if key == "imu_video":
                    assert abs(reference - tick) < 1e-8
                elif row.get("combined_source"):
                    assert reference == row["combined_source"]["received_monotonic"]
                expected = metadata(timeline.view(reference, saved["capture_generation"]))
                assert expected == saved, f"IMU metadata differs from exact source CSV at {row['frame_index']} / {key}"
                for stream in saved["streams"].values():
                    sample = stream["latest"]
                    if sample:
                        assert sample == by_id[sample["sample_id"]]
                        assert sample["received_monotonic"] <= reference
                        matched += 1
        assert matched > 0
        pixels = {}
        for name, key in (("combined", "combined_imu"), ("imu", "imu_video")):
            capture = cv2.VideoCapture(str(directory / (name + ".mp4")))
            checks = []
            try:
                indices = {len(rows)//3, len(rows)//2, len(rows)-1}
                step = max(1, round(state["fps"]*5))
                indices.update(range(step, len(rows), step))
                for index in sorted(indices):
                    row = rows[index]
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, image = capture.read()
                    assert ok
                    width, height = output_size("imu", state["width"], state["height"])
                    saved = row[key]
                    view = timeline.view(saved["reference_monotonic"], saved["capture_generation"])
                    label = "RGB-D SOURCE" if name == "combined" and row.get("combined_source") else "VIDEO TIMELINE"
                    expected = render_panel(view, width, height, row["t_s"], label)
                    actual = image[-height:]
                    mae = float(np.abs(expected.astype(np.int16)-actual.astype(np.int16)).mean())
                    # Compression permits a small pixel difference, but a blank
                    # panel or graph rendered from another time must fail.
                    signal = np.max(expected, axis=2) > 100
                    signal_mae = float(np.abs(expected.astype(np.int16)-actual.astype(np.int16))[signal].mean())
                    assert mae < 7. and signal_mae < 25., (name, index, mae, signal_mae)
                    assert signal.sum() > 3000
                    checks.append({"frame_index": index, "panel_mae": mae, "signal_mae": signal_mae})
                    if index == len(rows)//2:
                        cv2.imwrite(str(ROOT / "validation" / ("imu_recording_" + name + "_preview.jpg")), image)
            finally:
                capture.release()
            pixels[name] = checks
        report.update(ok=True, recording_id=state["id"], duration_s=state["duration_s"],
                      files=state["files"], output_dimensions=state["output_dimensions"],
                      video_frames=len(rows), matched_imu_display_samples=matched, pixel_checks=pixels,
                      source_freshness=freshness, performance=state.get("performance"),
                      dataset=state["dataset"], imu_visualization=state["imu_visualization"],
                      base_core_ok=base["core_ok"], pose_joints_unverified=base["pose_joints_unverified"])
        print("IMU video: source/tick alignment, every displayed sample vs CSV, and decoded panel pixels passed", flush=True)
    except Exception as error:
        import traceback
        report.update(error=str(error), traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    finally:
        report["finished_unix"] = time.time()
        (ROOT / "validation").mkdir(exist_ok=True)
        (ROOT / "validation/imu_video_live_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=8.)
    args = parser.parse_args()
    if not 8 <= args.seconds <= 300:
        parser.error("seconds must be between 8 and 300")
    raise SystemExit(main(args.seconds))
