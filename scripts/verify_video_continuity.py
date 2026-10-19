#!/usr/bin/env python3
"""Long real-camera recording: verify source freshness, CSV and all MP4s."""
import argparse
import json
from pathlib import Path
import sys
import time

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_dataset

ROOT = Path(__file__).resolve().parents[1]


def analyze(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
    result = {"id": manifest["id"], "duration_s": manifest["duration_s"],
              "dropped_frames": manifest["dropped_frames"], "dataset_dropped": manifest["dataset"]["dropped_total"],
              "camera_csv_rows": manifest["dataset"]["rows_written"]["camera"],
              "imu_visualization": manifest.get("imu_visualization"), "performance": manifest.get("performance"),
              "frames": len(rows), "freshness": {}}
    for key in ("sensor_source", "combined_source"):
        longest = run = 0
        last = None
        ages = []
        for row in rows:
            source = row.get(key)
            if source is None:
                run, last = 0, None
                continue
            frame_id = source["frame_id"]
            run = run + 1 if frame_id == last else 1
            longest = max(longest, run)
            last = frame_id
            ages.append((row["t_s"], manifest["start_monotonic"] + row["t_s"] - source["received_monotonic"]))
        result["freshness"][key] = {"max_age_s": max((a for _, a in ages), default=None),
            "longest_same_source_s": longest/manifest["fps"],
            "unique_sources": len({r[key]["frame_id"] for r in rows if r.get(key)}),
            "five_second_windows": [{"start_s": start, "max_age_s": max(a for t,a in ages if start <= t < start+5)}
                for start in range(0, int(manifest["duration_s"])+1, 5)
                if any(start <= t < start+5 for t,a in ages)]}
    return result, manifest, rows


def main(seconds, compare):
    report = {"ok": False, "started_unix": time.time(), "seconds_requested": seconds}
    try:
        result = verify_dataset.main(recording_seconds=seconds)
        base = json.loads((ROOT / "validation/imu_dataset_live_report.json").read_text())
        if result not in (0, 2) or not base.get("core_ok"):
            raise RuntimeError("Recording/CSV check failed: " + str(base.get("error")))
        directory = ROOT / "recordings" / base["final_recording"]["id"]
        analysis, manifest, rows = analyze(directory)
        report["recording"] = analysis
        assert manifest["imu_visualization"]["enabled"] is False
        assert "imu" not in manifest["files"]
        assert manifest["output_dimensions"]["combined"] == {"width": 1280, "height": 960}
        assert manifest["dropped_total"] == 0, analysis["dropped_frames"]
        assert analysis["dataset_dropped"] == 0
        for key, limit in (("sensor_source", .75), ("combined_source", 2.1)):
            values = analysis["freshness"][key]
            assert values["max_age_s"] < limit and values["longest_same_source_s"] < limit, (key, values)
        assert manifest["performance"]["mean_frame_ms"] < manifest["performance"]["frame_budget_ms"]
        decoded = {}
        for name in ("rgb", "combined"):
            capture = cv2.VideoCapture(str(directory / (name + ".mp4")))
            checks = []
            try:
                step = max(1, round(manifest["fps"] * 5))
                for index in sorted(set(range(0, len(rows), step)) | {len(rows)-1}):
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, image = capture.read()
                    assert ok, (name, index)
                    dimensions = manifest["output_dimensions"][name]
                    assert image.shape == (dimensions["height"], dimensions["width"], 3)
                    checks.append(index)
                    if index == (len(rows)//step//2)*step and name == "combined":
                        cv2.imwrite(str(ROOT / "validation/video_continuity_preview.jpg"), image)
            finally:
                capture.release()
            decoded[name] = checks
        report.update(ok=True, decoded_every_five_seconds=decoded,
                      raw_imu_exactly_preserved=base["checks"]["dataset_csv"]["native_imu_samples_exactly_preserved"],
                      csv_counts=base["checks"]["dataset_csv"]["counts"],
                      pose_joints_unverified=base["pose_joints_unverified"],
                      live_settings_unchanged=base["live_settings_unchanged"],
                      all_video_checks=base["checks"]["videos"],
                      previous_recordings=[analyze(ROOT / "recordings" / identifier)[0] for identifier in compare])
        print("PASS: long recording, frame freshness, no drops, CSV native IMU, MP4 playback and unchanged settings", flush=True)
    except Exception as error:
        import traceback
        report.update(error=str(error), traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    finally:
        report["finished_unix"] = time.time()
        (ROOT / "validation/video_continuity_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=90.)
    parser.add_argument("--compare", action="append", default=[])
    args = parser.parse_args()
    if not 15 <= args.seconds <= 300:
        parser.error("seconds must be between 15 and 300")
    if any((ROOT / "recordings" / value).resolve().parent != (ROOT / "recordings").resolve() for value in args.compare):
        parser.error("Invalid recording ID")
    raise SystemExit(main(args.seconds, args.compare))
