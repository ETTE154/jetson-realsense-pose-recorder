#!/usr/bin/env python3
"""Real D435i IMU/camera/Pose CSV verification, without changing live settings.

Run on the Jetson while the loopback server is already running. Creates one
short recording only if no user recording is active. Exit 0 means every path,
including real detected Pose joints, was verified; exit 2 means the sensor/CSV/
video checks passed but no actual Pose joints were available to verify.
The JSON report is written even if initial connectivity or preflight fails.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import traceback
import urllib.error
import urllib.request

import cv2  # Needed by the shared MP4 inspection helper.

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_csv import FIELDS, RECORD_TYPES
from verify_recording import inspect_recording

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8766"
DESTINATION = ROOT / "validation" / "imu_dataset_live_report.json"
ACTIVE = {"countdown", "recording", "stopping"}
TERMINAL = {"completed", "cancelled", "canceled", "error"}


def api(path, payload=None):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite(value, name):
    require(value is not None and value != "", f"Missing {name}")
    number = float(value)
    require(math.isfinite(number), f"Non-finite {name}")
    return number


def integer(value, name):
    number = finite(value, name)
    require(number.is_integer(), f"Non-integral {name}: {value}")
    return int(number)


def rec_status():
    return api("/api/recording/status")


class HistoryMonitor:
    """Incrementally read the actual UI endpoint before its bounded history rolls."""
    def __init__(self):
        initial = api("/api/imu?after=0&limit=4096")
        self.cursor = integer(initial["latest_id"], "initial latest_id")
        self.initial_id = self.cursor
        self.samples = {}
        self.calls = 0
        self.latest_imu = initial["imu"]
        self.observed_epochs = set()

    def poll(self):
        value = api(f"/api/imu?after={self.cursor}&limit=4096")
        self.calls += 1
        require(not value.get("truncated") and not value.get("missed_samples"),
                "IMU UI history lost samples between incremental polls")
        previous = self.cursor
        for sample in value["samples"]:
            sequence = integer(sample["sample_id"], "sample_id")
            require(sequence > previous and sequence not in self.samples,
                    "IMU incremental endpoint returned duplicate/out-of-order sample IDs")
            require(sequence == previous + 1,
                    "IMU incremental endpoint skipped an assigned sample ID")
            require(integer(sample["sequence"], "sequence") == sequence,
                    "UI sequence differs from native sample ID")
            kind = sample["type"]
            require(kind in {"gyro", "accel"} and sample["stream"] == kind,
                    "Invalid motion type/stream")
            generation = integer(sample["capture_generation"], "capture_generation")
            require(integer(sample["capture_epoch"], "capture_epoch") == generation,
                    "UI epoch differs from native motion generation")
            for key in ("x", "y", "z", "timestamp_ms", "received_time", "received_monotonic"):
                finite(sample[key], "motion " + key)
            integer(sample["frame_number"], "motion frame_number")
            require(bool(sample["timestamp_domain"]), "Motion clock domain is missing")
            require(sample["units"] in ({"rad/s"} if kind == "gyro" else {"m/s²", "m/s^2", "m/s2"}),
                    "Motion units are not SI")
            self.samples[sequence] = sample
            self.observed_epochs.add(generation)
            previous = sequence
        next_id = integer(value["next_id"], "next_id")
        require(next_id == previous, "IMU history cursor does not identify last delivered sample")
        self.cursor = next_id
        self.latest_imu = value["imu"]
        return value

    def observe_until(self, deadline):
        while time.monotonic() < deadline:
            self.poll()
            time.sleep(min(.1, max(0., deadline - time.monotonic())))
        self.poll()


def cadence(samples, profiles, *, label):
    """Check actual per-stream capture cadence against the opened SDK profile."""
    result = {}
    for kind in ("gyro", "accel"):
        rows = [sample for sample in samples if sample["type"] == kind]
        require(len(rows) >= 10, f"{label}: fewer than 10 actual {kind} samples")
        groups = {}
        for row in rows:
            groups.setdefault((row["capture_generation"], row["timestamp_domain"]), []).append(row)
        measured = []
        for (generation, domain), group in groups.items():
            group.sort(key=lambda item: item["received_monotonic"])
            if len(group) < 2:
                continue
            host_span = group[-1]["received_monotonic"] - group[0]["received_monotonic"]
            source_span = (group[-1]["timestamp_ms"] - group[0]["timestamp_ms"]) / 1000.
            require(host_span > 0 and source_span > 0, f"{label}: {kind} source clock did not advance")
            measured.append({"capture_generation": generation, "timestamp_domain": domain,
                             "samples": len(group), "span_s": host_span,
                             "received_rate_hz": (len(group)-1) / host_span,
                             "source_rate_hz": (len(group)-1) / source_span})
        require(bool(measured), f"{label}: {kind} cadence could not be measured")
        expected = next((float(profile["fps"]) for profile in profiles if profile.get("type") == kind), None)
        require(expected is not None and expected > 0, f"{label}: missing opened {kind} profile FPS")
        for group in measured:
            if group["samples"] >= 10 and group["span_s"] >= .5:
                for name in ("received_rate_hz", "source_rate_hz"):
                    require(.75 * expected <= group[name] <= 1.25 * expected,
                            f"{label}: {kind} {name}={group[name]:.2f} differs from SDK profile {expected}")
        result[kind] = {"total_samples": len(rows), "profile_fps": expected, "groups": measured}
    return result


def check_download(path, url):
    disk = path.read_bytes()
    with urllib.request.urlopen(BASE + url, timeout=30) as response:
        downloaded = response.read()
        content_type = response.headers.get("Content-Type")
    require(downloaded == disk, "Downloaded CSV bytes differ from finalized disk file")
    with urllib.request.urlopen(urllib.request.Request(BASE + url, method="HEAD"), timeout=10) as response:
        require(response.status == 200 and int(response.headers["Content-Length"]) == len(disk),
                "CSV HEAD size/status mismatch")
        require(response.headers.get("Accept-Ranges") == "bytes", "CSV HEAD lacks byte-range support")
        require(response.read() == b"", "CSV HEAD unexpectedly returned a body")
    length = min(128, len(disk))
    with urllib.request.urlopen(urllib.request.Request(
            BASE + url, headers={"Range": f"bytes=0-{length-1}"}), timeout=10) as response:
        require(response.status == 206 and response.read() == disk[:length], "CSV Range response mismatch")
        content_range = response.headers.get("Content-Range")
    return disk, {"bytes": len(disk), "sha256": hashlib.sha256(disk).hexdigest(),
                  "download_matches_disk": True, "content_type": content_type,
                  "head_ok": True, "range_ok": True, "content_range": content_range}


def assert_private_partial(identifier):
    path = f"/recordings/{identifier}/dataset.partial.csv"
    try:
        with urllib.request.urlopen(BASE + path, timeout=10) as response:
            response.read()
    except urllib.error.HTTPError as error:
        require(error.code == 404, f"Partial CSV returned unexpected HTTP {error.code}")
        return {"path": path, "status": error.code, "not_served": True}
    raise RuntimeError("In-progress dataset.partial.csv was publicly downloadable")


def inspect_csv(state, monitor):
    """Validate actual persisted rows against native callback history."""
    directory = ROOT / "recordings" / state["id"]
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    start, end = manifest["start_monotonic"], manifest["stop_monotonic"]
    require(start - manifest["requested_monotonic"] >= 5. - 1e-8,
            "Dataset recording starts before five-second capture deadline")
    dataset = state["dataset"]
    require(dataset["state"] == "completed" and not dataset.get("error"), "CSV writer did not complete")
    require(dataset.get("dropped_total", 0) == 0, "CSV writer dropped events during validation recording")
    require(dataset.get("rejected_invalid", 0) == 0, "CSV writer rejected invalid source samples")
    url = state["files"].get("dataset_csv")
    require(bool(url), "Final recording does not publish dataset_csv download")
    disk, transport = check_download(directory / "dataset.csv", url)
    require(disk.startswith(b"\xef\xbb\xbf"), "CSV is missing UTF-8 BOM")
    reader = csv.DictReader(io.StringIO(disk.decode("utf-8-sig"), newline=""))
    require(tuple(reader.fieldnames or ()) == FIELDS, "CSV schema header differs from DatasetCSV.FIELDS")
    rows = list(reader)
    require(bool(rows), "CSV contains zero actual rows")
    counts = {kind: 0 for kind in RECORD_TYPES}
    camera_sources = {}
    pose_events = {}
    imu_ids = set()
    csv_motion = []
    domains = set()
    generations = set()
    instances = set()
    valid_3d = 0
    previous_event = 0
    for index, row in enumerate(rows, start=1):
        require(integer(row["row_sequence"], "row_sequence") == index, "CSV row sequence is not contiguous")
        event = integer(row["event_index"], "event_index")
        require(event >= previous_event, "CSV event_index moved backwards")
        previous_event = event
        require(row["schema_version"] == "1.0" and row["recording_id"] == state["id"],
                "CSV schema/session identity is wrong")
        kind = row["record_type"]
        require(kind in counts, "Unknown CSV record type: " + kind)
        counts[kind] += 1
        stamp = finite(row["source_received_monotonic_s"], "source_received_monotonic_s")
        require(start <= stamp <= end, f"{kind} source timestamp is outside exact countdown/stop gate")
        elapsed = finite(row["source_elapsed_s"], "source_elapsed_s")
        require(abs(elapsed - (stamp - start)) < 1e-8, "source_elapsed_s is not capture-clock relative")
        require(finite(row["event_received_monotonic_s"], "event_received_monotonic_s") >= stamp,
                "CSV submit time precedes source receipt")
        require(bool(row["source_timestamp_domain"]), "CSV source timestamp domain is empty")
        finite(row["source_timestamp_ms"], "source_timestamp_ms")
        finite(row["source_received_unix_s"], "source_received_unix_s")
        generation = integer(row["source_generation"], "source_generation")
        generations.add(generation)
        domains.add((row["source_clock"], row["source_timestamp_domain"]))
        if kind == "camera":
            source = (generation, integer(row["source_frame_id"], "source_frame_id"))
            require(source not in camera_sources, "Duplicate original camera event")
            camera_sources[source] = row
            require(bool(row["device_serial"]), "Camera serial provenance is missing")
            for key in ("color_frame_number", "depth_frame_number"):
                integer(row[key], key)
            for key in ("depth_timestamp_ms", "camera_fx_px", "camera_fy_px", "depth_scale_m_per_unit"):
                finite(row[key], key)
            total = integer(row["depth_total_pixels"], "depth_total_pixels")
            valid = integer(row["depth_valid_pixels"], "depth_valid_pixels")
            require(total > 0 and 0 <= valid <= total, "Invalid depth pixel counts")
            require(abs(finite(row["depth_valid_fraction"], "depth_valid_fraction") - valid / total) < 1e-10,
                    "Depth fraction does not match valid pixel counts")
            require(row["raw_depth_file"] == "", "CSV unexpectedly claims continuous raw Z16 storage")
            require(row["video_mapping_file"] == "frames.jsonl", "Camera CSV lacks video sidecar mapping")
        elif kind.startswith("imu_"):
            sample_id = integer(row["imu_sample_id"], "imu_sample_id")
            require(sample_id not in imu_ids, "CSV duplicated an IMU sample ID")
            imu_ids.add(sample_id)
            sample = monitor.samples.get(sample_id)
            require(sample is not None, "CSV IMU sample was not present in incrementally observed UI history")
            motion_type = kind.removeprefix("imu_")
            require(row["imu_type"] == motion_type == sample["type"], "CSV changed IMU stream identity")
            require(generation == sample["capture_generation"], "CSV changed native IMU capture generation")
            require(integer(row["source_frame_number"], "source_frame_number") == sample["frame_number"],
                    "CSV changed native motion frame number")
            require(row["source_frame_id"] == "", "CSV assigned a synthetic video frame ID to asynchronous IMU")
            require(row["source_timestamp_domain"] == sample["timestamp_domain"], "CSV changed native motion clock domain")
            for column, field in (("source_timestamp_ms", "timestamp_ms"),
                                  ("source_received_unix_s", "received_time"),
                                  ("source_received_monotonic_s", "received_monotonic")):
                require(float(row[column]) == sample[field], f"CSV changed native IMU {field}")
            prefix = "accel" if motion_type == "accel" else "gyro"
            suffix = "m_s2" if motion_type == "accel" else "rad_s"
            for axis in ("x", "y", "z"):
                require(finite(row[f"{prefix}_{axis}_{suffix}"], "IMU axis") == sample[axis],
                        "CSV changed a native motion measurement")
                opposite = f"gyro_{axis}_rad_s" if prefix == "accel" else f"accel_{axis}_m_s2"
                require(row[opposite] == "", "CSV filled an unobserved opposite IMU stream")
            require(row["imu_units"] == sample["units"], "CSV changed native IMU units")
            require(bool(row["device_serial"]) and finite(row["imu_profile_fps"], "imu_profile_fps") > 0,
                    "IMU device serial/profile provenance is missing")
            csv_motion.append(sample)
        elif kind == "pose_frame":
            require(event not in pose_events, "Duplicate pose_frame event")
            pose_events[event] = row
            require(integer(row["pose_instance_count"], "pose_instance_count") >= 0, "Negative pose instance count")
        else:
            require(event in pose_events, "Pose joint has no matching pose_frame event")
            parent = pose_events[event]
            for key in ("source_generation", "source_frame_id", "source_timestamp_ms", "source_received_monotonic_s"):
                require(row[key] == parent[key], f"Pose joint changed its parent {key}")
            require(bool(row["joint_name"]), "Pose joint name is missing")
            integer(row["joint_index"], "joint_index")
            finite(row["joint_u_px"], "joint_u_px")
            finite(row["joint_v_px"], "joint_v_px")
            require(row["instance_id_scope"] == "source_frame", "Unexpected persistent instance identity")
            instances.add((event, row["instance_id"]))
            if row["joint_confidence_kind"] == "not_reported_display_mask":
                require(row["joint_confidence"] == "", "CSV presented a display mask as confidence")
            if row["joint_depth_valid"] == "1":
                for key in ("joint_x_m", "joint_y_m", "joint_z_m"):
                    finite(row[key], key)
                valid_3d += 1
            else:
                require(all(row[key] == "" for key in ("joint_x_m", "joint_y_m", "joint_z_m")),
                        "CSV emitted XYZ despite missing/invalid joint depth")
    require(counts == dataset["rows_written"], "Parsed CSV row counts differ from live writer status")
    require(len(rows) == dataset["rows_total"], "Parsed CSV total differs from live writer status")
    require(counts == manifest["dataset"]["rows_written"], "CSV counts differ from manifest")
    require(counts["camera"] >= 10 and counts["imu_accel"] >= 10 and counts["imu_gyro"] >= 10,
            "CSV does not contain actual camera, accel and gyro data")
    require(counts["pose_frame"] > 0, "CSV contains no Pose processing frame events")
    for row in pose_events.values():
        source = (integer(row["source_generation"], "Pose generation"), integer(row["source_frame_id"], "Pose source frame"))
        require(source in camera_sources, "Pose frame has no captured camera event in the same generation")
    in_gate_observed = {sample_id for sample_id, sample in monitor.samples.items()
                        if start <= sample["received_monotonic"] <= end}
    # This is a missing-native-sample check, not just an output-row count check.
    missing = sorted(in_gate_observed - imu_ids)
    extra = sorted(imu_ids - in_gate_observed)
    require(not missing and not extra,
            f"Native IMU sample coverage differs from CSV: missing={missing[:10]}, extra={extra[:10]}")
    profile_list = manifest["camera"]["imu"]["profiles"]
    native_cadence = cadence(csv_motion, profile_list, label="recorded CSV")
    ready_pose_events = sum(row["pose_state"] == "ready" for row in pose_events.values())
    pose_unverified = counts["pose_joint"] == 0
    return {"transport": transport, "rows_total": len(rows), "counts": counts,
            "start_monotonic": start, "stop_monotonic": end,
            "all_source_times_within_exact_gate": True,
            "native_imu_samples_exactly_preserved": True,
            "native_imu_samples": len(imu_ids), "native_cadence": native_cadence,
            "source_generations": sorted(generations),
            "source_domains": [list(pair) for pair in sorted(domains)],
            "camera_sources": len(camera_sources), "pose_frame_events": len(pose_events),
            "pose_ready_events": ready_pose_events, "pose_instances": len(instances),
            "pose_joints": counts["pose_joint"], "valid_depth_pose_joints": valid_3d,
            "pose_joints_unverified": pose_unverified,
            "pose_execution_unverified": ready_pose_events == 0,
            "partial_not_served": assert_private_partial(state["id"])}


def save_report(report):
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(DESTINATION)


def main(recording_seconds=8.):
    report = {"started_unix": time.time(), "ok": False, "core_ok": False,
              "verification_status": "failed", "pose_joints_unverified": True, "checks": {}}
    original = None
    owned_id = None
    monitor = None
    exit_code = 1
    try:
        original = api("/api/status")
        report["initial_status"] = original
        if rec_status()["state"] in ACTIVE:
            report["verification_status"] = "refused"
            raise RuntimeError("An existing user recording is active; it was not interrupted")
        require(original["camera"]["state"] == "streaming" and original["camera"]["active"],
                "Camera is not already streaming; verification will not start or stop the user's camera")
        imu = original.get("imu", original["camera"].get("imu", {}))
        require(imu.get("available") and imu.get("state") == "streaming" and not imu.get("stale"),
                "D435i Motion Module is not receiving fresh accelerometer and gyro data")
        monitor = HistoryMonitor()
        monitor.observe_until(time.monotonic() + 2.)
        initial_samples = list(monitor.samples.values())
        report["checks"]["live_imu_history"] = {
            "initial_id": monitor.initial_id, "latest_id": monitor.cursor,
            "incremental_polls": monitor.calls, "unique_samples": len(initial_samples),
            "capture_generations": sorted(monitor.observed_epochs),
            "no_history_sample_loss": True,
            "cadence": cadence(initial_samples, imu["profiles"], label="two-second live history"),
        }
        print("Live IMU: incremental UI endpoint preserves unique, finite native samples and configured cadence", flush=True)
        before_start = rec_status()
        require(before_start["state"] not in ACTIVE, "A user recording started during preflight; refusing to replace it")
        started = api("/api/recording/start", {})["result"]
        owned_id = started["id"]
        report["recording_id"] = owned_id
        requested = finite(started["requested_monotonic"], "requested_monotonic")
        deadline = finite(started["start_monotonic"], "start_monotonic")
        require(deadline - requested >= 5. - 1e-8, "Server recording delay is less than five seconds")
        monitor.observe_until(requested + 4.)
        at_four = rec_status()
        require(at_four["id"] == owned_id, "Verification recording was replaced externally")
        require(time.monotonic() < deadline, "Four-second checkpoint arrived after the five-second deadline")
        require(at_four["state"] == "countdown" and not any(at_four["frames_written"].values()),
                "Videos wrote frames before the five-second delay")
        require(at_four["dataset"]["rows_total"] == 0 and not any(at_four["dataset"]["rows_written"].values()),
                "CSV wrote rows before the five-second delay")
        report["checks"]["four_second_countdown"] = {"recording_status": at_four,
            "partial_not_served": assert_private_partial(owned_id)}
        print("At four seconds: zero CSV rows, zero video frames, partial CSV not downloadable", flush=True)
        while time.monotonic() < deadline + recording_seconds:
            monitor.observe_until(min(deadline + recording_seconds, time.monotonic() + .2))
            state = rec_status()
            require(state["id"] == owned_id and state["state"] in {"countdown", "recording"},
                    "Verification recording stopped or was replaced before its requested capture window")
        state = rec_status()
        require(state["id"] == owned_id and state["state"] == "recording", "Recording did not become active")
        report["checks"]["before_manual_stop"] = state
        api("/api/recording/stop", {})
        stop_deadline = time.monotonic() + 60.
        while time.monotonic() < stop_deadline:
            monitor.poll()
            completed = rec_status()
            require(completed["id"] == owned_id, "Recording changed while finishing; no unrelated recording was stopped")
            if completed["state"] in TERMINAL:
                break
            time.sleep(.2)
        else:
            raise RuntimeError("Timed out finalizing the verification recording")
        require(completed["state"] == "completed", "Verification recording failed to finalize: " + str(completed.get("error")))
        report["final_recording"] = completed
        report["checks"]["dataset_csv"] = inspect_csv(completed, monitor)
        print("CSV: download/disk hash, source clocks, units, IDs, row counts and exact native IMU coverage verified", flush=True)
        report["checks"]["videos"] = inspect_recording(completed)
        print("Videos: all finalized MP4s decode first/middle/last; dimensions, source timing, HEAD/Range verified", flush=True)
        dataset_check = report["checks"]["dataset_csv"]
        report["pose_joints_unverified"] = dataset_check["pose_joints_unverified"]
        report["pose_execution_unverified"] = dataset_check["pose_execution_unverified"]
        report["core_ok"] = True
        report["ok"] = not report["pose_joints_unverified"] and not report["pose_execution_unverified"]
        report["verification_status"] = "passed" if report["ok"] else "partial"
        if not report["ok"]:
            report["unverified_reason"] = "No actual detected Pose joints/ready inference were available; no joints were fabricated"
            print("PARTIAL: sensor/CSV/video checks passed; actual detected Pose joints remain unverified", flush=True)
            exit_code = 2
        else:
            print("Actual Pose joints were preserved in CSV and mapped to their captured RGB-D source frames", flush=True)
            exit_code = 0
    except Exception as error:
        report["error"] = str(error)
        report["exception_type"] = type(error).__name__
        report["traceback"] = traceback.format_exc()
        print("Dataset verification failed: " + str(error), flush=True)
    finally:
        # Never stop a pre-existing, replacement or unknown user's recording.
        if owned_id is not None:
            try:
                current = rec_status()
                if current.get("id") == owned_id and current.get("state") in ACTIVE:
                    api("/api/recording/stop", {})
                    cleanup_deadline = time.monotonic() + 60.
                    while time.monotonic() < cleanup_deadline:
                        current = rec_status()
                        if current.get("id") != owned_id or current.get("state") in TERMINAL:
                            break
                        time.sleep(.2)
                    require(current.get("id") == owned_id and current.get("state") in TERMINAL,
                            "Owned verification recording could not be finalized during cleanup")
                    report["cleanup_recording"] = current
            except Exception as error:
                report.update(ok=False, core_ok=False, verification_status="failed", cleanup_error=str(error))
                exit_code = 1
        try:
            final = api("/api/status")
            report["final_status"] = final
            if original is not None:
                unchanged = final["settings"] == original["settings"]
                report["live_settings_unchanged"] = unchanged
                # The script never changes settings and must not overwrite
                # independent user changes while it is running.
                if not unchanged:
                    report["settings_note"] = "Live settings changed externally; verification did not overwrite the user's choice"
            if original is not None and original["camera"]["active"]:
                require(final["camera"]["active"], "Camera stopped during verification; script did not stop or restart it")
        except Exception as error:
            report.update(ok=False, verification_status="failed", final_status_error=str(error))
            exit_code = 1
        report["finished_unix"] = time.time()
        save_report(report)
        print("Report: " + str(DESTINATION), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
