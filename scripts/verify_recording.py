#!/usr/bin/env python3
"""Real-camera recording checks. Restores the selected live model afterward."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.error
import urllib.request

import cv2

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8766"
STREAMS = ("rgb", "depth", "rgbd", "pose", "pose_rgbd", "combined", "imu")
TERMINAL = {"completed", "cancelled", "canceled", "error"}


def api(path, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(BASE + path, data=body,
                                     headers={"Content-Type": "application/json"} if body is not None else {})
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.load(response)


def recording():
    return api("/api/recording/status")


def frames(state):
    return state.get("frames_written", state.get("frames", {}))


def poll(getter, predicate, timeout=45):
    deadline = time.monotonic()+timeout
    last = None
    while time.monotonic() < deadline:
        last = getter()
        if predicate(last):
            return last
        time.sleep(.1)
    raise RuntimeError("Verification timed out: " + json.dumps(last))


def live_model(identifier):
    api("/api/settings", {"pose_enabled": True, "pose_model": identifier})
    return poll(lambda: api("/api/status"), lambda status: status["camera"]["state"] == "streaming" and
                status["pose"]["state"] == "ready" and status["pose"]["model_id"] == identifier,
                timeout=150)


def start():
    return api("/api/recording/start", {})["result"]


def stop():
    api("/api/recording/stop", {})
    value = poll(recording, lambda state: state["state"] in TERMINAL, timeout=45)
    if value["state"] == "error":
        raise RuntimeError("Recording failed: " + json.dumps(value))
    return value


def inspect_recording(state):
    directory = ROOT / "recordings" / state["id"]
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines() if line]
    if not rows:
        raise RuntimeError("Recording sidecar contains no output frames")
    start_time, stop_time = manifest["start_monotonic"], manifest["stop_monotonic"]
    if start_time - manifest["requested_monotonic"] < 5 - 1e-6:
        raise RuntimeError("Recording deadline is earlier than the five-second delay")
    sources = []
    for index, row in enumerate(rows):
        if row["frame_index"] != index:
            raise RuntimeError("Recording frame indices are not contiguous")
        for key in ("sensor_source", "pose_source", "combined_source"):
            source = row.get(key)
            if source:
                captured = source["received_monotonic"]
                if captured < start_time - 1e-6 or captured > stop_time + 1e-6:
                    raise RuntimeError(f"{key} capture is outside recording interval")
                sources.append(captured)
        if row.get("combined_pose_gap") is None and row.get("combined_source"):
            if row["combined_source"] != row.get("pose_source"):
                raise RuntimeError("Combined tiles do not share the skeleton source frame")
    if not sources:
        raise RuntimeError("Recording contains no real camera source frames")
    checks = {"id": state["id"], "manifest": manifest, "sidecar_rows": len(rows),
              "earliest_source_monotonic": min(sources), "deadline_monotonic": start_time,
              "all_sources_after_deadline": True, "videos": {}}
    for name in STREAMS:
        if name not in manifest["files"]:
            continue  # Older recordings predate IMU video.
        path = directory / f"{name}.mp4"
        if not path.is_file() or path.stat().st_size < 1000:
            raise RuntimeError(f"Finalized {name} MP4 is missing")
        capture = cv2.VideoCapture(str(path))
        dimensions = manifest["output_dimensions"][name]
        expected_shape = (dimensions["height"], dimensions["width"], 3)
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or count != len(rows):
                raise RuntimeError(f"{name} FPS/frame count does not match sidecar")
            duration = count/fps
            expected = manifest.get("recorded_duration_s", state.get("duration_s"))
            if expected is not None and abs(duration-expected) > 1/fps + .01:
                raise RuntimeError(f"{name} duration differs from wall-clock manifest")
            source_duration = manifest.get("source_duration_s", stop_time-start_time)
            if abs(duration-source_duration) > 1/fps + .1:
                raise RuntimeError(f"{name} playback speed does not preserve wall-clock time")
            decoded = []
            for index in sorted({0, count//2, count-1}):
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, image = capture.read()
                if not ok or image.shape != expected_shape:
                    raise RuntimeError(f"Cannot decode {name} frame {index}")
                decoded.append({"frame_index": index, "shape": list(image.shape),
                                "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest()})
        finally:
            capture.release()
        entry = {"bytes": path.stat().st_size, "fps": fps, "frames": count,
                 "duration_s": duration, "decoded_samples": decoded}
        executable = shutil.which("ffprobe")
        if executable:
            entry["ffprobe"] = json.loads(subprocess.check_output(
                [executable, "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=codec_name,width,height,r_frame_rate,nb_frames,duration", "-of", "json", str(path)],
                text=True, timeout=15))
        url = BASE + f"/recordings/{state['id']}/{name}.mp4"
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=8) as response:
            if int(response.headers["Content-Length"]) != path.stat().st_size:
                raise RuntimeError("Video HEAD size mismatch")
            entry["head_accept_ranges"] = response.headers.get("Accept-Ranges")
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-31"}), timeout=8) as response:
            body = response.read()
            if response.status != 206 or body != path.read_bytes()[:32]:
                raise RuntimeError("Video range seeking response failed")
            entry["range_status"] = response.status
            entry["content_range"] = response.headers.get("Content-Range")
        checks["videos"][name] = entry
    return checks


def main():
    report = {"started_unix": time.time(), "ok": False, "checks": {}}
    original = api("/api/status")
    original_model = original["settings"]["pose_model"]
    original_enabled = original["settings"]["pose_enabled"]
    report["original_model"] = original_model
    if recording()["state"] in {"countdown", "recording", "stopping"}:
        raise RuntimeError("An existing user recording is active; verification will not interrupt it")
    try:
        live_model("rtmpose")
        cancelled_start = start()
        time.sleep(.7)
        cancelled = stop()
        if cancelled["state"] not in {"cancelled", "canceled"} or any(frames(cancelled).values()):
            raise RuntimeError("Countdown cancellation created recorded frames")
        directory = ROOT / "recordings" / cancelled_start["id"]
        if any(directory.glob("*.mp4")):
            raise RuntimeError("Countdown cancellation created finalized videos")
        report["checks"]["countdown_cancel"] = cancelled
        print("Countdown cancellation: zero video frames and no finalized MP4", flush=True)

        beginning = start()
        requested_at = time.monotonic()
        duplicate_status = None
        try:
            start()
        except urllib.error.HTTPError as error:
            duplicate_status = error.code
        if duplicate_status not in (400, 409):
            raise RuntimeError("Duplicate recording start was accepted")
        deadline = requested_at + 4
        while time.monotonic() < deadline:
            time.sleep(max(0., min(.1, deadline-time.monotonic())))
        at_four = recording()
        if at_four["state"] != "countdown" or any(frames(at_four).values()):
            raise RuntimeError("Recording wrote frames before the five-second delay")
        if at_four["id"] != beginning["id"]:
            raise RuntimeError("Duplicate start replaced the active recording")
        print("Four-second check: countdown and zero output frames; duplicate start rejected", flush=True)
        active = poll(recording, lambda state: state["state"] == "recording" and frames(state).get("rgb", 0) >= 5, timeout=15)
        before_switch = frames(active)["rgb"]
        switched = live_model("mediapipe")
        time.sleep(1)
        after_switch = recording()
        if after_switch["id"] != beginning["id"] or frames(after_switch).get("rgb", 0) <= before_switch:
            raise RuntimeError("RGB recording stopped during model switching")
        completed = stop()
        if completed["state"] != "completed":
            raise RuntimeError("Recording did not complete")
        repeated = stop()
        if repeated["id"] != completed["id"] or repeated["state"] != "completed":
            raise RuntimeError("Repeated stop changed the finalized recording")
        report["checks"]["manual_stop"] = inspect_recording(completed)
        report["checks"]["manual_stop"].update(at_four_seconds=at_four, duplicate_start_status=duplicate_status,
                                                  rgb_frames_before_switch=before_switch,
                                                  rgb_frames_after_switch=frames(after_switch)["rgb"],
                                                  switched_pose_status=switched["pose"])
        print("Manual recording: six MP4s including combined decode, wall-time duration and source deadline hold, model switch keeps RGB advancing", flush=True)

        live_model("rtmpose")
        camera_recording = start()
        poll(recording, lambda state: state["state"] == "recording" and frames(state).get("rgb", 0) >= 5, timeout=20)
        time.sleep(.6)
        api("/api/camera/stop", {})
        automatically_stopped = poll(recording, lambda state: state["state"] in TERMINAL, timeout=30)
        if automatically_stopped["id"] != camera_recording["id"] or automatically_stopped["state"] != "completed":
            raise RuntimeError("Camera stop failed to finalize recording")
        if automatically_stopped.get("reason") not in {"camera_stopped", "camera_disconnected"}:
            raise RuntimeError("Camera stop reason was not preserved")
        report["checks"]["camera_stop"] = inspect_recording(automatically_stopped)
        api("/api/camera/start", {})
        live_model("rtmpose")
        print("Camera stop: recording finalized and six videos remain decodable; camera restarted", flush=True)
        report["ok"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        restore_error = None
        try:
            if recording()["state"] in {"countdown", "recording", "stopping"}:
                stop()
            if not api("/api/status")["camera"]["active"]:
                api("/api/camera/start", {})
            live_model(original_model)
            if not original_enabled:
                api("/api/settings", {"pose_enabled": False})
            report["final_status"] = api("/api/status")
        except Exception as error:
            restore_error = error
            report.update(ok=False, restore_error=str(error))
        finally:
            report["finished_unix"] = time.time()
            destination = ROOT / "validation/recording_live_report.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report, indent=2) + "\n")
        if restore_error:
            raise RuntimeError("Verification cleanup could not restore the original live model") from restore_error


if __name__ == "__main__":
    main()
