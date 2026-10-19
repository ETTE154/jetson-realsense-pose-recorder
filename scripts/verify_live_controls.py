#!/usr/bin/env python3
"""Real D435i live checks: pose off/on and camera stop/restart, with evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8766"


def api(path, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(BASE + path, data=body,
                                     headers={"Content-Type": "application/json"} if body else {})
    with urllib.request.urlopen(request, timeout=12) as response:
        return json.load(response)


def wait(predicate, timeout=35):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = api("/api/status")
        if predicate(status):
            return status
        time.sleep(.1)
    raise RuntimeError("Live control verification timed out: " + json.dumps(status))


def workers():
    rows = []
    for line in subprocess.check_output(["ps", "-eo", "pid,ppid,args"], text=True).splitlines()[1:]:
        pieces = line.strip().split(None, 2)
        if len(pieces) == 3 and str(ROOT / "live_worker.py") in pieces[2]:
            rows.append({"pid": int(pieces[0]), "ppid": int(pieces[1]), "command": pieces[2]})
    return rows


def read_exact(response, size):
    blocks = []
    while size:
        value = response.read(size)
        if not value:
            raise RuntimeError("MJPEG stream ended inside JPEG")
        blocks.append(value)
        size -= len(value)
    return b"".join(blocks)


def frames(name):
    output = []
    with urllib.request.urlopen(BASE + f"/stream/{name}.mjpg", timeout=8) as response:
        while len(output) < 3:
            boundary = response.readline()
            if not boundary:
                raise RuntimeError(f"{name} MJPEG ended before 3 frames")
            if boundary.strip() != b"--frame":
                continue
            headers = {}
            while True:
                line = response.readline()
                if line in (b"\r\n", b"\n"):
                    break
                key, value = line.decode().split(":", 1)
                headers[key.lower()] = value.strip()
            jpeg = read_exact(response, int(headers["content-length"]))
            image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape != (480, 640, 3):
                raise RuntimeError(f"{name} invalid decoded frame")
            output.append({"sequence": int(headers["x-sequence"]), "shape": list(image.shape),
                           "jpeg_bytes": len(jpeg), "sha256": hashlib.sha256(jpeg).hexdigest()})
    sequences = [frame["sequence"] for frame in output]
    if not all(a < b for a, b in zip(sequences, sequences[1:])):
        raise RuntimeError(f"{name} repeated frame sequence: {sequences}")
    return output


def main():
    report = {"started_unix": time.time(), "checks": {}, "ok": False}
    original = api("/api/status")
    if original["settings"]["pose_model"] != "rtmpose" or not original["settings"]["pose_enabled"]:
        raise RuntimeError("Expected active RTMPose before control verification")
    baseline_workers = workers()
    parents = {row["ppid"] for row in baseline_workers}
    if len(parents) != 1:
        raise RuntimeError("Cannot identify main live worker owner")
    server_pid = next(iter(parents))
    own_workers = lambda: [row for row in workers() if row["ppid"] == server_pid]
    report.update(initial_status=original, server_pid=server_pid, initial_workers=baseline_workers)
    try:
        api("/api/settings", {"pose_enabled": False})
        disabled = wait(lambda status: status["pose"]["state"] == "disabled")
        time.sleep(.2)
        disabled_workers = own_workers()
        if disabled_workers:
            raise RuntimeError("Disabled Pose retained live worker: " + str(disabled_workers))
        off = {"status": disabled, "workers": disabled_workers,
               "streams": {name: frames(name) for name in ("pose", "pose_rgbd")}}
        snapshot = api("/api/snapshot", {})
        captured = ROOT / "captures" / snapshot["id"]
        for first, second in (("rgb", "pose"), ("rgbd", "pose_rgbd")):
            if (captured / f"{first}.jpg").read_bytes() != (captured / f"{second}.jpg").read_bytes():
                raise RuntimeError("Disabled Pose fallback differs from its source background")
        off.update(snapshot_id=snapshot["id"], rgb_equals_pose=True, rgbd_equals_pose_rgbd=True)
        report["checks"]["pose_off"] = off
        print("Pose off: two MJPEG streams advance; source/fallback bytes equal; zero owned workers", flush=True)

        api("/api/settings", {"pose_enabled": True})
        restored = wait(lambda status: status["pose"]["state"] == "ready" and status["pose"]["model_id"] == "rtmpose")
        if len(own_workers()) != 1:
            raise RuntimeError("Expected exactly one live worker after re-enable")
        report["checks"]["pose_on"] = {"status": restored, "workers": own_workers(),
                                               "streams": {"pose": frames("pose")}}
        print("Pose on: RTMPose GPU ready; stream advances; exactly one worker", flush=True)

        before_stop = api("/api/status")["camera"]["frame_id"]
        api("/api/camera/stop", {})
        stopped = wait(lambda status: not status["camera"]["active"] and status["camera"]["state"] != "streaming")
        deadline = time.monotonic() + 5
        while own_workers() and time.monotonic() < deadline:
            time.sleep(.05)
        if own_workers():
            raise RuntimeError("Stopped camera retained live model worker")
        report["checks"]["camera_stop"] = {"status": stopped, "workers": own_workers()}
        print("Camera stop: inactive and zero owned live workers", flush=True)

        api("/api/camera/start", {})
        restarted = wait(lambda status: status["camera"]["state"] == "streaming" and
                         status["pose"]["state"] == "ready" and status["pose"]["model_id"] == "rtmpose" and
                         status["pose"]["frame_id"] > before_stop, timeout=60)
        active_workers = own_workers()
        if len(active_workers) != 1 or "--candidate rtmpose" not in active_workers[0]["command"]:
            raise RuntimeError("Restart did not retain exactly one RTMPose worker")
        report["checks"]["camera_restart"] = {"status": restarted, "workers": active_workers,
                                               "streams": {name: frames(name) for name in ("rgb", "depth", "rgbd", "pose", "pose_rgbd")}}
        report["ok"] = True
        print("Camera restart: new RTMPose frames; all five MJPEG streams advance and decode", flush=True)
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        # Restore the requested ready RTMPose state even if a verification fails.
        status = api("/api/status")
        if not status["settings"]["pose_enabled"]:
            api("/api/settings", {"pose_enabled": True})
        if not status["camera"]["active"]:
            api("/api/camera/start", {})
        report["finished_unix"] = time.time()
        report["final_status"] = api("/api/status")
        (ROOT / "validation/live_controls_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
