"""Bounded asynchronous H.264 recorder with a five-second capture-clock gate.

Individual CFR videos and a 2x2 composite share one monotonic timeline. Pose images retain their own RGB
source, low-rate results hold whole images, and gaps are explicit black frames.
No camera operation or model inference is performed by this module.
"""
from __future__ import annotations

import copy
from datetime import datetime
import heapq
import json
import math
from pathlib import Path
import queue
import shutil
import threading
import time
import uuid

import cv2
import numpy as np

from dataset_csv import DatasetCSV

STREAMS = ("rgb", "depth", "rgbd", "pose", "pose_rgbd")
SENSOR_STREAMS = STREAMS[:3]
POSE_STREAMS = STREAMS[3:]
OUTPUT_STREAMS = STREAMS + ("combined",)
COMBINED_TILES = ("rgb", "depth", "rgbd", "pose")
TERMINAL = {"completed", "cancelled", "error"}


def output_size(name, width, height):
    return (width * 2, height * 2) if name == "combined" else (width, height)


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _line(handle, value):
    handle.write(json.dumps(_safe(value), ensure_ascii=False, allow_nan=False) + "\n")


def _source(frame):
    output = {key: getattr(frame, key, None) for key in
              ("frame_id", "timestamp_ms", "timestamp_domain", "received_time", "received_monotonic",
               "capture_generation", "color_frame_number", "depth_frame_number",
               "depth_timestamp_ms", "depth_timestamp_domain")}
    output["depth_scale_m_per_unit"] = getattr(frame, "depth_scale", None)
    intrinsics = getattr(frame, "intrinsics", None)
    if intrinsics is not None:
        output["intrinsics"] = {key: getattr(intrinsics, key, None) for key in
                                ("width", "height", "ppx", "ppy", "fx", "fy", "model", "coeffs")}
    return _safe(output)


def _combined_frame(views, width, height, elapsed, source, model):
    """Copy four full-resolution panels without changing their aspect ratio."""
    canvas = np.empty((height * 2, width * 2, 3), np.uint8)
    for index, name in enumerate(COMBINED_TILES):
        y, x = (index // 2) * height, (index % 2) * width
        panel = canvas[y:y + height, x:x + width]
        panel[:] = views[name]
        if width >= 160 and height >= 100:
            cv2.rectangle(panel, (0, 0), (width - 1, 27), (15, 23, 29), -1)
            label = {"rgb": "RGB", "depth": "DEPTH", "rgbd": "RGB-D", "pose": "POSE"}[name]
            if name == "pose":
                label += " / " + str(model or "waiting")
            frame_id = source.get("frame_id") if source else None
            label += f"   frame {frame_id if frame_id is not None else '-'}   {elapsed:.2f}s"
            cv2.putText(panel, label, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, .48,
                        (232, 242, 246), 1, cv2.LINE_AA)
    return canvas


class Recorder:
    def __init__(self, output_dir, fps=15.0, countdown_seconds=5.0, queue_size=48,
                 write_delay_s=1.0, *, clock=None, wall_clock=None,
                 writer_factory=None, min_free_bytes=256 * 1024**2, pose_stale_after_s=2.0):
        self.output_dir = Path(output_dir).resolve()
        self.fps = float(fps)
        self.countdown_seconds = float(countdown_seconds)
        self.queue_size = int(queue_size)
        self.write_delay_s = float(write_delay_s)
        if not 1 <= self.fps <= 30 or self.countdown_seconds < 0 or self.queue_size < 2 or self.write_delay_s < 0:
            raise ValueError("Invalid recorder FPS, countdown, queue size, or write delay")
        self._clock = clock or time.monotonic
        self._wall = wall_clock or time.time
        self._writer_factory = writer_factory or (lambda path, fps, size: cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, size))
        self.min_free_bytes = int(min_free_bytes)
        self.pose_stale_after_s = float(pose_stale_after_s)
        self._lock = threading.RLock()
        self._session = None
        self._closed = False

    def start(self, camera=None, settings=None):
        with self._lock:
            if self._closed:
                raise ValueError("Recorder is closed")
            previous = self._session
            if previous is not None and (previous.state not in TERMINAL or previous.worker.is_alive()):
                raise ValueError("이미 녹화 또는 파일 마무리가 진행 중입니다.")
            from types import SimpleNamespace
            now, wall = self._clock(), self._wall()
            camera = copy.deepcopy(camera or {})
            width, height = int(camera.get("width", 640)), int(camera.get("height", 480))
            if width < 16 or height < 16 or width % 2 or height % 2:
                raise ValueError("Recording dimensions must be positive even numbers")
            identifier = datetime.fromtimestamp(wall).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
            session = SimpleNamespace(
                id=identifier, directory=self.output_dir / identifier, state="countdown",
                requested_mono=now, requested_wall=wall, start_mono=now + self.countdown_seconds,
                start_wall=wall + self.countdown_seconds, end_mono=None, reason=None, error=None,
                camera=camera, settings=copy.deepcopy(settings or {}), width=width, height=height,
                queue=queue.Queue(maxsize=self.queue_size), wake=threading.Event(), stop_requested=False,
                accepting=True, pose_epoch=now, sequence=0, written=0, files={},
                accepted={"sensor": 0, "pose": 0}, dropped={"sensor": 0, "pose": 0},
                excluded_before_deadline=0, rejected_invalid=0, late_packets=0,
                controls_dropped=0, final=None, worker=None, dataset=None, pose_csv_lock=threading.Lock(),
                performance={"processed_frames": 0, "total_frame_ms": 0., "max_frame_ms": 0.,
                             "max_backlog_s": 0., "max_queue_depth": 0},
            )
            session.worker = threading.Thread(target=self._run, args=(session,), name=f"record-{identifier}", daemon=True)
            self._session = session
            session.worker.start()
            return self._status_locked(session)

    def _status_locked(self, session):
        if session is None:
            return {"state": "idle", "id": None, "remaining_s": 0., "remaining_seconds": 0.,
                    "duration_s": 0., "elapsed_seconds": 0., "frames_written": {name: 0 for name in OUTPUT_STREAMS},
                    "frames": {name: 0 for name in OUTPUT_STREAMS}, "files": {}, "error": None,
                    "dataset": {"state": "idle", "rows_total": 0, "rows_written": {}, "dropped_total": 0, "error": None}}
        now = self._clock()
        duration = max(0., (session.end_mono if session.end_mono is not None else now) - session.start_mono)
        remaining = max(0., session.start_mono - now) if session.state == "countdown" else 0.
        frames = {name: session.written for name in OUTPUT_STREAMS}
        return {"state": session.state, "id": session.id, "remaining_s": round(remaining, 3),
                "remaining_seconds": round(remaining, 3), "duration_s": round(duration, 3),
                "elapsed_seconds": round(duration, 3), "source_duration_s": round(duration, 6),
                "recorded_duration_s": session.written / self.fps, "fps": self.fps, "video_fps": self.fps,
                "width": session.width, "height": session.height, "frames_written": frames, "frames": frames.copy(),
                "output_dimensions": {name: dict(zip(("width", "height"), output_size(name, session.width, session.height)))
                                      for name in OUTPUT_STREAMS},
                "combined_layout": {"rows": 2, "columns": 2, "tiles": list(COMBINED_TILES),
                                    "synchronization": "All four views use the Pose source frame; while Pose is unavailable, sensor views continue with a blank Pose tile."},
                "imu_visualization": {"enabled": False,
                                      "reason": "IMU movie recording disabled to prioritize video continuity; raw IMU remains in CSV and live UI"},
                "queue_depth": session.queue.qsize(), "queue_limit": self.queue_size,
                "performance": {**session.performance, "mean_frame_ms": session.performance["total_frame_ms"] /
                                max(1, session.performance["processed_frames"]), "frame_budget_ms": 1000. / self.fps},
                "accepted_sensor_frames": session.accepted["sensor"], "accepted_pose_frames": session.accepted["pose"],
                "dropped_frames": session.dropped.copy(), "dropped_total": sum(session.dropped.values()),
                "excluded_before_deadline": session.excluded_before_deadline,
                "late_packets": session.late_packets, "rejected_invalid": session.rejected_invalid,
                "files": session.files.copy(), "error": session.error, "reason": session.reason,
                "dataset": session.dataset.status() if session.dataset else {"state": session.state, "rows_total": 0, "rows_written": {}, "dropped_total": 0, "error": None},
                "start_monotonic": session.start_mono, "requested_monotonic": session.requested_mono,
                "stop_monotonic": session.end_mono, "requested_at": session.requested_wall,
                "scheduled_start_at": session.start_wall, "codec": "H.264 / avc1", "container": "MP4"}

    def status(self):
        with self._lock:
            return self._status_locked(self._session)

    def _submit(self, kind, frame, images, settings, people=None, pose=None):
        try:
            source_mono = float(getattr(frame, "received_monotonic", 0.0))
        except (TypeError, ValueError):
            source_mono = 0.
        with self._lock:
            session = self._session
            if session is None or not session.accepting:
                return False
            now = self._clock()
            if not math.isfinite(source_mono) or source_mono <= 0 or source_mono > now + .05:
                session.rejected_invalid += 1
                return False
            # Both gates are mandatory. A late result from a pre-countdown frame
            # can never become an accepted recording image.
            if now < session.start_mono or source_mono < session.start_mono:
                session.excluded_before_deadline += 1
                return False
            if kind == "pose" and source_mono < session.pose_epoch:
                session.dropped[kind] += 1
                return False
        names = SENSOR_STREAMS if kind == "sensor" else POSE_STREAMS
        # Keep the composite backgrounds paired with the skeleton's own source.
        if kind == "pose" and isinstance(images, dict) and all(name in images for name in SENSOR_STREAMS):
            names = POSE_STREAMS + SENSOR_STREAMS
        frozen = {}
        for name in names:
            data = images.get(name) if isinstance(images, dict) else None
            if isinstance(data, bytes):
                frozen[name] = data
            elif isinstance(data, (bytearray, memoryview)):
                frozen[name] = bytes(data)
            elif isinstance(data, np.ndarray) and data.dtype == np.uint8 and data.ndim == 3 and data.shape[2] == 3:
                frozen[name] = data.copy()
            else:
                with self._lock:
                    session.rejected_invalid += 1
                return False
        packet = {"kind": kind, "mono": source_mono, "source": _source(frame), "images": frozen,
                  "settings": copy.deepcopy(settings or {}), "people": copy.deepcopy(people), "pose": copy.deepcopy(pose)}
        with self._lock:
            # Atomic with stop() and start(): copying outside the lock never
            # permits a stopped or replaced session to accept another packet.
            if self._session is not session or not session.accepting:
                return False
            if kind == "pose" and source_mono < session.pose_epoch:
                session.dropped[kind] += 1
                return False
            session.sequence += 1
            packet["sequence"] = session.sequence
            try:
                session.queue.put_nowait(packet)
            except queue.Full:
                session.dropped[kind] += 1
                return False
            session.accepted[kind] += 1
            session.performance["max_queue_depth"] = max(session.performance["max_queue_depth"], session.queue.qsize())
            session.wake.set()
            return True

    def submit_sensor(self, frame, images, settings):
        return self._submit("sensor", frame, images, settings)

    def submit_camera_data(self, frame, settings=None):
        """Called at the native RGB-D rate, independently of video encoding."""
        with self._lock:
            session = self._session
            dataset = session.dataset if session is not None and session.accepting else None
            serial = session.camera.get("device", {}).get("serial") if session else None
        if dataset is None:
            return False
        return dataset.submit_camera(frame, settings=settings, video_refs={
            "device_serial": serial, "rgb_video_file": "rgb.mp4",
            "combined_video_file": "combined.mp4", "video_mapping_file": "frames.jsonl",
            "video_mapping_source_frame_id": getattr(frame, "frame_id", None),
            "video_mapping_source_received_monotonic_s": getattr(frame, "received_monotonic", None)})

    def submit_imu(self, sample):
        with self._lock:
            session = self._session
            dataset = session.dataset if session is not None and session.accepting else None
            if dataset is None:
                return False
            serial = session.camera.get("device", {}).get("serial")
            profiles = session.camera.get("imu", {}).get("profiles", [])
            fps = next((p.get("fps") for p in profiles if p.get("type") == sample.get("type")), None)
        return dataset.submit_imu({**sample, "serial": serial, "fps": fps})

    def submit_pose(self, frame, images, people, pose, settings):
        # Preserve model output even if the independent video queue is full.
        with self._lock:
            session = self._session
        if session is not None:
            # Serialize only Pose copying against invalidation. Native sensor
            # callbacks never acquire this lock, and the recorder lock stays free.
            with session.pose_csv_lock:
                with self._lock:
                    dataset = session.dataset if (session is self._session and session.accepting and
                        getattr(frame, "received_monotonic", 0.) >= session.pose_epoch) else None
                if dataset is not None:
                    dataset.submit_pose(frame, people, pose=pose, settings=settings)
        return self._submit("pose", frame, images, settings, people, pose)

    def invalidate_pose(self, reason="model_changed", settings=None):
        with self._lock:
            session = self._session
        if session is None:
            return False
        with session.pose_csv_lock:
            return self._invalidate_pose(session, reason, settings)

    def _invalidate_pose(self, session, reason, settings):
        with self._lock:
            if session is not self._session or not session.accepting:
                return False
            now = self._clock()
            session.pose_epoch = now
            session.sequence += 1
            packet = {"kind": "invalidate", "mono": max(session.start_mono, now),
                      "reason": str(reason), "settings": copy.deepcopy(settings or {}), "sequence": session.sequence}
            try:
                session.queue.put_nowait(packet)
            except queue.Full:
                # Control events cannot be lost behind data. Drop one queued
                # image instead; neither API call waits for the encoder.
                try:
                    removed = session.queue.get_nowait()
                    if removed["kind"] in session.dropped:
                        session.dropped[removed["kind"]] += 1
                    else:
                        session.controls_dropped += 1
                        packet["coalesced_through_monotonic"] = max(
                            packet.get("coalesced_through_monotonic", packet["mono"]),
                            removed.get("coalesced_through_monotonic", removed["mono"]))
                        packet["mono"] = min(packet["mono"], removed["mono"])
                        packet["reason"] = "control_events_coalesced"
                except queue.Empty:
                    pass
                session.queue.put_nowait(packet)
            session.wake.set()
            return True

    def stop(self, reason="user", wait=False):
        with self._lock:
            session = self._session
            if session is None or session.state in TERMINAL:
                return self._status_locked(session)
            if not session.stop_requested:
                session.accepting = False
                session.stop_requested = True
                session.end_mono = self._clock()
                session.reason = str(reason)
                session.state = "stopping"
                session.wake.set()
            worker = session.worker
            dataset = session.dataset
            cutoff = session.end_mono
        if dataset is not None:
            dataset.stop(cutoff, wait=False)
        if wait and worker is not threading.current_thread():
            worker.join(timeout=30.)
        return self.status()

    def close(self):
        with self._lock:
            self._closed = True
        return self.stop("app_shutdown", wait=True)

    def list_recordings(self, limit=20):
        rows = []
        if self.output_dir.is_dir():
            for path in sorted(self.output_dir.glob("*/manifest.json"), reverse=True)[:max(1, min(int(limit), 100))]:
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    rows.append({key: data.get(key) for key in ("id", "state", "duration_s", "recorded_duration_s", "files", "reason", "frames_written", "dataset", "imu_visualization")})
                except (OSError, ValueError):
                    continue
        return rows

    def _decode(self, packet, session):
        output = {}
        for name, data in packet["images"].items():
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if isinstance(data, bytes) else data
            if image is None or image.shape != (session.height, session.width, 3):
                raise ValueError(f"Invalid {name} recording image dimensions")
            output[name] = np.ascontiguousarray(image)
        return output

    def _run(self, session):
        writers, handles = {}, {}
        pending = []
        latest = {"sensor": None, "pose": None}
        settings = copy.deepcopy(session.settings)
        pose_gap = "waiting_for_pose"
        pose_valid_after = session.start_mono
        previous_ids = {"sensor": None, "pose": None, "combined": None}
        serial = 0
        error = None
        cancelled = False
        next_tick = 0
        last_check = self._clock()
        try:
            session.directory.mkdir(parents=True, exist_ok=False)
            dataset = DatasetCSV(session.directory / "dataset.csv", session.start_mono,
                                 recording_id=session.id, clock=self._clock)
            with self._lock:
                session.dataset = dataset
                if session.stop_requested:
                    dataset.stop(session.end_mono, wait=False)
            handles["events"] = (session.directory / "events.jsonl").open("w", encoding="utf-8")
            _line(handles["events"], {"event": "countdown_requested", "monotonic": session.requested_mono,
                                     "start_monotonic": session.start_mono, "settings": settings})
            while self._clock() < session.start_mono and not session.stop_requested:
                session.wake.wait(min(.05, max(0., session.start_mono - self._clock())))
                session.wake.clear()
            if session.stop_requested and session.end_mono < session.start_mono:
                cancelled = True
            else:
                if shutil.disk_usage(session.directory).free < self.min_free_bytes:
                    raise OSError("녹화 파일을 저장할 디스크 여유 공간이 부족합니다.")
                for name in OUTPUT_STREAMS:
                    temporary = session.directory / f"{name}.partial.mp4"
                    temporary.touch(exist_ok=False)
                    writer = self._writer_factory(temporary, self.fps, output_size(name, session.width, session.height))
                    writers[name] = writer
                    if not writer.isOpened():
                        raise RuntimeError("H.264(avc1) 녹화 코덱을 열 수 없습니다.")
                for name in ("frames", "poses", "sensor"):
                    handles[name] = (session.directory / f"{name}.jsonl").open("w", encoding="utf-8")
                with self._lock:
                    if not session.stop_requested:
                        session.state = "recording"
                _line(handles["events"], {"event": "recording_started", "monotonic": session.start_mono,
                                         "actual_writer_ready_monotonic": self._clock()})
                blank_sensor = np.zeros((session.height, session.width, 3), np.uint8)
                blank_pose = blank_sensor.copy()
                cv2.putText(blank_sensor, "NO SENSOR FRAME", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, .7, (180, 180, 180), 1)
                cv2.putText(blank_pose, "POSE GAP / no matching model result", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, .6, (180, 180, 180), 1)
                while True:
                    # Work per pass is bounded; arriving data cannot starve video
                    # ticks or turn this heap into an unlimited memory queue.
                    for _ in range(self.queue_size):
                        try:
                            packet = session.queue.get_nowait()
                        except queue.Empty:
                            break
                        first_unwritten = session.start_mono + next_tick / self.fps
                        if packet["mono"] < first_unwritten - 1. / self.fps:
                            if packet["kind"] == "invalidate":
                                packet["original_monotonic"] = packet["mono"]
                                packet["mono"] = first_unwritten
                                packet["late_control_applied"] = True
                            else:
                                with self._lock:
                                    session.late_packets += 1
                                    session.dropped[packet["kind"]] += 1
                                _line(handles["events"], {"event": "late_packet_dropped", "kind": packet["kind"],
                                                          "source_monotonic": packet["mono"], "first_unwritten_monotonic": first_unwritten})
                                continue
                        if len(pending) >= self.queue_size * 4:
                            if packet["kind"] != "invalidate":
                                with self._lock:
                                    session.dropped[packet["kind"]] += 1
                                continue
                            data_index = next((i for i, item in enumerate(pending) if item[2]["kind"] != "invalidate"), None)
                            if data_index is not None:
                                removed = pending.pop(data_index)[2]
                                heapq.heapify(pending)
                                with self._lock:
                                    session.dropped[removed["kind"]] += 1
                            else:
                                # A burst of control requests still has a strict
                                # memory bound and retains the earliest blanking
                                # barrier, so an old model is never held over it.
                                packet["coalesced_through_monotonic"] = max(
                                    packet.get("coalesced_through_monotonic", packet["mono"]),
                                    max(item[2].get("coalesced_through_monotonic", item[0]) for item in pending))
                                packet["mono"] = min(packet["mono"], pending[0][0])
                                packet["reason"] = "control_events_coalesced"
                                with self._lock:
                                    session.controls_dropped += len(pending)
                                pending.clear()
                        serial += 1
                        heapq.heappush(pending, (packet["mono"], serial, packet))
                    now = self._clock()
                    stopping = session.stop_requested
                    if stopping and session.queue.empty():
                        duration = max(0., session.end_mono - session.start_mono)
                        target = max(0, math.ceil(duration * self.fps - 1e-9)) - 1
                    else:
                        target = math.floor((now - self.write_delay_s - session.start_mono) * self.fps + 1e-9)
                    # Revisit input queues regularly even when catching up. A
                    # whole backlog rendered in one pass starved fresh images
                    # for tens of seconds while the IMU movie kept advancing.
                    batch_end = min(target, next_tick + 3)
                    while next_tick <= batch_end:
                        tick_started = time.perf_counter()
                        tick_mono = session.start_mono + next_tick / self.fps
                        while pending and pending[0][0] <= tick_mono + 1e-9:
                            _, _, packet = heapq.heappop(pending)
                            kind = packet["kind"]
                            if kind == "invalidate":
                                latest["pose"] = None
                                # A coalesced barrier represents every change
                                # through its final timestamp. Intermediate
                                # model results must not revive an old overlay.
                                # Conservatively leave the coalesced interval
                                # blank until a result captured after its end.
                                pose_valid_after = max(pose_valid_after, packet["mono"],
                                                       packet.get("coalesced_through_monotonic", packet["mono"]))
                                pose_gap = packet["reason"]
                                if packet["settings"]:
                                    settings = packet["settings"]
                                _line(handles["events"], {"event": "pose_invalidated", "monotonic": packet["mono"],
                                                          "reason": pose_gap, "settings": settings,
                                                          "coalesced_through_monotonic": packet.get("coalesced_through_monotonic"),
                                                          "pose_source_cutoff_monotonic": pose_valid_after,
                                                          "late_control_applied": packet.get("late_control_applied", False)})
                                continue
                            if kind == "pose" and packet["mono"] + 1e-9 < pose_valid_after:
                                with self._lock:
                                    session.dropped["pose"] += 1
                                _line(handles["events"], {"event": "invalidated_pose_dropped", "source": packet["source"],
                                                          "pose_source_cutoff_monotonic": pose_valid_after,
                                                          "reason": "control_events_coalesced"})
                                continue
                            try:
                                packet["decoded"] = self._decode(packet, session)
                            except ValueError:
                                with self._lock:
                                    session.rejected_invalid += 1
                                continue
                            latest[kind] = packet
                            if kind == "sensor":
                                settings = packet["settings"]
                                _line(handles["sensor"], {"source": packet["source"], "settings": settings})
                            else:
                                packet["pose_event_id"] = packet["sequence"]
                                _line(handles["poses"], {"pose_event_id": packet["sequence"], "source": packet["source"],
                                                         "people": packet["people"], "pose": packet["pose"], "settings": packet["settings"]})
                        sensor, pose = latest["sensor"], latest["pose"]
                        if pose is not None and tick_mono - pose["mono"] > self.pose_stale_after_s:
                            pose = None
                            pose_gap = "pose_stale"
                        for name in STREAMS:
                            chosen = sensor if name in SENSOR_STREAMS else pose
                            image = chosen["decoded"][name] if chosen is not None else (blank_sensor if name in SENSOR_STREAMS else blank_pose)
                            writers[name].write(image)
                        paired = pose is not None and all(name in pose["decoded"] for name in SENSOR_STREAMS)
                        combined_packet = pose if paired else sensor
                        combined_source = combined_packet["source"] if combined_packet else None
                        combined_model = (pose["settings"] if paired else settings).get("pose_model")
                        combined_gap = None if paired else ("matching_sensor_images_unavailable" if pose is not None else pose_gap)
                        views = {name: combined_packet["decoded"][name] if combined_packet else blank_sensor
                                 for name in SENSOR_STREAMS}
                        views["pose"] = pose["decoded"]["pose"] if paired else blank_pose
                        grid = _combined_frame(views, session.width, session.height,
                                               next_tick / self.fps, combined_source, combined_model)
                        writers["combined"].write(grid)
                        source_sensor = sensor["source"] if sensor is not None else None
                        source_pose = pose["source"] if pose is not None else None
                        ids = {"sensor": source_sensor.get("frame_id") if source_sensor else None,
                               "pose": source_pose.get("frame_id") if source_pose else None,
                               "combined": combined_source.get("frame_id") if combined_source else None}
                        _line(handles["frames"], {"frame_index": next_tick, "t_s": next_tick / self.fps,
                                                  "sensor_source": source_sensor, "pose_source": source_pose,
                                                  "sensor_held": ids["sensor"] is not None and ids["sensor"] == previous_ids["sensor"],
                                                  "pose_held": ids["pose"] is not None and ids["pose"] == previous_ids["pose"],
                                                  "pose_gap": None if pose is not None else pose_gap,
                                                  "pose_event_id": pose.get("pose_event_id") if pose else None,
                                                  "combined_source": combined_source,
                                                  "combined_pose_gap": combined_gap,
                                                  "combined_pose_model": combined_model,
                                                  "combined_held": ids["combined"] is not None and ids["combined"] == previous_ids["combined"],
                                                  "selected_pose_model": settings.get("pose_model")})
                        previous_ids = ids
                        next_tick += 1
                        with self._lock:
                            session.written = next_tick
                            perf = session.performance
                            frame_ms = (time.perf_counter()-tick_started)*1000.
                            perf["processed_frames"] += 1
                            perf["total_frame_ms"] += frame_ms
                            perf["max_frame_ms"] = max(perf["max_frame_ms"], frame_ms)
                            perf["max_backlog_s"] = max(perf["max_backlog_s"], self._clock()-self.write_delay_s-tick_mono)
                    if stopping and session.queue.empty() and next_tick > target:
                        break
                    if now - last_check >= 1.:
                        for handle in handles.values():
                            handle.flush()
                        last_check = now
                        if shutil.disk_usage(session.directory).free < self.min_free_bytes:
                            with self._lock:
                                if not session.stop_requested:
                                    session.accepting = False
                                    session.stop_requested = True
                                    session.end_mono = self._clock()
                                    session.reason = "disk_space_low"
                                    session.state = "stopping"
                    if next_tick <= target:
                        continue
                    session.wake.wait(.02)
                    session.wake.clear()
        except Exception as exc:
            error = str(exc)
            with self._lock:
                session.accepting = False
                session.stop_requested = True
                if session.end_mono is None:
                    session.end_mono = self._clock()
                session.reason = session.reason or "encoder_error"
                session.state = "stopping"
        finally:
            with self._lock:
                session.accepting = False
                if session.end_mono is None:
                    session.end_mono = self._clock()
            if session.dataset is not None:
                session.dataset.stop(session.end_mono, wait=True)
            # Only this worker owns/release()s writers, including every error
            # path. stop() and camera shutdown never touch encoder objects.
            for writer in writers.values():
                try:
                    writer.release()
                except Exception as exc:
                    error = error or str(exc)
            if "events" in handles:
                try:
                    _line(handles["events"], {"event": "recording_stopped", "monotonic": session.end_mono,
                                              "reason": session.reason, "frames_written": session.written,
                                              "dropped_frames": session.dropped, "error": error})
                except OSError as exc:
                    error = error or str(exc)
            for handle in handles.values():
                try:
                    handle.flush()
                    handle.close()
                except OSError as exc:
                    error = error or str(exc)
            try:
                files = {name: f"{name}.jsonl" for name in ("events", "sensor", "poses", "frames")
                         if (session.directory / f"{name}.jsonl").exists()}
                if not error and not cancelled and session.written:
                    for name in OUTPUT_STREAMS:
                        temporary = session.directory / f"{name}.partial.mp4"
                        final = session.directory / f"{name}.mp4"
                        temporary.replace(final)
                        files[name] = final.name
                if session.dataset is not None:
                    dataset_status = session.dataset.status()
                    if dataset_status["state"] == "completed":
                        files["dataset_csv"] = "dataset.csv"
                    elif dataset_status["state"] not in ("cancelled",):
                        error = error or dataset_status.get("error") or "Dataset CSV finalization did not complete"
                with self._lock:
                    session.accepting = False
                    session.error = error
                    if session.end_mono is None:
                        session.end_mono = self._clock()
                    terminal = "error" if error else "cancelled" if cancelled else "completed"
                    final_files = {key: f"/recordings/{session.id}/{filename}"
                                   for key, filename in (files | {"manifest": "manifest.json"}).items()}
                    manifest = self._status_locked(session) | {
                        "state": terminal, "files": final_files,
                        "schema_version": 4, "camera": session.camera, "initial_settings": session.settings,
                        "dataset_schema_version": "1.0",
                        "dataset_note": "CSV event rows preserve native asynchronous IMU, camera metadata/depth statistics and Pose joints. Image pixels and raw depth sequences are separate from CSV. Join by source generation and clocks; frames.jsonl maps CFR video slots.",
                        "countdown_seconds": self.countdown_seconds, "write_delay_s": self.write_delay_s,
                        "timing": "Common CFR monotonic timeline; missing slots repeat complete source images; leading/model-change/stale Pose gaps are black",
                        "pose_sync": "Pose and PoseRGBD are drawn on their own original RGB/RGB-D source frame; JSONL maps every encoded tick to source frame IDs",
                        "depth_note": "Depth video is a color visualization; raw Z16 depth sequence is not recorded",
                        "controls_dropped": session.controls_dropped,
                        "coordinate_note": "Depth-assisted joints are approximate camera optical coordinates in metres, not ground truth motion capture",
                    }
                _write_json(session.directory / "manifest.json", manifest)
                with self._lock:
                    session.files = final_files
                    session.state = terminal
            except Exception as exc:
                with self._lock:
                    session.state = "error"
                    session.error = error or str(exc)
