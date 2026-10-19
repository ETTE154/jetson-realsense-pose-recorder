"""One bounded, asynchronous sensor/pose CSV for dataset preprocessing.

Each row is an event, not a synthesized synchronized measurement. IMU samples
retain their own hardware clock, clock domain, generation and host receipt
times. Camera rows refer to the separate video mapping; a 30 Hz camera row is
not guaranteed to occur in a 15 Hz video. Empty cells mean unavailable values.
Camera XYZ values are local surface-depth estimates, not anatomical ground
truth. ``instance_id`` is local to a pose frame unless ``track_id`` is supplied.

Camera submission retains only the immutable depth array and scalar metadata,
never RGB. No file operation or depth calculation runs on a submitter thread.
The queue is bounded by event count, camera count AND retained depth bytes.
"""
from __future__ import annotations

import copy
import csv
import math
from pathlib import Path
import queue
import threading
import time

import numpy as np


FIELDS = (
    "schema_version", "recording_id", "row_sequence", "event_index", "record_type",
    "source_frame_id", "source_frame_number", "source_generation", "source_clock",
    "source_timestamp_ms", "source_timestamp_domain", "source_received_unix_s",
    "source_received_monotonic_s", "source_elapsed_s", "event_received_monotonic_s",
    "color_frame_number", "depth_frame_number", "depth_timestamp_ms", "depth_timestamp_domain",
    "device_serial", "image_width_px", "image_height_px", "depth_scale_m_per_unit",
    "depth_total_pixels", "depth_valid_pixels", "depth_valid_fraction",
    "depth_min_m", "depth_max_m", "depth_mean_m", "depth_median_m",
    "camera_fx_px", "camera_fy_px", "camera_ppx_px", "camera_ppy_px",
    "camera_distortion_model", "camera_distortion_coeff_0", "camera_distortion_coeff_1",
    "camera_distortion_coeff_2", "camera_distortion_coeff_3", "camera_distortion_coeff_4",
    "rgb_video_file", "combined_video_file", "video_mapping_file", "video_mapping_source_frame_id",
    "video_mapping_source_received_monotonic_s", "raw_depth_file",
    "imu_sample_id", "imu_type", "imu_units", "imu_profile_fps",
    "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
    "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
    "pose_model_id", "pose_model", "pose_backend", "pose_state", "pose_inference_ms",
    "pose_instance_count", "instance_id", "instance_id_scope", "track_id", "instance_label",
    "instance_score", "instance_score_kind", "handedness", "handedness_score",
    "bbox_left_px", "bbox_top_px", "bbox_right_px", "bbox_bottom_px",
    "joint_index", "joint_name", "joint_u_px", "joint_v_px", "joint_confidence",
    "joint_confidence_kind", "joint_displayed", "joint_depth_valid",
    "joint_x_m", "joint_y_m", "joint_z_m", "joint_depth_source", "joint_coordinate_frame",
    "joint_native_solver_weight", "setting_pose_enabled", "setting_confidence_threshold",
    "setting_depth_display_min_m", "setting_depth_display_max_m", "setting_roi_left_px",
    "setting_roi_top_px", "setting_roi_right_px", "setting_roi_bottom_px",
)
RECORD_TYPES = ("camera", "imu_accel", "imu_gyro", "pose_frame", "pose_joint")


def _number(value):
    """Keep nulls distinct from zero and suppress nonfinite CSV numbers."""
    if value is None or isinstance(value, (bool, str, bytes)):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else ""
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value)
    # csv quoting does not prevent spreadsheet formula execution. Preserve a
    # visible apostrophe in strings beginning with a formula/control prefix.
    stripped = text.lstrip(" \t\r\n\ufeff")
    if text.startswith(("\t", "\r", "\n")) or stripped.startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


class DatasetCSV:
    def __init__(self, path, start_monotonic, *, recording_id=None, queue_size=4096,
                 max_camera_frames=64, max_depth_bytes=64 * 1024**2, clock=None):
        self.path = Path(path)
        self.partial_path = self.path.with_name(self.path.stem + ".partial" + self.path.suffix)
        self.start_monotonic = float(start_monotonic)
        if not math.isfinite(self.start_monotonic) or self.start_monotonic <= 0:
            raise ValueError("start_monotonic must be a positive finite capture-clock deadline")
        if int(queue_size) < 1 or int(max_camera_frames) < 1 or int(max_depth_bytes) < 1:
            raise ValueError("Queue and depth memory limits must be positive")
        self.recording_id = str(recording_id or self.path.parent.name)
        self._clock = clock or time.monotonic
        self._queue = queue.Queue(maxsize=int(queue_size))
        self._max_camera_frames, self._max_depth_bytes = int(max_camera_frames), int(max_depth_bytes)
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._accepting, self._stop_mono = True, None
        self._state, self._error = "countdown", None
        self._accepted = {kind: 0 for kind in RECORD_TYPES}
        self._written = {kind: 0 for kind in RECORD_TYPES}
        self._dropped = {kind: 0 for kind in RECORD_TYPES}
        self._rejected_before = self._rejected_after = self._rejected_invalid = 0
        self._camera_pending = self._depth_bytes = self._sequence = self._row_sequence = 0
        self._last_written_source_mono = 0.
        self._worker = threading.Thread(target=self._run, name="dataset-csv", daemon=True)
        self._worker.start()

    def _source(self, source, kind):
        stamp = _number(_get(source, "received_monotonic"))
        generation = _get(source, "capture_generation", _get(source, "generation", _get(source, "stream_epoch")))
        is_imu = kind.startswith("imu_")
        return {
            "source_frame_id": None if is_imu else _get(source, "frame_id"),
            "source_frame_number": _get(source, "frame_number") if is_imu else _get(source, "color_frame_number"),
            "source_generation": generation,
            "source_clock": kind if is_imu else "camera_color",
            "source_timestamp_ms": _number(_get(source, "timestamp_ms")),
            "source_timestamp_domain": _get(source, "timestamp_domain"),
            "source_received_unix_s": _number(_get(source, "received_time")),
            "source_received_monotonic_s": stamp,
            "source_elapsed_s": None if stamp is None else stamp - self.start_monotonic,
        }

    @staticmethod
    def _settings(settings):
        settings = settings or {}
        result = {
            "setting_pose_enabled": settings.get("pose_enabled"),
            "setting_confidence_threshold": _number(settings.get("confidence")),
            "setting_depth_display_min_m": _number(settings.get("depth_min")),
            "setting_depth_display_max_m": _number(settings.get("depth_max")),
        }
        roi = settings.get("pose_roi")
        if isinstance(roi, (list, tuple)) and len(roi) == 4:
            result.update(zip(("setting_roi_left_px", "setting_roi_top_px", "setting_roi_right_px", "setting_roi_bottom_px"),
                              map(_number, roi)))
        return result

    def _enqueue(self, kind, source, payload, *, depth_bytes=0):
        now = self._clock()
        stamp = source["source_received_monotonic_s"]
        with self._lock:
            if not self._accepting:
                return False
            if stamp is None or stamp <= 0 or stamp > now:
                self._rejected_invalid += 1
                return False
            if stamp < self.start_monotonic or now < self.start_monotonic:
                self._rejected_before += 1
                return False
            if self._stop_mono is not None and stamp > self._stop_mono:
                self._rejected_after += 1
                return False
            if (kind == "camera" and (self._camera_pending >= self._max_camera_frames
                                      or self._depth_bytes + depth_bytes > self._max_depth_bytes)):
                self._dropped[kind] += 1
                return False
            event = (self._sequence + 1, kind, {**source, "event_received_monotonic_s": now}, payload, depth_bytes)
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                self._dropped[kind] += 1
                return False
            self._sequence += 1
            self._accepted[kind] += 1
            if kind == "camera":
                self._camera_pending += 1
                self._depth_bytes += depth_bytes
            self._state = "recording"
            self._wake.set()
            return True

    def submit_camera(self, frame, settings=None, video_refs=None):
        """Queue scalar metadata and a depth reference; RGB is never retained."""
        depth = _get(frame, "depth")
        if not isinstance(depth, np.ndarray) or depth.ndim != 2:
            with self._lock:
                self._rejected_invalid += 1
            return False
        intrinsics = _get(frame, "intrinsics")
        row = self._settings(settings)
        row.update({
            "color_frame_number": _get(frame, "color_frame_number"),
            "depth_frame_number": _get(frame, "depth_frame_number"),
            "depth_timestamp_ms": _number(_get(frame, "depth_timestamp_ms")),
            "depth_timestamp_domain": _get(frame, "depth_timestamp_domain"),
            "device_serial": _get(frame, "serial"),
            "image_width_px": int(depth.shape[1]), "image_height_px": int(depth.shape[0]),
            "depth_scale_m_per_unit": _number(_get(frame, "depth_scale")),
            "camera_fx_px": _number(_get(intrinsics, "fx")),
            "camera_fy_px": _number(_get(intrinsics, "fy")),
            "camera_ppx_px": _number(_get(intrinsics, "ppx")),
            "camera_ppy_px": _number(_get(intrinsics, "ppy")),
            "camera_distortion_model": str(_get(intrinsics, "model", "")),
            "rgb_video_file": "rgb.mp4", "combined_video_file": "combined.mp4",
            "video_mapping_file": "frames.jsonl", "video_mapping_source_frame_id": _get(frame, "frame_id"),
            "video_mapping_source_received_monotonic_s": _number(_get(frame, "received_monotonic")),
        })
        for index, coeff in enumerate(list(_get(intrinsics, "coeffs", []) or [])[:5]):
            row["camera_distortion_coeff_" + str(index)] = _number(coeff)
        if video_refs:
            row.update({key: value for key, value in video_refs.items() if key in FIELDS})
        return self._enqueue("camera", self._source(frame, "camera"), (row, depth), depth_bytes=depth.nbytes)

    def submit_imu(self, sample):
        kind = str(_get(sample, "type", _get(sample, "stream", ""))).lower()
        if kind not in ("accel", "gyro"):
            with self._lock:
                self._rejected_invalid += 1
            return False
        values = [_number(_get(sample, axis)) for axis in ("x", "y", "z")]
        if any(value is None for value in values):
            with self._lock:
                self._rejected_invalid += 1
            return False
        unit = "m/s^2" if kind == "accel" else "rad/s"
        supplied_unit = _get(sample, "units", unit)
        allowed_units = ("m/s^2", "m/s²", "m/s2") if kind == "accel" else ("rad/s",)
        if supplied_unit not in allowed_units:
            # Reject unconverted g / deg/s rather than mislabel physical units.
            with self._lock:
                self._rejected_invalid += 1
            return False
        payload = {"imu_sample_id": _get(sample, "sample_id"), "imu_type": kind,
                   "imu_units": supplied_unit, "imu_profile_fps": _get(sample, "fps"),
                   "device_serial": _get(sample, "serial")}
        names = ("accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2") if kind == "accel" else ("gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s")
        payload.update(zip(names, values))
        return self._enqueue("imu_" + kind, self._source(sample, "imu_" + kind), payload)

    def submit_pose(self, frame, people, pose=None, settings=None):
        # Caller may reuse its dictionaries immediately after this method.
        payload = (copy.deepcopy(people), copy.deepcopy(pose or {}), self._settings(settings))
        return self._enqueue("pose_frame", self._source(frame, "pose_frame"), payload)

    def stop(self, stop_monotonic=None, wait=True):
        end = self._clock() if stop_monotonic is None else float(stop_monotonic)
        if not math.isfinite(end) or end <= 0:
            raise ValueError("stop_monotonic must be positive and finite")
        with self._lock:
            if self._stop_mono is None:
                self._stop_mono = end
                self._accepting = False
                if self._state not in ("error", "completed", "cancelled"):
                    self._state = "stopping"
                self._wake.set()
        if wait and threading.current_thread() is not self._worker:
            self._worker.join(timeout=30.)
        return self.status()

    def status(self):
        with self._lock:
            state = self._state
            if state == "countdown" and self._clock() >= self.start_monotonic:
                state = "recording"
            return {"state": state, "recording_id": self.recording_id, "error": self._error,
                    "path": str(self.path) if state == "completed" else None,
                    "file": self.path.name if state == "completed" else None,
                    "start_monotonic": self.start_monotonic, "stop_monotonic": self._stop_mono,
                    "queue_depth": self._queue.qsize(), "pending_camera_frames": self._camera_pending,
                    "retained_depth_bytes": self._depth_bytes, "accepted": dict(self._accepted),
                    "rows_written": dict(self._written), "rows_total": self._row_sequence,
                    "dropped": dict(self._dropped), "dropped_total": sum(self._dropped.values()),
                    "excluded_before_deadline": self._rejected_before,
                    "excluded_after_stop": self._rejected_after, "rejected_invalid": self._rejected_invalid}

    def _write(self, writer, row, kind, event_index):
        row = {"schema_version": "1.0", "recording_id": self.recording_id,
               "row_sequence": self._row_sequence + 1, "event_index": event_index,
               "record_type": kind, **row}
        writer.writerow({key: _cell(value) for key, value in row.items() if key in FIELDS})
        with self._lock:
            self._row_sequence += 1
            self._written[kind] += 1
            self._last_written_source_mono = max(self._last_written_source_mono, row["source_received_monotonic_s"])

    def _filter_after_stop(self):
        """Handle a stop cutoff captured just before its callback reached us.

        Usually all queued rows are filtered before writing. A producer can
        race a supplied cutoff by a few microseconds and already have reached
        disk. This uncommon final pass makes the published stop gate exact.
        """
        with self._lock:
            cutoff = self._stop_mono
            if cutoff is None or self._last_written_source_mono <= cutoff:
                return
        filtered = self.partial_path.with_name(self.partial_path.stem + ".filter.csv")
        counts = {kind: 0 for kind in RECORD_TYPES}
        removed_events = set()
        total = 0
        try:
            with self.partial_path.open(newline="", encoding="utf-8-sig") as source:
                with filtered.open("x", newline="", encoding="utf-8-sig") as target:
                    reader, writer = csv.DictReader(source), csv.DictWriter(target, fieldnames=FIELDS, lineterminator="\n")
                    writer.writeheader()
                    for row in reader:
                        if float(row["source_received_monotonic_s"]) > cutoff:
                            removed_events.add(row["event_index"])
                            continue
                        total += 1
                        row["row_sequence"] = total
                        counts[row["record_type"]] += 1
                        writer.writerow(row)
            filtered.replace(self.partial_path)
        finally:
            filtered.unlink(missing_ok=True)
        with self._lock:
            self._written, self._row_sequence = counts, total
            self._rejected_after += len(removed_events)

    def _camera_row(self, payload):
        row, depth = payload
        row = dict(row)
        valid = depth[depth > 0]
        row.update(depth_total_pixels=int(depth.size), depth_valid_pixels=int(valid.size),
                   depth_valid_fraction=float(valid.size / depth.size) if depth.size else None)
        scale = row["depth_scale_m_per_unit"]
        if valid.size and scale is not None and scale > 0:
            row.update(depth_min_m=float(np.min(valid)) * scale, depth_max_m=float(np.max(valid)) * scale,
                       depth_mean_m=float(np.mean(valid)) * scale, depth_median_m=float(np.median(valid)) * scale)
        return row

    def _pose_rows(self, source, payload):
        people, pose, settings = payload
        base = {**source, **settings, "pose_model_id": pose.get("model_id"),
                "pose_model": pose.get("model"), "pose_backend": pose.get("backend"),
                "pose_state": pose.get("state"), "pose_inference_ms": _number(pose.get("inference_ms")),
                "pose_instance_count": len(people)}
        yield "pose_frame", base
        for instance_index, instance in enumerate(people):
            score_kind = instance.get("score_kind", "model_score")
            score = None if score_kind == "not_reported_display_mask" else _number(instance.get("score"))
            label = instance.get("label")
            handedness = label if str(label).lower() in ("left", "right") else instance.get("handedness")
            current = {**base, "instance_id": instance.get("instance_id", instance_index),
                       "instance_id_scope": "source_frame", "track_id": instance.get("track_id"),
                       "instance_label": label, "instance_score": score, "instance_score_kind": score_kind,
                       "handedness": handedness,
                       "handedness_score": score if score_kind == "handedness_classification" else None}
            box = instance.get("bbox")
            if box is not None and len(box) == 4:
                current.update(zip(("bbox_left_px", "bbox_top_px", "bbox_right_px", "bbox_bottom_px"), map(_number, box)))
            points = instance.get("depth_keypoints") or instance.get("keypoints") or []
            names = instance.get("keypoint_names") or []
            weights = instance.get("native_solver_weights") or []
            for joint_index, point in enumerate(points):
                if isinstance(point, dict):
                    joint = point
                else:
                    joint = {"u": point[0], "v": point[1], "confidence": point[2] if len(point) > 2 else None}
                confidence_kind = joint.get("confidence_kind", instance.get("keypoint_confidence_kind", "model_score"))
                confidence = None if confidence_kind == "not_reported_display_mask" else _number(joint.get("confidence"))
                depth_valid = joint.get("valid_depth")
                xyz = [_number(joint.get(key)) if depth_valid else None for key in ("x_m", "y_m", "z_m")]
                # Never export a valid flag paired with missing/nonfinite XYZ.
                if depth_valid and any(value is None for value in xyz):
                    depth_valid, xyz = False, [None, None, None]
                yield "pose_joint", {**current, "joint_index": joint_index,
                    "joint_name": joint.get("name", names[joint_index] if joint_index < len(names) else f"joint_{joint_index}"),
                    "joint_u_px": _number(joint.get("u")), "joint_v_px": _number(joint.get("v")),
                    "joint_confidence": confidence, "joint_confidence_kind": confidence_kind,
                    "joint_displayed": joint.get("displayed"), "joint_depth_valid": depth_valid,
                    "joint_x_m": xyz[0], "joint_y_m": xyz[1], "joint_z_m": xyz[2],
                    "joint_depth_source": joint.get("depth_source", "D435i local surface depth; not ground-truth joint depth"),
                    "joint_coordinate_frame": "color_camera_optical_x_right_y_down_z_forward",
                    "joint_native_solver_weight": _number(weights[joint_index]) if joint_index < len(weights) else None}

    def _run(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() or self.partial_path.exists():
                raise FileExistsError("Dataset CSV already exists; refusing to overwrite")
            with self.partial_path.open("x", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
                writer.writeheader()
                last_flush = time.monotonic()
                while True:
                    try:
                        index, kind, source, payload, depth_bytes = self._queue.get_nowait()
                    except queue.Empty:
                        with self._lock:
                            if not self._accepting:
                                break
                            self._wake.clear()
                        self._wake.wait(.1)
                        continue
                    try:
                        with self._lock:
                            excluded = self._stop_mono is not None and source["source_received_monotonic_s"] > self._stop_mono
                            if excluded:
                                self._rejected_after += 1
                        if excluded:
                            continue
                        if kind == "camera":
                            self._write(writer, {**source, **self._camera_row(payload)}, kind, index)
                        elif kind == "pose_frame":
                            for row_kind, row in self._pose_rows(source, payload):
                                self._write(writer, row, row_kind, index)
                        else:
                            self._write(writer, {**source, **payload}, kind, index)
                        if time.monotonic() - last_flush >= 1.:
                            handle.flush()
                            last_flush = time.monotonic()
                    finally:
                        with self._lock:
                            if kind == "camera":
                                self._camera_pending -= 1
                                self._depth_bytes -= depth_bytes
                        self._queue.task_done()
                handle.flush()
            with self._lock:
                cancelled = self._stop_mono is not None and self._stop_mono < self.start_monotonic
            if cancelled:
                self.partial_path.unlink(missing_ok=True)
            else:
                self._filter_after_stop()
                self.partial_path.replace(self.path)
            with self._lock:
                self._state = "cancelled" if cancelled else "completed"
        except Exception as error:
            with self._lock:
                self._accepting = False
                self._state, self._error = "error", str(error)
            # Release queued array references on writer failure; partial stays
            # private for diagnosis and is never published as a final CSV.
            while True:
                try:
                    _, kind, _, _, size = self._queue.get_nowait()
                except queue.Empty:
                    break
                with self._lock:
                    self._dropped[kind] += 1
                    if kind == "camera":
                        self._camera_pending -= 1
                        self._depth_bytes -= size
                self._queue.task_done()
