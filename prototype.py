"""Latest-frame RGB-D rendering and depth-assisted COCO pose visualization."""
from __future__ import annotations

import copy
from collections import deque
from datetime import datetime
import json
import logging
from pathlib import Path
import threading
import time
import uuid

import cv2
import numpy as np

from camera import Camera

LOG = logging.getLogger(__name__)
JOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
          "right_knee", "left_ankle", "right_ankle")
EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
         (0, 1), (0, 2), (1, 3), (2, 4))


def colorize_depth(depth, scale, lower, upper):
    """Color is display-only; zero/invalid pixels remain black."""
    metres = depth.astype(np.float32) * scale
    valid = depth > 0
    normalized = np.clip((metres - lower) / (upper - lower), 0, 1)
    palette = cv2.applyColorMap(np.uint8((1 - normalized) * 255), cv2.COLORMAP_TURBO)
    palette[~valid] = 0
    return palette


def enrich_people(people, frame, deproject=None):
    """Median 5x5 valid depth around a joint, deprojected in camera coordinates."""
    if deproject is None:
        import pyrealsense2 as rs
        deproject = rs.rs2_deproject_pixel_to_point
    enriched = []
    height, width = frame.depth.shape
    for person in people:
        result = {"bbox": person["bbox"], "score": person["score"], "keypoints": []}
        for name, point in zip(JOINTS, person["keypoints"]):
            u, v, confidence = map(float, point)
            joint = {"name": name, "u": u, "v": v, "confidence": confidence,
                     "x_m": None, "y_m": None, "z_m": None, "valid_depth": False}
            if confidence >= .3 and 0 <= u < width and 0 <= v < height:
                x, y = int(round(u)), int(round(v))
                x, y = min(width - 1, x), min(height - 1, y)
                patch = frame.depth[max(0, y-2):min(height, y+3),
                                    max(0, x-2):min(width, x+3)]
                values = patch[patch > 0]
                if len(values) >= 3:
                    depth_m = float(np.median(values)) * frame.depth_scale
                    xyz = deproject(frame.intrinsics, [u, v], depth_m)
                    if np.isfinite(xyz).all() and 0.1 <= depth_m <= 10:
                        joint.update(x_m=float(xyz[0]), y_m=float(xyz[1]),
                                     z_m=float(xyz[2]), valid_depth=True)
            result["keypoints"].append(joint)
        enriched.append(result)
    return enriched


def draw_people(image, people):
    canvas = image.copy()
    for person in people:
        x1, y1, x2, y2 = map(lambda v: int(round(v)), person["bbox"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (52, 219, 153), 1)
        points = person["keypoints"]
        for a, b in EDGES:
            if points[a]["confidence"] >= .3 and points[b]["confidence"] >= .3:
                pa = (int(points[a]["u"]), int(points[a]["v"]))
                pb = (int(points[b]["u"]), int(points[b]["v"]))
                cv2.line(canvas, pa, pb, (66, 220, 245), 2, cv2.LINE_AA)
        distances = []
        for joint in points:
            if joint["confidence"] < .3:
                continue
            p = (int(joint["u"]), int(joint["v"]))
            color = (80, 230, 110) if joint["valid_depth"] else (20, 155, 255)
            cv2.circle(canvas, p, 4, color, -1, cv2.LINE_AA)
            if joint["valid_depth"]:
                distances.append(joint["z_m"])
                if joint["name"] in ("left_wrist", "right_wrist", "nose"):
                    cv2.putText(canvas, f'{joint["z_m"]:.2f}m', (p[0]+6, p[1]-6),
                                cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1, cv2.LINE_AA)
        label = f'person {person["score"]:.2f}'
        if distances:
            label += f' | {float(np.median(distances)):.2f}m'
        cv2.putText(canvas, label, (max(0, x1), max(18, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (52, 219, 153), 1, cv2.LINE_AA)
    return canvas


def intrinsics_json(intrinsics):
    return {k: getattr(intrinsics, k) for k in ("width", "height", "ppx", "ppy", "fx", "fy")} | {
        "model": str(intrinsics.model), "coeffs": list(intrinsics.coeffs)}


class Application:
    def __init__(self, model_path, output_dir, input_size=640, backend="auto", preview_fps=15):
        self.camera = Camera()
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_size = input_size
        self.backend = backend
        self.preview_fps = preview_fps
        self.settings = {"confidence": .35, "depth_min": .3, "depth_max": 4.,
                         "pose_enabled": True, "pose_model": "yolo", "pose_roi": None}
        self.condition = threading.Condition(threading.RLock())
        self.actions = threading.Lock()
        self.pose = self._empty_pose("yolo", "loading")
        self._revision = 0
        self._estimator = None
        self._sequence = 0
        self._stream_sequences = {}
        self._images = {}
        self._processed = None
        self._snapshot_images = {}
        self._stop = threading.Event()
        from comparison.manager import ComparisonManager
        self.comparison = ComparisonManager(self)
        from recording import Recorder
        self.recorder = Recorder(self.output_dir.parent / "recordings", fps=float(preview_fps), countdown_seconds=5.0)
        self.camera.set_imu_listener(self.recorder.submit_imu)
        self.camera.set_frame_listener(self._receive_camera_data)
        self._worker = threading.Thread(target=self._process, name="rgbd-render", daemon=True)
        self._pose_worker = threading.Thread(target=self._infer, name="live-pose", daemon=True)
        self._worker.start()
        self._pose_worker.start()

    @staticmethod
    def _empty_pose(identifier, state):
        names = {"yolo": "YOLOv8n-pose", "mediapipe": "MediaPipe Hand Landmarker",
                 "rtmpose": "RTMPose-m Body7", "egoforce": "EgoForce HALO 2D"}
        labels = {"yolo": "감지된 사람", "mediapipe": "감지된 손",
                  "rtmpose": "ROI 추정", "egoforce": "처리한 손 영역"}
        return {"state": state, "model_id": identifier, "model": names[identifier],
                "backend": None, "enabled": state != "disabled", "inference_ms": None,
                "fps": 0., "people": 0, "instances": 0, "visible_keypoints": 0,
                "valid_depth_keypoints": 0, "frame_id": None, "received_time": None,
                "entity_label": labels[identifier], "metadata": {}, "error": None}

    def _clear_pose_locked(self):
        self.pose = self._empty_pose(self.settings["pose_model"],
                                     "loading" if self.settings["pose_enabled"] else "disabled")
        for name in ("pose", "pose_rgbd"):
            self._images.pop(name, None)
        self._processed = None
        self._snapshot_images = {}

    def invalidate_pose(self):
        """Discard an in-flight frame before a camera stop/restart."""
        with self.condition:
            self._revision += 1
            self._clear_pose_locked()
            estimator = self._estimator
            self._estimator = None
            self.condition.notify_all()
        self.recorder.invalidate_pose(reason="camera_reset")
        if estimator is not None:
            estimator.close()

    def status(self):
        camera = self.camera.snapshot()
        with self.condition:
            pose, settings = copy.deepcopy(self.pose), copy.deepcopy(self.settings)
        pose["enabled"] = settings["pose_enabled"]
        if not settings["pose_enabled"] and pose["state"] == "ready":
            pose["state"] = "disabled"
        pose["frame_age_ms"] = (max(0., (time.time()-pose["received_time"])*1000)
                                if pose.get("received_time") is not None else None)
        return {"camera": camera, "pose": pose, "settings": settings, "recording": self.recorder.status(),
                "imu": self._normalize_imu(camera.get("imu", {})),
                "system": {"name": "D435i Prototype Lab", "version": "1.4.2",
                           "preview_fps_limit": self.preview_fps, "depth_unit": "metres"}}

    def _receive_camera_data(self, frame):
        with self.condition:
            settings = self.settings.copy()
        self.recorder.submit_camera_data(frame, settings)

    @staticmethod
    def _normalize_imu(raw):
        result = {**raw, "capture_epoch": raw.get("capture_generation"),
                  "reason": raw.get("error")}
        for kind in ("accel", "gyro"):
            sample = raw.get("latest", {}).get(kind)
            result[kind] = ({**sample, "unit": sample.get("units"),
                             "sequence": sample.get("sample_id"),
                             "hz": raw.get("rates_hz", {}).get(kind, 0.),
                             "count": raw.get("counts", {}).get(kind, 0)} if sample else None)
        return result

    def imu_history(self, after=0, limit=512):
        history = self.camera.motion_history(after, limit)
        return {**history, "samples": [{**sample, "sequence": sample["sample_id"],
                   "stream": sample["type"], "capture_epoch": sample["capture_generation"]}
                   for sample in history["samples"]],
                "latest_sequence": history["next_id"],
                "truncated": history["missed_samples"] > 0,
                "imu": self._normalize_imu(self.camera.motion_snapshot())}

    def configure(self, values):
        from comparison.manager import validate_roi
        estimator = None
        with self.condition:
            settings = copy.deepcopy(self.settings)
            if set(values) - set(settings):
                raise ValueError("지원하지 않는 설정입니다.")
            for key in ("confidence", "depth_min", "depth_max"):
                if key in values:
                    if isinstance(values[key], bool) or not isinstance(values[key], (int, float)):
                        raise ValueError("설정은 숫자여야 합니다.")
                    value = float(values[key])
                    if not np.isfinite(value):
                        raise ValueError("유효한 숫자를 입력하세요.")
                    settings[key] = value
            if not .05 <= settings["confidence"] <= .95:
                raise ValueError("신뢰도는 0.05~0.95 범위입니다.")
            if not .1 <= settings["depth_min"] < settings["depth_max"] <= 10:
                raise ValueError("깊이 범위는 0.1~10m이며 최댓값이 더 커야 합니다.")
            if "pose_enabled" in values:
                if not isinstance(values["pose_enabled"], bool):
                    raise ValueError("pose_enabled는 true 또는 false여야 합니다.")
                settings["pose_enabled"] = values["pose_enabled"]
            if "pose_model" in values:
                if values["pose_model"] not in ("yolo", "mediapipe", "rtmpose", "egoforce"):
                    raise ValueError("지원하지 않는 Pose 모델입니다.")
                settings["pose_model"] = values["pose_model"]
            if "pose_roi" in values:
                settings["pose_roi"] = validate_roi(values["pose_roi"], self.camera.width, self.camera.height)
            changed = settings != self.settings
            restart = (settings["pose_model"] != self.settings["pose_model"] or
                       settings["pose_enabled"] != self.settings["pose_enabled"])
            self.settings = settings
            if changed:
                self._revision += 1
                self._clear_pose_locked()
            if restart:
                estimator = self._estimator
                self._estimator = None
            self.condition.notify_all()
        # This can interrupt model loading/inference without blocking RGB-D.
        if changed:
            self.recorder.invalidate_pose(reason="settings_changed", settings=settings)
        if estimator is not None:
            estimator.close()
        return copy.deepcopy(settings)

    def start_recording(self):
        if self.recorder.status()["state"] in ("countdown", "recording", "stopping"):
            raise ValueError("녹화 준비 또는 저장이 진행 중입니다. 종료 후 다시 시작하세요.")
        camera = self.camera.snapshot()
        if camera["state"] != "streaming":
            raise ValueError("카메라 영상이 수신된 뒤 녹화를 시작할 수 있습니다.")
        frame = self.camera.wait_frame(-1, timeout=0.)
        if frame is None:
            raise ValueError("녹화할 카메라 프레임을 아직 받지 못했습니다.")
        camera = {**camera, "color_intrinsics": intrinsics_json(frame.intrinsics),
                  "depth_alignment": "color", "depth_scale_m_per_unit": frame.depth_scale,
                  "depth_video": "display colormap; original Z16 is not recorded in the video"}
        with self.condition:
            settings = copy.deepcopy(self.settings)
        return self.recorder.start(camera=camera, settings=settings)

    @staticmethod
    def _encode(images):
        encoded = {}
        for name, image in images.items():
            ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            encoded[name] = jpeg.tobytes()
        return encoded

    @staticmethod
    def _backgrounds(frame, settings):
        depth = colorize_depth(frame.depth, frame.depth_scale, settings["depth_min"], settings["depth_max"])
        rgbd = cv2.addWeighted(frame.rgb, .60, depth, .40, 0)
        rgbd[frame.depth == 0] = frame.rgb[frame.depth == 0]
        return depth, rgbd

    def _publish_locked(self, images):
        self._sequence += 1
        self._images.update(images)
        for name in images:
            self._stream_sequences[name] = self._sequence
        self.condition.notify_all()

    def _process(self):
        """Render sensor streams independently of model load and inference."""
        had_camera = False
        try:
            after_id = -1
            next_preview = 0.
            while not self._stop.is_set():
                frame = self.camera.wait_frame(after_id, timeout=.5)
                if frame is None:
                    if self.camera.snapshot()["state"] != "streaming":
                        self.recorder.stop(reason="camera_disconnected")
                        if had_camera:
                            self.invalidate_pose()
                        had_camera = False
                        with self.condition:
                            self._images = {}
                            self._processed = None
                            self._snapshot_images = {}
                            self.pose.update(people=0, fps=0.)
                            self.condition.notify_all()
                    self._stop.wait(.1)
                    continue
                after_id = frame.frame_id
                had_camera = True
                now = time.monotonic()
                if now < next_preview:
                    self._stop.wait(min(next_preview - now, .1))
                    continue
                started = time.monotonic()
                with self.condition:
                    settings = copy.deepcopy(self.settings)
                    revision = self._revision
                depth_image, rgbd = self._backgrounds(frame, settings)
                images = self._encode({"rgb": frame.rgb, "depth": depth_image, "rgbd": rgbd})
                if not settings["pose_enabled"]:
                    images.update(pose=images["rgb"], pose_rgbd=images["rgbd"])
                camera_state = self.camera.snapshot()
                if camera_state["state"] != "streaming":
                    continue
                with self.condition:
                    if revision != self._revision:
                        continue
                    self._publish_locked(images)
                    if self.pose["state"] != "ready" or not settings["pose_enabled"]:
                        # No model result is claimed while it is loading/disabled.
                        fallback = {**images, "pose": images["rgb"], "pose_rgbd": images["rgbd"]}
                        self._processed = (frame, [], settings, copy.deepcopy(self.pose), camera_state["device"])
                        self._snapshot_images = fallback
                # File encoding stays outside the camera/application lock.
                self.recorder.submit_sensor(frame, images, settings)
                if not settings["pose_enabled"]:
                    disabled_pose = self._empty_pose(settings["pose_model"], "disabled")
                    disabled_pose.update(frame_id=frame.frame_id, received_time=frame.received_time)
                    self.recorder.submit_pose(frame, images,
                                              [], disabled_pose, settings)
                next_preview = started + 1 / self.preview_fps
        except Exception as error:
            LOG.exception("Rendering worker failed")
            self.recorder.stop(reason="render_error")
            with self.condition:
                self.pose.update(state="error", error=str(error))
                self._images = {}
                self._processed = None
                self.condition.notify_all()

    def _infer(self):
        from live_runtime import LiveEstimator
        from comparison.manager import enrich_instances, draw_instances
        estimator, active_id, failed_revision = None, None, None
        after_id, next_due = -1, 0.
        completed = deque(maxlen=60)
        try:
            while not self._stop.is_set():
                with self.condition:
                    settings, revision = copy.deepcopy(self.settings), self._revision
                identifier = settings["pose_model"]
                if not settings["pose_enabled"] or self.camera.snapshot()["state"] != "streaming":
                    if estimator is not None:
                        estimator.close()
                        with self.condition:
                            if self._estimator is estimator:
                                self._estimator = None
                        estimator, active_id = None, None
                    self._stop.wait(.1)
                    continue
                if failed_revision == revision:
                    self._stop.wait(.1)
                    continue
                if estimator is None or active_id != identifier:
                    if estimator is not None:
                        estimator.close()
                    try:
                        estimator = LiveEstimator(identifier, model_path=self.model_path if identifier == "yolo" else None,
                                                  input_size=self.input_size, backend=self.backend)
                    except Exception as error:
                        LOG.exception("Live model %s could not start", identifier)
                        with self.condition:
                            if revision == self._revision:
                                self.pose.update(state="error", error=str(error), fps=0.)
                                failed_revision = revision
                        estimator, active_id = None, None
                        continue
                    active_id = identifier
                    after_id, next_due = -1, 0.
                    completed.clear()
                    with self.condition:
                        if revision != self._revision or self._stop.is_set():
                            estimator.close()
                            estimator = None
                            continue
                        self._estimator = estimator
                    try:
                        # Load before acquiring the source image, so the first
                        # Pose frame is not as old as the model's loading time.
                        if hasattr(estimator, "prepare"):
                            estimator.prepare()
                    except Exception as error:
                        with self.condition:
                            if revision == self._revision and not self._stop.is_set():
                                self.pose.update(state="error", error=str(error), fps=0.)
                                failed_revision = revision
                            if self._estimator is estimator:
                                self._estimator = None
                        estimator.close()
                        estimator, active_id = None, None
                        continue
                frame = self.camera.wait_frame(after_id, timeout=.5)
                if frame is None:
                    continue
                if time.monotonic() < next_due:
                    self._stop.wait(min(.1, next_due-time.monotonic()))
                    continue
                after_id = frame.frame_id
                started = time.monotonic()
                try:
                    result = estimator.infer(frame.rgb, intrinsics=intrinsics_json(frame.intrinsics),
                                             roi=settings["pose_roi"], confidence=settings["confidence"])
                    inference_ms = result.get("inference_ms", (time.monotonic()-started)*1000)
                    threshold = settings["confidence"] if identifier == "rtmpose" else .3
                    instances, shown, measured = enrich_instances(result.get("instances", []), frame.depth,
                                                                 frame.depth_scale, intrinsics_json(frame.intrinsics), threshold)
                    depth_image, rgbd = self._backgrounds(frame, settings)
                    pose_images = self._encode({"pose": draw_instances(frame.rgb, instances, settings["pose_roi"]),
                                                "pose_rgbd": draw_instances(rgbd, instances, settings["pose_roi"])})
                    camera = self.camera.snapshot()
                    with self.condition:
                        if revision != self._revision or camera["state"] != "streaming" or self._stop.is_set():
                            continue
                        now = time.monotonic()
                        completed.append(now)
                        while len(completed) > 2 and now-completed[0] > 2.:
                            completed.popleft()
                        fps = ((len(completed)-1)/(completed[-1]-completed[0]) if len(completed) > 1
                               else 1/max(1/self.preview_fps, now-started))
                        metadata = {k: v for k, v in result.items() if k not in ("instances", "backend", "model", "metadata")}
                        metadata.update(result.get("metadata") or {})
                        self.pose.update(state="ready", model_id=identifier, model=result.get("model", active_id),
                                         backend=result.get("backend"), error=None, inference_ms=inference_ms,
                                         fps=round(fps, 1), people=len(instances), instances=len(instances),
                                         visible_keypoints=shown, valid_depth_keypoints=measured,
                                         load_ms=getattr(estimator, "load_ms", None),
                                         frame_id=frame.frame_id, received_time=frame.received_time,
                                         metadata=metadata)
                        # Pose is drawn on its own source frame, never on newer RGB.
                        people = [{**instance, "raw_keypoints": instance["keypoints"],
                                   "keypoints": instance["depth_keypoints"]} for instance in instances]
                        self._processed = (frame, people, settings, copy.deepcopy(self.pose), camera["device"])
                        self._snapshot_images = pose_images
                        self._publish_locked(pose_images)
                        pose_state = copy.deepcopy(self.pose)
                    self.recorder.submit_pose(frame, {**pose_images, "rgb": frame.rgb, "depth": depth_image, "rgbd": rgbd},
                                              people, pose_state, settings)
                    next_due = started + 1/self.preview_fps
                except Exception as error:
                    with self.condition:
                        current = revision == self._revision and not self._stop.is_set()
                        if current:
                            self.pose.update(state="error", error=str(error), fps=0., frame_id=None)
                            for name in ("pose", "pose_rgbd"):
                                self._images.pop(name, None)
                    if current:
                        self.recorder.invalidate_pose(reason="model_error", settings=settings)
                        LOG.exception("Live model %s failed", identifier)
                        failed_revision = revision
                    estimator.close()
                    with self.condition:
                        if self._estimator is estimator:
                            self._estimator = None
                    estimator, active_id = None, None
        except Exception as error:
            LOG.exception("Live inference worker failed")
            with self.condition:
                self.pose.update(state="error", error=str(error), fps=0.)
        finally:
            if estimator is not None:
                estimator.close()

    def wait_image(self, name, after, timeout=2):
        with self.condition:
            self.condition.wait_for(lambda: self._stop.is_set() or
                                    (self._stream_sequences.get(name, -1) > after and name in self._images), timeout)
            return self._stream_sequences.get(name, -1), self._images.get(name)

    def people(self):
        with self.condition:
            if self._processed is None or self.camera.snapshot()["state"] != "streaming":
                return {"frame_id": None, "timestamp_ms": None, "people": []}
            frame, people, *_ = self._processed
            return {"frame_id": frame.frame_id, "timestamp_ms": frame.timestamp_ms,
                    "pose_model": self.settings["pose_model"], "people": copy.deepcopy(people),
                    "instances": copy.deepcopy(people)}

    def snapshot(self):
        with self.condition:
            if self._processed is None or self.camera.snapshot()["state"] != "streaming":
                raise ValueError("카메라가 연결되고 영상이 수신된 뒤 저장할 수 있습니다.")
            frame, people, settings, pose_state, device = self._processed
            images = dict(getattr(self, "_snapshot_images", self._images))
        if not all(name in images for name in ("rgb", "depth", "rgbd")):
            depth_image, rgbd = self._backgrounds(frame, settings)
            images.update(self._encode({"rgb": frame.rgb, "depth": depth_image, "rgbd": rgbd}))
        identifier = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
        directory = self.output_dir / identifier
        directory.mkdir()
        files = {}
        try:
            for name, data in images.items():
                filename = f"{name}.jpg"
                (directory / filename).write_bytes(data)
                files[name] = f"/captures/{identifier}/{filename}"
            if not cv2.imwrite(str(directory / "depth16.png"), frame.depth):
                raise RuntimeError("깊이 PNG 저장 실패")
            metadata = {"frame_id": frame.frame_id, "timestamp_ms": frame.timestamp_ms,
                        "timestamp_domain": frame.timestamp_domain, "received_time": frame.received_time,
                        "received_monotonic": getattr(frame, "received_monotonic", None),
                        "depth_scale_m_per_unit": frame.depth_scale,
                        "depth_alignment": "color", "depth_intrinsics": intrinsics_json(frame.intrinsics),
                        "camera": device, "settings": settings,
                        "pose": pose_state, "people": people,
                        "coordinates": "camera optical frame; x right, y down, z forward; metres",
                        "note": "2D model keypoints projected using local median depth, not ground-truth 3D pose"}
            (directory / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            files["depth16"] = f"/captures/{identifier}/depth16.png"
            files["metadata"] = f"/captures/{identifier}/metadata.json"
            return {"id": identifier, "files": files}
        except Exception:
            LOG.exception("Snapshot failed: %s", identifier)
            raise

    def close(self):
        self._stop.set()
        self.recorder.close()
        self.comparison.close()
        with self.condition:
            estimator = self._estimator
            self._estimator = None
        if estimator is not None:
            estimator.close()
        self.camera.stop()
        with self.condition:
            self.condition.notify_all()
        self._worker.join(timeout=10)
        self._pose_worker.join(timeout=10)
