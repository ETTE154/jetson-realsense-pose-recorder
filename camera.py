"""RealSense capture with one thread owning every SDK pipeline operation."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from motion import MotionTracker


@dataclass(frozen=True)
class Frame:
    frame_id: int
    rgb: np.ndarray  # OpenCV BGR, independent of the SDK's buffer lifetime.
    depth: np.ndarray  # Original Z16 values, aligned to the color image.
    depth_scale: float
    intrinsics: Any
    timestamp_ms: float
    timestamp_domain: str
    received_time: float  # Unix time in seconds when capture completed.
    received_monotonic: float = 0.0  # Server receipt clock; recording delay gate.
    capture_generation: int = 0
    color_frame_number: int | None = None
    depth_frame_number: int | None = None
    depth_timestamp_ms: float | None = None
    depth_timestamp_domain: str | None = None


class Camera:
    """Keep the newest color/depth pair and reconnect after camera removal.

    ``start`` is nonblocking. Capture and SDK shutdown happen exclusively inside
    the worker, so API handlers never race ``wait_for_frames`` with ``stop``.
    ``wait_frame`` returns the newest frame, not a queue of historical images.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: str | None = None,
    ) -> None:
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("width, height, fps는 양수여야 합니다.")
        self.width = int(width)
        self.height = int(height)
        self.requested_fps = int(fps)
        self.serial = serial
        self._lifecycle_lock = threading.Lock()
        self._condition = threading.Condition(threading.RLock())
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._latest: Frame | None = None
        self._frame_id = 0
        self._state = "disconnected"
        self._device: dict[str, str] = {}
        self._error: str | None = None
        self._fps = 0.0
        self._capture_generation = 0
        self._motion = MotionTracker()
        self._frame_listener = None
        self._frame_listener_errors = 0

    def set_frame_listener(self, listener: Any) -> None:
        """Receive every owned RGB-D frame; listener must enqueue without I/O."""
        if listener is not None and not callable(listener):
            raise TypeError("Frame listener must be callable or None")
        with self._condition:
            self._frame_listener = listener

    def set_imu_listener(self, listener: Any) -> None:
        """Receive every gyro/accel sample, independently of video refresh rate."""
        self._motion.set_listener(listener)

    def motion_snapshot(self) -> dict[str, Any]:
        return self._motion.snapshot()

    def motion_history(self, after_id: int = 0, limit: int = 512) -> dict[str, Any]:
        return self._motion.history(after_id, limit)

    def start(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            return self._start()

    def _start(self) -> dict[str, Any]:
        with self._condition:
            # A worker still shutting down must finish before another opens USB.
            if self._thread is not None and self._thread.is_alive():
                return self._snapshot_locked()
            self._stop_event = threading.Event()
            self._state = "connecting"
            self._error = None
            self._fps = 0.0
            self._latest = None
            self._device = {}
            self._motion.set_state("connecting")
            self._thread = threading.Thread(
                target=self._capture,
                args=(self._stop_event,),
                name="realsense-capture",
                daemon=True,
            )
            self._thread.start()
            self._condition.notify_all()
            return self._snapshot_locked()

    def stop(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            return self._stop()

    def _stop(self) -> dict[str, Any]:
        with self._condition:
            worker = self._thread
            event = self._stop_event
            if event is not None:
                event.set()
            # Reject callbacks immediately; the capture owner performs SDK stop.
            self._motion.set_state("disconnected")
            self._latest = None
            self._fps = 0.0
            self._condition.notify_all()
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=6.0)
        with self._condition:
            if worker is not None and worker.is_alive():
                self._state = "error"
                self._error = "카메라 종료를 기다리는 중입니다. USB 응답 후 안전하게 종료됩니다."
            else:
                self._state = "disconnected"
                self._error = None
                self._device = {}
            self._condition.notify_all()
            return self._snapshot_locked()

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "state": self._state,
            "active": bool(self._thread is not None and self._thread.is_alive()
                           and self._stop_event is not None and not self._stop_event.is_set()),
            "device": dict(self._device),
            "error": self._error,
            "fps": round(self._fps, 2),
            "width": self.width,
            "height": self.height,
            "requested_fps": self.requested_fps,
            "frame_id": self._frame_id,
            "capture_generation": self._capture_generation,
            "frame_age_ms": round(max(0., time.monotonic() - self._latest.received_monotonic) * 1000., 2)
            if self._latest is not None else None,
            "frame_listener_errors": self._frame_listener_errors,
            "imu": self._motion.snapshot(),
        }

    def wait_frame(self, after_id: int, timeout: float = 1.0) -> Frame | None:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while True:
                if (
                    self._state == "streaming"
                    and self._latest is not None
                    and self._latest.frame_id > after_id
                    and self._stop_event is not None
                    and not self._stop_event.is_set()
                ):
                    return self._latest
                if self._stop_event is None or self._stop_event.is_set():
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

    def _set_state(
        self, state: str, error: str | None = None, device: dict[str, str] | None = None
    ) -> None:
        with self._condition:
            self._state = state
            self._error = error
            if device is not None:
                self._device = dict(device)
            if state != "streaming":
                self._latest = None
                self._fps = 0.0
            self._condition.notify_all()

    @staticmethod
    def _device_info(rs: Any, device: Any) -> dict[str, str]:
        result = {}
        for key, info in (
            ("name", rs.camera_info.name),
            ("serial", rs.camera_info.serial_number),
            ("firmware", rs.camera_info.firmware_version),
            ("usb", rs.camera_info.usb_type_descriptor),
        ):
            try:
                result[key] = str(device.get_info(info)) if device.supports(info) else ""
            except RuntimeError:
                result[key] = ""
        return result

    def _capture(self, stop: threading.Event) -> None:
        try:
            import pyrealsense2 as rs
        except Exception as exc:
            self._set_state("error", f"RealSense SDK를 불러올 수 없습니다: {exc}")
            self._motion.set_state("error", f"RealSense SDK를 불러올 수 없습니다: {exc}")
            return

        try:
            while not stop.is_set():
                pipeline = None
                motion_sensor = None
                try:
                    self._set_state("connecting")
                    self._motion.set_state("connecting")
                    context = rs.context()
                    devices = list(context.query_devices())
                    selected = None
                    for device in devices:
                        info = self._device_info(rs, device)
                        if self.serial and info.get("serial") != self.serial:
                            continue
                        # Do not silently select another camera when a serial was supplied.
                        if self.serial or "D435" in info.get("name", ""):
                            selected = device
                            break
                    if selected is None:
                        target = f" (시리얼 {self.serial})" if self.serial else ""
                        self._set_state(
                            "disconnected",
                            f"RealSense D435i{target}를 찾을 수 없습니다. USB 3.0 연결과 전원을 확인하세요. 자동 재연결 중입니다.",
                            {},
                        )
                        self._motion.set_state("disconnected", "D435i가 연결되지 않았습니다.")
                        stop.wait(2.0)
                        continue
                    info = self._device_info(rs, selected)
                    config = rs.config()
                    config.enable_device(info["serial"])
                    config.enable_stream(
                        rs.stream.color, self.width, self.height, rs.format.bgr8, self.requested_fps
                    )
                    config.enable_stream(
                        rs.stream.depth, self.width, self.height, rs.format.z16, self.requested_fps
                    )
                    pipeline = rs.pipeline(context)
                    profile = pipeline.start(config)
                    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
                    align = rs.align(rs.stream.color)
                    capture_times: deque[float] = deque(maxlen=60)
                    # A successfully opened pipeline is not streaming until a valid pair arrives.
                    self._set_state("connecting", device=info)
                    with self._condition:
                        self._capture_generation += 1
                        generation = self._capture_generation
                    motion_sensor = self._start_motion(rs, profile.get_device(), generation)
                    while not stop.is_set():
                        frames = pipeline.wait_for_frames(timeout_ms=1500)
                        aligned = align.process(frames)
                        color = aligned.get_color_frame()
                        depth = aligned.get_depth_frame()
                        if not color or not depth:
                            continue
                        rgb_array = np.asanyarray(color.get_data()).copy()
                        depth_array = np.asanyarray(depth.get_data()).copy()
                        if depth_array.shape != rgb_array.shape[:2]:
                            raise RuntimeError("정렬된 RGB와 깊이 영상 크기가 일치하지 않습니다.")
                        rgb_array.setflags(write=False)
                        depth_array.setflags(write=False)
                        intrinsics = color.profile.as_video_stream_profile().intrinsics
                        received_time = time.time()
                        now = time.monotonic()
                        capture_times.append(now)
                        fps = 0.0
                        if len(capture_times) > 1 and now > capture_times[0]:
                            fps = (len(capture_times) - 1) / (now - capture_times[0])
                        with self._condition:
                            if stop.is_set():
                                break
                            self._frame_id += 1
                            frame = Frame(
                                frame_id=self._frame_id,
                                rgb=rgb_array,
                                depth=depth_array,
                                depth_scale=depth_scale,
                                intrinsics=intrinsics,
                                timestamp_ms=float(color.get_timestamp()),
                                timestamp_domain=str(color.get_frame_timestamp_domain()),
                                received_time=received_time,
                                received_monotonic=now,
                                capture_generation=generation,
                                color_frame_number=int(color.get_frame_number())
                                if hasattr(color, "get_frame_number") else None,
                                depth_frame_number=int(depth.get_frame_number())
                                if hasattr(depth, "get_frame_number") else None,
                                depth_timestamp_ms=float(depth.get_timestamp()),
                                depth_timestamp_domain=str(depth.get_frame_timestamp_domain()),
                            )
                            self._latest = frame
                            self._fps = fps
                            self._state = "streaming"
                            self._error = None
                            frame_listener = self._frame_listener
                            self._condition.notify_all()
                        if frame_listener is not None and not stop.is_set():
                            try:
                                frame_listener(frame)
                            except Exception:
                                with self._condition:
                                    self._frame_listener_errors += 1
                except Exception as exc:
                    if not stop.is_set():
                        self._set_state(
                            "error",
                            f"카메라 영상을 받을 수 없습니다: {exc}. USB 연결과 SDK 권한을 확인하세요. 자동 재연결 중입니다.",
                            {},
                        )
                        self._motion.set_state("disconnected", f"카메라 연결이 중단되었습니다: {exc}")
                finally:
                    # No other thread calls SDK stop, even if API shutdown timed out.
                    if motion_sensor is not None:
                        try:
                            motion_sensor.stop()
                        except Exception:
                            pass
                        try:
                            motion_sensor.close()
                        except Exception:
                            pass
                    if pipeline is not None:
                        try:
                            pipeline.stop()
                        except Exception:
                            pass
                if not stop.is_set():
                    stop.wait(2.0)
        finally:
            if stop.is_set():
                self._set_state("disconnected", device={})
                self._motion.set_state("disconnected")

    def _start_motion(self, rs: Any, device: Any, generation: int) -> Any:
        """Open the independent motion sensor, leaving RGB-D pipeline unaffected."""
        opened = None
        try:
            gyro = getattr(rs.stream, "gyro", None)
            accel = getattr(rs.stream, "accel", None)
            if gyro is None or accel is None or not hasattr(device, "query_sensors"):
                self._motion.set_state("unavailable", "SDK에서 D435i Motion Module을 찾을 수 없습니다.")
                return None
            candidates = []
            for sensor in device.query_sensors():
                by_kind: dict[str, list[Any]] = {"gyro": [], "accel": []}
                for stream_profile in sensor.get_stream_profiles():
                    stream = stream_profile.stream_type()
                    if stream == gyro:
                        by_kind["gyro"].append(stream_profile)
                    elif stream == accel:
                        by_kind["accel"].append(stream_profile)
                if by_kind["gyro"] or by_kind["accel"]:
                    candidates.append((sensor, by_kind))
            if not candidates:
                self._motion.set_state(
                    "unavailable",
                    "D435i Motion Module이 SDK에 노출되지 않았습니다. HID/IIO 드라이버와 장치 권한을 확인하세요.",
                )
                return None
            sensor, by_kind = max(candidates, key=lambda item: sum(bool(v) for v in item[1].values()))
            selected = []
            descriptions = []
            for kind in ("gyro", "accel"):
                options = by_kind[kind]
                if not options:
                    continue
                target = 200 if kind == "gyro" else 63
                stream_profile = min(options, key=lambda value: abs(value.fps() - target))
                selected.append(stream_profile)
                descriptions.append({"type": kind, "fps": int(stream_profile.fps()),
                                     "format": str(stream_profile.format()),
                                     "units": "rad/s" if kind == "gyro" else "m/s²"})
            sensor.open(selected)
            opened = sensor
            self._motion.begin(generation, descriptions)

            def on_motion(frame: Any) -> None:
                received_time = time.time()
                received_monotonic = time.monotonic()
                try:
                    stream = frame.profile.stream_type()
                    kind = "gyro" if stream == gyro else "accel" if stream == accel else None
                    if kind is None:
                        return
                    vector = frame.as_motion_frame().get_motion_data()
                    self._motion.emit(
                        kind=kind, x=float(vector.x), y=float(vector.y), z=float(vector.z),
                        timestamp_ms=float(frame.get_timestamp()),
                        timestamp_domain=str(frame.get_frame_timestamp_domain()),
                        frame_number=int(frame.get_frame_number()), received_time=received_time,
                        received_monotonic=received_monotonic, capture_generation=generation,
                    )
                except Exception as exc:
                    self._motion.callback_error(f"IMU 샘플을 읽을 수 없습니다: {exc}")

            sensor.start(on_motion)
            return sensor
        except Exception as exc:
            if opened is not None:
                try:
                    opened.stop()
                except Exception:
                    pass
                try:
                    opened.close()
                except Exception:
                    pass
            self._motion.set_state("error", f"D435i IMU 스트림을 시작할 수 없습니다: {exc}")
            return None
