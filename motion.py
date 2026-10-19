"""Lossless motion sample handoff and bounded, explicitly aged UI history.

Only plain numbers leave the SDK callback. The optional listener must enqueue
quickly; CSV encoding and disk I/O belong to its consumer, never this callback.
"""

from __future__ import annotations

from collections import deque
import math
import threading
import time
from typing import Any, Callable


class MotionTracker:
    def __init__(self, capacity: int = 4096, stale_after_s: float = 1.0,
                 *, clock: Callable[[], float] = time.monotonic) -> None:
        self._lock = threading.RLock()
        self._history: deque[dict[str, Any]] = deque(maxlen=max(1, int(capacity)))
        self._clock = clock
        self._stale_after = float(stale_after_s)
        self._listener: Callable[[dict[str, Any]], Any] | None = None
        self._sample_id = 0
        self._generation = 0
        self._state = "disconnected"
        self._error: str | None = None
        self._available = False
        self._profiles: list[dict[str, Any]] = []
        self._latest: dict[str, dict[str, Any]] = {}
        self._times = {kind: deque(maxlen=1200) for kind in ("accel", "gyro")}
        self._counts = {"accel": 0, "gyro": 0}
        self._last_frame: dict[str, int] = {}
        self._frame_gaps = {"accel": 0, "gyro": 0}
        self._history_evicted = 0
        self._listener_errors = 0
        self._invalid_samples = 0
        self._late_callbacks = 0
        self._callback_errors = 0
        self._last_callback_error: str | None = None

    def set_listener(self, listener: Callable[[dict[str, Any]], Any] | None) -> None:
        if listener is not None and not callable(listener):
            raise TypeError("IMU listener must be callable or None")
        with self._lock:
            self._listener = listener

    def begin(self, generation: int, profiles: list[dict[str, Any]]) -> None:
        with self._lock:
            self._generation = int(generation)
            self._available = True
            self._profiles = [dict(profile) for profile in profiles]
            self._latest.clear()
            for values in self._times.values():
                values.clear()
            self._counts = {"accel": 0, "gyro": 0}
            self._last_frame.clear()
            self._frame_gaps = {"accel": 0, "gyro": 0}
            self._state = "initializing"
            self._error = None

    def set_state(self, state: str, error: str | None = None) -> None:
        with self._lock:
            self._state = state
            self._error = error
            if state in {"disconnected", "unavailable", "error", "connecting"}:
                self._available = False
                self._latest.clear()
                for values in self._times.values():
                    values.clear()

    def callback_error(self, error: str) -> None:
        """One malformed callback must not disable the subsequent valid samples."""
        with self._lock:
            self._callback_errors += 1
            self._last_callback_error = str(error)

    def emit(self, *, kind: str, x: float, y: float, z: float,
             timestamp_ms: float, timestamp_domain: str, frame_number: int,
             received_time: float, received_monotonic: float,
             capture_generation: int) -> dict[str, Any] | None:
        """Preserve every finite callback sample; no resampling or averaging."""
        numbers = (x, y, z, timestamp_ms, received_time, received_monotonic)
        if kind not in {"accel", "gyro"} or not all(math.isfinite(float(v)) for v in numbers):
            with self._lock:
                self._invalid_samples += 1
            return None
        with self._lock:
            if capture_generation != self._generation or not self._available:
                self._late_callbacks += 1
                return None
            self._sample_id += 1
            sample = {
                "sample_id": self._sample_id, "type": kind,
                "x": float(x), "y": float(y), "z": float(z),
                "units": "m/s²" if kind == "accel" else "rad/s",
                "timestamp_ms": float(timestamp_ms), "timestamp_domain": str(timestamp_domain),
                "frame_number": int(frame_number), "received_time": float(received_time),
                "received_monotonic": float(received_monotonic),
                "capture_generation": int(capture_generation),
            }
            previous = self._last_frame.get(kind)
            if previous is not None and frame_number > previous + 1:
                self._frame_gaps[kind] += frame_number - previous - 1
            self._last_frame[kind] = int(frame_number)
            if len(self._history) == self._history.maxlen:
                self._history_evicted += 1
            self._history.append(sample)
            self._latest[kind] = sample
            self._times[kind].append(float(received_monotonic))
            self._counts[kind] += 1
            self._state = "streaming"
            listener = self._listener
        if listener is not None:
            try:
                listener(dict(sample))
            except Exception:
                # A data sink failure must not terminate RGB or SDK callbacks.
                with self._lock:
                    self._listener_errors += 1
        return dict(sample)

    def snapshot(self) -> dict[str, Any]:
        now = self._clock()
        with self._lock:
            latest = {}
            rates = {}
            stream_stale = {}
            for kind in ("accel", "gyro"):
                sample = self._latest.get(kind)
                age = max(0.0, now - sample["received_monotonic"]) if sample else None
                stream_stale[kind] = age is None or age > self._stale_after
                latest[kind] = dict(sample, age_ms=round(age * 1000., 2)) if sample else None
                recent = [value for value in self._times[kind] if now - value <= 3.0]
                rates[kind] = round((len(recent) - 1) / (recent[-1] - recent[0]), 2) if (
                    len(recent) > 1 and recent[-1] > recent[0] and not stream_stale[kind]
                ) else 0.0
            stale = any(stream_stale.values())
            state = "stale" if self._state == "streaming" and stale else self._state
            return {
                "state": state, "available": self._available,
                "receiving": any(not value for value in stream_stale.values()),
                "error": self._error, "capture_generation": self._generation,
                "profiles": [dict(profile) for profile in self._profiles],
                "latest": latest, "rates_hz": rates, "counts": dict(self._counts),
                "stale": stale, "stream_stale": stream_stale,
                "stale_after_s": self._stale_after, "total_samples": self._sample_id,
                "frame_number_gaps": dict(self._frame_gaps),
                "history_capacity": self._history.maxlen, "history_evicted": self._history_evicted,
                "listener_errors": self._listener_errors, "invalid_samples": self._invalid_samples,
                "late_callbacks": self._late_callbacks,
                "callback_errors": self._callback_errors,
                "last_callback_error": self._last_callback_error,
            }

    def history(self, after_id: int = 0, limit: int = 512) -> dict[str, Any]:
        """Bounded UI history only; the listener receives samples before eviction."""
        after = max(0, int(after_id))
        maximum = max(1, min(int(limit), 4096))
        with self._lock:
            oldest = self._history[0]["sample_id"] if self._history else self._sample_id + 1
            samples = [dict(sample) for sample in self._history if sample["sample_id"] > after][:maximum]
            return {
                "samples": samples,
                "next_id": samples[-1]["sample_id"] if samples else max(after, self._sample_id),
                "latest_id": self._sample_id,
                "oldest_id": oldest,
                "missed_samples": max(0, oldest - after - 1),
                "capture_generation": self._generation,
            }
