"""Bounded IMU history and offline video panels on the capture receipt clock.

Rendering never runs in the SDK callback. Queries only include samples at or
before their reference time and never bridge device generations or gaps.
"""
from collections import deque, OrderedDict
import math
import threading

import cv2
import numpy as np


AXES = ("x", "y", "z")
COLORS = ((120, 140, 255), (150, 225, 95), (255, 185, 105))  # BGR


def panel_height(width):
    return 280 if width >= 800 else 160


def output_size(name, width, height):
    if name == "combined":
        return width * 2, height * 2 + panel_height(width * 2)
    if name == "imu":
        return width * 2, panel_height(width * 2)
    return width, height


class IMUTimeline:
    def __init__(self, start_monotonic, capacity=16384, window_s=10., stale_after_s=.5):
        self.start = float(start_monotonic)
        self.window_s, self.stale_after_s = float(window_s), float(stale_after_s)
        self._samples = deque(maxlen=max(1, int(capacity)))
        self._lock = threading.Lock()
        self._evicted_through = OrderedDict()
        self._accepted = self._evicted = self._invalid = 0

    def add(self, sample):
        try:
            kind = sample["type"]
            stamp = float(sample["received_monotonic"])
            values = [float(sample[axis]) for axis in AXES]
            generation = int(sample["capture_generation"])
            unit = sample["units"]
            expected = ("m/s²", "m/s^2", "m/s2") if kind == "accel" else ("rad/s",)
            if kind not in ("accel", "gyro") or unit not in expected or not all(map(math.isfinite, [stamp] + values)):
                raise ValueError("Invalid motion sample")
        except (KeyError, TypeError, ValueError, OverflowError):
            with self._lock:
                self._invalid += 1
            return False
        if stamp < self.start:
            return False
        frozen = {key: sample.get(key) for key in ("sample_id", "frame_number", "timestamp_ms", "timestamp_domain")}
        frozen.update(type=kind, received_monotonic=stamp, capture_generation=generation,
                      units=unit, **dict(zip(AXES, values)))
        with self._lock:
            if len(self._samples) == self._samples.maxlen:
                old = self._samples[0]
                key = (old["capture_generation"], old["type"])
                self._evicted_through[key] = max(old["received_monotonic"], self._evicted_through.get(key, 0.))
                self._evicted_through.move_to_end(key)
                while len(self._evicted_through) > 64:
                    self._evicted_through.popitem(last=False)
                self._evicted += 1
            self._samples.append(frozen)
            self._accepted += 1
        return True

    def status(self):
        with self._lock:
            return {"accepted_samples": self._accepted, "retained_samples": len(self._samples),
                    "capacity": self._samples.maxlen, "evicted_samples": self._evicted,
                    "invalid_samples": self._invalid, "window_s": self.window_s,
                    "stale_after_s": self.stale_after_s}

    def view(self, reference_monotonic, generation=None):
        reference = float(reference_monotonic)
        with self._lock:
            samples = list(self._samples)
            evicted = dict(self._evicted_through)
        samples = [s for s in samples if s["received_monotonic"] <= reference]
        if generation is None and samples:
            generation = max(samples, key=lambda s: s["received_monotonic"])["capture_generation"]
        samples = [s for s in samples if s["capture_generation"] == generation]
        start = max(self.start, reference - self.window_s)
        result = {"reference_monotonic": reference, "reference_elapsed_s": reference - self.start,
                  "capture_generation": generation, "window_start_monotonic": start,
                  "window_s": self.window_s, "streams": {}}
        for kind in ("accel", "gyro"):
            stream_samples = [s for s in samples if s["type"] == kind]
            latest = max(stream_samples, key=lambda s: s["received_monotonic"], default=None)
            age = reference - latest["received_monotonic"] if latest else None
            # Sort only the visible ten-second window, not the whole recording.
            points = sorted((s for s in stream_samples if s["received_monotonic"] >= start),
                            key=lambda s: s["received_monotonic"])
            truncated = evicted.get((generation, kind), -math.inf) >= start
            state = "missing" if latest is None else "stale" if age > self.stale_after_s else "receiving"
            result["streams"][kind] = {"state": state, "latest": latest, "age_ms": age * 1000 if age is not None else None,
                "points": points, "history_truncated": truncated,
                "hz": (len(points)-1)/(points[-1]["received_monotonic"]-points[0]["received_monotonic"])
                if len(points) > 1 and points[-1]["received_monotonic"] > points[0]["received_monotonic"] else 0.}
        return result


def metadata(view):
    return {key: value for key, value in view.items() if key != "streams"} | {
        "streams": {kind: {key: value for key, value in stream.items() if key != "points"} |
                    {"samples_in_window": len(stream["points"])} for kind, stream in view["streams"].items()}}


def render_panel(view, width, height, video_elapsed_s, reference_label="VIDEO TIMELINE"):
    canvas = np.full((height, width, 3), (29, 23, 16), np.uint8)
    if width < 200 or height < 130:
        return canvas
    text = lambda image, value, xy, color=(185, 203, 213), scale=.43: cv2.putText(
        image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    reference_elapsed = view["reference_elapsed_s"]
    text(canvas, f"IMU | {reference_label} {reference_elapsed:.2f}s | video {video_elapsed_s:.2f}s | last {view['window_s']:g}s",
         (14, 21), (220, 235, 242), .48)
    for index, kind in enumerate(("accel", "gyro")):
        left, right = index * width // 2, (index + 1) * width // 2
        panel = canvas[:, left:right]
        stream = view["streams"][kind]
        points, latest = stream["points"], stream["latest"]
        fresh = stream["state"] == "receiving"
        unit = "m/s^2" if kind == "accel" else "rad/s"
        state = f"{stream['hz']:.1f} Hz" if fresh else "NO SAMPLES" if latest is None else f"STALE {stream['age_ms']/1000:.2f}s"
        text(panel, f"{kind.upper()} [{unit}]   {state}", (14, 45), (100, 220, 205) if fresh else (130, 180, 240))
        for axis_index, axis in enumerate(AXES):
            value = f"{latest[axis]:+.3f}" if fresh else "--"
            text(panel, f"{axis.upper()} {value}", (14 + axis_index * (right-left-25)//3, 66), COLORS[axis_index])
        x0, x1 = 48, right - left - 14
        y0, y1 = 89, height - 30
        if x1 <= x0 or y1 <= y0:
            continue
        # Transform all three axes in NumPy. Python per-sample/per-axis drawing
        # exceeded the 15 FPS frame budget once a full ten seconds accumulated.
        values = np.array([[s[axis] for axis in AXES] for s in points], dtype=np.float64).reshape(-1, 3)
        bound = max(10. if kind == "accel" else .5, float(np.max(np.abs(values))) * 1.1 if len(values) else 0.)
        for fraction in (0., .5, 1.):
            y = round(y0 + fraction * (y1-y0))
            cv2.line(panel, (x0, y), (x1, y), (60, 51, 40), 1)
            text(panel, f"{bound*(1-2*fraction):+.1f}", (2, y+4), (125, 147, 158), .34)
        reference = view["reference_monotonic"]
        stamps = np.array([s["received_monotonic"] for s in points], dtype=np.float64)
        frames = np.array([s.get("frame_number") if s.get("frame_number") is not None else np.nan
                           for s in points], dtype=np.float64)
        cuts = np.flatnonzero((np.diff(stamps) > .15) | (np.diff(frames) > 1)) + 1
        xs = np.clip(np.rint(x1 - (reference-stamps)/view["window_s"] * (x1-x0)), x0, x1).astype(np.int32)
        ys = np.clip(np.rint((y0+y1)/2 - values/bound * (y1-y0)/2), y0, y1).astype(np.int32)
        for axis_index, axis in enumerate(AXES):
            # Preserve all samples and report gaps, including one-point segments.
            coordinates = np.column_stack((xs, ys[:, axis_index]))
            for segment in np.split(coordinates, cuts):
                if len(segment) == 1:
                    cv2.circle(panel, tuple(segment[0]), 1, COLORS[axis_index], -1)
                elif len(segment) > 1:
                    cv2.polylines(panel, [segment], False, COLORS[axis_index], 1, cv2.LINE_AA)
        text(panel, f"-{view['window_s']:g}s", (x0, height-10), (125, 147, 158), .35)
        text(panel, "0s", (x1-16, height-10), (125, 147, 158), .35)
        if stream["history_truncated"]:
            text(panel, "HISTORY BUFFER GAP", (x0+45, height-10), (130, 180, 240), .35)
    cv2.line(canvas, (width//2, 32), (width//2, height-4), (65, 52, 40), 1)
    return canvas
