"""Capture lifetime and reconnect tests; no physical camera is opened."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera import Camera


class FakeSDK(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("pyrealsense2")
        self.connected = True
        self.fail_stream = False
        self.fail_start = False
        self.active = 0
        self.max_active = 0
        self.stops = 0
        self.attempts = 0
        self.rgb = np.full((480, 640, 3), 12, dtype=np.uint8)
        self.depth = np.full((480, 640), 2500, dtype=np.uint16)
        self.camera_info = types.SimpleNamespace(
            name="name", serial_number="serial", firmware_version="firmware", usb_type_descriptor="usb"
        )
        self.stream = types.SimpleNamespace(color="color", depth="depth")
        self.format = types.SimpleNamespace(bgr8="bgr8", z16="z16")

    def context(self):
        return types.SimpleNamespace(query_devices=lambda: [self.device()] if self.connected else [])

    def device(self):
        info = {"name": "Intel RealSense D435I", "serial": "123", "firmware": "test", "usb": "3.2"}
        return types.SimpleNamespace(
            supports=lambda key: True,
            get_info=lambda key: info[key],
            first_depth_sensor=lambda: types.SimpleNamespace(get_depth_scale=lambda: 0.001),
        )

    def config(self):
        return types.SimpleNamespace(enable_device=lambda serial: None, enable_stream=lambda *args: None)

    def align(self, stream):
        return types.SimpleNamespace(process=lambda frames: frames)

    def image(self, depth=False):
        intrinsics = types.SimpleNamespace(width=640, height=480, fx=600.0, fy=600.0, ppx=320.0, ppy=240.0)
        return types.SimpleNamespace(
            profile=types.SimpleNamespace(as_video_stream_profile=lambda: types.SimpleNamespace(intrinsics=intrinsics)),
            get_data=lambda: self.depth if depth else self.rgb,
            get_timestamp=lambda: 1234.5,
            get_frame_timestamp_domain=lambda: "hardware_clock",
        )

    def pipeline(self, context):
        owner = self

        class Pipeline:
            def __init__(self):
                self.started = False

            def start(self, config):
                owner.attempts += 1
                if owner.fail_start:
                    raise RuntimeError("start failed")
                self.started = True
                owner.active += 1
                owner.max_active = max(owner.max_active, owner.active)
                return types.SimpleNamespace(get_device=owner.device)

            def wait_for_frames(self, timeout_ms=0):
                time.sleep(0.01)
                if owner.fail_stream or not owner.connected:
                    raise RuntimeError("unplugged")
                return types.SimpleNamespace(
                    get_color_frame=lambda: owner.image(), get_depth_frame=lambda: owner.image(True)
                )

            def stop(self):
                owner.stops += 1
                if self.started:
                    owner.active -= 1
                    self.started = False

        return Pipeline()


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class CameraTests(unittest.TestCase):
    def test_missing_sdk_reports_error_without_fake_frames(self):
        with patch.dict(sys.modules, {"pyrealsense2": None}):
            camera = Camera()
            try:
                camera.start()
                self.assertTrue(wait_until(lambda: camera.snapshot()["state"] == "error"))
                self.assertIn("SDK", camera.snapshot()["error"])
                self.assertIsNone(camera.wait_frame(0, 0.01))
            finally:
                camera.stop()

    def test_frame_owns_rgb_and_raw_depth_memory(self):
        sdk = FakeSDK()
        with patch.dict(sys.modules, {"pyrealsense2": sdk}):
            camera = Camera()
            try:
                camera.start()
                frame = camera.wait_frame(0, 1.0)
                self.assertIsNotNone(frame)
                sdk.rgb.fill(99)
                sdk.depth.fill(1000)
                self.assertEqual(int(frame.rgb[0, 0, 0]), 12)
                self.assertEqual(int(frame.depth[0, 0]), 2500)
                self.assertEqual(frame.depth.dtype, np.uint16)
                self.assertEqual(frame.depth_scale, 0.001)
                self.assertEqual(frame.timestamp_ms, 1234.5)
                self.assertEqual(frame.timestamp_domain, "hardware_clock")
                self.assertGreater(frame.received_time, 1_000_000_000)
                self.assertGreater(frame.received_monotonic, 0.)
                self.assertLessEqual(frame.received_monotonic, time.monotonic())
                json.dumps(camera.snapshot())
            finally:
                camera.stop()
            self.assertEqual(sdk.active, 0)
            self.assertEqual(sdk.stops, 1)

    def test_unplug_clears_previous_frame(self):
        sdk = FakeSDK()
        with patch.dict(sys.modules, {"pyrealsense2": sdk}):
            camera = Camera()
            try:
                camera.start()
                self.assertIsNotNone(camera.wait_frame(0, 1.0))
                sdk.connected = False
                self.assertTrue(wait_until(lambda: camera.snapshot()["state"] == "error"))
                self.assertIsNone(camera.wait_frame(0, 0.01))
                self.assertEqual(camera.snapshot()["fps"], 0.0)
            finally:
                camera.stop()

    def test_initial_missing_device_reconnects_automatically(self):
        sdk = FakeSDK()
        sdk.connected = False
        with patch.dict(sys.modules, {"pyrealsense2": sdk}):
            camera = Camera()
            try:
                camera.start()
                self.assertTrue(wait_until(lambda: camera.snapshot()["state"] == "disconnected"))
                self.assertIn("D435i", camera.snapshot()["error"])
                self.assertIsNone(camera.wait_frame(0, 0.01))
                sdk.connected = True
                self.assertIsNotNone(camera.wait_frame(0, 3.0))
                self.assertEqual(camera.snapshot()["state"], "streaming")
            finally:
                camera.stop()

    def test_failed_start_still_attempts_pipeline_cleanup(self):
        sdk = FakeSDK()
        sdk.fail_start = True
        with patch.dict(sys.modules, {"pyrealsense2": sdk}):
            camera = Camera()
            try:
                camera.start()
                self.assertTrue(wait_until(lambda: sdk.stops > 0))
                self.assertEqual(camera.snapshot()["state"], "error")
                self.assertIsNone(camera.wait_frame(0, 0.01))
            finally:
                camera.stop()
            self.assertEqual(sdk.active, 0)

    def test_overlapping_stop_and_start_have_only_one_pipeline(self):
        sdk = FakeSDK()
        with patch.dict(sys.modules, {"pyrealsense2": sdk}):
            camera = Camera()
            try:
                camera.start()
                self.assertIsNotNone(camera.wait_frame(0, 1.0))
                stop_worker = threading.Thread(target=camera.stop)
                start_worker = threading.Thread(target=camera.start)
                stop_worker.start()
                start_worker.start()
                stop_worker.join(2.0)
                start_worker.join(2.0)
                self.assertFalse(stop_worker.is_alive())
                self.assertFalse(start_worker.is_alive())
                self.assertEqual(sdk.max_active, 1)
            finally:
                camera.stop()
            self.assertEqual(sdk.active, 0)


if __name__ == "__main__":
    unittest.main()
