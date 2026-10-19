"""Raw motion timing, retention and independent SDK lifecycle regression tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera import Camera
from motion import MotionTracker
from test_camera import FakeSDK, wait_until


class MotionTests(unittest.TestCase):
    def make_tracker(self, capacity=8):
        self.now = 10.
        tracker = MotionTracker(capacity=capacity, clock=lambda: self.now)
        tracker.begin(1, [{"type": "gyro", "fps": 200}, {"type": "accel", "fps": 63}])
        return tracker

    def emit(self, tracker, kind="gyro", number=1, generation=1, **kwargs):
        values = dict(kind=kind, x=1., y=2., z=3., timestamp_ms=number * 5.,
                      timestamp_domain="hardware_clock", frame_number=number,
                      received_time=1_700_000_000. + self.now,
                      received_monotonic=self.now, capture_generation=generation)
        values.update(kwargs)
        return tracker.emit(**values)

    def test_every_raw_sample_reaches_listener_independent_of_ui_retention(self):
        tracker = self.make_tracker()
        received = []
        tracker.set_listener(received.append)
        for number in range(1, 501):
            self.now += .005
            self.emit(tracker, number=number)
        self.assertEqual(len(received), 500)
        self.assertEqual(len({row["sample_id"] for row in received}), 500)
        self.assertEqual(received[-1]["frame_number"], 500)
        self.assertEqual(received[-1]["timestamp_ms"], 2500.)
        self.assertEqual(received[-1]["units"], "rad/s")
        self.assertEqual(tracker.snapshot()["history_evicted"], 492)
        page = tracker.history(after_id=0, limit=3)
        self.assertEqual([row["sample_id"] for row in page["samples"]], [493, 494, 495])
        self.assertEqual(page["missed_samples"], 492)
        self.assertEqual(page["next_id"], 495)
        self.assertEqual([row["sample_id"] for row in tracker.history(495)["samples"]], list(range(496, 501)))

    def test_freshness_and_rates_do_not_display_stale_values_as_live(self):
        tracker = self.make_tracker()
        for number in range(1, 11):
            self.emit(tracker, "gyro", number)
            self.emit(tracker, "accel", number)
            self.now += .005
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot["state"], "streaming")
        self.assertFalse(snapshot["stale"])
        self.assertEqual(snapshot["rates_hz"], {"gyro": 200., "accel": 200.})
        self.assertEqual(snapshot["latest"]["accel"]["units"], "m/s²")
        self.now += 1.1
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot["state"], "stale")
        self.assertTrue(snapshot["stale"])
        self.assertFalse(snapshot["receiving"])
        self.assertEqual(snapshot["rates_hz"], {"gyro": 0., "accel": 0.})
        json.dumps(snapshot)

    def test_reconnect_clears_latest_and_rejects_previous_sensor_callbacks(self):
        tracker = self.make_tracker()
        self.emit(tracker)
        tracker.set_state("disconnected")
        self.assertIsNone(tracker.snapshot()["latest"]["gyro"])
        self.assertIsNone(self.emit(tracker, number=2))
        tracker.begin(2, [])
        self.assertIsNone(self.emit(tracker, number=3, generation=1))
        sample = self.emit(tracker, number=1, generation=2)
        self.assertEqual(sample["sample_id"], 2)
        self.assertEqual(sample["capture_generation"], 2)
        self.assertEqual(tracker.snapshot()["late_callbacks"], 2)
        self.assertEqual(tracker.snapshot()["counts"]["gyro"], 1)
        self.assertEqual(tracker.snapshot()["frame_number_gaps"]["gyro"], 0)

    def test_nonfinite_values_and_listener_failures_are_counted(self):
        tracker = self.make_tracker()
        self.assertIsNone(self.emit(tracker, x=float("nan")))
        def failed_sink(sample):
            raise OSError("storage full")
        tracker.set_listener(failed_sink)
        self.assertIsNotNone(self.emit(tracker, number=2))
        self.assertIsNotNone(self.emit(tracker, number=5))
        snapshot = tracker.snapshot()
        self.assertEqual(snapshot["invalid_samples"], 1)
        self.assertEqual(snapshot["listener_errors"], 2)
        self.assertEqual(snapshot["total_samples"], 2)
        self.assertEqual(snapshot["frame_number_gaps"]["gyro"], 2)

    def test_listener_is_not_invoked_under_tracker_lock(self):
        tracker = self.make_tracker()
        finished = threading.Event()
        def listener(sample):
            worker = threading.Thread(target=lambda: (tracker.snapshot(), finished.set()))
            worker.start()
            worker.join(.5)
            self.assertTrue(finished.is_set(), "IMU listener held the tracker lock")
        tracker.set_listener(listener)
        self.emit(tracker)
        self.assertTrue(finished.is_set())
        self.assertEqual(tracker.snapshot()["listener_errors"], 0)


class FakeMotionSDK(FakeSDK):
    def __init__(self):
        super().__init__()
        self.stream.gyro = "gyro"
        self.stream.accel = "accel"
        self.motion_open_error = False
        self.opened_profiles = []
        self.motion_stops = 0
        self.motion_closes = 0
        self.motion_callbacks = []
        owner = self
        class Sensor:
            def get_stream_profiles(self):
                return [owner.motion_profile("gyro", 400), owner.motion_profile("gyro", 200),
                        owner.motion_profile("accel", 63), owner.motion_profile("accel", 250)]
            def open(self, profiles):
                if owner.motion_open_error:
                    raise RuntimeError("HID permission denied")
                owner.opened_profiles.append(profiles)
            def start(self, callback):
                owner.motion_callbacks.append(callback)
                callback(owner.motion_frame("gyro"))
                callback(owner.motion_frame("accel"))
            def stop(self):
                owner.motion_stops += 1
            def close(self):
                owner.motion_closes += 1
        self.sensor = Sensor()

    def device(self):
        device = super().device()
        device.query_sensors = lambda: [self.sensor]
        return device

    def motion_profile(self, kind, fps):
        return types.SimpleNamespace(stream_type=lambda: kind, fps=lambda: fps, format=lambda: "motion_xyz32f")

    def motion_frame(self, kind):
        return types.SimpleNamespace(
            profile=self.motion_profile(kind, 200),
            as_motion_frame=lambda: types.SimpleNamespace(
                get_motion_data=lambda: types.SimpleNamespace(x=.1, y=9.8, z=-.2)),
            get_timestamp=lambda: 1234., get_frame_timestamp_domain=lambda: "hardware_clock",
            get_frame_number=lambda: 7,
        )


class MotionCameraTests(unittest.TestCase):
    def test_independent_sensor_profiles_callbacks_and_safe_shutdown(self):
        sdk = FakeMotionSDK()
        with patch.dict(sys.modules, {"pyrealsense2": sdk}):
            camera = Camera()
            received = []
            camera.set_imu_listener(received.append)
            try:
                camera.start()
                frame = camera.wait_frame(0, 1.)
                self.assertIsNotNone(frame)
                self.assertEqual(frame.capture_generation, 1)
                self.assertEqual(frame.depth_timestamp_ms, 1234.5)
                self.assertEqual(len(received), 2)
                self.assertEqual(received[0]["type"], "gyro")
                self.assertEqual(received[1]["type"], "accel")
                self.assertEqual(received[0]["timestamp_domain"], "hardware_clock")
                self.assertEqual(received[0]["frame_number"], 7)
                self.assertEqual([profile.fps() for profile in sdk.opened_profiles[0]], [200, 63])
                self.assertEqual(camera.snapshot()["imu"]["counts"], {"gyro": 1, "accel": 1})
                self.assertGreaterEqual(camera.snapshot()["frame_age_ms"], 0.)
                malformed = sdk.motion_frame("gyro")
                malformed.as_motion_frame = lambda: (_ for _ in ()).throw(RuntimeError("bad report"))
                sdk.motion_callbacks[0](malformed)
                sdk.motion_callbacks[0](sdk.motion_frame("gyro"))
                self.assertEqual(camera.motion_snapshot()["callback_errors"], 1)
                self.assertEqual(len(received), 3)
            finally:
                camera.stop()
            self.assertEqual(sdk.motion_stops, 1)
            self.assertEqual(sdk.motion_closes, 1)
            sdk.motion_callbacks[0](sdk.motion_frame("gyro"))
            self.assertEqual(len(received), 3)
            self.assertEqual(camera.snapshot()["imu"]["state"], "disconnected")

    def test_motion_permission_error_does_not_stop_video(self):
        sdk = FakeMotionSDK()
        sdk.motion_open_error = True
        with patch.dict(sys.modules, {"pyrealsense2": sdk}):
            camera = Camera()
            try:
                camera.start()
                self.assertIsNotNone(camera.wait_frame(0, 1.))
                self.assertEqual(camera.snapshot()["state"], "streaming")
                self.assertEqual(camera.snapshot()["imu"]["state"], "error")
                self.assertIn("HID permission", camera.snapshot()["imu"]["error"])
            finally:
                camera.stop()

    def test_frame_listener_receives_each_frame_and_sink_failure_is_isolated(self):
        sdk = FakeSDK()
        with patch.dict(sys.modules, {"pyrealsense2": sdk}):
            camera = Camera()
            frames = []
            def listener(frame):
                frames.append(frame)
                if len(frames) == 1:
                    raise OSError("disk full")
            camera.set_frame_listener(listener)
            try:
                camera.start()
                self.assertTrue(wait_until(lambda: len(frames) >= 8))
                self.assertEqual([frame.frame_id for frame in frames], list(range(1, len(frames) + 1)))
                self.assertEqual(camera.snapshot()["frame_listener_errors"], 1)
                self.assertEqual(camera.snapshot()["imu"]["state"], "unavailable")
            finally:
                camera.stop()


if __name__ == "__main__":
    unittest.main()
