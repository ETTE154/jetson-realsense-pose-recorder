"""Depth validity and projection checks independent of a physical camera."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import threading
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prototype import Application, JOINTS, colorize_depth, enrich_people


class DepthProjectionTests(unittest.TestCase):
    def test_invalid_depth_is_black(self):
        depth = np.array([[0, 1000], [2000, 3000]], dtype=np.uint16)
        colored = colorize_depth(depth, .001, .3, 4.)
        np.testing.assert_array_equal(colored[0, 0], [0, 0, 0])
        self.assertGreater(int(colored[1, 0].sum()), 0)
        np.testing.assert_array_equal(depth, [[0, 1000], [2000, 3000]])

    def test_metric_depth_uses_device_scale_and_median(self):
        depth = np.full((7, 7), 4000, dtype=np.uint16)
        depth[2, 2] = 65000
        depth[3, 2] = 0
        frame = SimpleNamespace(depth=depth, depth_scale=.0005, intrinsics=object())
        people = [{"bbox": [0, 0, 6, 6], "score": .9,
                   "keypoints": [[3., 3., .8] for _ in JOINTS]}]
        calls = []
        def project(intrinsics, uv, z):
            calls.append((uv, z))
            return [0.1, 0.2, z]
        result = enrich_people(people, frame, project)
        self.assertEqual(len(calls), 17)
        self.assertEqual(result[0]["keypoints"][0]["z_m"], 2.)
        self.assertTrue(result[0]["keypoints"][0]["valid_depth"])

    def test_missing_depth_low_confidence_and_outside_are_null(self):
        frame = SimpleNamespace(depth=np.zeros((7, 7), dtype=np.uint16),
                                depth_scale=.001, intrinsics=object())
        people = [{"bbox": [0, 0, 6, 6], "score": .9,
                   "keypoints": [[3, 3, .8], [-1, 3, .9], [3, 3, .1]] +
                                [[3, 3, .8]] * 14}]
        def project(*_):
            self.fail("Invalid depth must never be projected")
        result = enrich_people(people, frame, project)
        for joint in result[0]["keypoints"]:
            self.assertFalse(joint["valid_depth"])
            self.assertIsNone(joint["z_m"])


class SnapshotFidelityTests(unittest.TestCase):
    def test_saved_snapshot_keeps_frame_settings_and_lossless_z16(self):
        # Synthetic arrays deliberately include values that cannot fit in 8 bits.
        depth = np.array([[0, 1, 255, 256, 65535],
                          [1000, 4000, 12000, 32000, 50000]], dtype=np.uint16)
        rgb = np.full((2, 5, 3), (30, 90, 180), dtype=np.uint8)
        ok, jpeg = cv2.imencode(".jpg", rgb)
        self.assertTrue(ok)
        intrinsics = SimpleNamespace(width=5, height=2, ppx=2., ppy=.5,
                                     fx=10., fy=10., model="none", coeffs=[0.] * 5)
        frame = SimpleNamespace(frame_id=42, rgb=rgb, depth=depth, depth_scale=.0005,
                                intrinsics=intrinsics, timestamp_ms=1234.5,
                                timestamp_domain="hardware_clock", received_time=1_700_000_000.)
        frame_settings = {"confidence": .35, "depth_min": .3, "depth_max": 4., "pose_enabled": True}
        frame_pose = {"state": "ready", "backend": "synthetic-test", "people": 1}
        frame_device = {"name": "synthetic D435i", "serial": "frame-device"}
        people = [{"bbox": [0., 0., 4., 1.], "score": .9, "keypoints": []}]
        application = object.__new__(Application)
        application.condition = threading.Condition(threading.RLock())
        application.camera = SimpleNamespace(snapshot=lambda: {
            "state": "streaming", "device": {"serial": "current-device"}})
        application._processed = (frame, people, dict(frame_settings), dict(frame_pose), dict(frame_device))
        application._images = {name: jpeg.tobytes() for name in ("rgb", "depth", "rgbd", "pose", "pose_rgbd")}
        # These represent UI changes occurring after this frame was rendered.
        application.settings = {**frame_settings, "confidence": .9, "depth_max": 8., "pose_enabled": False}
        application.pose = {"state": "disabled", "people": 0}
        with tempfile.TemporaryDirectory(prefix="d435i-snapshot-test-") as temporary:
            application.output_dir = Path(temporary)
            saved = application.snapshot()
            directory = Path(temporary) / saved["id"]
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            restored = cv2.imread(str(directory / "depth16.png"), cv2.IMREAD_UNCHANGED)
            self.assertEqual(restored.dtype, np.uint16)
            np.testing.assert_array_equal(restored, depth)
            self.assertEqual(metadata["settings"], frame_settings)
            self.assertEqual(metadata["pose"], frame_pose)
            self.assertEqual(metadata["camera"], frame_device)
            self.assertEqual(metadata["people"], people)
            self.assertEqual(metadata["frame_id"], 42)
            self.assertEqual(metadata["timestamp_ms"], 1234.5)
            self.assertEqual(metadata["depth_scale_m_per_unit"], .0005)
            for name in application._images:
                self.assertEqual((directory / f"{name}.jpg").read_bytes(), application._images[name])
            self.assertIn("depth16", saved["files"])
            self.assertIn("metadata", saved["files"])


if __name__ == "__main__":
    unittest.main()
