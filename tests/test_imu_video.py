"""IMU movie queries must not leak future samples or join device epochs."""
import unittest
from unittest.mock import patch

import numpy as np

from imu_video import IMUTimeline, metadata, output_size, render_panel


def sample(number, stamp, kind="accel", generation=1, x=1.):
    return {"sample_id": number, "frame_number": number, "received_monotonic": stamp,
            "timestamp_ms": stamp*1000, "timestamp_domain": "hardware_clock",
            "capture_generation": generation, "type": kind,
            "units": "m/s²" if kind == "accel" else "rad/s", "x": x, "y": 2., "z": 9.8}


class IMUVideoTests(unittest.TestCase):
    def test_source_alignment_filters_future_and_other_generations(self):
        timeline = IMUTimeline(100.)
        timeline.add(sample(1, 100.1))
        timeline.add(sample(2, 100.2, "gyro"))
        timeline.add(sample(3, 100.3, x=999.))
        timeline.add(sample(4, 100.4, generation=2, x=555.))
        paired = timeline.view(100.25, generation=1)
        self.assertEqual(paired["streams"]["accel"]["latest"]["sample_id"], 1)
        self.assertEqual(paired["streams"]["gyro"]["latest"]["sample_id"], 2)
        self.assertEqual(len(paired["streams"]["accel"]["points"]), 1)
        # The independent IMU movie follows its own CFR tick, not held Pose.
        self.assertEqual(timeline.view(100.35)["streams"]["accel"]["latest"]["sample_id"], 3)
        reset = timeline.view(100.5)
        self.assertEqual(reset["capture_generation"], 2)
        self.assertIsNone(reset["streams"]["gyro"]["latest"])
        self.assertEqual([s["sample_id"] for s in reset["streams"]["accel"]["points"]], [4])

    def test_deadline_invalid_units_stale_and_history_overflow_are_explicit(self):
        timeline = IMUTimeline(100., capacity=2)
        self.assertFalse(timeline.add(sample(0, 99.9)))
        self.assertFalse(timeline.add({**sample(1, 100.), "units": "g"}))
        self.assertFalse(timeline.add(sample(1, 100., x=float("nan"))))
        for n in range(3):
            self.assertTrue(timeline.add(sample(n, 100.+n*.1)))
        self.assertEqual(timeline.status()["retained_samples"], 2)
        self.assertEqual(timeline.status()["evicted_samples"], 1)
        view = timeline.view(100.3)
        self.assertTrue(view["streams"]["accel"]["history_truncated"])
        self.assertEqual(view["streams"]["gyro"]["state"], "missing")
        self.assertEqual(timeline.view(101.)["streams"]["accel"]["state"], "stale")

    def test_renderer_marks_missing_stale_and_breaks_graph_at_report_gaps(self):
        timeline = IMUTimeline(100.)
        timeline.add(sample(1, 100.01))
        timeline.add(sample(2, 100.02))
        timeline.add(sample(4, 100.04))  # missing report 3
        timeline.add(sample(5, 100.05))
        view = timeline.view(101.)
        labels = []
        import cv2
        real_text, real_lines = cv2.putText, cv2.polylines
        segments = []
        def text(image, value, *args, **kwargs):
            labels.append(value)
            return real_text(image, value, *args, **kwargs)
        def lines(image, points, *args, **kwargs):
            segments.append(len(points[0]))
            return real_lines(image, points, *args, **kwargs)
        with patch("imu_video.cv2.putText", text), patch("imu_video.cv2.polylines", lines):
            image = render_panel(view, 1280, 280, 2., "RGB-D SOURCE")
        self.assertEqual(image.shape, (280, 1280, 3))
        self.assertTrue(any("STALE" in label for label in labels))
        self.assertTrue(any("NO SAMPLES" in label for label in labels))
        self.assertTrue(any("X --" in label for label in labels))
        self.assertEqual(segments, [2]*6, "Each of three axes must have two separate segments")
        self.assertGreater(np.unique(image.reshape(-1, 3), axis=0).shape[0], 30)
        self.assertNotIn("points", metadata(view)["streams"]["accel"])
        self.assertEqual(output_size("combined", 640, 480), (1280, 1240))
        self.assertEqual(output_size("imu", 640, 480), (1280, 280))


if __name__ == "__main__":
    unittest.main()
