"""Geometry and postprocessing tests; no camera, CUDA, or OpenCV required."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_runtime import decode_poses, letterbox_geometry, non_max_suppression


def raw_person(transform, bbox, confidence=0.9, joint=(250, 160, 0.8)):
    x1, y1, x2, y2 = bbox
    row = np.zeros(56, dtype=np.float32)
    row[:5] = [
        (x1 + x2) / 2 * transform.scale_x + transform.left,
        (y1 + y2) / 2 * transform.scale_y + transform.top,
        (x2 - x1) * transform.scale_x,
        (y2 - y1) * transform.scale_y,
        confidence,
    ]
    row[5:] = np.tile([
        joint[0] * transform.scale_x + transform.left,
        joint[1] * transform.scale_y + transform.top, joint[2],
    ], 17)
    return row


class PosePostprocessingTests(unittest.TestCase):
    def test_landscape_coordinates_survive_letterbox(self):
        transform = letterbox_geometry(480, 640, 640)
        self.assertEqual((transform.left, transform.top), (0, 80))
        row = raw_person(transform, (100, 50, 400, 440))
        result = decode_poses(row[None, :, None], transform)[0]
        np.testing.assert_allclose(result["bbox"], [100, 50, 400, 440], atol=1e-4)
        np.testing.assert_allclose(result["keypoints"][0], [250, 160, 0.8], atol=1e-4)
        self.assertEqual(len(result["keypoints"]), 17)
        json.dumps(result, allow_nan=False)

    def test_portrait_odd_dimensions_use_actual_rounded_scale(self):
        transform = letterbox_geometry(1001, 601, 320)
        self.assertEqual(transform.resized_height, 320)
        self.assertEqual(transform.resized_width, 192)
        row = raw_person(transform, (20, 50, 550, 950), joint=(300, 500, .7))
        result = decode_poses(row[None, None, :], transform)[0]
        np.testing.assert_allclose(result["bbox"], [20, 50, 550, 950], atol=1e-3)
        np.testing.assert_allclose(result["keypoints"][0], [300, 500, .7], atol=1e-3)

    def test_nms_removes_duplicate_and_keeps_separate_person(self):
        boxes = np.array([[0, 0, 100, 100], [2, 2, 98, 98], [200, 0, 300, 100]], dtype=np.float32)
        selected = non_max_suppression(boxes, np.array([.9, .8, .7]))
        self.assertEqual(selected, [0, 2])
        self.assertEqual(non_max_suppression(boxes, np.array([.9, .8, .7]), max_detections=1), [0])

    def test_rejects_low_confidence_nonfinite_and_invalid_boxes(self):
        transform = letterbox_geometry(480, 640, 640)
        low = raw_person(transform, (10, 10, 100, 100), confidence=.1)
        nan_row = raw_person(transform, (10, 10, 100, 100))
        nan_row[5] = np.nan
        negative = raw_person(transform, (100, 10, 10, 100))
        self.assertEqual(decode_poses(np.stack([low, nan_row, negative]).T[None], transform), [])

    def test_outside_joint_cannot_be_used_for_edge_depth(self):
        transform = letterbox_geometry(480, 640, 320)
        row = raw_person(transform, (-20, -30, 300, 300), joint=(-10, 500, .9))
        result = decode_poses(row[:, None], transform)[0]
        self.assertEqual(result["bbox"][:2], [0.0, 0.0])
        self.assertEqual(result["keypoints"][0], [0.0, 479.0, 0.0])

    def test_bad_output_shape_fails_explicitly(self):
        transform = letterbox_geometry(480, 640, 640)
        with self.assertRaisesRegex(ValueError, "56 pose features"):
            decode_poses(np.zeros((1, 84, 8400)), transform)
        self.assertEqual(decode_poses(np.zeros((1, 56, 0)), transform), [])


if __name__ == "__main__":
    unittest.main()
