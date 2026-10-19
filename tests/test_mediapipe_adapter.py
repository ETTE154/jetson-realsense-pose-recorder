"""Conversion tests cover API semantics without requiring the task runtime."""
from types import SimpleNamespace
import unittest

from comparison.mediapipe_adapter import convert_result


class ConversionTests(unittest.TestCase):
    def test_pixel_coordinates_and_explicit_score_meaning(self):
        landmarks = [SimpleNamespace(x=0.25 + i * .01, y=.5) for i in range(21)]
        category = SimpleNamespace(category_name="Right", score=.99)
        output = convert_result(SimpleNamespace(hand_landmarks=[landmarks], handedness=[[category]]), 640, 480)
        self.assertEqual(output[0]["keypoints"][0], [160.0, 240.0, 1.0])
        self.assertEqual(output[0]["label"], "Right")
        self.assertEqual(output[0]["keypoint_confidence_kind"], "not_reported_display_mask")
        self.assertEqual(output[0]["score_kind"], "handedness_classification")
        self.assertEqual(len(output[0]["keypoint_names"]), 21)
        self.assertEqual(len(output[0]["edges"]), 21)

    def test_empty_detections(self):
        self.assertEqual(convert_result(SimpleNamespace(hand_landmarks=[], handedness=[]), 640, 480), [])


if __name__ == "__main__":
    unittest.main()
