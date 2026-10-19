"""Coordinate and score checks that catch RGB/affine/depth sampling mistakes."""
import unittest
import numpy as np

from comparison.rtmpose_adapter import prepare_image, decode_simcc


class RTMPoseGeometryTest(unittest.TestCase):
    def test_center_maps_back_to_roi_center(self):
        image = np.zeros((480, 640, 3), np.uint8)
        tensor, transform = prepare_image(image, [100, 60, 300, 260])
        self.assertEqual(tensor.shape, (1, 3, 256, 192))
        x = np.zeros((1, 17, 384), np.float32)
        y = np.zeros((1, 17, 512), np.float32)
        x[:, :, 192] = .8
        y[:, :, 256] = .6
        points = np.asarray(decode_simcc(x, y, transform, image.shape))
        np.testing.assert_allclose(points[:, :2], np.tile([200, 160], (17, 1)), atol=1e-5)
        np.testing.assert_allclose(points[:, 2], .6)

    def test_rgb_normalization_matches_official_pipeline(self):
        image = np.full((256, 192, 3), [10, 20, 30], np.uint8)
        tensor, _ = prepare_image(image)
        expected = (np.asarray([30, 20, 10]) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
        np.testing.assert_allclose(tensor[0, :, 128, 96], expected, rtol=1e-5)

    def test_outside_prediction_does_not_acquire_edge_depth(self):
        image = np.zeros((480, 640, 3), np.uint8)
        _, transform = prepare_image(image)
        x = np.zeros((1, 17, 384), np.float32)
        y = np.zeros((1, 17, 512), np.float32)
        x[:, :, 0] = .8
        y[:, :, 0] = .8
        points = np.asarray(decode_simcc(x, y, transform, image.shape))
        self.assertTrue((points[:, 0] < 0).all())
        self.assertTrue((points[:, 2] == 0).all())

    def test_invalid_roi_rejected(self):
        for roi in ([1, 2, 1, 4], [0, 0, float("nan"), 10], [1000, 0, 1100, 20]):
            with self.assertRaises(ValueError):
                prepare_image(np.zeros((480, 640, 3), np.uint8), roi)


if __name__ == "__main__":
    unittest.main()
