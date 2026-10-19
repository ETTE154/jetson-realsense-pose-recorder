"""Dataset event fidelity, timing, bounded memory and safe publication tests."""
import csv
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_csv import DatasetCSV


class DatasetCSVTests(unittest.TestCase):
    def setUp(self):
        # Workspace paths are writable on restricted Windows runners as well.
        self.directory = tempfile.TemporaryDirectory(prefix="dataset-test-", dir=Path(__file__).resolve().parents[1])
        self.now = 100.
        self.writer = None
        self.gate = None

    def tearDown(self):
        if self.gate:
            self.gate.set()
        if self.writer:
            self.writer.stop(wait=True)
        self.directory.cleanup()

    def begin(self, **kwargs):
        self.path = Path(self.directory.name) / "session" / "dataset.csv"
        self.writer = DatasetCSV(self.path, 105., clock=lambda: self.now, **kwargs)
        return self.writer

    def frame(self, stamp=105., number=7, generation=2):
        return SimpleNamespace(frame_id=number, received_monotonic=stamp, received_time=1700000000.+stamp,
                               timestamp_ms=42., timestamp_domain="hardware_clock", capture_generation=generation,
                               color_frame_number=128, depth_frame_number=129, depth_timestamp_ms=41.9,
                               depth_timestamp_domain="hardware_clock", depth_scale=.001,
                               depth=np.array([[0, 1000], [2000, 3000]], np.uint16),
                               rgb=np.zeros((480, 640, 3), np.uint8),
                               intrinsics=SimpleNamespace(width=2, height=2, fx=10., fy=11., ppx=1., ppy=1.,
                                                          model="brown_conrady", coeffs=[.1, .2, .3, .4, .5]))

    def imu(self, stamp=105., sample_id=1, kind="accel", generation=9):
        return {"sample_id": sample_id, "type": kind, "x": 0., "y": -1., "z": 9.81,
                "units": "m/s^2" if kind == "accel" else "rad/s", "frame_number": sample_id+1000,
                "timestamp_ms": sample_id*2.5, "timestamp_domain": "global_time", "received_time": 1700000000+stamp,
                "received_monotonic": stamp, "capture_generation": generation}

    def rows(self):
        with self.path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def until(self, predicate):
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.005)
        self.fail("Timed out waiting for dataset writer")

    def test_five_second_source_gate_late_pose_and_stop_are_not_wall_clock_based(self):
        writer = self.begin()
        self.now = 104.99
        self.assertFalse(writer.submit_imu(self.imu(104.99)))
        self.now = 106.
        self.assertFalse(writer.submit_camera(self.frame(104.999)))
        self.assertFalse(writer.submit_pose(self.frame(104.9), []))
        self.assertTrue(writer.submit_imu(self.imu(105.)))
        self.assertTrue(writer.submit_camera(self.frame(105.1)))
        self.now = 106.2
        stopped = writer.stop(106.2)
        self.assertEqual(stopped["state"], "completed")
        self.assertFalse(writer.submit_imu(self.imu(106.1, 99)))
        self.assertEqual(stopped["excluded_before_deadline"], 3)
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(105 <= float(row["source_received_monotonic_s"]) <= 106.2 for row in rows))
        self.assertEqual(rows[0]["source_timestamp_domain"], "global_time")
        self.assertEqual(rows[0]["source_received_unix_s"], "1700000105.0")
        self.assertEqual(rows[1]["source_timestamp_domain"], "hardware_clock")

    def test_all_async_imu_samples_preserve_clock_epochs_units_and_zero(self):
        writer = self.begin()
        self.now = 107.
        for index in range(180):
            kind = "accel" if index % 2 else "gyro"
            self.assertTrue(writer.submit_imu(self.imu(105+index/100., index, kind, index//90)))
        writer.stop()
        rows = self.rows()
        self.assertEqual(len(rows), 180)
        self.assertEqual([int(row["imu_sample_id"]) for row in rows], list(range(180)))
        self.assertEqual({row["source_generation"] for row in rows}, {"0", "1"})
        self.assertEqual(rows[0]["gyro_x_rad_s"], "0.0")
        self.assertEqual(rows[0]["accel_x_m_s2"], "")
        self.assertEqual(rows[1]["accel_y_m_s2"], "-1.0")
        self.assertEqual(rows[1]["gyro_y_rad_s"], "")
        self.assertTrue(all(row["source_frame_id"] == "" for row in rows))
        self.assertEqual(rows[9]["source_frame_number"], "1009")

    def test_camera_depth_stats_intrinsics_and_video_reference_do_not_invent_raw_pixels(self):
        writer = self.begin()
        self.now = 106.
        self.assertTrue(writer.submit_camera(self.frame(), {"pose_enabled": True, "confidence": .3,
                                                           "pose_roi": [1, 2, 3, 4]}))
        writer.stop()
        row, = self.rows()
        self.assertEqual(row["depth_valid_pixels"], "3")
        self.assertEqual(float(row["depth_valid_fraction"]), .75)
        self.assertEqual(float(row["depth_median_m"]), 2.)
        self.assertEqual(row["color_frame_number"], "128")
        self.assertEqual(row["depth_frame_number"], "129")
        self.assertEqual(row["depth_timestamp_ms"], "41.9")
        self.assertEqual(row["camera_fx_px"], "10.0")
        self.assertEqual(row["camera_distortion_coeff_4"], "0.5")
        self.assertEqual(row["rgb_video_file"], "rgb.mp4")
        self.assertEqual(row["video_mapping_file"], "frames.jsonl")
        self.assertEqual(row["video_mapping_source_frame_id"], "7")
        self.assertEqual(row["raw_depth_file"], "")
        self.assertEqual(row["setting_roi_bottom_px"], "4.0")

    def test_pose_frame_and_joints_preserve_null_confidence_and_invalid_depth(self):
        writer = self.begin()
        self.now = 106.
        person = {"label": "Left", "score": .9, "score_kind": "handedness_classification", "bbox": [1, 2, 3, 4],
                  "keypoint_confidence_kind": "not_reported_display_mask", "native_solver_weights": [.8, .7],
                  "keypoints": [{"name": "wrist", "u": 10., "v": 20., "confidence": 1., "displayed": True,
                                 "valid_depth": True, "x_m": .1, "y_m": -.2, "z_m": 1.2},
                                {"name": "tip", "u": 30., "v": 40., "confidence": 1., "displayed": False,
                                 "valid_depth": False, "x_m": 99., "y_m": 99., "z_m": 99.}]}
        self.assertTrue(writer.submit_pose(self.frame(), [person], {"model_id": "mediapipe", "state": "ready"}))
        person["keypoints"][0]["u"] = 999  # submit owns a copy, no stale/reused mutation.
        writer.stop()
        frame, wrist, tip = self.rows()
        self.assertEqual(frame["record_type"], "pose_frame")
        self.assertEqual(frame["pose_instance_count"], "1")
        self.assertEqual(wrist["record_type"], "pose_joint")
        self.assertEqual(wrist["joint_confidence"], "")
        self.assertEqual(wrist["joint_u_px"], "10.0")
        self.assertEqual(wrist["handedness_score"], "0.9")
        self.assertEqual(wrist["joint_depth_valid"], "1")
        self.assertEqual(wrist["joint_y_m"], "-0.2")
        self.assertEqual(tip["joint_depth_valid"], "0")
        self.assertEqual(tip["joint_x_m"], "")
        self.assertEqual(tip["joint_displayed"], "0")
        self.assertEqual(wrist["instance_id_scope"], "source_frame")

    def test_bounded_camera_memory_does_not_retain_rgb_and_imu_can_continue(self):
        self.gate = threading.Event()
        entered = threading.Event()
        original = DatasetCSV._camera_row
        def delayed(writer, payload):
            entered.set()
            self.gate.wait(3)
            return original(writer, payload)
        with patch.object(DatasetCSV, "_camera_row", delayed):
            writer = self.begin(max_camera_frames=2, max_depth_bytes=16)
            self.now = 106.
            self.assertTrue(writer.submit_camera(self.frame()))
            self.assertTrue(entered.wait(1))
            self.assertTrue(writer.submit_camera(self.frame(number=8)))
            self.assertFalse(writer.submit_camera(self.frame(number=9)))
            self.assertTrue(writer.submit_imu(self.imu()))
            status = writer.status()
            self.assertEqual(status["retained_depth_bytes"], 16)
            self.assertEqual(status["pending_camera_frames"], 2)
            self.assertEqual(status["dropped"]["camera"], 1)
            queued_camera = next(item for item in writer._queue.queue if item[1] == "camera")
            metadata, depth = queued_camera[3]
            self.assertNotIn("rgb", metadata)
            self.assertIsInstance(depth, np.ndarray)
            self.gate.set()
            self.assertEqual(writer.stop()["state"], "completed")
        self.assertEqual(writer.status()["retained_depth_bytes"], 0)

    def test_queue_overflow_is_nonblocking_and_error_never_publishes_final(self):
        self.gate = threading.Event()
        entered = threading.Event()
        def failure(writer, payload):
            entered.set()
            self.gate.wait(3)
            raise OSError("deliberate disk failure")
        with patch.object(DatasetCSV, "_camera_row", failure):
            writer = self.begin(queue_size=2)
            self.now = 106.
            self.assertTrue(writer.submit_camera(self.frame()))
            self.assertTrue(entered.wait(1))
            started = time.monotonic()
            self.assertTrue(writer.submit_imu(self.imu(sample_id=1)))
            self.assertTrue(writer.submit_imu(self.imu(sample_id=2)))
            self.assertFalse(writer.submit_imu(self.imu(sample_id=3)))
            self.assertLess(time.monotonic()-started, .25)
            self.assertFalse(self.path.exists())
            self.assertIsNone(writer.status()["file"])
            self.gate.set()
            self.until(lambda: writer.status()["state"] == "error")
            status = writer.stop()
        self.assertIn("disk failure", status["error"])
        self.assertFalse(self.path.exists())
        self.assertIsNone(status["file"])
        self.assertEqual(status["dropped"]["imu_accel"], 3)
        self.assertEqual(status["retained_depth_bytes"], 0)

    def test_cancel_and_duplicate_stop_and_spreadsheet_strings(self):
        writer = self.begin()
        self.now = 104.
        self.assertEqual(writer.stop()["state"], "cancelled")
        self.assertFalse(self.path.exists())
        self.assertFalse(writer.partial_path.exists())
        self.assertEqual(writer.stop()["state"], "cancelled")
        self.writer = None
        writer = self.begin(recording_id="=unsafe()")
        self.now = 106.
        frame = self.frame()
        frame.timestamp_domain = " \t@formula"
        self.assertTrue(writer.submit_pose(frame, [], {"model_id": "+model"}))
        writer.stop()
        row, = self.rows()
        self.assertEqual(row["recording_id"], "'=unsafe()")
        self.assertEqual(row["source_timestamp_domain"], "' \t@formula")
        self.assertEqual(row["pose_model_id"], "'+model")

    def test_invalid_future_nonfinite_and_missing_sample_data_are_rejected(self):
        writer = self.begin()
        self.now = 106.
        self.assertFalse(writer.submit_imu(self.imu(107.)))
        bad = self.imu()
        bad["x"] = float("nan")
        self.assertFalse(writer.submit_imu(bad))
        bad = self.imu()
        bad["type"] = "orientation"
        self.assertFalse(writer.submit_imu(bad))
        bad = self.imu()
        bad.pop("received_monotonic")
        self.assertFalse(writer.submit_imu(bad))
        writer.stop()
        self.assertEqual(self.rows(), [])
        self.assertEqual(writer.status()["rejected_invalid"], 4)

    def test_stop_cutoff_race_filters_rows_already_written_including_all_pose_joints(self):
        writer = self.begin()
        self.now = 107.
        self.assertTrue(writer.submit_imu(self.imu(105.5)))
        self.assertTrue(writer.submit_imu(self.imu(106.5, 2)))
        self.assertTrue(writer.submit_pose(self.frame(106.5), [{"keypoints": [{"u": 1, "v": 2}]}]))
        self.until(lambda: writer.status()["rows_total"] == 4)
        # Model/camera callbacks can run between caller capturing this cutoff
        # and its stop call reaching DatasetCSV. The published CSV must obey it.
        status = writer.stop(106.)
        self.assertEqual(status["state"], "completed")
        row, = self.rows()
        self.assertEqual(row["source_received_monotonic_s"], "105.5")
        self.assertEqual(status["rows_total"], 1)
        self.assertEqual(status["excluded_after_stop"], 2)
        self.assertEqual(status["rows_written"]["pose_joint"], 0)

    def test_unsupported_imu_units_cannot_be_mislabeled_as_si_units(self):
        writer = self.begin()
        self.now = 106.
        gyro = self.imu(kind="gyro")
        gyro["units"] = "deg/s"
        self.assertFalse(writer.submit_imu(gyro))
        accel = self.imu()
        accel["units"] = "g"
        self.assertFalse(writer.submit_imu(accel))
        writer.stop()
        self.assertEqual(self.rows(), [])
        self.assertEqual(writer.status()["rejected_invalid"], 2)


if __name__ == "__main__":
    unittest.main()
