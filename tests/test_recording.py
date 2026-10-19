"""Recording clock, backpressure, lifecycle and publication invariants."""
import csv
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np

from recording import Recorder, STREAMS, OUTPUT_STREAMS, output_size


class Clock:
    def __init__(self):
        self.now, self.wall = 100., 1700000000.
    def monotonic(self): return self.now
    def time(self): return self.wall


class Writer:
    def __init__(self, fail=False, on_write=None):
        self.images, self.released, self.fail = [], 0, fail
        self.on_write = on_write
    def isOpened(self): return True
    def write(self, image):
        if self.fail: raise RuntimeError("deliberate encoder failure")
        self.images.append(image.copy())
        if self.on_write:
            self.on_write(image)
    def release(self): self.released += 1


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="recording-clock-test-", dir=Path(__file__).resolve().parents[1])
        self.clock = Clock()
        self.writers = {}
        self.factory_entered, self.factory_gate = threading.Event(), threading.Event()
        self.factory_gate.set()
        self.fail_writer = False
        self.on_rgb_write = None
        def factory(path, fps, size):
            self.factory_entered.set()
            if not self.factory_gate.wait(3): raise RuntimeError("test writer factory gate timed out")
            hook = (lambda image: self.on_rgb_write(image) if self.on_rgb_write else None) if Path(path).name == "rgb.partial.mp4" else None
            writer = Writer(self.fail_writer, hook)
            writer.size = size
            self.writers[Path(path).name.split(".")[0]] = writer
            return writer
        self.recorder = Recorder(self.directory.name, fps=10, countdown_seconds=5,
                                 write_delay_s=1, queue_size=8, clock=self.clock.monotonic,
                                 wall_clock=self.clock.time, writer_factory=factory, min_free_bytes=0)
    def tearDown(self):
        self.factory_gate.set()
        self.recorder.close()
        self.directory.cleanup()
    def advance(self, value):
        self.clock.now = value
        if self.recorder._session: self.recorder._session.wake.set()
    def until(self, predicate):
        deadline = time.monotonic()+4
        while time.monotonic() < deadline:
            if predicate(): return
            time.sleep(.005)
        self.fail("Timed out waiting for recorder test condition")
    def frame(self, number, stamp):
        return SimpleNamespace(frame_id=number, received_monotonic=stamp, received_time=1700000000+stamp-100,
                               timestamp_ms=number*33., timestamp_domain="test", depth_scale=.001)
    def sensor(self, number, stamp, color, settings=None):
        images = {name: np.full((16, 16, 3), color, np.uint8) for name in STREAMS[:3]}
        return self.recorder.submit_sensor(self.frame(number, stamp), images, settings or {"pose_model": "yolo"})
    def begin(self):
        return self.recorder.start({"width": 16, "height": 16}, {"pose_model": "yolo"})
    def rows(self, identifier):
        path = Path(self.directory.name)/identifier/"frames.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]
    def data_frame(self, number, stamp, generation=2):
        frame = self.frame(number, stamp)
        frame.depth = np.full((16, 16), 1500, np.uint16)
        frame.depth.setflags(write=False)
        frame.capture_generation = generation
        frame.color_frame_number = number + 1000
        frame.depth_frame_number = number + 2000
        frame.depth_timestamp_ms = number*33.-.2
        frame.depth_timestamp_domain = "depth_hardware_clock"
        frame.intrinsics = SimpleNamespace(fx=10., fy=10., ppx=8., ppy=8., model="none", coeffs=[0.]*5)
        return frame
    def imu_sample(self, number, stamp, kind="gyro", generation=2):
        return {"sample_id": number, "type": kind, "x": 0., "y": -.1, "z": 9.8,
                "units": "rad/s" if kind == "gyro" else "m/s²", "timestamp_ms": number*2.5,
                "timestamp_domain": "imu_hardware_clock", "frame_number": number+9000,
                "received_time": 1700000000+stamp-100, "received_monotonic": stamp,
                "capture_generation": generation}
    def dataset_rows(self, identifier):
        with (Path(self.directory.name)/identifier/"dataset.csv").open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_predeadline_source_is_rejected_even_when_result_arrives_late(self):
        started = self.begin()
        self.advance(104.9)
        self.assertFalse(self.sensor(1, 104.9, 10))
        self.clock.wall += 3600  # Wall-clock jumps do not move the capture gate.
        self.advance(106.)
        self.assertFalse(self.sensor(2, 104.999, 20))
        self.assertTrue(self.sensor(3, 105., 30))
        self.recorder.stop(wait=True)
        rows = self.rows(started["id"])
        sources = [row["sensor_source"] for row in rows if row["sensor_source"]]
        self.assertTrue(sources)
        self.assertTrue(all(source["received_monotonic"] >= 105 for source in sources))
        self.assertEqual({source["frame_id"] for source in sources}, {3})
        self.assertGreaterEqual(self.recorder.status()["excluded_before_deadline"], 2)

    def test_sparse_sources_keep_two_second_wall_time_and_finalize_tail(self):
        self.factory_gate.clear()
        started = self.begin()
        self.advance(105.)
        self.assertTrue(self.factory_entered.wait(1))
        self.assertTrue(self.sensor(1, 105., 10))
        self.advance(105.7)
        self.assertTrue(self.sensor(2, 105.7, 80))
        self.advance(106.4)
        self.assertTrue(self.sensor(3, 106.4, 160))
        self.advance(107.)
        self.recorder.stop()
        self.factory_gate.set()
        self.until(lambda: self.recorder.status()["state"] == "completed")
        self.assertEqual(set(self.writers), set(OUTPUT_STREAMS))
        self.assertTrue(all(len(writer.images) == 20 for writer in self.writers.values()))
        self.assertTrue(all(writer.released == 1 for writer in self.writers.values()))
        colors = [int(image[0, 0, 0]) for image in self.writers["rgb"].images]
        self.assertEqual(colors, [10]*7+[80]*7+[160]*6)
        rows = self.rows(started["id"])
        self.assertEqual([row["frame_index"] for row in rows], list(range(20)))
        self.assertEqual(rows[-1]["t_s"], 1.9)
        self.assertEqual(self.recorder.status()["recorded_duration_s"], 2.)
        self.assertTrue((Path(self.directory.name)/started["id"]/"manifest.json").is_file())

    def test_catchup_revisits_new_inputs_before_rendering_the_whole_backlog(self):
        accepted = []
        def inject(image):
            if not accepted:
                # Arrival during a slow encoder's first catch-up pass.
                accepted.append(self.sensor(2, 105.45, 180))
        self.on_rgb_write = inject
        self.factory_gate.clear()
        started = self.begin()
        self.advance(105.)
        self.assertTrue(self.factory_entered.wait(1))
        self.assertTrue(self.sensor(1, 105., 10))
        self.advance(107.)
        self.factory_gate.set()
        self.until(lambda: self.recorder.status()["frames_written"]["rgb"] >= 11)
        self.recorder.stop(wait=True)
        self.assertEqual(accepted, [True])
        row = next(row for row in self.rows(started["id"]) if row["t_s"] == .5)
        self.assertEqual(row["sensor_source"]["frame_id"], 2,
                         "New arrivals must be drained between catch-up batches")
        self.assertTrue(np.all(self.writers["rgb"].images[5] == 180))

    def test_combined_views_share_the_pose_source_even_with_newer_sensor_images(self):
        self.factory_gate.clear()
        started = self.begin()
        self.advance(105.)
        self.assertTrue(self.factory_entered.wait(1))
        paired = {name: np.full((16, 16, 3), value, np.uint8)
                  for name, value in zip(STREAMS, (10, 20, 30, 40, 50))}
        self.assertTrue(self.recorder.submit_pose(self.frame(1, 105.), paired, [], {}, {"pose_model": "yolo"}))
        for image in paired.values():
            image.fill(250)  # The recorder must own its matching source images.
        self.advance(105.3)
        self.assertTrue(self.sensor(2, 105.3, 180))
        self.advance(106.)
        self.recorder.stop()
        self.factory_gate.set()
        self.until(lambda: self.recorder.status()["state"] == "completed")
        image = self.writers["combined"].images[-1]
        self.assertEqual(self.writers["combined"].size, output_size("combined", 16, 16))
        self.assertEqual(image.shape, (32, 32, 3))
        for y, x, value in ((8, 8, 10), (8, 24, 20), (24, 8, 30), (24, 24, 40)):
            self.assertTrue(np.all(image[y, x] == value))
        self.assertEqual(int(self.writers["rgb"].images[-1][0, 0, 0]), 180)
        rows = self.rows(started["id"])
        self.assertTrue(all(row["combined_source"]["frame_id"] == 1 for row in rows))
        self.assertTrue(all(row["combined_pose_gap"] is None for row in rows))
        status = self.recorder.status()
        self.assertEqual(status["output_dimensions"]["combined"], {"width": 32, "height": 32})
        self.assertTrue(status["files"]["combined"].endswith("/combined.mp4"))

    def test_combined_model_change_continues_sensor_views_with_a_blank_pose(self):
        self.factory_gate.clear()
        started = self.begin()
        self.advance(105.)
        self.assertTrue(self.factory_entered.wait(1))
        paired = {name: np.full((16, 16, 3), 70, np.uint8) for name in STREAMS}
        self.recorder.submit_pose(self.frame(1, 105.), paired, [], {}, {"pose_model": "yolo"})
        self.advance(105.5)
        self.recorder.invalidate_pose(settings={"pose_model": "mediapipe"})
        self.sensor(2, 105.5, 180, {"pose_model": "mediapipe"})
        self.advance(106.)
        self.recorder.stop()
        self.factory_gate.set()
        self.until(lambda: self.recorder.status()["state"] == "completed")
        image = self.writers["combined"].images[-1]
        for y, x in ((8, 8), (8, 24), (24, 8)):
            self.assertTrue(np.all(image[y, x] == 180))
        self.assertTrue(np.all(image[16:32, 16:] == 0))
        last = self.rows(started["id"])[-1]
        self.assertEqual(last["combined_source"]["frame_id"], 2)
        self.assertEqual(last["combined_pose_model"], "mediapipe")
        self.assertIsNotNone(last["combined_pose_gap"])

    def test_raw_imu_csv_is_retained_without_graph_movie_work(self):
        self.factory_gate.clear()
        started = self.begin()
        self.advance(105.5)
        self.assertTrue(self.factory_entered.wait(1))
        paired = {name: np.full((16, 16, 3), 70, np.uint8) for name in STREAMS}
        self.assertTrue(self.recorder.submit_imu(self.imu_sample(1, 105.01, "accel")))
        self.assertTrue(self.recorder.submit_imu(self.imu_sample(2, 105.015, "gyro")))
        self.assertTrue(self.recorder.submit_imu(self.imu_sample(3, 105.2, "accel")))
        self.assertTrue(self.recorder.submit_pose(self.data_frame(1, 105.02), paired, [], {}, {"pose_model": "yolo"}))
        self.assertTrue(self.recorder.submit_sensor(self.data_frame(2, 105.3),
                        {name: paired[name] for name in STREAMS[:3]}, {"pose_model": "yolo"}))
        self.recorder.stop()
        self.factory_gate.set()
        self.until(lambda: self.recorder.status()["state"] == "completed")
        rows = self.rows(started["id"])
        later = next(row for row in rows if row["t_s"] == .3)
        self.assertEqual(later["combined_source"]["frame_id"], 1)
        self.assertNotIn("imu_video", later)
        self.assertNotIn("combined_imu", later)
        self.assertNotIn("imu", self.writers)
        self.assertNotIn("imu", self.recorder.status()["files"])
        self.assertFalse(self.recorder.status()["imu_visualization"]["enabled"])
        imu_rows = [row for row in self.dataset_rows(started["id"]) if row["record_type"].startswith("imu_")]
        self.assertEqual([int(row["imu_sample_id"]) for row in imu_rows], [1, 2, 3])
        self.assertEqual(self.writers["combined"].size, (32, 32))

    def test_cancel_duplicate_start_and_repeated_stop_are_session_safe(self):
        first = self.begin()
        with self.assertRaises(ValueError): self.begin()
        self.advance(102.)
        cancelled = self.recorder.stop(wait=True)
        self.assertEqual(cancelled["state"], "cancelled")
        self.assertFalse(self.writers)
        self.assertEqual(self.recorder.stop()["id"], first["id"])
        self.assertFalse(list((Path(self.directory.name)/first["id"]).glob("*.mp4")))
        second = self.begin()
        self.assertNotEqual(first["id"], second["id"])
        self.recorder.stop(wait=True)
        for identifier in (first["id"], second["id"]):
            manifest = json.loads((Path(self.directory.name)/identifier/"manifest.json").read_text())
            self.assertEqual(manifest["id"], identifier)

    def test_stop_during_frame_copy_rejects_the_late_enqueue(self):
        self.factory_gate.clear()
        self.begin()
        self.advance(105.)
        self.assertTrue(self.factory_entered.wait(1))
        copying, release = threading.Event(), threading.Event()
        class BlockingImage(np.ndarray):
            def copy(image):
                copying.set()
                if not release.wait(2): raise RuntimeError("copy test gate timed out")
                return np.asarray(image).copy()
        array = np.zeros((16, 16, 3), np.uint8).view(BlockingImage)
        accepted = []
        thread = threading.Thread(target=lambda: accepted.append(self.recorder.submit_sensor(
            self.frame(1, 105.), {name: array for name in STREAMS[:3]}, {})))
        thread.start()
        self.assertTrue(copying.wait(1))
        self.advance(105.2)
        self.recorder.stop()
        release.set()
        thread.join(2)
        self.factory_gate.set()
        self.assertEqual(accepted, [False])
        self.until(lambda: self.recorder.status()["state"] == "completed")
        self.assertEqual(self.recorder.status()["accepted_sensor_frames"], 0)

    def test_encoder_error_releases_all_writers_and_rejects_further_input(self):
        self.fail_writer = True
        self.begin()
        self.advance(106.)
        self.until(lambda: self.recorder.status()["state"] == "error")
        self.assertTrue(all(writer.released == 1 for writer in self.writers.values()))
        self.assertFalse(self.sensor(1, 106., 20))
        self.assertIn("deliberate encoder failure", self.recorder.status()["error"])

    def test_queue_overflow_is_bounded_and_model_change_clears_old_pose(self):
        self.factory_gate.clear()
        started = self.begin()
        self.advance(105.)
        self.assertTrue(self.factory_entered.wait(1))
        pose_images = {name: np.full((16, 16, 3), 200, np.uint8) for name in STREAMS[3:]}
        # Ensure the control event evicts a sensor packet, preserving the old
        # Pose long enough to prove that invalidation actually clears it.
        self.assertTrue(self.sensor(0, 105., 40))
        self.assertTrue(self.recorder.submit_pose(self.frame(1, 105.), pose_images, [], {}, {"pose_model": "yolo"}))
        for index in range(20): self.sensor(index+2, 105., 40)
        self.assertLessEqual(self.recorder.status()["queue_depth"], 8)
        self.assertGreater(self.recorder.status()["dropped_total"], 0)
        self.advance(105.5)
        self.recorder.invalidate_pose(settings={"pose_model": "mediapipe"})
        self.advance(106.)
        self.recorder.stop()
        self.factory_gate.set()
        self.until(lambda: self.recorder.status()["state"] == "completed")
        rows = self.rows(started["id"])
        self.assertTrue(any(row["pose_source"] is not None for row in rows if row["t_s"] < .5))
        self.assertTrue(all(row["pose_source"] is None for row in rows if row["t_s"] >= .5))
        self.assertEqual(len(self.writers["rgb"].images), 10)

    def test_coalesced_controls_do_not_revive_an_intermediate_model_pose(self):
        self.recorder.queue_size = 2
        self.factory_gate.clear()
        started = self.begin()
        self.advance(105.)
        self.assertTrue(self.factory_entered.wait(1))
        self.recorder.invalidate_pose(settings={"pose_model": "rtmpose"})
        self.advance(105.2)
        images = {name: np.full((16, 16, 3), 200, np.uint8) for name in STREAMS[3:]}
        self.assertTrue(self.recorder.submit_pose(self.frame(1, 105.2), images, [], {}, {"pose_model": "rtmpose"}))
        self.advance(105.4)
        self.recorder.invalidate_pose(settings={"pose_model": "mediapipe"})
        self.advance(106.)
        self.recorder.stop()
        self.factory_gate.set()
        self.until(lambda: self.recorder.status()["state"] == "completed")
        rows = self.rows(started["id"])
        self.assertTrue(all(row["pose_source"] is None for row in rows if row["t_s"] >= .4),
                        "Intermediate RTMPose must not be held after the MediaPipe change")

    def test_dataset_native_camera_imu_and_pose_survive_video_backpressure(self):
        self.factory_gate.clear()
        started = self.recorder.start({"width": 16, "height": 16, "device": {"serial": "camera-test"},
                                       "imu": {"profiles": [{"type": "gyro", "fps": 200},
                                                            {"type": "accel", "fps": 100}]}},
                                      {"pose_model": "mediapipe"})
        self.advance(106.)
        self.assertTrue(self.factory_entered.wait(1))
        for index in range(30):
            self.assertTrue(self.recorder.submit_camera_data(self.data_frame(index, 105.+index/30.)))
        for index in range(200):
            self.assertTrue(self.recorder.submit_imu(self.imu_sample(index, 105.+index/200.)))
        for index in range(100):
            self.assertTrue(self.recorder.submit_imu(self.imu_sample(index+200, 105.+index/100., "accel")))
        # Fill the independent eight-packet video queue; no extra camera CSV
        # rows should be emitted by preview/encoder submissions.
        for index in range(20):
            self.sensor(index+100, 105.+index/25., 40)
        pose_images = {name: np.full((16, 16, 3), 100, np.uint8) for name in STREAMS[3:]}
        for index in range(3):
            person = {"label": "Left", "score": .9, "score_kind": "handedness_classification",
                      "keypoint_confidence_kind": "not_reported_display_mask",
                      "keypoints": [{"name": "wrist", "u": 5., "v": 6., "confidence": None,
                                     "valid_depth": True, "x_m": .1, "y_m": .2, "z_m": 1.5}]}
            self.assertFalse(self.recorder.submit_pose(self.data_frame(300+index, 105.25+index*.2),
                pose_images, [person], {"state": "ready", "model_id": "mediapipe"}, {"pose_model": "mediapipe"}))
        self.assertGreater(self.recorder.status()["dropped_total"], 0)
        self.advance(106.1)
        self.recorder.stop()
        self.factory_gate.set()
        self.until(lambda: self.recorder.status()["state"] == "completed")
        status = self.recorder.status()
        self.assertEqual(status["dataset"]["rows_written"], {"camera": 30, "imu_accel": 100, "imu_gyro": 200,
                                                             "pose_frame": 3, "pose_joint": 3})
        self.assertEqual(status["dataset"]["dropped_total"], 0)
        self.assertEqual(status["dataset"]["state"], "completed")
        self.assertTrue(status["files"]["dataset_csv"].endswith("/dataset.csv"))
        rows = self.dataset_rows(started["id"])
        self.assertEqual(len(rows), 336)
        self.assertEqual({int(row["source_frame_id"]) for row in rows if row["record_type"] == "camera"}, set(range(30)))
        gyro = next(row for row in rows if row["record_type"] == "imu_gyro")
        self.assertEqual(gyro["device_serial"], "camera-test")
        self.assertEqual(gyro["imu_profile_fps"], "200")
        self.assertEqual(gyro["source_timestamp_domain"], "imu_hardware_clock")

    def test_dataset_countdown_cancel_has_no_csv_or_partial_download(self):
        started = self.begin()
        self.until(lambda: self.recorder._session.dataset is not None)
        self.advance(104.)
        self.assertFalse(self.recorder.submit_camera_data(self.data_frame(1, 104.)))
        self.assertFalse(self.recorder.submit_imu(self.imu_sample(1, 104.)))
        stopped = self.recorder.stop(wait=True)
        self.assertEqual(stopped["state"], "cancelled")
        self.assertEqual(stopped["dataset"]["state"], "cancelled")
        self.assertNotIn("dataset_csv", stopped["files"])
        self.assertFalse(list((Path(self.directory.name)/started["id"]).glob("dataset*.csv")))
        self.assertEqual(stopped["dataset"]["rows_total"], 0)

    def test_encoder_failure_finalizes_sensor_csv_at_same_stop_cutoff(self):
        self.factory_gate.clear()
        self.fail_writer = True
        started = self.begin()
        self.advance(106.)
        self.assertTrue(self.factory_entered.wait(1))
        self.assertTrue(self.recorder.submit_camera_data(self.data_frame(1, 105.1)))
        self.assertTrue(self.recorder.submit_imu(self.imu_sample(1, 105.2)))
        self.factory_gate.set()
        self.until(lambda: self.recorder.status()["state"] == "error")
        status = self.recorder.status()
        self.assertIn("encoder failure", status["error"])
        self.assertEqual(status["dataset"]["state"], "completed")
        self.assertEqual(status["dataset"]["stop_monotonic"], status["stop_monotonic"])
        self.assertEqual(status["dataset"]["rows_total"], 2)
        self.assertIn("dataset_csv", status["files"])
        self.assertFalse(self.recorder.submit_imu(self.imu_sample(2, 105.5)))
        manifest = json.loads((Path(self.directory.name)/started["id"]/"manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["dataset"]["state"], "completed")
        self.assertEqual(manifest["dataset"]["rows_total"], 2)
        self.assertEqual(len(self.dataset_rows(started["id"])), 2)

    def test_dataset_stop_publishes_exact_cutoff_and_sessions_do_not_leak(self):
        self.factory_gate.clear()
        first = self.begin()
        self.advance(106.)
        self.assertTrue(self.factory_entered.wait(1))
        self.assertFalse(self.recorder.submit_camera_data(self.data_frame(0, 104.999)))
        self.assertFalse(self.recorder.submit_imu(self.imu_sample(0, 104.999)))
        self.assertTrue(self.recorder.submit_camera_data(self.data_frame(1, 105.)))
        self.assertTrue(self.recorder.submit_imu(self.imu_sample(1, 105.)))
        self.advance(106.25)
        stopped = self.recorder.stop()
        self.assertNotIn("dataset_csv", stopped["files"], "No CSV link before terminal manifest publication")
        self.assertFalse(self.recorder.submit_camera_data(self.data_frame(2, 106.2)))
        self.assertFalse(self.recorder.submit_imu(self.imu_sample(2, 106.2)))
        self.factory_gate.set()
        self.until(lambda: self.recorder.status()["state"] == "completed")
        status = self.recorder.status()
        self.assertEqual(status["stop_monotonic"], 106.25)
        self.assertEqual(status["dataset"]["stop_monotonic"], 106.25)
        self.assertEqual({row["source_received_monotonic_s"] for row in self.dataset_rows(first["id"])}, {"105.0"})
        self.until(lambda: not self.recorder._session.worker.is_alive())
        second = self.begin()  # Its deadline is now 111.25, independent of the old session.
        self.until(lambda: self.recorder._session.dataset is not None)
        self.advance(112.)
        self.assertFalse(self.recorder.submit_imu(self.imu_sample(99, 106.2)))
        self.assertTrue(self.recorder.submit_imu(self.imu_sample(100, 111.5, generation=3)))
        second_status = self.recorder.stop(wait=True)
        self.assertEqual(second_status["state"], "completed")
        row, = self.dataset_rows(second["id"])
        self.assertEqual(row["imu_sample_id"], "100")
        self.assertEqual(row["source_generation"], "3")
        self.assertEqual(row["recording_id"], second["id"])

    def test_pose_csv_commit_and_invalidation_serialize_without_blocking_native_sensors(self):
        self.factory_gate.clear()
        started = self.begin()
        self.advance(106.)
        self.assertTrue(self.factory_entered.wait(1))
        session = self.recorder._session
        dataset = session.dataset
        original_submit = dataset.submit_pose
        entered, release_pose = threading.Event(), threading.Event()
        invalidate_started, invalidate_done, sensors_done = threading.Event(), threading.Event(), threading.Event()
        results, errors, order = [], [], []
        images = {name: np.zeros((16, 16, 3), np.uint8) for name in STREAMS[3:]}

        def delayed_submit(*args, **kwargs):
            order.append("pose_copy_started")
            entered.set()
            if not release_pose.wait(3):
                raise RuntimeError("Pose serialization test timed out")
            result = original_submit(*args, **kwargs)
            order.append("pose_committed")
            return result

        def pose_worker():
            try:
                self.recorder.submit_pose(self.data_frame(1, 105.5), images, [],
                                          {"state": "ready", "model_id": "yolo"}, {"pose_model": "yolo"})
            except Exception as error:
                errors.append(error)

        def invalidate_worker():
            try:
                invalidate_started.set()
                results.append(self.recorder.invalidate_pose(settings={"pose_model": "rtmpose"}))
                order.append("pose_invalidated")
            except Exception as error:
                errors.append(error)
            finally:
                invalidate_done.set()

        def sensor_worker():
            try:
                # These submissions run while Pose copying is deliberately
                # blocked and invalidation is waiting for its separate lock.
                for index in range(50):
                    results.append(self.recorder.submit_imu(self.imu_sample(index, 105.6+index*.002)))
                for index in range(5):
                    results.append(self.recorder.submit_camera_data(self.data_frame(100+index, 105.6+index*.02)))
            except Exception as error:
                errors.append(error)
            finally:
                sensors_done.set()

        dataset.submit_pose = delayed_submit
        pose_thread = threading.Thread(target=pose_worker, daemon=True)
        invalidation_thread = threading.Thread(target=invalidate_worker, daemon=True)
        sensor_thread = threading.Thread(target=sensor_worker, daemon=True)
        try:
            pose_thread.start()
            self.assertTrue(entered.wait(1))
            self.advance(106.2)
            invalidation_thread.start()
            self.assertTrue(invalidate_started.wait(1))
            self.assertFalse(invalidate_done.wait(.05), "Invalidation interleaved a still-uncommitted Pose copy")
            sensor_thread.start()
            self.assertTrue(sensors_done.wait(.5), "Pose serialization held the native sensor callback lock")
            self.assertFalse(invalidate_done.is_set())
            self.assertTrue(all(results))
        finally:
            release_pose.set()
            for thread in (pose_thread, invalidation_thread, sensor_thread):
                if thread.ident is not None:
                    thread.join(2)
            dataset.submit_pose = original_submit
            self.factory_gate.set()
        self.assertFalse(pose_thread.is_alive(), "Pose commit deadlocked")
        self.assertFalse(invalidation_thread.is_alive(), "Pose invalidation deadlocked")
        self.assertFalse(sensor_thread.is_alive(), "Native sensor callback deadlocked")
        self.assertEqual(errors, [])
        self.assertTrue(all(results))
        self.assertLess(order.index("pose_committed"), order.index("pose_invalidated"))
        # After the barrier an old source cannot produce another CSV row.
        self.assertFalse(self.recorder.submit_pose(self.data_frame(2, 105.7), images, [],
                         {"state": "ready", "model_id": "yolo"}, {"pose_model": "yolo"}))
        self.advance(106.4)
        self.assertTrue(self.recorder.submit_pose(self.data_frame(3, 106.3), images, [],
                        {"state": "ready", "model_id": "rtmpose"}, {"pose_model": "rtmpose"}))
        self.recorder.stop(wait=True)
        status = self.recorder.status()
        self.assertEqual(status["state"], "completed")
        self.assertEqual(status["dataset"]["rows_written"]["imu_gyro"], 50)
        self.assertEqual(status["dataset"]["rows_written"]["camera"], 5)
        pose_rows = [row for row in self.dataset_rows(started["id"]) if row["record_type"] == "pose_frame"]
        self.assertEqual([(row["source_frame_id"], row["pose_model_id"]) for row in pose_rows],
                         [("1", "yolo"), ("3", "rtmpose")])


if __name__ == "__main__":
    unittest.main()
