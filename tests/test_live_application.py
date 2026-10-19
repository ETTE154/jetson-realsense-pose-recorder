"""Live switching must preserve sensor cadence and pose/source correspondence."""
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from camera import Frame
from prototype import Application


class FakeCamera:
    width, height = 64, 48
    def __init__(self):
        self.condition = threading.Condition()
        self.active = False
        self.frame = None
        self.counter = 0
        self.thread = None
        self.frame_listener = None
    def set_frame_listener(self, listener):
        self.frame_listener = listener
    def set_imu_listener(self, listener):
        self.imu_listener = listener
    def start(self):
        self.active = True
        self.thread = threading.Thread(target=self.produce, daemon=True)
        self.thread.start()
        return self.snapshot()
    def produce(self):
        while self.active:
            with self.condition:
                self.counter += 1
                intrinsics = SimpleNamespace(width=64, height=48, fx=50., fy=50., ppx=32., ppy=24.,
                                             model='none', coeffs=[0.]*5)
                self.frame = Frame(self.counter, np.full((48, 64, 3), 30+self.counter%150, np.uint8),
                                   np.full((48, 64), 1000+self.counter, np.uint16), .001, intrinsics,
                                   self.counter*20., 'test_clock', time.time(), time.monotonic())
                self.condition.notify_all()
            if self.frame_listener:
                self.frame_listener(self.frame)
            time.sleep(.02)
    def stop(self):
        self.active = False
        with self.condition: self.condition.notify_all()
        if self.thread: self.thread.join(1)
        return self.snapshot()
    def snapshot(self):
        return {'state': 'streaming' if self.active else 'disconnected', 'active': self.active,
                'frame_id': self.counter, 'fps': 50., 'width': 64, 'height': 48,
                'device': {'serial': 'synthetic-test-camera'}}
    def wait_frame(self, after_id, timeout=.5):
        with self.condition:
            self.condition.wait_for(lambda: not self.active or (self.frame and self.frame.frame_id > after_id), timeout)
            return self.frame if self.active and self.frame and self.frame.frame_id > after_id else None


class FakeEstimator:
    blocked = set()
    fail_constructor = set()
    made = []
    def __init__(self, identifier, **kwargs):
        if identifier in self.fail_constructor: raise RuntimeError('deliberate unavailable runtime')
        self.identifier = identifier
        self.closed = threading.Event()
        self.entered = threading.Event()
        self.gate = threading.Event()
        if identifier not in self.blocked: self.gate.set()
        self.made.append(self)
    def infer(self, bgr, **kwargs):
        self.entered.set()
        self.gate.wait(3)
        if self.closed.is_set(): raise RuntimeError('closed test worker')
        return {'model': self.identifier, 'backend': 'synthetic test runtime',
                'instances': [{'keypoints': [[25., 20., 1.]], 'keypoint_names': ['wrist'], 'edges': [],
                               'bbox': [10., 10., 35., 30.], 'score': 1.,
                               'keypoint_confidence_kind': 'not_reported_display_mask'}]}
    def close(self):
        self.closed.set()
        self.gate.set()


class LiveApplicationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='live-app-test-')
        FakeEstimator.blocked, FakeEstimator.fail_constructor, FakeEstimator.made = set(), set(), []
        self.camera_patch = patch('prototype.Camera', FakeCamera)
        self.runtime_patch = patch('live_runtime.LiveEstimator', FakeEstimator)
        self.log_patch = patch('prototype.LOG.exception')
        self.camera_patch.start()
        self.runtime_patch.start()
        self.log_patch.start()
        self.app = Application('test.engine', Path(self.directory.name), preview_fps=15)
    def tearDown(self):
        self.app.close()
        self.log_patch.stop()
        self.runtime_patch.stop()
        self.camera_patch.stop()
        self.directory.cleanup()
    def until(self, predicate, message='condition'):
        deadline = time.monotonic()+3
        while time.monotonic() < deadline:
            if predicate(): return
            time.sleep(.01)
        self.fail('Timed out waiting for '+message)
    def test_rgbd_keeps_advancing_and_switch_interrupts_loading(self):
        FakeEstimator.blocked.add('egoforce')
        self.app.configure({'pose_model': 'egoforce'})
        self.app.camera.start()
        self.until(lambda: FakeEstimator.made and FakeEstimator.made[-1].entered.is_set())
        blocked = FakeEstimator.made[-1]
        self.until(lambda: 'rgb' in self.app._images)
        sequence, before = self.app.wait_image('rgb', -1)
        next_sequence, after = self.app.wait_image('rgb', sequence)
        self.assertGreater(next_sequence, sequence)
        self.assertNotEqual(before, after)
        self.assertEqual(self.app.status()['pose']['state'], 'loading')
        self.app.configure({'pose_model': 'mediapipe'})
        self.until(lambda: self.app.status()['pose']['state'] == 'ready')
        self.assertTrue(blocked.closed.is_set())
        self.assertEqual(self.app.status()['pose']['model_id'], 'mediapipe')
        self.assertTrue(self.app._pose_worker.is_alive())
    def test_snapshot_uses_pose_source_not_newer_rgb(self):
        self.app.camera.start()
        self.until(lambda: self.app.status()['pose']['state'] == 'ready')
        estimator = FakeEstimator.made[-1]
        estimator.gate.clear()
        self.until(lambda: self.app.camera.counter > self.app.status()['pose']['frame_id']+5)
        saved = self.app.snapshot()
        directory = Path(self.directory.name)/saved['id']
        meta = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        rgb = cv2.imdecode(np.frombuffer((directory/'rgb.jpg').read_bytes(), np.uint8), 1)
        depth = cv2.imdecode(np.frombuffer((directory/'depth16.png').read_bytes(), np.uint8), -1)
        self.assertEqual(meta['frame_id'], meta['pose']['frame_id'])
        self.assertTrue(np.all(depth == 1000+meta['frame_id']))
        self.assertAlmostEqual(float(rgb.mean()), 30+meta['frame_id']%150, delta=1.)
        pose_sequence, _ = self.app.wait_image('pose', -1)
        rgb_sequence, _ = self.app.wait_image('rgb', -1)
        self.app.wait_image('rgb', rgb_sequence)
        unchanged, _ = self.app.wait_image('pose', pose_sequence, timeout=.03)
        self.assertEqual(unchanged, pose_sequence, 'RGB updates must not emit duplicate Pose frames')
    def test_unavailable_model_can_recover_by_selecting_another(self):
        FakeEstimator.fail_constructor.add('egoforce')
        self.app.configure({'pose_model': 'egoforce'})
        self.app.camera.start()
        self.until(lambda: self.app.status()['pose']['state'] == 'error')
        self.app.configure({'pose_model': 'rtmpose'})
        self.until(lambda: self.app.status()['pose']['state'] == 'ready')
        self.assertEqual(self.app.status()['pose']['model_id'], 'rtmpose')
    def test_invalid_configuration_is_atomic_and_roi_is_copied(self):
        original = self.app.status()['settings']
        for values in ({'pose_model': 'unknown'}, {'pose_roi': [0, 0, 1000, 20]},
                       {'pose_enabled': 'yes'}, {'confidence': float('nan')}):
            with self.subTest(values=values), self.assertRaises(ValueError): self.app.configure(values)
            self.assertEqual(self.app.status()['settings'], original)
        roi = [10, 10, 30, 35]
        self.app.configure({'pose_roi': roi})
        roi[0] = 0
        self.assertEqual(self.app.status()['settings']['pose_roi'], [10., 10., 30., 35.])


if __name__ == '__main__':
    unittest.main()
