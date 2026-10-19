"""Comparison invariants: one input, explicit score meanings, real depth only."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from comparison import manager as module


class ComparisonTests(unittest.TestCase):
    def test_cancelled_worker_never_spawns_a_child(self):
        worker = module.ModelWorker('mediapipe')
        cancelled = threading.Event()
        cancelled.set()
        with patch.object(module.subprocess, 'Popen') as launch:
            with self.assertRaises(RuntimeError): worker.start(cancelled)
            launch.assert_not_called()

    def test_cancel_during_process_creation_closes_the_published_child(self):
        created, release = threading.Event(), threading.Event()
        cancelled = threading.Event()
        class FakeProcess:
            def __init__(self):
                self.stdin = io.StringIO()
                self.stdout = io.StringIO(module.PREFIX + '{"ready":true,"load_ms":1}\n')
                self.terminated = False
            def poll(self): return 0 if self.terminated else None
            def terminate(self): self.terminated = True
            def wait(self, **kwargs): return 0
        process = FakeProcess()
        def launch(*args, **kwargs):
            created.set()
            self.assertTrue(release.wait(2))
            return process
        errors = []
        with tempfile.TemporaryDirectory(prefix='worker-test-') as temporary:
            runtime = Path(temporary)/'python'
            runtime.touch()
            with patch.object(module, 'ROOT', Path(temporary)), patch.object(module, 'python_path', return_value=runtime), \
                    patch.object(module.subprocess, 'Popen', side_effect=launch):
                worker = module.ModelWorker('mediapipe')
                def start():
                    try: worker.start(cancelled)
                    except RuntimeError as error: errors.append(str(error))
                startup = threading.Thread(target=start)
                startup.start()
                self.assertTrue(created.wait(2))
                cancelled.set()
                closer = threading.Thread(target=worker.close)
                closer.start()
                release.set()
                startup.join(2)
                closer.join(2)
                self.assertFalse(startup.is_alive())
                self.assertFalse(closer.is_alive())
                self.assertTrue(process.terminated)
                self.assertIsNone(worker.process)

    def test_roi_rejects_outside_nonfinite_and_degenerate(self):
        self.assertEqual(module.validate_roi([1, 2, 20, 30], 640, 480), [1., 2., 20., 30.])
        for roi in ([0, 0, 641, 10], [0, 0, 10, 481], [4, 4, 3, 20],
                    [0, 0, 7, 40], [0, 0, float('nan'), 40], [False, 0, 20, 40]):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                module.validate_roi(roi, 640, 480)

    def test_depth_uses_scale_median_and_masks_are_not_probabilities(self):
        depth = np.full((10, 10), 2000, np.uint16)
        depth[5, 5] = 65000
        instance = {'keypoints': [[5., 5., 1.]], 'keypoint_names': ['wrist'],
                    'keypoint_confidence_kind': 'not_reported_display_mask'}
        output, shown, measured = module.enrich_instances([instance], depth, .0005, {}, .95,
                                                         deproject=lambda uv, z: [0., 0., z])
        joint = output[0]['depth_keypoints'][0]
        self.assertEqual((shown, measured), (1, 1))
        self.assertEqual(joint['z_m'], 1.)
        self.assertIsNone(joint['confidence'])
        self.assertNotIn('depth_keypoints', instance)

    def test_invalid_depth_and_low_score_never_get_coordinates(self):
        instance = {'keypoints': [[5, 5, .9], [-1, 5, .9], [5, 5, .1]]}
        def invalid_projection(*args):
            self.fail('Missing depth must not be projected')
        output, shown, measured = module.enrich_instances([instance], np.zeros((10, 10), np.uint16),
                                                         .001, {}, .3, deproject=invalid_projection)
        self.assertEqual((shown, measured), (1, 0))
        self.assertTrue(all(p['z_m'] is None for p in output[0]['depth_keypoints']))
        with self.assertRaises(ValueError):
            module.enrich_instances([{'keypoints': [[float('inf'), 2, .9]]}],
                                    np.zeros((10, 10), np.uint16), .001, {}, .3,
                                    deproject=invalid_projection)

    def test_same_lossless_frame_shared_hints_and_model_failure_are_retained(self):
        rgb = np.arange(12*16*3, dtype=np.uint8).reshape(12, 16, 3)
        depth = np.full((12, 16), 1200, np.uint16)
        metadata = {'frame_id': 42, 'depth_scale_m_per_unit': .001, 'depth_intrinsics': {}}
        hints = [{'keypoints': [[5., 5., 1.]], 'keypoint_names': ['wrist'],
                  'keypoint_confidence_kind': 'not_reported_display_mask'}]
        calls = []
        class FakeWorker:
            load_ms = 5.
            def __init__(self, identifier): self.identifier = identifier
            def start(self, cancel_event=None): pass
            def close(self): pass
            def infer(self, request, cancel_event=None):
                raw = Path(request['image']).read_bytes()
                calls.append((self.identifier, hashlib.sha256(raw).hexdigest(), request['hand_hints']))
                np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), rgb)
                if self.identifier == 'rtmpose': raise RuntimeError('deliberate test failure')
                return {'inference_ms': 10., 'result': {'instances': hints, 'backend': 'test-double'}}
        real_enrich = module.enrich_instances
        def enrich(*args): return real_enrich(*args, deproject=lambda uv, z: [0., 0., z])
        with tempfile.TemporaryDirectory(prefix='compare-test-') as temporary:
            app = SimpleNamespace(output_dir=Path(temporary)/'captures')
            with patch.object(module, 'ROOT', Path(temporary)), patch.object(module, 'ModelWorker', FakeWorker), \
                    patch.object(module, 'enrich_instances', side_effect=enrich), \
                    patch.object(module.LOG, 'exception'):
                manager = module.ComparisonManager(app)
                manager._source = lambda source: (rgb, depth, metadata, b'original baseline')
                job = manager.run({'source': 'live'})
                # Wait for completion, without assuming inference ordering by wall time.
                for _ in range(200):
                    if not manager.job_by_id(job['id'])['busy']: break
                    threading.Event().wait(.01)
                job = manager.job_by_id(job['id'])
                self.assertFalse(job['busy'])
                self.assertEqual([c[0] for c in calls], list(module.IDS))
                self.assertEqual({c[1] for c in calls}, {job['source']['sha256']})
                self.assertEqual(calls[0][2], [])
                self.assertEqual(calls[2][2], hints)
                self.assertEqual(job['results']['rtmpose']['state'], 'error')
                self.assertIn('deliberate test failure', job['results']['rtmpose']['error'])
                self.assertEqual(job['results']['egoforce']['state'], 'done')
                directory = Path(temporary)/'comparisons'/job['id']
                np.testing.assert_array_equal(cv2.imread(str(directory/'depth16.png'), -1), depth)
                # Disk persistence follows the final state update in the worker thread.
                for _ in range(100):
                    saved = json.loads((directory/'result.json').read_text(encoding='utf-8'))
                    if not saved['busy']: break
                    threading.Event().wait(.01)
                self.assertFalse(saved['busy'])
                self.assertFalse((directory/'rtmpose.jpg').exists())
                manager.close()


if __name__ == '__main__':
    unittest.main()
