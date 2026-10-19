"""Raw-frame integrity and real process interruption during loading/inference."""
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

try:
    from .. import live_runtime
    from ..live_protocol import encode_frame, read_frame, write_all, MAX_HEADER
except ImportError:
    import live_runtime
    from live_protocol import encode_frame, read_frame, write_all, MAX_HEADER


class FragmentedInput(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 7))


class FragmentedOutput(io.BytesIO):
    def write(self, data):
        return super().write(data[:5])


class LiveProtocolTests(unittest.TestCase):
    def test_lossless_noncontiguous_frame_and_metadata(self):
        original = np.arange(9*13*3, dtype=np.uint8).reshape(9, 13, 3)[:, ::-1]
        prefix, pixels = encode_frame(original, sequence=77, intrinsics={"fx": 300.}, roi=[0, 0, 8, 8])
        header, image = read_frame(FragmentedInput(prefix + pixels.tobytes()))
        np.testing.assert_array_equal(image, original)
        self.assertEqual(header["sequence"], 77)
        self.assertEqual(header["intrinsics"], {"fx": 300.})
        self.assertEqual(header["roi"], [0, 0, 8, 8])
        self.assertEqual(header["nbytes"], original.nbytes)

    def test_short_pipe_writes_are_completed(self):
        output = FragmentedOutput()
        write_all(output, memoryview(b"abc123" * 101))
        self.assertEqual(output.getvalue(), b"abc123" * 101)

    def test_truncated_and_oversized_frame_rejected(self):
        prefix, pixels = encode_frame(np.zeros((3, 4, 3), np.uint8), sequence=1)
        for value in (prefix[:2], prefix + pixels.tobytes()[:-1]):
            with self.assertRaises(EOFError):
                read_frame(io.BytesIO(value))
        with self.assertRaises(ValueError):
            read_frame(io.BytesIO(struct.pack("!I", MAX_HEADER+1)))
        invalid = json.dumps({"shape": [2, 3, 3], "nbytes": 17}).encode()
        with self.assertRaises(ValueError):
            read_frame(io.BytesIO(struct.pack("!I", len(invalid)) + invalid))
        self.assertIsNone(read_frame(io.BytesIO()))

    def _interrupt_process(self, child_source, large_frame=False):
        original_popen = subprocess.Popen
        def spawn(_command, **kwargs):
            return original_popen([sys.executable, "-u", "-c", child_source], **kwargs)
        failures = []
        entered = threading.Event()
        with patch.object(live_runtime, "python_path", return_value=Path(sys.executable)), \
                patch.object(live_runtime.subprocess, "Popen", side_effect=spawn):
            estimator = live_runtime.LiveEstimator("rtmpose")
            def infer():
                entered.set()
                try:
                    estimator.infer(np.zeros((480, 640, 3) if large_frame else (1, 1, 3), np.uint8))
                except RuntimeError as error:
                    failures.append(str(error))
            thread = threading.Thread(target=infer)
            thread.start()
            self.assertTrue(entered.wait(1))
            if large_frame:
                self.assertTrue(estimator._ready.wait(3))
            time.sleep(.05)
            started = time.monotonic()
            process = estimator._process
            estimator.close()
            elapsed = time.monotonic()-started
            thread.join(2)
            self.assertFalse(thread.is_alive(), "inference wait must wake on close")
            self.assertIsNotNone(process.poll(), "model process must exit")
            self.assertTrue(failures)
            self.assertLess(elapsed, 3)
            estimator.close()  # idempotent

    def test_close_interrupts_model_loading(self):
        self._interrupt_process("import time; time.sleep(60)")

    def test_close_interrupts_blocked_raw_frame_write(self):
        # Real pipe fills because the child never reads stdin. close must not
        # wait on the caller's inference lock or the blocked writer.
        self._interrupt_process('import time; print(\'LIVE_RESULT {"ready": true}\', flush=True); time.sleep(60)', large_frame=True)

    def test_inference_timeout_also_covers_blocked_pipe_write(self):
        original_popen = subprocess.Popen
        def spawn(_command, **kwargs):
            source = 'import time; print(\'LIVE_RESULT {"ready": true}\', flush=True); time.sleep(60)'
            return original_popen([sys.executable, "-u", "-c", source], **kwargs)
        with patch.object(live_runtime, "python_path", return_value=Path(sys.executable)), \
                patch.object(live_runtime.subprocess, "Popen", side_effect=spawn):
            estimator = live_runtime.LiveEstimator("rtmpose", inference_timeout=.1)
            process = estimator._process
            try:
                with self.assertRaisesRegex(RuntimeError, "transfer timed out"):
                    estimator.infer(np.zeros((480, 640, 3), np.uint8))
                self.assertIsNotNone(process.poll())
            finally:
                estimator.close()


if __name__ == "__main__":
    unittest.main()
