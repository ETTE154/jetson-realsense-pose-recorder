"""The UI receives raw sample cursors and explicit gaps, never interpolated data."""
import json
import threading
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request

from motion import MotionTracker
from prototype import Application
from server import Server, handler_for


class IMUAPITests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.tracker = MotionTracker(capacity=3, clock=lambda: self.now)
        self.tracker.begin(7, [{"type": "gyro", "fps": 200}, {"type": "accel", "fps": 63}])
        self.app = Application.__new__(Application)
        self.app.camera = SimpleNamespace(motion_history=self.tracker.history, motion_snapshot=self.tracker.snapshot)
        self.server = Server(("127.0.0.1", 0), handler_for(self.app))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:" + str(self.server.server_address[1])

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def emit(self, number):
        self.now += .01
        self.tracker.emit(kind="gyro" if number % 2 else "accel", x=number, y=-number, z=0.,
            timestamp_ms=number*5., timestamp_domain="hardware_clock", frame_number=number,
            received_time=1700000000+self.now, received_monotonic=self.now, capture_generation=7)

    def get(self, query):
        with urllib.request.urlopen(self.url+"/api/imu"+query, timeout=3) as response:
            return json.load(response)

    def test_cursor_does_not_skip_page_and_exposes_eviction(self):
        for number in range(5):
            self.emit(number)
        first = self.get("?after=0&limit=2")
        self.assertTrue(first["truncated"])
        self.assertEqual([row["sequence"] for row in first["samples"]], [3, 4])
        self.assertEqual(first["latest_sequence"], 4)
        second = self.get("?after=4")
        self.assertEqual([row["sequence"] for row in second["samples"]], [5])
        self.assertFalse(second["truncated"])
        self.assertEqual(second["imu"]["capture_epoch"], 7)
        self.assertEqual(second["samples"][0]["timestamp_domain"], "hardware_clock")
        self.assertEqual(second["imu"]["accel"]["unit"], "m/s²")
        self.assertEqual(second["imu"]["accel"]["count"], 3)

    def test_stale_disconnected_and_epoch_do_not_fabricate_values(self):
        self.emit(0)
        self.emit(1)
        self.now += 2
        result = self.get("")
        self.assertEqual(result["imu"]["state"], "stale")
        self.assertGreater(result["imu"]["accel"]["age_ms"], 1000)
        self.tracker.set_state("unavailable", "Motion Module missing")
        result = self.get("?after=2")
        self.assertEqual(result["samples"], [])
        self.assertIsNone(result["imu"]["accel"])
        self.assertFalse(result["imu"]["available"])
        self.tracker.begin(8, [])
        self.assertEqual(self.get("?after=2")["imu"]["capture_epoch"], 8)

    def test_invalid_cursor_and_unbounded_limit_are_rejected(self):
        for query in ("?after=-1", "?after=abc", "?limit=0", "?limit=4097"):
            with self.subTest(query=query), self.assertRaises(urllib.error.HTTPError) as error:
                self.get(query)
            self.assertEqual(error.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
