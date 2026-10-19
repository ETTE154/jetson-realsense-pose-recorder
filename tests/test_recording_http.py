"""Large recording downloads must support seek ranges and reject traversal."""
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request

from server import Server, handler_for


class RecordingDownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='recording-http-test-')
        root = Path(self.directory.name)
        (root/'session').mkdir()
        self.data = bytes(range(256))*10
        (root/'session'/'rgb.mp4').write_bytes(self.data)
        (root/'session'/'unfinished.mp4.part').write_bytes(self.data)
        (root/'session'/'rgb.partial.mp4').write_bytes(self.data)
        self.csv_data = '\ufeffrecord_type,accel_x_m_s2\nimu_accel,0.1\n'.encode('utf-8')
        (root/'session'/'dataset.csv').write_bytes(self.csv_data)
        (root/'session'/'dataset.partial.csv').write_bytes(self.csv_data)
        app = SimpleNamespace(recorder=SimpleNamespace(output_dir=root))
        self.server = Server(('127.0.0.1', 0), handler_for(app))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:'+str(self.server.server_address[1])
    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.directory.cleanup()
    def request(self, path='/recordings/session/rgb.mp4', headers=None, method=None):
        return urllib.request.urlopen(urllib.request.Request(self.url+path, headers=headers or {}, method=method), timeout=3)
    def test_full_head_and_middle_range(self):
        with self.request() as response:
            self.assertEqual(response.read(), self.data)
            self.assertEqual(response.headers['Accept-Ranges'], 'bytes')
        with self.request(method='HEAD') as response:
            self.assertEqual(int(response.headers['Content-Length']), len(self.data))
            self.assertEqual(response.read(), b'')
        with self.request(headers={'Range': 'bytes=10-29'}) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), self.data[10:30])
            self.assertEqual(response.headers['Content-Range'], f'bytes 10-29/{len(self.data)}')
    def test_suffix_open_ended_and_invalid_ranges(self):
        with self.request(headers={'Range': 'bytes=-12'}) as response:
            self.assertEqual(response.read(), self.data[-12:])
        with self.request(headers={'Range': 'bytes=2500-'}) as response:
            self.assertEqual(response.read(), self.data[2500:])
        for value in ('bytes=99999-', 'bytes=20-10', 'bytes=-0', 'bytes=0-1,3-4'):
            with self.subTest(value=value), self.assertRaises(urllib.error.HTTPError) as raised:
                self.request(headers={'Range': value})
            self.assertEqual(raised.exception.code, 416)
    def test_unfinished_files_and_parent_paths_are_not_served(self):
        for path in ('/recordings/session/unfinished.mp4.part', '/recordings/session/rgb.partial.mp4',
                     '/recordings/session/dataset.partial.csv',
                     '/recordings/%2e%2e/camera.py'):
            with self.subTest(path=path), self.assertRaises(urllib.error.HTTPError) as raised:
                self.request(path)
            self.assertEqual(raised.exception.code, 404)

    def test_dataset_csv_is_utf8_attachment_with_exact_download_bytes(self):
        path = '/recordings/session/dataset.csv'
        with self.request(path) as response:
            self.assertEqual(response.read(), self.csv_data)
            self.assertEqual(response.headers['Content-Type'], 'text/csv; charset=utf-8')
            self.assertIn('attachment', response.headers['Content-Disposition'])
        with self.request(path, method='HEAD') as response:
            self.assertEqual(int(response.headers['Content-Length']), len(self.csv_data))
            self.assertEqual(response.read(), b'')
        with self.request(path, headers={'Range': 'bytes=0-2'}) as response:
            self.assertEqual(response.status, 206)
            self.assertEqual(response.read(), b'\xef\xbb\xbf')


if __name__ == '__main__':
    unittest.main()
