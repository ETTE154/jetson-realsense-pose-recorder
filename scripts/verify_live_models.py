"""Exercise real model selection and each live MJPEG feed through the HTTP API."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8766')
    parser.add_argument('--models', nargs='+', choices=('yolo', 'mediapipe', 'rtmpose', 'egoforce'),
                        default=['yolo', 'mediapipe', 'rtmpose', 'egoforce'])
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=180.)
    parser.add_argument('--leave-last', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('validation/live_model_validation.json'))
    args = parser.parse_args()
    if not 2 <= args.frames <= 20: parser.error('--frames must be 2..20')
    def api(path, values=None):
        request = urllib.request.Request(args.url+path,
                                         data=None if values is None else json.dumps(values).encode(),
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=15) as response: return json.load(response)
    def stream(name):
        started = time.monotonic()
        items = []
        with urllib.request.urlopen(args.url+'/stream/'+name+'.mjpg', timeout=15) as response:
            while len(items) < args.frames:
                line = response.readline()
                if not line: raise RuntimeError(name+' stream ended unexpectedly')
                if not line.startswith(b'--frame'): continue
                headers = {}
                while True:
                    line = response.readline()
                    if line in (b'\r\n', b'\n'): break
                    key, value = line.decode().split(':', 1)
                    headers[key.lower()] = value.strip()
                size = int(headers['content-length'])
                data = response.read(size)
                image = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
                assert image is not None and image.shape == (480, 640, 3), name
                items.append({'sequence': int(headers['x-sequence']), 'sha256': hashlib.sha256(data).hexdigest()})
        assert all(b['sequence'] > a['sequence'] for a, b in zip(items, items[1:])), name+' duplicate frame sequence'
        return {'decoded_frames': len(items), 'elapsed_s': time.monotonic()-started, 'frames': items}
    previous = api('/api/status')['settings']
    report = {'models': {}, 'started_at': time.time()}
    try:
        for model in args.models:
            api('/api/settings', {'pose_model': model, 'pose_enabled': True, 'pose_roi': None})
            deadline = time.monotonic()+args.timeout
            while True:
                status = api('/api/status')
                pose = status['pose']
                if pose['model_id'] == model and pose['state'] == 'error':
                    raise RuntimeError(model+': '+str(pose['error']))
                if pose['model_id'] == model and pose['state'] == 'ready' and pose['frame_id'] is not None: break
                if time.monotonic() >= deadline: raise TimeoutError(model+' never became ready')
                time.sleep(.2)
            assert status['camera']['state'] == 'streaming'
            feeds = {name: stream(name) for name in ('rgb', 'depth', 'rgbd', 'pose', 'pose_rgbd')}
            after = api('/api/status')
            assert after['pose']['model_id'] == model and after['pose']['frame_id'] > pose['frame_id'], model+' Pose did not advance'
            assert after['camera']['frame_id'] > status['camera']['frame_id'], 'Capture did not advance'
            if model == 'egoforce':
                assert 'cpu' not in after['pose']['backend'].lower(), 'CPU fallback is forbidden'
                assert any(token in after['pose']['backend'].lower() for token in ('cuda', 'gpu')), 'CUDA backend must be explicit'
            report['models'][model] = {'pose': after['pose'], 'camera': after['camera'], 'streams': feeds}
            print(model, after['pose']['backend'], 'fps', after['pose']['fps'],
                  'ms', round(after['pose']['inference_ms'], 2), 'points', after['pose']['visible_keypoints'], flush=True)
        report['ok'] = True
    finally:
        if not args.leave_last: api('/api/settings', previous)
        report['finished_at'] = time.time()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
