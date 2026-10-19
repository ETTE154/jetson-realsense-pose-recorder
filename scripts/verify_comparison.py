"""Verify a completed real comparison without opening or stopping the camera."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8766')
    parser.add_argument('--job')
    parser.add_argument('--output', type=Path, default=Path('logs/comparison_validation.json'))
    args = parser.parse_args()
    def read(path):
        with urllib.request.urlopen(args.url + path, timeout=15) as response:
            return response.read()
    job = json.loads(read('/api/compare/job/'+args.job)) if args.job else json.loads(read('/api/compare/state'))['job']
    if not job or job['busy'] or job['state'] != 'done':
        raise RuntimeError('A completed comparison is required; no model success is assumed')
    source = read(job['source']['image_url'])
    digest = hashlib.sha256(source).hexdigest()
    assert digest == job['source']['sha256'], 'Saved source hash mismatch'
    dimensions = (job['source']['height'], job['source']['width'], 3)
    assert cv2.imdecode(np.frombuffer(source, np.uint8), 1).shape == dimensions
    report = {'job': job['id'], 'source': job['source'], 'roi': job['roi'], 'models': {}}
    for candidate in ('mediapipe', 'rtmpose', 'egoforce'):
        result = job['results'][candidate]
        assert result['state'] == 'done', (candidate, result)
        assert result['input_sha256'] == digest, 'Candidates did not share the same input'
        image = cv2.imdecode(np.frombuffer(read(result['image_url']), np.uint8), 1)
        assert image is not None and image.shape == dimensions, candidate
        displayed = measured = count = 0
        for instance in result['instances']:
            assert np.isfinite(instance['keypoints']).all()
            for point in instance['depth_keypoints']:
                count += 1
                displayed += int(point['displayed'])
                measured += int(point['valid_depth'])
                if point['valid_depth']:
                    assert point['displayed'] and np.isfinite([point['x_m'], point['y_m'], point['z_m']]).all()
                else:
                    assert point['x_m'] is None and point['y_m'] is None and point['z_m'] is None
                if instance.get('keypoint_confidence_kind') == 'not_reported_display_mask':
                    assert point['confidence'] is None, 'Display mask must not be exported as a probability'
        assert displayed == result['visible_keypoints']
        assert measured == result['valid_depth_keypoints']
        report['models'][candidate] = {'backend': result['backend'], 'inference_ms': result['inference_ms'],
                                       'returned_points': count, 'displayed_points': displayed,
                                       'valid_surface_depth_points': measured, 'image_decoded': True}
    status = json.loads(read('/api/status'))
    report['camera'] = status['camera']
    assert status['camera']['state'] == 'streaming', 'Original camera stream must remain active'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
