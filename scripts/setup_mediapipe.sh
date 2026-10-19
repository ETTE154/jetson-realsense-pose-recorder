#!/usr/bin/env bash
# Run on Jetson, from any directory. Does not alter .venv/system packages.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv --system-site-packages .mediapipe-venv
.mediapipe-venv/bin/python -m pip install --upgrade pip
.mediapipe-venv/bin/python -m pip install 'numpy==1.26.4' 'absl-py~=2.3' certifi 'sounddevice~=0.5' 'flatbuffers~=25.9'
# OpenCV and matplotlib come from the Jetson's system site packages.
.mediapipe-venv/bin/python -m pip install --no-deps 'mediapipe==1.1.1'
mkdir -p models/comparison
.mediapipe-venv/bin/python - <<'PY'
from pathlib import Path
import hashlib
import urllib.request
url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
target = Path('models/comparison/hand_landmarker.task')
if not target.is_file():
    temporary = target.with_suffix('.download')
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(target)
expected = 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'
actual = hashlib.sha256(target.read_bytes()).hexdigest()
if actual != expected:
    raise RuntimeError(f'Unexpected hand model checksum: {actual}')
import mediapipe, cv2, numpy
print('MediaPipe', mediapipe.__version__, 'OpenCV', cv2.__version__, 'NumPy', numpy.__version__)
print('Model:', target.resolve(), target.stat().st_size, 'bytes')
PY
