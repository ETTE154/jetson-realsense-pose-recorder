#!/usr/bin/env bash
# Jetson Orin NX / JetPack 6.2: isolated MediaPipe CPU and HALO CUDA runtimes.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/setup_mediapipe.sh
.venv/bin/python scripts/setup_rtmpose.py
.venv/bin/python scripts/setup_egoforce.py --download
bash scripts/setup_egoforce_gpu.sh
echo 'Three candidates prepared. Open /compare on the existing local server.'
