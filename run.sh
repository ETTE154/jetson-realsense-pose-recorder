#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -f .realsense-rsusb/python/pyrealsense2/__init__.py ]]; then
  export PYTHONPATH="$PWD/.realsense-rsusb/python${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="$PWD/.realsense-rsusb/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec .venv/bin/python server.py --host 127.0.0.1 --port 8766 --model models/yolov8n-pose.engine --auto-start "$@"
