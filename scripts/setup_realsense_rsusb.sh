#!/usr/bin/env bash
# Project-local SDK with user-space UVC/HID; leaves system SDK/kernel untouched.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$project_dir/third_party/librealsense-rsusb"
install_dir="$project_dir/.realsense-rsusb"
source_revision="625279aa1fe903528a818f48746ee47c1b5fefd3"
mkdir -p -- "$project_dir/third_party"
if [[ ! -d "$source_dir/.git" ]]; then
  git clone --depth 1 --branch v2.59.1 https://github.com/realsenseai/librealsense.git "$source_dir"
fi
actual_revision="$(git -C "$source_dir" rev-parse HEAD)"
if [[ "$actual_revision" != "$source_revision" ]]; then
  printf 'Unexpected librealsense revision: %s (expected %s). Existing source was preserved.\n' "$actual_revision" "$source_revision" >&2
  exit 1
fi
printf 'Building librealsense revision %s\n' "$actual_revision"
cmake -S "$source_dir" -B "$source_dir/build-rsusb" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_INSTALL_PREFIX="$install_dir" -DCMAKE_INSTALL_RPATH="$install_dir/lib" \
  -DPYTHON_INSTALL_DIR="$install_dir/python/pyrealsense2" \
  -DPython_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3 \
  -DFORCE_RSUSB_BACKEND=ON -DBUILD_PYTHON_BINDINGS=ON \
  -DBUILD_WITH_CUDA=OFF -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF \
  -DBUILD_TOOLS=OFF -DBUILD_UNIT_TESTS=OFF -DBUILD_GLSL_EXTENSIONS=OFF \
  -DBUILD_WITH_DDS=OFF -DBUILD_ROSBAG2=OFF -DBUILD_RS2_ALL=OFF \
  -DCHECK_FOR_UPDATES=OFF -DENABLE_AI_ASSISTANT=OFF -DENABLE_STATS=OFF
cmake --build "$source_dir/build-rsusb" --parallel 4
cmake --install "$source_dir/build-rsusb"
printf 'RSUSB SDK installed in %s\n' "$install_dir"
