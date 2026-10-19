#!/usr/bin/env bash
# Official NVIDIA Jetson PyTorch wheel, in a dedicated project environment.
# Does not replace JetPack, CUDA, system Python, or the existing CPU environment.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(uname -m)" == "aarch64" ]] || { echo 'This runtime targets Jetson aarch64.' >&2; exit 1; }
python3 -m venv --system-site-packages .egoforce-gpu-venv
runtime_python="$PWD/.egoforce-gpu-venv/bin/python"
asset_dir="$PWD/models/egoforce/runtime"
mkdir -p "$asset_dir"
wheel_name='torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl'
archive_name='libcusparse_lt-linux-aarch64-0.6.2.3-archive'
wheel_url="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/$wheel_name"
archive_url="https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/$archive_name.tar.xz"
fetch_asset() {
    local asset_url="$1" asset_path="$2"
    if [[ ! -s "$asset_path" ]]; then
        curl --fail --location --retry 3 --continue-at - --output "$asset_path.part" "$asset_url"
        mv -- "$asset_path.part" "$asset_path"
    fi
}
fetch_asset "$wheel_url" "$asset_dir/$wheel_name"
fetch_asset "$archive_url" "$asset_dir/$archive_name.tar.xz"
printf '%s  %s\n' '6f75fd2d2ef840ede1a90dbcf40a5458214bee26cc803fa510cda2e8978d972a' "$asset_dir/$wheel_name" | sha256sum --check --status
printf '%s  %s\n' 'b081a82a884754fd7ab1f53aa9cd1a943d782b87268c3657991c287af3179811' "$asset_dir/$archive_name.tar.xz" | sha256sum --check --status
if [[ ! -s "$asset_dir/$archive_name/lib/libcusparseLt.so.0" ]]; then
    tar -xJf "$asset_dir/$archive_name.tar.xz" -C "$asset_dir" "$archive_name/lib"
fi
export LD_LIBRARY_PATH="$asset_dir/$archive_name/lib:${LD_LIBRARY_PATH:-}"
"$runtime_python" -m pip install --upgrade pip
"$runtime_python" -m pip install 'numpy==1.26.4' filelock typing_extensions 'sympy==1.13.1' networkx jinja2 fsspec pyyaml huggingface_hub safetensors pillow
"$runtime_python" -m pip install --no-deps "$asset_dir/$wheel_name" 'timm==1.0.22'
# PyPI's prebuilt torchvision ABI does not match NVIDIA's prerelease PyTorch.
# Build the official release against this exact wheel. HALO uses torchvision's
# Python utilities through timm; its neural network still executes on CUDA.
if ! "$runtime_python" -c 'import torchvision, timm; assert torchvision.__version__.startswith("0.20.0")' >/dev/null 2>&1; then
    "$runtime_python" -m pip install ninja
    fetch_asset 'https://github.com/pytorch/vision/archive/refs/tags/v0.20.0.tar.gz' "$asset_dir/vision-v0.20.0.tar.gz"
    printf '%s  %s\n' 'b59d9896c5c957c6db0018754bbd17d079c5102b82b9be0b438553b40a7b6029' "$asset_dir/vision-v0.20.0.tar.gz" | sha256sum --check --status
    mkdir -p "$asset_dir/source"
    if [[ ! -f "$asset_dir/source/vision-0.20.0/setup.py" ]]; then
        tar -xzf "$asset_dir/vision-v0.20.0.tar.gz" -C "$asset_dir/source"
    fi
    PATH="$PWD/.egoforce-gpu-venv/bin:$PATH" CUDA_VISIBLE_DEVICES='' MAX_JOBS=2 BUILD_VERSION=0.20.0 \
        TORCHVISION_USE_VIDEO_CODEC=0 TORCHVISION_USE_FFMPEG=0 TORCHVISION_USE_NVJPEG=0 \
        "$runtime_python" -m pip install --no-build-isolation --no-deps --force-reinstall "$asset_dir/source/vision-0.20.0"
fi
"$runtime_python" -m pip install 'absl-py~=2.3' certifi 'sounddevice~=0.5' 'flatbuffers~=25.9'
"$runtime_python" -m pip install --no-deps 'mediapipe==1.1.1'
"$runtime_python" - <<'PY'
from pathlib import Path
import hashlib, json, time
import torch, torchvision, timm, mediapipe
if not torch.cuda.is_available():
    raise RuntimeError('Official NVIDIA wheel did not initialize CUDA; CPU fallback is not accepted')
device = torch.cuda.get_device_properties(0)
started = time.perf_counter()
x = torch.randn(512, 512, device='cuda', dtype=torch.float16)
y = x @ x.T
torch.cuda.synchronize()
if not torch.isfinite(y).all().item():
    raise RuntimeError('CUDA FP16 matrix verification produced nonfinite results')
root = Path('models/egoforce/runtime')
artifacts = {}
for path in root.glob('*'):
    if path.is_file() and path.suffix in ('.whl', '.xz', '.gz'):
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8*1024*1024), b''):
                digest.update(block)
        artifacts[path.name] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
report = {'torch': torch.__version__, 'torchvision': torchvision.__version__, 'timm': timm.__version__,
          'mediapipe': mediapipe.__version__, 'cuda': torch.version.cuda,
          'device': device.name, 'capability': [device.major,device.minor],
          'fp16_matrix_verified': True, 'verification_ms': (time.perf_counter()-started)*1000,
          'wheel_source': 'https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/',
          'cusparselt_source': 'https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/',
          'torchvision_source': 'https://github.com/pytorch/vision/tree/v0.20.0',
          'torchvision_build': 'CPU utility ops compiled against NVIDIA PyTorch ABI; HALO model runs CUDA FP16',
          'artifacts': artifacts}
(root/'gpu_runtime_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
PY
