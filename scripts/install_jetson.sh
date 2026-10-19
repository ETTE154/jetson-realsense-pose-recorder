#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_UDEV=0
for argument in "$@"; do
    case "$argument" in
        --install-udev) INSTALL_UDEV=1 ;;
        -h|--help)
            echo "Usage: bash scripts/install_jetson.sh [--install-udev]"
            echo "--install-udev: D435i(8086:0b3a) 전용 video 그룹 USB 권한 규칙을 sudo로 설치합니다."
            exit 0 ;;
        *) echo "알 수 없는 옵션: $argument" >&2; exit 2 ;;
    esac
done
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
    echo "이 설치 스크립트는 Linux aarch64 Jetson용입니다." >&2
    exit 1
fi
cd -- "$PROJECT_ROOT"
if ! python3 -c 'import cv2, numpy, tensorrt' >/dev/null 2>&1; then
    echo "시스템 Python의 OpenCV, NumPy, TensorRT를 확인하세요: python3 -c 'import cv2,numpy,tensorrt'" >&2
    exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
    if ! python3 -m venv --system-site-packages .venv; then
        echo "가상환경 생성에 실패했습니다. python3-venv 설치 여부를 확인하세요." >&2
        exit 1
    fi
fi
.venv/bin/python -m pip install --disable-pip-version-check 'pyrealsense2==2.59.1.11053'
.venv/bin/python -c 'import cv2,numpy,tensorrt,pyrealsense2; print("Python dependencies ready. TensorRT:",tensorrt.__version__)'

RULE_NAME="99-d435i-prototype.rules"
RULE_TARGET="/etc/udev/rules.d/$RULE_NAME"
RULE_CONTENT='SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="8086", ATTR{idProduct}=="0b3a", MODE="0660", GROUP="video"'
if [[ -e "$RULE_TARGET" || -L "$RULE_TARGET" ]]; then
    echo "D435i 전용 규칙 파일이 이미 있습니다. 기존 파일을 변경하지 않습니다: $RULE_TARGET"
elif [[ "$INSTALL_UDEV" == 1 ]]; then
    mkdir -p -- "$PROJECT_ROOT/.setup"
    RULE_LOCAL="$PROJECT_ROOT/.setup/$RULE_NAME"
    if [[ -e "$RULE_LOCAL" || -L "$RULE_LOCAL" ]]; then
        if [[ -L "$RULE_LOCAL" ]] || ! cmp -s <(printf '%s\n' "$RULE_CONTENT") "$RULE_LOCAL"; then
            echo "기존 로컬 규칙 파일이 D435i 전용 설정과 다릅니다. 덮어쓰지 않습니다: $RULE_LOCAL" >&2
            exit 1
        fi
    else
        (set -o noclobber; printf '%s\n' "$RULE_CONTENT" > "$RULE_LOCAL")
    fi
    chmod 0644 -- "$RULE_LOCAL"
    echo "D435i(8086:0b3a)에만 MODE=0660, GROUP=video를 적용합니다: $RULE_LOCAL"
    echo "sudo로 $RULE_TARGET 에 설치합니다. 기존 대상 파일은 덮어쓰지 않습니다."
    sudo cp --no-clobber --preserve=mode -- "$RULE_LOCAL" "$RULE_TARGET"
    sudo udevadm control --reload-rules
    sudo udevadm trigger --subsystem-match=usb --attr-match=idVendor=8086 --attr-match=idProduct=0b3a
else
    echo "D435i 전용 USB 권한 규칙이 없습니다. 필요하면 다음을 실행하세요:"
    echo "  bash scripts/install_jetson.sh --install-udev"
fi
if ! id -nG | tr ' ' '\n' | grep -qx video; then
    echo "현재 사용자가 video 그룹에 없습니다. video 그룹 사용자로 실행해야 카메라에 접근할 수 있습니다."
fi
mkdir -p -- models captures logs
echo "설치 완료. ONNX 모델을 models/ 에 준비하고 아래를 실행하세요:"
echo "  .venv/bin/python scripts/build_engine.py"
echo "  bash run.sh"
