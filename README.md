# Jetson RealSense Pose Recorder

Jetson Orin NX와 RealSense **D435i**로 RGB·Depth·RGB-D·Pose를 실시간 확인하고, 영상과 IMU·관절 데이터를 데이터셋 전처리용으로 수집하는 프로토타입입니다. 연산과 저장은 Jetson에서 수행하며 PC 브라우저는 SSH 터널로 접속합니다.

백업 기준은 **v1.4.2 / 2026-10-19**입니다. 새 장치에서는 아래 설치가 필요합니다. 소스·설치 스크립트·테스트·문서를 포함하며, 가상환경·모델 가중치·TensorRT 엔진·사용자 촬영 데이터는 포함하지 않습니다.

## 에이전트에게 재현을 맡길 때

다음 요청과 이 저장소 주소를 전달하면 됩니다. SSH 접속 정보는 별도로 전달하고 저장소에 기록하지 마세요.

> 이 저장소의 README와 AGENTS.md를 읽고 Jetson Orin NX에 재현해 주세요. 먼저 JetPack/CUDA/TensorRT/Python과 USB 연결을 확인하고 기존 환경과 녹화를 보존하세요. README 순서대로 기본 환경, RSUSB SDK, YOLO 엔진, 세 후보 모델을 준비하세요. 서버는 127.0.0.1:8766으로 실행하고 SSH 터널로 연결하세요. 자동 테스트, 실제 RGB/Depth/IMU 수신, 모델 전환, 90초 녹화와 CSV 검증을 수행하고 결과를 보고하세요. 손이 없는 검사에서 관절 검증 성공을 주장하지 말고 설치 불가·미검증 항목은 원인과 함께 구분해 주세요.

설치 순서는 **환경 확인 → 시스템 의존성 → 기본 venv/USB 권한 → RSUSB → YOLO 내보내기/엔진 → 후보 모델 → 서버/터널 → 실제 검증**입니다. RGB-D와 YOLO만 필요하면 후보 모델 설치를 생략할 수 있습니다. 원래의 네 모델 전환 기능을 모두 재현하려면 후보 모델도 설치하세요.

## 기능과 현재 제한

- RGB, RGB 정렬 Depth, RGB-D, Pose 스트리밍과 동일 원본 프레임 스냅샷.
- 메인 페이지에서 YOLOv8n-pose / MediaPipe Hand / RTMPose-m / EgoForce HALO 실시간 전환. `/compare`에서는 같은 정지 프레임으로 후보 세 개를 비교합니다.
- 녹화 시작 후 **5초 준비 시간**, 이후 **15fps** MP4 6개와 통합 `dataset.csv` 저장.
- `combined.mp4`는 **1280×960** 4분할: 왼쪽 위 RGB, 오른쪽 위 Depth, 왼쪽 아래 RGB-D, 오른쪽 아래 Pose.
- 가속도·각속도 원본 샘플의 CSV 저장 및 UI 수치·최근 10초 6축 그래프.

**IMU 그래프를 영상 파일로 저장하는 기능은 비활성화한 상태가 정상입니다.** 그래프 렌더링과 추가 인코딩으로 장시간 녹화에서 영상이 멈췄던 문제를 피하기 위한 결정입니다. IMU의 실시간 그래프와 원본 CSV 수집은 유지합니다. `imu.mp4`를 만들거나 합본에 IMU 패널을 다시 붙이지 마세요. 근거는 [검증 기록](docs/validation.md)에 있습니다.

CSV에는 RGB 픽셀이나 연속 원본 Z16 배열을 넣지 않습니다. Depth MP4는 표시용 색상 영상입니다. 원본 Z16 한 프레임은 스냅샷의 `depth16.png`로 저장합니다. 관절 정확도·IMU 바이어스·영상/IMU 외부 보정·정밀 시각 동기화를 완료한 모션 캡처 시스템은 아닙니다.

## 1. 하드웨어와 기준 환경 확인 — Jetson

실기 검증한 조합입니다. 다른 JetPack·Python·TensorRT 버전은 별도 호환성 확인이 필요합니다.

| 항목 | 검증 환경 |
|---|---|
| 장치 | Jetson Orin NX 16GB, Linux aarch64 |
| OS / JetPack | Ubuntu 22.04 / JetPack 6.2.1, L4T 36.4.7 |
| CUDA / TensorRT | 12.6 / 10.3.0 |
| 기본 Python | 3.10.12 |
| 기본 OpenCV / NumPy | 4.8.0 / 1.21.5, 시스템 패키지를 venv에서 사용 |
| 카메라 | D435i, 검증 펌웨어 5.17.3.10 |
| 연결 / RGB-D | USB 3.2 / 640×480·30fps |
| IMU 프로파일 | 가속도 63Hz, 자이로 200Hz |
| 미리보기 / 저장 | 최대 15fps 기본값; Pose 처리 속도는 모델에 따라 다름 |

D435i를 Jetson의 USB 3 포트에 데이터 케이블로 직결하세요. C-to-C도 **5Gbps 이상 데이터 전송을 지원하는 케이블과 호스트 포트**이면 사용할 수 있습니다. 모델과 빌드 캐시를 위해 **20GiB 이상 여유 공간을 권장**하며 이는 정확한 최소 용량은 아닙니다. 녹화 공간은 별도로 확보하세요. 최초 설치에는 GitHub·PyPI·NVIDIA·Google Storage·OpenMMLab·Hugging Face 접속이 필요합니다.

```bash
uname -m
cat /etc/nv_tegra_release
dpkg-query -W nvidia-jetpack nvidia-l4t-core
python3 --version
python3 -c 'import cv2,numpy,tensorrt; print(cv2.__version__,numpy.__version__,tensorrt.__version__)'
lsusb -d 8086:0b3a
lsusb -t
df -h .
```

`aarch64`, Python 3.10과 GPU 라이브러리 import를 확인하세요. USB 트리의 카메라는 `5000M` 이상이어야 합니다. `480M`이면 USB 2 연결입니다. 기존 서버가 있으면 `/api/status`의 `recording.state`를 먼저 확인하고 `countdown`, `recording`, `stopping` 중에는 서버나 카메라를 중단하지 마세요.

## 2. 복제와 시스템 의존성 — Jetson

`~/workspace`는 예시 설치 위치이며 이전 사용자 이름이나 IP에 의존하지 않습니다.

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone https://github.com/ETTE154/jetson-realsense-pose-recorder.git
cd jetson-realsense-pose-recorder
git rev-parse HEAD

sudo apt-get update
sudo apt-get install -y \
  git curl build-essential cmake pkg-config python3-dev python3-venv \
  libusb-1.0-0-dev libssl-dev libudev-dev \
  libgl1 libglib2.0-0 libportaudio2 python3-matplotlib \
  libjpeg-dev libpng-dev zlib1g-dev
```

CUDA·TensorRT·OpenCV는 기존 JetPack 환경을 씁니다. 시스템 Python에 `pip install --upgrade`를 실행하거나 JetPack·커널·펌웨어를 자동 업데이트하지 마세요. GPU 패키지가 없으면 JetPack SDK 구성을 확인한 뒤 진행합니다.

## 3. 기본 환경과 USB 권한 — Jetson

```bash
bash scripts/install_jetson.sh --install-udev
id -nG
```

`.venv`를 `--system-site-packages`로 만들고 `pyrealsense2==2.59.1.11053`을 설치합니다. 시스템 OpenCV·NumPy·TensorRT를 쓰므로 `.venv`에 범용 `opencv-python`이나 새 NumPy를 덧씌우지 마세요.

`--install-udev`는 아래 D435i 전용 규칙을 `/etc/udev/rules.d/99-d435i-prototype.rules`에 만듭니다. 기존 파일은 덮어쓰지 않으므로 내용이 다르면 확인해야 합니다.

```udev
SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTR{idVendor}=="8086", ATTR{idProduct}=="0b3a", MODE="0660", GROUP="video"
```

사용자에게 `video` 그룹이 없으면 다음을 실행하고 **SSH를 재접속한 뒤** 반영을 확인합니다. 카메라도 한 번 뺐다 연결하세요.

```bash
sudo usermod -aG video "$(id -un)"
# SSH 재접속 후 프로젝트로 돌아와서 실행
id -nG
```

서버는 일반 사용자로 실행합니다. 모든 USB 기기에 대한 권한 확대나 `sudo bash run.sh`는 필요하지 않습니다.

## 4. IMU용 RSUSB SDK — Jetson

기준 Jetson에서는 pip 기본 V4L2/IIO 경로에 Motion Module이 나타나지 않았습니다. **IMU까지 재현하려면 이 단계를 생략하지 마세요.** 공식 librealsense v2.59.1을 사용자 공간 RSUSB 방식으로 프로젝트 안에 빌드합니다. 커널 패치는 적용하지 않습니다. [공식 Jetson 설치 안내](https://github.com/realsenseai/librealsense/blob/master/doc/installation_jetson.md)

```bash
bash scripts/setup_realsense_rsusb.sh

# 직접 Python 검사를 할 때도 프로젝트 SDK를 선택
export PYTHONPATH="$PWD/.realsense-rsusb/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$PWD/.realsense-rsusb/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

.venv/bin/python - <<'PY'
import pyrealsense2 as rs
print('SDK module:', rs.__file__)
devices = rs.context().query_devices()
if not len(devices):
    raise SystemExit('D435i가 없습니다. USB와 권한을 확인하세요.')
for device in devices:
    print(device.get_info(rs.camera_info.name), device.get_info(rs.camera_info.usb_type_descriptor))
    sensors = device.query_sensors()
    for sensor in sensors:
        print('Sensor:', sensor.get_info(rs.camera_info.name))
    motion = [p for sensor in sensors for p in sensor.get_stream_profiles()
              if p.stream_type() in (rs.stream.accel, rs.stream.gyro)]
    print('Motion profiles:', [str(p) for p in motion])
    if not {rs.stream.accel, rs.stream.gyro}.issubset({p.stream_type() for p in motion}):
        raise SystemExit('Motion Module 미인식. SDK 선택과 연결을 확인하세요.')
PY
```

SDK import 경로가 `.realsense-rsusb/python/` 아래이고 Stereo·RGB·Motion Module 및 accel/gyro 프로파일이 보이는지 확인하세요. `run.sh`는 같은 SDK 선택을 자동으로 합니다. 다른 터미널에서 직접 Python을 실행할 때는 위 `export` 두 줄도 다시 실행하세요.

빌드 원본 commit은 `625279aa1fe903528a818f48746ee47c1b5fefd3`으로 고정합니다. 최초 빌드는 시간이 걸립니다. 실패 로그를 확인하고 최신 SDK로 자동 교체하지 마세요.

## 5. YOLO ONNX와 TensorRT 엔진

모델 파일은 Git에 포함하지 않습니다. 공식 가중치에서 ONNX를 만들고 **실행할 Jetson에서** TensorRT 엔진을 생성합니다. 다른 GPU나 TensorRT 버전의 `.engine`은 재사용하지 마세요.

### 5-A. ONNX 생성 — 별도 Linux x86_64 PC 권장 / 또는 Jetson 격리 환경

아래는 bash 예시입니다. PC에서도 저장소를 clone하고 루트에서 실행하세요. 내보내기 전용 `.export-venv`를 사용하며 추론용 `.venv`나 EgoForce 환경을 바꾸지 않습니다. [PyTorch CPU 버전 조합](https://pytorch.org/get-started/previous-versions/)

```bash
python3 -m venv .export-venv
.export-venv/bin/python -m pip install --upgrade pip
.export-venv/bin/python -m pip install \
  'torch==2.8.0' 'torchvision==0.23.0' \
  --index-url https://download.pytorch.org/whl/cpu
.export-venv/bin/python -m pip install \
  'numpy==1.26.4' 'ultralytics==8.3.230' 'onnx==1.17.0'
.export-venv/bin/python scripts/export_model.py
```

해당 OS/아키텍처에 CPU wheel이 없다면 지원되는 x86_64 Linux PC에서 만드세요. Jetson GPU용 PyTorch를 CPU판으로 바꾸지 마세요.

스크립트는 [공식 YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt)를 받아 `models/yolov8n-pose.onnx`와 메타데이터 JSON을 생성합니다. 입력 `[1,3,640,640]`, raw 출력 `[1,56,8400]`, opset13, dynamic/NMS 없음이며 ONNX checker와 shape 검사를 수행합니다. 참조 가중치 SHA-256은 `c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212`입니다.

다른 PC에서 생성했다면 실제 접속 정보로 아래 대상을 바꿔 ONNX와 JSON을 Jetson의 `models/`에 복사하세요.

```bash
scp models/yolov8n-pose.onnx models/yolov8n-pose.json \
  USER@JETSON_IP:~/workspace/jetson-realsense-pose-recorder/models/
```

torch/exporter 버전 차이로 참조 백업과 ONNX 바이트 해시가 달라질 수 있습니다. 메타데이터, shape와 실제 추론도 확인하세요.

### 5-B. 엔진 생성 — Jetson

```bash
.venv/bin/python scripts/build_engine.py \
  --onnx models/yolov8n-pose.onnx \
  --output models/yolov8n-pose.engine \
  --workspace-mib 512
```

지원 GPU에서는 FP16을 사용하고 새 빌드 성공 후 기존 엔진을 교체합니다. 기준기에서는 최초 빌드에 약 8.5분이 걸렸습니다. 파일이 있다는 사실만으로 실제 추론 성공으로 판단하지 마세요.

## 6. 후보 3모델 추가 — Jetson

원래 모델 전환 기능을 모두 쓰려면 실행합니다. **EgoForce 약 3GB 가중치 다운로드와 torchvision 빌드**가 포함되어 수분 이상 걸립니다.

```bash
bash scripts/setup_comparison.sh
.egoforce-gpu-venv/bin/python scripts/check_egoforce_cuda.py
```

| 모델 | 환경 | 실행과 출력 |
|---|---|---|
| YOLOv8n-pose | `.venv` | TensorRT GPU, 사람 검출＋COCO17점 |
| MediaPipe Hand | `.mediapipe-venv` | CPU, 손목·손가락21점 |
| RTMPose-m Body7 | `.venv` | TensorRT GPU, 전체/ROI의 신체17점. ROI를 사람 검출 수로 해석하지 않음 |
| EgoForce HALO | `.egoforce-gpu-venv` | HALO CUDA FP16, 손21점＋선택 전완3점. 손 검출은 MediaPipe CPU |

실패한 단계는 개별 명령으로 재개할 수 있습니다.

```bash
bash scripts/setup_mediapipe.sh
.venv/bin/python scripts/setup_rtmpose.py
.venv/bin/python scripts/setup_egoforce.py --download
bash scripts/setup_egoforce_gpu.sh
```

MediaPipe는 `1.1.1`, 격리 환경 NumPy는 `1.26.4`입니다. EgoForce는 NVIDIA 공식 Jetson PyTorch `2.5.0a0+872d972e41.nv24.08`(CUDA12.6), torchvision `0.20.0`, 프로젝트 내 cuSPARSELt `0.6.2.3`을 사용합니다. 설치 스크립트는 wheel·라이브러리·소스 해시와 CUDA FP16 연산을 검사합니다. **HALO의 CPU fallback은 허용하지 않습니다.**

EgoForce 구현은 공식 HALO의 **2D 손·전완 모드**입니다. 전체 공식 데모·MANO mesh·힘 추정 재현은 아닙니다. 손이 안 보이는 팔만의 입력을 반드시 처리하지는 못하며, 손 검출이 없을 때 손을 감싸는 수동 ROI를 줄 수 있습니다. 구현 범위와 고정 revision·라이선스는 [EgoForce 안내](COMPARISON_EGOFORCE.md)를 읽으세요.

## 7. 실행과 PC 접속

### Jetson 실행

```bash
bash run.sh --backend tensorrt
```

포그라운드 실행이며 `Ctrl+C`로 종료합니다. 카메라가 없으면 연결 상태를 표시하고 재접속을 시도합니다. 다른 Jetson 터미널에서 상태를 확인할 수 있습니다.

```bash
curl --fail http://127.0.0.1:8766/api/status
```

`run.sh`는 `127.0.0.1:8766`, YOLO 엔진과 카메라 자동 시작을 지정합니다. 옵션은 `.venv/bin/python server.py --help`를 확인하세요. 표준 검증 스크립트도 8766 포트를 사용합니다. HTTP 서버 자체에는 로그인 기능이 없으므로 PC는 SSH 터널로 접속합니다.

### PC SSH 터널 — Windows PowerShell / Linux / macOS

`USER`, `JETSON_IP`를 실제 값으로 바꾸세요. 첫 SSH 호스트 키는 실기기와 대조하고 키 불일치를 무시하는 옵션을 사용하지 마세요.

```bash
ssh -N -L 127.0.0.1:8766:127.0.0.1:8766 USER@JETSON_IP
```

터미널을 유지한 상태로 PC 브라우저에서 **http://127.0.0.1:8766/** 을 엽니다. Jetson 자체 브라우저도 같은 URL입니다. 기존 장치의 IP·사용자명·호스트 키를 소스에 고정할 필요가 없습니다.

## 8. 재현 검증

아래는 **Jetson의 저장소 루트**에서 실행합니다. SDK를 직접 쓰는 Python 검사는 4단계의 `PYTHONPATH`, `LD_LIBRARY_PATH`를 설정해야 합니다.

### 자동 테스트 — 카메라를 열지 않음

```bash
.venv/bin/python -m unittest discover -s tests -v
# Node.js 18 이상이 있을 때. 실제 브라우저 렌더링과 별개인 UI 로직 검사
node tests/test_imu_ui.js
```

기준기에서는 Python **82개**와 Node UI 검사가 통과했습니다. 일부 테스트는 `pyrealsense2` 좌표 계산을 쓰므로 SDK 미설치 PC에서는 실패할 수 있습니다.

### GPU 추론 — 카메라 불필요

사용자 사진 대신 공개 참조 이미지를 받습니다.

```bash
mkdir -p models
curl --fail --location \
  https://raw.githubusercontent.com/ultralytics/ultralytics/v8.3.230/ultralytics/assets/bus.jpg \
  --output models/test_bus.jpg
.venv/bin/python scripts/verify_runtime.py
```

결과는 `logs/gpu_validation.json`입니다. `--camera`를 추가하면 SDK로 직접 장치를 엽니다. 이때는 **사용자 녹화 종료 후 서버를 멈춘 상태에서** 실행하세요. 아래 HTTP 검사에서는 서버를 계속 실행합니다.

### 실제 센서·모델·CSV — 서버 실행 중

1. UI에서 RGB·Depth·RGB-D 프레임 갱신, USB3, 가속도/각속도 수신과 6축 그래프를 확인합니다. 정지한 수치와 미수신을 구분합니다.
2. 모델을 바꾸며 실행 장치와 프레임 번호 증가를 확인합니다. 손 모델에는 손과 손가락을, YOLO에는 사람을 보여 주세요.
3. 아래 검사는 실제 테스트 녹화를 만듭니다. **사용자 녹화 중에는 실행하지 마세요.** 모델 전환 검사 전에도 현재 녹화 상태를 확인합니다.

```bash
.venv/bin/python scripts/verify_live_models.py
.venv/bin/python scripts/verify_dataset.py
.venv/bin/python scripts/verify_video_continuity.py --seconds 90
```

`verify_live_models.py`는 모델을 순서대로 선택해 5종 MJPEG를 검사한 뒤 기존 설정을 복원합니다. 스트림 갱신만으로 관절 검출 정확도를 검증했다고 판단하지 마세요.

`verify_dataset.py`는 5초 준비＋약 8초 녹화, 원본 IMU 대조, CSV/영상 다운로드와 디코딩 등을 검사합니다. 종료 코드 **0**은 실제 관절 포함 통과, **2**는 센서/CSV/영상은 통과했지만 실제 관절이 없는 부분 검증, **1**은 실패입니다. 코드2이면 손을 실제로 비춘 뒤 다시 검증하고, 부분 검증을 전체 성공으로 기록하지 마세요.

`verify_video_continuity.py --seconds 90`은 장시간 프레임 진행·지연·영상/CSV 누락·6영상 재생을 검사하고 `validation/video_continuity_report.json`을 생성합니다. 이 검사가 통과해도 손이 없으면 관절 검출을 새로 검증한 것은 아닙니다. IMU 영상 기능을 다시 켜 `verify_imu_video.py`를 통과시키려 하지 마세요.

실제 브라우저에서도 모델 변경, 5초 카운트다운, 녹화 완료 링크, 4분할 영상 끝까지 재생, CSV의 `imu_accel`/`imu_gyro`/`pose_joint`를 확인하세요. 새 장치 결과와 [과거 기준기 검증](docs/validation.md)을 구분해 보고합니다.

## 저장 파일과 데이터셋 가공

녹화는 `recordings/<recording-id>/`, 스냅샷은 `captures/<capture-id>/`, 비교는 `comparisons/<comparison-id>/`에 저장합니다. 브라우저를 닫아도 녹화는 이어지므로 녹화 정지 버튼으로 마무리하고 완료를 기다려 다운로드하세요.

| 파일 | 내용 |
|---|---|
| `combined.mp4` | 4분할 1280×960·15fps |
| `rgb.mp4`, `depth.mp4`, `rgbd.mp4` | 개별 640×480·15fps 표시 영상 |
| `pose.mp4`, `pose_rgbd.mp4` | 추론한 동일 원본＋골격 영상 |
| `dataset.csv` | UTF-8 BOM, camera / imu_accel / imu_gyro / pose_frame / pose_joint |
| `manifest.json` | 시각, 설정, 보정값, 파일, 누락 수·처리 시간 |
| `frames.jsonl` | MP4와 실제 RGB/Pose 원본, 반복·빈 구간 매핑 |
| `sensor.jsonl`, `poses.jsonl`, `events.jsonl` | 원본 정보, 관절, 설정 변경 등 이벤트 |

IMU는 비디오 15fps로 줄이지 않고 수신 원본 샘플마다 저장합니다. 가속도는 **m/s²(중력 포함)**, 각속도는 **rad/s**입니다. 위치·자세 quaternion은 생성하지 않습니다.

CSV 빈칸을 0으로 채우지 말고 하드웨어 시계 domain과 연결 세대를 보존하세요. Pose-camera는 `recording_id + source_generation + source_frame_id`로 대응합니다. CSV는 기록 순서이며 시간 정렬에는 `source_elapsed_s`를 씁니다. 전체 열 설명과 pandas 예시는 [CSV 사양](docs/dataset_csv.md)에 있습니다. pandas는 가공용 별도 환경에서 사용하세요.

Pose가 느리면 합본 4화면은 같은 원본 전체를 반복합니다. 모델 준비·전환·오래된 Pose 구간은 빈 화면과 JSONL 사유를 남깁니다. 새 RGB에 오래된 골격을 얹지 않습니다. 추론 주기의 짧은 반복과 달리, 긴 시간 원본이 진행하지 않거나 누락 수가 증가하면 점검해야 합니다.

## 문제 해결

| 증상 | 확인과 대처 |
|---|---|
| USB에 D435i 없음 | Jetson 연결 여부, USB3 데이터 케이블/포트, 직결, `lsusb`, `lsusb -t` |
| 권한 오류 | D435i 전용 udev, video 그룹, SSH 재접속, 카메라 재연결 |
| RGB-D는 나오지만 IMU 없음 | 4단계 RSUSB 빌드/import 경로와 Motion Module 확인. pip SDK import 성공만으로 IMU 성공 아님 |
| Depth가 검거나 누락 | USB 속도, 좌우 IR, 거리·반사·가림 확인. 0은 무효 깊이. `scripts/diagnose_depth.py`는 카메라를 중단/재개하므로 코드를 먼저 읽고 녹화가 없을 때만 실행 |
| 카메라 사용 중 | 다른 서버/Viewer 확인. 녹화 상태를 확인하지 않고 프로세스를 kill하지 않기 |
| TensorRT load 실패 | 현 Jetson/TensorRT에서 엔진 재생성, ONNX shape 확인 |
| H.264 녹화 실패 | `scripts/probe_recording_codec.py` 합성 검사로 avc1 open/15프레임 decode 확인. pip OpenCV 덮어쓰기 여부 확인 |
| EgoForce CPU/오류 | 전용 venv, NVIDIA wheel, 내부 cuSPARSELt, CUDA smoke check 확인. CPU torch로 교체 금지 |
| 손만 보이면 YOLO 실패 | MediaPipe Hand로 변경. EgoForce도 손 검출에 의존. 점 개수를 정확도로 해석하지 않기 |
| 긴 녹화 지연/멈춤 | dropped_total, CSV 누락, performance, 디스크, tegrastats 확인. IMU 영상 비활성 유지 |
| PC 페이지 접속 실패 | Jetson curl, SSH 터널 유지, PC 8766 포트 충돌, 실제 SSH 접속 대상 확인 |

직접 SDK를 여는 진단 도구는 서버와 카메라 점유가 충돌할 수 있습니다. 스크립트 help/docstring을 먼저 읽고 녹화 중인 카메라를 중단하지 마세요.

## 소스와 백업 범위

| 경로 | 역할 |
|---|---|
| `server.py`, `prototype.py` | HTTP/API, 제어, 렌더링·스냅샷 |
| `camera.py`, `motion.py` | RGB-D/IMU 수집, 재연결·이력 |
| `pose_runtime.py`, `live_*.py`, `comparison/` | TensorRT와 모델별 worker·비교 |
| `recording.py`, `dataset_csv.py` | 5초 대기, 영상 큐, CSV |
| `static/` | 실시간/비교 UI |
| `scripts/`, `tests/` | 설치·모델 빌드·진단·실기 검증 / 회귀 검사 |
| `imu_video.py` | 이전 IMU 영상 연구용 렌더러. 활성 녹화 경로에서 호출하지 않음 |
| `models/yolov8n-pose.json` | 백업 당시 모델 참조 메타데이터. 실제 모델은 재생성 |

`.gitignore`는 venv·SDK 빌드·외부 소스·가중치·엔진·로그·촬영/녹화/비교 데이터·검증 생성물을 제외합니다. **수집 소프트웨어의 백업이며 데이터셋 자체의 백업은 아닙니다.** 데이터셋은 MP4·CSV·JSONL·manifest를 세션 폴더 단위로 별도 보관하세요. 원본 사진과 실제 관절 좌표도 Git에는 포함하지 않았습니다.

모델별 배포 조건은 [Ultralytics](https://www.ultralytics.com/license), [MediaPipe](https://github.com/google-ai-edge/mediapipe), [RTMPose/MMPose](https://github.com/open-mmlab/mmpose), [EgoForce](https://github.com/dfki-av/EgoForce)를 확인하세요. 특히 EgoForce 공식 코드·가중치는 CC-BY-NC-4.0입니다. 본 프로젝트 독자 코드의 배포 라이선스는 미지정이며 제3자 모델의 상업적 이용·재배포를 일괄 허가하지 않습니다.
