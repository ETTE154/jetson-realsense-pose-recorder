# EgoForce 비교 구현 범위

이 문서의 실측값은 기준 장치에서 얻은 과거 기록입니다. 언급된 `captures/`와
`validation/`의 사용자 영상·실좌표·생성 보고서는 GitHub에 포함하지 않습니다.
새 장치에서는 직접 스냅샷을 촬영하고 아래 명령의 `<capture-id>`를 실제 값으로
바꿔 검증하세요. 현재 설치 순서와 검증 요약은 [README](README.md)와
[기준 장치 검증 기록](docs/validation.md)에 있습니다.

이 비교 화면의 **EgoForce HALO · 2D / 손 영역 공유**는 [공식 EgoForce 저장소](https://github.com/dfki-av/EgoForce)의 HALO 신경망과 [공식 공개 가중치](https://huggingface.co/chris10/EgoForce)를 실제로 실행합니다. 손가락·손목 21개 위치를 출력하고, 사용자가 영상에 보이는 전완 영역을 지정하면 해당 영역에서 전완 3개 위치도 출력합니다.

이 모드는 RGB 이미지 위에서 손 관절 추정 결과를 비교하기 위한 구현입니다. 공식 전체 데모의 RTMDet·YOLO 검출기와 MANO 메쉬, 카메라 공간 3D 복원 과정은 포함하지 않습니다. 대신 첫 번째 후보인 MediaPipe가 찾은 손 영역을 HALO 입력으로 공유합니다. 따라서 화면의 EgoForce 결과는 독립적인 손 검출 성능을 평가한 결과가 아닙니다. 보이지 않는 팔의 생성 추정점은 표시하거나 깊이 측정값으로 저장하지 않습니다.

## 입력 영역

- MediaPipe가 손을 찾은 경우: 해당 손 바운딩 박스를 1.25배 확장해 원래 RGB 이미지에서 손 crop을 만듭니다. RGB 카메라의 실제 내부 파라미터와 공식 `DemoHandArmLoader`로 crop 위치를 인코딩합니다.
- 이때 사용자가 영역을 추가로 지정하면: 그 영역을 영상에 **실제로 보이는 전완** crop으로 사용합니다. 어깨는 출력하지 않습니다.
- MediaPipe가 손을 찾지 못했고 수동 영역이 있으면: 수동 영역을 손 crop으로 해석합니다. 손이 들어 있는 작은 영역을 지정해야 합니다. 이 경우 전완점은 출력하지 않습니다.
- 손 검출과 수동 영역이 모두 없으면: 결과를 0개로 반환하고 손 영역이 필요하다는 상태를 표시합니다.

영역 지정 결과는 `metadata.crop_details`에 기록됩니다. 모든 출력 좌표는 원본 RGB 이미지의 2D 픽셀 좌표입니다. D435i 깊이를 결합한 3D 표면 좌표는 비교 서버에서 별도로 계산하며 EgoForce의 단안 3D 복원 결과와 구분합니다.

## 환경과 공개 자산

가중치는 3,021,671,106바이트입니다. 프로젝트 디렉터리 안에 저장하고 다운로드 후 SHA-256을 검증합니다. 공식 설치 스크립트는 실행하지 않습니다.

```bash
python scripts/setup_egoforce.py --download
```

다운로드 위치는 `models/egoforce/`입니다. `manifest.json`에 소스 커밋, 모델 저장소 revision, 다운로드 크기와 해시를 기록합니다. 체크포인트는 PyTorch의 `weights_only=True`, `mmap=True`로 읽고 모든 state key를 `strict=True`로 검사합니다. 공식 소스는 렌더러를 가져오는 패키지 초기화만 건너뛰고 실제 네트워크 연산과 crop loader를 그대로 사용합니다.

현재 배포는 **GPU 필수**입니다. `.egoforce-gpu-venv`의 NVIDIA Jetson용 PyTorch로 HALO를 CUDA FP16에서 실행합니다. CUDA가 없으면 명시적으로 실패하며 CPU로 자동 전환하지 않습니다. 이전 CPU 검증용 `.egoforce-venv`와 시스템 환경은 보존합니다. 공식 전체 데모의 Torch-TensorRT·MMCV·PyTorch3D와 호환되는 환경을 설치했다고 간주하지 않습니다. 첫 실행에는 가중치 검증과 신경망 초기화 시간이 추가됩니다. 한 번에 손 하나씩 처리하고 CPU 전처리 스레드를 기본 4개로 제한합니다.

```bash
bash scripts/setup_egoforce_gpu.sh
.egoforce-gpu-venv/bin/python scripts/check_egoforce_cuda.py
```

GPU 설치는 [NVIDIA의 Jetson PyTorch 안내](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)와 [NVIDIA v61 공식 배포 파일](https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/)을 사용합니다. 이 장치의 JetPack 6.2.1 / CUDA 12.6에서 `torch 2.5.0a0+872d972e41.nv24.08`, Orin SM 8.7의 실제 FP16 CUDA 행렬 연산을 확인했습니다. torchvision 0.20.0은 [공식 소스](https://github.com/pytorch/vision/tree/v0.20.0)에서 이 PyTorch의 ABI에 맞춰 빌드합니다. torchvision의 유틸리티 연산은 CPU 빌드이며 HALO 신경망은 GPU에서 실행됩니다. MediaPipe의 손 영역 검출도 CPU에서 실행합니다.

필요한 cuSPARSELt 0.6.2.3은 NVIDIA 공식 압축 파일을 프로젝트 안에만 풉니다. 어댑터는 이 파일을 절대 경로로 미리 로드하므로 시스템 `ldconfig`나 `/usr/local/cuda`를 변경하지 않습니다. 다운로드 wheel·cuSPARSELt·torchvision 소스 압축 파일은 고정 SHA-256으로 검증합니다. 설치 후 CUDA 검증 및 아티팩트 출처는 `models/egoforce/runtime/gpu_runtime_manifest.json`에 기록합니다.

고정한 자산은 다음과 같습니다.

- 소스: `480ffb358516d0d7f971ec9da99b7bb07635731e`
- 모델: `5a57b4026babc5b35cb53627c2e1c709e1276b65`
- 체크포인트 SHA-256: `722fe84ba2b6ab3569c3a95000c61a9a77c7b3a41cc76585f891b8bc4fe82870`
- 라이선스: 공식 코드·가중치 **CC-BY-NC-4.0**. 상업적 제품 적용 시 해당 조건을 검토해야 합니다.

RGB 입력과 내부 파라미터 해상도는 같아야 합니다. 현재 어댑터는 왜곡 계수가 0인 보정된 pinhole RGB를 지원합니다. 현재 D435i 저장 영상은 계수가 모두 0입니다. 왜곡이 남은 영상은 부정확한 좌표를 그대로 출력하는 대신 지원 범위 오류를 표시합니다.

## 저장 영상 직접 검증

```bash
.egoforce-gpu-venv/bin/python scripts/verify_egoforce.py \
  --image captures/<capture-id>/rgb.jpg \
  --metadata captures/<capture-id>/metadata.json \
  --roi 244,280,343,459 \
  --output validation/egoforce_check.json
```

`--roi`는 원본 영상에서 직접 확인한 손 영역으로 바꿉니다. `--hand-hints`에 MediaPipe 결과 JSON을 지정하면 손 영역을 공유할 수 있습니다. 이때 `--roi`는 전완 영역이 됩니다. 도구는 원시 결과 JSON과 같은 이름의 JPEG 오버레이를 저장합니다.

HALO가 출력하는 `hand_kpt_w`·`arm_kpt_w`는 3D 위치 계산에도 사용되는 학습된 가중치이며 다른 후보의 검출 확률과 보정된 공통 점수가 아닙니다. 낮은 가중치가 반드시 손 검출 실패를 뜻하지 않습니다. 이 값은 `native_solver_weights`와 crop 메타데이터에 원본 그대로 보관합니다. 화면 표시용 관절 배열의 세 번째 값은 1.0인 표시 마스크이며 검출 신뢰도를 뜻하지 않습니다. 이는 점별 신뢰도를 제공하지 않는 MediaPipe와 같은 처리입니다. 정답 관절 주석이 없는 사용자 영상에서는 출력 점 개수와 지연 시간을 측정하며 정확도 백분율을 산출하지 않습니다.

## GPU 실측

같은 Jetson Orin NX에서 공개 HALO 가중치와 원본 D435i RGB 두 장을 CUDA FP16으로 실행했습니다. 각 입력을 한 번 워밍업한 뒤 3회 측정했습니다. 실제 파라미터가 `cuda:0`, `torch.float16`인 것을 검사했고 모든 손 좌표와 solver weight가 유한한 값임을 확인했습니다.

| 입력 | HALO 전체 처리 | CUDA 이벤트로 측정한 GPU 구간 |
|---|---:|---:|
| 첫 손 영상 | 평균 95.97ms | 평균 90.38ms |
| 두 번째 손 영상 | 평균 79.95ms | 평균 74.82ms |

이 시간에는 별도의 MediaPipe 손 검출 시간이 포함되지 않습니다. CPU에서 MediaPipe로 손 영역을 찾고 GPU에서 HALO를 실행하는 전체 실시간 처리 속도는 화면에 따로 표시합니다. 측정 당시 카메라 서버와 torchvision CPU 빌드가 동작 중이어서 통제된 성능 벤치마크는 아닙니다. 새 프로세스 모델 초기화는 약 15.2초, CUDA 최대 할당 메모리는 해당 검증에서 약 2178MiB였습니다. 첫 추론에는 cuDNN의 일회성 커널 선택 시간이 추가될 수 있습니다.

두 번째 영상의 동일한 공유 손 영역에서 기존 CPU FP32 출력과 비교한 21개 2D 좌표 차이는 평균 0.0404픽셀, 최대 0.1526픽셀이었습니다. 이는 GPU 변환 전후의 수치 정합 확인이며 실제 관절 위치에 대한 정확도 평가가 아닙니다. 전완 영역을 추가하는 CUDA FP16 24점 분기도 정상 출력과 유한성을 확인했습니다.

원시 결과와 실제 오버레이는 `validation/egoforce_gpu_verified.json`, `egoforce_gpu_verified_1.jpg`, `egoforce_gpu_verified_2.jpg`, `egoforce_gpu_arm24.*`에 있습니다. 재검증은 다음 명령으로 실행합니다.

```bash
.egoforce-gpu-venv/bin/python scripts/verify_egoforce_gpu.py \
  --capture captures/<capture-id-1> \
  --capture captures/<capture-id-2> \
  --repeats 3 --output validation/egoforce_gpu_verified.json
```

## 이전 CPU 구성의 Jetson 실측

2026-10-18 Jetson Orin NX의 별도 환경에서 `PyTorch 2.8.0+cpu`, CPU 스레드 4개로 공식 HALO와 사용자 원본 D435i RGB 영상 두 장을 실행했습니다. 기존 카메라·GPU Pose 서버가 켜져 있는 상태이며 통제된 성능 벤치마크는 아닙니다.

| 입력 | 손 영역 공급 | 출력 | 모델 처리 시간 |
|---|---|---|---|
| 첫 손 입력 | 수동 손 영역 | 손 21점 | 3.79초 |
| 두 번째 손 입력 | MediaPipe 손 영역 | 손 21점 | 3.88초 |
| 두 번째 영상 + 전완 영역 | MediaPipe 손 영역 + 수동 전완 | 손·전완 24점 | 3.96초 |

가중치 검증과 모델 초기화는 각 새 프로세스에서 약 12~13초였습니다. 비교 서버는 모델 프로세스를 유지해 후속 비교에서 초기화를 반복하지 않습니다. 결과 JSON과 실제 골격 오버레이는 `validation/compare_egoforce_jetson_sample1.*`, `sample2.*`, `arm24.*`에 보관했습니다. 손 골격 출력은 시각적으로 확인했으며 가려진 손가락과 전완 관절의 실제 위치 정확도는 별도 정답 데이터로 검증하지 않았습니다. 이 CPU 구성은 캡처 후 비교용이며 30fps 실시간 구성은 아닙니다.
