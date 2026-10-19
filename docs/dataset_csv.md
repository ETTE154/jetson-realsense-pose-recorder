# 센서·Pose 데이터셋 CSV

녹화 세션의 `dataset.csv`는 카메라, IMU, Pose를 **이벤트별 행**으로 저장합니다.
하나의 행에 모든 센서 값을 채우기 위해 보간하거나 마지막 값을 반복하지 않습니다.
가속도·자이로·카메라·Pose의 수신 주기가 다르므로, 원본 이벤트를 보존한 뒤 학습 목적에 맞춰 정렬·리샘플링하도록 구성했습니다.

CSV는 UTF-8 BOM 인코딩이며, 헤더가 포함됩니다. 현재 `schema_version`은 `1.0`입니다.
녹화 준비 중에는 `.partial.csv`로 쓰고, 정상 종료 후 `dataset.csv`로 공개합니다.
녹화 시작 후 5초 대기 구간에 수신한 원본은 제외합니다. Pose 추론이 늦게 끝나더라도 원본 카메라 프레임이 대기 구간에 속하면 제외합니다.

## 행 종류

| `record_type` | 한 행의 의미 | 주요 값 |
|---|---|---|
| `camera` | 캡처한 RGB·정렬 깊이 한 쌍 | 영상 크기, 카메라 보정값, 깊이 유효율·거리 통계, 원본 프레임 번호 |
| `imu_accel` | 가속도계의 독립 샘플 하나 | `accel_x_m_s2`, `accel_y_m_s2`, `accel_z_m_s2` |
| `imu_gyro` | 자이로의 독립 샘플 하나 | `gyro_x_rad_s`, `gyro_y_rad_s`, `gyro_z_rad_s` |
| `pose_frame` | 한 원본 영상에 대한 Pose 처리 결과 | 모델·실행 상태·추론 시간·검출 인스턴스 수 |
| `pose_joint` | 해당 처리 결과의 한 인스턴스에 속한 관절 하나 | 관절 이름·픽셀 좌표·가용한 신뢰도·유효한 깊이 기반 3D 좌표 |

카메라 행은 캡처 콜백에서 제출하므로 미리보기의 프레임 건너뛰기와 독립적입니다.
IMU 행도 영상 주기로 줄이지 않고 SDK 콜백의 각 샘플을 제출합니다.
저장 큐가 밀리거나 쓰기가 실패한 경우에는 상태·매니페스트의 누락·오류 카운터를 확인해야 합니다.
행이 없다는 이유만으로 해당 시점의 실제 움직임이 0이었다고 해석하면 안 됩니다.

## 공통 식별자와 시각

| 열 | 의미와 사용법 |
|---|---|
| `recording_id` | 녹화 세션 식별자. 여러 CSV를 합칠 때 유지합니다. |
| `row_sequence` | 파일 내 행 순번. 시간 순서라는 보장은 없습니다. |
| `event_index` | CSV 수신 이벤트 순번. 같은 Pose 이벤트의 `pose_frame`과 모든 `pose_joint`가 공유합니다. |
| `source_generation` | 카메라 캡처 연결 세대. 재연결·재시작 후 증가합니다. 다른 세대의 하드웨어 시각을 이어 붙이지 않습니다. |
| `source_frame_id` | 프로그램의 원본 RGB-D 프레임 ID. Pose의 원본 카메라 행을 찾을 때 사용합니다. IMU에서는 비어 있습니다. |
| `source_frame_number` | SDK 원본 프레임 번호. 영상에서는 컬러 번호, IMU에서는 해당 motion 스트림 번호입니다. |
| `source_clock` | `camera_color`, `imu_accel`, `imu_gyro` 중 하나. Pose는 원본 컬러 시각을 사용합니다. |
| `source_timestamp_ms` | SDK 원본 시각, 밀리초. `source_timestamp_domain`과 함께 읽습니다. |
| `source_timestamp_domain` | 예: `hardware_clock`, `system_time`, `global_time`. 서로 다른 도메인을 무조건 같은 시계로 취급하면 안 됩니다. |
| `source_received_unix_s` | 원본을 프로그램이 수신한 Unix 시각, 초. PC 벽시계 보정의 영향을 받을 수 있습니다. |
| `source_received_monotonic_s` | 원본의 서버 수신 monotonic 시각, 초. 동일 서버 프로세스의 경과 시간 정렬에 사용합니다. |
| `source_elapsed_s` | 원본 수신 monotonic 시각에서 5초 대기 후 녹화 시작 시각을 뺀 값, 초. |
| `event_received_monotonic_s` | CSV 저장기로 이벤트가 제출된 시각. Pose는 추론 후 제출되므로 원본 수신보다 늦습니다. |

CSV는 제출 순서로 쓰입니다. 느린 Pose 결과가 뒤늦게 도착할 수 있으므로 시간축을 만들 때는 `source_elapsed_s`를 정렬합니다.
Pose와 카메라의 정확한 원본 대응은 `recording_id`, `source_generation`, `source_frame_id`로 찾습니다.
동일 Pose 처리 결과의 관절들을 묶을 때는 `recording_id`, `event_index`를 사용합니다.

서버 수신 시각은 USB 전송·스케줄링 지연을 포함합니다. 이를 이용한 근접 시각 조인은 전처리를 위한 근사이며, 하드웨어 노출 시각의 정밀 동기화가 아닙니다.
카메라·IMU의 SDK 시각도 도메인과 연결 세대, 실제 시계 관계를 확인한 뒤 비교해야 합니다.

## 카메라·깊이

- `image_width_px`, `image_height_px`: 정렬된 영상 크기, 픽셀.
- `color_frame_number`, `depth_frame_number`: SDK 컬러·깊이 번호.
- `depth_timestamp_ms`, `depth_timestamp_domain`: 원본 깊이 프레임의 별도 시각과 도메인.
- `depth_scale_m_per_unit`: Z16 원시 정수 한 단위에 해당하는 미터 값.
- `depth_total_pixels`, `depth_valid_pixels`, `depth_valid_fraction`: 전체 픽셀, 0보다 큰 깊이 픽셀, 그 비율.
- `depth_min_m`, `depth_max_m`, `depth_mean_m`, `depth_median_m`: 0보다 큰 깊이값만으로 계산한 거리 통계, 미터.
- `camera_fx_px`, `camera_fy_px`, `camera_ppx_px`, `camera_ppy_px`: 정렬 대상인 컬러 카메라의 내부 보정값, 픽셀.
- `camera_distortion_model`, `camera_distortion_coeff_0`…`4`: SDK 왜곡 모델·계수.

깊이 통계는 화면의 표시 범위 설정으로 잘라 계산하지 않습니다.
`setting_depth_display_min_m`, `setting_depth_display_max_m`은 색상 표시 범위입니다.
`depth_valid_fraction`이 높더라도 움직임·재질·정렬 오차가 없다는 보장은 없습니다.

CSV에는 RGB 픽셀이나 원본 Z16 깊이 배열 전체를 넣지 않습니다.
`raw_depth_file`이 비어 있으면 그 카메라 프레임의 연속 원본 깊이 파일이 저장됐다는 뜻이 아닙니다.
깊이 색상 MP4에서 Z16 원본 거리값을 복원할 수 없습니다.
픽셀 단위 RGB-D 학습에 원본 깊이 전체가 필요하면 별도의 무손실 깊이 저장 형식을 사용해야 합니다.

## IMU

`imu_accel`은 m/s², `imu_gyro`는 rad/s 단위입니다.
가속도는 중력 성분을 포함한 센서 측정값이며, 중력 제거·적분·자세 융합을 수행하지 않습니다.
`imu_sample_id`는 프로그램 수명 중 증가하는 IMU 샘플 ID입니다.
`imu_type`, `imu_units`, 가용한 `imu_profile_fps`와 `device_serial`도 함께 읽습니다.
프로파일 FPS가 비어 있으면 샘플 간 원본 시각 차이와 UI의 실제 수신률을 확인합니다.

가속도 행의 자이로 열과 자이로 행의 가속도 열은 비어 있습니다.
이것은 해당 측정값이 0이라는 뜻이 아닙니다.
프로그램은 위치, 속도, 쿼터니언, 절대 방위를 생성하지 않습니다.

IMU XYZ는 SDK motion 스트림의 좌표계입니다.
현재 CSV는 IMU와 영상 간 외부 보정 행렬·IMU 잡음/바이어스 보정 모델을 제공하지 않습니다.
따라서 IMU XYZ를 그대로 컬러 카메라 XYZ와 같은 축이라고 가정하면 안 됩니다.
영상·관성 융합 학습 전에 장치 외부 보정과 시간 보정이 필요합니다.

## Pose

`pose_model_id`, `pose_model`, `pose_backend`, `pose_state`, `pose_inference_ms`로 처리 조건을 구분합니다.
`pose_frame`의 `pose_instance_count=0`은 그 이벤트에 인스턴스가 없음을 나타냅니다.
Pose 활성화 여부와 상태도 함께 확인합니다. 행 자체가 없는 구간을 미검출 결과로 채우지 않습니다.

`instance_id`의 기본 유효 범위는 한 원본 프레임입니다(`instance_id_scope=source_frame`).
다음 프레임에서 같은 번호가 나온다고 같은 사람·손이라는 뜻이 아닙니다.
연속 추적 ID는 모델이 `track_id`를 제공할 때만 사용합니다.

| 열 | 의미 |
|---|---|
| `instance_label`, `handedness` | 인스턴스 이름·가용한 왼손/오른손 분류 |
| `instance_score`, `instance_score_kind` | 모델이 보고한 점수와 그 종류 |
| `handedness_score` | 왼손/오른손 분류 점수. 관절 검출 확률과 구분합니다. |
| `bbox_left_px`…`bbox_bottom_px` | 원본 컬러 이미지의 경계 상자, 픽셀 |
| `joint_index`, `joint_name` | 모델별 관절 순서·이름. 모델마다 관절 구성이 다릅니다. |
| `joint_u_px`, `joint_v_px` | 원본 컬러 이미지의 2D 위치, 픽셀 |
| `joint_confidence`, `joint_confidence_kind` | 관절 점수와 종류. 모델이 보고하지 않으면 비어 있습니다. |
| `joint_displayed` | 표시 임계값 등에 의해 화면에 표시한 관절인지 여부 |
| `joint_depth_valid` | 유효한 깊이 기반 XYZ를 얻었는지 여부 |
| `joint_x_m`, `joint_y_m`, `joint_z_m` | 깊이 기반 3D 위치, 미터 |
| `joint_depth_source` | 3D 위치를 얻은 방법 |
| `joint_coordinate_frame` | 컬러 광학 좌표계: X 오른쪽, Y 아래, Z 전방 |
| `joint_native_solver_weight` | 가용한 모델 내부 solver weight. 검출 확률로 취급하지 않습니다. |

깊이 기반 관절 XYZ는 해당 픽셀 근처의 D435i 표면 깊이로 구한 근사값입니다.
관절의 해부학적 중심이나 모션 캡처 정답 좌표가 아닙니다.
가려진 관절은 배경의 깊이를 읽을 수도 있습니다.
`joint_depth_valid=0`이면 XYZ는 비어 있으며, 0 m 좌표로 채우지 않습니다.
`joint_confidence_kind=not_reported_display_mask`인 경우 표시 마스크를 신뢰도 점수로 내보내지 않습니다.

`setting_pose_enabled`, `setting_confidence_threshold`, `setting_roi_*`와 깊이 표시 설정은 제출 시점의 처리 조건입니다.
비교 학습에서는 모델·관절 체계·ROI·점수 종류가 다른 구간을 구분합니다.

## 영상 파일과 원본 대응

CSV의 `rgb_video_file`, `combined_video_file`, `video_mapping_file`은 같은 녹화 폴더 안의 파일을 가리킵니다.
`frames.jsonl`은 MP4의 각 인코딩 프레임을 원본 카메라·Pose 프레임에 연결합니다.
`frame_index`는 0부터 시작하며, `t_s`는 영상 시간, `sensor_source`, `pose_source`, `combined_source`는 각 출력의 원본입니다.

원본 카메라는 목표 30fps, MP4는 15fps입니다. 따라서 CSV의 모든 카메라 행에 대응하는 MP4 프레임이 있지는 않습니다.
반대로 Pose가 느리면 같은 원본이 영상 여러 프레임에서 유지될 수 있습니다.
`sensor_held`, `pose_held`, `combined_held`와 gap 필드를 확인하고, 같은 원본을 서로 다른 정답 샘플로 중복 학습하지 않도록 처리합니다.
4분할 영상은 네 타일이 같은 원본에 대응하도록 구성되므로 RGB 개별 영상보다 오래된 Pose 원본을 유지할 수 있습니다.
영상 프레임 번호에 2를 곱해 카메라 ID로 추정하지 말고 JSONL 매핑을 사용합니다.

IMU 그래프를 포함하는 **이전 버전** `combined.mp4`는 RGB-D 원본 수신 시각을 기준으로 표시합니다.
`frames.jsonl`의 `combined_imu`는 해당 원본 시각까지의 실제 샘플과 상태를 나타냅니다.
별도 `imu.mp4`는 공통 CFR 재생 시각을 사용하며 `imu_video`에 대응합니다.
느린 Pose 때문에 합본 원본이 반복되면 IMU 패널도 해당 원본 시점으로 유지됩니다.
원본 IMU 데이터셋에는 시각화 영상의 15fps 값을 사용하지 말고 CSV의 독립 측정 행을 사용합니다.
현재 버전에서는 녹화 연속성 확보를 위해 IMU 영상 저장을 해제했으며, CSV의 IMU 원본 측정과 실시간 UI 그래프는 계속 제공합니다.

## pandas 전처리 예시

다음 예시는 원본 이벤트를 보존해서 읽고, 유효한 깊이 관절을 선택합니다.

```python
from pathlib import Path
import pandas as pd

folder = Path("recordings/녹화_세션_ID")
df = pd.read_csv(folder / "dataset.csv", encoding="utf-8-sig")
camera = df.loc[df.record_type.eq("camera")].copy()
gyro = df.loc[df.record_type.eq("imu_gyro")].copy()
accel = df.loc[df.record_type.eq("imu_accel")].copy()
joint = df.loc[df.record_type.eq("pose_joint")].copy()

# 빈 XYZ를 0으로 채우지 않습니다.
joint = joint.loc[joint.joint_depth_valid.eq(1)].copy()
join_keys = ["recording_id", "source_generation", "source_frame_id"]
joint_with_camera = joint.drop(columns=["depth_valid_fraction"]).merge(
    camera[join_keys + ["depth_valid_fraction"]],
    on=join_keys, how="left", validate="many_to_one",
    suffixes=("", "_camera"),
)
```

서버 수신 시각을 이용해 각 카메라 프레임에 가장 가까운 자이로 샘플을 붙이는 예시입니다.
서로 다른 세대를 섞지 않으며, 허용 시차 밖의 값은 NaN으로 남깁니다.

```python
import numpy as np

keys = ["recording_id", "source_generation"]
matched_parts = []
for group_key, video_group in camera.groupby(keys, dropna=False):
    imu_group = gyro.loc[
        gyro.recording_id.eq(group_key[0])
        & gyro.source_generation.eq(group_key[1])
    ]
    left = video_group.sort_values("source_elapsed_s").copy()
    cols = ["source_elapsed_s", "imu_sample_id",
            "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s"]
    right = imu_group[cols].sort_values("source_elapsed_s").rename(
        columns={"source_elapsed_s": "gyro_source_elapsed_s"}
    )
    # camera 행의 같은 이름의 빈 IMU 열을 제거합니다.
    left = left.drop(columns=cols[1:], errors="ignore")
    if right.empty:
        for name in right.columns:
            left[name] = np.nan
        matched = left
    else:
        matched = pd.merge_asof(
            left, right, left_on="source_elapsed_s",
            right_on="gyro_source_elapsed_s",
            direction="nearest", tolerance=0.020,
        )
    matched["gyro_match_delta_s"] = (
        matched.gyro_source_elapsed_s - matched.source_elapsed_s
    )
    matched_parts.append(matched)

camera_with_gyro = pd.concat(matched_parts, ignore_index=True) if matched_parts else pd.DataFrame()
```

`tolerance=0.020`은 이 예시에서 선택한 20ms 허용값입니다. 실제 동작·지연 분포에 맞춰 결정합니다.
양방향 근접 조인은 미래 샘플을 선택할 수 있으므로, 실시간 예측 학습에서는 `direction="backward"` 등 인과 조건을 검토합니다.
가속도와 자이로를 같은 주기로 변환할 때도 누락·재연결·긴 공백을 넘겨 보간하지 않습니다.

영상 매핑은 다음과 같이 원본 ID를 추출할 수 있습니다.

```python
import json

with (folder / "frames.jsonl").open(encoding="utf-8") as handle:
    mapping = [json.loads(line) for line in handle]
video_map = pd.DataFrame([
    {
        "video_frame_index": item["frame_index"],
        "video_t_s": item["t_s"],
        "source_frame_id": (item.get("combined_source") or {}).get("frame_id"),
        "source_received_monotonic_s": (item.get("combined_source") or {}).get("received_monotonic"),
        "held": item.get("combined_held"),
        "gap": item.get("combined_pose_gap"),
    }
    for item in mapping
])
```

데이터셋 배포 시 CSV만 옮기면 픽셀 입력·영상 매핑·설정 이력을 잃을 수 있습니다.
필요한 MP4, `frames.jsonl`, `manifest.json`을 녹화 세션 폴더와 함께 보관합니다.
