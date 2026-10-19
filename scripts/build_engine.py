#!/usr/bin/env python3
"""Build a raw TensorRT 10 plan on the Jetson that will run inference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
import time


ROOT = Path(__file__).resolve().parent.parent


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("양수를 입력하세요.")
    return result


def build(onnx_path: Path, output_path: Path, workspace_mib: int = 512) -> None:
    import tensorrt as trt

    if workspace_mib <= 0:
        raise ValueError("workspace_mib must be positive")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX 모델이 없습니다: {onnx_path}")
    if onnx_path.resolve() == output_path.resolve():
        raise ValueError("ONNX 입력과 엔진 출력은 다른 파일이어야 합니다.")
    if int(trt.__version__.split(".", 1)[0]) < 10:
        raise RuntimeError(f"TensorRT 10 이상이 필요합니다. 현재 버전: {trt.__version__}")
    logger = trt.Logger(trt.Logger.WARNING)
    if not trt.init_libnvinfer_plugins(logger, ""):
        raise RuntimeError("TensorRT 플러그인을 초기화하지 못했습니다.")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path.resolve())):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX 모델을 해석하지 못했습니다:\n{errors}")
    if network.num_inputs != 1:
        raise ValueError(f"단일 입력 모델이 필요합니다. 입력 수: {network.num_inputs}")
    input_tensor = network.get_input(0)
    shape = tuple(int(v) for v in input_tensor.shape)
    if shape != (1, 3, 640, 640):
        raise ValueError(f"고정 입력 (1, 3, 640, 640)이 필요합니다. 현재 입력: {shape}")
    if network.num_outputs != 1:
        raise ValueError("YOLOv8 pose의 단일 raw 출력 모델이 필요합니다. NMS 없이 내보내세요.")
    output_shape = tuple(int(v) for v in network.get_output(0).shape)
    if len(output_shape) != 3 or output_shape[:2] != (1, 56) or output_shape[2] <= 0:
        raise ValueError(f"raw COCO pose 출력 (1, 56, anchors)이 필요합니다: {output_shape}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 1024 * 1024)
    use_fp16 = bool(builder.platform_has_fast_fp16)
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    precision = "FP16" if use_fp16 else "FP32"
    print(f"TensorRT {trt.__version__}: {shape} -> {output_shape}", flush=True)
    print(f"{precision} 엔진 빌드 시작, workspace {workspace_mib} MiB. 몇 분 걸릴 수 있습니다.", flush=True)
    started = time.monotonic()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT 엔진 빌드에 실패했습니다. 위의 TensorRT 로그를 확인하세요.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output_path.parent, prefix=output_path.name + ".", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(bytes(serialized))
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    elapsed = time.monotonic() - started
    print(f"완료: {output_path.resolve()} ({output_path.stat().st_size:,} bytes, {elapsed:.1f}초)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Jetson에서 YOLOv8n-pose TensorRT 엔진을 만듭니다.")
    parser.add_argument("--onnx", type=Path, default=ROOT / "models" / "yolov8n-pose.onnx")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "yolov8n-pose.engine")
    parser.add_argument("--workspace-mib", type=positive_int, default=512)
    args = parser.parse_args()
    try:
        build(args.onnx, args.output, args.workspace_mib)
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f"오류: {exc}\n")


if __name__ == "__main__":
    main()
