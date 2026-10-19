#!/usr/bin/env python3
"""Download official OpenMMLab RTMPose-m COCO17 and build a Jetson TRT plan."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"


def download(destination):
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "model.zip"
    if not archive.is_file():
        request = urllib.request.Request(URL, headers={"User-Agent": "RealSensePosePrototype/1.0"})
        temporary = archive.with_suffix(".zip.part")
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as source:
        for member in source.namelist():
            name = Path(member).name
            if name in ("end2end.onnx", "deploy.json", "pipeline.json", "detail.json"):
                # Flatten known names; never extract untrusted relative paths.
                (destination / name).write_bytes(source.read(member))
    onnx = destination / "end2end.onnx"
    metadata = {"model": "RTMPose-m Body7 COCO17", "source": URL,
                "sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
                "input_shape": [1, 3, 256, 192], "keypoint_count": 17,
                "detector": "bypassed; full image or selected ROI"}
    (destination / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return onnx


def build(onnx, workspace_mib=512):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx.resolve())):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    if network.num_inputs != 1 or network.num_outputs != 2:
        raise ValueError("Expected one RTMPose input and two SimCC outputs")
    image = network.get_input(0)
    expected = (1, 3, 256, 192)
    if len(image.shape) != 4 or any(a > 0 and a != b for a, b in zip(image.shape, expected)):
        raise ValueError(f"Unexpected RTMPose input: {image.shape}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 1024 * 1024)
    if any(v <= 0 for v in image.shape):
        profile = builder.create_optimization_profile()
        profile.set_shape(image.name, expected, expected, expected)
        config.add_optimization_profile(profile)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    started = time.monotonic()
    print(f"Building RTMPose TensorRT {trt.__version__}: {image.shape} -> " +
          str([tuple(network.get_output(i).shape) for i in range(network.num_outputs)]), flush=True)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("RTMPose TensorRT build failed")
    output = onnx.with_suffix(".engine")
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(bytes(plan))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    print(f"Built {output} in {time.monotonic() - started:.1f}s", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "models" / "rtmpose")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--workspace-mib", type=int, default=512)
    args = parser.parse_args()
    if args.workspace_mib <= 0:
        parser.error("--workspace-mib must be positive")
    model = download(args.destination)
    if not args.download_only:
        build(model, args.workspace_mib)
