"""Export the official YOLOv8n COCO pose checkpoint to a static raw ONNX.

Run in an export-only environment containing ultralytics==8.3.230 and
onnx==1.17.0 plus a compatible CPU torch/torchvision pair. The Jetson inference
application does not need those export dependencies. No packages are installed
by this script; missing requirements fail explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "yolov8n-pose.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "yolov8n-pose.onnx")
    parser.add_argument("--input-size", type=int, default=640)
    args = parser.parse_args()
    if args.input_size < 32 or args.input_size % 32:
        parser.error("--input-size must be a positive multiple of 32")
    weights = args.weights.expanduser().resolve()
    output = args.output.expanduser().resolve()
    weights.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".export-cache"))
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    if not weights.is_file():
        if weights.name != "yolov8n-pose.pt":
            raise FileNotFoundError(f"Custom checkpoint does not exist: {weights}")
        temporary = weights.with_suffix(".pt.download")
        urllib.request.urlretrieve(CHECKPOINT_URL, temporary)
        temporary.replace(weights)

    from ultralytics import YOLO
    import onnx
    import torch

    model = YOLO(str(weights), task="pose")
    # Explicit options keep this compatible with OpenCV DNN and TensorRT 10.3.
    exported = Path(model.export(
        format="onnx", imgsz=args.input_size, opset=13, simplify=False,
        dynamic=False, nms=False, batch=1, device="cpu", half=False,
    )).resolve()
    if exported != output:
        shutil.copy2(exported, output)
    graph = onnx.load(str(output))
    onnx.checker.check_model(graph)
    input_shape = [d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [d.dim_value for d in graph.graph.output[0].type.tensor_type.shape.dim]
    expected_input = [1, 3, args.input_size, args.input_size]
    if input_shape != expected_input or len(output_shape) != 3 or output_shape[:2] != [1, 56]:
        raise RuntimeError(f"Unexpected pose tensor shapes: input={input_shape}, output={output_shape}")
    metadata = {
        "model": "yolov8n-pose", "checkpoint_url": CHECKPOINT_URL,
        "opset": 13, "input_shape": input_shape, "output_shape": output_shape,
        "input_dtype": "float32", "keypoints": 17, "layout": "COCO",
        "nms": False, "dynamic": False,
        "torch_version": torch.__version__,
        "onnx_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "license": "Ultralytics AGPL-3.0 / Enterprise; review before redistribution",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"onnx": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
