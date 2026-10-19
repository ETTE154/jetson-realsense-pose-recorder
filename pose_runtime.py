"""YOLOv8 pose inference without PyTorch or PyCUDA.

TensorRT engines must expose one static NCHW image tensor and one raw
COCO-pose output tensor (1, 56, anchors). Use an ONNX model for CPU fallback.
The 17 keypoints follow COCO order; coordinates are in the original RGB frame.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    """Exact rounded resize and integer padding used to form a model image."""

    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    top: int
    left: int

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.original_width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.original_height


def letterbox_geometry(height: int, width: int, size: int) -> LetterboxTransform:
    if min(height, width, size) <= 0:
        raise ValueError("Image dimensions and model input size must be positive")
    scale = min(size / width, size / height)
    resized_width = max(1, min(size, round(width * scale)))
    resized_height = max(1, min(size, round(height * scale)))
    return LetterboxTransform(
        height, width, resized_height, resized_width,
        (size - resized_height) // 2, (size - resized_width) // 2,
    )


def prepare_image(bgr: np.ndarray, size: int) -> tuple[np.ndarray, LetterboxTransform]:
    """BGR uint8 HWC -> RGB float32 NCHW, with centered 114 padding."""
    import cv2

    if not isinstance(bgr, np.ndarray) or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image with shape (height, width, 3)")
    if bgr.dtype != np.uint8:
        raise ValueError("Expected a uint8 BGR image")
    transform = letterbox_geometry(bgr.shape[0], bgr.shape[1], size)
    resized = cv2.resize(
        bgr, (transform.resized_width, transform.resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[
        transform.top:transform.top + transform.resized_height,
        transform.left:transform.left + transform.resized_width,
    ] = resized
    tensor = padded[:, :, ::-1].transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor, dtype=np.float32) / np.float32(255.0), transform


def non_max_suppression(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45,
    max_detections: int = 20,
) -> list[int]:
    """Class-agnostic IoU NMS for the single person class."""
    if boxes.size == 0 or max_detections <= 0:
        return []
    order = np.argsort(-scores, kind="stable")
    area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size and len(keep) < max_detections:
        index = int(order[0])
        keep.append(index)
        rest = order[1:]
        if not rest.size:
            break
        left_top = np.maximum(boxes[index, :2], boxes[rest, :2])
        right_bottom = np.minimum(boxes[index, 2:], boxes[rest, 2:])
        overlap = np.maximum(0, right_bottom - left_top)
        intersection = overlap[:, 0] * overlap[:, 1]
        union = area[index] + area[rest] - intersection
        iou = intersection / np.maximum(union, np.finfo(np.float32).eps)
        order = rest[iou <= iou_threshold]
    return keep


def decode_poses(
    prediction: np.ndarray, transform: LetterboxTransform,
    confidence: float = 0.35, iou_threshold: float = 0.45,
    max_detections: int = 20,
) -> list[dict[str, Any]]:
    """Decode raw YOLOv8-pose output and undo preprocessing coordinates."""
    rows = np.asarray(prediction)
    if rows.ndim == 3 and rows.shape[0] == 1:
        rows = rows[0]
    if rows.ndim != 2:
        raise ValueError(f"Expected raw YOLOv8 pose output [1,56,N], got {rows.shape}")
    if rows.shape[0] == 56:
        rows = rows.T
    elif rows.shape[1] != 56:
        raise ValueError(f"Expected 56 pose features, got output shape {rows.shape}")
    rows = rows.astype(np.float32, copy=False)
    mask = (
        np.isfinite(rows).all(axis=1) & (rows[:, 4] >= confidence)
        & (rows[:, 2] > 0) & (rows[:, 3] > 0)
    )
    rows = rows[mask]
    if not len(rows):
        return []
    boxes = np.empty((len(rows), 4), dtype=np.float32)
    boxes[:, :2] = rows[:, :2] - rows[:, 2:4] / 2
    boxes[:, 2:] = rows[:, :2] + rows[:, 2:4] / 2
    # Suppress in model coordinates, before clipping changes overlap at edges.
    selected = non_max_suppression(boxes, rows[:, 4], iou_threshold, max_detections)
    results = []
    for index in selected:
        bbox = boxes[index].copy()
        bbox[[0, 2]] = (bbox[[0, 2]] - transform.left) / transform.scale_x
        bbox[[1, 3]] = (bbox[[1, 3]] - transform.top) / transform.scale_y
        bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, transform.original_width - 1)
        bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, transform.original_height - 1)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        points = rows[index, 5:].reshape(17, 3).copy()
        points[:, 0] = (points[:, 0] - transform.left) / transform.scale_x
        points[:, 1] = (points[:, 1] - transform.top) / transform.scale_y
        # An out-of-frame joint must not acquire an unrelated depth at the edge.
        outside = (
            (points[:, 0] < 0) | (points[:, 0] > transform.original_width - 1)
            | (points[:, 1] < 0) | (points[:, 1] > transform.original_height - 1)
        )
        points[:, 0] = np.clip(points[:, 0], 0, transform.original_width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, transform.original_height - 1)
        points[:, 2] = np.clip(points[:, 2], 0, 1)
        points[outside, 2] = 0
        results.append({
            "bbox": bbox.tolist(),
            "score": float(np.clip(rows[index, 4], 0, 1)),
            "keypoints": points.tolist(),
        })
    return results


class _CudaRuntime:
    """Small CUDA Runtime ABI binding; keeps system CUDA as the only dependency."""

    def __init__(self) -> None:
        candidates = [
            ctypes.util.find_library("cudart"), "libcudart.so.12", "libcudart.so",
            "/usr/local/cuda/lib64/libcudart.so.12",
            "/usr/local/cuda/targets/aarch64-linux/lib/libcudart.so.12",
        ]
        self.lib = None
        errors = []
        for name in dict.fromkeys(candidate for candidate in candidates if candidate):
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError as exc:
                errors.append(str(exc))
        if self.lib is None:
            raise RuntimeError("CUDA runtime library unavailable: " + "; ".join(errors))
        signatures = {
            "cudaSetDevice": [ctypes.c_int],
            "cudaMalloc": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
            "cudaFree": [ctypes.c_void_p],
            "cudaMemcpy": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
            "cudaStreamCreateWithFlags": [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint],
            "cudaStreamSynchronize": [ctypes.c_void_p],
            "cudaStreamDestroy": [ctypes.c_void_p],
        }
        for name, signature in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = signature
            function.restype = ctypes.c_int
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def call(self, name: str, *args: Any) -> None:
        code = getattr(self.lib, name)(*args)
        if code:
            detail = self.lib.cudaGetErrorString(code)
            text = detail.decode("utf-8", "replace") if detail else "unknown CUDA error"
            raise RuntimeError(f"{name} failed ({code}): {text}")


def _engine_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    # Ultralytics .engine exports may prefix raw plan bytes with length + JSON.
    if len(data) > 8:
        length = int.from_bytes(data[:4], "little")
        if 0 < length < min(len(data) - 4, 1_000_000):
            try:
                metadata = json.loads(data[4:4 + length].decode("utf-8"))
            except (UnicodeError, ValueError):
                pass
            else:
                if isinstance(metadata, dict):
                    return data[4 + length:]
    return data


class _TensorRTBackend:
    def __init__(self, model_path: Path, input_size: int) -> None:
        import tensorrt as trt

        self.cuda = _CudaRuntime()
        self.buffers: list[ctypes.c_void_p] = []
        self.stream = ctypes.c_void_p()
        self.context = self.engine = self.runtime = None
        self.cuda.call("cudaSetDevice", 0)
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        try:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(_engine_bytes(model_path))
            if self.engine is None:
                raise RuntimeError(
                    "TensorRT engine could not be loaded. Rebuild it on this Jetson "
                    "using its installed TensorRT version."
                )
            inputs, outputs = [], []
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                target = inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs
                target.append(name)
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError("Pose engine must have exactly one image input and one raw output")
            self.input_name, self.output_name = inputs[0], outputs[0]
            self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
            self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
            if any(dimension <= 0 for dimension in self.input_shape + self.output_shape):
                raise ValueError("Dynamic TensorRT shapes are unsupported; export a static pose engine")
            if self.input_shape != (1, 3, input_size, input_size):
                raise ValueError(
                    f"Engine input is {self.input_shape}, requested (1,3,{input_size},{input_size}). "
                    "Use the matching --input-size or rebuild the engine."
                )
            if len(self.output_shape) != 3 or self.output_shape[0] != 1 or 56 not in self.output_shape[1:]:
                raise ValueError(f"Expected raw COCO pose output [1,56,N], got {self.output_shape}")
            self.input_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.input_name)))
            self.output_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.output_name)))
            for name, dtype in [(self.input_name, self.input_dtype), (self.output_name, self.output_dtype)]:
                if dtype not in (np.dtype(np.float32), np.dtype(np.float16)):
                    raise ValueError(f"Unsupported tensor dtype {dtype} for {name}; use FP16 or FP32")
                if self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                    raise ValueError(f"Tensor {name} must use contiguous LINEAR format")
                if self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                    raise ValueError(f"Tensor {name} must use CUDA device memory")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Could not create TensorRT execution context")
            self.host_output = np.empty(self.output_shape, dtype=self.output_dtype)
            sizes = [int(np.prod(self.input_shape)) * self.input_dtype.itemsize, self.host_output.nbytes]
            for name, size in zip((self.input_name, self.output_name), sizes):
                pointer = ctypes.c_void_p()
                self.cuda.call("cudaMalloc", ctypes.byref(pointer), size)
                self.buffers.append(pointer)
                if not self.context.set_tensor_address(name, int(pointer.value)):
                    raise RuntimeError(f"TensorRT rejected device buffer for {name}")
            self.cuda.call("cudaStreamCreateWithFlags", ctypes.byref(self.stream), 1)
        except Exception:
            self.close()
            raise
        self.name = f"TensorRT {trt.__version__} (GPU)"

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        self.cuda.call("cudaSetDevice", 0)
        host_input = np.ascontiguousarray(tensor, dtype=self.input_dtype)
        # Synchronous memcpy accepts ordinary NumPy memory (no pinned allocator).
        self.cuda.call("cudaMemcpy", self.buffers[0], host_input.ctypes.data, host_input.nbytes, 1)
        if not self.context.execute_async_v3(stream_handle=int(self.stream.value)):
            raise RuntimeError("TensorRT inference failed")
        self.cuda.call("cudaStreamSynchronize", self.stream)
        self.cuda.call(
            "cudaMemcpy", self.host_output.ctypes.data, self.buffers[1], self.host_output.nbytes, 2,
        )
        return self.host_output

    def close(self) -> None:
        # Cleanup is best-effort so the original inference/setup error is preserved.
        if self.cuda is None:
            return
        try:
            self.cuda.call("cudaSetDevice", 0)
            if self.stream.value:
                self.cuda.call("cudaStreamSynchronize", self.stream)
        except Exception:
            pass
        self.context = None
        for pointer in self.buffers:
            try:
                self.cuda.call("cudaFree", pointer)
            except Exception:
                pass
        self.buffers.clear()
        if self.stream.value:
            try:
                self.cuda.call("cudaStreamDestroy", self.stream)
            except Exception:
                pass
            self.stream = ctypes.c_void_p()
        self.engine = self.runtime = None


class _OpenCVBackend:
    def __init__(self, model_path: Path) -> None:
        import cv2

        # OpenCV's native file loader fails on some Windows Unicode paths.
        self.net = cv2.dnn.readNetFromONNX(np.frombuffer(model_path.read_bytes(), dtype=np.uint8))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.name = "OpenCV DNN (CPU)"

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        self.net.setInput(tensor)
        return self.net.forward()

    def close(self) -> None:
        self.net = None


class PoseEstimator:
    """Thread-serialized person pose estimator with a JSON-safe result API.

    Instantiate and close in the capture/inference worker. ``auto`` selects a
    same-stem .engine next to an ONNX model when available. Engine failures are
    reported explicitly instead of silently hiding a broken GPU installation.
    """

    def __init__(
        self, model_path: str | Path, confidence: float = 0.35,
        input_size: int = 640, backend: str = "auto",
        iou_threshold: float = 0.45, max_detections: int = 20,
    ) -> None:
        if not 0 <= confidence <= 1 or not 0 <= iou_threshold <= 1:
            raise ValueError("Confidence and IoU thresholds must be between 0 and 1")
        if input_size < 32 or input_size % 32:
            raise ValueError("Model input size must be a positive multiple of 32")
        if max_detections < 1:
            raise ValueError("max_detections must be positive")
        path = Path(model_path).expanduser().resolve()
        requested_backend = backend.lower()
        if requested_backend not in {"auto", "tensorrt", "opencv", "cpu"}:
            raise ValueError("backend must be auto, tensorrt, opencv, or cpu")
        if requested_backend == "auto" and path.suffix.lower() == ".onnx" and path.with_suffix(".engine").exists():
            path = path.with_suffix(".engine")
        if not path.is_file():
            raise FileNotFoundError(f"Pose model not found: {path}")
        self.confidence = float(confidence)
        self.input_size = int(input_size)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.model_path = str(path)
        self._lock = threading.Lock()
        self._closed = False
        use_tensorrt = requested_backend == "tensorrt" or (requested_backend == "auto" and path.suffix.lower() in {".engine", ".plan"})
        if use_tensorrt:
            if path.suffix.lower() not in {".engine", ".plan"}:
                raise ValueError("TensorRT backend requires a prebuilt .engine or .plan model")
            self._runtime = _TensorRTBackend(path, self.input_size)
        else:
            if path.suffix.lower() != ".onnx":
                raise ValueError("OpenCV backend requires an .onnx model")
            self._runtime = _OpenCVBackend(path)
        self.backend = self._runtime.name
        self.backend_name = self.backend

    def infer(self, bgr: np.ndarray) -> list[dict[str, Any]]:
        with self._lock:
            if self._closed:
                raise RuntimeError("PoseEstimator has been closed")
            tensor, transform = prepare_image(bgr, self.input_size)
            prediction = self._runtime.infer(tensor)
            return decode_poses(
                prediction, transform, self.confidence,
                self.iou_threshold, self.max_detections,
            )

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._runtime.close()

    def __enter__(self) -> "PoseEstimator":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
