"""RTMPose-m Body7/COCO17, detector-free full-frame or pixel-ROI inference.

The official OpenMMLab deployment pipeline supplies RGB ImageNet normalization,
1.25 bbox padding, aspect-preserving top-down affine, and SimCC ratio 2 decode.
This adapter follows that metadata, including the MMPose minimum-axis score.
An ROI is a user-specified single-person hypothesis, not a person detection.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import threading
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
MODEL_NAME = "RTMPose-m Body7 (COCO 17)"
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
]
EDGES = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9],
         [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13],
         [13, 15], [12, 14], [14, 16]]


def prepare_image(bgr: np.ndarray, roi=None):
    """Return RGB normalized NCHW input and invertible affine metadata."""
    import cv2

    if not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected uint8 BGR image (H,W,3)")
    height, width = bgr.shape[:2]
    if min(height, width) < 1:
        raise ValueError("Image dimensions must be positive")
    bbox = np.asarray(roi if roi is not None else [0, 0, width, height], dtype=np.float32)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        raise ValueError("ROI must be finite pixel [x1,y1,x2,y2]")
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width)
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height)
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError("ROI must have positive width and height within image")
    center = (bbox[:2] + bbox[2:]) * np.float32(.5)
    scale = (bbox[2:] - bbox[:2]) * np.float32(1.25)
    aspect = 192 / 256
    if scale[0] > scale[1] * aspect:
        scale[1] = scale[0] / aspect
    else:
        scale[0] = scale[1] * aspect
    # Rotation is zero. This is exactly TopDownAffine's three-point transform.
    ratio = np.asarray([192, 256], dtype=np.float32) / scale
    matrix = np.asarray([[ratio[0], 0, 96 - center[0] * ratio[0]],
                         [0, ratio[1], 128 - center[1] * ratio[1]]], dtype=np.float32)
    warped = cv2.warpAffine(bgr, matrix, (192, 256), flags=cv2.INTER_LINEAR)
    rgb = warped[:, :, ::-1].astype(np.float32)
    rgb = (rgb - np.asarray([123.675, 116.28, 103.53], np.float32)) / np.asarray([58.395, 57.12, 57.375], np.float32)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]), (center, scale, bbox)


def decode_simcc(simcc_x, simcc_y, transform, image_shape):
    """MMPose SimCC argmax/min-axis confidence; outside-image confidence=0."""
    x = np.asarray(simcc_x, dtype=np.float32)
    y = np.asarray(simcc_y, dtype=np.float32)
    if x.shape != (1, 17, 384) or y.shape != (1, 17, 512):
        raise ValueError(f"Expected COCO17 SimCC (1,17,384)/(1,17,512), got {x.shape}/{y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("RTMPose returned non-finite SimCC values")
    center, scale, _ = transform
    points = np.stack([np.argmax(x[0], axis=-1), np.argmax(y[0], axis=-1)], axis=-1).astype(np.float32) / 2
    points = points / np.asarray([192, 256], np.float32) * scale + center - scale / 2
    scores = np.minimum(np.max(x[0], axis=-1), np.max(y[0], axis=-1))
    height, width = image_shape[:2]
    outside = (points[:, 0] < 0) | (points[:, 0] >= width) | (points[:, 1] < 0) | (points[:, 1] >= height)
    scores = np.clip(scores, 0, 1)
    scores[outside] = 0
    # Retain true projected coordinates; never turn out-of-frame predictions into
    # apparently measured edge pixels. The common renderer/depth sampler gates score.
    return np.column_stack([points, scores]).tolist()


class _TensorRTSimCC:
    def __init__(self, path: Path):
        import tensorrt as trt
        try:
            from ..pose_runtime import _CudaRuntime
        except ImportError:
            from pose_runtime import _CudaRuntime

        self.cuda = _CudaRuntime()
        self.buffers = []
        self.stream = ctypes.c_void_p()
        self.context = self.engine = self.runtime = None
        self.cuda.call("cudaSetDevice", 0)
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        try:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
            if self.engine is None:
                raise RuntimeError("RTMPose engine is invalid; rebuild on this Jetson")
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            if len(inputs) != 1 or len(names) != 3:
                raise ValueError("RTMPose engine needs one input and two SimCC outputs")
            self.input_name = inputs[0]
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Could not create RTMPose TensorRT context")
            if not self.context.set_input_shape(self.input_name, (1, 3, 256, 192)):
                raise ValueError("RTMPose engine must accept (1,3,256,192)")
            self.arrays = {}
            self.pointers = {}
            for name in names:
                shape = tuple(self.context.get_tensor_shape(name))
                if not shape or any(v <= 0 for v in shape):
                    raise ValueError(f"Unresolved TensorRT shape: {name} {shape}")
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                if dtype not in (np.dtype(np.float32), np.dtype(np.float16)):
                    raise ValueError(f"Unsupported tensor dtype {dtype}")
                if self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR or self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                    raise ValueError("Expected contiguous device tensors")
                array = np.empty(shape, dtype=dtype)
                pointer = ctypes.c_void_p()
                self.cuda.call("cudaMalloc", ctypes.byref(pointer), array.nbytes)
                self.buffers.append(pointer)
                if not self.context.set_tensor_address(name, int(pointer.value)):
                    raise RuntimeError(f"Could not bind TensorRT tensor {name}")
                self.arrays[name], self.pointers[name] = array, pointer
            outputs = [n for n in names if n != self.input_name]
            self.output_x = next(n for n in outputs if self.arrays[n].shape == (1, 17, 384))
            self.output_y = next(n for n in outputs if self.arrays[n].shape == (1, 17, 512))
            self.cuda.call("cudaStreamCreateWithFlags", ctypes.byref(self.stream), 1)
        except Exception:
            self.close()
            raise
        self.name = f"TensorRT {trt.__version__} (GPU)"

    def infer(self, tensor):
        self.cuda.call("cudaSetDevice", 0)
        data = np.ascontiguousarray(tensor, dtype=self.arrays[self.input_name].dtype)
        self.cuda.call("cudaMemcpy", self.pointers[self.input_name], data.ctypes.data, data.nbytes, 1)
        if not self.context.execute_async_v3(stream_handle=int(self.stream.value)):
            raise RuntimeError("RTMPose TensorRT execution failed")
        self.cuda.call("cudaStreamSynchronize", self.stream)
        for name in (self.output_x, self.output_y):
            array = self.arrays[name]
            self.cuda.call("cudaMemcpy", array.ctypes.data, self.pointers[name], array.nbytes, 2)
        return self.arrays[self.output_x], self.arrays[self.output_y]

    def close(self):
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


class _OpenCVSimCC:
    def __init__(self, path):
        import cv2
        self.net = cv2.dnn.readNetFromONNX(np.frombuffer(path.read_bytes(), np.uint8))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.names = self.net.getUnconnectedOutLayersNames()
        self.name = "OpenCV DNN (CPU)"

    def infer(self, tensor):
        self.net.setInput(tensor)
        outputs = self.net.forward(self.names)
        x = next(v for v in outputs if v.shape == (1, 17, 384))
        y = next(v for v in outputs if v.shape == (1, 17, 512))
        return x, y

    def close(self):
        self.net = None


class RTMPoseAdapter:
    name = MODEL_NAME
    model = MODEL_NAME

    def __init__(self, model_path=None, roi=None, confidence=.3):
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        path = Path(model_path) if model_path else ROOT / "models" / "rtmpose" / "end2end.onnx"
        if path.suffix == ".onnx" and path.with_suffix(".engine").is_file():
            path = path.with_suffix(".engine")
        if not path.is_file():
            raise FileNotFoundError(f"RTMPose model missing: {path}. Run scripts/setup_rtmpose.py on Jetson.")
        self.roi = roi
        self.confidence = float(confidence)
        self.model_path = str(path)
        self._lock = threading.Lock()
        self._closed = False
        self.runtime = _TensorRTSimCC(path) if path.suffix in (".engine", ".plan") else _OpenCVSimCC(path)
        self.backend = self.runtime.name

    def infer(self, bgr) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("RTMPoseAdapter is closed")
            tensor, transform = prepare_image(bgr, self.roi)
            points = decode_simcc(*self.runtime.infer(tensor), transform, bgr.shape)
            visible = sum(p[2] >= self.confidence for p in points)
            instance = {
                "bbox": transform[2].tolist(), "score": float(max(p[2] for p in points)),
                "label": "body ROI", "keypoints": points, "keypoint_names": KEYPOINT_NAMES,
                "edges": EDGES, "coordinate_kind": "image_2d", "visible_keypoints": visible,
                "roi_hypothesis": True,
            }
            return {"instances": [instance], "backend": self.backend, "model": self.model,
                    "keypoint_threshold": self.confidence, "person_detector": False,
                    "roi": transform[2].tolist(),
                    "notes": "Single-person full-frame/ROI hypothesis; person detector bypassed. Partial-arm accuracy is unverified. Confidence scores are model-specific."}

    def close(self):
        with self._lock:
            if not self._closed:
                self.runtime.close()
                self._closed = True
