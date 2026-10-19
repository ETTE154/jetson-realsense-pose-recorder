"""Genuine EgoForce HALO 2D inference with shared or manually selected hand ROI.

This adapter runs the original HALO network and strict official checkpoint.
It deliberately does not claim the upstream RTMDet+YOLO detection pipeline or
MANO/ray-space 3D reconstruction. Those require a substantially different GPU
software stack. The comparison UI must display the crop source explicitly.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

MODEL_NAME = "EgoForce HALO · 2D / shared hand ROI"
SOURCE_REVISION = "480ffb358516d0d7f971ec9da99b7bb07635731e"
CHECKPOINT_SHA256 = "722fe84ba2b6ab3569c3a95000c61a9a77c7b3a41cc76585f891b8bc4fe82870"
HAND_NAMES = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
              [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
              [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official EgoForce source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _source_modules(source):
    """Avoid executing upstream package initializers that import 3D renderers.

    No source code or model operation is replaced. Only package resolution is
    narrowed to the actual 2D inference dependencies in this isolated worker.
    """
    prefix = "_d435_egoforce_official"
    for package, relative in [(prefix, ""), (prefix + ".models", "models"), (prefix + ".utils", "utils")]:
        module = ModuleType(package)
        module.__path__ = [str(source / relative)]
        sys.modules[package] = module
    previous = {name: sys.modules.get(name) for name in ("utils", "utils.rotations", "utils.augmentation")}
    try:
        sys.modules["utils"] = sys.modules[prefix + ".utils"]
        sys.modules["utils.rotations"] = _load_module(prefix + ".utils.rotations", source / "utils/rotations.py")
        sys.modules["utils.augmentation"] = _load_module(prefix + ".utils.augmentation", source / "utils/augmentation.py")
        halo = _load_module(prefix + ".models.halo", source / "models/halo.py")
        loader = _load_module(prefix + ".loader", source / "demo/demo_hand_arm_loader.py")
        camera = _load_module(prefix + ".pinhole", source / "camera_models/pinhole.py")
        yield halo.HALO, loader.DemoHandArmLoader, camera.PinholeCameraModel
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _config():
    # Values are the official settings.py inference architecture/preprocessing.
    return SimpleNamespace(
        NUM_JOINTS_PER_HAND=21,
        POSE_3D=SimpleNamespace(ROT_6D=True, DECODER_ITRS=3, IMAGE_SIZE=[224, 224],
                                HEATMAP_SIZE=[56, 56], TARGET_TYPE="gaussian", SIGMA=2),
    )


def _box(box, width, height, expand=1.0):
    import numpy as np
    values = np.asarray(box, dtype=float).reshape(-1)
    if len(values) != 4 or not np.isfinite(values).all():
        raise ValueError("ROI must be finite pixel coordinates [x1,y1,x2,y2]")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI has zero or negative area")
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = (x2 - x1) * expand / 2, (y2 - y1) * expand / 2
    result = [max(0, int(center_x - half_w)), max(0, int(center_y - half_h)),
              min(width, int(center_x + half_w)), min(height, int(center_y + half_h))]
    if result[2] - result[0] < 8 or result[3] - result[1] < 8:
        raise ValueError("Visible ROI must be at least 8 x 8 pixels")
    return result


class EgoForceAdapter:
    def __init__(self, model_path=None, **kwargs):
        root = Path(__file__).resolve().parents[1]
        # NVIDIA's Jetson wheel requires cuSPARSELt. Load only the private
        # project copy by absolute path; no system linker configuration changes.
        private_sparse = root / "models/egoforce/runtime/libcusparse_lt-linux-aarch64-0.6.2.3-archive/lib/libcusparseLt.so.0"
        self._private_sparse_handle = None
        if os.name == "posix" and private_sparse.is_file():
            self._private_sparse_handle = ctypes.CDLL(str(private_sparse), mode=ctypes.RTLD_GLOBAL)
        import torch
        self.torch = torch
        requested = os.environ.get("EGOFORCE_DEVICE", "cuda")
        if requested != "cuda":
            raise RuntimeError("EgoForce requires EGOFORCE_DEVICE=cuda; CPU fallback is disabled")
        if not torch.cuda.is_available():
            raise RuntimeError("EgoForce GPU mode needs CUDA-enabled PyTorch in .egoforce-gpu-venv; CPU fallback is disabled")
        self.device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        configured = model_path or os.environ.get("EGOFORCE_CHECKPOINT") or root / "models/egoforce/model_weights.pth"
        self.checkpoint = Path(configured).resolve()
        self.model_root = self.checkpoint.parent
        self.source = self.model_root / "source"
        if not self.checkpoint.is_file():
            raise RuntimeError("EgoForce checkpoint is missing. Run: python scripts/setup_egoforce.py --download (3.02 GB public official weights)")
        manifest_path = self.model_root / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("EgoForce verified provenance manifest missing; run scripts/setup_egoforce.py --download")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("source_revision") != SOURCE_REVISION or self.manifest.get("checkpoint_sha256") != CHECKPOINT_SHA256:
            raise RuntimeError("EgoForce manifest does not match the reviewed official model revision")
        if self.checkpoint.stat().st_size != self.manifest["checkpoint_size"]:
            raise RuntimeError("EgoForce checkpoint size is incorrect; re-run the verified downloader")
        for relative, digest in self.manifest["sources"].items():
            path = self.source / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"EgoForce official source integrity check failed: {relative}")
        # Full checkpoint SHA is verified on first activation in each worker.
        digest = hashlib.sha256()
        with self.checkpoint.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != CHECKPOINT_SHA256:
            raise RuntimeError("EgoForce checkpoint SHA-256 mismatch")
        torch.set_num_threads(max(1, min(4, int(os.environ.get("EGOFORCE_THREADS", "4")))))
        self.cfg = _config()
        with _source_modules(self.source) as (HALO, Loader, Camera):
            self.Loader = Loader
            self.Camera = Camera
            self.model = HALO(self.cfg)
        # weights_only forbids arbitrary checkpoint pickle globals. mmap avoids
        # a second resident 3 GB copy while strict loading validates every key.
        weights = torch.load(str(self.checkpoint), map_location="cpu", weights_only=True, mmap=True)
        self.model.load_state_dict(weights, strict=True, assign=True)
        del weights
        self.model.eval().requires_grad_(False)
        self.model.to(self.device, dtype=torch.float16)
        self.intrinsics = None
        self.gpu_name = torch.cuda.get_device_name(self.device)
        self.backend = f"PyTorch {torch.__version__} · CUDA {torch.version.cuda} · {self.gpu_name} · FP16 · official HALO"

    def set_intrinsics(self, intrinsics):
        self.intrinsics = intrinsics

    def infer(self, bgr, intrinsics=None, roi=None, hand_hints=None):
        import cv2
        import numpy as np
        torch = self.torch
        height, width = bgr.shape[:2]
        calibration = intrinsics or self.intrinsics
        if not isinstance(calibration, dict):
            raise ValueError("EgoForce requires actual RGB camera intrinsics; no synthetic calibration is used")
        fx, fy = float(calibration["fx"]), float(calibration["fy"])
        cx, cy = float(calibration.get("ppx", calibration.get("cx"))), float(calibration.get("ppy", calibration.get("cy")))
        if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
            raise ValueError("EgoForce received invalid RGB camera intrinsics")
        for key, actual in [("width", width), ("height", height)]:
            if key in calibration and int(calibration[key]) != actual:
                raise ValueError("EgoForce intrinsics resolution must match the image supplied to the worker")
        coeffs = np.asarray(calibration.get("coeffs", []), dtype=float)
        distortion_note = "pinhole RGB intrinsics"
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if coeffs.size and np.any(np.abs(coeffs) > 1e-7):
            raise ValueError("EgoForce HALO 2D adapter requires rectified RGB with zero distortion coefficients so overlay and RGB-D coordinates remain consistent")
        camera = self.Camera([fx, fy], [cx, cy], width, height)
        hints = []
        for hint in hand_hints or []:
            if not isinstance(hint, dict) or "bbox" not in hint:
                continue
            label = str(hint.get("label", "hand"))
            hints.append((hint, label))
        if not hints and roi is not None:
            hints = [({"bbox": roi, "score": 1.0}, "manual hand ROI")]
        instances = []
        details = []
        began = time.perf_counter()
        gpu_kernel_ms = 0.0
        # One hand per forward pass limits activation memory on Orin NX 16 GB.
        for hint, label in hints[:2]:
            box = _box(hint["bbox"], width, height, expand=1.25)
            boxes = {"hand": {"bbox": box, "keypoint": np.zeros((3, 2))}}
            arm_box = None
            if roi is not None and hand_hints:
                arm_box = _box(roi, width, height)
                boxes["arm"] = {"bbox": arm_box, "keypoint": np.zeros((3, 2))}
            loader = self.Loader(self.cfg, camera, undistort_inp=False, hand_type="left" if "left" in label.lower() else "right")
            data, meta = loader.transform(rgb, boxes)
            inputs = [data[name].unsqueeze(0).unsqueeze(0).to(self.device).contiguous() for name in ("hand_crop", "hand_sparse_kpe", "arm_crop", "arm_sparse_kpe")]
            if self.device.type == "cuda":
                inputs = [value.half() for value in inputs]
            gpu_started = torch.cuda.Event(enable_timing=True)
            gpu_finished = torch.cuda.Event(enable_timing=True)
            gpu_started.record()
            with torch.inference_mode():
                outputs = self.model(*inputs)
            gpu_finished.record()
            gpu_finished.synchronize()
            gpu_kernel_ms += gpu_started.elapsed_time(gpu_finished)
            hand_xy = outputs["hand_kpts_2d"][0, 0].float().cpu().numpy()
            hand_score = outputs["hand_kpt_w"][0, 0].float().cpu().numpy().reshape(-1)
            if not np.isfinite(hand_xy).all() or not np.isfinite(hand_score).all():
                raise RuntimeError("EgoForce GPU produced nonfinite hand outputs; result was not accepted")
            hand_xy = hand_xy / 224.0 * np.asarray(meta["hand_crop_size"]) + np.asarray(meta["hand_bbox"][:2])
            # HALO's learned solver weights are not confidence probabilities.
            # The comparison protocol uses a 1.0 display mask when no calibrated
            # per-keypoint confidence is reported, as it does for MediaPipe.
            points = np.column_stack([hand_xy, np.ones(len(hand_xy))]).tolist()
            solver_weights = hand_score.tolist()
            names = list(HAND_NAMES)
            edges = list(HAND_EDGES)
            if arm_box is not None:
                arm_xy = outputs["arm_kpts_2d"][0, 0].float().cpu().numpy()
                arm_score = outputs["arm_kpt_w"][0, 0].float().cpu().numpy().reshape(-1)
                if not np.isfinite(arm_xy).all() or not np.isfinite(arm_score).all():
                    raise RuntimeError("EgoForce GPU produced nonfinite forearm outputs; result was not accepted")
                arm_xy = arm_xy / 224.0 * np.asarray(meta["arm_crop_size"]) + np.asarray(meta["arm_bbox"][:2])
                points += np.column_stack([arm_xy, np.ones(len(arm_xy))]).tolist()
                solver_weights += arm_score.tolist()
                names += ["forearm_elbow", "forearm_mid", "forearm_wrist"]
                edges += [[21, 22], [22, 23], [23, 0]]
            instances.append({
                "bbox": box, "score": 1.0, "score_kind": "not_reported_display_mask", "label": label,
                "keypoints": points, "keypoint_names": names, "edges": edges,
                "coordinate_kind": "image_2d", "estimated_geometry": None,
                "keypoint_confidence_kind": "not_reported_display_mask",
                "native_solver_weights": solver_weights,
            })
            details.append({"hand_crop": box, "arm_crop": arm_box,
                            "hand_crop_source": "MediaPipe shared detection" if hand_hints else "manual hand ROI",
                            "arm_visible": arm_box is not None,
                            "hidden_arm_points_omitted": arm_box is None,
                            "native_solver_weights": solver_weights})
            del outputs, inputs
        return {"instances": instances, "backend": self.backend, "model": MODEL_NAME,
                "metadata": {
                    "source_revision": SOURCE_REVISION, "checkpoint_sha256": CHECKPOINT_SHA256,
                    "device": "cuda", "gpu_name": self.gpu_name,
                    "precision": "float16", "cpu_fallback": False,
                    "license": "CC-BY-NC-4.0", "scope": "official HALO 2D heads; shared hand ROI; no MANO mesh or ray-space 3D reconstruction",
                    "crop_details": details, "calibration": distortion_note,
                    "no_hand_reason": None if hints else "No shared hand detection. Select a visible HAND ROI to evaluate manually.",
                    "model_ms": round((time.perf_counter() - began) * 1000, 2),
                    "gpu_kernel_ms": round(gpu_kernel_ms, 2),
                    "gpu_allocated_mib": round(torch.cuda.memory_allocated(self.device) / 1024**2, 1),
                    "confidence_note": "Keypoint third column is a 1.0 display mask, not confidence. Native learned solver weights are preserved separately and are not detection probabilities or comparable confidence scores.",
                }}

    def close(self):
        pass
