"""Capture once, run three genuine models, retain images and raw results."""
from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time
import uuid

import cv2
import numpy as np

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
IDS = ("mediapipe", "rtmpose", "egoforce")
PREFIX = "COMPARE_RESULT "
ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
SPECS = {
    "mediapipe": {"name": "MediaPipe Hand Landmarker", "task": "손·손목 21점",
                  "description": "손 전용 검출 · CPU", "env": ".mediapipe-venv",
                  "asset": "models/comparison/hand_landmarker.task"},
    "rtmpose": {"name": "RTMPose-m", "task": "신체 관절 17점",
                "description": "사람 검출 생략 · 지정 영역에서 신체 관절 추정", "env": ".venv",
                "asset": "models/rtmpose/end2end.onnx"},
    "egoforce": {"name": "EgoForce HALO", "task": "손·전완 2D 관절",
                 "description": "공식 손·전완 모델 · CUDA FP16 · 손 영역 공유 · 3D mesh 제외", "env": ".egoforce-gpu-venv",
                 "asset": "models/egoforce/model_weights.pth"},
}


def python_path(identifier):
    override = os.environ.get("COMPARE_" + identifier.upper() + "_PYTHON")
    if override:
        return Path(override)
    directory = ROOT / SPECS[identifier]["env"]
    return directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def validate_roi(value, width, height):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("영역은 [왼쪽, 위, 오른쪽, 아래] 숫자 4개여야 합니다.")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError("영역 좌표에 유효한 숫자를 입력하세요.")
    x1, y1, x2, y2 = map(float, value)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and x2-x1 >= 8 and y2-y1 >= 8):
        raise ValueError("선택 영역은 영상 안에 있어야 하며 가로·세로가 8픽셀 이상이어야 합니다.")
    return [x1, y1, x2, y2]


def enrich_instances(instances, depth, scale, intrinsics, threshold, deproject=None):
    if deproject is None:
        import pyrealsense2 as rs
        parameters = rs.intrinsics()
        for key in ("width", "height", "ppx", "ppy", "fx", "fy"):
            setattr(parameters, key, intrinsics[key])
        model_name = str(intrinsics.get("model", "none")).split(".")[-1]
        parameters.model = getattr(rs.distortion, model_name, rs.distortion.none)
        parameters.coeffs = intrinsics.get("coeffs", [0.]*5)
        deproject = lambda uv, z: rs.rs2_deproject_pixel_to_point(parameters, uv, z)
    height, width = depth.shape
    output, shown, measured = [], 0, 0
    for original in instances[:20]:
        instance = copy.deepcopy(original)
        points = np.asarray(instance.get("keypoints", []), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) > 256 or not np.isfinite(points).all():
            raise ValueError("모델이 유효하지 않은 관절 좌표를 반환했습니다.")
        names = instance.get("keypoint_names") or [f"joint_{i}" for i in range(len(points))]
        mask_only = instance.get("keypoint_confidence_kind") == "not_reported_display_mask"
        coordinates = []
        for index, (u, v, confidence) in enumerate(points):
            selected = bool((confidence > 0 if mask_only else confidence >= threshold)
                            and 0 <= u < width and 0 <= v < height)
            joint = {"name": names[index] if index < len(names) else f"joint_{index}",
                     "u": float(u), "v": float(v), "confidence": None if mask_only else float(confidence),
                     "confidence_kind": instance.get("keypoint_confidence_kind", "model_score"),
                     "displayed": selected, "valid_depth": False,
                     "x_m": None, "y_m": None, "z_m": None,
                     "depth_source": "D435i local surface depth; not ground-truth joint depth"}
            if selected:
                shown += 1
                x, y = min(width-1, round(u)), min(height-1, round(v))
                patch = depth[max(0, y-2):min(height, y+3), max(0, x-2):min(width, x+3)]
                values = patch[patch > 0]
                if len(values) >= 3:
                    z = float(np.median(values))*scale
                    if .1 <= z <= 10:
                        xyz = deproject([float(u), float(v)], z)
                        if np.isfinite(xyz).all():
                            joint.update(valid_depth=True, x_m=float(xyz[0]), y_m=float(xyz[1]), z_m=float(xyz[2]))
                            measured += 1
            coordinates.append(joint)
        instance["depth_keypoints"] = coordinates
        output.append(instance)
    return output, shown, measured


def draw_instances(bgr, instances, roi=None):
    image = bgr.copy()
    if roi:
        cv2.rectangle(image, tuple(map(round, roi[:2])), tuple(map(round, roi[2:])), (200, 170, 70), 1)
    for number, instance in enumerate(instances):
        color = [(65, 220, 170), (235, 170, 70), (180, 100, 235)][number % 3]
        points = instance["depth_keypoints"]
        for edge in instance.get("edges", []):
            if len(edge) != 2:
                continue
            a, b = map(int, edge)
            if not (0 <= a < len(points) and 0 <= b < len(points)):
                continue
            if points[a]["displayed"] and points[b]["displayed"]:
                cv2.line(image, (round(points[a]["u"]), round(points[a]["v"])),
                         (round(points[b]["u"]), round(points[b]["v"])), color, 2, cv2.LINE_AA)
        for point in points:
            if not point["displayed"]:
                continue
            location = (round(point["u"]), round(point["v"]))
            cv2.circle(image, location, 3, color if point["valid_depth"] else (40, 160, 255), -1, cv2.LINE_AA)
            if point["valid_depth"] and ("wrist" in point["name"].lower() or point["name"] == "joint_0"):
                cv2.putText(image, f'{point["z_m"]:.2f}m', (location[0]+4, location[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
    return image


class ModelWorker:
    def __init__(self, identifier):
        self.identifier = identifier
        self.process = None
        self.responses = queue.Queue()
        self.load_ms = None
        self.log = None
        self.lifecycle = threading.RLock()

    @staticmethod
    def _read(process, responses):
        try:
            for line in process.stdout:
                if line.startswith(PREFIX):
                    try:
                        responses.put(json.loads(line[len(PREFIX):]))
                    except ValueError:
                        continue
        except (OSError, ValueError):
            pass
        finally:
            responses.put({"worker_exited": True})

    def _response(self, timeout):
        try:
            response = self.responses.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise RuntimeError("모델 응답 시간이 초과됐습니다. 로그를 확인한 뒤 다시 실행하세요.")
        if response.get("worker_exited"):
            raise RuntimeError("모델 실행이 종료됐습니다. 모델 로그를 확인하세요.")
        return response

    def start(self, cancel_event=None):
        # Publish the new child atomically with cancellation. Never hold this
        # lock while waiting for a heavy model to load or return inference.
        with self.lifecycle:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("비교를 중지했습니다.")
            if self.process is not None and self.process.poll() is None:
                return
            self.close()
            self.responses = queue.Queue()
            runtime = python_path(self.identifier)
            if not runtime.is_file():
                raise RuntimeError(f'{SPECS[self.identifier]["name"]} 실행 환경이 준비되지 않았습니다.')
            (ROOT / "logs").mkdir(exist_ok=True)
            self.log = (ROOT / "logs" / f"compare_{self.identifier}.log").open("a", encoding="utf-8")
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4")
            self.process = subprocess.Popen([str(runtime), "-u", "-m", "comparison.worker", "--candidate", self.identifier],
                                            cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.log, text=True, encoding="utf-8", bufsize=1,
                                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            threading.Thread(target=self._read, args=(self.process, self.responses),
                             name=f"{self.identifier}-stdout", daemon=True).start()
        response = self._response(300)
        if not response.get("ready"):
            self.close()
            raise RuntimeError(response.get("error", "모델 초기화에 실패했습니다."))
        self.load_ms = response.get("load_ms")

    def infer(self, request, cancel_event=None):
        self.start(cancel_event)
        with self.lifecycle:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("비교를 중지했습니다.")
            if self.process is None or self.process.poll() is not None:
                raise RuntimeError("모델 실행이 종료됐습니다.")
            self.process.stdin.write(json.dumps(request, ensure_ascii=True, allow_nan=False) + "\n")
            self.process.stdin.flush()
        response = self._response(600)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "모델 추론에 실패했습니다."))
        return response

    def close(self):
        with self.lifecycle:
            self._close_locked()

    def _close_locked(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            for channel in (process.stdin, process.stdout):
                if channel:
                    channel.close()
        if self.log is not None:
            self.log.close()
            self.log = None


class ComparisonManager:
    def __init__(self, application):
        self.app = application
        self.output_dir = ROOT / "comparisons"
        self.output_dir.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.job = None
        self.cancel_event = threading.Event()
        self.closed = False
        self.workers = {name: ModelWorker(name) for name in IDS}

    def captures(self):
        rows = []
        for path in sorted(self.app.output_dir.glob("*/metadata.json"), reverse=True)[:100]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                rows.append({"id": path.parent.name, "frame_id": data.get("frame_id"),
                             "time": data.get("received_time"), "rgb_url": f"/captures/{path.parent.name}/rgb.jpg"})
            except (OSError, ValueError):
                continue
        return rows

    def state(self):
        models = []
        for identifier in IDS:
            spec = SPECS[identifier]
            runtime = python_path(identifier).is_file()
            asset = (ROOT / spec["asset"]).is_file()
            adapter = (ROOT / "comparison" / f"{identifier}_adapter.py").is_file()
            models.append({"id": identifier, **{k: spec[k] for k in ("name", "task", "description")},
                           "available": runtime and asset and adapter, "python_exists": runtime,
                           "assets_ready": asset and adapter})
        history = []
        for path in sorted(self.output_dir.glob("*/result.json"), reverse=True)[:20]:
            try:
                entry = json.loads(path.read_text(encoding="utf-8"))
                history.append({k: entry.get(k) for k in ("id", "state", "source", "finished_at")})
            except (OSError, ValueError):
                continue
        with self.lock:
            job = copy.deepcopy(self.job)
        return {"models": models, "job": job, "captures": self.captures(), "history": history}

    def job_by_id(self, identifier):
        if not ID_PATTERN.fullmatch(identifier):
            raise ValueError("유효하지 않은 비교 ID입니다.")
        with self.lock:
            if self.job and self.job["id"] == identifier:
                return copy.deepcopy(self.job)
        path = self.output_dir / identifier / "result.json"
        if not path.is_file():
            raise ValueError("비교 결과가 없습니다.")
        return json.loads(path.read_text(encoding="utf-8"))

    def _source(self, identifier):
        if identifier == "live":
            from prototype import intrinsics_json
            with self.app.condition:
                if self.app._processed is None or self.app.camera.snapshot()["state"] != "streaming":
                    raise ValueError("현재 카메라 영상이 없습니다. 카메라 연결을 확인하세요.")
                frame, _, settings, pose, device = self.app._processed
                rgb, depth = frame.rgb.copy(), frame.depth.copy()
                metadata = {"frame_id": frame.frame_id, "timestamp_ms": frame.timestamp_ms,
                            "received_time": frame.received_time, "depth_scale_m_per_unit": frame.depth_scale,
                            "depth_intrinsics": intrinsics_json(frame.intrinsics), "camera": device}
                baseline = self.app._images.get("pose") if settings.get("pose_model", "yolo") == "yolo" else None
            return rgb, depth, metadata, baseline
        if not isinstance(identifier, str) or not ID_PATTERN.fullmatch(identifier):
            raise ValueError("유효하지 않은 스냅샷 ID입니다.")
        directory = self.app.output_dir / identifier
        if not (directory / "metadata.json").is_file():
            raise ValueError("스냅샷을 찾을 수 없습니다.")
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        rgb = cv2.imdecode(np.frombuffer((directory / "rgb.jpg").read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        depth = cv2.imdecode(np.frombuffer((directory / "depth16.png").read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or depth.dtype != np.uint16 or depth.shape != rgb.shape[:2]:
            raise ValueError("스냅샷의 RGB와 원본 깊이를 읽을 수 없습니다.")
        baseline = ((directory / "pose.jpg").read_bytes()
                    if (directory / "pose.jpg").is_file() and metadata.get("settings", {}).get("pose_model", "yolo") == "yolo"
                    else None)
        return rgb, depth, metadata, baseline

    def run(self, payload):
        source = payload.get("source", "live")
        threshold = payload.get("draw_threshold", .3)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not .05 <= threshold <= .95:
            raise ValueError("표시 기준은 0.05~0.95 숫자여야 합니다.")
        with self.lock:
            if self.closed:
                raise ValueError("프로그램이 종료 중입니다.")
            if self.job and self.job.get("busy"):
                raise ValueError("현재 비교가 진행 중입니다. 완료 후 다시 실행하세요.")
            rgb, depth, metadata, baseline = self._source(source)
            roi = validate_roi(payload.get("roi"), rgb.shape[1], rgb.shape[0])
            identifier = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
            directory = self.output_dir / identifier
            directory.mkdir()
            ok, encoded = cv2.imencode(".png", rgb)
            if not ok:
                raise RuntimeError("비교 원본 저장 실패")
            source_bytes = encoded.tobytes()
            (directory / "source.png").write_bytes(source_bytes)
            if not cv2.imwrite(str(directory / "depth16.png"), depth):
                raise RuntimeError("비교 깊이 저장 실패")
            (directory / "source_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            if baseline:
                (directory / "baseline_yolo.jpg").write_bytes(baseline)
            self.cancel_event = threading.Event()
            self.job = {"id": identifier, "state": "queued", "busy": True,
                        "started_at": datetime.now().astimezone().isoformat(), "finished_at": None,
                        "source": {"capture_id": source, "frame_id": metadata.get("frame_id"),
                                   "timestamp_ms": metadata.get("timestamp_ms"),
                                   "image_url": f"/comparisons/{identifier}/source.png",
                                   "baseline_url": f"/comparisons/{identifier}/baseline_yolo.jpg" if baseline else None,
                                   "width": rgb.shape[1], "height": rgb.shape[0],
                                   "sha256": hashlib.sha256(source_bytes).hexdigest()},
                        "roi": roi, "draw_threshold": float(threshold), "error": None,
                        "results": {key: {"state": "queued", "model": SPECS[key]["name"], "error": None} for key in IDS},
                        "notes": ["동일한 RGB 프레임을 각 후보에 순차 입력합니다. 정답 관절을 주석하지 않았으므로 정확도 점수는 제공하지 않습니다.",
                                  "관절 수와 신뢰도 정의가 모델별로 달라 숫자를 정확도 순위로 비교할 수 없습니다.",
                                  "표시 기준은 RTMPose 관절 점수에 적용합니다. MediaPipe와 EgoForce는 관절별 확률을 제공하지 않아 화면 안의 반환 좌표를 표시합니다.",
                                  "EgoForce는 공식 HALO의 2D 출력이며 손 ROI는 MediaPipe와 공유합니다. 3D mesh 비교가 아닙니다."]}
            threading.Thread(target=self._execute, args=(identifier, directory, rgb, depth, metadata, roi, float(threshold)),
                             name="candidate-comparison", daemon=True).start()
            return copy.deepcopy(self.job)

    def _save(self, directory):
        with self.lock:
            data = copy.deepcopy(self.job)
        temporary = directory / "result.json.tmp"
        temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        temporary.replace(directory / "result.json")

    def _execute(self, identifier, directory, rgb, depth, metadata, roi, threshold):
        hand_hints = []
        try:
            with self.lock:
                self.job["state"] = "running"
            for candidate in IDS:
                if self.cancel_event.is_set():
                    break
                with self.lock:
                    self.job["results"][candidate]["state"] = "loading"
                self._save(directory)
                try:
                    worker = self.workers[candidate]
                    worker.start(self.cancel_event)
                    with self.lock:
                        self.job["results"][candidate]["state"] = "running"
                    response = worker.infer({"op": "infer", "image": str(directory / "source.png"),
                                             "roi": roi, "intrinsics": metadata["depth_intrinsics"],
                                             "hand_hints": hand_hints}, self.cancel_event)
                    result = response["result"]
                    raw_instances = result.get("instances", [])
                    if candidate == "mediapipe":
                        hand_hints = raw_instances
                    instances, shown, measured = enrich_instances(raw_instances, depth,
                                                                  metadata["depth_scale_m_per_unit"],
                                                                  metadata["depth_intrinsics"], threshold)
                    overlay = draw_instances(rgb, instances, roi)
                    if not cv2.imwrite(str(directory / f"{candidate}.jpg"), overlay):
                        raise RuntimeError("모델 결과 이미지 저장 실패")
                    result_metadata = {k: v for k, v in result.items()
                                       if k not in ("instances", "backend", "model", "metadata")}
                    result_metadata.update(result.get("metadata") or {})
                    completed = {"state": "done", "model": result.get("model", SPECS[candidate]["name"]),
                                 "backend": result.get("backend", "unknown"),
                                 "input_sha256": self.job["source"]["sha256"],
                                 "inference_ms": response["inference_ms"], "load_ms": worker.load_ms,
                                 "instances": instances, "visible_keypoints": shown, "valid_depth_keypoints": measured,
                                 "image_url": f"/comparisons/{identifier}/{candidate}.jpg", "error": None,
                                 "metadata": result_metadata}
                    with self.lock:
                        self.job["results"][candidate] = completed
                except Exception as exc:
                    LOG.exception("Comparison model %s failed", candidate)
                    with self.lock:
                        self.job["results"][candidate].update(
                            state="skipped" if self.cancel_event.is_set() else "error",
                            error=None if self.cancel_event.is_set() else str(exc))
                self._save(directory)
            with self.lock:
                cancelled = self.cancel_event.is_set()
                self.job.update(state="cancelled" if cancelled else "done", busy=False,
                                finished_at=datetime.now().astimezone().isoformat())
                for result in self.job["results"].values():
                    if result["state"] in ("queued", "loading", "running"):
                        result["state"] = "skipped"
                # Persist final state before another run can replace self.job.
                self._save(directory)
        except Exception as exc:
            LOG.exception("Comparison job failed")
            with self.lock:
                self.job.update(state="error", busy=False, error=str(exc),
                                finished_at=datetime.now().astimezone().isoformat())
                self._save(directory)

    def cancel(self):
        with self.lock:
            self.cancel_event.set()
            for worker in self.workers.values():
                worker.close()
        return {"cancel_requested": True}

    def close(self):
        self.closed = True
        self.cancel()
