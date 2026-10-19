"""Persistent model subprocesses; no CUDA context or frame files in the server.

Creation starts loading asynchronously. ``close`` can interrupt loading, a pipe
write or inference, and never waits for the inference serialization lock.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

try:
    from .live_protocol import PREFIX, encode_frame, write_all
except ImportError:
    from live_protocol import PREFIX, encode_frame, write_all

ROOT = Path(__file__).resolve().parent
ENVIRONMENTS = {"yolo": ".venv", "mediapipe": ".mediapipe-venv", "rtmpose": ".venv", "egoforce": ".egoforce-gpu-venv"}


def python_path(identifier):
    override = os.environ.get("LIVE_" + identifier.upper() + "_PYTHON")
    if override:
        return Path(override).expanduser().resolve()
    return ROOT / ENVIRONMENTS[identifier] / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


class LiveEstimator:
    def __init__(self, identifier, model_path=None, input_size=640, backend="auto",
                 startup_timeout=240, inference_timeout=120):
        if identifier not in ENVIRONMENTS:
            raise ValueError("Unknown live model: " + str(identifier))
        if startup_timeout <= 0 or inference_timeout <= 0:
            raise ValueError("Live model timeouts must be positive")
        self.identifier = identifier
        self.startup_timeout = float(startup_timeout)
        self.inference_timeout = float(inference_timeout)
        self.backend = None
        self.model = identifier
        self.load_ms = None
        self._ready_metadata = {}
        self._closed = threading.Event()
        self._ready = threading.Event()
        self._state_lock = threading.RLock()
        self._infer_lock = threading.Lock()
        self._responses = queue.Queue()
        self._ready_error = None
        self._sequence = 0
        self._process = None
        self._log = None
        runtime = python_path(identifier)
        if not runtime.is_file():
            raise RuntimeError(f"{identifier} live runtime is missing: {runtime}")
        (ROOT / "logs").mkdir(exist_ok=True)
        self.log_path = ROOT / "logs" / f"live_{identifier}.log"
        command = [str(runtime), "-u", str(ROOT / "live_worker.py"), "--candidate", identifier,
                   "--input-size", str(input_size), "--backend", str(backend)]
        if model_path is not None:
            command.extend(["--model", str(Path(model_path).expanduser().resolve())])
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
        if identifier == "egoforce":
            # The live EgoForce option explicitly requires GPU execution.
            env["EGOFORCE_DEVICE"] = "cuda"
        extra_libraries = os.environ.get("LIVE_" + identifier.upper() + "_LIBRARY_PATH")
        if extra_libraries:
            env["LD_LIBRARY_PATH"] = extra_libraries + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        with self._state_lock:
            self._log = self.log_path.open("ab", buffering=0)
            try:
                self._process = subprocess.Popen(command, cwd=ROOT, env=env,
                                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                 stderr=self._log, bufsize=0,
                                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            except Exception:
                self._log.close()
                self._log = None
                raise
            process = self._process
        self._reader_thread = threading.Thread(target=self._read, args=(process,), name=f"live-{identifier}-stdout", daemon=True)
        self._reader_thread.start()

    def _read(self, process):
        try:
            for raw in iter(process.stdout.readline, b""):
                line = raw.decode("utf-8", "replace")
                if not line.startswith(PREFIX):
                    continue
                try:
                    message = json.loads(line[len(PREFIX):])
                except (ValueError, TypeError):
                    continue
                if "ready" in message:
                    with self._state_lock:
                        if not message["ready"]:
                            self._ready_error = message.get("error", "Live model initialization failed")
                        else:
                            self.backend = message.get("backend")
                            self.model = message.get("model", self.identifier)
                            self.load_ms = message.get("load_ms")
                            self._ready_metadata = message.get("metadata") or {}
                    self._ready.set()
                else:
                    self._responses.put(message)
        except (OSError, ValueError):
            pass
        finally:
            if not self._ready.is_set():
                self._ready_error = f"Live model exited during loading; see {self.log_path}"
                self._ready.set()
            self._responses.put({"worker_exited": True})

    def _check_open(self):
        if self._closed.is_set():
            raise RuntimeError("Live estimator was closed")

    def _await_ready(self):
        deadline = time.monotonic() + self.startup_timeout
        while not self._ready.wait(.05):
            self._check_open()
            if time.monotonic() >= deadline:
                self.close()
                raise RuntimeError(f"{self.identifier} initialization timed out; see {self.log_path}")
        self._check_open()
        if self._ready_error:
            raise RuntimeError(self._ready_error)

    def prepare(self):
        """Wait for model loading before the caller acquires its source frame."""
        self._await_ready()
        with self._state_lock:
            self._check_open()
            return {"backend": self.backend, "model": self.model, "load_ms": self.load_ms}

    def infer(self, bgr, intrinsics=None, roi=None, confidence=.35, hand_hints=None):
        # One caller at a time; there is no unbounded frame queue. The camera
        # owner supplies its latest captured frame after each inference.
        with self._infer_lock:
            self._check_open()
            self._await_ready()
            self._sequence += 1
            sequence = self._sequence
            header, pixels = encode_frame(bgr, sequence=sequence, intrinsics=intrinsics,
                                          roi=roi, confidence=confidence, hand_hints=hand_hints)
            with self._state_lock:
                self._check_open()
                process = self._process
            if process is None or process.poll() is not None:
                raise RuntimeError(f"Live model process exited; see {self.log_path}")
            deadline = time.monotonic() + self.inference_timeout
            transmitted = threading.Event()
            write_errors = []
            def transmit():
                try:
                    # Keep writes out of the lifecycle lock. A stalled child may
                    # fill its pipe before inference starts; the same deadline
                    # covers that case and close() releases the blocked writer.
                    write_all(process.stdin, header)
                    write_all(process.stdin, pixels)
                    process.stdin.flush()
                except (BrokenPipeError, OSError, ValueError) as exc:
                    write_errors.append(exc)
                finally:
                    transmitted.set()
            threading.Thread(target=transmit, name=f"live-{self.identifier}-input", daemon=True).start()
            while not transmitted.wait(.05):
                self._check_open()
                if time.monotonic() >= deadline:
                    self.close()
                    raise RuntimeError(f"{self.identifier} frame transfer timed out; see {self.log_path}")
            self._check_open()
            if write_errors:
                raise RuntimeError(f"Live model input closed; see {self.log_path}") from write_errors[0]
            while True:
                self._check_open()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.close()
                    raise RuntimeError(f"{self.identifier} inference timed out; see {self.log_path}")
                try:
                    message = self._responses.get(timeout=min(.1, remaining))
                except queue.Empty:
                    continue
                self._check_open()
                if message.get("worker_exited"):
                    raise RuntimeError(f"Live model process exited; see {self.log_path}")
                if message.get("sequence") != sequence:
                    continue
                if not message.get("ok"):
                    raise RuntimeError(message.get("error", "Live model inference failed"))
                result = message["result"]
                result.setdefault("backend", self.backend)
                result.setdefault("model", self.model)
                result.setdefault("metadata", {})
                result["metadata"].update(copy.deepcopy(self._ready_metadata))
                result["metadata"]["worker_inference_ms"] = message["inference_ms"]
                result["metadata"]["transport"] = "lossless raw BGR pipe; no per-frame files"
                result["inference_ms"] = message["inference_ms"]
                self.backend, self.model = result["backend"], result["model"]
                return result

    def close(self):
        with self._state_lock:
            if self._closed.is_set():
                return
            self._closed.set()
            self._ready.set()
            self._responses.put({"worker_exited": True})
            process, self._process = self._process, None
            log, self._log = self._log, None
        if process is not None:
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
                except ProcessLookupError:
                    pass
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass
        if log is not None:
            log.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
