"""Jetson-local HTTP UI; RealSense capture and TensorRT stay on the Jetson."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import mimetypes
from pathlib import Path
import re
import signal
import threading
from urllib.parse import parse_qs, unquote, urlsplit

from prototype import Application

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("d435i")


def handler_for(app):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            LOG.debug(fmt, *args)

        def send_bytes(self, data, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)

        def send_file(self, path):
            """Stream finalized recordings with byte ranges for seeking."""
            size = path.stat().st_size
            start, end, partial = 0, size-1, False
            value = self.headers.get("Range")
            if value:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
                if not match or not any(match.groups()) or size == 0:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                first, last = match.groups()
                if first:
                    start, end = int(first), min(int(last), size-1) if last else size-1
                else:
                    start, end = max(0, size-int(last)), size-1
                if start >= size or start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                partial = True
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", "text/csv; charset=utf-8" if path.suffix == ".csv" else
                             mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            if path.suffix == ".csv":
                self.send_header("Content-Disposition", 'attachment; filename="dataset.csv"')
            self.send_header("Content-Length", str(max(0, end-start+1)))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command != "HEAD":
                with path.open("rb") as source:
                    source.seek(start)
                    remaining = end-start+1
                    while remaining > 0:
                        block = source.read(min(256*1024, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)

        def do_HEAD(self):
            if urlsplit(self.path).path.startswith("/recordings/"):
                return self.do_GET()
            return self.send_bytes(b"", "text/plain", 405)

        def send_json(self, data, status=200):
            self.send_bytes(json.dumps(data, ensure_ascii=False, allow_nan=False).encode(),
                            "application/json; charset=utf-8", status)

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            try:
                if path == "/api/status":
                    return self.send_json(app.status())
                if path == "/api/imu":
                    query = parse_qs(urlsplit(self.path).query)
                    try:
                        after = int(query.get("after", ["0"])[0])
                        limit = int(query.get("limit", ["512"])[0])
                        if after < 0 or not 1 <= limit <= 4096:
                            raise ValueError()
                    except (TypeError, ValueError):
                        return self.send_json({"ok": False, "error": "after는 0 이상 정수, limit는 1~4096 정수여야 합니다."}, 400)
                    return self.send_json(app.imu_history(after, limit))
                if path == "/api/people":
                    return self.send_json(app.people())
                if path == "/api/recording/status":
                    return self.send_json(app.recorder.status())
                if path == "/api/recordings":
                    return self.send_json({"recordings": app.recorder.list_recordings()})
                if path == "/api/compare/state":
                    return self.send_json(app.comparison.state())
                if path.startswith("/api/compare/job/"):
                    return self.send_json(app.comparison.job_by_id(path[len("/api/compare/job/"):]))
                if path.startswith("/stream/") and path.endswith(".mjpg"):
                    name = path.removeprefix("/stream/").removesuffix(".mjpg")
                    if name not in ("rgb", "depth", "rgbd", "pose", "pose_rgbd"):
                        return self.send_json({"ok": False, "error": "없는 영상입니다."}, 404)
                    return self.stream(name)
                if path.startswith("/recordings/"):
                    root = app.recorder.output_dir.resolve()
                    candidate = (root / path[len("/recordings/"):]).resolve()
                    if (not candidate.is_relative_to(root) or not candidate.is_file() or
                            ".partial." in candidate.name or
                            candidate.suffix.lower() not in (".mp4", ".avi", ".json", ".jsonl", ".csv")):
                        return self.send_json({"ok": False, "error": "녹화 파일을 찾을 수 없습니다."}, 404)
                    return self.send_file(candidate)
                elif path.startswith("/comparisons/"):
                    root, relative = app.comparison.output_dir.resolve(), path[len("/comparisons/"):]
                elif path.startswith("/captures/"):
                    root, relative = app.output_dir.resolve(), path[len("/captures/"):]
                else:
                    relative = "index.html" if path == "/" else "compare.html" if path in ("/compare", "/compare/") else path.lstrip("/")
                    root = (ROOT / "static").resolve()
                candidate = (root / relative).resolve()
                if not candidate.is_relative_to(root) or not candidate.is_file():
                    return self.send_json({"ok": False, "error": "파일을 찾을 수 없습니다."}, 404)
                content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
                if content_type.startswith("text/") or content_type == "application/javascript":
                    content_type += "; charset=utf-8"
                return self.send_bytes(candidate.read_bytes(), content_type)
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
            except Exception:
                LOG.exception("GET failed")
                self.send_json({"ok": False, "error": "요청 처리에 실패했습니다."}, 500)

        def stream(self, name):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store, no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            sequence = -1
            idle_timeouts = 0
            self.connection.settimeout(8)
            while not app._stop.is_set():
                new_sequence, data = app.wait_image(name, sequence)
                if data is None or new_sequence == sequence:
                    idle_timeouts += 1
                    if idle_timeouts >= 4:
                        break
                    continue
                idle_timeouts = 0
                sequence = new_sequence
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                 str(len(data)).encode() + b"\r\nX-Sequence: " +
                                 str(sequence).encode() + b"\r\n\r\n" + data + b"\r\n")
                self.wfile.flush()

        def do_POST(self):
            # Camera controls accept only requests originating at this host.
            origin = self.headers.get("Origin")
            if (origin and urlsplit(origin).netloc != self.headers.get("Host")) or self.headers.get("Sec-Fetch-Site") == "cross-site":
                self.close_connection = True
                return self.send_json({"ok": False, "error": "다른 사이트의 제어 요청은 허용하지 않습니다."}, 403)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 <= size <= 8192:
                    self.close_connection = True
                    return self.send_json({"ok": False, "error": "요청 크기가 너무 큽니다."}, 413)
                payload = json.loads(self.rfile.read(size) or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("JSON 객체가 필요합니다.")
                path = urlsplit(self.path).path
                with app.actions:
                    if app._stop.is_set():
                        raise ValueError("프로그램이 종료 중입니다.")
                    if path == "/api/camera/start":
                        app.invalidate_pose()
                        result = app.camera.start()
                    elif path == "/api/camera/stop":
                        app.recorder.stop(reason="camera_stopped")
                        app.invalidate_pose()
                        result = app.camera.stop()
                        with app.condition:
                            app._images = {}
                            app._processed = None
                            app.pose.update(people=0, fps=0.)
                    elif path == "/api/settings":
                        result = app.configure(payload)
                    elif path == "/api/snapshot":
                        return self.send_json({"ok": True, **app.snapshot()})
                    elif path == "/api/recording/start":
                        result = app.start_recording()
                    elif path == "/api/recording/stop":
                        result = app.recorder.stop(reason="user")
                    elif path == "/api/compare/run":
                        return self.send_json({"ok": True, "job": app.comparison.run(payload)}, 202)
                    elif path == "/api/compare/cancel":
                        return self.send_json({"ok": True, **app.comparison.cancel()})
                    else:
                        return self.send_json({"ok": False, "error": "없는 명령입니다."}, 404)
                self.send_json({"ok": True, "result": result})
            except (ValueError, json.JSONDecodeError) as error:
                self.send_json({"ok": False, "error": str(error)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                LOG.exception("POST failed")
                self.send_json({"ok": False, "error": str(error)}, 500)
    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser(description="RealSense D435i RGB-D pose prototype")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolov8n-pose.engine")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "captures")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--backend", choices=("auto", "tensorrt", "opencv"), default="auto")
    parser.add_argument("--preview-fps", type=int, default=15)
    parser.add_argument("--auto-start", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.preview_fps <= 30:
        parser.error("preview-fps must be between 1 and 30")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Application(args.model, args.output_dir, args.input_size, args.backend, args.preview_fps)
    try:
        server = Server((args.host, args.port), handler_for(app))
    except Exception:
        app.close()
        raise
    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if args.auto_start:
        app.camera.start()
    LOG.info("D435i Prototype Lab http://%s:%s", args.host, args.port)
    try:
        server.serve_forever(poll_interval=.25)
    finally:
        app.close()
        server.server_close()


if __name__ == "__main__":
    main()
