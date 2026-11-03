from __future__ import annotations

import io
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from threading import Lock, Thread
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from numpy.typing import NDArray
from PIL import Image


class PreviewState:
    """Latest-frame mailbox. Rendering and camera application stay on the owner thread."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._images: dict[str, bytes] = {}
        self._status: dict[str, Any] = {"phase": "starting", "frames": 0}
        self._camera: dict[str, float] | None = None

    def publish(self, overview: NDArray[np.uint8], wrist: NDArray[np.uint8], **status: Any) -> None:
        images = {}
        for name, pixels in (("overview", overview), ("wrist", wrist)):
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="JPEG", quality=85)
            images[name] = buffer.getvalue()
        json.dumps(status, allow_nan=False)
        with self._lock:
            self._images = images
            self._status.update(status)
            self._status["frames"] += 1

    def update(self, **status: Any) -> None:
        json.dumps(status, allow_nan=False)
        with self._lock:
            self._status.update(status)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def image(self, name: str) -> bytes | None:
        with self._lock:
            return self._images.get(name)

    def set_camera(self, values: dict[str, Any]) -> None:
        bounds = {"azimuth": (-360, 360), "elevation": (-89, -5), "distance": (0.3, 4)}
        if not values or set(values) - set(bounds):
            raise ValueError("Unsupported camera fields")
        camera = {}
        for name, value in values.items():
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError("Camera values must be finite numbers")
            low, high = bounds[name]
            if not low <= value <= high:
                raise ValueError(f"Camera {name} outside bounds")
            camera[name] = float(value)
        with self._lock:
            self._camera = {**(self._camera or {}), **camera}

    def take_camera(self) -> dict[str, float] | None:
        with self._lock:
            camera, self._camera = self._camera, None
            return camera


class WebViewer:
    """Loopback HTTP server, intended for local use or an SSH tunnel."""

    def __init__(self, state: PreviewState, port: int = 8000) -> None:
        html = files("robot_dock.visualization").joinpath("viewer.html").read_bytes()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def reply(self, code: int, body: bytes, content_type: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                path = urlsplit(self.path).path
                if path == "/":
                    self.reply(200, html, "text/html; charset=utf-8")
                elif path == "/status":
                    self.reply(200, json.dumps(state.status()).encode(), "application/json")
                elif path in ("/frame/overview.jpg", "/frame/wrist.jpg"):
                    image = state.image(path.split("/")[-1].split(".")[0])
                    self.reply(
                        200 if image else 503,
                        image or b"Waiting for renderer",
                        "image/jpeg" if image else "text/plain",
                    )
                else:
                    self.reply(404, b"Not found", "text/plain")

            def do_POST(self) -> None:
                if self.path != "/camera":
                    self.reply(404, b"Not found", "text/plain")
                    return
                origin = self.headers.get("Origin")
                if origin is not None and origin != "http://" + self.headers.get("Host", ""):
                    self.reply(403, b"Cross-origin camera request rejected", "text/plain")
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if (
                        not 0 < length <= 1024
                        or self.headers.get_content_type() != "application/json"
                    ):
                        raise ValueError("Expected a small JSON object")
                    values = json.loads(self.rfile.read(length))
                    if not isinstance(values, dict):
                        raise ValueError("Expected an object")
                    if state.status().get("phase") in ("finished", "closed"):
                        self.reply(409, b"Run finished; camera is frozen", "text/plain")
                        return
                    state.set_camera(values)
                except (ValueError, TypeError):
                    self.reply(400, b"Invalid camera command", "text/plain")
                    return
                self.reply(202, b'{"queued":true}', "application/json")

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_port
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)
