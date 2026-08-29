"""Loopback HTTP transport for JSON-RPC."""

from __future__ import annotations

import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any
from urllib.parse import urlsplit

from .protocol import JsonRpcRouter

SERVICE_NAME = "geometry-editing-blender-rpc"
API_VERSION = "2.26"
DEFAULT_MAX_REQUEST_BYTES = 1_048_576


class _ThreadingHttpServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    router: JsonRpcRouter
    access_token: str
    max_request_bytes: int


class _RequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "GeometryEditingRPC/1.0.0"

    @property
    def rpc_server(self) -> _ThreadingHttpServer:
        return self.server  # type: ignore[return-value]

    def do_GET(self) -> None:
        """Serve the health endpoint."""
        path = self._path()
        if path == "/health":
            self._send_json(
                200,
                {
                    "service": SERVICE_NAME,
                    "api_version": API_VERSION,
                    "status": "ok",
                },
            )
            return
        if path == "/favicon.ico":
            self._send_empty(204)
            return
        if path == "/rpc":
            self._send_json(405, {"error": "Use POST for JSON-RPC"})
            return
        self._send_json(404, {"error": "Not found"})

    def do_HEAD(self) -> None:
        """Return no content for unsupported HEAD requests."""
        self._send_empty(404)

    def do_POST(self) -> None:
        """Handle JSON-RPC requests."""
        if self._path() != "/rpc":
            self._send_json(404, {"error": "Not found"})
            return
        if not self._is_authorized():
            self._send_json(
                401,
                {"error": "Unauthorized"},
                extra_headers={"WWW-Authenticate": "Bearer"},
            )
            return

        content_type = self.headers.get_content_type()
        if content_type != "application/json":
            self._send_json(
                415,
                {"error": "Content-Type must be application/json"},
            )
            return

        raw_length = self.headers.get("Content-Length")
        try:
            content_length = int(raw_length) if raw_length is not None else -1
        except ValueError:
            content_length = -1
        if content_length < 0:
            self._send_json(411, {"error": "Content-Length is required"})
            return
        if content_length > self.rpc_server.max_request_bytes:
            self._send_json(413, {"error": "Request body is too large"})
            return

        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(200, JsonRpcRouter.parse_error())
            return

        response = self.rpc_server.router.handle(payload)
        if response is None:
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            return
        self._send_json(200, response)

    def do_OPTIONS(self) -> None:
        """Reject browser cross-origin probing without enabling CORS."""
        self._send_json(405, {"error": "Method not allowed"})

    def log_message(self, format_string: str, *args: Any) -> None:
        """Keep Blender's console free from per-request HTTP logs."""

    def _path(self) -> str:
        return urlsplit(self.path).path

    def _is_authorized(self) -> bool:
        token = self.rpc_server.access_token
        if not token:
            return True
        actual = self.headers.get("Authorization", "")
        return hmac.compare_digest(actual, f"Bearer {token}")

    def _send_json(
        self,
        status: int,
        payload: Any,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        if extra_headers:
            for name, value in extra_headers.items():
                self.send_header(name, value)
        self.end_headers()
        self.wfile.write(encoded)
        self.close_connection = True

    def _send_empty(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True


class LocalRpcServer:
    """Own the HTTP server and its background accept loop."""

    def __init__(
        self,
        router: JsonRpcRouter,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        access_token: str = "",
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    ) -> None:
        if host != "127.0.0.1":
            raise ValueError("Only the IPv4 loopback address is allowed")
        self._httpd = _ThreadingHttpServer((host, port), _RequestHandler)
        self._httpd.router = router
        self._httpd.access_token = access_token
        self._httpd.max_request_bytes = max_request_bytes
        self._thread: threading.Thread | None = None

    @property
    def host(self) -> str:
        return str(self._httpd.server_address[0])

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start accepting requests on a daemon thread."""
        if self.is_running:
            return
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="geometry-editing-rpc-http",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop accepting requests and release the socket."""
        if self._thread is None:
            self._httpd.server_close()
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2.0)
        self._thread = None
