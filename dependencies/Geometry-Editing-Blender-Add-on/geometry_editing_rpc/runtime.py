"""Lifecycle management for the Blender RPC runtime."""

from __future__ import annotations

import threading
from typing import Any

from .blender_api import BlenderRpcApi
from .dispatcher import (
    BlenderMainThreadDispatcher,
    DispatcherClosedError,
    DispatcherTimeoutError,
)
from .protocol import JsonRpcError, JsonRpcRouter
from .server import LocalRpcServer


class RpcRuntime:
    """Own the dispatcher, API implementation, and HTTP transport."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._dispatcher: BlenderMainThreadDispatcher | None = None
        self._server: LocalRpcServer | None = None
        self._request_timeout = 30.0

    def start(
        self,
        *,
        port: int,
        access_token: str,
        request_timeout: float,
    ) -> dict[str, Any]:
        """Start the server, returning its effective configuration."""
        with self._lock:
            if self._server is not None and self._server.is_running:
                raise RuntimeError("RPC server is already running")

            dispatcher = BlenderMainThreadDispatcher()
            api = BlenderRpcApi()
            self._request_timeout = request_timeout

            def resolver(method: str, params: Any) -> Any:
                try:
                    return dispatcher.submit(
                        lambda: api.call(method, params),
                        timeout=self._request_timeout,
                    )
                except DispatcherTimeoutError as error:
                    raise JsonRpcError(
                        -32001,
                        "Blender main-thread execution timed out",
                        {
                            "may_have_completed": error.may_have_completed,
                        },
                    ) from error
                except DispatcherClosedError as error:
                    raise JsonRpcError(
                        -32002,
                        "Blender RPC service is stopping",
                    ) from error

            try:
                server = LocalRpcServer(
                    JsonRpcRouter(resolver),
                    port=port,
                    access_token=access_token,
                )
                server.start()
            except Exception:
                dispatcher.close()
                raise

            self._dispatcher = dispatcher
            self._server = server
            return self.status()

    def stop(self) -> None:
        """Stop all RPC activity and release the listening socket."""
        with self._lock:
            dispatcher = self._dispatcher
            server = self._server
            self._dispatcher = None
            self._server = None

        if dispatcher is not None:
            dispatcher.close()
        if server is not None:
            server.stop()

    def restart(
        self,
        *,
        port: int,
        access_token: str,
        request_timeout: float,
    ) -> dict[str, Any]:
        """Restart the runtime with a new configuration."""
        self.stop()
        return self.start(
            port=port,
            access_token=access_token,
            request_timeout=request_timeout,
        )

    def status(self) -> dict[str, Any]:
        """Return transport status without touching Blender state."""
        with self._lock:
            server = self._server
            running = server is not None and server.is_running
            return {
                "running": running,
                "host": server.host if server is not None else "127.0.0.1",
                "port": server.port if server is not None else None,
                "endpoint": (
                    f"http://{server.host}:{server.port}/rpc"
                    if server is not None
                    else None
                ),
                "request_timeout": self._request_timeout,
            }


runtime = RpcRuntime()
