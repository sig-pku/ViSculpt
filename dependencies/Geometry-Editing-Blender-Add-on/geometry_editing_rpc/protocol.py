"""JSON-RPC 2.0 protocol handling independent of Blender."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

JSON_VALUE = Any
Resolver = Callable[[str, JSON_VALUE], JSON_VALUE]


class JsonRpcError(Exception):
    """An error that can be returned to a JSON-RPC client."""

    def __init__(
        self,
        code: int,
        message: str,
        data: JSON_VALUE | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class JsonRpcRouter:
    """Validate JSON-RPC requests and invoke a resolver."""

    def __init__(self, resolver: Resolver, *, max_batch_size: int = 64) -> None:
        self._resolver = resolver
        self._max_batch_size = max_batch_size

    def handle(self, payload: JSON_VALUE) -> JSON_VALUE | None:
        """Handle one request or a batch and return a JSON-compatible response."""
        if isinstance(payload, list):
            if not payload:
                return self.error_response(
                    None,
                    JsonRpcError(-32600, "Invalid Request"),
                )
            if len(payload) > self._max_batch_size:
                return self.error_response(
                    None,
                    JsonRpcError(
                        -32600,
                        "Invalid Request",
                        {"reason": "Batch is too large"},
                    ),
                )

            responses = [
                response
                for item in payload
                if (response := self._handle_one(item)) is not None
            ]
            return responses or None

        return self._handle_one(payload)

    def _handle_one(self, request: JSON_VALUE) -> dict[str, JSON_VALUE] | None:
        if not isinstance(request, dict):
            return self.error_response(
                None,
                JsonRpcError(-32600, "Invalid Request"),
            )

        has_id = "id" in request
        request_id = request.get("id")
        if has_id and not self._valid_request_id(request_id):
            return self.error_response(
                None,
                JsonRpcError(
                    -32600,
                    "Invalid Request",
                    {"reason": "id must be a string, number, or null"},
                ),
            )

        if request.get("jsonrpc") != "2.0":
            return self.error_response(
                None,
                JsonRpcError(
                    -32600,
                    "Invalid Request",
                    {"reason": 'jsonrpc must equal "2.0"'},
                ),
            )

        method = request.get("method")
        if not isinstance(method, str) or not method:
            return self.error_response(
                None,
                JsonRpcError(
                    -32600,
                    "Invalid Request",
                    {"reason": "method must be a non-empty string"},
                ),
            )

        params = request.get("params", {})
        if not isinstance(params, (dict, list)):
            if not has_id:
                return None
            return self.error_response(
                request_id,
                JsonRpcError(-32602, "Invalid params"),
            )

        try:
            result = self._resolver(method, params)
        except JsonRpcError as error:
            if not has_id:
                return None
            return self.error_response(request_id, error)
        except Exception as error:  # noqa: BLE001
            if not has_id:
                return None
            return self.error_response(
                request_id,
                JsonRpcError(
                    -32603,
                    "Internal error",
                    {"type": type(error).__name__},
                ),
            )

        if not has_id:
            return None
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }

    @staticmethod
    def error_response(
        request_id: JSON_VALUE,
        error: JsonRpcError,
    ) -> dict[str, JSON_VALUE]:
        """Build a JSON-RPC error response."""
        error_object: dict[str, JSON_VALUE] = {
            "code": error.code,
            "message": error.message,
        }
        if error.data is not None:
            error_object["data"] = error.data
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": error_object,
        }

    @staticmethod
    def parse_error() -> dict[str, JSON_VALUE]:
        """Build the standard JSON-RPC parse error response."""
        return JsonRpcRouter.error_response(
            None,
            JsonRpcError(-32700, "Parse error"),
        )

    @staticmethod
    def _valid_request_id(value: JSON_VALUE) -> bool:
        if value is None or isinstance(value, str):
            return True
        return isinstance(value, (int, float)) and not isinstance(value, bool)
