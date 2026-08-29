"""Errors exposed by the Blender RPC bridge."""

from __future__ import annotations

import json

from .types import JsonValue


class BlenderRpcBridgeError(RuntimeError):
    """Base class for bridge failures outside JSON-RPC responses."""

    error_type = "bridge_error"

    def details(self) -> dict[str, JsonValue]:
        """Return structured error details safe for an Agent tool."""
        return {}

    def as_payload(self) -> dict[str, JsonValue]:
        """Return a stable JSON-compatible tool error."""
        error: dict[str, JsonValue] = {
            "type": self.error_type,
            "message": str(self),
        }
        error.update(self.details())
        return {"bridge_error": error}

    def as_json(self) -> str:
        """Serialize the stable tool error."""
        return json.dumps(
            self.as_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
        )


class BlenderRpcRequestError(BlenderRpcBridgeError):
    """The tool supplied a value that cannot be sent as JSON."""

    error_type = "invalid_request_json"


class BlenderRpcTransportError(BlenderRpcBridgeError):
    """The Blender RPC endpoint could not be reached."""

    error_type = "transport_error"


class BlenderRpcTimeoutError(BlenderRpcTransportError):
    """The Blender RPC request exceeded the client timeout."""

    error_type = "timeout"


class BlenderRpcResponseError(BlenderRpcBridgeError):
    """The Server returned an unusable response."""

    error_type = "invalid_server_response"


class BlenderRpcHttpError(BlenderRpcBridgeError):
    """The Server returned a non-success HTTP status."""

    error_type = "http_error"

    def __init__(
        self,
        status: int,
        reason: str,
        server_response: JsonValue | None,
    ) -> None:
        self.status = status
        self.reason = reason
        self.server_response = server_response
        super().__init__(f"Blender RPC Server returned HTTP {status}: {reason}")

    def details(self) -> dict[str, JsonValue]:
        """Preserve the Server reply for the Agent tool."""
        return {
            "status": self.status,
            "server_response": self.server_response,
        }
