"""HTTP JSON-RPC client for the Blender Add-on."""

from __future__ import annotations

import json
import socket
from http.client import HTTPResponse
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from .config import BlenderRpcConfig
from .errors import (
    BlenderRpcHttpError,
    BlenderRpcRequestError,
    BlenderRpcResponseError,
    BlenderRpcTimeoutError,
    BlenderRpcTransportError,
)
from .types import JsonValue


class BlenderRpcClient:
    """Forward JSON messages to the local Blender RPC Server."""

    def __init__(self, config: BlenderRpcConfig | None = None) -> None:
        self.config = config or BlenderRpcConfig.from_env()

    def send(self, payload: JsonValue) -> JsonValue:
        """Send one JSON-RPC message or batch and return the Server response."""
        request_body = _encode_request(payload)
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "agentic-geometry-editing-bridge/0.1",
        }
        if self.config.access_token:
            headers["Authorization"] = (
                f"Bearer {self.config.access_token}"
            )
        request = Request(
            self.config.endpoint,
            data=request_body,
            headers=headers,
            method="POST",
        )

        try:
            opener = build_opener(ProxyHandler({}))
            with opener.open(
                request,
                timeout=self.config.timeout,
            ) as response:
                return self._read_success_response(response)
        except HTTPError as error:
            raise self._build_http_error(error) from error
        except (TimeoutError, socket.timeout) as error:
            raise BlenderRpcTimeoutError(
                f"Blender RPC request timed out after "
                f"{self.config.timeout:g} seconds"
            ) from error
        except URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                raise BlenderRpcTimeoutError(
                    f"Blender RPC request timed out after "
                    f"{self.config.timeout:g} seconds"
                ) from error
            raise BlenderRpcTransportError(
                f"Cannot connect to Blender RPC Server at "
                f"{self.config.endpoint}: {error.reason}"
            ) from error

    def forward_json(self, request_json: str) -> str:
        """Forward a JSON string and return the response as a JSON string."""
        if not isinstance(request_json, str):
            raise BlenderRpcRequestError("request_json must be a string")
        try:
            payload = json.loads(
                request_json,
                parse_constant=_reject_nonstandard_number,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise BlenderRpcRequestError(
                f"request_json is not valid JSON: {error}"
            ) from error

        response = self.send(cast(JsonValue, payload))
        return json.dumps(
            response,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    def _read_success_response(self, response: HTTPResponse) -> JsonValue:
        status = response.status
        body = self._read_limited(response)
        if status == 204:
            if body:
                raise BlenderRpcResponseError(
                    "Blender RPC Server returned a body with HTTP 204"
                )
            return None
        if not body:
            raise BlenderRpcResponseError(
                "Blender RPC Server returned an empty response"
            )
        return _decode_response(body)

    def _build_http_error(self, error: HTTPError) -> BlenderRpcHttpError:
        try:
            body = self._read_limited(error)
        finally:
            error.close()
        server_response = _decode_http_error_body(body)
        return BlenderRpcHttpError(
            status=error.code,
            reason=str(error.reason),
            server_response=server_response,
        )

    def _read_limited(self, response: HTTPResponse | HTTPError) -> bytes:
        raw_content_length = response.headers.get("Content-Length")
        if raw_content_length is not None:
            try:
                content_length = int(raw_content_length)
            except ValueError:
                content_length = -1
            if content_length > self.config.max_response_bytes:
                raise BlenderRpcResponseError(
                    "Blender RPC Server response exceeds "
                    f"{self.config.max_response_bytes} bytes"
                )

        body = response.read(self.config.max_response_bytes + 1)
        if len(body) > self.config.max_response_bytes:
            raise BlenderRpcResponseError(
                "Blender RPC Server response exceeds "
                f"{self.config.max_response_bytes} bytes"
            )
        return body


def _encode_request(payload: JsonValue) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise BlenderRpcRequestError(
            f"request payload is not JSON serializable: {error}"
        ) from error


def _decode_response(body: bytes) -> JsonValue:
    try:
        response = json.loads(
            body.decode("utf-8"),
            parse_constant=_reject_nonstandard_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise BlenderRpcResponseError(
            f"Blender RPC Server returned invalid JSON: {error}"
        ) from error
    return cast(JsonValue, response)


def _decode_http_error_body(body: bytes) -> JsonValue:
    if not body:
        return None
    try:
        return _decode_response(body)
    except BlenderRpcResponseError:
        return body.decode("utf-8", errors="replace")


def _reject_nonstandard_number(value: str) -> None:
    raise ValueError(f"non-standard JSON number {value}")
