"""Configuration for the local Blender RPC bridge."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

PORT_ENV = "AGENTIC_GEOMETRY_BLENDER_RPC_PORT"
TOKEN_ENV = "AGENTIC_GEOMETRY_BLENDER_RPC_ACCESS_TOKEN"
TIMEOUT_ENV = "AGENTIC_GEOMETRY_BLENDER_RPC_TIMEOUT"
MAX_RESPONSE_BYTES_ENV = "AGENTIC_GEOMETRY_BLENDER_RPC_MAX_RESPONSE_BYTES"
DEFAULT_PORT = 8765
DEFAULT_ACCESS_TOKEN = ""
DEFAULT_TIMEOUT = 35.0
DEFAULT_MAX_RESPONSE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class BlenderRpcConfig:
    """Connection settings for the loopback Blender RPC Server."""

    port: int = DEFAULT_PORT
    access_token: str = DEFAULT_ACCESS_TOKEN
    timeout: float = DEFAULT_TIMEOUT
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES

    def __post_init__(self) -> None:
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 1 <= self.port <= 65535
        ):
            raise ValueError("port must be between 1 and 65535")
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number")
        if (
            isinstance(self.max_response_bytes, bool)
            or not isinstance(self.max_response_bytes, int)
            or self.max_response_bytes <= 0
        ):
            raise ValueError("max_response_bytes must be positive")
        if not isinstance(self.access_token, str):
            raise TypeError("access_token must be a string")
        if "\r" in self.access_token or "\n" in self.access_token:
            raise ValueError("access_token must not contain line breaks")

    @property
    def endpoint(self) -> str:
        """Return the fixed loopback JSON-RPC endpoint."""
        return f"http://127.0.0.1:{self.port}/rpc"

    @classmethod
    def from_env(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> BlenderRpcConfig:
        """Load optional connection overrides from environment variables."""
        values = os.environ if environment is None else environment
        return cls(
            port=_read_int(values, PORT_ENV, DEFAULT_PORT),
            access_token=values.get(TOKEN_ENV, DEFAULT_ACCESS_TOKEN),
            timeout=_read_float(values, TIMEOUT_ENV, DEFAULT_TIMEOUT),
            max_response_bytes=_read_int(
                values,
                MAX_RESPONSE_BYTES_ENV,
                DEFAULT_MAX_RESPONSE_BYTES,
            ),
        )


def _read_int(
    environment: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    raw_value = environment.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def _read_float(
    environment: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    raw_value = environment.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
