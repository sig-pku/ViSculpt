"""Configuration for the local SAM3 Gradio service."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass

PORT_ENV = "AGENTIC_GEOMETRY_SAM3_GRADIO_PORT"
TIMEOUT_ENV = "AGENTIC_GEOMETRY_SAM3_GRADIO_TIMEOUT"
DEFAULT_PORT = 7860
DEFAULT_TIMEOUT = 300.0


@dataclass(frozen=True, slots=True)
class Sam3GradioConfig:
    """Connection settings for the loopback SAM3 Gradio service."""

    port: int = DEFAULT_PORT
    timeout: float = DEFAULT_TIMEOUT

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

    @property
    def service_url(self) -> str:
        """Return the fixed loopback Gradio service URL."""
        return f"http://127.0.0.1:{self.port}"

    @classmethod
    def from_env(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> Sam3GradioConfig:
        """Load optional connection overrides from environment variables."""
        values = os.environ if environment is None else environment
        return cls(
            port=_read_int(values, PORT_ENV, DEFAULT_PORT),
            timeout=_read_float(values, TIMEOUT_ENV, DEFAULT_TIMEOUT),
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
