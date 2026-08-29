"""Validation for deterministic Sculpt brush stroke-quality settings."""

from __future__ import annotations

import re
from typing import Any

from .protocol import JsonRpcError

BRUSH_QUALITY_PARAMETER_RNA_NAMES = {
    "stroke_method": "stroke_method",
    "spacing": "spacing",
    "use_space_attenuation": "use_space_attenuation",
    "auto_smooth_factor": "auto_smooth_factor",
}

BRUSH_QUALITY_PARAMETERS = frozenset(BRUSH_QUALITY_PARAMETER_RNA_NAMES)
STROKE_METHOD_VALUES = (
    "DOTS",
    "DRAG_DOT",
    "SPACE",
    "AIRBRUSH",
    "ANCHORED",
    "LINE",
    "CURVE",
)


def parse_brush_quality_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Return validated quality settings explicitly present in params."""
    parsed: dict[str, Any] = {}
    if "stroke_method" in params:
        parsed["stroke_method"] = _enum_identifier(
            params["stroke_method"],
            name="stroke_method",
            supported_values=STROKE_METHOD_VALUES,
        )
    if "spacing" in params:
        parsed["spacing"] = _integer(
            params["spacing"],
            name="spacing",
            minimum=1,
            maximum=1_000,
        )
    if "use_space_attenuation" in params:
        value = params["use_space_attenuation"]
        if not isinstance(value, bool):
            raise _invalid("use_space_attenuation must be a boolean")
        parsed["use_space_attenuation"] = value
    if "auto_smooth_factor" in params:
        parsed["auto_smooth_factor"] = _number(
            params["auto_smooth_factor"],
            name="auto_smooth_factor",
            minimum=0.0,
            maximum=1.0,
        )
    return parsed


def _enum_identifier(
    value: Any,
    *,
    name: str,
    supported_values: tuple[str, ...],
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _invalid(
            f"{name} must be a non-empty string",
            supported_values=list(supported_values),
        )
    identifier = re.sub(r"[^A-Z0-9]+", "_", value.strip().upper()).strip("_")
    if identifier not in supported_values:
        raise _invalid(
            f"{name} is not supported",
            value=value,
            supported_values=list(supported_values),
        )
    return identifier


def _number(
    value: Any,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not minimum <= float(value) <= maximum
    ):
        raise _invalid(
            f"{name} must be a number in range",
            minimum=minimum,
            maximum=maximum,
        )
    return float(value)


def _integer(
    value: Any,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _invalid(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise _invalid(
            f"{name} is out of range",
            minimum=minimum,
            maximum=maximum,
        )
    return value


def _invalid(reason: str, **data: Any) -> JsonRpcError:
    return JsonRpcError(
        -32602,
        "Invalid params",
        {"reason": reason, **data},
    )
