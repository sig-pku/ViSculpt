"""Validation shared by the Pose Brush RPC implementation and tests."""

from __future__ import annotations

import re
from typing import Any

from .protocol import JsonRpcError

POSE_BRUSH_PARAMETER_DEFAULTS: dict[str, Any] = {
    "deformation_target": "GEOMETRY",
    "rotation_origins": "FACE_SETS",
    "pose_origin_offset": 0.0,
    "smooth_iterations": 100,
    "pose_ik_segments": 1,
    "connected_only": False,
    "max_element_distance": 0.1,
}

POSE_BRUSH_PARAMETER_RNA_NAMES = {
    "deformation_target": "deform_target",
    "rotation_origins": "pose_origin_type",
    "pose_origin_offset": "pose_offset",
    "smooth_iterations": "pose_smooth_iterations",
    "pose_ik_segments": "pose_ik_segments",
    "connected_only": "use_connected_only",
    "max_element_distance": "disconnected_distance_max",
}

POSE_BRUSH_PARAMETERS = frozenset(POSE_BRUSH_PARAMETER_RNA_NAMES)
DEFORMATION_TARGET_VALUES = ("GEOMETRY", "CLOTH_SIM")
ROTATION_ORIGIN_VALUES = ("TOPOLOGY", "FACE_SETS", "FACE_SETS_FK")


def parse_pose_brush_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """Return validated Pose-only settings explicitly present in params."""
    parsed: dict[str, Any] = {}
    if "deformation_target" in params:
        parsed["deformation_target"] = _enum_identifier(
            params["deformation_target"],
            name="deformation_target",
            supported_values=DEFORMATION_TARGET_VALUES,
        )
    if "rotation_origins" in params:
        parsed["rotation_origins"] = _enum_identifier(
            params["rotation_origins"],
            name="rotation_origins",
            supported_values=ROTATION_ORIGIN_VALUES,
        )
    if "pose_origin_offset" in params:
        parsed["pose_origin_offset"] = _number(
            params["pose_origin_offset"],
            name="pose_origin_offset",
            minimum=0.0,
            maximum=2.0,
        )
    if "smooth_iterations" in params:
        parsed["smooth_iterations"] = _integer(
            params["smooth_iterations"],
            name="smooth_iterations",
            minimum=0,
            maximum=100,
        )
    if "pose_ik_segments" in params:
        parsed["pose_ik_segments"] = _integer(
            params["pose_ik_segments"],
            name="pose_ik_segments",
            minimum=1,
            maximum=20,
        )
    if "connected_only" in params:
        parsed["connected_only"] = _boolean(
            params["connected_only"],
            name="connected_only",
        )
    if "max_element_distance" in params:
        parsed["max_element_distance"] = _number(
            params["max_element_distance"],
            name="max_element_distance",
            minimum=0.0,
            maximum=10.0,
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
    aliases = {
        "CLOTH_SIMULATION": "CLOTH_SIM",
    }
    identifier = aliases.get(identifier, identifier)
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


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise _invalid(f"{name} must be a boolean")
    return value


def _invalid(reason: str, **data: Any) -> JsonRpcError:
    return JsonRpcError(
        -32602,
        "Invalid params",
        {"reason": reason, **data},
    )
