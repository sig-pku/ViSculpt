"""LangGraph tool for configuring Blender Sculpt settings."""

from __future__ import annotations

import asyncio
import json
import math
import time
from enum import StrEnum
from typing import cast
from uuid import uuid4

from langchain_core.tools import StructuredTool, ToolException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from visculpt.bridge import (
    BlenderRpcBridgeError,
    BlenderRpcClient,
    BlenderRpcConfig,
    JsonValue,
)

_BRUSH_ACTIVATION_MAX_ATTEMPTS = 12
_BRUSH_ACTIVATION_RETRY_DELAY = 0.25


class PoseDeformationTarget(StrEnum):
    """Deformation targets supported by Blender's Pose Sculpt brush."""

    GEOMETRY = "GEOMETRY"
    CLOTH_SIM = "CLOTH_SIM"


class PoseRotationOrigins(StrEnum):
    """Rotation-origin modes supported by Blender's Pose Sculpt brush."""

    TOPOLOGY = "TOPOLOGY"
    FACE_SETS = "FACE_SETS"
    FACE_SETS_FK = "FACE_SETS_FK"


class SculptStrokeMethod(StrEnum):
    """Stroke placement modes exposed by Blender Sculpt brushes."""

    DOTS = "DOTS"
    DRAG_DOT = "DRAG_DOT"
    SPACE = "SPACE"
    AIRBRUSH = "AIRBRUSH"
    ANCHORED = "ANCHORED"
    LINE = "LINE"
    CURVE = "CURVE"


class SetSculptSettingsInput(BaseModel):
    """Input schema shown to the Agent model."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    sculpt_brush: str = Field(
        min_length=1,
        max_length=128,
        description=(
            "Sculpt brush name, such as Draw, Clay Strips, Crease, or Smooth."
        ),
    )
    brush_size: int = Field(
        ge=1,
        le=10_000,
        description="Blender View Brush Size in pixel diameter, from 1 to 10000.",
    )
    brush_strength: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Brush Strength from 0 to 1; finite values above 1 are clamped to 1."
        ),
    )
    brush_direction: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Exact Direction value exposed by get_state for the selected local "
            "brush; pass null or omit it when Direction is unsupported."
        ),
    )
    dyntopo_enabled: bool = Field(
        description="Whether to enable Dyntopo.",
    )
    dyntopo_detail_size: float = Field(
        ge=0.5,
        le=40.0,
        description="Dyntopo Relative Detail Size from 0.5 to 40.",
    )
    use_unified_size: bool = Field(
        default=False,
        description="Whether to use Unified Size; defaults to false.",
    )
    use_unified_strength: bool = Field(
        default=False,
        description="Whether to use Unified Strength; defaults to false.",
    )
    use_size_pressure: bool = Field(
        default=False,
        description="Whether to use Size Pressure; defaults to false.",
    )
    use_strength_pressure: bool = Field(
        default=False,
        description="Whether to use Strength Pressure; defaults to false.",
    )
    stroke_method: SculptStrokeMethod | None = Field(
        default=None,
        description="Optional Blender Stroke Method; omit to preserve it.",
    )
    brush_spacing_percent: int | None = Field(
        default=None,
        ge=1,
        le=1_000,
        strict=True,
        description="Optional Brush Spacing as a diameter percentage, 1 to 1000.",
    )
    use_space_attenuation: bool | None = Field(
        default=None,
        strict=True,
        description="Optional Space Attenuation; omit to preserve it.",
    )
    auto_smooth_factor: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional Auto-Smooth from 0 to 1; omit to preserve it.",
    )
    deformation_target: PoseDeformationTarget = Field(
        default=PoseDeformationTarget.GEOMETRY,
        description=(
            "Pose brush Deformation Target, applied only to an active Pose brush; "
            "defaults to Geometry."
        ),
    )
    rotation_origins: PoseRotationOrigins = Field(
        default=PoseRotationOrigins.FACE_SETS,
        description=(
            "Pose brush Rotation Origins, applied only to an active Pose brush; "
            "defaults to Face Sets."
        ),
    )
    pose_origin_offset: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Pose Origin Offset from 0 to 2; defaults to 0.",
    )
    smooth_iterations: int = Field(
        default=100,
        ge=0,
        le=100,
        strict=True,
        description="Pose Smooth Iterations from 0 to 100; defaults to 100.",
    )
    pose_ik_segments: int = Field(
        default=1,
        ge=1,
        le=20,
        strict=True,
        description="Pose IK Segments from 1 to 20; defaults to 1.",
    )
    connected_only: bool = Field(
        default=False,
        strict=True,
        description=(
            "Pose Connected Only; defaults to false so virtual adjacency can "
            "cross nearby disconnected mesh islands."
        ),
    )
    max_element_distance: float = Field(
        default=0.1,
        ge=0.0,
        le=10.0,
        description=(
            "Pose Max Element Distance from 0 to 10; defaults to 0.1."
        ),
    )

    @field_validator("sculpt_brush")
    @classmethod
    def normalize_sculpt_brush(cls, value: str) -> str:
        """Match the Add-on's bounded brush-name validation."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("sculpt_brush must not be blank")
        if any(ord(character) < 32 for character in stripped):
            raise ValueError("sculpt_brush contains unsupported characters")
        return stripped

    @field_validator("brush_direction", mode="before")
    @classmethod
    def normalize_brush_direction(cls, value: object) -> object:
        """Normalize a Blender Direction identifier when one is supplied."""
        if isinstance(value, str):
            direction = value.strip().upper()
            if not direction or any(
                ord(character) < 32 for character in direction
            ):
                raise ValueError("brush_direction is invalid")
            return direction
        return value

    @field_validator("brush_strength", mode="before")
    @classmethod
    def clamp_brush_strength(cls, value: object) -> object:
        """Clamp a finite oversized Strength to Blender's standard range."""
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and value > 1.0
        ):
            return 1.0
        return value

    @field_validator("deformation_target", mode="before")
    @classmethod
    def normalize_deformation_target(cls, value: object) -> object:
        """Accept Blender identifiers and matching UI labels."""
        if isinstance(value, str):
            identifier = _enum_identifier(value)
            return {
                "CLOTH_SIMULATION": PoseDeformationTarget.CLOTH_SIM.value,
            }.get(identifier, identifier)
        return value

    @field_validator("rotation_origins", mode="before")
    @classmethod
    def normalize_rotation_origins(cls, value: object) -> object:
        """Accept Blender identifiers and matching UI labels."""
        if isinstance(value, str):
            return _enum_identifier(value)
        return value

    @field_validator("stroke_method", mode="before")
    @classmethod
    def normalize_stroke_method(cls, value: object) -> object:
        """Accept Blender identifiers and matching UI labels."""
        if isinstance(value, str):
            return _enum_identifier(value)
        return value

    @field_validator(
        "pose_origin_offset",
        "smooth_iterations",
        "pose_ik_segments",
        "max_element_distance",
        "auto_smooth_factor",
        mode="before",
    )
    @classmethod
    def reject_boolean_numeric_settings(cls, value: object) -> object:
        """Prevent JSON booleans from becoming numeric settings."""
        if isinstance(value, bool):
            raise ValueError("Numeric settings must not be booleans")
        return value


def create_set_sculpt_settings_tool(
    *,
    client: BlenderRpcClient | None = None,
    config: BlenderRpcConfig | None = None,
) -> StructuredTool:
    """Create a LangGraph-compatible Blender Sculpt settings Tool."""
    if client is not None and config is not None:
        raise ValueError("client and config cannot both be provided")
    rpc_client = client or BlenderRpcClient(config)

    def invoke(
        sculpt_brush: str,
        brush_size: int,
        brush_strength: float,
        dyntopo_enabled: bool,
        dyntopo_detail_size: float,
        brush_direction: str | None = None,
        use_unified_size: bool = False,
        use_unified_strength: bool = False,
        use_size_pressure: bool = False,
        use_strength_pressure: bool = False,
        stroke_method: SculptStrokeMethod | None = None,
        brush_spacing_percent: int | None = None,
        use_space_attenuation: bool | None = None,
        auto_smooth_factor: float | None = None,
        deformation_target: PoseDeformationTarget = (
            PoseDeformationTarget.GEOMETRY
        ),
        rotation_origins: PoseRotationOrigins = (
            PoseRotationOrigins.FACE_SETS
        ),
        pose_origin_offset: float = 0.0,
        smooth_iterations: int = 100,
        pose_ik_segments: int = 1,
        connected_only: bool = False,
        max_element_distance: float = 0.1,
    ) -> JsonValue:
        try:
            return _execute_settings_transaction(
                rpc_client,
                sculpt_brush=sculpt_brush,
                brush_size=brush_size,
                brush_strength=brush_strength,
                brush_direction=brush_direction,
                dyntopo_enabled=dyntopo_enabled,
                dyntopo_detail_size=dyntopo_detail_size,
                use_unified_size=use_unified_size,
                use_unified_strength=use_unified_strength,
                use_size_pressure=use_size_pressure,
                use_strength_pressure=use_strength_pressure,
                stroke_method=stroke_method,
                brush_spacing_percent=brush_spacing_percent,
                use_space_attenuation=use_space_attenuation,
                auto_smooth_factor=auto_smooth_factor,
                deformation_target=deformation_target,
                rotation_origins=rotation_origins,
                pose_origin_offset=pose_origin_offset,
                smooth_iterations=smooth_iterations,
                pose_ik_segments=pose_ik_segments,
                connected_only=connected_only,
                max_element_distance=max_element_distance,
            )
        except BlenderRpcBridgeError as error:
            raise ToolException(error.as_json()) from error

    async def ainvoke(
        sculpt_brush: str,
        brush_size: int,
        brush_strength: float,
        dyntopo_enabled: bool,
        dyntopo_detail_size: float,
        brush_direction: str | None = None,
        use_unified_size: bool = False,
        use_unified_strength: bool = False,
        use_size_pressure: bool = False,
        use_strength_pressure: bool = False,
        stroke_method: SculptStrokeMethod | None = None,
        brush_spacing_percent: int | None = None,
        use_space_attenuation: bool | None = None,
        auto_smooth_factor: float | None = None,
        deformation_target: PoseDeformationTarget = (
            PoseDeformationTarget.GEOMETRY
        ),
        rotation_origins: PoseRotationOrigins = (
            PoseRotationOrigins.FACE_SETS
        ),
        pose_origin_offset: float = 0.0,
        smooth_iterations: int = 100,
        pose_ik_segments: int = 1,
        connected_only: bool = False,
        max_element_distance: float = 0.1,
    ) -> JsonValue:
        try:
            return await asyncio.to_thread(
                _execute_settings_transaction,
                rpc_client,
                sculpt_brush=sculpt_brush,
                brush_size=brush_size,
                brush_strength=brush_strength,
                brush_direction=brush_direction,
                dyntopo_enabled=dyntopo_enabled,
                dyntopo_detail_size=dyntopo_detail_size,
                use_unified_size=use_unified_size,
                use_unified_strength=use_unified_strength,
                use_size_pressure=use_size_pressure,
                use_strength_pressure=use_strength_pressure,
                stroke_method=stroke_method,
                brush_spacing_percent=brush_spacing_percent,
                use_space_attenuation=use_space_attenuation,
                auto_smooth_factor=auto_smooth_factor,
                deformation_target=deformation_target,
                rotation_origins=rotation_origins,
                pose_origin_offset=pose_origin_offset,
                smooth_iterations=smooth_iterations,
                pose_ik_segments=pose_ik_segments,
                connected_only=connected_only,
                max_element_distance=max_element_distance,
            )
        except BlenderRpcBridgeError as error:
            raise ToolException(error.as_json()) from error

    return StructuredTool.from_function(
        func=invoke,
        coroutine=ainvoke,
        name="set_blender_sculpt_settings",
        description=(
            "Set Blender sculpt brush type, Size, Strength, Direction, "
            "Use Unified Size/Strength、Use Size/Strength Pressure，"
            "optional Stroke Method, Spacing, Space Attenuation, Auto-Smooth, "
            "Dyntopo, and Relative Detail Size. An active Pose brush also receives "
            "Deformation Target, Rotation Origins, Origin Offset, Smooth "
            "Iterations, IK Segments, Connected Only, and "
            "Max Element Distance。"
            "All four Unified/Pressure switches default to false and Strength is "
            "limited to 0..1. Direction must come from "
            "get_blender_sculpt_brush_capabilities; use null when unsupported. "
            "The active mesh must already be in Sculpt Mode."
        ),
        args_schema=SetSculptSettingsInput,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: _invalid_input_error(),
    )


def _execute_settings(
    client: BlenderRpcClient,
    *,
    sculpt_brush: str,
    brush_size: int,
    brush_strength: float,
    brush_direction: str | None,
    dyntopo_enabled: bool,
    dyntopo_detail_size: float,
    use_unified_size: bool,
    use_unified_strength: bool,
    use_size_pressure: bool,
    use_strength_pressure: bool,
    stroke_method: SculptStrokeMethod | None,
    brush_spacing_percent: int | None,
    use_space_attenuation: bool | None,
    auto_smooth_factor: float | None,
    deformation_target: PoseDeformationTarget,
    rotation_origins: PoseRotationOrigins,
    pose_origin_offset: float,
    smooth_iterations: int,
    pose_ik_segments: int,
    connected_only: bool,
    max_element_distance: float,
) -> JsonValue:
    responses: list[JsonValue] = []
    completed_methods: list[JsonValue] = []

    activation_succeeded = False
    active_brush_type: str | None = None
    for attempt in range(1, _BRUSH_ACTIVATION_MAX_ATTEMPTS + 1):
        response = client.send(
            _build_request(
                method="activate_sculpt_brush",
                params={"brush": sculpt_brush},
            )
        )
        responses.append(
            {
                "method": "activate_sculpt_brush",
                "attempt": attempt,
                "response": response,
            }
        )
        if not _has_json_rpc_error(response):
            activation_succeeded = True
            active_brush_type = _activated_sculpt_brush_type(response)
            completed_methods.append("activate_sculpt_brush")
            break
        if not _is_retryable_brush_activation_error(response):
            break
        if attempt < _BRUSH_ACTIVATION_MAX_ATTEMPTS:
            time.sleep(_BRUSH_ACTIVATION_RETRY_DELAY)

    if not activation_succeeded:
        return _error_result(
            failed_method="activate_sculpt_brush",
            completed_methods=completed_methods,
            responses=responses,
        )

    pose_brush_active = (
        active_brush_type == "POSE"
        if active_brush_type is not None
        else sculpt_brush.casefold() == "pose"
    )
    pose_settings = (
        _pose_brush_params(
            deformation_target=deformation_target,
            rotation_origins=rotation_origins,
            pose_origin_offset=pose_origin_offset,
            smooth_iterations=smooth_iterations,
            pose_ik_segments=pose_ik_segments,
            connected_only=connected_only,
            max_element_distance=max_element_distance,
        )
        if pose_brush_active
        else {}
    )
    steps: tuple[tuple[str, dict[str, JsonValue]], ...] = (
        ("set_use_unified_size", {"enabled": use_unified_size}),
        ("set_use_unified_strength", {"enabled": use_unified_strength}),
        ("set_use_size_pressure", {"enabled": use_size_pressure}),
        ("set_use_strength_pressure", {"enabled": use_strength_pressure}),
        (
            "set_sculpt_brush",
            _sculpt_brush_params(
                brush_size=brush_size,
                brush_strength=brush_strength,
                brush_direction=brush_direction,
                stroke_method=stroke_method,
                brush_spacing_percent=brush_spacing_percent,
                use_space_attenuation=use_space_attenuation,
                auto_smooth_factor=auto_smooth_factor,
                pose_settings=pose_settings,
            ),
        ),
        (
            "set_dyntopo",
            {
                "enabled": dyntopo_enabled,
                "detail_size": dyntopo_detail_size,
            },
        ),
    )
    for method, params in steps:
        response = client.send(_build_request(method=method, params=params))
        responses.append(
            {
                "method": method,
                "attempt": 1,
                "response": response,
            }
        )
        if _has_json_rpc_error(response):
            return _error_result(
                failed_method=method,
                completed_methods=completed_methods,
                responses=responses,
            )
        completed_methods.append(method)

    applied_settings: dict[str, JsonValue] = {
        "sculpt_brush": sculpt_brush,
        "brush_size": brush_size,
        "brush_strength": brush_strength,
        "brush_direction": brush_direction,
        "use_unified_size": use_unified_size,
        "use_unified_strength": use_unified_strength,
        "use_size_pressure": use_size_pressure,
        "use_strength_pressure": use_strength_pressure,
        "dyntopo_enabled": dyntopo_enabled,
        "dyntopo_detail_size": dyntopo_detail_size,
    }
    applied_settings.update(
        {
            name: value
            for name, value in (
                (
                    "stroke_method",
                    _enum_value(stroke_method)
                    if stroke_method is not None
                    else None,
                ),
                ("brush_spacing_percent", brush_spacing_percent),
                ("use_space_attenuation", use_space_attenuation),
                ("auto_smooth_factor", auto_smooth_factor),
            )
            if value is not None
        }
    )
    applied_settings.update(pose_settings)
    return {
        "status": "success",
        "completed_methods": completed_methods,
        "rpc_responses": responses,
        "applied_settings": applied_settings,
        "direction_mapping": {
            "requested": brush_direction,
            "rpc_brush_direction": brush_direction,
            "stroke_brush_toggle": (
                "SMOOTH" if brush_direction == "SMOOTH" else "None"
            ),
        },
    }


def _execute_settings_transaction(
    client: BlenderRpcClient,
    *,
    sculpt_brush: str,
    brush_size: int,
    brush_strength: float,
    brush_direction: str | None,
    dyntopo_enabled: bool,
    dyntopo_detail_size: float,
    use_unified_size: bool,
    use_unified_strength: bool,
    use_size_pressure: bool,
    use_strength_pressure: bool,
    stroke_method: SculptStrokeMethod | None,
    brush_spacing_percent: int | None,
    use_space_attenuation: bool | None,
    auto_smooth_factor: float | None,
    deformation_target: PoseDeformationTarget,
    rotation_origins: PoseRotationOrigins,
    pose_origin_offset: float,
    smooth_iterations: int,
    pose_ik_segments: int,
    connected_only: bool,
    max_element_distance: float,
) -> JsonValue:
    """Apply one strict, compensating Blender-side settings transaction."""
    pose_settings = _pose_brush_params(
        deformation_target=deformation_target,
        rotation_origins=rotation_origins,
        pose_origin_offset=pose_origin_offset,
        smooth_iterations=smooth_iterations,
        pose_ik_segments=pose_ik_segments,
        connected_only=connected_only,
        max_element_distance=max_element_distance,
    )
    brush_params = _sculpt_brush_params(
        brush_size=brush_size,
        brush_strength=brush_strength,
        brush_direction=brush_direction,
        stroke_method=stroke_method,
        brush_spacing_percent=brush_spacing_percent,
        use_space_attenuation=use_space_attenuation,
        auto_smooth_factor=auto_smooth_factor,
        pose_settings=(
            pose_settings if "pose" in sculpt_brush.casefold() else {}
        ),
    )
    transaction_params: dict[str, JsonValue] = {
        "sculpt_brush": sculpt_brush,
        "brush": brush_params,
        "use_unified_size": use_unified_size,
        "use_unified_strength": use_unified_strength,
        "use_size_pressure": use_size_pressure,
        "use_strength_pressure": use_strength_pressure,
        "dyntopo": {
            "enabled": dyntopo_enabled,
            "detail_size": dyntopo_detail_size,
        },
    }
    responses: list[JsonValue] = []
    for attempt in range(1, _BRUSH_ACTIVATION_MAX_ATTEMPTS + 1):
        request = _build_request(
            method="set_sculpt_settings",
            params=transaction_params,
        )
        response = client.send(request)
        responses.append(
            {
                "method": "set_sculpt_settings",
                "attempt": attempt,
                "response": response,
            }
        )
        validation_error = _json_rpc_validation_error(
            response,
            expected_id=str(request["id"]),
        )
        if validation_error is not None:
            return _error_result(
                failed_method="set_sculpt_settings",
                completed_methods=[],
                responses=[*responses, {"validation_error": validation_error}],
            )
        if _has_json_rpc_error(response):
            if (
                _is_retryable_brush_activation_error(response)
                and attempt < _BRUSH_ACTIVATION_MAX_ATTEMPTS
            ):
                time.sleep(_BRUSH_ACTIVATION_RETRY_DELAY)
                continue
            return _error_result(
                failed_method="set_sculpt_settings",
                completed_methods=[],
                responses=responses,
            )
        result = response.get("result")
        if (
            not isinstance(result, dict)
            or result.get("transaction") != "sculpt-settings/v1"
            or result.get("status") != "committed"
        ):
            return _error_result(
                failed_method="set_sculpt_settings",
                completed_methods=[],
                responses=[
                    *responses,
                    {
                        "validation_error": (
                            "RPC result did not confirm committed "
                            "sculpt-settings/v1"
                        )
                    },
                ],
            )
        applied_settings: dict[str, JsonValue] = {
            "sculpt_brush": sculpt_brush,
            "brush_size": brush_size,
            "brush_strength": brush_strength,
            "brush_direction": brush_direction,
            "use_unified_size": use_unified_size,
            "use_unified_strength": use_unified_strength,
            "use_size_pressure": use_size_pressure,
            "use_strength_pressure": use_strength_pressure,
            "dyntopo_enabled": dyntopo_enabled,
            "dyntopo_detail_size": dyntopo_detail_size,
        }
        applied_settings.update(
            {
                name: value
                for name, value in (
                    (
                        "stroke_method",
                        _enum_value(stroke_method)
                        if stroke_method is not None
                        else None,
                    ),
                    ("brush_spacing_percent", brush_spacing_percent),
                    ("use_space_attenuation", use_space_attenuation),
                    ("auto_smooth_factor", auto_smooth_factor),
                )
                if value is not None
            }
        )
        transaction_sculpt = result.get("sculpt")
        active_brush = (
            transaction_sculpt.get("brush")
            if isinstance(transaction_sculpt, dict)
            else None
        )
        if (
            isinstance(active_brush, dict)
            and active_brush.get("sculpt_brush_type") == "POSE"
        ):
            applied_settings.update(pose_settings)
        return {
            "status": "success",
            "completed_methods": ["set_sculpt_settings"],
            "rpc_responses": responses,
            "applied_settings": applied_settings,
            "direction_mapping": {
                "requested": brush_direction,
                "rpc_brush_direction": brush_direction,
                "stroke_brush_toggle": (
                    "SMOOTH" if brush_direction == "SMOOTH" else "None"
                ),
            },
            "transaction": cast(dict[str, JsonValue], result),
        }
    raise AssertionError("unreachable Sculpt settings retry state")


def _json_rpc_validation_error(
    response: JsonValue,
    *,
    expected_id: str,
) -> str | None:
    """Return a strict JSON-RPC envelope error without accepting ambiguity."""
    if not isinstance(response, dict):
        return "response must be an object"
    if response.get("jsonrpc") != "2.0":
        return 'jsonrpc must equal "2.0"'
    if response.get("id") != expected_id:
        return "response id does not match the request"
    has_result = "result" in response
    has_error = "error" in response
    if has_result == has_error:
        return "response must contain exactly one of result or error"
    if has_error and not isinstance(response.get("error"), dict):
        return "error must be an object"
    if has_result and not isinstance(response.get("result"), dict):
        return "result must be an object"
    return None


def _sculpt_brush_params(
    *,
    brush_size: int,
    brush_strength: float,
    brush_direction: str | None,
    stroke_method: SculptStrokeMethod | None,
    brush_spacing_percent: int | None,
    use_space_attenuation: bool | None,
    auto_smooth_factor: float | None,
    pose_settings: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Map Agent semantics to Blender's actual Brush RNA fields."""
    params: dict[str, JsonValue] = {
        "size": brush_size,
        "strength": brush_strength,
    }
    if brush_direction is not None:
        params["direction"] = brush_direction
    if stroke_method is not None:
        params["stroke_method"] = _enum_value(stroke_method)
    if brush_spacing_percent is not None:
        params["spacing"] = brush_spacing_percent
    if use_space_attenuation is not None:
        params["use_space_attenuation"] = use_space_attenuation
    if auto_smooth_factor is not None:
        params["auto_smooth_factor"] = auto_smooth_factor
    params.update(pose_settings)
    return params


def _pose_brush_params(
    *,
    deformation_target: PoseDeformationTarget | str,
    rotation_origins: PoseRotationOrigins | str,
    pose_origin_offset: float,
    smooth_iterations: int,
    pose_ik_segments: int,
    connected_only: bool,
    max_element_distance: float,
) -> dict[str, JsonValue]:
    """Map Pose UI semantics to the Add-on's canonical RPC identifiers."""
    return {
        "deformation_target": _enum_value(deformation_target),
        "rotation_origins": _enum_value(rotation_origins),
        "pose_origin_offset": pose_origin_offset,
        "smooth_iterations": smooth_iterations,
        "pose_ik_segments": pose_ik_segments,
        "connected_only": connected_only,
        "max_element_distance": max_element_distance,
    }


def _activated_sculpt_brush_type(response: JsonValue) -> str | None:
    """Read the actual activated brush type from the RPC response."""
    if not isinstance(response, dict):
        return None
    result = response.get("result")
    if not isinstance(result, dict):
        return None
    brush = result.get("brush")
    if not isinstance(brush, dict):
        return None
    value = brush.get("sculpt_brush_type")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().upper()


def _enum_identifier(value: str) -> str:
    """Normalize a short Blender enum identifier or UI label."""
    return "_".join(value.strip().upper().replace("-", " ").split())


def _enum_value(value: StrEnum | str) -> str:
    """Return the canonical string for validated StrEnum input."""
    return value.value if isinstance(value, StrEnum) else value


def _build_request(
    *,
    method: str,
    params: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    return {
        "jsonrpc": "2.0",
        "id": f"set-sculpt-settings-{method}-{uuid4().hex}",
        "method": method,
        "params": params,
    }


def _has_json_rpc_error(response: JsonValue) -> bool:
    return (
        isinstance(response, dict)
        and isinstance(response.get("error"), dict)
    )


def _is_retryable_brush_activation_error(response: JsonValue) -> bool:
    if not isinstance(response, dict):
        return False
    error = response.get("error")
    if not isinstance(error, dict) or error.get("code") != -32022:
        return False
    data = error.get("data")
    if not isinstance(data, dict):
        return False
    if data.get("retryable") is True:
        return True
    cause = data.get("cause")
    return isinstance(cause, dict) and cause.get("retryable") is True


def _error_result(
    *,
    failed_method: str,
    completed_methods: list[JsonValue],
    responses: list[JsonValue],
) -> dict[str, JsonValue]:
    return {
        "status": "error",
        "failed_method": failed_method,
        "completed_methods": completed_methods,
        "rpc_responses": responses,
    }


def _invalid_input_error() -> str:
    return json.dumps(
        {
            "bridge_error": {
                "type": "invalid_tool_input",
                "message": (
                    "Sculpt settings Tool input must provide a supported "
                    "brush, size, 0-to-1 strength, optional runtime "
                    "direction, optional "
                    "Unified/Pressure toggles, Dyntopo state, detail size, "
                    "and valid Pose-only settings"
                ),
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
