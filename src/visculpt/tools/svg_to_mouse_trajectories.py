"""LangGraph Tool that converts SVG line art into mouse trajectories."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from visculpt.bridge import JsonValue
from visculpt.vision import (
    SvgTrajectoryGenerationError,
    generate_svg_mouse_trajectories,
)

from .generate_svg_pattern import (
    SvgPatternValidationError,
    validate_svg_pattern,
)

_MAX_SVG_CHARACTERS = 131_072


class SvgToMouseTrajectoriesInput(BaseModel):
    """Validated SVG markup accepted by the deterministic Tool."""

    model_config = ConfigDict(extra="forbid")

    svg: str = Field(
        min_length=1,
        max_length=_MAX_SVG_CHARACTERS,
        description="Complete 512x512 black-and-white SVG pattern string.",
    )

    @field_validator("svg")
    @classmethod
    def strip_svg(cls, value: str) -> str:
        """Remove surrounding whitespace without modifying SVG content."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped


def create_svg_to_mouse_trajectories_tool(
    *,
    point_spacing_pixels: float = 4.0,
    flattening_spacing_pixels: float = 1.0,
) -> StructuredTool:
    """Create the standalone deterministic SVG trajectory Tool."""
    _validate_factory_settings(
        point_spacing_pixels=point_spacing_pixels,
        flattening_spacing_pixels=flattening_spacing_pixels,
    )

    def invoke(svg: str) -> dict[str, JsonValue]:
        try:
            validated_svg = validate_svg_pattern(svg)
        except SvgPatternValidationError as error:
            raise ToolException(
                _trajectory_error("invalid_svg", str(error))
            ) from error
        try:
            result = generate_svg_mouse_trajectories(
                svg=validated_svg,
                point_spacing_pixels=point_spacing_pixels,
                flattening_spacing_pixels=flattening_spacing_pixels,
            )
        except SvgTrajectoryGenerationError as error:
            raise ToolException(
                _trajectory_error("generation_error", str(error))
            ) from error
        return {
            "input": {
                "media_type": "image/svg+xml",
                "character_count": len(validated_svg),
                "sha256": hashlib.sha256(
                    validated_svg.encode("utf-8")
                ).hexdigest(),
            },
            "result": result.as_payload(),
        }

    async def ainvoke(svg: str) -> dict[str, JsonValue]:
        return await asyncio.to_thread(invoke, svg)

    return StructuredTool.from_function(
        func=invoke,
        coroutine=ainvoke,
        name="svg_to_mouse_trajectories",
        description=(
            "Deterministically convert a validated 512x512 SVG line drawing into "
            "uniformly sampled mouse trajectories. Each path is one press, move, "
            "and release gesture in the SVG top-left-origin 0..512 coordinates."
        ),
        args_schema=SvgToMouseTrajectoriesInput,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: _trajectory_error(
            "invalid_tool_input",
            "svg must be a non-empty validated SVG string and no unknown "
            "fields are allowed",
        ),
    )


def _validate_factory_settings(
    *,
    point_spacing_pixels: float,
    flattening_spacing_pixels: float,
) -> None:
    if (
        not math.isfinite(point_spacing_pixels)
        or point_spacing_pixels <= 0.0
    ):
        raise ValueError("point_spacing_pixels must be finite and positive")
    if (
        not math.isfinite(flattening_spacing_pixels)
        or flattening_spacing_pixels <= 0.0
    ):
        raise ValueError(
            "flattening_spacing_pixels must be finite and positive"
        )
    if flattening_spacing_pixels > point_spacing_pixels:
        raise ValueError(
            "flattening_spacing_pixels must not exceed "
            "point_spacing_pixels"
        )


def _trajectory_error(error_type: str, message: str) -> str:
    return json.dumps(
        {
            "svg_trajectory_error": {
                "type": error_type,
                "message": message,
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
