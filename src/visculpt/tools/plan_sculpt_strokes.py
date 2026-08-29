"""LangGraph tool for deterministic Sculpt stroke planning."""

from __future__ import annotations

import json
import os
from pathlib import Path

from langchain_core.tools import StructuredTool, ToolException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from visculpt.bridge import JsonValue
from visculpt.vision import (
    SculptStrokePlanningError,
    plan_sculpt_strokes,
)

class ScreenshotRegionInput(BaseModel):
    """VIEW_3D region fields returned by get_blender_screenshot."""

    model_config = ConfigDict(extra="ignore")

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ScreenshotCoordinateScaleInput(BaseModel):
    """Framebuffer-to-region scale returned by get_blender_screenshot."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    x: float = Field(gt=0.0)
    y: float = Field(gt=0.0)


class ScreenshotMetadataInput(BaseModel):
    """Screenshot fields required for operator-coordinate mapping."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    width: int = Field(gt=0)
    height: int = Field(gt=0)
    region: ScreenshotRegionInput
    coordinate_scale: ScreenshotCoordinateScaleInput
    window_index: int | None = Field(default=None, ge=0)
    area_index: int | None = Field(default=None, ge=0)


class PlanSculptStrokesInput(BaseModel):
    """Input contract for deterministic trajectory generation."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    cleaned_mask_path: str = Field(min_length=1)
    sculpt_brush: str = Field(min_length=1, max_length=128)
    brush_size: int = Field(
        ge=1,
        le=10_000,
        description="Blender View Brush Size as a pixel diameter.",
    )
    brush_strength: float = Field(ge=0.0, le=1.0)
    brush_direction: str | None = Field(default=None, max_length=64)
    screenshot_metadata: ScreenshotMetadataInput | None = None
    output_dir: str | None = None

    @field_validator("cleaned_mask_path", "sculpt_brush", "output_dir")
    @classmethod
    def strip_string_value(cls, value: str | None) -> str | None:
        """Normalize Agent-provided strings."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped

    @field_validator("brush_direction", mode="before")
    @classmethod
    def normalize_direction(cls, value: object) -> object:
        """Accept case-insensitive direction names."""
        if isinstance(value, str):
            direction = value.strip().upper()
            if not direction or any(
                ord(character) < 32 for character in direction
            ):
                raise ValueError("brush_direction is invalid")
            return direction
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> PlanSculptStrokesInput:
        """Validate the cleaned mask and output path contract."""
        mask = _expanded_path(self.cleaned_mask_path)
        if not mask.is_absolute() or not mask.is_file():
            raise ValueError(
                "cleaned_mask_path must reference an existing absolute file"
            )
        if self.output_dir is not None:
            output = _expanded_path(self.output_dir)
            if not output.is_absolute():
                raise ValueError("output_dir must be absolute")
            if output.exists() and not output.is_dir():
                raise ValueError("output_dir must reference a directory")
        return self


def create_plan_sculpt_strokes_tool() -> StructuredTool:
    """Create the standalone deterministic stroke-planning Tool."""

    def invoke(
        cleaned_mask_path: str,
        sculpt_brush: str,
        brush_size: int,
        brush_strength: float,
        brush_direction: str | None = None,
        screenshot_metadata: ScreenshotMetadataInput | None = None,
        output_dir: str | None = None,
    ) -> dict[str, JsonValue]:
        try:
            metadata = (
                screenshot_metadata.model_dump()
                if screenshot_metadata is not None
                else None
            )
            result = plan_sculpt_strokes(
                cleaned_mask_path=cleaned_mask_path,
                sculpt_brush=sculpt_brush,
                brush_size=brush_size,
                brush_strength=brush_strength,
                brush_direction=brush_direction,
                screenshot_metadata=metadata,
                output_dir=output_dir,
            )
            return {
                "input": {
                    "cleaned_mask_path": cleaned_mask_path,
                    "sculpt_brush": sculpt_brush,
                    "brush_size": brush_size,
                    "brush_strength": brush_strength,
                    "brush_direction": brush_direction,
                    "screenshot_metadata": metadata,
                },
                "result": result.as_payload(),
            }
        except SculptStrokePlanningError as error:
            raise ToolException(_planning_error(error)) from error

    return StructuredTool.from_function(
        func=invoke,
        name="plan_sculpt_strokes",
        description=(
            "Generate deterministic uniform-coverage trajectories from a cleaned "
            "semantic mask and final sculpt parameters, including operator calls "
            "ready for bpy.ops.sculpt.brush_stroke."
        ),
        args_schema=PlanSculptStrokesInput,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: _invalid_input_error(),
    )


def _expanded_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value)))


def _invalid_input_error() -> str:
    return json.dumps(
        {
            "stroke_planning_error": {
                "type": "invalid_tool_input",
                "message": (
                    "Stroke planning requires an existing absolute cleaned "
                    "mask path, valid Sculpt settings, matching screenshot "
                    "metadata, and an optional absolute output directory"
                ),
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _planning_error(error: SculptStrokePlanningError) -> str:
    return json.dumps(
        {
            "stroke_planning_error": {
                "type": "planning_error",
                "message": str(error),
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
