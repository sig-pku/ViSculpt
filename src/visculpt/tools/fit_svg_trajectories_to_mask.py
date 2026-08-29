"""LangGraph Tool for fitting SVG trajectories inside a SAM3 mask."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Literal

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
    MaskTrajectoryFitConfig,
    MaskTrajectoryFitError,
    fit_svg_trajectories_to_mask,
)


class SvgTrajectoryPointInput(BaseModel):
    """One source SVG coordinate in the inclusive 0..512 canvas."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    x: float = Field(ge=0.0, le=512.0)
    y: float = Field(ge=0.0, le=512.0)


class SvgMouseTrajectoryInput(BaseModel):
    """One complete mouse-down to mouse-up SVG gesture."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)

    id: str = Field(min_length=1, max_length=128)
    source: dict[str, JsonValue] = Field(default_factory=dict)
    closed: bool = False
    points: list[SvgTrajectoryPointInput] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def normalize_identifier(cls, value: str) -> str:
        """Reject empty identifiers after trimming whitespace."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("id must not be empty")
        return normalized


class SvgMouseTrajectoryPlanInput(BaseModel):
    """Relevant subset of svg_to_mouse_trajectories output."""

    model_config = ConfigDict(extra="ignore")

    format: Literal["svg-mouse-trajectories/v1"]
    trajectories: list[SvgMouseTrajectoryInput] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_ids(self) -> SvgMouseTrajectoryPlanInput:
        """Keep gesture identity unambiguous through the transform."""
        identifiers = [item.id for item in self.trajectories]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("trajectory ids must be unique")
        return self


class FitSvgTrajectoriesToMaskInput(BaseModel):
    """SAM3 mask and SVG trajectory plan accepted by the Tool."""

    model_config = ConfigDict(extra="forbid")

    mask_path: str = Field(
        min_length=1,
        description="Absolute cleaned-mask path from segment_with_sam3.",
    )
    trajectory_plan: SvgMouseTrajectoryPlanInput
    scale_tier: Literal["SMALL", "MEDIUM", "LARGE"] = Field(
        default="MEDIUM",
        description=(
            "Semantic pattern scale within the mask, used to compute adaptive "
            "boundary clearance."
        ),
    )

    @field_validator("mask_path")
    @classmethod
    def normalize_mask_path(cls, value: str) -> str:
        """Normalize but do not silently resolve an Agent path."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("mask_path must not be empty")
        return stripped

    @field_validator("scale_tier", mode="before")
    @classmethod
    def normalize_scale_tier(cls, value: object) -> object:
        """Accept case-insensitive scale-tier names."""
        return value.strip().upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_mask_path(self) -> FitSvgTrajectoriesToMaskInput:
        """Require an explicit existing local mask file."""
        path = Path(
            os.path.expandvars(os.path.expanduser(self.mask_path))
        )
        if not path.is_absolute() or not path.is_file():
            raise ValueError(
                "mask_path must reference an existing absolute file"
            )
        return self


def create_fit_svg_trajectories_to_mask_tool(
    *,
    config: MaskTrajectoryFitConfig | None = None,
) -> StructuredTool:
    """Create the deterministic mask-constrained trajectory fitting Tool."""
    settings = config or MaskTrajectoryFitConfig()

    def invoke(
        mask_path: str,
        trajectory_plan: SvgMouseTrajectoryPlanInput,
        scale_tier: Literal["SMALL", "MEDIUM", "LARGE"] = "MEDIUM",
    ) -> dict[str, JsonValue]:
        try:
            result = fit_svg_trajectories_to_mask(
                mask_path=mask_path,
                trajectory_plan=trajectory_plan.model_dump(mode="json"),
                scale_tier=scale_tier,
                config=settings,
            )
        except MaskTrajectoryFitError as error:
            raise ToolException(
                _fit_error("fitting_error", str(error))
            ) from error
        return {
            "input": {
                "mask_path": str(Path(mask_path).expanduser().resolve()),
                "source_format": trajectory_plan.format,
                "trajectory_count": len(trajectory_plan.trajectories),
                "scale_tier": scale_tier,
            },
            "result": result.as_payload(),
        }

    async def ainvoke(
        mask_path: str,
        trajectory_plan: SvgMouseTrajectoryPlanInput,
        scale_tier: Literal["SMALL", "MEDIUM", "LARGE"] = "MEDIUM",
    ) -> dict[str, JsonValue]:
        return await asyncio.to_thread(
            invoke,
            mask_path,
            trajectory_plan,
            scale_tier,
        )

    return StructuredTool.from_function(
        func=invoke,
        coroutine=ainvoke,
        name="fit_svg_trajectories_to_mask",
        description=(
            "Fit every svg_to_mouse_trajectories path inside a cleaned SAM 3 "
            "mask using one rotation, translation, and scale. SMALL, MEDIUM, or "
            "LARGE sets adaptive clearance; prefer zero rotation and the center."
        ),
        args_schema=FitSvgTrajectoriesToMaskInput,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: _fit_error(
            "invalid_tool_input",
            "mask_path must be an existing absolute image and "
            "trajectory_plan must be svg-mouse-trajectories/v1; "
            "scale_tier must be SMALL, MEDIUM, or LARGE",
        ),
    )


def _fit_error(error_type: str, message: str) -> str:
    return json.dumps(
        {
            "trajectory_fit_error": {
                "type": error_type,
                "message": message,
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
