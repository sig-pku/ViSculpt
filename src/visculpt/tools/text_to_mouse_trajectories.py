"""LangGraph Tool for drawing English text inside a SAM3 mask."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Literal

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from visculpt.bridge import JsonValue
from visculpt.vision import (
    MaskTrajectoryFitConfig,
    TextTrajectoryConfig,
    TextTrajectoryGenerationError,
    generate_text_mouse_trajectories,
)


class TextToMouseTrajectoriesInput(BaseModel):
    """English text, exact font family, and an existing SAM3 mask."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=1,
        max_length=256,
        description="English ASCII text to draw.",
    )
    font_name: str = Field(
        min_length=1,
        max_length=128,
        description="Exact family name of a locally available font.",
    )
    mask_path: str = Field(
        min_length=1,
        description="Absolute cleaned-mask path from segment_with_sam3.",
    )
    scale_tier: Literal["SMALL", "MEDIUM", "LARGE"] = Field(
        default="MEDIUM",
        description=(
            "Semantic scale tier for text within the mask, used to compute "
            "adaptive boundary clearance."
        ),
    )

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        """Normalize spaces and reject non-English characters."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must not be empty")
        if any(ord(character) < 32 or ord(character) > 126 for character in stripped):
            raise ValueError(
                "text supports printable English ASCII characters only"
            )
        return " ".join(stripped.split())

    @field_validator("font_name")
    @classmethod
    def normalize_font_name(cls, value: str) -> str:
        """Require a family name instead of a font file path."""
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("font_name must not be empty")
        if any(character in normalized for character in ("/", "\\", "\0")):
            raise ValueError("font_name must not be a file path")
        return normalized

    @field_validator("scale_tier", mode="before")
    @classmethod
    def normalize_scale_tier(cls, value: object) -> object:
        """Accept case-insensitive scale-tier names."""
        return value.strip().upper() if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_mask_path(self) -> TextToMouseTrajectoriesInput:
        """Require an explicit existing local mask file."""
        path = Path(
            os.path.expandvars(os.path.expanduser(self.mask_path))
        )
        if not path.is_absolute() or not path.is_file():
            raise ValueError(
                "mask_path must reference an existing absolute file"
            )
        return self


def create_text_to_mouse_trajectories_tool(
    *,
    config: TextTrajectoryConfig | None = None,
    fit_config: MaskTrajectoryFitConfig | None = None,
) -> StructuredTool:
    """Create the deterministic mask-constrained text trajectory Tool."""
    settings = config or TextTrajectoryConfig()

    def invoke(
        text: str,
        font_name: str,
        mask_path: str,
        scale_tier: Literal["SMALL", "MEDIUM", "LARGE"] = "MEDIUM",
    ) -> dict[str, JsonValue]:
        try:
            result = generate_text_mouse_trajectories(
                text=text,
                font_name=font_name,
                mask_path=mask_path,
                scale_tier=scale_tier,
                config=settings,
                fit_config=fit_config,
            )
        except TextTrajectoryGenerationError as error:
            raise ToolException(
                _text_trajectory_error("generation_error", str(error))
            ) from error
        return {
            "input": {
                "text": text,
                "font_name": font_name,
                "mask_path": str(Path(mask_path).expanduser().resolve()),
                "scale_tier": scale_tier,
            },
            "result": result.as_payload(),
        }

    async def ainvoke(
        text: str,
        font_name: str,
        mask_path: str,
        scale_tier: Literal["SMALL", "MEDIUM", "LARGE"] = "MEDIUM",
    ) -> dict[str, JsonValue]:
        return await asyncio.to_thread(
            invoke,
            text,
            font_name,
            mask_path,
            scale_tier,
        )

    return StructuredTool.from_function(
        func=invoke,
        coroutine=ainvoke,
        name="text_to_mouse_trajectories",
        description=(
            "Skeletonize solid English glyphs from a local font into centerline "
            "mouse trajectories and deterministically search horizontal and "
            "vertical layouts. SMALL, MEDIUM, and LARGE tiers control adaptive "
            "clearance while keeping all trajectories inside and near the center "
            "of the cleaned SAM 3 mask."
        ),
        args_schema=TextToMouseTrajectoriesInput,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: _text_trajectory_error(
            "invalid_tool_input",
            "text must be printable English ASCII, font_name must be a "
            "family name, mask_path must be an existing absolute image, and "
            "scale_tier must be SMALL, MEDIUM, or LARGE",
        ),
    )


def _text_trajectory_error(error_type: str, message: str) -> str:
    return json.dumps(
        {
            "text_trajectory_error": {
                "type": error_type,
                "message": message,
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
