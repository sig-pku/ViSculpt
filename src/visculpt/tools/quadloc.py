"""LangGraph Tool for QuadLoc four-way visual localization."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from visculpt.bridge import JsonValue
from visculpt.vision.quadloc import (
    QuadLocConfig,
    QuadLocError,
    QuadLocLocator,
    QuadLocVlm,
)


class QuadLocInput(BaseModel):
    """Two-field input contract shown to the Agent model."""

    model_config = ConfigDict(extra="forbid")

    image_path: str = Field(
        min_length=1,
        description="Absolute local path of the screenshot to localize.",
    )
    location_description: str = Field(
        min_length=1,
        max_length=512,
        description="Text describing the target operation location.",
    )
    output_dir: str | None = Field(
        default=None,
        description="Optional absolute artifact directory owned by the workflow.",
    )

    @field_validator("image_path")
    @classmethod
    def strip_image_path(cls, value: str) -> str:
        """Strip path edges without changing valid internal spaces."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped

    @field_validator("location_description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        """Normalize repeated whitespace in the target description."""
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_image_path(self) -> QuadLocInput:
        """Reject ambiguous or missing local screenshot paths."""
        image_path = Path(
            os.path.expandvars(os.path.expanduser(self.image_path))
        )
        if not image_path.is_absolute():
            raise ValueError("image_path must be absolute")
        if not image_path.is_file():
            raise ValueError("image_path must reference an existing file")
        if self.output_dir is not None:
            output_dir = Path(
                os.path.expandvars(os.path.expanduser(self.output_dir))
            )
            if not output_dir.is_absolute():
                raise ValueError("output_dir must be absolute")
        return self


def create_quadloc_tool(
    *,
    llm: QuadLocVlm,
    segment_tool: BaseTool,
    config: QuadLocConfig | None = None,
    workdir: Path | None = None,
) -> StructuredTool:
    """Create a LangGraph-compatible QuadLoc localization Tool."""
    # Delay import to avoid a tools/workflow package initialization cycle.
    from visculpt.workflow.prompts import (
        QUADLOC_SYSTEM_PROMPT,
        quadloc_user_prompt,
    )

    locator = QuadLocLocator(
        llm=llm,
        segment_tool=segment_tool,
        system_prompt=QUADLOC_SYSTEM_PROMPT,
        prompt_builder=quadloc_user_prompt,
        config=config,
        workdir=workdir,
    )

    def invoke(
        image_path: str,
        location_description: str,
        output_dir: str | None = None,
    ) -> dict[str, JsonValue]:
        try:
            result = locator.locate(
                image_path=image_path,
                location_description=location_description,
                artifact_root=output_dir,
            )
            return {
                "input": {
                    "image_path": image_path,
                    "location_description": location_description,
                    "output_dir": output_dir,
                },
                "result": result.as_payload(),
            }
        except QuadLocError as error:
            raise ToolException(_quadloc_error(error)) from error

    async def ainvoke(
        image_path: str,
        location_description: str,
        output_dir: str | None = None,
    ) -> dict[str, JsonValue]:
        try:
            result = await asyncio.to_thread(
                locator.locate,
                image_path=image_path,
                location_description=location_description,
                artifact_root=output_dir,
            )
            return {
                "input": {
                    "image_path": image_path,
                    "location_description": location_description,
                    "output_dir": output_dir,
                },
                "result": result.as_payload(),
            }
        except QuadLocError as error:
            raise ToolException(_quadloc_error(error)) from error

    return StructuredTool.from_function(
        func=invoke,
        coroutine=ainvoke,
        name="quadloc",
        description=(
            "Recursively localize an operation point with QuadLoc quadrant search, "
            "then correct it into the cleaned SAM 3 model mask and return "
            "screenshot pixel coordinates (x, y)."
        ),
        args_schema=QuadLocInput,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: _invalid_input_error(),
    )


def _invalid_input_error() -> str:
    return json.dumps(
        {
            "quadloc_error": {
                "type": "invalid_tool_input",
                "message": (
                    "QuadLoc input must use an existing absolute image path "
                    "and a non-empty location description"
                ),
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _quadloc_error(error: QuadLocError) -> str:
    return json.dumps(
        {
            "quadloc_error": {
                "type": _error_type(error),
                "message": str(error),
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _error_type(error: QuadLocError) -> str:
    name = type(error).__name__
    mapping = {
        "QuadLocInputError": "input_error",
        "QuadLocVlmError": "vlm_error",
        "QuadLocSearchError": "search_exhausted",
        "QuadLocNoModelMaskError": "no_model_mask",
        "QuadLocSegmentationError": "segmentation_error",
        "QuadLocArtifactError": "artifact_error",
    }
    return mapping.get(name, "quadloc_error")
