"""LangGraph Tool for selecting one semantic mask component with a VLM."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from visculpt.bridge import JsonValue
from visculpt.vision.mask_component_selection import (
    MaskComponentSelectionError,
    MaskComponentVlm,
    select_mask_component,
)


class SelectMaskComponentInput(BaseModel):
    """Source image, cleaned mask, and semantic target description."""

    model_config = ConfigDict(extra="forbid")

    image_path: str = Field(
        min_length=1,
        description="Absolute Blender screenshot path aligned to the cleaned mask.",
    )
    cleaned_mask_path: str = Field(
        min_length=1,
        description="Absolute cleaned-mask path with one or more components.",
    )
    part_description: str = Field(
        min_length=1,
        max_length=512,
        description="Description of the model part to select among components.",
    )
    overlay_opacity: float = Field(default=0.45, ge=0.0, le=1.0)
    output_dir: str | None = Field(
        default=None,
        description="Optional absolute output directory.",
    )

    @field_validator("image_path", "cleaned_mask_path")
    @classmethod
    def normalize_path(cls, value: str) -> str:
        """Strip path edges without changing valid internal spaces."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("path must not be empty")
        return normalized

    @field_validator("part_description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        """Normalize repeated whitespace in the semantic target."""
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("part_description must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_local_paths(self) -> SelectMaskComponentInput:
        """Require unambiguous existing local input files."""
        for field_name in ("image_path", "cleaned_mask_path"):
            value = getattr(self, field_name)
            path = Path(os.path.expandvars(os.path.expanduser(value)))
            if not path.is_absolute():
                raise ValueError(f"{field_name} must be absolute")
            if not path.is_file():
                raise ValueError(f"{field_name} must reference a file")
        if self.output_dir is not None:
            output = Path(
                os.path.expandvars(os.path.expanduser(self.output_dir))
            )
            if not output.is_absolute():
                raise ValueError("output_dir must be absolute")
        return self


def create_select_mask_component_tool(
    *,
    llm: MaskComponentVlm,
    llm_role: str = "translator",
    workdir: Path | None = None,
) -> StructuredTool:
    """Create a sync/async semantic component-selection Tool."""
    role = llm_role.strip()
    if not role:
        raise ValueError("llm_role must not be empty")

    # Delay import to avoid a tools/workflow package initialization cycle.
    from visculpt.workflow.prompts import (
        MASK_COMPONENT_SELECTOR_SYSTEM_PROMPT,
        mask_component_selector_user_prompt,
    )

    def invoke(
        image_path: str,
        cleaned_mask_path: str,
        part_description: str,
        overlay_opacity: float = 0.45,
        output_dir: str | None = None,
    ) -> dict[str, JsonValue]:
        try:
            result = select_mask_component(
                llm=llm,
                image_path=image_path,
                cleaned_mask_path=cleaned_mask_path,
                part_description=part_description,
                system_prompt=MASK_COMPONENT_SELECTOR_SYSTEM_PROMPT,
                prompt_builder=mask_component_selector_user_prompt,
                llm_role=role,
                overlay_opacity=overlay_opacity,
                output_dir=output_dir,
                workdir=workdir,
            )
            return {
                "input": {
                    "image_path": image_path,
                    "cleaned_mask_path": cleaned_mask_path,
                    "part_description": part_description,
                    "overlay_opacity": overlay_opacity,
                    "output_dir": output_dir,
                },
                "result": result.as_payload(),
            }
        except MaskComponentSelectionError as error:
            raise ToolException(_tool_error(error)) from error

    async def ainvoke(
        image_path: str,
        cleaned_mask_path: str,
        part_description: str,
        overlay_opacity: float = 0.45,
        output_dir: str | None = None,
    ) -> dict[str, JsonValue]:
        try:
            result = await asyncio.to_thread(
                select_mask_component,
                llm=llm,
                image_path=image_path,
                cleaned_mask_path=cleaned_mask_path,
                part_description=part_description,
                system_prompt=MASK_COMPONENT_SELECTOR_SYSTEM_PROMPT,
                prompt_builder=mask_component_selector_user_prompt,
                llm_role=role,
                overlay_opacity=overlay_opacity,
                output_dir=output_dir,
                workdir=workdir,
            )
            return {
                "input": {
                    "image_path": image_path,
                    "cleaned_mask_path": cleaned_mask_path,
                    "part_description": part_description,
                    "overlay_opacity": overlay_opacity,
                    "output_dir": output_dir,
                },
                "result": result.as_payload(),
            }
        except MaskComponentSelectionError as error:
            raise ToolException(_tool_error(error)) from error

    return StructuredTool.from_function(
        func=invoke,
        coroutine=ainvoke,
        name="select_mask_component",
        description=(
            "Label every connected component in a cleaned mask, ask the Translator "
            "VLM to select the component matching the part description, and output "
            "a mask containing only that component."
        ),
        args_schema=SelectMaskComponentInput,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: _invalid_input_error(),
    )


def _tool_error(error: MaskComponentSelectionError) -> str:
    mapping = {
        "MaskComponentSelectionInputError": "input_error",
        "MaskComponentSelectionVlmError": "vlm_error",
        "MaskComponentSelectionArtifactError": "artifact_error",
    }
    return json.dumps(
        {
            "mask_component_selection_error": {
                "type": mapping.get(type(error).__name__, "tool_error"),
                "message": str(error),
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _invalid_input_error() -> str:
    return json.dumps(
        {
            "mask_component_selection_error": {
                "type": "invalid_tool_input",
                "message": (
                    "Mask component selection requires an existing absolute "
                    "image, an existing absolute cleaned mask, a non-empty "
                    "part description, and an optional absolute output dir"
                ),
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
