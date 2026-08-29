"""LangGraph Tool for SAM3-assisted Blender Face Set segmentation."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from visculpt.bridge import (
    BlenderRpcClient,
    BlenderRpcConfig,
    JsonValue,
)
from visculpt.vision.part_segmentation import (
    PartSegmentationConfig,
    PartSegmentationError,
    PartSegmentationRunner,
    PartSegmentationVlm,
)

from .select_mask_component import create_select_mask_component_tool


class PartSegmentationWithSam3Input(BaseModel):
    """Input contract for automatic or user-constrained part segmentation."""

    model_config = ConfigDict(extra="forbid")

    image_path: str = Field(
        min_length=1,
        description="Absolute local path of a Blender VIEW_3D screenshot.",
    )
    part_description: str = Field(
        min_length=1,
        max_length=512,
        description="Description of the model part to divide into Face Sets.",
    )
    output_dir: str | None = Field(
        default=None,
        description="Optional absolute artifact directory owned by the workflow.",
    )
    parent_mask_path: str | None = Field(
        default=None,
        description=(
            "Optional user-confirmed parent-part mask; when present, SAM 3 does "
            "not segment the parent part again."
        ),
    )

    @field_validator("image_path")
    @classmethod
    def normalize_image_path(cls, value: str) -> str:
        """Strip surrounding whitespace from the screenshot path."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("image_path must not be empty")
        return normalized

    @field_validator("part_description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        """Normalize repeated whitespace in the target description."""
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("part_description must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_image_path(self) -> PartSegmentationWithSam3Input:
        """Reject ambiguous or missing screenshot paths."""
        image_path = Path(
            os.path.expandvars(os.path.expanduser(self.image_path))
        )
        if not image_path.is_absolute():
            raise ValueError("image_path must be absolute")
        if not image_path.is_file():
            raise ValueError("image_path must reference an existing file")
        for field_name in ("output_dir", "parent_mask_path"):
            value = getattr(self, field_name)
            if value is None:
                continue
            path = Path(os.path.expandvars(os.path.expanduser(value)))
            if not path.is_absolute():
                raise ValueError(f"{field_name} must be absolute")
            if field_name == "parent_mask_path" and not path.is_file():
                raise ValueError(
                    "parent_mask_path must reference an existing file"
                )
        return self


def create_part_segmentation_with_sam3_tool(
    *,
    llm: PartSegmentationVlm,
    segment_tool: BaseTool,
    mask_component_selector_tool: BaseTool | None = None,
    client: BlenderRpcClient | None = None,
    rpc_config: BlenderRpcConfig | None = None,
    config: PartSegmentationConfig | None = None,
    workdir: Path | None = None,
) -> StructuredTool:
    """Create the composite SAM3-to-Face-Set LangGraph Tool."""
    if client is not None and rpc_config is not None:
        raise ValueError("client and rpc_config cannot both be provided")
    rpc_client = client or BlenderRpcClient(rpc_config)
    resolved_config = config or PartSegmentationConfig()
    component_selector = (
        mask_component_selector_tool
        if mask_component_selector_tool is not None
        else create_select_mask_component_tool(
            llm=llm,
            llm_role=resolved_config.llm_role,
            workdir=workdir,
        )
    )

    # Delay import to avoid a tools/workflow package initialization cycle.
    from visculpt.workflow.prompts import (
        PART_SEGMENTATION_SYSTEM_PROMPT,
        part_segmentation_user_prompt,
    )

    runner = PartSegmentationRunner(
        llm=llm,
        segment_tool=segment_tool,
        mask_component_selector_tool=component_selector,
        rpc_client=rpc_client,
        system_prompt=PART_SEGMENTATION_SYSTEM_PROMPT,
        prompt_builder=part_segmentation_user_prompt,
        config=resolved_config,
        workdir=workdir,
    )

    def invoke(
        image_path: str,
        part_description: str,
        output_dir: str | None = None,
        parent_mask_path: str | None = None,
    ) -> dict[str, JsonValue]:
        try:
            result = runner.run(
                image_path=image_path,
                part_description=part_description,
                artifact_root=output_dir,
                parent_mask_path=parent_mask_path,
            )
            return {
                "input": {
                    "image_path": image_path,
                    "part_description": part_description,
                    "output_dir": output_dir,
                    "parent_mask_path": parent_mask_path,
                },
                "result": result.as_payload(),
            }
        except PartSegmentationError as error:
            raise ToolException(_part_segmentation_error(error)) from error

    async def ainvoke(
        image_path: str,
        part_description: str,
        output_dir: str | None = None,
        parent_mask_path: str | None = None,
    ) -> dict[str, JsonValue]:
        try:
            result = await asyncio.to_thread(
                runner.run,
                image_path=image_path,
                part_description=part_description,
                artifact_root=output_dir,
                parent_mask_path=parent_mask_path,
            )
            return {
                "input": {
                    "image_path": image_path,
                    "part_description": part_description,
                    "output_dir": output_dir,
                    "parent_mask_path": parent_mask_path,
                },
                "result": result.as_payload(),
            }
        except PartSegmentationError as error:
            raise ToolException(_part_segmentation_error(error)) from error

    return StructuredTool.from_function(
        func=invoke,
        coroutine=ainvoke,
        name="part_segmentation_with_sam3",
        description=(
            "Use a VLM to plan the parent instance and kinematic subparts from a "
            "Blender screenshot and part description. Constrain SAM 3 segmentation "
            "with the parent mask, then create one Blender Lasso Face Set per "
            "subpart. This tool modifies the current sculpt mesh directly."
        ),
        args_schema=PartSegmentationWithSam3Input,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: _invalid_input_error(),
    )


def _invalid_input_error() -> str:
    return json.dumps(
        {
            "part_segmentation_error": {
                "type": "invalid_tool_input",
                "message": (
                    "Part segmentation requires an existing absolute Blender "
                    "screenshot path and a non-empty model-part description"
                ),
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _part_segmentation_error(error: PartSegmentationError) -> str:
    mapping = {
        "PartSegmentationInputError": "input_error",
        "PartSegmentationVlmError": "vlm_error",
        "PartSegmentationSam3Error": "sam3_error",
        "PartSegmentationParentMaskError": "parent_mask_unavailable",
        "PartSegmentationLassoError": "lasso_error",
        "PartSegmentationRpcError": "rpc_error",
        "PartSegmentationArtifactError": "artifact_error",
    }
    return json.dumps(
        {
            "part_segmentation_error": {
                "type": mapping.get(type(error).__name__, "tool_error"),
                "message": str(error),
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
