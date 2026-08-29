"""LangGraph tool for text-prompted SAM3 image segmentation."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from visculpt.bridge import JsonValue
from visculpt.vision import (
    MaskCleanupError,
    Sam3ClientError,
    Sam3GradioClient,
    Sam3GradioConfig,
    clean_instance_mask_archive,
    clean_segmentation_mask,
    render_cleaned_mask_overlay,
)


class SegmentWithSam3Input(BaseModel):
    """Segmentation-only input schema shown to the Agent model."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    image_path: str = Field(
        min_length=1,
        description="Absolute path of the local image to segment.",
    )
    prompt: str = Field(
        min_length=1,
        max_length=512,
        description="Short English text prompt describing the target object.",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.05,
        le=0.95,
        description="SAM 3 instance confidence threshold.",
    )
    overlay_opacity: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Mask opacity in the segmentation overlay.",
    )
    output_dir: str | None = Field(
        default=None,
        description="Optional absolute directory for saved results.",
    )

    @field_validator("image_path", "prompt", "output_dir")
    @classmethod
    def strip_string_value(cls, value: str | None) -> str | None:
        """Remove surrounding whitespace from Agent-provided strings."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped

    @model_validator(mode="after")
    def validate_local_paths(self) -> SegmentWithSam3Input:
        """Reject ambiguous or unusable local paths before inference."""
        image_path = _expanded_path(self.image_path)
        if not image_path.is_absolute():
            raise ValueError("image_path must be absolute")
        if not image_path.is_file():
            raise ValueError("image_path must reference an existing file")

        if self.output_dir is not None:
            output_dir = _expanded_path(self.output_dir)
            if not output_dir.is_absolute():
                raise ValueError("output_dir must be absolute")
            if output_dir.exists() and not output_dir.is_dir():
                raise ValueError("output_dir must reference a directory")
        return self


def create_segment_with_sam3_tool(
    *,
    client: Sam3GradioClient | None = None,
    config: Sam3GradioConfig | None = None,
) -> StructuredTool:
    """Create the segmentation-only LangGraph Tool."""
    if client is not None and config is not None:
        raise ValueError("client and config cannot both be provided")
    sam3_client = client or Sam3GradioClient(config)

    def invoke(
        image_path: str,
        prompt: str,
        confidence_threshold: float = 0.5,
        overlay_opacity: float = 0.45,
        output_dir: str | None = None,
    ) -> dict[str, JsonValue]:
        try:
            return _segment_and_clean(
                sam3_client,
                image_path=image_path,
                prompt=prompt,
                confidence_threshold=confidence_threshold,
                overlay_opacity=overlay_opacity,
                output_dir=output_dir,
            )
        except Sam3ClientError as error:
            raise ToolException(error.as_json()) from error
        except MaskCleanupError as error:
            raise ToolException(_cleanup_error(error)) from error

    async def ainvoke(
        image_path: str,
        prompt: str,
        confidence_threshold: float = 0.5,
        overlay_opacity: float = 0.45,
        output_dir: str | None = None,
    ) -> dict[str, JsonValue]:
        try:
            return await asyncio.to_thread(
                _segment_and_clean,
                sam3_client,
                image_path=image_path,
                prompt=prompt,
                confidence_threshold=confidence_threshold,
                overlay_opacity=overlay_opacity,
                output_dir=output_dir,
            )
        except Sam3ClientError as error:
            raise ToolException(error.as_json()) from error
        except MaskCleanupError as error:
            raise ToolException(_cleanup_error(error)) from error

    return StructuredTool.from_function(
        func=invoke,
        coroutine=ainvoke,
        name="segment_with_sam3",
        description=(
            "Call the SAM 3 Gradio HTTP service with a local image and English "
            "prompt, returning a cleaned mask, overlay, and metadata independent "
            "of sculpt parameters."
        ),
        args_schema=SegmentWithSam3Input,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: _invalid_input_error(),
    )


def _segment_and_clean(
    client: Sam3GradioClient,
    *,
    image_path: str,
    prompt: str,
    confidence_threshold: float,
    overlay_opacity: float,
    output_dir: str | None,
) -> dict[str, JsonValue]:
    result = client.segment(
        image_path=image_path,
        prompt=prompt,
        confidence_threshold=confidence_threshold,
        overlay_opacity=overlay_opacity,
        output_dir=output_dir,
    )
    cleaned = clean_segmentation_mask(
        result.mask_path,
        output_dir=output_dir,
    )
    cleaned_instances = clean_instance_mask_archive(
        result.instance_masks_path,
        output_dir=output_dir,
    )
    cleaned_overlay_path = render_cleaned_mask_overlay(
        image_path,
        cleaned.cleaned_mask_path,
        opacity=overlay_opacity,
        output_dir=output_dir,
    )
    payload = result.as_payload()
    result_payload = payload["result"]
    assert isinstance(result_payload, dict)
    result_payload.update(cleaned.as_payload())
    result_payload["cleaned_overlay_path"] = cleaned_overlay_path
    metadata_instances = result.metadata.get("instances")
    if (
        not isinstance(metadata_instances, list)
        or len(metadata_instances) != len(cleaned_instances)
    ):
        raise MaskCleanupError(
            "SAM3 metadata does not match independently cleaned instances"
        )
    instance_payloads: list[JsonValue] = []
    for cleaned_instance in cleaned_instances:
        metadata = metadata_instances[cleaned_instance.instance_index]
        if not isinstance(metadata, dict):
            raise MaskCleanupError("SAM3 instance metadata is not an object")
        instance_payload = dict(metadata)
        instance_payload.update(cleaned_instance.as_payload())
        instance_payloads.append(instance_payload)
    result_payload["instances"] = instance_payloads
    return payload


def _expanded_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value)))


def _invalid_input_error() -> str:
    return json.dumps(
        {
            "sam3_error": {
                "type": "invalid_tool_input",
                "message": (
                    "SAM3 input must use an existing absolute image path, "
                    "a non-empty prompt, valid thresholds, and an optional "
                    "absolute output directory"
                ),
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _cleanup_error(error: MaskCleanupError) -> str:
    return json.dumps(
        {
            "sam3_error": {
                "type": "mask_cleanup_error",
                "message": str(error),
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
