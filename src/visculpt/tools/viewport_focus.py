"""LangGraph tools for deterministic Blender viewport ROI focus."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Literal, Self
from uuid import uuid4

from langchain_core.tools import StructuredTool, ToolException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from visculpt.bridge import (
    BlenderRpcBridgeError,
    BlenderRpcClient,
    BlenderRpcConfig,
    JsonValue,
)


class ViewportRoiInput(BaseModel):
    """Top-left screenshot-space ROI bounds."""

    model_config = ConfigDict(extra="forbid")

    x_min: float = Field(ge=0.0)
    y_min: float = Field(ge=0.0)
    x_max: float = Field(gt=0.0)
    y_max: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_rectangle(self) -> Self:
        """Require a finite non-empty rectangle."""
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("ROI coordinates must be finite")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("ROI must have positive width and height")
        return self


class ViewportDimensionsInput(BaseModel):
    """Positive width and height pair."""

    model_config = ConfigDict(extra="forbid")

    width: int = Field(ge=2, strict=True)
    height: int = Field(ge=2, strict=True)


class ViewportImageInput(ViewportDimensionsInput):
    """Screenshot dimensions and origin stored in a viewport snapshot."""

    origin: Literal["TOP_LEFT"]


class ViewportCoordinateScaleInput(BaseModel):
    """Framebuffer pixels per Blender region coordinate."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(gt=0.0)
    y: float = Field(gt=0.0)


class ViewportStateInput(BaseModel):
    """Versioned state returned by the Blender ROI focus RPC."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["viewport-state/v1"]
    window_index: int = Field(ge=0, strict=True)
    area_index: int = Field(ge=0, strict=True)
    region: ViewportDimensionsInput
    image: ViewportImageInput
    coordinate_scale: ViewportCoordinateScaleInput
    view_perspective: Literal["ORTHO", "PERSP", "CAMERA"]
    view_location: list[float]
    view_rotation: list[float]
    view_distance: float = Field(ge=0.0)
    view_camera_offset: list[float]
    view_camera_zoom: float
    lens: float = Field(gt=0.0)
    perspective_matrix: list[list[float]]

    @field_validator(
        "view_location",
        "view_rotation",
        "view_camera_offset",
        mode="after",
    )
    @classmethod
    def validate_vectors(
        cls,
        value: list[float],
        info: ValidationInfo,
    ) -> list[float]:
        """Validate vector lengths and finite values."""
        name = info.field_name
        expected = {
            "view_location": 3,
            "view_rotation": 4,
            "view_camera_offset": 2,
        }[name]
        if len(value) != expected or not all(
            math.isfinite(item) for item in value
        ):
            raise ValueError(f"{name} must contain {expected} finite numbers")
        return value

    @field_validator("perspective_matrix", mode="after")
    @classmethod
    def validate_matrix(cls, value: list[list[float]]) -> list[list[float]]:
        """Require one finite 4x4 projection matrix."""
        if len(value) != 4 or any(len(row) != 4 for row in value):
            raise ValueError("perspective_matrix must be 4x4")
        if not all(math.isfinite(item) for row in value for item in row):
            raise ValueError("perspective_matrix must be finite")
        return value


class FocusViewportRoiInput(BaseModel):
    """Input schema for deterministic ROI focus."""

    model_config = ConfigDict(extra="forbid")

    roi: ViewportRoiInput
    image_width: int = Field(ge=2, le=100_000, strict=True)
    image_height: int = Field(ge=2, le=100_000, strict=True)
    margin_ratio: float = Field(default=0.2, ge=0.0, le=2.0)
    maximum_zoom_factor: float = Field(default=12.0, ge=1.0, le=100.0)
    window_index: int | None = Field(default=None, ge=0, strict=True)
    area_index: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def validate_roi_inside_image(self) -> Self:
        """Keep the requested ROI inside its source screenshot."""
        if (
            self.roi.x_max > self.image_width
            or self.roi.y_max > self.image_height
        ):
            raise ValueError("ROI must be inside the screenshot")
        return self


class RestoreViewportStateInput(BaseModel):
    """Input schema for applying a captured viewport state."""

    model_config = ConfigDict(extra="forbid")

    snapshot: ViewportStateInput
    require_region_match: bool = True


def create_focus_viewport_roi_tool(
    *,
    client: BlenderRpcClient | None = None,
    config: BlenderRpcConfig | None = None,
) -> StructuredTool:
    """Create the Blender orthographic ROI focus Tool."""
    rpc_client = _rpc_client(client=client, config=config)

    def invoke(**kwargs: object) -> JsonValue:
        payload = FocusViewportRoiInput.model_validate(kwargs)
        return _send(rpc_client, _focus_request(payload))

    async def ainvoke(**kwargs: object) -> JsonValue:
        payload = FocusViewportRoiInput.model_validate(kwargs)
        return await asyncio.to_thread(
            _send,
            rpc_client,
            _focus_request(payload),
        )

    return StructuredTool.from_function(
        func=invoke,
        coroutine=ainvoke,
        name="focus_blender_viewport_roi",
        description=(
            "Pan and zoom a screenshot-coordinate ROI to the center of an "
            "orthographic viewport, returning before/after snapshots and the "
            "exact 2D affine transform."
        ),
        args_schema=FocusViewportRoiInput,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: _invalid_input_error("focus"),
    )


def create_restore_viewport_state_tool(
    *,
    client: BlenderRpcClient | None = None,
    config: BlenderRpcConfig | None = None,
) -> StructuredTool:
    """Create the Tool that applies an exact viewport snapshot."""
    rpc_client = _rpc_client(client=client, config=config)

    def invoke(
        snapshot: ViewportStateInput,
        require_region_match: bool = True,
    ) -> JsonValue:
        return _send(
            rpc_client,
            _restore_request(snapshot, require_region_match),
        )

    async def ainvoke(
        snapshot: ViewportStateInput,
        require_region_match: bool = True,
    ) -> JsonValue:
        return await asyncio.to_thread(
            _send,
            rpc_client,
            _restore_request(snapshot, require_region_match),
        )

    return StructuredTool.from_function(
        func=invoke,
        coroutine=ainvoke,
        name="restore_blender_viewport_state",
        description=(
            "Apply the exact snapshot returned by focus_blender_viewport_roi and "
            "verify projection, pan, zoom, and viewport dimensions."
        ),
        args_schema=RestoreViewportStateInput,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: _invalid_input_error("restore"),
    )


def _rpc_client(
    *,
    client: BlenderRpcClient | None,
    config: BlenderRpcConfig | None,
) -> BlenderRpcClient:
    if client is not None and config is not None:
        raise ValueError("client and config cannot both be provided")
    return client or BlenderRpcClient(config)


def _send(
    client: BlenderRpcClient,
    request: dict[str, JsonValue],
) -> JsonValue:
    try:
        return client.send(request)
    except BlenderRpcBridgeError as error:
        raise ToolException(error.as_json()) from error


def _focus_request(payload: FocusViewportRoiInput) -> dict[str, JsonValue]:
    return {
        "jsonrpc": "2.0",
        "id": f"focus-viewport-roi-{uuid4().hex}",
        "method": "focus_viewport_roi",
        "params": payload.model_dump(mode="json", exclude_none=True),
    }


def _restore_request(
    snapshot: ViewportStateInput,
    require_region_match: bool,
) -> dict[str, JsonValue]:
    return {
        "jsonrpc": "2.0",
        "id": f"restore-viewport-state-{uuid4().hex}",
        "method": "restore_viewport_state",
        "params": {
            "snapshot": snapshot.model_dump(mode="json"),
            "require_region_match": require_region_match,
        },
    }


def _invalid_input_error(operation: str) -> str:
    return json.dumps(
        {
            "bridge_error": {
                "type": "invalid_tool_input",
                "message": f"Viewport {operation} input is invalid",
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
