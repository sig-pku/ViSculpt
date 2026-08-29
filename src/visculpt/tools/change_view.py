"""LangGraph tool for changing the Blender operation view."""

from __future__ import annotations

import asyncio
import json
from enum import StrEnum
from typing import Self
from uuid import uuid4

from langchain_core.tools import StructuredTool, ToolException
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from visculpt.bridge import (
    BlenderRpcBridgeError,
    BlenderRpcClient,
    BlenderRpcConfig,
    JsonValue,
)


class BlenderView(StrEnum):
    """Standard views supported by the Blender RPC Add-on."""

    FRONT = "FRONT"
    BACK = "BACK"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    TOP = "TOP"
    BOTTOM = "BOTTOM"
    CAMERA = "CAMERA"


class ViewFrame(StrEnum):
    """Framing behavior applied after changing the view."""

    KEEP = "KEEP"
    SELECTED = "SELECTED"
    ALL = "ALL"


class ViewProjection(StrEnum):
    """Projection modes supported by standard Blender views."""

    ORTHOGRAPHIC = "ORTHOGRAPHIC"
    PERSPECTIVE = "PERSPECTIVE"


class ChangeViewInput(BaseModel):
    """Input schema shown to the Agent model."""

    model_config = ConfigDict(extra="forbid")

    view: BlenderView = Field(
        description=(
            "Target standard view: FRONT, BACK, LEFT, RIGHT, TOP, BOTTOM, "
            "or CAMERA."
        )
    )
    frame: ViewFrame = Field(
        default=ViewFrame.KEEP,
        description=(
            "Framing after the change: KEEP preserves framing, SELECTED "
            "focuses selected objects, and ALL frames all objects."
        ),
    )
    projection: ViewProjection = Field(
        default=ViewProjection.ORTHOGRAPHIC,
        description=(
            "Projection for standard views; the workflow uses orthographic."
        ),
    )
    all_viewports: bool = Field(
        default=False,
        description=(
            "Whether to update every 3D Viewport in the target window, or in "
            "all windows when no target window is specified."
        ),
    )
    align_active: bool = Field(
        default=False,
        description="Whether to align the standard view to active-object axes.",
    )
    window_index: int | None = Field(
        default=None,
        ge=0,
        description="Optional Blender window index.",
    )
    area_index: int | None = Field(
        default=None,
        ge=0,
        description="Optional 3D Viewport area index.",
    )

    @field_validator("view", "frame", "projection", mode="before")
    @classmethod
    def normalize_enum_value(cls, value: object) -> object:
        """Accept case-insensitive string values from an Agent."""
        if isinstance(value, str):
            return value.upper()
        return value

    @model_validator(mode="after")
    def validate_viewport_target(self) -> Self:
        """Reject target combinations unsupported by the Add-on."""
        if self.all_viewports and self.area_index is not None:
            raise ValueError(
                "area_index cannot be combined with all_viewports"
            )
        return self


def create_change_view_tool(
    *,
    client: BlenderRpcClient | None = None,
    config: BlenderRpcConfig | None = None,
) -> StructuredTool:
    """Create a LangGraph-compatible Blender view Tool."""
    if client is not None and config is not None:
        raise ValueError("client and config cannot both be provided")
    rpc_client = client or BlenderRpcClient(config)

    def invoke(
        view: BlenderView,
        frame: ViewFrame = ViewFrame.KEEP,
        projection: ViewProjection = ViewProjection.ORTHOGRAPHIC,
        all_viewports: bool = False,
        align_active: bool = False,
        window_index: int | None = None,
        area_index: int | None = None,
    ) -> JsonValue:
        request = _build_request(
            view=view,
            frame=frame,
            projection=projection,
            all_viewports=all_viewports,
            align_active=align_active,
            window_index=window_index,
            area_index=area_index,
        )
        try:
            return rpc_client.send(request)
        except BlenderRpcBridgeError as error:
            raise ToolException(error.as_json()) from error

    async def ainvoke(
        view: BlenderView,
        frame: ViewFrame = ViewFrame.KEEP,
        projection: ViewProjection = ViewProjection.ORTHOGRAPHIC,
        all_viewports: bool = False,
        align_active: bool = False,
        window_index: int | None = None,
        area_index: int | None = None,
    ) -> JsonValue:
        request = _build_request(
            view=view,
            frame=frame,
            projection=projection,
            all_viewports=all_viewports,
            align_active=align_active,
            window_index=window_index,
            area_index=area_index,
        )
        try:
            return await asyncio.to_thread(rpc_client.send, request)
        except BlenderRpcBridgeError as error:
            raise ToolException(error.as_json()) from error

    return StructuredTool.from_function(
        func=invoke,
        coroutine=ainvoke,
        name="change_blender_view",
        description=(
            "Change the Blender operation view to a standard axis or camera "
            "view while preserving, selecting, or reframing visible objects."
        ),
        args_schema=ChangeViewInput,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: _invalid_input_error(),
    )


def _build_request(
    *,
    view: BlenderView,
    frame: ViewFrame,
    projection: ViewProjection,
    all_viewports: bool,
    align_active: bool,
    window_index: int | None,
    area_index: int | None,
) -> dict[str, JsonValue]:
    params: dict[str, JsonValue] = {
        "view": view.value,
        "frame": frame.value,
        "projection": projection.value,
        "all_viewports": all_viewports,
        "align_active": align_active,
    }
    if window_index is not None:
        params["window_index"] = window_index
    if area_index is not None:
        params["area_index"] = area_index
    return {
        "jsonrpc": "2.0",
        "id": f"change-view-{uuid4().hex}",
        "method": "set_view",
        "params": params,
    }


def _invalid_input_error() -> str:
    return json.dumps(
        {
            "bridge_error": {
                "type": "invalid_tool_input",
                "message": (
                    "View Tool input must use a supported view, frame, "
                    "and viewport target"
                ),
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
