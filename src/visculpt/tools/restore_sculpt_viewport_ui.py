"""LangGraph tool for restoring Blender Sculpt viewport UI state."""

from __future__ import annotations

import asyncio
import json
from typing import Literal
from uuid import uuid4

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from visculpt.bridge import (
    BlenderRpcBridgeError,
    BlenderRpcClient,
    BlenderRpcConfig,
    JsonValue,
)


class SculptViewportSpaceUiInput(BaseModel):
    """Restorable SpaceView3D values supported across Blender versions."""

    model_config = ConfigDict(extra="forbid")

    show_region_asset_shelf: StrictBool | None = None
    show_region_toolbar: StrictBool | None = None
    show_region_ui: StrictBool | None = None
    show_region_header: StrictBool | None = None
    show_region_tool_header: StrictBool | None = None
    show_gizmo_navigate: StrictBool | None = None


class SculptViewportOverlayUiInput(BaseModel):
    """Restorable View3DOverlay values managed by the workflow."""

    model_config = ConfigDict(extra="forbid")

    show_floor: StrictBool | None = None
    show_ortho_grid: StrictBool | None = None
    show_text: StrictBool | None = None
    show_stats: StrictBool | None = None


class SculptViewportUiSnapshotInput(BaseModel):
    """Exact snapshot returned by enter_blender_sculpt_mode."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    window_index: int = Field(ge=0, strict=True)
    area_index: int = Field(ge=0, strict=True)
    space: SculptViewportSpaceUiInput
    overlay: SculptViewportOverlayUiInput

    @model_validator(mode="after")
    def require_restorable_property(self) -> SculptViewportUiSnapshotInput:
        """Reject empty snapshots before sending an RPC request."""
        space = self.space.model_dump(exclude_none=True)
        overlay = self.overlay.model_dump(exclude_none=True)
        if not space and not overlay:
            raise ValueError("snapshot contains no restorable UI properties")
        return self


class RestoreSculptViewportUiInput(BaseModel):
    """Input schema shown to the Agent model."""

    model_config = ConfigDict(extra="forbid")

    snapshot: SculptViewportUiSnapshotInput = Field(
        description=(
            "The viewport_ui.snapshot returned by enter_blender_sculpt_mode."
        )
    )


def create_restore_sculpt_viewport_ui_tool(
    *,
    client: BlenderRpcClient | None = None,
    config: BlenderRpcConfig | None = None,
) -> StructuredTool:
    """Create a LangGraph-compatible viewport UI restore Tool."""
    if client is not None and config is not None:
        raise ValueError("client and config cannot both be provided")
    rpc_client = client or BlenderRpcClient(config)

    def invoke(snapshot: SculptViewportUiSnapshotInput) -> JsonValue:
        request = _build_request(snapshot)
        try:
            return rpc_client.send(request)
        except BlenderRpcBridgeError as error:
            raise ToolException(error.as_json()) from error

    async def ainvoke(snapshot: SculptViewportUiSnapshotInput) -> JsonValue:
        request = _build_request(snapshot)
        try:
            return await asyncio.to_thread(rpc_client.send, request)
        except BlenderRpcBridgeError as error:
            raise ToolException(error.as_json()) from error

    return StructuredTool.from_function(
        func=invoke,
        coroutine=ainvoke,
        name="restore_blender_sculpt_viewport_ui",
        description=(
            "Restore the Blender 3D Viewport asset shelf, toolbar, sidebar, main "
            "header, tool settings, grid, general information, statistics, and "
            "navigation controls from the enter_blender_sculpt_mode snapshot, "
            "then verify the result on a later Blender timer step."
        ),
        args_schema=RestoreSculptViewportUiInput,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: _invalid_input_error(),
    )


def _build_request(
    snapshot: SculptViewportUiSnapshotInput,
) -> dict[str, JsonValue]:
    return {
        "jsonrpc": "2.0",
        "id": f"restore-sculpt-viewport-ui-{uuid4().hex}",
        "method": "restore_sculpt_viewport_ui",
        "params": {
            "snapshot": snapshot.model_dump(
                mode="json",
                exclude_none=True,
            )
        },
    }


def _invalid_input_error() -> str:
    return json.dumps(
        {
            "bridge_error": {
                "type": "invalid_tool_input",
                "message": (
                    "snapshot must be the version 1.0 viewport UI state "
                    "returned by enter_blender_sculpt_mode"
                ),
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
