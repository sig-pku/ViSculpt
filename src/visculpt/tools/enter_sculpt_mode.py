"""LangGraph tool for entering Blender Sculpt Mode."""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from visculpt.bridge import (
    BlenderRpcBridgeError,
    BlenderRpcClient,
    BlenderRpcConfig,
    JsonValue,
)


class EnterSculptModeInput(BaseModel):
    """Input schema shown to the Agent model."""

    model_config = ConfigDict(extra="forbid")

    window_index: int | None = Field(
        default=None,
        ge=0,
        description="Optional Blender window index.",
    )
    area_index: int | None = Field(
        default=None,
        ge=0,
        description="Optional 3D Viewport area index in the target window.",
    )
    hide_viewport_ui: bool = Field(
        default=False,
        strict=True,
        description=(
            "Whether to hide viewport controls after the Sculpting screen and "
            "regions stabilize, returning a delayed-verified restore snapshot."
        ),
    )


def create_enter_sculpt_mode_tool(
    *,
    client: BlenderRpcClient | None = None,
    config: BlenderRpcConfig | None = None,
) -> StructuredTool:
    """Create a LangGraph-compatible Blender Sculpt Mode Tool."""
    if client is not None and config is not None:
        raise ValueError("client and config cannot both be provided")
    rpc_client = client or BlenderRpcClient(config)

    def invoke(
        window_index: int | None = None,
        area_index: int | None = None,
        hide_viewport_ui: bool = False,
    ) -> JsonValue:
        request = _build_request(
            window_index=window_index,
            area_index=area_index,
            hide_viewport_ui=hide_viewport_ui,
        )
        try:
            return rpc_client.send(request)
        except BlenderRpcBridgeError as error:
            raise ToolException(error.as_json()) from error

    async def ainvoke(
        window_index: int | None = None,
        area_index: int | None = None,
        hide_viewport_ui: bool = False,
    ) -> JsonValue:
        request = _build_request(
            window_index=window_index,
            area_index=area_index,
            hide_viewport_ui=hide_viewport_ui,
        )
        try:
            return await asyncio.to_thread(rpc_client.send, request)
        except BlenderRpcBridgeError as error:
            raise ToolException(error.as_json()) from error

    return StructuredTool.from_function(
        func=invoke,
        coroutine=ainvoke,
        name="enter_blender_sculpt_mode",
        description=(
            "Switch the target Blender window to the Sculpting workspace and "
            "enter Sculpt Mode on the active mesh. Optionally hide viewport UI "
            "that obstructs screenshots; return only after screen, region, and "
            "delayed UI readback are stable."
        ),
        args_schema=EnterSculptModeInput,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: _invalid_input_error(),
    )


def _build_request(
    *,
    window_index: int | None,
    area_index: int | None,
    hide_viewport_ui: bool,
) -> dict[str, JsonValue]:
    params: dict[str, JsonValue] = {}
    if window_index is not None:
        params["window_index"] = window_index
    if area_index is not None:
        params["area_index"] = area_index
    if hide_viewport_ui:
        params["hide_viewport_ui"] = True
    return {
        "jsonrpc": "2.0",
        "id": f"enter-sculpt-mode-{uuid4().hex}",
        "method": "enter_sculpt_mode",
        "params": params,
    }


def _invalid_input_error() -> str:
    return json.dumps(
        {
            "bridge_error": {
                "type": "invalid_tool_input",
                "message": (
                    "Sculpt Mode Tool indices must be non-negative integers "
                    "and hide_viewport_ui must be a boolean"
                ),
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
