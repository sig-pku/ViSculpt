"""LangGraph tool for reading local Blender Sculpt brush capabilities."""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict

from visculpt.bridge import (
    BlenderRpcBridgeError,
    BlenderRpcClient,
    BlenderRpcConfig,
    JsonValue,
)


class GetSculptBrushCapabilitiesInput(BaseModel):
    """Empty input contract for the read-only capability query."""

    model_config = ConfigDict(extra="forbid")


def create_get_sculpt_brush_capabilities_tool(
    *,
    client: BlenderRpcClient | None = None,
    config: BlenderRpcConfig | None = None,
) -> StructuredTool:
    """Create a Tool that reads Blender version and Sculpt brush metadata."""
    if client is not None and config is not None:
        raise ValueError("client and config cannot both be provided")
    rpc_client = client or BlenderRpcClient(config)

    def invoke() -> JsonValue:
        try:
            return rpc_client.send(_build_request())
        except BlenderRpcBridgeError as error:
            raise ToolException(error.as_json()) from error

    async def ainvoke() -> JsonValue:
        try:
            return await asyncio.to_thread(rpc_client.send, _build_request())
        except BlenderRpcBridgeError as error:
            raise ToolException(error.as_json()) from error

    return StructuredTool.from_function(
        func=invoke,
        coroutine=ainvoke,
        name="get_blender_sculpt_brush_capabilities",
        description=(
            "Read the local Blender version, exact names of all available sculpt "
            "brushes, and valid direction values for each brush through RPC "
            "get_state. This read-only tool does not modify Blender."
        ),
        args_schema=GetSculptBrushCapabilitiesInput,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: _invalid_input_error(),
    )


def _build_request() -> dict[str, JsonValue]:
    return {
        "jsonrpc": "2.0",
        "id": f"get-sculpt-brush-capabilities-{uuid4().hex}",
        "method": "get_state",
        "params": {
            "include_objects": False,
            "include_viewports": False,
        },
    }


def _invalid_input_error() -> str:
    return json.dumps(
        {
            "bridge_error": {
                "type": "invalid_tool_input",
                "message": (
                    "Sculpt brush capability Tool does not accept input "
                    "parameters"
                ),
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
