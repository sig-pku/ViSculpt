"""LangGraph-compatible Agent tool for Blender RPC."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field

from .client import BlenderRpcClient
from .config import BlenderRpcConfig
from .errors import BlenderRpcBridgeError
from .types import JsonValue

_TOOL_DESCRIPTION = (
    "Forward a complete JSON-RPC 2.0 envelope to the local Blender Geometry "
    "Editing RPC Server and return its complete JSON response. Supported calls "
    "include ping, get_state, set_view, get_screenshot, and rpc.discover. "
    "State-changing methods may modify the open Blender project."
)


class BlenderRpcToolInput(BaseModel):
    """Input schema shown to the Agent model."""

    model_config = ConfigDict(extra="forbid")

    request: dict[str, Any] | list[Any] = Field(
        description=(
            "A complete JSON-RPC 2.0 request object or batch request array."
        )
    )


def create_blender_rpc_tool(
    *,
    client: BlenderRpcClient | None = None,
    config: BlenderRpcConfig | None = None,
) -> StructuredTool:
    """Create the Agent tool backed by a Blender RPC client."""
    if client is not None and config is not None:
        raise ValueError("client and config cannot both be provided")
    rpc_client = client or BlenderRpcClient(config)

    def invoke(request: object) -> JsonValue:
        try:
            return rpc_client.send(cast(JsonValue, request))
        except BlenderRpcBridgeError as error:
            raise ToolException(error.as_json()) from error

    async def ainvoke(request: object) -> JsonValue:
        try:
            return await asyncio.to_thread(
                rpc_client.send,
                cast(JsonValue, request),
            )
        except BlenderRpcBridgeError as error:
            raise ToolException(error.as_json()) from error

    return StructuredTool.from_function(
        func=invoke,
        coroutine=ainvoke,
        name="blender_rpc",
        description=_TOOL_DESCRIPTION,
        args_schema=BlenderRpcToolInput,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: json.dumps(
            {
                "bridge_error": {
                    "type": "invalid_tool_input",
                    "message": (
                        "Tool input must contain a JSON-RPC request object "
                        "or batch array in 'request'"
                    ),
                }
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )
