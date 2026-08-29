"""LangGraph tool for restoring Blender state from a .blend file."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from visculpt.bridge import (
    BlenderRpcBridgeError,
    BlenderRpcClient,
    BlenderRpcConfig,
    JsonValue,
)


class LoadBlenderStateInput(BaseModel):
    """Input schema shown to the Agent model."""

    model_config = ConfigDict(extra="forbid")

    filepath: str = Field(
        min_length=1,
        description="Absolute path of an existing .blend file to restore.",
    )
    load_ui: bool = Field(
        default=True,
        description="Whether to restore saved UI, workspace, and viewport layout.",
    )

    @field_validator("filepath")
    @classmethod
    def validate_filepath(cls, value: str) -> str:
        """Require an existing absolute .blend file path."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("filepath must not be blank")
        expanded = Path(
            os.path.expandvars(os.path.expanduser(stripped))
        )
        if not expanded.is_absolute():
            raise ValueError("filepath must be absolute")
        if expanded.suffix.lower() != ".blend":
            raise ValueError("filepath must end in .blend")
        if not expanded.is_file():
            raise ValueError("filepath must reference an existing file")
        return stripped


def create_load_blender_state_tool(
    *,
    client: BlenderRpcClient | None = None,
    config: BlenderRpcConfig | None = None,
) -> StructuredTool:
    """Create a LangGraph-compatible Blender state-load Tool."""
    if client is not None and config is not None:
        raise ValueError("client and config cannot both be provided")
    rpc_client = client or BlenderRpcClient(config)

    def invoke(
        filepath: str,
        load_ui: bool = True,
    ) -> JsonValue:
        request = _build_request(filepath=filepath, load_ui=load_ui)
        try:
            return rpc_client.send(request)
        except BlenderRpcBridgeError as error:
            raise ToolException(error.as_json()) from error

    async def ainvoke(
        filepath: str,
        load_ui: bool = True,
    ) -> JsonValue:
        request = _build_request(filepath=filepath, load_ui=load_ui)
        try:
            return await asyncio.to_thread(rpc_client.send, request)
        except BlenderRpcBridgeError as error:
            raise ToolException(error.as_json()) from error

    return StructuredTool.from_function(
        func=invoke,
        coroutine=ainvoke,
        name="load_blender_state",
        description=(
            "Restore the complete Blender project state from an existing .blend "
            "file. This replaces the current project and discards unsaved "
            "changes; embedded scripts are never executed automatically."
        ),
        args_schema=LoadBlenderStateInput,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: _invalid_input_error(),
    )


def _build_request(
    *,
    filepath: str,
    load_ui: bool,
) -> dict[str, JsonValue]:
    return {
        "jsonrpc": "2.0",
        "id": f"load-blender-state-{uuid4().hex}",
        "method": "load_blend_file",
        "params": {
            "filepath": filepath,
            "load_ui": load_ui,
        },
    }


def _invalid_input_error() -> str:
    return json.dumps(
        {
            "bridge_error": {
                "type": "invalid_tool_input",
                "message": (
                    "Blender State Load Tool filepath must reference an "
                    "existing absolute .blend file and load_ui must be a "
                    "boolean"
                ),
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
