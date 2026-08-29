"""LangGraph tool for saving the current Blender state."""

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


class SaveBlenderStateInput(BaseModel):
    """Input schema shown to the Agent model."""

    model_config = ConfigDict(extra="forbid")

    filepath: str = Field(
        min_length=1,
        description="Absolute path for the target Blender .blend snapshot.",
    )
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite an existing target; defaults to false.",
    )
    compress: bool = Field(
        default=False,
        description="Whether to use native Blender compression; defaults to false.",
    )

    @field_validator("filepath")
    @classmethod
    def validate_filepath(cls, value: str) -> str:
        """Require a non-empty absolute .blend snapshot path."""
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
        return stripped


def create_save_blender_state_tool(
    *,
    client: BlenderRpcClient | None = None,
    config: BlenderRpcConfig | None = None,
) -> StructuredTool:
    """Create a LangGraph-compatible Blender state-save Tool."""
    if client is not None and config is not None:
        raise ValueError("client and config cannot both be provided")
    rpc_client = client or BlenderRpcClient(config)

    def invoke(
        filepath: str,
        overwrite: bool = False,
        compress: bool = False,
    ) -> JsonValue:
        request = _build_request(
            filepath=filepath,
            overwrite=overwrite,
            compress=compress,
        )
        try:
            return rpc_client.send(request)
        except BlenderRpcBridgeError as error:
            raise ToolException(error.as_json()) from error

    async def ainvoke(
        filepath: str,
        overwrite: bool = False,
        compress: bool = False,
    ) -> JsonValue:
        request = _build_request(
            filepath=filepath,
            overwrite=overwrite,
            compress=compress,
        )
        try:
            return await asyncio.to_thread(rpc_client.send, request)
        except BlenderRpcBridgeError as error:
            raise ToolException(error.as_json()) from error

    return StructuredTool.from_function(
        func=invoke,
        coroutine=ainvoke,
        name="save_blender_state",
        description=(
            "Save the complete current Blender project as a .blend snapshot at "
            "an absolute path. Copy mode preserves the active project path and "
            "existing files are not overwritten by default."
        ),
        args_schema=SaveBlenderStateInput,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: _invalid_input_error(),
    )


def _build_request(
    *,
    filepath: str,
    overwrite: bool,
    compress: bool,
) -> dict[str, JsonValue]:
    return {
        "jsonrpc": "2.0",
        "id": f"save-blender-state-{uuid4().hex}",
        "method": "save_blend_file",
        "params": {
            "filepath": filepath,
            "overwrite": overwrite,
            "compress": compress,
        },
    }


def _invalid_input_error() -> str:
    return json.dumps(
        {
            "bridge_error": {
                "type": "invalid_tool_input",
                "message": (
                    "Blender State Save Tool filepath must be a non-empty "
                    "absolute .blend path; overwrite and compress must be "
                    "booleans"
                ),
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
