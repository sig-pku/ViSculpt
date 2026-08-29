"""LangGraph tool for capturing a Blender window screenshot."""

from __future__ import annotations

import asyncio
import json
import os
from enum import StrEnum
from pathlib import Path
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


class ScreenshotOutput(StrEnum):
    """Screenshot output modes supported by the Blender RPC Add-on."""

    BASE64 = "base64"
    FILE = "file"


class GetScreenshotInput(BaseModel):
    """Input schema shown to the Agent model."""

    model_config = ConfigDict(extra="forbid")

    output: ScreenshotOutput = Field(
        default=ScreenshotOutput.BASE64,
        description=(
            "Screenshot output mode: base64 embeds PNG data in the RPC response; "
            "file asks Blender to save a PNG."
        ),
    )
    filepath: str | None = Field(
        default=None,
        description=(
            "Absolute PNG path required for output=file and forbidden for "
            "output=base64."
        ),
    )
    window_index: int | None = Field(
        default=None,
        ge=0,
        description="Optional Blender window index.",
    )
    redraw: bool = Field(
        default=True,
        description="Whether Blender should redraw the target window first.",
    )

    @field_validator("output", mode="before")
    @classmethod
    def normalize_output(cls, value: object) -> object:
        """Accept case-insensitive string values from an Agent."""
        if isinstance(value, str):
            return value.lower()
        return value

    @field_validator("filepath")
    @classmethod
    def normalize_filepath(cls, value: str | None) -> str | None:
        """Match the Add-on's handling of surrounding whitespace."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("filepath must be a non-empty string")
        return stripped

    @model_validator(mode="after")
    def validate_output_target(self) -> Self:
        """Reject output combinations unsupported by the Add-on."""
        if self.output is ScreenshotOutput.BASE64:
            if self.filepath is not None:
                raise ValueError(
                    "filepath cannot be used with base64 output"
                )
            return self

        if self.filepath is None:
            raise ValueError("filepath is required for file output")
        expanded_path = Path(
            os.path.expandvars(os.path.expanduser(self.filepath))
        )
        if not expanded_path.is_absolute():
            raise ValueError("filepath must be absolute")
        if expanded_path.suffix.lower() != ".png":
            raise ValueError("filepath must end in .png")
        return self


def create_get_screenshot_tool(
    *,
    client: BlenderRpcClient | None = None,
    config: BlenderRpcConfig | None = None,
) -> StructuredTool:
    """Create a LangGraph-compatible Blender screenshot Tool."""
    if client is not None and config is not None:
        raise ValueError("client and config cannot both be provided")
    rpc_client = client or BlenderRpcClient(config)

    def invoke(
        output: ScreenshotOutput = ScreenshotOutput.BASE64,
        filepath: str | None = None,
        window_index: int | None = None,
        redraw: bool = True,
    ) -> JsonValue:
        request = _build_request(
            output=output,
            filepath=filepath,
            window_index=window_index,
            redraw=redraw,
        )
        try:
            return rpc_client.send(request)
        except BlenderRpcBridgeError as error:
            raise ToolException(error.as_json()) from error

    async def ainvoke(
        output: ScreenshotOutput = ScreenshotOutput.BASE64,
        filepath: str | None = None,
        window_index: int | None = None,
        redraw: bool = True,
    ) -> JsonValue:
        request = _build_request(
            output=output,
            filepath=filepath,
            window_index=window_index,
            redraw=redraw,
        )
        try:
            return await asyncio.to_thread(rpc_client.send, request)
        except BlenderRpcBridgeError as error:
            raise ToolException(error.as_json()) from error

    return StructuredTool.from_function(
        func=invoke,
        coroutine=ainvoke,
        name="get_blender_screenshot",
        description=(
            "Capture a PNG screenshot of a Blender viewport. Return Base64 in "
            "result.data of the complete JSON-RPC response by default, or ask "
            "Blender to save it to an absolute path."
        ),
        args_schema=GetScreenshotInput,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: _invalid_input_error(),
    )


def _build_request(
    *,
    output: ScreenshotOutput,
    filepath: str | None,
    window_index: int | None,
    redraw: bool,
) -> dict[str, JsonValue]:
    params: dict[str, JsonValue] = {
        "output": output.value,
        "redraw": redraw,
    }
    if filepath is not None:
        params["filepath"] = filepath
    if window_index is not None:
        params["window_index"] = window_index
    return {
        "jsonrpc": "2.0",
        "id": f"get-screenshot-{uuid4().hex}",
        "method": "get_screenshot",
        "params": params,
    }


def _invalid_input_error() -> str:
    return json.dumps(
        {
            "bridge_error": {
                "type": "invalid_tool_input",
                "message": (
                    "Screenshot Tool input must use a supported output mode, "
                    "filepath, and window target"
                ),
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
