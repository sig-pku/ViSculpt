"""LangGraph tool for executing an operator-ready Blender Sculpt stroke."""

from __future__ import annotations

import asyncio
import json
from enum import StrEnum
from typing import Literal, Self
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


class SculptStrokeMode(StrEnum):
    """Execution modes accepted by Blender's Sculpt stroke operator."""

    NORMAL = "NORMAL"
    INVERT = "INVERT"


class SculptStrokeBrushToggle(StrEnum):
    """Temporary brush toggles exposed by the controlled RPC method."""

    NONE = "None"
    SMOOTH = "SMOOTH"
    ERASE = "ERASE"
    MASK = "MASK"


class SculptStrokeLocationMode(StrEnum):
    """How Blender resolves 3D locations for a screen-space stroke."""

    AUTO = "AUTO"
    SURFACE_RAYCAST = "SURFACE_RAYCAST"
    ANCHORED_DRAG = "ANCHORED_DRAG"


class SculptStrokeExecutionMode(StrEnum):
    """How the RPC server applies one validated Sculpt gesture."""

    AUTO = "AUTO"
    DIRECT_DABS = "DIRECT_DABS"
    PAINT_CURVE = "PAINT_CURVE"
    ANCHORED_DRAG = "ANCHORED_DRAG"


class OperatorStrokeElementInput(BaseModel):
    """One Blender OperatorStrokeElement from segment_with_sam3."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str = Field(min_length=1, max_length=128)
    location: list[float] = Field(min_length=3, max_length=3)
    mouse: list[float] = Field(min_length=2, max_length=2)
    mouse_event: list[float] = Field(min_length=2, max_length=2)
    pressure: float = Field(ge=0.0, le=1.0)
    size: float = Field(ge=1.0, le=10_000.0)
    x_tilt: float = Field(ge=-1.0, le=1.0)
    y_tilt: float = Field(ge=-1.0, le=1.0)
    time: float = Field(ge=0.0)
    is_start: bool

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Match the Add-on's printable stroke-name contract."""
        if any(ord(character) < 32 for character in value):
            raise ValueError("name contains unsupported characters")
        return value

    @model_validator(mode="after")
    def validate_mouse_coordinates(self) -> Self:
        """Require the same nonnegative region coordinates in both fields."""
        if any(
            abs(first - second) > 1e-6
            for first, second in zip(
                self.mouse,
                self.mouse_event,
                strict=True,
            )
        ):
            raise ValueError("mouse and mouse_event must match")
        if any(coordinate < 0.0 for coordinate in self.mouse_event):
            raise ValueError("mouse coordinates must be nonnegative")
        return self


class ExecuteSculptStrokeInput(BaseModel):
    """Input schema shown to the Agent model."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    operation_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description=(
            "Optional transaction-scoped idempotency ID persisted in the .blend."
        ),
    )

    stroke: list[OperatorStrokeElementInput] = Field(
        min_length=1,
        max_length=100_000,
        description=(
            "One operator_call.kwargs.stroke array from plan_sculpt_strokes; a "
            "tool call represents one mouse press-to-release gesture."
        ),
    )
    mode: SculptStrokeMode = Field(
        default=SculptStrokeMode.NORMAL,
        description="Sculpt operator mode: NORMAL or INVERT.",
    )
    brush_toggle: SculptStrokeBrushToggle = Field(
        default=SculptStrokeBrushToggle.NONE,
        description="Temporary brush toggle; defaults to None.",
    )
    pen_flip: bool = Field(
        default=False,
        description="Whether to use the stylus eraser side.",
    )
    execution_mode: SculptStrokeExecutionMode = Field(
        default=SculptStrokeExecutionMode.AUTO,
        description=(
            "DIRECT_DABS executes precomputed dabs, PAINT_CURVE reuses Blender "
            "spacing and overlap attenuation, and ANCHORED_DRAG fixes the origin."
        ),
    )
    location_mode: SculptStrokeLocationMode = Field(
        default=SculptStrokeLocationMode.AUTO,
        description=(
            "AUTO selects by active brush, SURFACE_RAYCAST hits the surface per "
            "point, and ANCHORED_DRAG hits once at the origin and keeps dragging."
        ),
    )
    ignore_background_click: Literal[True] = Field(
        default=True,
        description="Must be true to ignore points that miss the sculpt surface.",
    )
    window_index: int | None = Field(
        default=None,
        ge=0,
        description="Optional Blender window index.",
    )
    area_index: int | None = Field(
        default=None,
        ge=0,
        description="Optional VIEW_3D area index corresponding to the screenshot.",
    )

    @field_validator("mode", mode="before")
    @classmethod
    def normalize_mode(cls, value: object) -> object:
        """Accept case-insensitive operator modes."""
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("brush_toggle", mode="before")
    @classmethod
    def normalize_brush_toggle(cls, value: object) -> object:
        """Accept case-insensitive toggle strings including Blender's None."""
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        if normalized.casefold() == "none":
            return "None"
        return normalized.upper()

    @field_validator("location_mode", mode="before")
    @classmethod
    def normalize_location_mode(cls, value: object) -> object:
        """Accept case-insensitive location-mode values."""
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("execution_mode", mode="before")
    @classmethod
    def normalize_execution_mode(cls, value: object) -> object:
        """Accept case-insensitive execution-mode values."""
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @model_validator(mode="after")
    def validate_single_mouse_gesture(self) -> Self:
        """Keep one Tool invocation equal to one mouse-down gesture."""
        if not self.stroke[0].is_start:
            raise ValueError("the first stroke element must start the gesture")
        if any(element.is_start for element in self.stroke[1:]):
            raise ValueError("only the first stroke element may set is_start")
        times = [element.time for element in self.stroke]
        if times != sorted(times):
            raise ValueError("stroke time must be monotonically nondecreasing")
        return self


def create_execute_sculpt_stroke_tool(
    *,
    client: BlenderRpcClient | None = None,
    config: BlenderRpcConfig | None = None,
) -> StructuredTool:
    """Create a LangGraph-compatible Blender Sculpt stroke Tool."""
    if client is not None and config is not None:
        raise ValueError("client and config cannot both be provided")
    rpc_client = client or BlenderRpcClient(config)

    def invoke(
        stroke: list[OperatorStrokeElementInput],
        operation_id: str | None = None,
        mode: SculptStrokeMode = SculptStrokeMode.NORMAL,
        brush_toggle: SculptStrokeBrushToggle = (
            SculptStrokeBrushToggle.NONE
        ),
        pen_flip: bool = False,
        execution_mode: SculptStrokeExecutionMode = (
            SculptStrokeExecutionMode.AUTO
        ),
        location_mode: SculptStrokeLocationMode = (
            SculptStrokeLocationMode.AUTO
        ),
        ignore_background_click: Literal[True] = True,
        window_index: int | None = None,
        area_index: int | None = None,
    ) -> JsonValue:
        request = _build_request(
            stroke=stroke,
            operation_id=operation_id,
            mode=mode,
            brush_toggle=brush_toggle,
            pen_flip=pen_flip,
            execution_mode=execution_mode,
            location_mode=location_mode,
            ignore_background_click=ignore_background_click,
            window_index=window_index,
            area_index=area_index,
        )
        try:
            return rpc_client.send(request)
        except BlenderRpcBridgeError as error:
            raise ToolException(error.as_json()) from error

    async def ainvoke(
        stroke: list[OperatorStrokeElementInput],
        operation_id: str | None = None,
        mode: SculptStrokeMode = SculptStrokeMode.NORMAL,
        brush_toggle: SculptStrokeBrushToggle = (
            SculptStrokeBrushToggle.NONE
        ),
        pen_flip: bool = False,
        execution_mode: SculptStrokeExecutionMode = (
            SculptStrokeExecutionMode.AUTO
        ),
        location_mode: SculptStrokeLocationMode = (
            SculptStrokeLocationMode.AUTO
        ),
        ignore_background_click: Literal[True] = True,
        window_index: int | None = None,
        area_index: int | None = None,
    ) -> JsonValue:
        request = _build_request(
            stroke=stroke,
            operation_id=operation_id,
            mode=mode,
            brush_toggle=brush_toggle,
            pen_flip=pen_flip,
            execution_mode=execution_mode,
            location_mode=location_mode,
            ignore_background_click=ignore_background_click,
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
        name="execute_blender_sculpt_stroke",
        description=(
            "Execute one Blender sculpt stroke generated by plan_sculpt_strokes. "
            "Pass one operator_call kwargs object and its context window_index "
            "and area_index. Sculpt Mode and matching brush settings must already "
            "be active. Draw uses PAINT_CURVE, Smear uses DIRECT_DABS, and Drag "
            "uses ANCHORED_DRAG. Invoke this tool sequentially for a full plan."
        ),
        args_schema=ExecuteSculptStrokeInput,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: _invalid_input_error(),
    )


def _build_request(
    *,
    stroke: list[OperatorStrokeElementInput],
    operation_id: str | None,
    mode: SculptStrokeMode,
    brush_toggle: SculptStrokeBrushToggle,
    pen_flip: bool,
    execution_mode: SculptStrokeExecutionMode,
    location_mode: SculptStrokeLocationMode,
    ignore_background_click: Literal[True],
    window_index: int | None,
    area_index: int | None,
) -> dict[str, JsonValue]:
    params: dict[str, JsonValue] = {
        "stroke": [
            element.model_dump(mode="json") for element in stroke
        ],
        "mode": mode.value,
        "brush_toggle": brush_toggle.value,
        "pen_flip": pen_flip,
        "execution_mode": execution_mode.value,
        "location_mode": location_mode.value,
        "ignore_background_click": ignore_background_click,
    }
    if operation_id is not None:
        params["operation_id"] = operation_id
    if window_index is not None:
        params["window_index"] = window_index
    if area_index is not None:
        params["area_index"] = area_index
    return {
        "jsonrpc": "2.0",
        "id": f"execute-sculpt-stroke-{uuid4().hex}",
        "method": "sculpt_brush_stroke",
        "params": params,
    }


def _invalid_input_error() -> str:
    return json.dumps(
        {
            "bridge_error": {
                "type": "invalid_tool_input",
                "message": (
                    "Sculpt stroke Tool input must contain one complete, "
                    "ordered plan_sculpt_strokes stroke with safe operator "
                    "flags and optional nonnegative viewport indices"
                ),
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
