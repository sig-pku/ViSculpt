"""Validate operator-ready Sculpt stroke requests without importing bpy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NoReturn

from .protocol import JsonRpcError

MAX_SCULPT_STROKE_ELEMENTS = 100_000
MAX_PAINT_CURVE_POINTS = 2_048
PAINT_CURVE_SIMPLIFICATION_TOLERANCE = 0.75

_REQUEST_FIELDS = {
    "operation_id",
    "stroke",
    "mode",
    "brush_toggle",
    "pen_flip",
    "execution_mode",
    "location_mode",
    "ignore_background_click",
    "window_index",
    "area_index",
}
_STROKE_FIELDS = {
    "name",
    "location",
    "mouse",
    "mouse_event",
    "pressure",
    "size",
    "x_tilt",
    "y_tilt",
    "time",
    "is_start",
}
_MODES = {"NORMAL", "INVERT"}
SCULPT_STROKE_LOCATION_MODES = {
    "AUTO",
    "SURFACE_RAYCAST",
    "ANCHORED_DRAG",
}
SCULPT_STROKE_EXECUTION_MODES = {
    "AUTO",
    "DIRECT_DABS",
    "PAINT_CURVE",
    "ANCHORED_DRAG",
}
_ANCHORED_DRAG_BRUSH_TYPES = {
    "BOUNDARY",
    "ELASTIC_DEFORM",
    "GRAB",
    "POSE",
    "ROTATE",
    "SNAKE_HOOK",
    "THUMB",
}
_BRUSH_TOGGLES = {
    "none": "None",
    "smooth": "SMOOTH",
    "erase": "ERASE",
    "mask": "MASK",
}


@dataclass(frozen=True, slots=True)
class SculptBrushStrokeRequest:
    """Validated parameters for one mouse-down Sculpt gesture."""

    operation_id: str | None
    window_index: int | None
    area_index: int | None
    execution_mode: str
    location_mode: str
    operator_kwargs: dict[str, Any]
    element_count: int
    mouse_min: tuple[float, float]
    mouse_max: tuple[float, float]


def parse_sculpt_brush_stroke_request(
    params: dict[str, Any],
    *,
    region_width: int,
    region_height: int,
) -> SculptBrushStrokeRequest:
    """Validate a segment_with_sam3 operator-call kwargs object."""
    unknown = sorted(set(params) - _REQUEST_FIELDS)
    if unknown:
        _invalid("request contains unknown parameters", unknown=unknown)

    window_index = _optional_index(params, "window_index")
    area_index = _optional_index(params, "area_index")
    operation_id = _optional_operation_id(params.get("operation_id"))
    mode = _mode(params.get("mode", "NORMAL"))
    brush_toggle = _brush_toggle(params.get("brush_toggle", "None"))
    pen_flip = _boolean(params.get("pen_flip", False), "pen_flip")
    execution_mode = _execution_mode(params.get("execution_mode", "AUTO"))
    location_mode = _location_mode(params.get("location_mode", "AUTO"))
    ignore_background_click = _boolean(
        params.get("ignore_background_click", True),
        "ignore_background_click",
    )
    if not ignore_background_click:
        _invalid(
            "ignore_background_click must be true for safe Sculpt replay"
        )

    raw_stroke = params.get("stroke")
    if not isinstance(raw_stroke, list):
        _invalid("stroke must be a JSON array")
    if not raw_stroke:
        _invalid("stroke must contain at least one element")
    if len(raw_stroke) > MAX_SCULPT_STROKE_ELEMENTS:
        _invalid(
            "stroke exceeds the element safety limit",
            maximum=MAX_SCULPT_STROKE_ELEMENTS,
        )
    if region_width <= 0 or region_height <= 0:
        raise ValueError("region dimensions must be positive")

    stroke: list[dict[str, Any]] = []
    previous_time: float | None = None
    mouse_min_x = math.inf
    mouse_min_y = math.inf
    mouse_max_x = -math.inf
    mouse_max_y = -math.inf
    for index, raw_element in enumerate(raw_stroke):
        element = _stroke_element(
            raw_element,
            index=index,
            region_width=region_width,
            region_height=region_height,
        )
        if index == 0 and not element["is_start"]:
            _invalid("the first stroke element must set is_start=true")
        if index > 0 and element["is_start"]:
            _invalid(
                "only the first stroke element may set is_start=true",
                element_index=index,
            )
        element_time = element["time"]
        if previous_time is not None and element_time < previous_time:
            _invalid(
                "stroke element time must be monotonically nondecreasing",
                element_index=index,
            )
        previous_time = element_time
        mouse_x, mouse_y = element["mouse_event"]
        mouse_min_x = min(mouse_min_x, mouse_x)
        mouse_min_y = min(mouse_min_y, mouse_y)
        mouse_max_x = max(mouse_max_x, mouse_x)
        mouse_max_y = max(mouse_max_y, mouse_y)
        stroke.append(element)

    _validate_execution_contract(
        execution_mode=execution_mode,
        location_mode=location_mode,
        mode=mode,
        brush_toggle=brush_toggle,
        pen_flip=pen_flip,
        stroke=stroke,
    )

    return SculptBrushStrokeRequest(
        operation_id=operation_id,
        window_index=window_index,
        area_index=area_index,
        execution_mode=execution_mode,
        location_mode=location_mode,
        operator_kwargs={
            "stroke": stroke,
            "mode": mode,
            "brush_toggle": brush_toggle,
            "pen_flip": pen_flip,
            "ignore_background_click": ignore_background_click,
        },
        element_count=len(stroke),
        mouse_min=(mouse_min_x, mouse_min_y),
        mouse_max=(mouse_max_x, mouse_max_y),
    )


def _optional_operation_id(value: Any) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 256
        or any(ord(character) < 32 for character in value)
    ):
        _invalid(
            "operation_id must be a printable non-empty string",
            maximum_length=256,
        )
    return value.strip()


def resolve_sculpt_stroke_execution_mode(
    requested_mode: str,
    *,
    sculpt_brush_type: str,
    brush_toggle: str,
    cloth_deform_type: str | None = None,
) -> str:
    """Resolve the replay mechanism from explicit brush semantics."""
    mode = _execution_mode(requested_mode)
    brush_type = str(sculpt_brush_type or "").strip().upper()
    cloth_type = str(cloth_deform_type or "").strip().upper()
    anchored_brush = brush_type in _ANCHORED_DRAG_BRUSH_TYPES or (
        brush_type == "CLOTH" and cloth_type == "GRAB"
    )
    supports_anchored_drag = anchored_brush and brush_toggle == "None"

    if mode == "AUTO":
        return "ANCHORED_DRAG" if supports_anchored_drag else "DIRECT_DABS"
    if mode == "ANCHORED_DRAG" and not supports_anchored_drag:
        _invalid(
            "ANCHORED_DRAG execution requires an anchored Sculpt brush "
            "without a temporary brush toggle",
            sculpt_brush_type=brush_type or None,
            brush_toggle=brush_toggle,
        )
    if mode == "DIRECT_DABS" and anchored_brush:
        _invalid(
            "DIRECT_DABS is reserved for non-anchored surface brushes",
            sculpt_brush_type=brush_type or None,
        )
    if mode == "PAINT_CURVE" and anchored_brush:
        _invalid(
            "PAINT_CURVE is reserved for continuous surface brushes",
            sculpt_brush_type=brush_type or None,
        )
    return mode


def resolve_sculpt_stroke_location_mode(
    requested_mode: str,
    *,
    sculpt_brush_type: str,
    brush_toggle: str,
    cloth_deform_type: str | None = None,
) -> str:
    """Resolve AUTO without importing Blender and reject unsafe pairings."""
    mode = _location_mode(requested_mode)
    brush_type = str(sculpt_brush_type or "").strip().upper()
    cloth_type = str(cloth_deform_type or "").strip().upper()
    anchored_brush = brush_type in _ANCHORED_DRAG_BRUSH_TYPES or (
        brush_type == "CLOTH" and cloth_type == "GRAB"
    )
    uses_temporary_brush = brush_toggle != "None"
    supports_anchored_drag = anchored_brush and not uses_temporary_brush

    if mode == "AUTO":
        return (
            "ANCHORED_DRAG"
            if supports_anchored_drag
            else "SURFACE_RAYCAST"
        )
    if mode == "ANCHORED_DRAG" and not supports_anchored_drag:
        _invalid(
            "ANCHORED_DRAG requires an anchored Sculpt brush without a "
            "temporary brush toggle",
            sculpt_brush_type=brush_type or None,
            brush_toggle=brush_toggle,
        )
    return mode


def _validate_execution_contract(
    *,
    execution_mode: str,
    location_mode: str,
    mode: str,
    brush_toggle: str,
    pen_flip: bool,
    stroke: list[dict[str, Any]],
) -> None:
    """Reject combinations that the selected replay path cannot represent."""
    if execution_mode == "DIRECT_DABS" and location_mode == "ANCHORED_DRAG":
        _invalid(
            "DIRECT_DABS cannot use ANCHORED_DRAG location semantics"
        )
    if execution_mode == "ANCHORED_DRAG" and location_mode not in {
        "AUTO",
        "ANCHORED_DRAG",
    }:
        _invalid(
            "ANCHORED_DRAG execution requires AUTO or ANCHORED_DRAG "
            "location_mode"
        )
    if execution_mode != "PAINT_CURVE":
        return
    if location_mode not in {"AUTO", "SURFACE_RAYCAST"}:
        _invalid(
            f"{execution_mode} requires AUTO or SURFACE_RAYCAST "
            "location_mode"
        )
    if mode != "NORMAL" or brush_toggle != "None" or pen_flip:
        _invalid(
            f"{execution_mode} supports NORMAL mode without brush toggle "
            "or pen flip"
        )
    for index, element in enumerate(stroke):
        if (
            abs(float(element["pressure"]) - 1.0) > 1e-6
            or abs(float(element["x_tilt"])) > 1e-6
            or abs(float(element["y_tilt"])) > 1e-6
        ):
            _invalid(
                f"{execution_mode} requires mouse-equivalent full "
                "pressure and zero tilt",
                element_index=index,
            )


def prepare_sculpt_paint_curve_points(
    stroke: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    """Simplify a screen-space gesture for Blender's native Paint Curve."""
    rounded: list[tuple[int, int]] = []
    for element in stroke:
        mouse = element["mouse_event"]
        point = (int(round(float(mouse[0]))), int(round(float(mouse[1]))))
        if not rounded or point != rounded[-1]:
            rounded.append(point)
    if len(rounded) <= 2:
        return rounded

    simplified = _simplify_polyline(
        rounded,
        tolerance=PAINT_CURVE_SIMPLIFICATION_TOLERANCE,
    )
    if len(simplified) > MAX_PAINT_CURVE_POINTS:
        _invalid(
            "PAINT_CURVE gesture remains too complex after simplification",
            point_count=len(simplified),
            maximum=MAX_PAINT_CURVE_POINTS,
        )
    return simplified


def _simplify_polyline(
    points: list[tuple[int, int]],
    *,
    tolerance: float,
) -> list[tuple[int, int]]:
    """Apply deterministic Ramer-Douglas-Peucker simplification."""
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    tolerance_squared = tolerance * tolerance
    while stack:
        start, end = stack.pop()
        farthest_index = -1
        farthest_distance = tolerance_squared
        for index in range(start + 1, end):
            distance = _point_segment_distance_squared(
                points[index],
                points[start],
                points[end],
            )
            if distance > farthest_distance:
                farthest_distance = distance
                farthest_index = index
        if farthest_index >= 0:
            keep.add(farthest_index)
            stack.append((farthest_index, end))
            stack.append((start, farthest_index))
    return [points[index] for index in sorted(keep)]


def _point_segment_distance_squared(
    point: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
) -> float:
    """Return squared screen-space distance from a point to a segment."""
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    if dx == 0.0 and dy == 0.0:
        return float(
            (point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2
        )
    projection = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / (dx * dx + dy * dy)
    projection = min(1.0, max(0.0, projection))
    projected_x = start[0] + projection * dx
    projected_y = start[1] + projection * dy
    return (point[0] - projected_x) ** 2 + (
        point[1] - projected_y
    ) ** 2


def anchor_sculpt_drag_stroke(
    stroke: list[dict[str, Any]],
    initial_location: list[float],
) -> list[dict[str, Any]]:
    """Copy a stroke and bind every element to one object-space anchor."""
    if len(initial_location) != 3 or not all(
        math.isfinite(float(value)) for value in initial_location
    ):
        raise ValueError("initial_location must contain three finite values")
    anchor = [float(value) for value in initial_location]
    return [
        {
            **element,
            "location": anchor.copy(),
        }
        for element in stroke
    ]


def _stroke_element(
    value: object,
    *,
    index: int,
    region_width: int,
    region_height: int,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _invalid("each stroke element must be an object", element_index=index)
    unknown = sorted(set(value) - _STROKE_FIELDS)
    missing = sorted(_STROKE_FIELDS - set(value))
    if unknown or missing:
        _invalid(
            "stroke element fields do not match OperatorStrokeElement",
            element_index=index,
            unknown=unknown,
            missing=missing,
        )

    name = value["name"]
    if (
        not isinstance(name, str)
        or not name
        or len(name) > 128
        or any(ord(character) < 32 for character in name)
    ):
        _invalid(
            "stroke element name must be 1 to 128 printable characters",
            element_index=index,
        )

    location = _vector(
        value["location"],
        length=3,
        label="location",
        element_index=index,
    )
    mouse = _vector(
        value["mouse"],
        length=2,
        label="mouse",
        element_index=index,
    )
    mouse_event = _vector(
        value["mouse_event"],
        length=2,
        label="mouse_event",
        element_index=index,
    )
    if any(
        abs(first - second) > 1e-6
        for first, second in zip(mouse, mouse_event, strict=True)
    ):
        _invalid(
            "mouse and mouse_event must contain the same coordinates",
            element_index=index,
        )
    if not 0.0 <= mouse_event[0] < region_width:
        _invalid(
            "mouse_event x is outside the target VIEW_3D region",
            element_index=index,
            region_width=region_width,
        )
    if not 0.0 <= mouse_event[1] < region_height:
        _invalid(
            "mouse_event y is outside the target VIEW_3D region",
            element_index=index,
            region_height=region_height,
        )

    return {
        "name": name,
        "location": location,
        "mouse": mouse,
        "mouse_event": mouse_event,
        "pressure": _number(
            value["pressure"],
            label="pressure",
            element_index=index,
            minimum=0.0,
            maximum=1.0,
        ),
        "size": _number(
            value["size"],
            label="size",
            element_index=index,
            minimum=1.0,
            maximum=10_000.0,
        ),
        "x_tilt": _number(
            value["x_tilt"],
            label="x_tilt",
            element_index=index,
            minimum=-1.0,
            maximum=1.0,
        ),
        "y_tilt": _number(
            value["y_tilt"],
            label="y_tilt",
            element_index=index,
            minimum=-1.0,
            maximum=1.0,
        ),
        "time": _number(
            value["time"],
            label="time",
            element_index=index,
            minimum=0.0,
        ),
        "is_start": _boolean(
            value["is_start"],
            "is_start",
            element_index=index,
        ),
    }


def _mode(value: object) -> str:
    if not isinstance(value, str) or value.upper() not in _MODES:
        _invalid("mode must be NORMAL or INVERT")
    return value.upper()


def _location_mode(value: object) -> str:
    if not isinstance(value, str):
        _invalid("location_mode must be a string")
    normalized = value.strip().upper()
    if normalized not in SCULPT_STROKE_LOCATION_MODES:
        _invalid(
            "location_mode must be AUTO, SURFACE_RAYCAST, or "
            "ANCHORED_DRAG"
        )
    return normalized


def _execution_mode(value: object) -> str:
    if not isinstance(value, str):
        _invalid("execution_mode must be a string")
    normalized = value.strip().upper()
    if normalized not in SCULPT_STROKE_EXECUTION_MODES:
        _invalid(
            "execution_mode must be AUTO, DIRECT_DABS, PAINT_CURVE, or "
            "ANCHORED_DRAG"
        )
    return normalized


def _brush_toggle(value: object) -> str:
    if not isinstance(value, str):
        _invalid("brush_toggle must be a string")
    normalized = _BRUSH_TOGGLES.get(value.casefold())
    if normalized is None:
        _invalid(
            "brush_toggle must be None, SMOOTH, ERASE, or MASK"
        )
    return normalized


def _optional_index(params: dict[str, Any], name: str) -> int | None:
    if name not in params:
        return None
    value = params[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _invalid(f"{name} must be a nonnegative integer")
    return value


def _vector(
    value: object,
    *,
    length: int,
    label: str,
    element_index: int,
) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        _invalid(
            f"{label} must be an array of length {length}",
            element_index=element_index,
        )
    return [
        _number(
            item,
            label=f"{label}[{item_index}]",
            element_index=element_index,
        )
        for item_index, item in enumerate(value)
    ]


def _number(
    value: object,
    *,
    label: str,
    element_index: int,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _invalid(
            f"{label} must be a finite number",
            element_index=element_index,
        )
    result = float(value)
    if not math.isfinite(result):
        _invalid(
            f"{label} must be a finite number",
            element_index=element_index,
        )
    if minimum is not None and result < minimum:
        _invalid(
            f"{label} is below the supported range",
            element_index=element_index,
            minimum=minimum,
        )
    if maximum is not None and result > maximum:
        _invalid(
            f"{label} is above the supported range",
            element_index=element_index,
            maximum=maximum,
        )
    return result


def _boolean(
    value: object,
    label: str,
    *,
    element_index: int | None = None,
) -> bool:
    if not isinstance(value, bool):
        data: dict[str, Any] = {}
        if element_index is not None:
            data["element_index"] = element_index
        _invalid(f"{label} must be a boolean", **data)
    return value


def _invalid(reason: str, **data: Any) -> NoReturn:
    raise JsonRpcError(
        -32602,
        "Invalid params",
        {"reason": reason, **data},
    )
