"""Validate Sculpt Face Set lasso paths without importing bpy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NoReturn

from .protocol import JsonRpcError

MAX_FACE_SET_LASSO_POINTS = 100_000
MIN_FACE_SET_LASSO_POINTS = 3

_REQUEST_FIELDS = {
    "path",
    "use_front_faces_only",
    "show_face_sets",
    "window_index",
    "area_index",
}
_PATH_FIELDS = {"name", "loc", "time"}


@dataclass(frozen=True, slots=True)
class FaceSetLassoRequest:
    """Validated kwargs for one Face Set lasso operator call."""

    window_index: int | None
    area_index: int | None
    show_face_sets: bool
    operator_kwargs: dict[str, Any]
    point_count: int
    mouse_min: tuple[float, float]
    mouse_max: tuple[float, float]


def parse_face_set_lasso_request(
    params: dict[str, Any],
    *,
    region_width: int,
    region_height: int,
) -> FaceSetLassoRequest:
    """Validate a VIEW_3D region-local OperatorMousePath."""
    unknown = sorted(set(params) - _REQUEST_FIELDS)
    if unknown:
        _invalid("request contains unknown parameters", unknown=unknown)

    window_index = _optional_index(params, "window_index")
    area_index = _optional_index(params, "area_index")
    use_front_faces_only = _boolean(
        params.get("use_front_faces_only", False),
        "use_front_faces_only",
    )
    show_face_sets = _boolean(
        params.get("show_face_sets", False),
        "show_face_sets",
    )
    raw_path = params.get("path")
    if not isinstance(raw_path, list):
        _invalid("path must be a JSON array")
    if len(raw_path) < MIN_FACE_SET_LASSO_POINTS:
        _invalid(
            "path must contain at least three points",
            minimum=MIN_FACE_SET_LASSO_POINTS,
        )
    if len(raw_path) > MAX_FACE_SET_LASSO_POINTS:
        _invalid(
            "path exceeds the point safety limit",
            maximum=MAX_FACE_SET_LASSO_POINTS,
        )
    if region_width <= 0 or region_height <= 0:
        raise ValueError("region dimensions must be positive")

    operator_path: list[dict[str, Any]] = []
    unique_pixels: set[tuple[int, int]] = set()
    previous_time: float | None = None
    mouse_min_x = math.inf
    mouse_min_y = math.inf
    mouse_max_x = -math.inf
    mouse_max_y = -math.inf

    for index, raw_point in enumerate(raw_path):
        point = _path_point(
            raw_point,
            index=index,
            default_time=(
                previous_time if previous_time is not None else 0.0
            ),
            region_width=region_width,
            region_height=region_height,
        )
        point_time = point["time"]
        if previous_time is not None and point_time < previous_time:
            _invalid(
                "path point time must be monotonically nondecreasing",
                point_index=index,
            )
        previous_time = point_time

        mouse_x, mouse_y = point["loc"]
        mouse_min_x = min(mouse_min_x, mouse_x)
        mouse_min_y = min(mouse_min_y, mouse_y)
        mouse_max_x = max(mouse_max_x, mouse_x)
        mouse_max_y = max(mouse_max_y, mouse_y)
        unique_pixels.add((int(mouse_x), int(mouse_y)))
        operator_path.append(point)

    if len(unique_pixels) < MIN_FACE_SET_LASSO_POINTS:
        _invalid(
            "path must contain at least three distinct pixel locations",
            minimum=MIN_FACE_SET_LASSO_POINTS,
        )
    if mouse_min_x == mouse_max_x or mouse_min_y == mouse_max_y:
        _invalid("path must span a nonzero width and height")

    return FaceSetLassoRequest(
        window_index=window_index,
        area_index=area_index,
        show_face_sets=show_face_sets,
        operator_kwargs={
            "path": operator_path,
            "use_front_faces_only": use_front_faces_only,
        },
        point_count=len(operator_path),
        mouse_min=(mouse_min_x, mouse_min_y),
        mouse_max=(mouse_max_x, mouse_max_y),
    )


def _path_point(
    value: object,
    *,
    index: int,
    default_time: float,
    region_width: int,
    region_height: int,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _invalid("each path point must be an object", point_index=index)
    unknown = sorted(set(value) - _PATH_FIELDS)
    if unknown:
        _invalid(
            "path point contains unknown fields",
            point_index=index,
            unknown=unknown,
        )
    if "loc" not in value:
        _invalid("path point loc is required", point_index=index)

    name = value.get("name")
    if name is not None and (
        not isinstance(name, str)
        or len(name) > 128
        or any(ord(character) < 32 for character in name)
    ):
        _invalid(
            "path point name must be a printable string up to 128 characters",
            point_index=index,
        )

    loc = _vector(value["loc"], index=index)
    if not 0.0 <= loc[0] < region_width:
        _invalid(
            "path point x is outside the target VIEW_3D region",
            point_index=index,
            region_width=region_width,
        )
    if not 0.0 <= loc[1] < region_height:
        _invalid(
            "path point y is outside the target VIEW_3D region",
            point_index=index,
            region_height=region_height,
        )

    return {
        # Some Python-to-RNA paths inherit the required name field.
        "name": name or "",
        "loc": loc,
        "time": _number(
            value.get("time", default_time),
            label="time",
            point_index=index,
            minimum=0.0,
        ),
    }


def _optional_index(params: dict[str, Any], name: str) -> int | None:
    if name not in params:
        return None
    value = params[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _invalid(f"{name} must be a nonnegative integer")
    return value


def _vector(value: object, *, index: int) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        _invalid(
            "loc must be an array of length 2",
            point_index=index,
        )
    return [
        _number(
            item,
            label=f"loc[{item_index}]",
            point_index=index,
        )
        for item_index, item in enumerate(value)
    ]


def _number(
    value: object,
    *,
    label: str,
    point_index: int,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _invalid(
            f"{label} must be a finite number",
            point_index=point_index,
        )
    result = float(value)
    if not math.isfinite(result):
        _invalid(
            f"{label} must be a finite number",
            point_index=point_index,
        )
    if minimum is not None and result < minimum:
        _invalid(
            f"{label} is below the supported range",
            point_index=point_index,
            minimum=minimum,
        )
    return result


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        _invalid(f"{label} must be a boolean")
    return value


def _invalid(reason: str, **data: Any) -> NoReturn:
    raise JsonRpcError(
        -32602,
        "Invalid params",
        {"reason": reason, **data},
    )
