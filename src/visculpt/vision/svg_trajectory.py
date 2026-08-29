"""Deterministic mouse-trajectory generation from validated SVG line art."""

from __future__ import annotations

import math
from dataclasses import dataclass
from io import StringIO

from svgelements import Close, Path, SVG, Shape

from visculpt.bridge import JsonValue

_CANVAS_MIN = 0.0
_CANVAS_MAX = 512.0
_LENGTH_ERROR = 0.01
_MAX_FLATTENING_SEGMENTS = 16_384
_MAX_OUTPUT_POINTS = 100_000
_POINT_EPSILON = 1e-7
_FILLABLE_TAGS = {
    "circle",
    "ellipse",
    "path",
    "polygon",
    "polyline",
    "rect",
}

type Point = tuple[float, float]


class SvgTrajectoryGenerationError(ValueError):
    """Raised when validated SVG cannot produce bounded trajectories."""


@dataclass(frozen=True, slots=True)
class SvgMouseTrajectoryResult:
    """JSON-compatible deterministic SVG trajectory plan."""

    trajectory_plan: dict[str, JsonValue]

    def as_payload(self) -> dict[str, JsonValue]:
        """Return the trajectory plan used by LangGraph state."""
        return self.trajectory_plan


def generate_svg_mouse_trajectories(
    *,
    svg: str,
    point_spacing_pixels: float = 4.0,
    flattening_spacing_pixels: float = 1.0,
) -> SvgMouseTrajectoryResult:
    """Convert validated 512x512 SVG geometry into uniform mouse paths."""
    _validate_sampling_parameters(
        point_spacing_pixels=point_spacing_pixels,
        flattening_spacing_pixels=flattening_spacing_pixels,
    )
    try:
        document = SVG.parse(
            StringIO(svg),
            reify=True,
            on_error="raise",
        )
    except Exception as error:
        raise SvgTrajectoryGenerationError(
            f"SVG geometry parsing failed: {error}"
        ) from error
    if not isinstance(document, SVG):
        raise SvgTrajectoryGenerationError(
            "SVG geometry parser did not return a document"
        )

    trajectories: list[dict[str, JsonValue]] = []
    source_shape_index = -1
    for element in document.elements():
        if not isinstance(element, Shape):
            continue
        source_shape_index += 1
        tag = _shape_tag(element)
        visible_stroke = _is_black(element.stroke) and (
            _stroke_width(element) > 0.0
        )
        visible_fill = tag in _FILLABLE_TAGS and _is_black(element.fill)
        if not visible_stroke and not visible_fill:
            continue
        paint_mode = "stroke" if visible_stroke else "fill_outline"
        try:
            source_path = Path(element)
            source_path.reify()
            subpaths = list(source_path.as_subpaths())
        except Exception as error:
            raise SvgTrajectoryGenerationError(
                f"Failed to normalize {tag} geometry: {error}"
            ) from error

        for subpath_index, raw_subpath in enumerate(subpaths, start=1):
            subpath = Path(raw_subpath)
            originally_closed = bool(subpath) and isinstance(
                subpath[-1], Close
            )
            sampled = _flatten_path(
                subpath,
                spacing=flattening_spacing_pixels,
            )
            if (
                paint_mode == "fill_outline"
                and sampled
                and not _points_close(sampled[0], sampled[-1])
            ):
                sampled.append(sampled[0])
                originally_closed = True

            clipped_paths = _clip_polyline_to_canvas(sampled)
            for clipped_path in clipped_paths:
                points = _uniformly_resample(
                    clipped_path,
                    spacing=point_spacing_pixels,
                )
                if not points:
                    continue
                closed = (
                    originally_closed
                    and len(clipped_paths) == 1
                    and len(points) > 1
                    and _points_close(points[0], points[-1])
                )
                trajectory_id = f"trajectory-{len(trajectories) + 1:03d}"
                point_payload = [
                    {
                        "x": _rounded_coordinate(point[0]),
                        "y": _rounded_coordinate(point[1]),
                    }
                    for point in points
                ]
                trajectories.append(
                    {
                        "id": trajectory_id,
                        "source": {
                            "element_index": source_shape_index,
                            "tag": tag,
                            "subpath_index": subpath_index,
                            "paint_mode": paint_mode,
                        },
                        "closed": closed,
                        "length_pixels": round(
                            _polyline_length(points), 6
                        ),
                        "point_count": len(point_payload),
                        "points": point_payload,
                    }
                )

    if not trajectories:
        raise SvgTrajectoryGenerationError(
            "SVG contains no drawable black geometry inside the 0..512 "
            "canvas"
        )
    total_points = sum(
        int(trajectory["point_count"]) for trajectory in trajectories
    )
    if total_points > _MAX_OUTPUT_POINTS:
        raise SvgTrajectoryGenerationError(
            "Generated trajectory plan exceeds the 100000-point safety "
            "limit"
        )

    plan: dict[str, JsonValue] = {
        "format": "svg-mouse-trajectories/v1",
        "algorithm": {
            "name": "flatten_clip_uniform_arclength",
            "deterministic": True,
            "flattening_spacing_pixels": flattening_spacing_pixels,
            "point_spacing_pixels": point_spacing_pixels,
            "off_canvas_policy": "CLIP",
        },
        "coordinate_system": {
            "width": 512,
            "height": 512,
            "origin": "TOP_LEFT",
            "x_range": [0.0, 512.0],
            "y_range": [0.0, 512.0],
            "units": "SVG_USER_UNITS",
        },
        "gesture_contract": {
            "one_trajectory_per_mouse_gesture": True,
            "first_point": "MOUSE_DOWN",
            "intermediate_points": "MOUSE_MOVE",
            "last_point": "MOUSE_UP",
        },
        "trajectories": trajectories,
        "summary": {
            "trajectory_count": len(trajectories),
            "point_count": total_points,
            "closed_trajectory_count": sum(
                int(bool(trajectory["closed"]))
                for trajectory in trajectories
            ),
            "total_length_pixels": round(
                sum(
                    float(trajectory["length_pixels"])
                    for trajectory in trajectories
                ),
                6,
            ),
        },
    }
    return SvgMouseTrajectoryResult(trajectory_plan=plan)


def _validate_sampling_parameters(
    *,
    point_spacing_pixels: float,
    flattening_spacing_pixels: float,
) -> None:
    for name, value in (
        ("point_spacing_pixels", point_spacing_pixels),
        ("flattening_spacing_pixels", flattening_spacing_pixels),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise SvgTrajectoryGenerationError(
                f"{name} must be a finite positive number"
            )
    if flattening_spacing_pixels > point_spacing_pixels:
        raise SvgTrajectoryGenerationError(
            "flattening_spacing_pixels must not exceed "
            "point_spacing_pixels"
        )


def _shape_tag(element: Shape) -> str:
    values = getattr(element, "values", {})
    tag = values.get("tag") if isinstance(values, dict) else None
    if isinstance(tag, str) and tag:
        return tag.casefold()
    return type(element).__name__.casefold()


def _is_black(paint: object) -> bool:
    return str(paint).strip().casefold() in {
        "black",
        "#000",
        "#000000",
        "rgb(0,0,0)",
    }


def _stroke_width(element: Shape) -> float:
    try:
        width = float(element.stroke_width)
    except (TypeError, ValueError):
        return 0.0
    return width if math.isfinite(width) else 0.0


def _flatten_path(path: Path, *, spacing: float) -> list[Point]:
    try:
        length = float(path.length(error=_LENGTH_ERROR, min_depth=4))
    except Exception as error:
        raise SvgTrajectoryGenerationError(
            f"SVG path length calculation failed: {error}"
        ) from error
    if not math.isfinite(length) or length < 0.0:
        raise SvgTrajectoryGenerationError(
            "SVG path has a non-finite length"
        )
    segment_count = max(1, int(math.ceil(length / spacing)))
    segment_count = min(segment_count, _MAX_FLATTENING_SEGMENTS)
    points: list[Point] = []
    for index in range(segment_count + 1):
        try:
            raw_point = path.point(
                index / segment_count,
                error=_LENGTH_ERROR,
            )
        except Exception as error:
            raise SvgTrajectoryGenerationError(
                f"SVG path sampling failed: {error}"
            ) from error
        if raw_point is None:
            continue
        point = (float(raw_point.x), float(raw_point.y))
        if not all(math.isfinite(value) for value in point):
            raise SvgTrajectoryGenerationError(
                "SVG path contains a non-finite coordinate"
            )
        _append_distinct(points, point)
    return points


def _clip_polyline_to_canvas(points: list[Point]) -> list[list[Point]]:
    if not points:
        return []
    if len(points) == 1:
        return [[_bound_point(points[0])]] if _inside_canvas(points[0]) else []

    clipped_paths: list[list[Point]] = []
    current: list[Point] = []
    for start, end in zip(points, points[1:]):
        clipped = _clip_segment(start, end)
        if clipped is None:
            _flush_path(current, clipped_paths)
            current = []
            continue
        clipped_start, clipped_end = clipped
        if current and not _points_close(current[-1], clipped_start):
            _flush_path(current, clipped_paths)
            current = []
        _append_distinct(current, clipped_start)
        _append_distinct(current, clipped_end)
    _flush_path(current, clipped_paths)
    return clipped_paths


def _clip_segment(start: Point, end: Point) -> tuple[Point, Point] | None:
    x0, y0 = start
    x1, y1 = end
    delta_x = x1 - x0
    delta_y = y1 - y0
    lower = 0.0
    upper = 1.0
    for coefficient, distance in (
        (-delta_x, x0 - _CANVAS_MIN),
        (delta_x, _CANVAS_MAX - x0),
        (-delta_y, y0 - _CANVAS_MIN),
        (delta_y, _CANVAS_MAX - y0),
    ):
        if math.isclose(coefficient, 0.0, abs_tol=_POINT_EPSILON):
            if distance < 0.0:
                return None
            continue
        ratio = distance / coefficient
        if coefficient < 0.0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return None
    clipped_start = (
        x0 + lower * delta_x,
        y0 + lower * delta_y,
    )
    clipped_end = (
        x0 + upper * delta_x,
        y0 + upper * delta_y,
    )
    return _bound_point(clipped_start), _bound_point(clipped_end)


def _uniformly_resample(points: list[Point], *, spacing: float) -> list[Point]:
    distinct: list[Point] = []
    for point in points:
        _append_distinct(distinct, point)
    if len(distinct) <= 1:
        return distinct

    segment_lengths = [
        math.dist(start, end)
        for start, end in zip(distinct, distinct[1:])
    ]
    total_length = sum(segment_lengths)
    if total_length <= _POINT_EPSILON:
        return [distinct[0]]
    step_count = max(1, int(math.ceil(total_length / spacing)))
    targets = [
        total_length * index / step_count
        for index in range(step_count + 1)
    ]
    result: list[Point] = []
    segment_index = 0
    segment_start_distance = 0.0
    for target in targets:
        while (
            segment_index < len(segment_lengths) - 1
            and target
            > segment_start_distance + segment_lengths[segment_index]
        ):
            segment_start_distance += segment_lengths[segment_index]
            segment_index += 1
        segment_length = segment_lengths[segment_index]
        if segment_length <= _POINT_EPSILON:
            point = distinct[segment_index]
        else:
            ratio = (target - segment_start_distance) / segment_length
            start = distinct[segment_index]
            end = distinct[segment_index + 1]
            point = (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            )
        _append_distinct(result, _bound_point(point))
    return result


def _polyline_length(points: list[Point]) -> float:
    return sum(
        math.dist(start, end)
        for start, end in zip(points, points[1:])
    )


def _inside_canvas(point: Point) -> bool:
    return (
        _CANVAS_MIN <= point[0] <= _CANVAS_MAX
        and _CANVAS_MIN <= point[1] <= _CANVAS_MAX
    )


def _bound_point(point: Point) -> Point:
    return (
        min(_CANVAS_MAX, max(_CANVAS_MIN, point[0])),
        min(_CANVAS_MAX, max(_CANVAS_MIN, point[1])),
    )


def _append_distinct(points: list[Point], point: Point) -> None:
    if not points or not _points_close(points[-1], point):
        points.append(point)


def _points_close(first: Point, second: Point) -> bool:
    return math.dist(first, second) <= _POINT_EPSILON


def _flush_path(current: list[Point], output: list[list[Point]]) -> None:
    if current:
        output.append(current.copy())


def _rounded_coordinate(value: float) -> float:
    rounded = round(min(_CANVAS_MAX, max(_CANVAS_MIN, value)), 6)
    return 0.0 if rounded == -0.0 else rounded
