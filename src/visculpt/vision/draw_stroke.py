"""Deterministic Blender Sculpt Draw planning from fitted trajectories."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw, UnidentifiedImageError

from visculpt.bridge import JsonValue

from .sculpt_stroke import ScreenshotViewportMapping

_MOUSE_SPEED_PIXELS_PER_SECOND = 600.0
_MAX_STROKE_ELEMENTS = 100_000
_PALETTE = (
    "#e63946",
    "#2a9d8f",
    "#457b9d",
    "#f4a261",
    "#9b5de5",
    "#00b4d8",
    "#ff006e",
    "#6a994e",
)

type Point = tuple[float, float]


class DrawStrokePlanningError(ValueError):
    """Raised when fitted Draw trajectories cannot form operator calls."""


@dataclass(frozen=True, slots=True)
class DrawStrokePlanResult:
    """Persisted Draw operator plan and selected-view visualization."""

    stroke_plan_path: str
    visualization_path: str
    stroke_plan: dict[str, JsonValue]

    def as_payload(self) -> dict[str, JsonValue]:
        """Return the workflow-compatible result fragment."""
        return {
            "stroke_plan_path": self.stroke_plan_path,
            "visualization_path": self.visualization_path,
            "stroke_plan": self.stroke_plan,
        }


@dataclass(frozen=True, slots=True)
class _Trajectory:
    trajectory_id: str
    points: tuple[Point, ...]
    closed: bool


def plan_draw_strokes(
    *,
    image_path: str | Path,
    trajectory_plan: Mapping[str, object],
    screenshot_metadata: Mapping[str, object] | None,
    output_dir: str | Path,
    brush_size_ratio: float,
    minimum_brush_size: int,
    maximum_brush_size: int,
    brush_direction: str,
    size_multiplier: float = 1.0,
    brush_strength: float = 1.0,
) -> DrawStrokePlanResult:
    """Convert mask-fitted paths into surface-raycast Draw gestures."""
    image_source = _resolved_file(image_path, label="image_path")
    image = _load_image(image_source)
    trajectories = _parse_trajectories(
        trajectory_plan,
        image_width=image.width,
        image_height=image.height,
    )
    _validate_settings(
        brush_size_ratio=brush_size_ratio,
        minimum_brush_size=minimum_brush_size,
        maximum_brush_size=maximum_brush_size,
        size_multiplier=size_multiplier,
        brush_strength=brush_strength,
        brush_direction=brush_direction,
    )
    mapping = ScreenshotViewportMapping.from_metadata(
        image_width=image.width,
        image_height=image.height,
        metadata=screenshot_metadata,
    )
    base_size, reference_extent = _resolve_brush_size(
        trajectories,
        mapping=mapping,
        ratio=brush_size_ratio,
        minimum=minimum_brush_size,
        maximum=maximum_brush_size,
    )
    brush_size = min(
        maximum_brush_size,
        max(minimum_brush_size, round(base_size * size_multiplier)),
    )
    operator_calls = [
        _operator_call(
            trajectory,
            sequence=index,
            mapping=mapping,
            brush_size=brush_size,
        )
        for index, trajectory in enumerate(trajectories, start=1)
    ]
    element_count = sum(
        len(call["kwargs"]["stroke"])  # type: ignore[index]
        for call in operator_calls
    )
    if element_count > _MAX_STROKE_ELEMENTS:
        raise DrawStrokePlanningError(
            "Draw plan exceeds the maximum stroke element count"
        )

    plan: dict[str, JsonValue] = {
        "format": "blender-sculpt-draw-plan/v1",
        "algorithm": {
            "name": "mask_fitted_polyline_surface_raycast",
            "deterministic": True,
            "one_trajectory_per_mouse_gesture": True,
            "closed_endpoint_policy": "OMIT_DUPLICATE_FINAL_POINT",
            "mouse_speed_pixels_per_second": (
                _MOUSE_SPEED_PIXELS_PER_SECOND
            ),
            "brush_size_policy": (
                "SHORTER_POSITIVE_FITTED_BBOX_EXTENT_RATIO"
            ),
        },
        "coordinate_mapping": mapping.as_payload(),
        "sculpt_settings": {
            "brush_size": brush_size,
            "brush_strength": brush_strength,
            "brush_direction": brush_direction,
        },
        "size_resolution": {
            "reference_extent_pixels": round(reference_extent, 6),
            "brush_size_ratio": brush_size_ratio,
            "base_brush_size": base_size,
            "size_multiplier": size_multiplier,
            "resolved_brush_size": brush_size,
            "minimum_brush_size": minimum_brush_size,
            "maximum_brush_size": maximum_brush_size,
        },
        "summary": {
            "region_count": 1,
            "trajectory_count": len(trajectories),
            "operator_call_count": len(operator_calls),
            "stroke_element_count": element_count,
        },
        "operator_calls": operator_calls,
    }
    directory = _resolved_directory(output_dir)
    plan_path = directory / "draw-stroke-plan.json"
    visualization_path = directory / "draw-trajectory-overlay.png"
    _write_json(plan_path, plan)
    _render_visualization(
        image=image,
        trajectories=trajectories,
        brush_size=brush_size,
        output_path=visualization_path,
    )
    return DrawStrokePlanResult(
        stroke_plan_path=str(plan_path),
        visualization_path=str(visualization_path),
        stroke_plan=plan,
    )


def _parse_trajectories(
    plan: Mapping[str, object],
    *,
    image_width: int,
    image_height: int,
) -> tuple[_Trajectory, ...]:
    supported = {
        "mask-fitted-svg-mouse-trajectories/v1",
        "mask-fitted-text-mouse-trajectories/v1",
    }
    if plan.get("format") not in supported:
        raise DrawStrokePlanningError(
            "trajectory_plan must be a supported mask-fitted format"
        )
    coordinate_system = plan.get("coordinate_system")
    if not isinstance(coordinate_system, Mapping):
        raise DrawStrokePlanningError(
            "trajectory_plan is missing coordinate_system"
        )
    if (
        coordinate_system.get("width") != image_width
        or coordinate_system.get("height") != image_height
        or coordinate_system.get("origin") != "TOP_LEFT"
    ):
        raise DrawStrokePlanningError(
            "trajectory coordinates do not match the selected screenshot"
        )
    raw = plan.get("trajectories")
    if not isinstance(raw, list) or not raw:
        raise DrawStrokePlanningError(
            "trajectory_plan must contain at least one trajectory"
        )
    parsed: list[_Trajectory] = []
    identifiers: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, Mapping):
            raise DrawStrokePlanningError(
                f"trajectory {index} must be an object"
            )
        identifier = item.get("id")
        points_value = item.get("points")
        if not isinstance(identifier, str) or not identifier.strip():
            raise DrawStrokePlanningError(
                f"trajectory {index} has no valid id"
            )
        if identifier in identifiers:
            raise DrawStrokePlanningError("trajectory ids must be unique")
        if not isinstance(points_value, list) or not points_value:
            raise DrawStrokePlanningError(
                f"trajectory {identifier} has no points"
            )
        points: list[Point] = []
        for point in points_value:
            if not isinstance(point, Mapping):
                raise DrawStrokePlanningError("trajectory point is invalid")
            x = point.get("x")
            y = point.get("y")
            if (
                isinstance(x, bool)
                or isinstance(y, bool)
                or not isinstance(x, (int, float))
                or not isinstance(y, (int, float))
                or not math.isfinite(float(x))
                or not math.isfinite(float(y))
                or not 0.0 <= float(x) <= image_width - 1.0
                or not 0.0 <= float(y) <= image_height - 1.0
            ):
                raise DrawStrokePlanningError(
                    "trajectory point lies outside the selected screenshot"
                )
            normalized = (float(x), float(y))
            if not points or math.dist(points[-1], normalized) > 1e-9:
                points.append(normalized)
        if not points:
            raise DrawStrokePlanningError(
                f"trajectory {identifier} contains no usable points"
            )
        closed = bool(item.get("closed", False))
        if (
            closed
            and len(points) > 1
            and math.dist(points[0], points[-1]) <= 1e-9
        ):
            # The brush radius joins endpoints without sculpting closure twice.
            points.pop()
        identifiers.add(identifier)
        parsed.append(
            _Trajectory(
                trajectory_id=identifier,
                points=tuple(points),
                closed=closed,
            )
        )
    return tuple(parsed)


def _resolve_brush_size(
    trajectories: Sequence[_Trajectory],
    *,
    mapping: ScreenshotViewportMapping,
    ratio: float,
    minimum: int,
    maximum: int,
) -> tuple[int, float]:
    region_points = [
        (point[0] / mapping.scale_x, point[1] / mapping.scale_y)
        for trajectory in trajectories
        for point in trajectory.points
    ]
    x_values = [point[0] for point in region_points]
    y_values = [point[1] for point in region_points]
    spans = [
        value
        for value in (
            max(x_values) - min(x_values),
            max(y_values) - min(y_values),
        )
        if value > 1e-6
    ]
    reference_extent = min(spans) if spans else 1.0
    resolved = min(maximum, max(minimum, round(reference_extent * ratio)))
    return resolved, reference_extent


def _operator_call(
    trajectory: _Trajectory,
    *,
    sequence: int,
    mapping: ScreenshotViewportMapping,
    brush_size: int,
) -> dict[str, JsonValue]:
    name = f"draw-{sequence:03d}-{trajectory.trajectory_id}"[:128]
    stroke: list[JsonValue] = []
    elapsed = 0.0
    previous: Point | None = None
    for index, point in enumerate(trajectory.points):
        region_point = (
            min(
                mapping.region_width - 1.0,
                max(0.0, point[0] / mapping.scale_x),
            ),
            min(
                mapping.region_height - 1.0,
                max(0.0, point[1] / mapping.scale_y),
            ),
        )
        if previous is not None:
            elapsed += math.dist(previous, region_point) / (
                _MOUSE_SPEED_PIXELS_PER_SECOND
            )
        mouse = [
            round(region_point[0], 6),
            round(mapping.region_height - 1.0 - region_point[1], 6),
        ]
        stroke.append(
            {
                "name": name,
                "location": [0.0, 0.0, 0.0],
                "mouse": mouse,
                "mouse_event": mouse.copy(),
                "pressure": 1.0,
                "size": float(brush_size),
                "x_tilt": 0.0,
                "y_tilt": 0.0,
                "time": round(elapsed, 6),
                "is_start": index == 0,
            }
        )
        previous = region_point
    return {
        "region_id": 1,
        "path_kind": "draw",
        "path_name": name,
        "closed": trajectory.closed,
        "operator": "bpy.ops.sculpt.brush_stroke",
        "context": {
            "window_index": mapping.window_index,
            "area_index": mapping.area_index,
            "region_type": "WINDOW",
            "coordinate_space": "VIEW_3D_WINDOW_REGION",
        },
        "kwargs": {
            "stroke": stroke,
            "mode": "NORMAL",
            "brush_toggle": "None",
            "pen_flip": False,
            "execution_mode": "PAINT_CURVE",
            "location_mode": "SURFACE_RAYCAST",
            "ignore_background_click": True,
        },
    }


def _render_visualization(
    *,
    image: Image.Image,
    trajectories: Sequence[_Trajectory],
    brush_size: int,
    output_path: Path,
) -> None:
    rendered = image.convert("RGBA")
    overlay = Image.new("RGBA", rendered.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_width = max(3, min(10, round(brush_size * 0.35)))
    point_radius = max(2, min(6, round(line_width * 0.65)))
    for index, trajectory in enumerate(trajectories):
        color = _PALETTE[index % len(_PALETTE)]
        points = list(trajectory.points)
        if len(points) >= 2:
            rendered_points = (
                [*points, points[0]] if trajectory.closed else points
            )
            draw.line(
                rendered_points,
                fill=color,
                width=line_width,
                joint="curve",
            )
        for x, y in points:
            draw.ellipse(
                (
                    x - point_radius,
                    y - point_radius,
                    x + point_radius,
                    y + point_radius,
                ),
                fill=color,
            )
    composed = Image.alpha_composite(rendered, overlay).convert("RGB")
    temporary = output_path.with_name(
        f".{output_path.name}.{uuid4().hex}.tmp"
    )
    try:
        composed.save(temporary, format="PNG")
        os.replace(temporary, output_path)
    except OSError as error:
        raise DrawStrokePlanningError(
            f"cannot write Draw trajectory visualization: {output_path}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _validate_settings(
    *,
    brush_size_ratio: float,
    minimum_brush_size: int,
    maximum_brush_size: int,
    size_multiplier: float,
    brush_strength: float,
    brush_direction: str,
) -> None:
    if (
        not math.isfinite(brush_size_ratio)
        or not 0.001 <= brush_size_ratio <= 1.0
    ):
        raise DrawStrokePlanningError(
            "brush_size_ratio must be between 0.001 and 1"
        )
    if not 1 <= minimum_brush_size <= maximum_brush_size <= 10_000:
        raise DrawStrokePlanningError("Draw brush-size bounds are invalid")
    if not math.isfinite(size_multiplier) or size_multiplier < 1.0:
        raise DrawStrokePlanningError(
            "size_multiplier must be finite and at least 1"
        )
    if not math.isfinite(brush_strength) or not 0.0 <= brush_strength <= 1.0:
        raise DrawStrokePlanningError(
            "brush_strength must be between 0 and 1"
        )
    if brush_direction not in {"ADD", "SUBTRACT"}:
        raise DrawStrokePlanningError(
            "brush_direction must be ADD or SUBTRACT"
        )


def _load_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise DrawStrokePlanningError(f"cannot read image: {path}") from error


def _resolved_file(value: str | Path, *, label: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()
    if not path.is_file():
        raise DrawStrokePlanningError(f"{label} is not a file: {path}")
    return path


def _resolved_directory(value: str | Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise DrawStrokePlanningError(
            f"cannot create Draw output directory: {path}"
        ) from error
    return path


def _write_json(path: Path, payload: Mapping[str, JsonValue]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                dict(payload),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as error:
        raise DrawStrokePlanningError(
            f"cannot write Draw stroke plan: {path}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
