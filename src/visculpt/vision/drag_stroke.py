"""Deterministic straight-line Drag planning for Blender Sculpt Mode."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw, UnidentifiedImageError

from visculpt.bridge import JsonValue

from .sculpt_stroke import ScreenshotViewportMapping

_MOUSE_SPEED_PIXELS_PER_SECOND = 600.0


class DragStrokePlanningError(ValueError):
    """Raised when a requested 2D Drag cannot produce a safe gesture."""


@dataclass(frozen=True, slots=True)
class DragStrokePlanResult:
    """Persisted straight Drag plan and its diagnostic visualization."""

    stroke_plan_path: str
    visualization_path: str
    stroke_plan: dict[str, JsonValue]

    def as_payload(self) -> dict[str, JsonValue]:
        """Return the result fragment used by the workflow."""
        return {
            "stroke_plan_path": self.stroke_plan_path,
            "visualization_path": self.visualization_path,
            "stroke_plan": self.stroke_plan,
        }


def plan_drag_stroke(
    *,
    image_path: str | Path,
    start_x: int,
    start_y: int,
    direction_x: float,
    direction_y: float,
    distance_pixels: float,
    distance_multiplier: float,
    brush_size: int,
    brush_strength: float,
    brush_direction: str | None,
    screenshot_metadata: dict[str, object],
    output_dir: str | Path,
    minimum_distance_pixels: float,
    maximum_distance_ratio: float,
    stroke_spacing_pixels: float,
) -> DragStrokePlanResult:
    """Create one uniformly sampled straight Sculpt operator call."""
    source = _resolve_image(image_path)
    image = _load_image(source)
    _validate_parameters(
        image=image,
        start_x=start_x,
        start_y=start_y,
        direction_x=direction_x,
        direction_y=direction_y,
        distance_pixels=distance_pixels,
        distance_multiplier=distance_multiplier,
        brush_size=brush_size,
        brush_strength=brush_strength,
        minimum_distance_pixels=minimum_distance_pixels,
        maximum_distance_ratio=maximum_distance_ratio,
        stroke_spacing_pixels=stroke_spacing_pixels,
    )
    mapping = ScreenshotViewportMapping.from_metadata(
        image_width=image.width,
        image_height=image.height,
        metadata=screenshot_metadata,
    )
    magnitude = math.hypot(direction_x, direction_y)
    unit_x = direction_x / magnitude
    unit_y = direction_y / magnitude
    maximum_distance = math.hypot(image.width, image.height) * (
        maximum_distance_ratio
    )
    requested_distance = min(
        maximum_distance,
        max(minimum_distance_pixels, distance_pixels * distance_multiplier),
    )
    boundary_distance = _distance_to_image_boundary(
        x=float(start_x),
        y=float(start_y),
        unit_x=unit_x,
        unit_y=unit_y,
        width=image.width,
        height=image.height,
    )
    applied_distance = min(requested_distance, boundary_distance)
    if applied_distance < 1.0:
        raise DragStrokePlanningError(
            "drag direction leaves the screenshot immediately from its "
            "localized start point"
        )
    end_x = float(start_x) + unit_x * applied_distance
    end_y = float(start_y) + unit_y * applied_distance
    sample_count = max(
        2,
        int(math.ceil(applied_distance / stroke_spacing_pixels)) + 1,
    )
    screenshot_points = [
        (
            float(start_x) + (end_x - float(start_x)) * index / (
                sample_count - 1
            ),
            float(start_y) + (end_y - float(start_y)) * index / (
                sample_count - 1
            ),
        )
        for index in range(sample_count)
    ]
    stroke = _operator_stroke(
        screenshot_points,
        mapping=mapping,
        brush_size=brush_size,
    )
    plan: dict[str, JsonValue] = {
        "format": "blender-sculpt-drag-plan/v2",
        "algorithm": {
            "name": "normalized_clipped_uniform_line",
            "deterministic": True,
            "coordinate_origin": "TOP_LEFT",
            "stroke_spacing_pixels": stroke_spacing_pixels,
            "mouse_speed_pixels_per_second": (
                _MOUSE_SPEED_PIXELS_PER_SECOND
            ),
        },
        "coordinate_mapping": mapping.as_payload(),
        "sculpt_settings": {
            "brush_size": brush_size,
            "brush_strength": brush_strength,
            "brush_direction": brush_direction,
        },
        "gesture": {
            "start": {"x": start_x, "y": start_y},
            "end": {"x": round(end_x, 6), "y": round(end_y, 6)},
            "unit_direction": {
                "x": round(unit_x, 9),
                "y": round(unit_y, 9),
            },
            "vlm_distance_pixels": distance_pixels,
            "distance_multiplier": distance_multiplier,
            "requested_distance_pixels": round(requested_distance, 6),
            "applied_distance_pixels": round(applied_distance, 6),
            "clipped_to_viewport": (
                applied_distance + 1e-6 < requested_distance
            ),
            "sample_count": sample_count,
        },
        "summary": {
            "region_count": 1,
            "operator_call_count": 1,
            "stroke_element_count": sample_count,
        },
        "operator_calls": [
            {
                "region_id": 1,
                "path_kind": "drag",
                "path_name": "drag-main-001",
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
                    "execution_mode": "ANCHORED_DRAG",
                    "location_mode": "ANCHORED_DRAG",
                    "ignore_background_click": True,
                },
            }
        ],
    }
    directory = _resolve_output_dir(output_dir)
    plan_path = directory / "drag-stroke-plan.json"
    visualization_path = directory / "drag-trajectory-overlay.png"
    _write_json(plan_path, plan)
    _render_visualization(
        image,
        points=screenshot_points,
        output_path=visualization_path,
    )
    return DragStrokePlanResult(
        stroke_plan_path=str(plan_path),
        visualization_path=str(visualization_path),
        stroke_plan=plan,
    )


def _operator_stroke(
    points: list[tuple[float, float]],
    *,
    mapping: ScreenshotViewportMapping,
    brush_size: int,
) -> list[JsonValue]:
    elements: list[JsonValue] = []
    elapsed = 0.0
    previous: tuple[float, float] | None = None
    for index, point in enumerate(points):
        if previous is not None:
            elapsed += math.dist(previous, point) / (
                _MOUSE_SPEED_PIXELS_PER_SECOND
            )
        region_x = min(
            mapping.region_width - 1.0,
            max(0.0, point[0] / mapping.scale_x),
        )
        region_y_from_top = min(
            mapping.region_height - 1.0,
            max(0.0, point[1] / mapping.scale_y),
        )
        mouse = [
            round(region_x, 6),
            round(mapping.region_height - 1.0 - region_y_from_top, 6),
        ]
        elements.append(
            {
                "name": "drag-main-001",
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
        previous = point
    return elements


def _distance_to_image_boundary(
    *,
    x: float,
    y: float,
    unit_x: float,
    unit_y: float,
    width: int,
    height: int,
) -> float:
    candidates: list[float] = []
    if unit_x > 1e-12:
        candidates.append((width - 1.0 - x) / unit_x)
    elif unit_x < -1e-12:
        candidates.append(x / -unit_x)
    if unit_y > 1e-12:
        candidates.append((height - 1.0 - y) / unit_y)
    elif unit_y < -1e-12:
        candidates.append(y / -unit_y)
    positive = [value for value in candidates if value >= 0.0]
    if not positive:
        raise DragStrokePlanningError("drag direction has no image extent")
    return min(positive)


def _validate_parameters(
    *,
    image: Image.Image,
    start_x: int,
    start_y: int,
    direction_x: float,
    direction_y: float,
    distance_pixels: float,
    distance_multiplier: float,
    brush_size: int,
    brush_strength: float,
    minimum_distance_pixels: float,
    maximum_distance_ratio: float,
    stroke_spacing_pixels: float,
) -> None:
    values = (
        direction_x,
        direction_y,
        distance_pixels,
        distance_multiplier,
        brush_strength,
        minimum_distance_pixels,
        maximum_distance_ratio,
        stroke_spacing_pixels,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise DragStrokePlanningError("drag parameters must be finite")
    if not 0 <= start_x < image.width or not 0 <= start_y < image.height:
        raise DragStrokePlanningError("drag start is outside the screenshot")
    if math.hypot(direction_x, direction_y) <= 1e-9:
        raise DragStrokePlanningError("drag direction must be nonzero")
    if distance_pixels <= 0.0 or distance_multiplier <= 0.0:
        raise DragStrokePlanningError("drag distance must be positive")
    if isinstance(brush_size, bool) or not 1 <= brush_size <= 10_000:
        raise DragStrokePlanningError("brush_size must be between 1 and 10000")
    if not 0.0 <= brush_strength <= 1.0:
        raise DragStrokePlanningError("brush_strength must be between 0 and 1")
    if minimum_distance_pixels < 1.0:
        raise DragStrokePlanningError("minimum drag distance must be positive")
    if not 0.0 < maximum_distance_ratio <= 1.0:
        raise DragStrokePlanningError("maximum distance ratio is invalid")
    if stroke_spacing_pixels <= 0.0:
        raise DragStrokePlanningError("stroke spacing must be positive")


def _resolve_image(value: str | Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()
    if not path.is_file():
        raise DragStrokePlanningError(f"drag screenshot is not a file: {path}")
    return path


def _load_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise DragStrokePlanningError(
            f"cannot read drag screenshot: {path}"
        ) from error


def _resolve_output_dir(value: str | Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise DragStrokePlanningError(
            f"cannot create drag output directory: {path}"
        ) from error
    return path


def _render_visualization(
    image: Image.Image,
    *,
    points: list[tuple[float, float]],
    output_path: Path,
) -> None:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    start_x, start_y = points[0]
    end_x, end_y = points[-1]
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    distance = math.hypot(delta_x, delta_y)
    unit_x = delta_x / distance
    unit_y = delta_y / distance
    perpendicular_x = -unit_y
    perpendicular_y = unit_x

    visual_scale = min(image.size)
    line_width = max(4, round(visual_scale * 0.006))
    start_radius = max(6, round(visual_scale * 0.009))
    desired_head_length = max(14, round(visual_scale * 0.026))
    head_length = max(2.0, min(float(desired_head_length), distance * 0.42))
    desired_half_width = max(8, round(visual_scale * 0.014))
    head_half_width = max(
        2.0,
        min(float(desired_half_width), head_length * 0.68),
    )
    base_x = end_x - unit_x * head_length
    base_y = end_y - unit_y * head_length
    arrow_head = [
        (round(end_x), round(end_y)),
        (
            round(base_x + perpendicular_x * head_half_width),
            round(base_y + perpendicular_y * head_half_width),
        ),
        (
            round(base_x - perpendicular_x * head_half_width),
            round(base_y - perpendicular_y * head_half_width),
        ),
    ]
    start = (round(start_x), round(start_y))
    end = (round(end_x), round(end_y))
    outline_color = (10, 18, 28)
    arrow_color = (36, 183, 255)
    start_color = (47, 214, 116)

    # A dark outline keeps the arrow legible on light and dark backgrounds.
    draw.line((start, end), fill=outline_color, width=line_width + 4)
    draw.line((start, end), fill=arrow_color, width=line_width)
    draw.polygon(arrow_head, fill=outline_color)
    inset_head_length = max(1.0, head_length - 2.0)
    inset_half_width = max(1.0, head_half_width - 2.0)
    inset_base_x = end_x - unit_x * inset_head_length
    inset_base_y = end_y - unit_y * inset_head_length
    draw.polygon(
        [
            end,
            (
                round(inset_base_x + perpendicular_x * inset_half_width),
                round(inset_base_y + perpendicular_y * inset_half_width),
            ),
            (
                round(inset_base_x - perpendicular_x * inset_half_width),
                round(inset_base_y - perpendicular_y * inset_half_width),
            ),
        ],
        fill=arrow_color,
    )
    draw.ellipse(
        (
            start[0] - start_radius - 2,
            start[1] - start_radius - 2,
            start[0] + start_radius + 2,
            start[1] + start_radius + 2,
        ),
        fill=outline_color,
    )
    draw.ellipse(
        (
            start[0] - start_radius,
            start[1] - start_radius,
            start[0] + start_radius,
            start[1] + start_radius,
        ),
        fill=start_color,
    )
    temporary = output_path.with_name(
        f".{output_path.name}.{uuid4().hex}.tmp"
    )
    try:
        canvas.save(temporary, format="PNG")
        os.replace(temporary, output_path)
    except OSError as error:
        raise DragStrokePlanningError(
            f"cannot write drag visualization: {output_path}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: dict[str, JsonValue]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError as error:
        raise DragStrokePlanningError(
            f"cannot write drag stroke plan: {path}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
