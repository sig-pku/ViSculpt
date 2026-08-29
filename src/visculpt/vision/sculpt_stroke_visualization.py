"""Render Blender Sculpt stroke plans on SAM3 result images."""

from __future__ import annotations

import colorsys
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

type Point = tuple[float, float]
type RgbColor = tuple[int, int, int]

_PATH_COLORS: dict[str, RgbColor] = {
    "boundary": (0, 210, 255),
    "interior": (255, 190, 0),
}
_FALLBACK_COLOR: RgbColor = (170, 120, 255)
_START_COLOR: RgbColor = (40, 230, 90)
_END_COLOR: RgbColor = (255, 70, 55)
_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


class SculptStrokeVisualizationError(ValueError):
    """Raised when a stroke plan cannot be rendered safely."""


@dataclass(frozen=True, slots=True)
class SculptStrokeVisualizationResult:
    """Paths of the two persisted diagnostic renderings."""

    mask_visualization_path: str
    overlay_visualization_path: str


@dataclass(frozen=True, slots=True)
class SculptStrokePolylineVisualizationResult:
    """Paths of the two mouse-trajectory-only renderings."""

    mask_visualization_path: str
    overlay_visualization_path: str


@dataclass(frozen=True, slots=True)
class _ImageMapping:
    source_width: int
    source_height: int
    region_width: int
    region_height: int

    @property
    def scale_x(self) -> float:
        return self.source_width / self.region_width

    @property
    def scale_y(self) -> float:
        return self.source_height / self.region_height

    def mouse_to_image(self, mouse: object) -> Point:
        """Convert bottom-left region coordinates to image pixel centers."""
        values = _required_sequence(mouse, "stroke.mouse_event", length=2)
        mouse_x = _required_number(values[0], "stroke.mouse_event[0]")
        mouse_y = _required_number(values[1], "stroke.mouse_event[1]")
        if not (0.0 <= mouse_x < self.region_width):
            raise SculptStrokeVisualizationError(
                "stroke mouse_event x is outside the target region"
            )
        if not (0.0 <= mouse_y < self.region_height):
            raise SculptStrokeVisualizationError(
                "stroke mouse_event y is outside the target region"
            )

        # Scale around pixel centers to avoid half-pixel offsets on 2x HiDPI.
        image_x = (mouse_x + 0.5) * self.scale_x - 0.5
        logical_top_y = self.region_height - 1.0 - mouse_y
        image_y = (logical_top_y + 0.5) * self.scale_y - 0.5
        return (
            min(max(image_x, 0.0), self.source_width - 1.0),
            min(max(image_y, 0.0), self.source_height - 1.0),
        )


@dataclass(frozen=True, slots=True)
class _RenderedStroke:
    call_index: int
    path_kind: str
    color: RgbColor
    points: tuple[Point, ...]
    pressures: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _PolylineStroke:
    color: RgbColor
    points: tuple[Point, ...]


def render_sculpt_stroke_visualizations(
    *,
    stroke_plan: Mapping[str, object],
    mask_path: str | Path,
    overlay_path: str | Path,
    output_dir: str | Path,
) -> SculptStrokeVisualizationResult:
    """Overlay an operator-ready trajectory on mask and overlay images."""
    resolved_mask = _required_image_path(mask_path, "mask_path")
    resolved_overlay = _required_image_path(overlay_path, "overlay_path")
    target_directory = Path(output_dir).expanduser().resolve()
    if target_directory.exists() and not target_directory.is_dir():
        raise SculptStrokeVisualizationError(
            "output_dir must reference a directory"
        )
    target_directory.mkdir(parents=True, exist_ok=True)

    mapping = _image_mapping(stroke_plan)
    brush_size = _brush_size(stroke_plan)
    strokes = _rendered_strokes(stroke_plan, mapping)
    base_name = _result_base_name(resolved_mask)
    mask_output = target_directory / (
        f"{base_name}-stroke-trajectory-on-mask.png"
    )
    overlay_output = target_directory / (
        f"{base_name}-stroke-trajectory-on-overlay.png"
    )

    _render_image(
        source_path=resolved_mask,
        output_path=mask_output,
        mapping=mapping,
        brush_size=brush_size,
        strokes=strokes,
    )
    _render_image(
        source_path=resolved_overlay,
        output_path=overlay_output,
        mapping=mapping,
        brush_size=brush_size,
        strokes=strokes,
    )
    return SculptStrokeVisualizationResult(
        mask_visualization_path=str(mask_output),
        overlay_visualization_path=str(overlay_output),
    )


def render_sculpt_stroke_polyline_visualizations(
    *,
    stroke_plan: Mapping[str, object],
    mask_path: str | Path,
    overlay_path: str | Path,
    output_dir: str | Path,
) -> SculptStrokePolylineVisualizationResult:
    """Render one independently colored polyline per mouse-down gesture."""
    resolved_mask = _required_image_path(mask_path, "mask_path")
    resolved_overlay = _required_image_path(overlay_path, "overlay_path")
    target_directory = Path(output_dir).expanduser().resolve()
    if target_directory.exists() and not target_directory.is_dir():
        raise SculptStrokeVisualizationError(
            "output_dir must reference a directory"
        )
    target_directory.mkdir(parents=True, exist_ok=True)

    mapping = _image_mapping(stroke_plan)
    strokes = _polyline_strokes(stroke_plan, mapping)
    base_name = _result_base_name(resolved_mask)
    mask_output = target_directory / (
        f"{base_name}-mouse-trajectories-on-mask.png"
    )
    overlay_output = target_directory / (
        f"{base_name}-mouse-trajectories-on-overlay.png"
    )
    _render_polyline_image(
        source_path=resolved_mask,
        output_path=mask_output,
        mapping=mapping,
        strokes=strokes,
    )
    _render_polyline_image(
        source_path=resolved_overlay,
        output_path=overlay_output,
        mapping=mapping,
        strokes=strokes,
    )
    return SculptStrokePolylineVisualizationResult(
        mask_visualization_path=str(mask_output),
        overlay_visualization_path=str(overlay_output),
    )


def _render_image(
    *,
    source_path: Path,
    output_path: Path,
    mapping: _ImageMapping,
    brush_size: float,
    strokes: tuple[_RenderedStroke, ...],
) -> None:
    base = _load_source_image(source_path, mapping)

    footprint_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    path_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    footprint_draw = ImageDraw.Draw(footprint_layer, "RGBA")
    path_draw = ImageDraw.Draw(path_layer, "RGBA")
    radius_x = brush_size * 0.5 * mapping.scale_x
    radius_y = brush_size * 0.5 * mapping.scale_y
    display_scale = math.sqrt(mapping.scale_x * mapping.scale_y)
    line_width = max(2, round(2.5 * display_scale))
    event_radius_base = max(1.5, 1.6 * display_scale)
    endpoint_radius = max(4.0, 4.5 * display_scale)

    for stroke in strokes:
        for point, pressure in zip(
            stroke.points,
            stroke.pressures,
            strict=True,
        ):
            footprint_draw.ellipse(
                (
                    point[0] - radius_x,
                    point[1] - radius_y,
                    point[0] + radius_x,
                    point[1] + radius_y,
                ),
                fill=(*stroke.color, round(10 + 30 * pressure)),
            )

        if len(stroke.points) > 1:
            path_draw.line(
                stroke.points,
                fill=(*stroke.color, 235),
                width=line_width,
                joint="curve",
            )
        for point, pressure in zip(
            stroke.points,
            stroke.pressures,
            strict=True,
        ):
            event_radius = event_radius_base * (0.65 + pressure * 0.7)
            path_draw.ellipse(
                (
                    point[0] - event_radius,
                    point[1] - event_radius,
                    point[0] + event_radius,
                    point[1] + event_radius,
                ),
                fill=(*stroke.color, round(125 + 130 * pressure)),
            )

        _draw_endpoint(
            path_draw,
            point=stroke.points[0],
            radius=endpoint_radius,
            color=_START_COLOR,
        )
        _draw_endpoint(
            path_draw,
            point=stroke.points[-1],
            radius=endpoint_radius,
            color=_END_COLOR,
        )
        _draw_call_index(
            path_draw,
            point=stroke.points[0],
            call_index=stroke.call_index,
            scale=display_scale,
        )

    _draw_legend(path_draw, width=base.width, scale=display_scale)
    rendered = Image.alpha_composite(base, footprint_layer)
    rendered = Image.alpha_composite(rendered, path_layer)
    rendered.convert("RGB").save(output_path, format="PNG")


def _render_polyline_image(
    *,
    source_path: Path,
    output_path: Path,
    mapping: _ImageMapping,
    strokes: tuple[_PolylineStroke, ...],
) -> None:
    base = _load_source_image(source_path, mapping)
    path_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(path_layer, "RGBA")
    display_scale = math.sqrt(mapping.scale_x * mapping.scale_y)
    line_width = max(2, round(4.0 * display_scale))
    point_radius = max(
        round(4.0 * display_scale),
        min(
            round(12.0 * display_scale),
            round(min(base.size) * 0.007),
        ),
    )
    for stroke in strokes:
        points = stroke.points
        if len(points) > 1:
            draw.line(
                points,
                fill=(*stroke.color, 255),
                width=line_width,
                joint="curve",
            )
        # Draw every sample as a solid dot so one-point gestures stay visible.
        for point in points:
            draw.ellipse(
                (
                    point[0] - point_radius,
                    point[1] - point_radius,
                    point[0] + point_radius,
                    point[1] + point_radius,
                ),
                fill=(*stroke.color, 255),
            )
    rendered = Image.alpha_composite(base, path_layer)
    rendered.convert("RGB").save(output_path, format="PNG")


def _load_source_image(
    source_path: Path,
    mapping: _ImageMapping,
) -> Image.Image:
    try:
        with Image.open(source_path) as source:
            source.load()
            if source.size != (
                mapping.source_width,
                mapping.source_height,
            ):
                raise SculptStrokeVisualizationError(
                    f"{source_path.name} dimensions do not match the "
                    "stroke plan source image"
                )
            return source.convert("RGBA")
    except (OSError, UnidentifiedImageError) as error:
        raise SculptStrokeVisualizationError(
            f"cannot read visualization source image: {source_path}"
        ) from error


def _draw_endpoint(
    draw: ImageDraw.ImageDraw,
    *,
    point: Point,
    radius: float,
    color: RgbColor,
) -> None:
    draw.ellipse(
        (
            point[0] - radius,
            point[1] - radius,
            point[0] + radius,
            point[1] + radius,
        ),
        fill=(*color, 245),
        outline=(10, 10, 10, 255),
        width=max(1, round(radius * 0.3)),
    )


def _draw_call_index(
    draw: ImageDraw.ImageDraw,
    *,
    point: Point,
    call_index: int,
    scale: float,
) -> None:
    font_size = max(9, round(11 * scale))
    font = ImageFont.load_default(size=font_size)
    position = (point[0] + 5 * scale, point[1] + 3 * scale)
    draw.text(
        position,
        str(call_index),
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=max(1, round(scale)),
        stroke_fill=(0, 0, 0, 230),
    )


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    scale: float,
) -> None:
    if width < round(520 * scale):
        return
    font_size = max(9, round(12 * scale))
    font = ImageFont.load_default(size=font_size)
    padding = round(8 * scale)
    item_gap = round(18 * scale)
    labels = (
        ("boundary", _PATH_COLORS["boundary"]),
        ("interior", _PATH_COLORS["interior"]),
        ("start", _START_COLOR),
        ("end", _END_COLOR),
    )
    text_widths = [
        draw.textbbox((0, 0), label, font=font)[2] for label, _ in labels
    ]
    box_width = (
        padding * 2
        + sum(text_widths)
        + len(labels) * round(13 * scale)
        + (len(labels) - 1) * item_gap
    )
    box_height = round(30 * scale)
    draw.rounded_rectangle(
        (padding, padding, padding + box_width, padding + box_height),
        radius=round(5 * scale),
        fill=(0, 0, 0, 170),
    )
    cursor_x = padding * 2
    center_y = padding + box_height / 2
    swatch_radius = max(3, round(4 * scale))
    for (label, color), text_width in zip(
        labels,
        text_widths,
        strict=True,
    ):
        draw.ellipse(
            (
                cursor_x,
                center_y - swatch_radius,
                cursor_x + swatch_radius * 2,
                center_y + swatch_radius,
            ),
            fill=(*color, 255),
        )
        cursor_x += swatch_radius * 2 + round(5 * scale)
        draw.text(
            (cursor_x, center_y - font_size / 2),
            label,
            font=font,
            fill=(255, 255, 255, 255),
        )
        cursor_x += text_width + item_gap


def _image_mapping(stroke_plan: Mapping[str, object]) -> _ImageMapping:
    coordinate_mapping = _required_mapping(
        stroke_plan.get("coordinate_mapping"),
        "stroke_plan.coordinate_mapping",
    )
    source = _required_mapping(
        coordinate_mapping.get("source_image"),
        "stroke_plan.coordinate_mapping.source_image",
    )
    target = _required_mapping(
        coordinate_mapping.get("target_region"),
        "stroke_plan.coordinate_mapping.target_region",
    )
    return _ImageMapping(
        source_width=_required_positive_int(
            source.get("width"),
            "source_image.width",
        ),
        source_height=_required_positive_int(
            source.get("height"),
            "source_image.height",
        ),
        region_width=_required_positive_int(
            target.get("width"),
            "target_region.width",
        ),
        region_height=_required_positive_int(
            target.get("height"),
            "target_region.height",
        ),
    )


def _brush_size(stroke_plan: Mapping[str, object]) -> float:
    settings = _required_mapping(
        stroke_plan.get("sculpt_settings"),
        "stroke_plan.sculpt_settings",
    )
    brush_size = _required_number(
        settings.get("brush_size"),
        "stroke_plan.sculpt_settings.brush_size",
    )
    if brush_size <= 0.0:
        raise SculptStrokeVisualizationError("brush_size must be positive")
    return brush_size


def _rendered_strokes(
    stroke_plan: Mapping[str, object],
    mapping: _ImageMapping,
) -> tuple[_RenderedStroke, ...]:
    calls = _required_sequence(
        stroke_plan.get("operator_calls"),
        "stroke_plan.operator_calls",
    )
    if not calls:
        raise SculptStrokeVisualizationError(
            "stroke_plan.operator_calls must not be empty"
        )
    rendered: list[_RenderedStroke] = []
    for call_index, raw_call in enumerate(calls, start=1):
        call = _required_mapping(
            raw_call,
            f"stroke_plan.operator_calls[{call_index - 1}]",
        )
        path_kind = call.get("path_kind")
        if not isinstance(path_kind, str) or not path_kind:
            raise SculptStrokeVisualizationError(
                "operator call path_kind must be a non-empty string"
            )
        kwargs = _required_mapping(call.get("kwargs"), "operator kwargs")
        elements = _required_sequence(kwargs.get("stroke"), "kwargs.stroke")
        if not elements:
            raise SculptStrokeVisualizationError(
                "operator call stroke must not be empty"
            )
        points: list[Point] = []
        pressures: list[float] = []
        for element_index, raw_element in enumerate(elements):
            element = _required_mapping(
                raw_element,
                f"kwargs.stroke[{element_index}]",
            )
            points.append(mapping.mouse_to_image(element.get("mouse_event")))
            pressure = _required_number(
                element.get("pressure"),
                f"kwargs.stroke[{element_index}].pressure",
            )
            if not 0.0 <= pressure <= 1.0:
                raise SculptStrokeVisualizationError(
                    "stroke pressure must be in the range 0 to 1"
                )
            pressures.append(pressure)
        rendered.append(
            _RenderedStroke(
                call_index=call_index,
                path_kind=path_kind,
                color=_PATH_COLORS.get(path_kind, _FALLBACK_COLOR),
                points=tuple(points),
                pressures=tuple(pressures),
            )
        )
    return tuple(rendered)


def _polyline_strokes(
    stroke_plan: Mapping[str, object],
    mapping: _ImageMapping,
) -> tuple[_PolylineStroke, ...]:
    """Read only gesture boundaries and mouse coordinates from a plan."""
    calls = _required_sequence(
        stroke_plan.get("operator_calls"),
        "stroke_plan.operator_calls",
    )
    if not calls:
        raise SculptStrokeVisualizationError(
            "stroke_plan.operator_calls must not be empty"
        )
    rendered: list[_PolylineStroke] = []
    for call_index, raw_call in enumerate(calls):
        call = _required_mapping(
            raw_call,
            f"stroke_plan.operator_calls[{call_index}]",
        )
        kwargs = _required_mapping(call.get("kwargs"), "operator kwargs")
        elements = _required_sequence(kwargs.get("stroke"), "kwargs.stroke")
        if not elements:
            raise SculptStrokeVisualizationError(
                "operator call stroke must not be empty"
            )
        points: list[Point] = []
        for element_index, raw_element in enumerate(elements):
            element = _required_mapping(
                raw_element,
                f"kwargs.stroke[{element_index}]",
            )
            points.append(mapping.mouse_to_image(element.get("mouse_event")))
        rendered.append(
            _PolylineStroke(
                color=_polyline_color(call_index),
                points=tuple(points),
            )
        )
    return tuple(rendered)


def _polyline_color(index: int) -> RgbColor:
    """Assign a deterministic, visually separated color to each gesture."""
    hue = (0.37 + index * _GOLDEN_RATIO_CONJUGATE) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 1.0)
    return (
        round(red * 255),
        round(green * 255),
        round(blue * 255),
    )


def _required_image_path(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise SculptStrokeVisualizationError(
            f"{label} must reference an existing image file"
        )
    return path


def _result_base_name(mask_path: Path) -> str:
    stem = mask_path.stem
    for suffix in ("-mask", "_mask"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _required_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SculptStrokeVisualizationError(f"{label} must be an object")
    return value


def _required_sequence(
    value: object,
    label: str,
    *,
    length: int | None = None,
) -> list[object] | tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise SculptStrokeVisualizationError(f"{label} must be an array")
    if length is not None and len(value) != length:
        raise SculptStrokeVisualizationError(
            f"{label} must contain exactly {length} values"
        )
    return value


def _required_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SculptStrokeVisualizationError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise SculptStrokeVisualizationError(f"{label} must be finite")
    return result


def _required_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SculptStrokeVisualizationError(
            f"{label} must be a positive integer"
        )
    return value
