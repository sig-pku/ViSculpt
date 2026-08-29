"""Deterministic screen-space stroke planning for Blender Sculpt Mode."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from visculpt.bridge import JsonValue

type MaskArray = NDArray[np.uint8]
type Point = tuple[float, float]

_MASK_THRESHOLD = 128
_BRUSH_RADIUS_FACTOR = 0.5
_EFFECTIVE_RADIUS_FACTOR = 0.65
_LANE_PITCH_FACTOR = 2.00
_EVENT_STEP_FACTOR = 0.24
_MOUSE_SPEED_PIXELS_PER_SECOND = 600.0
_MAX_STROKE_ELEMENTS = 100_000


class SculptStrokePlanningError(ValueError):
    """Raised when a SAM3 mask cannot produce a safe stroke plan."""


@dataclass(frozen=True, slots=True)
class ScreenshotViewportMapping:
    """Map a screenshot mask to a Blender VIEW_3D WINDOW region."""

    image_width: int
    image_height: int
    region_width: int
    region_height: int
    scale_x: float
    scale_y: float
    window_index: int | None
    area_index: int | None
    source: str

    @classmethod
    def from_metadata(
        cls,
        *,
        image_width: int,
        image_height: int,
        metadata: Mapping[str, object] | None,
    ) -> ScreenshotViewportMapping:
        """Validate optional get_screenshot metadata."""
        if metadata is None:
            return cls(
                image_width=image_width,
                image_height=image_height,
                region_width=image_width,
                region_height=image_height,
                scale_x=1.0,
                scale_y=1.0,
                window_index=None,
                area_index=None,
                source="unit_scale_assumption",
            )

        metadata_width = _required_positive_int(metadata, "width")
        metadata_height = _required_positive_int(metadata, "height")
        if (metadata_width, metadata_height) != (
            image_width,
            image_height,
        ):
            raise SculptStrokePlanningError(
                "screenshot_metadata dimensions do not match the SAM3 mask"
            )

        region = _required_mapping(metadata, "region")
        coordinate_scale = _required_mapping(metadata, "coordinate_scale")
        region_width = _required_positive_int(region, "width")
        region_height = _required_positive_int(region, "height")
        scale_x = _required_positive_float(coordinate_scale, "x")
        scale_y = _required_positive_float(coordinate_scale, "y")

        # Screenshot crops may round by one pixel; reject only clear mismatches.
        expected_width = region_width * scale_x
        expected_height = region_height * scale_y
        width_tolerance = max(2.0, image_width * 0.01)
        height_tolerance = max(2.0, image_height * 0.01)
        if abs(expected_width - image_width) > width_tolerance:
            raise SculptStrokePlanningError(
                "screenshot_metadata coordinate_scale.x is inconsistent"
            )
        if abs(expected_height - image_height) > height_tolerance:
            raise SculptStrokePlanningError(
                "screenshot_metadata coordinate_scale.y is inconsistent"
            )

        return cls(
            image_width=image_width,
            image_height=image_height,
            region_width=region_width,
            region_height=region_height,
            scale_x=scale_x,
            scale_y=scale_y,
            window_index=_optional_nonnegative_int(metadata, "window_index"),
            area_index=_optional_nonnegative_int(metadata, "area_index"),
            source="get_blender_screenshot",
        )

    def as_payload(self) -> dict[str, JsonValue]:
        """Return the coordinate contract used by Blender operator calls."""
        return {
            "source": self.source,
            "source_image": {
                "width": self.image_width,
                "height": self.image_height,
                "origin": "TOP_LEFT",
            },
            "target_region": {
                "width": self.region_width,
                "height": self.region_height,
                "origin": "BOTTOM_LEFT",
                "coordinate_space": "VIEW_3D_WINDOW_REGION",
                "window_index": self.window_index,
                "area_index": self.area_index,
            },
            "coordinate_scale": {
                "x": self.scale_x,
                "y": self.scale_y,
            },
            "transform": (
                "resize mask to region dimensions; "
                "mouse=(x, region_height-1-y)"
            ),
        }


@dataclass(slots=True)
class _PlannedPath:
    region_id: int
    kind: str
    sequence: int
    points: list[Point]
    closed: bool = False
    pressures: list[float] = field(default_factory=list)

    @property
    def name(self) -> str:
        return (
            f"region-{self.region_id:03d}-{self.kind}-"
            f"{self.sequence:03d}"
        )


@dataclass(frozen=True, slots=True)
class SculptStrokePlanResult:
    """Persisted Blender stroke-planning result."""

    stroke_plan_path: str
    stroke_plan: dict[str, JsonValue]

    def as_payload(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible Tool payload fragment."""
        return {
            "stroke_plan_path": self.stroke_plan_path,
            "stroke_plan": self.stroke_plan,
        }


def plan_sculpt_strokes(
    *,
    cleaned_mask_path: str | Path,
    sculpt_brush: str,
    brush_size: int,
    brush_strength: float,
    brush_direction: str | None,
    screenshot_metadata: Mapping[str, object] | None,
    output_dir: str | Path | None,
) -> SculptStrokePlanResult:
    """Create Blender brush_stroke calls from a cleaned semantic mask."""
    _validate_sculpt_parameters(
        sculpt_brush=sculpt_brush,
        brush_size=brush_size,
        brush_strength=brush_strength,
        brush_direction=brush_direction,
    )
    resolved_mask = _resolve_mask_path(cleaned_mask_path)
    grayscale = _load_grayscale_mask(resolved_mask)
    image_height, image_width = grayscale.shape
    mapping = ScreenshotViewportMapping.from_metadata(
        image_width=image_width,
        image_height=image_height,
        metadata=screenshot_metadata,
    )

    try:
        logical_mask = _resize_and_threshold(grayscale, mapping)
    except cv2.error as error:
        raise SculptStrokePlanningError(
            f"OpenCV mask preparation failed: {error}"
        ) from error
    if not np.any(logical_mask):
        raise SculptStrokePlanningError(
            "cleaned SAM3 mask has no foreground"
        )

    try:
        plan = _build_plan(
            logical_mask,
            mapping=mapping,
            sculpt_brush=sculpt_brush,
            brush_size=brush_size,
            brush_strength=brush_strength,
            brush_direction=brush_direction,
        )
    except cv2.error as error:
        raise SculptStrokePlanningError(
            f"OpenCV stroke planning failed: {error}"
        ) from error
    target_directory = _resolve_plan_output_dir(
        output_dir=output_dir,
        mask_path=resolved_mask,
    )
    stroke_plan_path = _output_path(
        mask_path=resolved_mask,
        output_dir=target_directory,
    )
    _write_json(stroke_plan_path, plan)
    return SculptStrokePlanResult(
        stroke_plan_path=str(stroke_plan_path),
        stroke_plan=plan,
    )


def _build_plan(
    mask: MaskArray,
    *,
    mapping: ScreenshotViewportMapping,
    sculpt_brush: str,
    brush_size: int,
    brush_strength: float,
    brush_direction: str | None,
) -> dict[str, JsonValue]:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    components = [
        (label, int(stats[label, cv2.CC_STAT_AREA]))
        for label in range(1, component_count)
    ]
    components.sort(key=lambda item: (-item[1], item[0]))

    # Blender View Brush Size is a diameter; the planner uses true radius.
    brush_radius = max(0.5, brush_size * _BRUSH_RADIUS_FACTOR)
    effective_radius = (
        brush_radius * _EFFECTIVE_RADIUS_FACTOR * brush_strength
    )
    lane_pitch = max(
        1.0,
        brush_radius * _LANE_PITCH_FACTOR * brush_strength,
    )
    event_step = max(1.0, brush_radius * _EVENT_STEP_FACTOR)
    all_paths: list[_PlannedPath] = []
    region_payloads: list[JsonValue] = []

    for region_id, (source_label, area) in enumerate(components, start=1):
        component = np.where(labels == source_label, 255, 0).astype(np.uint8)
        paths, region_payload = _plan_component(
            component,
            region_id=region_id,
            source_label=source_label,
            area=area,
            effective_radius=effective_radius,
            lane_pitch=lane_pitch,
            event_step=event_step,
        )
        all_paths.extend(paths)
        region_payloads.append(region_payload)

    element_count = sum(len(path.points) for path in all_paths)
    if element_count > _MAX_STROKE_ELEMENTS:
        raise SculptStrokePlanningError(
            "generated stroke plan exceeds the 100000-element safety limit"
        )

    _assign_full_pressure(all_paths)
    operator_calls = [
        _operator_call(
            path,
            mapping=mapping,
            brush_size=brush_size,
            brush_direction=brush_direction,
        )
        for path in all_paths
    ]

    return {
        "format": "blender-sculpt-stroke-plan/v3",
        "algorithm": {
            "name": "strength_scaled_safe_center_boundary_pca_raster",
            "deterministic": True,
            "connectivity": 8,
            "mask_threshold": _MASK_THRESHOLD,
            "brush_size_semantics": "diameter_pixels",
            "brush_radius_factor": _BRUSH_RADIUS_FACTOR,
            "effective_coverage_radius_factor": _EFFECTIVE_RADIUS_FACTOR,
            "effective_coverage_radius_formula": (
                "brush_radius * 0.65 * brush_strength"
            ),
            "effective_coverage_radius_pixels": round(
                effective_radius,
                6,
            ),
            "lane_pitch_factor": _LANE_PITCH_FACTOR,
            "lane_pitch_formula": (
                "max(1.0, brush_radius * 2.00 * brush_strength)"
            ),
            "lane_pitch_pixels": round(lane_pitch, 6),
            "event_step_factor": _EVENT_STEP_FACTOR,
            "pressure_model": "constant full pressure",
        },
        "coordinate_mapping": mapping.as_payload(),
        "sculpt_settings": {
            "sculpt_brush": sculpt_brush,
            "brush_size": brush_size,
            "brush_strength": brush_strength,
            "brush_direction": brush_direction,
        },
        "execution_requirements": {
            "context": "SCULPT mode in the mapped VIEW_3D WINDOW region",
            "radius_unit": "VIEW",
            "brush_size_semantics": "diameter_pixels",
            "brush_radius_pixels": round(brush_radius, 6),
            "stroke_method": "SPACE",
            "event_pressure": 1.0,
            "event_spacing_is_precomputed": True,
            "operator_exec_should_not_interpolate": True,
            "note": (
                "The plan uses full event pressure and does not require the "
                "RPC server to alter pressure, falloff, or spacing settings."
            ),
        },
        "summary": {
            "region_count": len(components),
            "operator_call_count": len(operator_calls),
            "stroke_element_count": element_count,
            "pressure_min": 1.0,
            "pressure_max": 1.0,
        },
        "regions": region_payloads,
        "operator_calls": operator_calls,
    }


def _plan_component(
    component: MaskArray,
    *,
    region_id: int,
    source_label: int,
    area: int,
    effective_radius: float,
    lane_pitch: float,
    event_step: float,
) -> tuple[list[_PlannedPath], dict[str, JsonValue]]:
    distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    maximum_clearance = float(np.max(distance))
    used_thin_region_fallback = maximum_clearance < effective_radius
    center_clearance = (
        max(1.0, maximum_clearance * 0.5)
        if used_thin_region_fallback
        else effective_radius
    )
    center_mask = np.where(
        (component > 0)
        & (distance >= min(center_clearance, maximum_clearance)),
        255,
        0,
    ).astype(np.uint8)
    if not np.any(center_mask):
        center_mask = np.where(
            (component > 0) & (distance >= maximum_clearance),
            255,
            0,
        ).astype(np.uint8)

    axis, center = _principal_axis(center_mask)
    raw_paths: list[tuple[str, list[Point], bool]] = []
    raw_paths.extend(_boundary_paths(center_mask))
    raw_paths.extend(
        _interior_raster_paths(
            center_mask,
            axis=axis,
            center=center,
            lane_pitch=lane_pitch,
        )
    )
    if not raw_paths:
        y_values, x_values = np.nonzero(component)
        raw_paths.append(
            (
                "thin-region",
                [(float(x_values[0]), float(y_values[0]))],
                False,
            )
        )

    paths: list[_PlannedPath] = []
    kind_counts: dict[str, int] = {}
    for kind, points, closed in raw_paths:
        resampled = _uniform_resample(
            points,
            step=event_step,
            closed=closed,
        )
        if not resampled:
            continue
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        paths.append(
            _PlannedPath(
                region_id=region_id,
                kind=kind,
                sequence=kind_counts[kind],
                points=resampled,
                closed=closed,
            )
        )

    x, y, width, height = cv2.boundingRect(component)
    angle = math.degrees(math.atan2(axis[1], axis[0]))
    kind_summary: dict[str, int] = {}
    for path in paths:
        kind_summary[path.kind] = kind_summary.get(path.kind, 0) + 1
    return paths, {
        "region_id": region_id,
        "source_label": source_label,
        "area_pixels": area,
        "bbox_top_left": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        },
        "principal_axis_degrees_in_image_space": round(angle, 6),
        "effective_coverage_radius_pixels": round(effective_radius, 6),
        "lane_pitch_pixels": round(lane_pitch, 6),
        "thin_region_fallback": used_thin_region_fallback,
        "trajectory": {
            "representation": "ordered_strokes_with_pen_lifts",
            "path_names": [path.name for path in paths],
            "pen_lift_count": max(0, len(paths) - 1),
        },
        "path_counts": kind_summary,
        "operator_call_count": len(paths),
        "stroke_element_count": sum(len(path.points) for path in paths),
    }


def _boundary_paths(mask: MaskArray) -> list[tuple[str, list[Point], bool]]:
    contours, _ = cv2.findContours(
        mask.copy(),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    ordered: list[tuple[tuple[int, int, int, int, int], list[Point]]] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        points = [
            (float(item[0][0]), float(item[0][1])) for item in contour
        ]
        normalized = _normalize_closed_contour(points)
        ordered.append(((y, x, -width * height, height, width), normalized))
    ordered.sort(key=lambda item: item[0])
    return [("boundary", points, True) for _, points in ordered]


def _normalize_closed_contour(points: list[Point]) -> list[Point]:
    start = min(
        range(len(points)),
        key=lambda index: (points[index][1], points[index][0]),
    )
    rotated = points[start:] + points[:start]
    if len(rotated) > 2:
        forward = (rotated[1][1], rotated[1][0])
        backward = (rotated[-1][1], rotated[-1][0])
        if backward < forward:
            rotated = [rotated[0], *reversed(rotated[1:])]
    return rotated


def _interior_raster_paths(
    mask: MaskArray,
    *,
    axis: NDArray[np.float64],
    center: NDArray[np.float64],
    lane_pitch: float,
) -> list[tuple[str, list[Point], bool]]:
    y_values, x_values = np.nonzero(mask)
    if len(x_values) == 0:
        return []
    coordinates = np.column_stack((x_values, y_values)).astype(np.float64)
    normal = np.array([-axis[1], axis[0]], dtype=np.float64)
    relative = coordinates - center
    u_values = relative @ axis
    v_values = relative @ normal
    u_min = float(np.min(u_values)) - 2.0
    u_max = float(np.max(u_values)) + 2.0
    v_min = float(np.min(v_values))
    v_max = float(np.max(v_values))
    v_center = (v_min + v_max) / 2.0
    first_index = math.ceil((v_min - v_center) / lane_pitch)
    last_index = math.floor((v_max - v_center) / lane_pitch)
    lane_values = [
        v_center + lane_index * lane_pitch
        for lane_index in range(first_index, last_index + 1)
    ]
    if not lane_values:
        lane_values = [v_center]

    segments: list[list[Point]] = []
    for lane_number, lane_value in enumerate(lane_values):
        start = center + axis * u_min + normal * lane_value
        end = center + axis * u_max + normal * lane_value
        line_mask = np.zeros_like(mask)
        cv2.line(
            line_mask,
            _rounded_point(start),
            _rounded_point(end),
            255,
            thickness=1,
            lineType=cv2.LINE_8,
        )
        intersection = cv2.bitwise_and(line_mask, mask)
        count, run_labels, stats, _ = cv2.connectedComponentsWithStats(
            intersection,
            connectivity=8,
        )
        runs: list[tuple[float, list[Point]]] = []
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) == 0:
                continue
            run_y, run_x = np.nonzero(run_labels == label)
            run_points = np.column_stack((run_x, run_y)).astype(np.float64)
            projections = (run_points - center) @ axis
            first = run_points[int(np.argmin(projections))]
            last = run_points[int(np.argmax(projections))]
            runs.append(
                (
                    float(np.min(projections)),
                    [
                        (float(first[0]), float(first[1])),
                        (float(last[0]), float(last[1])),
                    ],
                )
            )
        runs.sort(key=lambda item: item[0])
        if lane_number % 2 == 1:
            runs.reverse()
            runs = [(position, list(reversed(points))) for position, points in runs]
        segments.extend(points for _, points in runs)
    merged = _merge_interior_segments(
        segments,
        mask=mask,
        maximum_connector_length=lane_pitch * 1.75,
    )
    return [("interior", points, False) for points in merged]


def _merge_interior_segments(
    segments: list[list[Point]],
    *,
    mask: MaskArray,
    maximum_connector_length: float,
) -> list[list[Point]]:
    if not segments:
        return []
    merged: list[list[Point]] = []
    current = segments[0].copy()
    for segment in segments[1:]:
        can_connect = (
            math.dist(current[-1], segment[0])
            <= maximum_connector_length
            and _segment_is_inside_mask(current[-1], segment[0], mask)
        )
        if can_connect:
            current.extend(segment)
        else:
            merged.append(current)
            current = segment.copy()
    merged.append(current)
    return merged


def _segment_is_inside_mask(
    start: Point,
    end: Point,
    mask: MaskArray,
) -> bool:
    connector = np.zeros_like(mask)
    cv2.line(
        connector,
        _rounded_point(start),
        _rounded_point(end),
        255,
        thickness=1,
        lineType=cv2.LINE_8,
    )
    outside = cv2.bitwise_and(connector, cv2.bitwise_not(mask))
    return not np.any(outside)


def _principal_axis(
    mask: MaskArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    y_values, x_values = np.nonzero(mask)
    coordinates = np.column_stack((x_values, y_values)).astype(np.float64)
    center = np.mean(coordinates, axis=0)
    if len(coordinates) < 2:
        return np.array([1.0, 0.0]), center
    relative = coordinates - center
    covariance = relative.T @ relative / len(coordinates)
    xx = float(covariance[0, 0])
    xy = float(covariance[0, 1])
    yy = float(covariance[1, 1])
    anisotropy = math.hypot(xx - yy, 2.0 * xy)
    tolerance = max(1e-12, (xx + yy) * 1e-9)
    if anisotropy <= tolerance:
        axis = np.array([1.0, 0.0])
    else:
        angle = 0.5 * math.atan2(2.0 * xy, xx - yy)
        axis = np.array([math.cos(angle), math.sin(angle)])
    # Fix the eigenvector sign so identical inputs cannot reverse the path.
    if axis[0] < 0.0 or (abs(axis[0]) < 1e-12 and axis[1] < 0.0):
        axis = -axis
    return axis.astype(np.float64), center


def _uniform_resample(
    points: list[Point],
    *,
    step: float,
    closed: bool,
) -> list[Point]:
    if not points:
        return []
    if len(points) == 1:
        return points.copy()

    source = points.copy()
    if closed:
        source.append(source[0])
    compact = [source[0]]
    for point in source[1:]:
        if math.dist(compact[-1], point) > 1e-9:
            compact.append(point)
    if len(compact) == 1:
        return [compact[0]]

    segment_lengths = [
        math.dist(compact[index], compact[index + 1])
        for index in range(len(compact) - 1)
    ]
    cumulative = np.concatenate(
        (np.array([0.0]), np.cumsum(segment_lengths, dtype=np.float64))
    )
    total = float(cumulative[-1])
    if total <= 1e-9:
        return [compact[0]]
    if closed:
        sample_distances = list(np.arange(0.0, total, step))
        if not sample_distances:
            sample_distances = [0.0]
        sample_distances.append(total)
    else:
        sample_distances = list(np.arange(0.0, total, step))
        if not sample_distances or total - sample_distances[-1] > 1e-9:
            sample_distances.append(total)

    result: list[Point] = []
    segment_index = 0
    for distance in sample_distances:
        while (
            segment_index < len(segment_lengths) - 1
            and distance > cumulative[segment_index + 1]
        ):
            segment_index += 1
        length = segment_lengths[segment_index]
        ratio = (
            0.0
            if length <= 1e-12
            else (distance - cumulative[segment_index]) / length
        )
        start = compact[segment_index]
        end = compact[segment_index + 1]
        result.append(
            (
                start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio,
            )
        )
    return result


def _assign_full_pressure(paths: list[_PlannedPath]) -> None:
    for path in paths:
        path.pressures = [1.0] * len(path.points)


def _operator_call(
    path: _PlannedPath,
    *,
    mapping: ScreenshotViewportMapping,
    brush_size: int,
    brush_direction: str | None,
) -> dict[str, JsonValue]:
    elements: list[JsonValue] = []
    elapsed = 0.0
    previous: Point | None = None
    for index, (point, pressure) in enumerate(
        zip(path.points, path.pressures, strict=True)
    ):
        if previous is not None:
            elapsed += math.dist(previous, point) / (
                _MOUSE_SPEED_PIXELS_PER_SECOND
            )
        mouse = [
            round(point[0], 6),
            round(mapping.region_height - 1.0 - point[1], 6),
        ]
        elements.append(
            {
                "name": path.name,
                "location": [0.0, 0.0, 0.0],
                "mouse": mouse,
                "mouse_event": mouse.copy(),
                "pressure": pressure,
                "size": float(brush_size),
                "x_tilt": 0.0,
                "y_tilt": 0.0,
                "time": round(elapsed, 6),
                "is_start": index == 0,
            }
        )
        previous = point

    return {
        "region_id": path.region_id,
        "path_kind": path.kind,
        "path_name": path.name,
        "operator": "bpy.ops.sculpt.brush_stroke",
        "context": {
            "window_index": mapping.window_index,
            "area_index": mapping.area_index,
            "region_type": "WINDOW",
            "coordinate_space": "VIEW_3D_WINDOW_REGION",
        },
        "kwargs": {
            "stroke": elements,
            "mode": "NORMAL",
            "brush_toggle": (
                "SMOOTH" if brush_direction == "SMOOTH" else "None"
            ),
            "pen_flip": False,
            "execution_mode": "DIRECT_DABS",
            "location_mode": "SURFACE_RAYCAST",
            "ignore_background_click": True,
        },
    }


def _resize_and_threshold(
    grayscale: MaskArray,
    mapping: ScreenshotViewportMapping,
) -> MaskArray:
    if grayscale.shape != (mapping.region_height, mapping.region_width):
        grayscale = cv2.resize(
            grayscale,
            (mapping.region_width, mapping.region_height),
            interpolation=cv2.INTER_LINEAR,
        )
    return np.where(grayscale >= _MASK_THRESHOLD, 255, 0).astype(np.uint8)


def _load_grayscale_mask(path: Path) -> MaskArray:
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.uint8)
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise SculptStrokePlanningError(
            f"cannot read SAM3 mask image: {path}"
        ) from error


def _resolve_mask_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()
    if not path.is_file():
        raise SculptStrokePlanningError(f"SAM3 mask is not a file: {path}")
    return path


def _validate_sculpt_parameters(
    *,
    sculpt_brush: str,
    brush_size: int,
    brush_strength: float,
    brush_direction: str | None,
) -> None:
    if (
        not isinstance(sculpt_brush, str)
        or not sculpt_brush.strip()
        or len(sculpt_brush) > 128
        or any(ord(character) < 32 for character in sculpt_brush)
    ):
        raise SculptStrokePlanningError("sculpt_brush is invalid")
    if (
        isinstance(brush_size, bool)
        or not isinstance(brush_size, int)
        or not 1 <= brush_size <= 10_000
    ):
        raise SculptStrokePlanningError(
            "brush_size must be an integer between 1 and 10000"
        )
    if (
        isinstance(brush_strength, bool)
        or not isinstance(brush_strength, (int, float))
        or not math.isfinite(float(brush_strength))
        or not 0.0 <= float(brush_strength) <= 1.0
    ):
        raise SculptStrokePlanningError(
            "brush_strength must be between 0 and 1"
        )
    if brush_direction is not None and (
        not isinstance(brush_direction, str)
        or not brush_direction.strip()
        or brush_direction != brush_direction.strip().upper()
        or len(brush_direction) > 64
        or any(ord(character) < 32 for character in brush_direction)
    ):
        raise SculptStrokePlanningError(
            "brush_direction must be null or an uppercase Blender identifier"
        )


def _resolve_plan_output_dir(
    *,
    output_dir: str | Path | None,
    mask_path: Path,
) -> Path:
    directory = (
        mask_path.parent
        if output_dir is None
        else Path(
            os.path.expandvars(os.path.expanduser(str(output_dir)))
        ).resolve()
    )
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise SculptStrokePlanningError(
            f"cannot create stroke-plan output directory: {directory}"
        ) from error
    return directory


def _output_path(*, mask_path: Path, output_dir: Path) -> Path:
    stem = mask_path.stem
    if stem.endswith("-cleaned-mask"):
        stem = stem[: -len("-cleaned-mask")]
    return output_dir / f"{stem}-sculpt-stroke-plan.json"


def _write_json(path: Path, payload: dict[str, JsonValue]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError as error:
        raise SculptStrokePlanningError(
            f"cannot write stroke plan: {path}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _required_mapping(
    mapping: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise SculptStrokePlanningError(
            f"screenshot_metadata.{key} must be an object"
        )
    return value


def _required_positive_int(
    mapping: Mapping[str, object],
    key: str,
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SculptStrokePlanningError(
            f"screenshot_metadata.{key} must be a positive integer"
        )
    return value


def _required_positive_float(
    mapping: Mapping[str, object],
    key: str,
) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SculptStrokePlanningError(
            f"screenshot_metadata.{key} must be a positive number"
        )
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise SculptStrokePlanningError(
            f"screenshot_metadata.{key} must be a positive number"
        )
    return number


def _optional_nonnegative_int(
    mapping: Mapping[str, object],
    key: str,
) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SculptStrokePlanningError(
            f"screenshot_metadata.{key} must be a non-negative integer"
        )
    return value


def _rounded_point(value: Point | NDArray[np.float64]) -> tuple[int, int]:
    return int(round(float(value[0]))), int(round(float(value[1])))
