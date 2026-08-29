"""Fit SVG mouse trajectories inside a binary segmentation mask."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from visculpt.bridge import JsonValue

type Point = tuple[float, float]
type MaskArray = NDArray[np.uint8]
type TrajectoryScaleTier = Literal["SMALL", "MEDIUM", "LARGE"]

_MASK_THRESHOLD = 128
_POINT_EPSILON = 1e-9
_MAX_EXACT_CANDIDATES = 2_048
_SCALE_TIERS = {"SMALL", "MEDIUM", "LARGE"}


class MaskTrajectoryFitError(ValueError):
    """Raised when trajectories cannot be placed safely inside a mask."""


@dataclass(frozen=True, slots=True)
class MaskTrajectoryFitConfig:
    """Deterministic coarse-to-fine fitting settings."""

    containment_margin_pixels: float = 1.0
    small_boundary_clearance_ratio: float = 0.45
    medium_boundary_clearance_ratio: float = 0.28
    large_boundary_clearance_ratio: float = 0.05
    optimization_max_dimension: int = 640
    rotation_coarse_step_degrees: float = 15.0
    rotation_refine_step_degrees: float = 3.0
    rotation_fine_step_degrees: float = 0.5
    scale_search_iterations: int = 10

    def __post_init__(self) -> None:
        """Validate bounded deterministic search settings."""
        if (
            not math.isfinite(self.containment_margin_pixels)
            or self.containment_margin_pixels < 0.0
            or self.containment_margin_pixels > 32.0
        ):
            raise ValueError(
                "containment_margin_pixels must be between 0 and 32"
            )
        clearance_ratios = (
            self.large_boundary_clearance_ratio,
            self.medium_boundary_clearance_ratio,
            self.small_boundary_clearance_ratio,
        )
        if any(
            not math.isfinite(ratio) or not 0.0 < ratio < 1.0
            for ratio in clearance_ratios
        ):
            raise ValueError(
                "boundary clearance ratios must be between 0 and 1"
            )
        if not (
            self.large_boundary_clearance_ratio
            < self.medium_boundary_clearance_ratio
            < self.small_boundary_clearance_ratio
        ):
            raise ValueError(
                "boundary clearance ratios must satisfy large < medium < "
                "small"
            )
        if not 64 <= self.optimization_max_dimension <= 2_048:
            raise ValueError(
                "optimization_max_dimension must be between 64 and 2048"
            )
        steps = (
            self.rotation_coarse_step_degrees,
            self.rotation_refine_step_degrees,
            self.rotation_fine_step_degrees,
        )
        if any(not math.isfinite(step) or step <= 0.0 for step in steps):
            raise ValueError("rotation search steps must be finite and positive")
        if not (
            self.rotation_fine_step_degrees
            <= self.rotation_refine_step_degrees
            <= self.rotation_coarse_step_degrees
            <= 180.0
        ):
            raise ValueError(
                "rotation steps must satisfy fine <= refine <= coarse <= 180"
            )
        if not 4 <= self.scale_search_iterations <= 20:
            raise ValueError(
                "scale_search_iterations must be between 4 and 20"
            )


@dataclass(frozen=True, slots=True)
class MaskFittedTrajectoryResult:
    """Transformed trajectories and fitting diagnostics."""

    trajectory_plan: dict[str, JsonValue]

    def as_payload(self) -> dict[str, JsonValue]:
        """Return the JSON-compatible transformed plan."""
        return self.trajectory_plan


@dataclass(frozen=True, slots=True)
class _SourceTrajectory:
    trajectory_id: str
    closed: bool
    source: dict[str, JsonValue]
    points: tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class _PreparedMask:
    component: MaskArray
    safe_component: MaskArray
    crop: MaskArray
    crop_x: int
    crop_y: int
    target_center: Point
    component_count: int
    component_area: int
    safe_area: int
    scale_tier: TrajectoryScaleTier
    boundary_clearance_ratio: float
    target_clearance_pixels: float
    maximum_clearance_pixels: float


@dataclass(frozen=True, slots=True)
class _Placement:
    angle_degrees: float
    scale: float
    center_x: float
    center_y: float
    center_distance: float


@dataclass(frozen=True, slots=True)
class _RasterTemplate:
    pixels: MaskArray
    center_x: float
    center_y: float


def fit_svg_trajectories_to_mask(
    *,
    mask_path: str | Path,
    trajectory_plan: Mapping[str, object],
    scale_tier: str = "MEDIUM",
    config: MaskTrajectoryFitConfig | None = None,
) -> MaskFittedTrajectoryResult:
    """Fit trajectories inside an adaptive scale-tier safety inset."""
    settings = config or MaskTrajectoryFitConfig()
    normalized_scale_tier = _normalize_scale_tier(scale_tier)
    source_trajectories = _parse_trajectory_plan(trajectory_plan)
    source_center = _source_center(source_trajectories)
    mask = _load_mask(mask_path)
    prepared = _prepare_mask(
        mask,
        settings=settings,
        scale_tier=normalized_scale_tier,
    )
    target_local = (
        prepared.target_center[0] - prepared.crop_x,
        prepared.target_center[1] - prepared.crop_y,
    )
    placement = _maximize_placement(
        trajectories=source_trajectories,
        source_center=source_center,
        angle_degrees=0.0,
        mask=prepared.crop,
        target_center=target_local,
        iterations=settings.scale_search_iterations + 2,
    )
    rotation_fallback_used = placement is None
    if placement is None:
        optimization_mask, scale_x, scale_y = _optimization_mask(
            prepared.crop,
            maximum_dimension=settings.optimization_max_dimension,
        )
        optimization_target = (
            target_local[0] * scale_x,
            target_local[1] * scale_y,
        )
        selected_angle = _select_rotation(
            trajectories=source_trajectories,
            source_center=source_center,
            mask=optimization_mask,
            target_center=optimization_target,
            settings=settings,
        )
        final_candidates = {
            _normalized_angle(selected_angle),
            _normalized_angle(
                selected_angle - settings.rotation_fine_step_degrees
            ),
            _normalized_angle(
                selected_angle + settings.rotation_fine_step_degrees
            ),
        }
        final_placements = [
            candidate
            for angle in sorted(final_candidates)
            if (
                candidate := _maximize_placement(
                    trajectories=source_trajectories,
                    source_center=source_center,
                    angle_degrees=angle,
                    mask=prepared.crop,
                    target_center=target_local,
                    iterations=settings.scale_search_iterations + 2,
                )
            )
            is not None
        ]
        if final_placements:
            placement = _best_placement(final_placements)
    if placement is None:
        raise MaskTrajectoryFitError(
            "No rotation, scale, and translation can place the trajectories "
            "inside the selected mask component"
        )
    absolute_placement = _Placement(
        angle_degrees=placement.angle_degrees,
        scale=placement.scale,
        center_x=placement.center_x + prepared.crop_x,
        center_y=placement.center_y + prepared.crop_y,
        center_distance=placement.center_distance,
    )
    transformed = _transform_trajectories(
        source_trajectories,
        source_center=source_center,
        placement=absolute_placement,
    )
    containment = _verify_containment(
        transformed,
        component=prepared.component,
        safe_component=prepared.safe_component,
    )
    plan = _result_plan(
        mask_path=Path(mask_path).expanduser().resolve(),
        mask_shape=mask.shape,
        prepared=prepared,
        source_center=source_center,
        placement=absolute_placement,
        transformed=transformed,
        containment=containment,
        settings=settings,
        rotation_fallback_used=rotation_fallback_used,
    )
    return MaskFittedTrajectoryResult(trajectory_plan=plan)


def _parse_trajectory_plan(
    plan: Mapping[str, object],
) -> tuple[_SourceTrajectory, ...]:
    if plan.get("format") != "svg-mouse-trajectories/v1":
        raise MaskTrajectoryFitError(
            "trajectory_plan.format must be svg-mouse-trajectories/v1"
        )
    raw_trajectories = plan.get("trajectories")
    if not isinstance(raw_trajectories, list) or not raw_trajectories:
        raise MaskTrajectoryFitError(
            "trajectory_plan must contain at least one trajectory"
        )
    trajectories: list[_SourceTrajectory] = []
    identifiers: set[str] = set()
    for raw_trajectory in raw_trajectories:
        if not isinstance(raw_trajectory, Mapping):
            raise MaskTrajectoryFitError("trajectory must be an object")
        identifier = raw_trajectory.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise MaskTrajectoryFitError(
                "each trajectory must have a non-empty id"
            )
        identifier = identifier.strip()
        if identifier in identifiers:
            raise MaskTrajectoryFitError("trajectory ids must be unique")
        identifiers.add(identifier)
        closed = raw_trajectory.get("closed", False)
        if not isinstance(closed, bool):
            raise MaskTrajectoryFitError("trajectory.closed must be boolean")
        raw_points = raw_trajectory.get("points")
        if not isinstance(raw_points, list) or not raw_points:
            raise MaskTrajectoryFitError(
                "each trajectory must contain at least one point"
            )
        points = tuple(_trajectory_point(point) for point in raw_points)
        raw_source = raw_trajectory.get("source", {})
        source = (
            dict(raw_source)
            if isinstance(raw_source, Mapping)
            else {}
        )
        trajectories.append(
            _SourceTrajectory(
                trajectory_id=identifier,
                closed=closed,
                source=source,
                points=points,
            )
        )
    return tuple(trajectories)


def _trajectory_point(value: object) -> Point:
    if not isinstance(value, Mapping):
        raise MaskTrajectoryFitError("trajectory point must be an object")
    coordinates: list[float] = []
    for axis in ("x", "y"):
        raw_coordinate = value.get(axis)
        if isinstance(raw_coordinate, bool) or not isinstance(
            raw_coordinate, (int, float)
        ):
            raise MaskTrajectoryFitError(
                f"trajectory point {axis} must be numeric"
            )
        coordinate = float(raw_coordinate)
        if not math.isfinite(coordinate) or not 0.0 <= coordinate <= 512.0:
            raise MaskTrajectoryFitError(
                f"trajectory point {axis} must be within 0..512"
            )
        coordinates.append(coordinate)
    return coordinates[0], coordinates[1]


def _load_mask(mask_path: str | Path) -> MaskArray:
    path = Path(mask_path).expanduser().resolve()
    if not path.is_file():
        raise MaskTrajectoryFitError(
            "mask_path must reference an existing image file"
        )
    try:
        with Image.open(path) as image:
            grayscale = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    except (OSError, UnidentifiedImageError) as error:
        raise MaskTrajectoryFitError(
            f"Cannot read segmentation mask {path}: {error}"
        ) from error
    if grayscale.ndim != 2 or grayscale.size == 0:
        raise MaskTrajectoryFitError("segmentation mask is empty or invalid")
    return np.where(grayscale >= _MASK_THRESHOLD, 255, 0).astype(np.uint8)


def _normalize_scale_tier(value: str) -> TrajectoryScaleTier:
    if not isinstance(value, str):
        raise MaskTrajectoryFitError("scale_tier must be a string")
    normalized = value.strip().upper()
    if normalized not in _SCALE_TIERS:
        raise MaskTrajectoryFitError(
            "scale_tier must be SMALL, MEDIUM, or LARGE"
        )
    return cast(TrajectoryScaleTier, normalized)


def _distance_to_background(component: MaskArray) -> NDArray[np.float32]:
    """Treat pixels outside the screenshot as mask background."""
    padded = np.pad(component, 1, mode="constant", constant_values=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    return distance[1:-1, 1:-1]


def _boundary_clearance_ratio(
    settings: MaskTrajectoryFitConfig,
    scale_tier: TrajectoryScaleTier,
) -> float:
    return {
        "SMALL": settings.small_boundary_clearance_ratio,
        "MEDIUM": settings.medium_boundary_clearance_ratio,
        "LARGE": settings.large_boundary_clearance_ratio,
    }[scale_tier]


def _prepare_mask(
    mask: MaskArray,
    *,
    settings: MaskTrajectoryFitConfig,
    scale_tier: TrajectoryScaleTier,
) -> _PreparedMask:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    if component_count <= 1:
        raise MaskTrajectoryFitError(
            "segmentation mask contains no foreground component"
        )
    foreground_labels = range(1, component_count)
    selected_label = max(
        foreground_labels,
        key=lambda label: int(stats[label, cv2.CC_STAT_AREA]),
    )
    component = np.where(labels == selected_label, 255, 0).astype(np.uint8)
    component_area = int(stats[selected_label, cv2.CC_STAT_AREA])
    distance = _distance_to_background(component)
    maximum_clearance = float(np.max(distance))
    clearance_ratio = _boundary_clearance_ratio(settings, scale_tier)
    target_clearance = max(
        settings.containment_margin_pixels,
        maximum_clearance * clearance_ratio,
    )
    safe_component = np.where(
        distance >= target_clearance,
        255,
        0,
    ).astype(np.uint8)
    safe_area = int(np.count_nonzero(safe_component))
    if safe_area == 0:
        raise MaskTrajectoryFitError(
            "selected mask component disappears after applying the "
            f"{scale_tier} boundary clearance"
        )
    ys, xs = np.nonzero(safe_component)
    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1
    target_center = _interior_centroid(safe_component, xs=xs, ys=ys)
    return _PreparedMask(
        component=component,
        safe_component=safe_component,
        crop=safe_component[top:bottom, left:right],
        crop_x=left,
        crop_y=top,
        target_center=target_center,
        component_count=component_count - 1,
        component_area=component_area,
        safe_area=safe_area,
        scale_tier=scale_tier,
        boundary_clearance_ratio=clearance_ratio,
        target_clearance_pixels=target_clearance,
        maximum_clearance_pixels=maximum_clearance,
    )


def _interior_centroid(
    mask: MaskArray,
    *,
    xs: NDArray[np.int64],
    ys: NDArray[np.int64],
) -> Point:
    centroid_x = float(np.mean(xs))
    centroid_y = float(np.mean(ys))
    rounded_x = int(round(centroid_x))
    rounded_y = int(round(centroid_y))
    if (
        0 <= rounded_x < mask.shape[1]
        and 0 <= rounded_y < mask.shape[0]
        and mask[rounded_y, rounded_x] != 0
    ):
        return centroid_x, centroid_y
    distances = (xs.astype(np.float64) - centroid_x) ** 2 + (
        ys.astype(np.float64) - centroid_y
    ) ** 2
    index = int(np.argmin(distances))
    return float(xs[index]), float(ys[index])


def _optimization_mask(
    mask: MaskArray,
    *,
    maximum_dimension: int,
) -> tuple[MaskArray, float, float]:
    height, width = mask.shape
    ratio = min(1.0, maximum_dimension / max(height, width))
    if math.isclose(ratio, 1.0):
        return np.where(mask != 0, 1, 0).astype(np.uint8), 1.0, 1.0
    target_width = max(1, int(round(width * ratio)))
    target_height = max(1, int(round(height * ratio)))
    resized = cv2.resize(
        mask,
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return (
        np.where(resized != 0, 1, 0).astype(np.uint8),
        target_width / width,
        target_height / height,
    )


def _source_center(trajectories: tuple[_SourceTrajectory, ...]) -> Point:
    points = [point for trajectory in trajectories for point in trajectory.points]
    return (
        (min(point[0] for point in points) + max(point[0] for point in points))
        / 2.0,
        (min(point[1] for point in points) + max(point[1] for point in points))
        / 2.0,
    )


def _select_rotation(
    *,
    trajectories: tuple[_SourceTrajectory, ...],
    source_center: Point,
    mask: MaskArray,
    target_center: Point,
    settings: MaskTrajectoryFitConfig,
) -> float:
    cache: dict[float, _Placement | None] = {}

    def evaluate(angle: float) -> _Placement | None:
        normalized = _normalized_angle(angle)
        key = round(normalized, 6)
        if key not in cache:
            cache[key] = _maximize_placement(
                trajectories=trajectories,
                source_center=source_center,
                angle_degrees=normalized,
                mask=mask,
                target_center=target_center,
                iterations=settings.scale_search_iterations,
            )
        return cache[key]

    coarse_angles = _angle_range(
        start=0.0,
        stop=360.0,
        step=settings.rotation_coarse_step_degrees,
    )
    coarse = [placement for angle in coarse_angles if (placement := evaluate(angle))]
    if not coarse:
        raise MaskTrajectoryFitError(
            "No coarse rotation candidate can fit inside the mask"
        )
    seeds = sorted(coarse, key=_placement_sort_key)[:2]
    refined: list[_Placement] = coarse.copy()
    for seed in seeds:
        for offset in _angle_range(
            start=-settings.rotation_coarse_step_degrees,
            stop=settings.rotation_coarse_step_degrees
            + settings.rotation_refine_step_degrees * 0.5,
            step=settings.rotation_refine_step_degrees,
        ):
            placement = evaluate(seed.angle_degrees + offset)
            if placement is not None:
                refined.append(placement)
    best_refined = _best_placement(refined)
    fine: list[_Placement] = refined.copy()
    for offset in _angle_range(
        start=-settings.rotation_refine_step_degrees,
        stop=settings.rotation_refine_step_degrees
        + settings.rotation_fine_step_degrees * 0.5,
        step=settings.rotation_fine_step_degrees,
    ):
        placement = evaluate(best_refined.angle_degrees + offset)
        if placement is not None:
            fine.append(placement)
    return _best_placement(fine).angle_degrees


def _maximize_placement(
    *,
    trajectories: tuple[_SourceTrajectory, ...],
    source_center: Point,
    angle_degrees: float,
    mask: MaskArray,
    target_center: Point,
    iterations: int,
) -> _Placement | None:
    rotated = _rotated_paths(
        trajectories,
        source_center=source_center,
        angle_degrees=angle_degrees,
    )
    upper = _scale_upper_bound(rotated, mask_shape=mask.shape)
    if upper <= 0.0:
        return None
    lower = 0.0
    best: _Placement | None = None
    for _ in range(iterations):
        candidate_scale = (lower + upper) / 2.0
        placement = _find_placement(
            rotated_paths=rotated,
            closed=tuple(item.closed for item in trajectories),
            scale=candidate_scale,
            angle_degrees=angle_degrees,
            mask=mask,
            target_center=target_center,
        )
        if placement is None:
            upper = candidate_scale
        else:
            lower = candidate_scale
            best = placement
    if best is not None:
        return best
    return _find_placement(
        rotated_paths=rotated,
        closed=tuple(item.closed for item in trajectories),
        scale=min(upper, 1e-3),
        angle_degrees=angle_degrees,
        mask=mask,
        target_center=target_center,
    )


def _rotated_paths(
    trajectories: tuple[_SourceTrajectory, ...],
    *,
    source_center: Point,
    angle_degrees: float,
) -> tuple[tuple[Point, ...], ...]:
    radians = math.radians(angle_degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    paths: list[tuple[Point, ...]] = []
    for trajectory in trajectories:
        rotated: list[Point] = []
        for x, y in trajectory.points:
            local_x = x - source_center[0]
            local_y = y - source_center[1]
            rotated.append(
                (
                    cosine * local_x - sine * local_y,
                    sine * local_x + cosine * local_y,
                )
            )
        paths.append(tuple(rotated))
    return tuple(paths)


def _scale_upper_bound(
    paths: tuple[tuple[Point, ...], ...],
    *,
    mask_shape: tuple[int, int],
) -> float:
    points = [point for path in paths for point in path]
    span_x = max(point[0] for point in points) - min(point[0] for point in points)
    span_y = max(point[1] for point in points) - min(point[1] for point in points)
    bounds: list[float] = []
    if span_x > _POINT_EPSILON:
        bounds.append(max(0.0, mask_shape[1] - 1.0) / span_x)
    if span_y > _POINT_EPSILON:
        bounds.append(max(0.0, mask_shape[0] - 1.0) / span_y)
    if not bounds:
        return 1.0
    return min(bounds)


def _find_placement(
    *,
    rotated_paths: tuple[tuple[Point, ...], ...],
    closed: tuple[bool, ...],
    scale: float,
    angle_degrees: float,
    mask: MaskArray,
    target_center: Point,
) -> _Placement | None:
    template = _raster_template(
        rotated_paths,
        closed=closed,
        scale=scale,
    )
    template_height, template_width = template.pixels.shape
    mask_height, mask_width = mask.shape
    if template_width > mask_width or template_height > mask_height:
        return None
    inverted = np.where(mask == 0, 1, 0).astype(np.uint8)
    mismatch = cv2.matchTemplate(
        inverted,
        template.pixels,
        cv2.TM_CCORR,
    )
    candidate_y, candidate_x = np.nonzero(mismatch <= 0.25)
    if candidate_x.size == 0:
        return None
    center_x = candidate_x.astype(np.float64) + template.center_x
    center_y = candidate_y.astype(np.float64) + template.center_y
    distances = (center_x - target_center[0]) ** 2 + (
        center_y - target_center[1]
    ) ** 2
    if candidate_x.size > _MAX_EXACT_CANDIDATES:
        selected = np.argpartition(
            distances,
            _MAX_EXACT_CANDIDATES - 1,
        )[:_MAX_EXACT_CANDIDATES]
        selected = selected[np.argsort(distances[selected])]
    else:
        selected = np.argsort(distances)
    template_pixels = template.pixels != 0
    for candidate_index in selected:
        x = int(candidate_x[candidate_index])
        y = int(candidate_y[candidate_index])
        window = mask[y : y + template_height, x : x + template_width]
        if np.all(window[template_pixels] != 0):
            return _Placement(
                angle_degrees=_normalized_angle(angle_degrees),
                scale=scale,
                center_x=float(x + template.center_x),
                center_y=float(y + template.center_y),
                center_distance=math.sqrt(float(distances[candidate_index])),
            )
    return None


def _raster_template(
    paths: tuple[tuple[Point, ...], ...],
    *,
    closed: tuple[bool, ...],
    scale: float,
) -> _RasterTemplate:
    scaled_paths = [
        tuple((point[0] * scale, point[1] * scale) for point in path)
        for path in paths
    ]
    points = [point for path in scaled_paths for point in path]
    minimum_x = math.floor(min(point[0] for point in points)) - 1
    minimum_y = math.floor(min(point[1] for point in points)) - 1
    maximum_x = math.ceil(max(point[0] for point in points)) + 1
    maximum_y = math.ceil(max(point[1] for point in points)) + 1
    width = max(1, maximum_x - minimum_x + 1)
    height = max(1, maximum_y - minimum_y + 1)
    pixels = np.zeros((height, width), dtype=np.uint8)
    shift_x = -minimum_x
    shift_y = -minimum_y
    for path, is_closed in zip(scaled_paths, closed):
        coordinates = np.rint(
            np.asarray(
                [
                    (point[0] + shift_x, point[1] + shift_y)
                    for point in path
                ],
                dtype=np.float64,
            )
        ).astype(np.int32)
        if len(coordinates) == 1:
            x, y = coordinates[0]
            pixels[y, x] = 1
        else:
            cv2.polylines(
                pixels,
                [coordinates.reshape((-1, 1, 2))],
                isClosed=is_closed,
                color=1,
                thickness=1,
                lineType=cv2.LINE_8,
            )
    return _RasterTemplate(
        pixels=pixels,
        center_x=float(shift_x),
        center_y=float(shift_y),
    )


def _transform_trajectories(
    trajectories: tuple[_SourceTrajectory, ...],
    *,
    source_center: Point,
    placement: _Placement,
) -> list[dict[str, JsonValue]]:
    radians = math.radians(placement.angle_degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    transformed: list[dict[str, JsonValue]] = []
    for trajectory in trajectories:
        points: list[dict[str, JsonValue]] = []
        exact_points: list[Point] = []
        for x, y in trajectory.points:
            local_x = x - source_center[0]
            local_y = y - source_center[1]
            transformed_point = (
                placement.center_x
                + placement.scale * (cosine * local_x - sine * local_y),
                placement.center_y
                + placement.scale * (sine * local_x + cosine * local_y),
            )
            exact_points.append(transformed_point)
            points.append(
                {
                    "x": _rounded(transformed_point[0]),
                    "y": _rounded(transformed_point[1]),
                }
            )
        transformed.append(
            {
                "id": trajectory.trajectory_id,
                "source": trajectory.source,
                "closed": trajectory.closed,
                "length_pixels": round(
                    _polyline_length(exact_points), 6
                ),
                "point_count": len(points),
                "points": points,
            }
        )
    return transformed


def _verify_containment(
    trajectories: list[dict[str, JsonValue]],
    *,
    component: MaskArray,
    safe_component: MaskArray,
) -> dict[str, JsonValue]:
    raster = np.zeros_like(component)
    for trajectory in trajectories:
        raw_points = trajectory["points"]
        assert isinstance(raw_points, list)
        coordinates = np.rint(
            np.asarray(
                [
                    (float(point["x"]), float(point["y"]))
                    for point in raw_points
                    if isinstance(point, dict)
                ],
                dtype=np.float64,
            )
        ).astype(np.int32)
        if len(coordinates) == 1:
            x, y = coordinates[0]
            if 0 <= x < raster.shape[1] and 0 <= y < raster.shape[0]:
                raster[y, x] = 255
        elif len(coordinates) > 1:
            cv2.polylines(
                raster,
                [coordinates.reshape((-1, 1, 2))],
                isClosed=bool(trajectory["closed"]),
                color=255,
                thickness=1,
                lineType=cv2.LINE_8,
            )
    stroke_pixels = raster != 0
    stroke_count = int(np.count_nonzero(stroke_pixels))
    if stroke_count == 0:
        raise MaskTrajectoryFitError(
            "transformed trajectories contain no rasterized pixels"
        )
    safe_count = int(np.count_nonzero(stroke_pixels & (safe_component != 0)))
    if safe_count != stroke_count:
        raise MaskTrajectoryFitError(
            "internal containment verification failed after fitting"
        )
    distance = _distance_to_background(component)
    minimum_clearance = float(np.min(distance[stroke_pixels]))
    return {
        "verified": True,
        "stroke_pixel_count": stroke_count,
        "inside_safe_mask_pixel_count": safe_count,
        "containment_ratio": 1.0,
        "minimum_clearance_pixels": round(minimum_clearance, 6),
        "verification": "RASTERIZED_COMPLETE_POLYLINES",
    }


def _result_plan(
    *,
    mask_path: Path,
    mask_shape: tuple[int, int],
    prepared: _PreparedMask,
    source_center: Point,
    placement: _Placement,
    transformed: list[dict[str, JsonValue]],
    containment: dict[str, JsonValue],
    settings: MaskTrajectoryFitConfig,
    rotation_fallback_used: bool,
) -> dict[str, JsonValue]:
    radians = math.radians(placement.angle_degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    a = placement.scale * cosine
    b = placement.scale * sine
    c = -placement.scale * sine
    d = placement.scale * cosine
    translate_x = placement.center_x - a * source_center[0] - c * source_center[1]
    translate_y = placement.center_y - b * source_center[0] - d * source_center[1]
    total_points = sum(int(item["point_count"]) for item in transformed)
    return {
        "format": "mask-fitted-svg-mouse-trajectories/v1",
        "source_format": "svg-mouse-trajectories/v1",
        "algorithm": {
            "name": "adaptive_scale_tier_template_containment",
            "deterministic": True,
            "objective_priority": [
                "SATISFY_SCALE_TIER_BOUNDARY_CLEARANCE",
                "PRESERVE_ZERO_ROTATION",
                "MAXIMIZE_UNIFORM_SCALE_WITHIN_TIER_SAFE_REGION",
                "MINIMIZE_DISTANCE_TO_MASK_CENTER",
            ],
            "rotation_policy": (
                "ZERO_DEGREES_FIRST; SEARCH_ROTATION_ONLY_IF_INFEASIBLE"
            ),
            "rotation_fallback_used": rotation_fallback_used,
            "complete_polyline_containment": True,
        },
        "coordinate_system": {
            "width": mask_shape[1],
            "height": mask_shape[0],
            "origin": "TOP_LEFT",
            "x_range": [0.0, float(mask_shape[1] - 1)],
            "y_range": [0.0, float(mask_shape[0] - 1)],
            "units": "MASK_PIXELS",
        },
        "gesture_contract": {
            "one_trajectory_per_mouse_gesture": True,
            "first_point": "MOUSE_DOWN",
            "intermediate_points": "MOUSE_MOVE",
            "last_point": "MOUSE_UP",
        },
        "transform": {
            "rotation_degrees": round(placement.angle_degrees, 6),
            "uniform_scale": round(placement.scale, 9),
            "translation": {
                "x": round(translate_x, 6),
                "y": round(translate_y, 6),
            },
            "affine_matrix": [
                round(a, 9),
                round(b, 9),
                round(c, 9),
                round(d, 9),
                round(translate_x, 6),
                round(translate_y, 6),
            ],
            "source_center": {
                "x": round(source_center[0], 6),
                "y": round(source_center[1], 6),
            },
            "fitted_center": {
                "x": round(placement.center_x, 6),
                "y": round(placement.center_y, 6),
            },
            "mask_center": {
                "x": round(prepared.target_center[0], 6),
                "y": round(prepared.target_center[1], 6),
            },
            "center_offset_pixels": round(placement.center_distance, 6),
        },
        "sizing": {
            "scale_tier": prepared.scale_tier,
            "policy": "ADAPTIVE_MASK_INTERIOR_CLEARANCE",
            "boundary_clearance_ratio": round(
                prepared.boundary_clearance_ratio,
                6,
            ),
            "target_boundary_clearance_pixels": round(
                prepared.target_clearance_pixels,
                6,
            ),
            "maximum_mask_clearance_pixels": round(
                prepared.maximum_clearance_pixels,
                6,
            ),
            "resolved_uniform_scale": round(placement.scale, 9),
        },
        "mask": {
            "path": str(mask_path),
            "width": mask_shape[1],
            "height": mask_shape[0],
            "threshold": _MASK_THRESHOLD,
            "foreground_component_count": prepared.component_count,
            "selected_component_policy": "LARGEST_AREA",
            "selected_component_area": prepared.component_area,
            "safe_component_area": prepared.safe_area,
            "containment_margin_pixels": settings.containment_margin_pixels,
            "adaptive_boundary_clearance_pixels": round(
                prepared.target_clearance_pixels,
                6,
            ),
        },
        "search": {
            "optimization_max_dimension": settings.optimization_max_dimension,
            "rotation_coarse_step_degrees": (
                settings.rotation_coarse_step_degrees
            ),
            "rotation_refine_step_degrees": (
                settings.rotation_refine_step_degrees
            ),
            "rotation_fine_step_degrees": (
                settings.rotation_fine_step_degrees
            ),
            "scale_search_iterations": settings.scale_search_iterations,
            "small_boundary_clearance_ratio": (
                settings.small_boundary_clearance_ratio
            ),
            "medium_boundary_clearance_ratio": (
                settings.medium_boundary_clearance_ratio
            ),
            "large_boundary_clearance_ratio": (
                settings.large_boundary_clearance_ratio
            ),
        },
        "containment": containment,
        "trajectories": transformed,
        "summary": {
            "trajectory_count": len(transformed),
            "point_count": total_points,
            "closed_trajectory_count": sum(
                int(bool(item["closed"])) for item in transformed
            ),
            "total_length_pixels": round(
                sum(float(item["length_pixels"]) for item in transformed),
                6,
            ),
        },
    }


def _best_placement(placements: list[_Placement]) -> _Placement:
    return sorted(placements, key=_placement_sort_key)[0]


def _placement_sort_key(placement: _Placement) -> tuple[float, float, float]:
    return (
        -placement.scale,
        placement.center_distance,
        placement.angle_degrees,
    )


def _angle_range(*, start: float, stop: float, step: float) -> list[float]:
    count = max(0, int(math.ceil((stop - start) / step)))
    return [start + index * step for index in range(count)]


def _normalized_angle(angle: float) -> float:
    normalized = angle % 360.0
    return 0.0 if math.isclose(normalized, 360.0) else normalized


def _polyline_length(points: list[Point]) -> float:
    return sum(
        math.dist(first, second)
        for first, second in zip(points, points[1:])
    )


def _rounded(value: float) -> float:
    rounded = round(value, 6)
    return 0.0 if rounded == -0.0 else rounded
