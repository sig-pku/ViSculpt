"""Deterministic target-mask correction for Drag mouse-down points."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, UnidentifiedImageError

from visculpt.bridge import JsonValue

type MaskArray = NDArray[np.uint8]


class DragAnchorCorrectionError(ValueError):
    """Raised when a target mask cannot provide a safe Drag anchor."""


@dataclass(frozen=True, slots=True)
class DragAnchorCorrectionResult:
    """Corrected anchor and deterministic distance-transform diagnostics."""

    input_x: int
    input_y: int
    corrected_x: int
    corrected_y: int
    correction_applied: bool
    input_inside_target: bool
    shift_pixels: float
    brush_radius_pixels: float
    minimum_margin_pixels: float
    brush_radius_ratio: float
    component_depth_ratio: float
    requested_margin_pixels: float
    safe_margin_pixels: float
    input_interior_distance_pixels: float
    corrected_interior_distance_pixels: float
    component_label: int
    component_area_pixels: int
    component_max_interior_distance_pixels: float

    def as_payload(self) -> dict[str, JsonValue]:
        """Return JSON-safe state metadata for the corrected anchor."""
        return {
            "algorithm": {
                "name": "connected-component-distance-transform/v1",
                "deterministic": True,
                "mask_semantics": "part_to_be_changed",
                "safe_margin_formula": (
                    "min(max(minimum_margin_pixels, brush_radius_pixels * "
                    "brush_radius_ratio), "
                    "component_max_interior_distance_pixels * "
                    "component_depth_ratio)"
                ),
            },
            "input_coordinate": {"x": self.input_x, "y": self.input_y},
            "coordinate": {
                "x": self.corrected_x,
                "y": self.corrected_y,
            },
            "correction_applied": self.correction_applied,
            "input_inside_target": self.input_inside_target,
            "shift_pixels": round(self.shift_pixels, 6),
            "brush_radius_pixels": round(self.brush_radius_pixels, 6),
            "parameters": {
                "minimum_margin_pixels": round(
                    self.minimum_margin_pixels,
                    6,
                ),
                "brush_radius_ratio": round(self.brush_radius_ratio, 6),
                "component_depth_ratio": round(
                    self.component_depth_ratio,
                    6,
                ),
            },
            "requested_margin_pixels": round(
                self.requested_margin_pixels,
                6,
            ),
            "safe_margin_pixels": round(self.safe_margin_pixels, 6),
            "input_interior_distance_pixels": round(
                self.input_interior_distance_pixels,
                6,
            ),
            "corrected_interior_distance_pixels": round(
                self.corrected_interior_distance_pixels,
                6,
            ),
            "component": {
                "label": self.component_label,
                "area_pixels": self.component_area_pixels,
                "max_interior_distance_pixels": round(
                    self.component_max_interior_distance_pixels,
                    6,
                ),
            },
        }


def correct_drag_anchor(
    *,
    cleaned_mask_path: str | Path,
    start_x: int,
    start_y: int,
    brush_size: int,
    minimum_margin_pixels: float,
    brush_radius_ratio: float,
    component_depth_ratio: float,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> DragAnchorCorrectionResult:
    """Move a Drag point to the nearest safe pixel in its target component."""
    _validate_parameters(
        start_x=start_x,
        start_y=start_y,
        brush_size=brush_size,
        minimum_margin_pixels=minimum_margin_pixels,
        brush_radius_ratio=brush_radius_ratio,
        component_depth_ratio=component_depth_ratio,
        expected_width=expected_width,
        expected_height=expected_height,
    )
    mask_path = _resolve_mask(cleaned_mask_path)
    binary = _load_binary_mask(mask_path)
    height, width = binary.shape
    if expected_width is not None and width != expected_width:
        raise DragAnchorCorrectionError(
            "Drag target mask width does not match the screenshot"
        )
    if expected_height is not None and height != expected_height:
        raise DragAnchorCorrectionError(
            "Drag target mask height does not match the screenshot"
        )
    if not 0 <= start_x < width or not 0 <= start_y < height:
        raise DragAnchorCorrectionError(
            "QuadLoc coordinate is outside the Drag target mask image"
        )
    if not np.any(binary):
        raise DragAnchorCorrectionError(
            "Drag target mask contains no foreground"
        )

    try:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    except cv2.error as error:
        raise DragAnchorCorrectionError(
            f"OpenCV connected-component analysis failed: {error}"
        ) from error
    if count <= 1:
        raise DragAnchorCorrectionError(
            "Drag target mask contains no connected foreground component"
        )

    input_inside_target = bool(binary[start_y, start_x])
    component_label = (
        int(labels[start_y, start_x])
        if input_inside_target
        else _nearest_foreground_label(
            labels,
            start_x=start_x,
            start_y=start_y,
        )
    )
    component = np.where(labels == component_label, 255, 0).astype(np.uint8)
    distance = _interior_distance(component)
    maximum_distance = float(np.max(distance))
    if not math.isfinite(maximum_distance) or maximum_distance <= 0.0:
        raise DragAnchorCorrectionError(
            "Selected Drag target component has no measurable interior"
        )
    if maximum_distance < minimum_margin_pixels:
        raise DragAnchorCorrectionError(
            "Selected Drag target component is too thin for the minimum "
            "safe anchor margin"
        )

    brush_radius = float(brush_size) * 0.5
    requested_margin = max(
        minimum_margin_pixels,
        brush_radius * brush_radius_ratio,
    )
    safe_margin = min(
        requested_margin,
        maximum_distance * component_depth_ratio,
    )
    safe_core = component.astype(bool) & (distance >= safe_margin)
    if not np.any(safe_core):
        raise DragAnchorCorrectionError(
            "Selected Drag target component has no safe interior anchor"
        )

    input_distance = (
        float(distance[start_y, start_x]) if input_inside_target else 0.0
    )
    if bool(safe_core[start_y, start_x]):
        corrected_x, corrected_y = start_x, start_y
    else:
        corrected_x, corrected_y = _nearest_pixel(
            safe_core,
            start_x=start_x,
            start_y=start_y,
        )
    corrected_distance = float(distance[corrected_y, corrected_x])
    shift = math.hypot(corrected_x - start_x, corrected_y - start_y)

    return DragAnchorCorrectionResult(
        input_x=start_x,
        input_y=start_y,
        corrected_x=corrected_x,
        corrected_y=corrected_y,
        correction_applied=corrected_x != start_x or corrected_y != start_y,
        input_inside_target=input_inside_target,
        shift_pixels=shift,
        brush_radius_pixels=brush_radius,
        minimum_margin_pixels=minimum_margin_pixels,
        brush_radius_ratio=brush_radius_ratio,
        component_depth_ratio=component_depth_ratio,
        requested_margin_pixels=requested_margin,
        safe_margin_pixels=safe_margin,
        input_interior_distance_pixels=input_distance,
        corrected_interior_distance_pixels=corrected_distance,
        component_label=component_label,
        component_area_pixels=int(stats[component_label, cv2.CC_STAT_AREA]),
        component_max_interior_distance_pixels=maximum_distance,
    )


def _interior_distance(component: MaskArray) -> NDArray[np.float32]:
    # Zero padding preserves true boundary distances for edge-touching masks.
    padded = np.pad(component, 1, mode="constant", constant_values=0)
    try:
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    except cv2.error as error:
        raise DragAnchorCorrectionError(
            f"OpenCV distance transform failed: {error}"
        ) from error
    return distance[1:-1, 1:-1]


def _nearest_foreground_label(
    labels: NDArray[np.int32],
    *,
    start_x: int,
    start_y: int,
) -> int:
    foreground = labels > 0
    x, y = _nearest_pixel(
        foreground,
        start_x=start_x,
        start_y=start_y,
    )
    return int(labels[y, x])


def _nearest_pixel(
    candidates: NDArray[np.bool_],
    *,
    start_x: int,
    start_y: int,
) -> tuple[int, int]:
    y_values, x_values = np.nonzero(candidates)
    if len(x_values) == 0:
        raise DragAnchorCorrectionError("No candidate Drag anchor exists")
    delta_x = x_values.astype(np.int64) - start_x
    delta_y = y_values.astype(np.int64) - start_y
    squared_distance = delta_x * delta_x + delta_y * delta_y
    index = int(np.argmin(squared_distance))
    return int(x_values[index]), int(y_values[index])


def _load_binary_mask(path: Path) -> MaskArray:
    try:
        with Image.open(path) as image:
            grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise DragAnchorCorrectionError(
            f"Cannot read Drag target mask: {path}"
        ) from error
    return np.where(grayscale >= 128, 255, 0).astype(np.uint8)


def _resolve_mask(value: str | Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()
    if not path.is_file():
        raise DragAnchorCorrectionError(
            f"Drag target mask is not a file: {path}"
        )
    return path


def _validate_parameters(
    *,
    start_x: int,
    start_y: int,
    brush_size: int,
    minimum_margin_pixels: float,
    brush_radius_ratio: float,
    component_depth_ratio: float,
    expected_width: int | None,
    expected_height: int | None,
) -> None:
    for name, value in (("start_x", start_x), ("start_y", start_y)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise DragAnchorCorrectionError(f"{name} must be an integer")
    if isinstance(brush_size, bool) or not 1 <= brush_size <= 10_000:
        raise DragAnchorCorrectionError(
            "brush_size must be between 1 and 10000"
        )
    values = (
        minimum_margin_pixels,
        brush_radius_ratio,
        component_depth_ratio,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise DragAnchorCorrectionError(
            "Safe Drag Anchor parameters must be finite"
        )
    if minimum_margin_pixels < 0.0:
        raise DragAnchorCorrectionError(
            "minimum_margin_pixels cannot be negative"
        )
    if not 0.0 <= brush_radius_ratio <= 1.0:
        raise DragAnchorCorrectionError(
            "brush_radius_ratio must be between 0 and 1"
        )
    if not 0.0 < component_depth_ratio <= 1.0:
        raise DragAnchorCorrectionError(
            "component_depth_ratio must be between 0 and 1"
        )
    for name, value in (
        ("expected_width", expected_width),
        ("expected_height", expected_height),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            raise DragAnchorCorrectionError(
                f"{name} must be a positive integer when provided"
            )
