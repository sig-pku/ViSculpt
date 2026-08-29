"""Deterministic anchor-aware Brush Size resolution for Drag gestures."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from visculpt.bridge import JsonValue


class DragBrushSizeResolutionError(ValueError):
    """Raised when a bound Drag target cannot resolve a Brush Size."""


@dataclass(frozen=True, slots=True)
class DragBrushSizeResolution:
    """Auditable anchor-relative Brush Size expressed in Blender UI pixels."""

    preliminary_brush_size: int
    anchor_aware_brush_size: int
    brush_size: int
    foreground_pixels: int
    extent_percentile: float
    anchor_extent_pixels: float
    support_radius_ratio: float
    support_radius_pixels: float
    coordinate_scale_x: float
    coordinate_scale_y: float

    def as_payload(self) -> dict[str, JsonValue]:
        """Return the resolution evidence stored in LangGraph State."""
        selected = (
            "ANCHOR_AWARE_SIZE_DOMINATES"
            if self.anchor_aware_brush_size
            > self.preliminary_brush_size
            else "MASK_RELATIVE_SIZE_DOMINATES"
        )
        return {
            "algorithm": {
                "name": "anchor-relative-drag-brush-size/v1",
                "deterministic": True,
                "size_semantics": "BLENDER_UI_PIXEL_DIAMETER",
                "extent_statistic": "ANCHOR_DISTANCE_PERCENTILE",
            },
            "preliminary_brush_size": self.preliminary_brush_size,
            "anchor_aware_brush_size": self.anchor_aware_brush_size,
            "brush_size": self.brush_size,
            "foreground_pixels": self.foreground_pixels,
            "extent_percentile": round(self.extent_percentile, 6),
            "anchor_extent_pixels": round(
                self.anchor_extent_pixels,
                6,
            ),
            "support_radius_ratio": round(
                self.support_radius_ratio,
                6,
            ),
            "support_radius_pixels": round(
                self.support_radius_pixels,
                6,
            ),
            "coordinate_scale": {
                "x": round(self.coordinate_scale_x, 9),
                "y": round(self.coordinate_scale_y, 9),
            },
            "reason_codes": [
                "ANCHOR_AWARE_DRAG_SIZE",
                selected,
            ],
        }


def resolve_drag_brush_size(
    *,
    component_mask_path: str | Path,
    anchor_x: int,
    anchor_y: int,
    preliminary_brush_size: int,
    support_radius_ratio: float,
    extent_percentile: float,
    maximum_brush_size: int,
    screenshot_metadata: Mapping[str, object],
) -> DragBrushSizeResolution:
    """Resolve Drag Size from the robust target extent around its anchor."""
    _validate_parameters(
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        preliminary_brush_size=preliminary_brush_size,
        support_radius_ratio=support_radius_ratio,
        extent_percentile=extent_percentile,
        maximum_brush_size=maximum_brush_size,
    )
    mask = _load_mask(component_mask_path)
    image_width = _positive_integer(screenshot_metadata, "width")
    image_height = _positive_integer(screenshot_metadata, "height")
    if mask.shape != (image_height, image_width):
        raise DragBrushSizeResolutionError(
            "Drag component mask dimensions do not match the screenshot"
        )
    if not 0 <= anchor_x < image_width or not 0 <= anchor_y < image_height:
        raise DragBrushSizeResolutionError(
            "Drag anchor is outside the component mask"
        )
    if not bool(mask[anchor_y, anchor_x]):
        raise DragBrushSizeResolutionError(
            "Drag anchor is outside the bound target component"
        )

    region = screenshot_metadata.get("region")
    if not isinstance(region, Mapping):
        raise DragBrushSizeResolutionError(
            "Screenshot metadata is missing its VIEW_3D region"
        )
    region_width = _positive_integer(region, "width")
    region_height = _positive_integer(region, "height")
    scale_x = image_width / region_width
    scale_y = image_height / region_height

    y_values, x_values = np.nonzero(mask)
    if len(x_values) == 0:
        raise DragBrushSizeResolutionError(
            "Drag component mask contains no foreground"
        )
    delta_x = (x_values.astype(np.float64) - anchor_x) / scale_x
    delta_y = (y_values.astype(np.float64) - anchor_y) / scale_y
    distances = np.hypot(delta_x, delta_y)
    anchor_extent = float(
        np.percentile(distances, extent_percentile)
    )
    if not math.isfinite(anchor_extent) or anchor_extent < 0.0:
        raise DragBrushSizeResolutionError(
            "Drag anchor extent is not finite"
        )
    support_radius = anchor_extent * support_radius_ratio
    anchor_aware_size = max(1, round(2.0 * support_radius))
    final_size = min(
        maximum_brush_size,
        max(preliminary_brush_size, anchor_aware_size),
    )
    return DragBrushSizeResolution(
        preliminary_brush_size=preliminary_brush_size,
        anchor_aware_brush_size=anchor_aware_size,
        brush_size=final_size,
        foreground_pixels=int(len(x_values)),
        extent_percentile=extent_percentile,
        anchor_extent_pixels=anchor_extent,
        support_radius_ratio=support_radius_ratio,
        support_radius_pixels=support_radius,
        coordinate_scale_x=scale_x,
        coordinate_scale_y=scale_y,
    )


def _load_mask(path_value: str | Path) -> np.ndarray:
    path = Path(
        os.path.expandvars(os.path.expanduser(str(path_value)))
    ).resolve()
    if not path.is_file():
        raise DragBrushSizeResolutionError(
            f"Drag component mask is not a file: {path}"
        )
    try:
        with Image.open(path) as image:
            grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise DragBrushSizeResolutionError(
            f"Cannot read Drag component mask: {path}"
        ) from error
    return grayscale >= 128


def _positive_integer(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DragBrushSizeResolutionError(
            f"Screenshot metadata field {key} must be a positive integer"
        )
    return value


def _validate_parameters(
    *,
    anchor_x: int,
    anchor_y: int,
    preliminary_brush_size: int,
    support_radius_ratio: float,
    extent_percentile: float,
    maximum_brush_size: int,
) -> None:
    for name, value in (("anchor_x", anchor_x), ("anchor_y", anchor_y)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise DragBrushSizeResolutionError(
                f"{name} must be an integer"
            )
    for name, value in (
        ("preliminary_brush_size", preliminary_brush_size),
        ("maximum_brush_size", maximum_brush_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise DragBrushSizeResolutionError(
                f"{name} must be a positive integer"
            )
    if preliminary_brush_size > maximum_brush_size:
        raise DragBrushSizeResolutionError(
            "preliminary_brush_size cannot exceed maximum_brush_size"
        )
    if (
        isinstance(support_radius_ratio, bool)
        or not isinstance(support_radius_ratio, (int, float))
        or not math.isfinite(float(support_radius_ratio))
        or not 0.0 < float(support_radius_ratio) <= 1.0
    ):
        raise DragBrushSizeResolutionError(
            "support_radius_ratio must be in (0, 1]"
        )
    if (
        isinstance(extent_percentile, bool)
        or not isinstance(extent_percentile, (int, float))
        or not math.isfinite(float(extent_percentile))
        or not 0.0 < float(extent_percentile) <= 100.0
    ):
        raise DragBrushSizeResolutionError(
            "extent_percentile must be in (0, 100]"
        )
