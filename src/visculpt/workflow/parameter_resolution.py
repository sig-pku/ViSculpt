"""Resolve semantic Sculpt intent into mask-relative execution settings."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from visculpt.bridge import JsonValue

from .config import (
    ParameterResolutionConfig,
    SculptOperationDefaultsConfig,
)
from .models import BrushScale, SculptIntent


class ParameterResolutionError(ValueError):
    """Raised when a cleaned mask cannot produce safe Sculpt settings."""


class ResolvedSculptSettings(BaseModel):
    """Concrete settings sent to Blender and the stroke planner."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    sculpt_brush: str
    brush_size: int = Field(ge=1, le=10_000)
    brush_strength: float = Field(ge=0.0, le=1.0)
    brush_direction: str | None
    dyntopo_enabled: bool
    dyntopo_detail_size: float
    use_unified_size: bool = False
    use_unified_strength: bool = False
    use_size_pressure: bool = False
    use_strength_pressure: bool = False


class StrokePolicy(BaseModel):
    """Deterministic execution policy resolved for one attempt."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    pass_count: int = Field(ge=1, le=20)
    dose_multiplier: float = Field(ge=1.0)


class ResolutionContext(BaseModel):
    """Auditable scale calculation used to resolve Brush Size."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    mask_width: int = Field(gt=0)
    mask_height: int = Field(gt=0)
    foreground_pixels: int = Field(gt=0)
    bbox_short_axis_px: int = Field(gt=0)
    equivalent_diameter_px: float = Field(gt=0.0)
    target_scale_px: float = Field(gt=0.0)
    brush_size_ratio: float = Field(gt=0.0)
    size_multiplier: float = Field(gt=0.0)
    reason_codes: list[str]


class ResolvedSculptPlan(BaseModel):
    """Latest execution contract derived after segmentation."""

    model_config = ConfigDict(extra="forbid")

    settings: ResolvedSculptSettings
    stroke_policy: StrokePolicy
    resolution_context: ResolutionContext


def resolve_sculpt_plan(
    *,
    intent: SculptIntent | Mapping[str, object],
    cleaned_mask_path: str | Path,
    screenshot_metadata: Mapping[str, object],
    config: ParameterResolutionConfig,
    operation_defaults: SculptOperationDefaultsConfig,
    retry_directive: Mapping[str, JsonValue] | None = None,
) -> ResolvedSculptPlan:
    """Resolve a semantic intent against the selected-view mask scale."""
    parsed_intent = (
        intent
        if isinstance(intent, SculptIntent)
        else SculptIntent.model_validate(intent)
    )
    mask = _load_mask(cleaned_mask_path)
    logical = _logical_mask(mask, screenshot_metadata)
    y_values, x_values = np.nonzero(logical)
    if not len(x_values):
        raise ParameterResolutionError("cleaned mask has no foreground")
    if parsed_intent.brush_scale is None:
        raise ParameterResolutionError(
            "non-Pose execution requires brush_scale"
        )
    if parsed_intent.brush_strength is None:
        raise ParameterResolutionError(
            "non-Pose execution requires brush_strength"
        )

    width = int(x_values.max() - x_values.min() + 1)
    height = int(y_values.max() - y_values.min() + 1)
    area = int(len(x_values))
    equivalent_diameter = 2.0 * math.sqrt(area / math.pi)
    target_scale = min(float(min(width, height)), equivalent_diameter)
    ratio = {
        BrushScale.LOCAL: config.local_size_ratio,
        BrushScale.REGIONAL: config.regional_size_ratio,
        BrushScale.BROAD: config.broad_size_ratio,
    }[parsed_intent.brush_scale]

    directive = {} if retry_directive is None else retry_directive
    size_multiplier = _positive_directive(
        directive,
        "size_multiplier",
        default=1.0,
    )
    dose_multiplier = _positive_directive(
        directive,
        "dose_multiplier",
        default=1.0,
    )
    requested_pass_count = int(
        round(
            _positive_directive(
                directive,
                "pass_count",
                default=1.0,
            )
        )
    )
    pass_count = min(
        config.maximum_pass_count,
        max(1, requested_pass_count),
    )
    brush_size = min(
        config.maximum_brush_size,
        max(1, round(target_scale * ratio * size_multiplier)),
    )
    strength = min(
        config.maximum_brush_strength,
        parsed_intent.brush_strength * dose_multiplier,
    )
    reason_codes = [
        f"{parsed_intent.brush_scale.value}_MASK_RELATIVE_SIZE",
        f"{parsed_intent.effect_intensity.value}_EFFECT",
    ]
    if size_multiplier > 1.0:
        reason_codes.append("RETRY_INCREASED_SIZE")
    if dose_multiplier > 1.0 or pass_count > 1:
        reason_codes.append("RETRY_INCREASED_DOSE")

    return ResolvedSculptPlan(
        settings=ResolvedSculptSettings(
            sculpt_brush=parsed_intent.sculpt_brush,
            brush_size=brush_size,
            brush_strength=round(strength, 6),
            brush_direction=parsed_intent.brush_direction,
            dyntopo_enabled=operation_defaults.dyntopo_enabled,
            dyntopo_detail_size=operation_defaults.dyntopo_detail_size,
            use_unified_size=False,
            use_unified_strength=False,
            use_size_pressure=False,
            use_strength_pressure=False,
        ),
        stroke_policy=StrokePolicy(
            pass_count=pass_count,
            dose_multiplier=round(dose_multiplier, 6),
        ),
        resolution_context=ResolutionContext(
            mask_width=logical.shape[1],
            mask_height=logical.shape[0],
            foreground_pixels=area,
            bbox_short_axis_px=min(width, height),
            equivalent_diameter_px=round(equivalent_diameter, 6),
            target_scale_px=round(target_scale, 6),
            brush_size_ratio=ratio,
            size_multiplier=round(size_multiplier, 6),
            reason_codes=reason_codes,
        ),
    )


def _load_mask(path_value: str | Path) -> np.ndarray:
    path = Path(
        os.path.expandvars(os.path.expanduser(str(path_value)))
    ).resolve()
    if not path.is_file():
        raise ParameterResolutionError(f"cleaned mask is not a file: {path}")
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.uint8)
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise ParameterResolutionError(
            f"cannot read cleaned mask: {path}"
        ) from error


def _logical_mask(
    mask: np.ndarray,
    metadata: Mapping[str, object],
) -> np.ndarray:
    width = _positive_int(metadata, "width")
    height = _positive_int(metadata, "height")
    if mask.shape != (height, width):
        raise ParameterResolutionError(
            "cleaned mask dimensions do not match screenshot metadata"
        )
    region = metadata.get("region")
    if not isinstance(region, Mapping):
        raise ParameterResolutionError("screenshot metadata is missing region")
    region_width = _positive_int(region, "width")
    region_height = _positive_int(region, "height")
    if mask.shape != (region_height, region_width):
        mask = cv2.resize(
            mask,
            (region_width, region_height),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask >= 128


def _positive_int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ParameterResolutionError(f"{key} must be a positive integer")
    return value


def _positive_directive(
    directive: Mapping[str, JsonValue],
    key: str,
    *,
    default: float,
) -> float:
    value = directive.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterResolutionError(f"retry {key} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ParameterResolutionError(f"retry {key} must be positive")
    return number
