"""One-sided, target-only minimum visible-effect validation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from visculpt.bridge import JsonValue

from .config import MinimumEffectConfig
from .models import EffectIntensity

_BRUSH_RADIUS_FACTOR = 0.5


class MinimumEffectError(ValueError):
    """Raised when minimum-effect inputs violate the evaluation contract."""


class MinimumEffectVerdict(StrEnum):
    """One-sided visibility outcomes."""

    NO_EFFECT = "NO_EFFECT"
    TOO_SUBTLE = "TOO_SUBTLE"
    VISIBLE = "VISIBLE"
    INCONCLUSIVE = "INCONCLUSIVE"


class MinimumEffectBaseline(BaseModel):
    """Validated baseline context prepared before Sculpt execution."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    ready: bool
    capture_attempt_count: int = Field(default=1, ge=1, le=5)
    baseline_a_path: str
    baseline_b_path: str
    evaluation_mask_path: str | None
    before_roi_path: str | None
    noise_threshold: float = Field(ge=0.0, le=255.0)
    evaluation_pixel_count: int = Field(ge=0)
    roi_xyxy: list[int] | None
    reason_codes: list[str]


class MinimumEffectMetrics(BaseModel):
    """Only the approved target-region image-change measurements."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    noise_threshold: float = Field(ge=0.0, le=255.0)
    target_mean_abs_diff: float = Field(ge=0.0, le=255.0)
    target_changed_fraction: float = Field(ge=0.0, le=1.0)
    evaluation_pixel_count: int = Field(ge=0)


class MinimumEffectRequirements(BaseModel):
    """Configured lower bounds applied to one intent intensity."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    minimum_mean_abs_diff: float = Field(ge=0.0, le=255.0)
    minimum_changed_fraction: float = Field(ge=0.0, le=1.0)


class MinimumEffectResult(BaseModel):
    """Persisted one-sided effect-gate result."""

    model_config = ConfigDict(extra="forbid")

    verdict: MinimumEffectVerdict
    metrics: MinimumEffectMetrics
    requirements: MinimumEffectRequirements
    reason_codes: list[str]
    artifact_paths: dict[str, str]


def prepare_minimum_effect_baseline(
    *,
    baseline_a_path: str | Path,
    baseline_b_path: str | Path,
    cleaned_mask_path: str | Path,
    stroke_plan: Mapping[str, JsonValue],
    output_dir: str | Path,
    config: MinimumEffectConfig,
    evaluation_strategy: str = "stroke_intersection",
) -> MinimumEffectBaseline:
    """Validate two pre-execution captures and persist the target mask."""
    baseline_a = _load_rgb(baseline_a_path, label="Baseline A")
    baseline_b = _load_rgb(baseline_b_path, label="Baseline B")
    if baseline_a.shape != baseline_b.shape:
        return _inconclusive_baseline(
            baseline_a_path,
            baseline_b_path,
            "BASELINE_DIMENSIONS_MISMATCH",
        )
    semantic_mask = _load_mask(cleaned_mask_path)
    if semantic_mask.shape != baseline_a.shape[:2]:
        return _inconclusive_baseline(
            baseline_a_path,
            baseline_b_path,
            "MASK_DIMENSIONS_MISMATCH",
        )

    if evaluation_strategy == "stroke_intersection":
        footprint = _stroke_footprint(
            stroke_plan,
            image_width=baseline_a.shape[1],
            image_height=baseline_a.shape[0],
        )
        evaluation_mask = np.logical_and(semantic_mask, footprint)
    elif evaluation_strategy == "semantic_mask":
        evaluation_mask = semantic_mask
    else:
        raise MinimumEffectError(
            "evaluation_strategy must be stroke_intersection or "
            "semantic_mask"
        )
    evaluation_pixels = int(np.count_nonzero(evaluation_mask))
    if evaluation_pixels < config.minimum_evaluation_pixels:
        return _inconclusive_baseline(
            baseline_a_path,
            baseline_b_path,
            "EVALUATION_MASK_TOO_SMALL",
            evaluation_pixel_count=evaluation_pixels,
        )

    comparison_a = _comparison_image(baseline_a, config=config)
    comparison_b = _comparison_image(baseline_b, config=config)
    baseline_delta = _pixel_delta(comparison_a, comparison_b)
    noise_threshold = max(
        config.pixel_delta_floor,
        float(
            np.percentile(
                baseline_delta[evaluation_mask],
                config.baseline_noise_percentile,
            )
        ),
    )
    if noise_threshold > config.maximum_baseline_noise:
        return _inconclusive_baseline(
            baseline_a_path,
            baseline_b_path,
            "BASELINE_NOISE_TOO_HIGH",
            noise_threshold=noise_threshold,
            evaluation_pixel_count=evaluation_pixels,
        )

    directory = _ensure_directory(output_dir)
    mask_path = directory / "effect-evaluation-mask.png"
    _write_image(
        mask_path,
        np.where(evaluation_mask, 255, 0).astype(np.uint8),
    )
    roi = _padded_roi(
        evaluation_mask,
        padding_ratio=config.roi_padding_ratio,
    )
    x0, y0, x1, y1 = roi
    before_roi_path = directory / "effect-before-roi.png"
    _write_image(
        before_roi_path,
        baseline_a[y0:y1, x0:x1],
    )
    return MinimumEffectBaseline(
        ready=True,
        baseline_a_path=str(_resolved_path(baseline_a_path)),
        baseline_b_path=str(_resolved_path(baseline_b_path)),
        evaluation_mask_path=str(mask_path),
        before_roi_path=str(before_roi_path),
        noise_threshold=round(noise_threshold, 6),
        evaluation_pixel_count=evaluation_pixels,
        roi_xyxy=[x0, y0, x1, y1],
        reason_codes=[],
    )


def evaluate_minimum_effect(
    *,
    baseline: MinimumEffectBaseline | Mapping[str, object],
    after_path: str | Path,
    effect_intensity: EffectIntensity | str,
    output_dir: str | Path,
    config: MinimumEffectConfig,
) -> MinimumEffectResult:
    """Measure only target-region pixel change against a lower bound."""
    prepared = (
        baseline
        if isinstance(baseline, MinimumEffectBaseline)
        else MinimumEffectBaseline.model_validate(baseline)
    )
    intensity = EffectIntensity(effect_intensity)
    requirements = _requirements(config, intensity)
    if not prepared.ready or prepared.evaluation_mask_path is None:
        return MinimumEffectResult(
            verdict=MinimumEffectVerdict.INCONCLUSIVE,
            metrics=MinimumEffectMetrics(
                noise_threshold=prepared.noise_threshold,
                target_mean_abs_diff=0.0,
                target_changed_fraction=0.0,
                evaluation_pixel_count=prepared.evaluation_pixel_count,
            ),
            requirements=requirements,
            reason_codes=[*prepared.reason_codes],
            artifact_paths={},
        )

    baseline_a = _load_rgb(prepared.baseline_a_path, label="Baseline A")
    baseline_b = _load_rgb(prepared.baseline_b_path, label="Baseline B")
    after = _load_rgb(after_path, label="After")
    evaluation_mask = _load_mask(prepared.evaluation_mask_path)
    if (
        baseline_a.shape != baseline_b.shape
        or baseline_a.shape != after.shape
        or evaluation_mask.shape != after.shape[:2]
    ):
        return MinimumEffectResult(
            verdict=MinimumEffectVerdict.INCONCLUSIVE,
            metrics=MinimumEffectMetrics(
                noise_threshold=prepared.noise_threshold,
                target_mean_abs_diff=0.0,
                target_changed_fraction=0.0,
                evaluation_pixel_count=prepared.evaluation_pixel_count,
            ),
            requirements=requirements,
            reason_codes=["EVALUATION_DIMENSIONS_MISMATCH"],
            artifact_paths={},
        )

    comparison_a = _comparison_image(baseline_a, config=config)
    comparison_b = _comparison_image(baseline_b, config=config)
    comparison_after = _comparison_image(after, config=config)
    reference = (comparison_a + comparison_b) / 2.0
    delta = np.abs(comparison_after - reference).mean(axis=2)
    target_delta = delta[evaluation_mask]
    mean_diff = float(target_delta.mean())
    changed_fraction = float(
        np.count_nonzero(target_delta > prepared.noise_threshold)
        / target_delta.size
    )

    if (
        mean_diff <= prepared.noise_threshold
        and changed_fraction <= config.no_effect_fraction
    ):
        verdict = MinimumEffectVerdict.NO_EFFECT
        reasons = ["TARGET_CHANGE_NOT_ABOVE_BASELINE_NOISE"]
    elif (
        mean_diff >= requirements.minimum_mean_abs_diff
        and changed_fraction >= requirements.minimum_changed_fraction
    ):
        verdict = MinimumEffectVerdict.VISIBLE
        reasons = ["MINIMUM_VISIBLE_EFFECT_REACHED"]
    else:
        verdict = MinimumEffectVerdict.TOO_SUBTLE
        reasons = []
        if mean_diff < requirements.minimum_mean_abs_diff:
            reasons.append("MEAN_CHANGE_BELOW_MINIMUM")
        if changed_fraction < requirements.minimum_changed_fraction:
            reasons.append("CHANGED_FRACTION_BELOW_MINIMUM")

    directory = _ensure_directory(output_dir)
    roi = prepared.roi_xyxy
    if roi is None:
        raise MinimumEffectError("prepared baseline is missing roi_xyxy")
    x0, y0, x1, y1 = roi
    after_roi_path = directory / "effect-after-roi.png"
    _write_image(after_roi_path, after[y0:y1, x0:x1])
    heatmap = np.zeros((*evaluation_mask.shape, 3), dtype=np.uint8)
    scaled = np.clip(delta * 12.0, 0.0, 255.0).astype(np.uint8)
    colored = cv2.cvtColor(
        cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO),
        cv2.COLOR_BGR2RGB,
    )
    heatmap[evaluation_mask] = colored[evaluation_mask]
    heatmap_path = directory / "effect-difference-heatmap.png"
    _write_image(heatmap_path, heatmap)

    artifacts = {
        "evaluation_mask_path": prepared.evaluation_mask_path,
        "before_roi_path": prepared.before_roi_path or "",
        "after_roi_path": str(after_roi_path),
        "difference_heatmap_path": str(heatmap_path),
    }
    return MinimumEffectResult(
        verdict=verdict,
        metrics=MinimumEffectMetrics(
            noise_threshold=prepared.noise_threshold,
            target_mean_abs_diff=round(mean_diff, 6),
            target_changed_fraction=round(changed_fraction, 6),
            evaluation_pixel_count=int(target_delta.size),
        ),
        requirements=requirements,
        reason_codes=reasons,
        artifact_paths=artifacts,
    )


def unmeasured_minimum_effect_result(
    *,
    verdict: MinimumEffectVerdict,
    effect_intensity: EffectIntensity | str,
    config: MinimumEffectConfig,
    baseline: MinimumEffectBaseline | Mapping[str, object] | None = None,
    reason_codes: list[str],
) -> MinimumEffectResult:
    """Create a typed result when deterministic measurement cannot run."""
    if verdict not in {
        MinimumEffectVerdict.VISIBLE,
        MinimumEffectVerdict.INCONCLUSIVE,
    }:
        raise MinimumEffectError(
            "unmeasured verdict must be VISIBLE or INCONCLUSIVE"
        )
    prepared = (
        None
        if baseline is None
        else baseline
        if isinstance(baseline, MinimumEffectBaseline)
        else MinimumEffectBaseline.model_validate(baseline)
    )
    return MinimumEffectResult(
        verdict=verdict,
        metrics=MinimumEffectMetrics(
            noise_threshold=(
                prepared.noise_threshold if prepared is not None else 0.0
            ),
            target_mean_abs_diff=0.0,
            target_changed_fraction=0.0,
            evaluation_pixel_count=(
                prepared.evaluation_pixel_count
                if prepared is not None
                else 0
            ),
        ),
        requirements=_requirements(
            config,
            EffectIntensity(effect_intensity),
        ),
        reason_codes=reason_codes,
        artifact_paths={},
    )


def retry_directive_for_minimum_effect(
    result: MinimumEffectResult,
    *,
    previous: Mapping[str, JsonValue] | None,
    retry_multiplier: float,
) -> dict[str, JsonValue]:
    """Increase only the deficient execution dimensions."""
    prior = {} if previous is None else previous
    dose = _number(prior.get("dose_multiplier"), default=1.0)
    size = _number(prior.get("size_multiplier"), default=1.0)
    pass_count = int(_number(prior.get("pass_count"), default=1.0))
    mean_met = (
        result.metrics.target_mean_abs_diff
        >= result.requirements.minimum_mean_abs_diff
    )
    fraction_met = (
        result.metrics.target_changed_fraction
        >= result.requirements.minimum_changed_fraction
    )
    if not mean_met:
        dose *= retry_multiplier
    if not fraction_met:
        size *= retry_multiplier
    if not mean_met and not fraction_met and dose > retry_multiplier:
        pass_count += 1
    return {
        "cause": result.verdict.value,
        "dose_multiplier": round(dose, 6),
        "size_multiplier": round(size, 6),
        "pass_count": pass_count,
    }


def _stroke_footprint(
    stroke_plan: Mapping[str, JsonValue],
    *,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    mapping = stroke_plan.get("coordinate_mapping")
    if not isinstance(mapping, Mapping):
        raise MinimumEffectError("stroke plan is missing coordinate_mapping")
    target = mapping.get("target_region")
    if not isinstance(target, Mapping):
        raise MinimumEffectError("stroke plan is missing target_region")
    region_width = _positive_int(target, "width")
    region_height = _positive_int(target, "height")
    footprint = np.zeros((region_height, region_width), dtype=np.uint8)
    calls = stroke_plan.get("operator_calls")
    if not isinstance(calls, list) or not calls:
        raise MinimumEffectError("stroke plan has no operator_calls")
    for call in calls:
        if not isinstance(call, Mapping):
            raise MinimumEffectError("stroke plan contains an invalid call")
        kwargs = call.get("kwargs")
        if not isinstance(kwargs, Mapping):
            raise MinimumEffectError("operator call is missing kwargs")
        elements = kwargs.get("stroke")
        if not isinstance(elements, list) or not elements:
            continue
        previous: tuple[int, int] | None = None
        previous_radius = 1
        for element in elements:
            if not isinstance(element, Mapping):
                continue
            mouse = element.get("mouse_event")
            size = element.get("size")
            if (
                not isinstance(mouse, list)
                or len(mouse) != 2
                or not all(isinstance(value, (int, float)) for value in mouse)
                or not isinstance(size, (int, float))
            ):
                continue
            x = int(round(float(mouse[0])))
            y = region_height - 1 - int(round(float(mouse[1])))
            point = (
                min(max(x, 0), region_width - 1),
                min(max(y, 0), region_height - 1),
            )
            # Blender View Brush Size is a diameter; the footprint uses radius.
            radius = max(1, round(float(size) * _BRUSH_RADIUS_FACTOR))
            cv2.circle(footprint, point, radius, 255, thickness=-1)
            if previous is not None:
                cv2.line(
                    footprint,
                    previous,
                    point,
                    255,
                    thickness=max(previous_radius, radius) * 2,
                )
            previous = point
            previous_radius = radius
    if not np.any(footprint):
        raise MinimumEffectError("stroke plan produced an empty footprint")
    if footprint.shape != (image_height, image_width):
        footprint = cv2.resize(
            footprint,
            (image_width, image_height),
            interpolation=cv2.INTER_NEAREST,
        )
    return footprint >= 128


def _requirements(
    config: MinimumEffectConfig,
    intensity: EffectIntensity,
) -> MinimumEffectRequirements:
    values = {
        EffectIntensity.SUBTLE: (
            config.subtle_minimum_mean_abs_diff,
            config.subtle_minimum_changed_fraction,
        ),
        EffectIntensity.MEDIUM_VISIBLE: (
            config.medium_minimum_mean_abs_diff,
            config.medium_minimum_changed_fraction,
        ),
        EffectIntensity.STRONG: (
            config.strong_minimum_mean_abs_diff,
            config.strong_minimum_changed_fraction,
        ),
    }[intensity]
    return MinimumEffectRequirements(
        minimum_mean_abs_diff=values[0],
        minimum_changed_fraction=values[1],
    )


def _inconclusive_baseline(
    baseline_a_path: str | Path,
    baseline_b_path: str | Path,
    reason: str,
    *,
    noise_threshold: float = 0.0,
    evaluation_pixel_count: int = 0,
) -> MinimumEffectBaseline:
    return MinimumEffectBaseline(
        ready=False,
        baseline_a_path=str(_resolved_path(baseline_a_path)),
        baseline_b_path=str(_resolved_path(baseline_b_path)),
        evaluation_mask_path=None,
        before_roi_path=None,
        noise_threshold=round(noise_threshold, 6),
        evaluation_pixel_count=evaluation_pixel_count,
        roi_xyxy=None,
        reason_codes=[reason],
    )


def _padded_roi(
    mask: np.ndarray,
    *,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    y_values, x_values = np.nonzero(mask)
    x0 = int(x_values.min())
    x1 = int(x_values.max()) + 1
    y0 = int(y_values.min())
    y1 = int(y_values.max()) + 1
    padding = round(max(x1 - x0, y1 - y0) * padding_ratio)
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(mask.shape[1], x1 + padding),
        min(mask.shape[0], y1 + padding),
    )


def _pixel_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(
        left.astype(np.float32) - right.astype(np.float32)
    ).mean(axis=2)


def _comparison_image(
    image: np.ndarray,
    *,
    config: MinimumEffectConfig,
) -> np.ndarray:
    """Suppress subpixel viewport redraw noise before effect measurement."""
    comparison = image.astype(np.float32)
    kernel = config.comparison_blur_kernel_size
    if kernel > 1:
        comparison = cv2.GaussianBlur(
            comparison,
            (kernel, kernel),
            sigmaX=0.0,
            sigmaY=0.0,
        )
    return comparison


def _load_rgb(path_value: str | Path, *, label: str) -> np.ndarray:
    path = _resolved_path(path_value)
    if not path.is_file():
        raise MinimumEffectError(f"{label} is not a file: {path}")
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise MinimumEffectError(f"cannot read {label}: {path}") from error


def _load_mask(path_value: str | Path) -> np.ndarray:
    path = _resolved_path(path_value)
    if not path.is_file():
        raise MinimumEffectError(f"mask is not a file: {path}")
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.uint8) >= 128
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise MinimumEffectError(f"cannot read mask: {path}") from error


def _ensure_directory(value: str | Path) -> Path:
    path = _resolved_path(value)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise MinimumEffectError(f"cannot create output directory: {path}") from error
    return path


def _write_image(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        Image.fromarray(value).save(temporary, format="PNG")
        os.replace(temporary, path)
    except OSError as error:
        raise MinimumEffectError(f"cannot write evaluation image: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _resolved_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def _positive_int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MinimumEffectError(f"{key} must be a positive integer")
    return value


def _number(value: JsonValue | None, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MinimumEffectError("retry directive must contain numbers")
    return float(value)
