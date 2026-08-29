"""Deterministic QuadLoc recursion around a structured multimodal VLM."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator

from visculpt.bridge import JsonValue


class QuadLocError(RuntimeError):
    """Base class for user-facing QuadLoc failures."""


class QuadLocInputError(QuadLocError):
    """Raised when a direct core call receives invalid input."""


class QuadLocVlmError(QuadLocError):
    """Raised when the VLM cannot provide a usable quadrant decision."""


class QuadLocSearchError(QuadLocError):
    """Raised when the bounded recursive search cannot locate the target."""


class QuadLocSegmentationError(QuadLocError):
    """Raised when model-mask segmentation or correction fails."""


class QuadLocNoModelMaskError(QuadLocSegmentationError):
    """Raised when SAM3 deterministically returns no whole-model mask."""


class QuadLocArtifactError(QuadLocError):
    """Raised when a diagnostic artifact cannot be persisted."""


class QuadLocRegion(StrEnum):
    """Four overlay colors plus the explicit target-absent answer."""

    RED = "RED"
    GREEN = "GREEN"
    BLUE = "BLUE"
    YELLOW = "YELLOW"
    NONE = "NONE"


class QuadLocDecision(BaseModel):
    """Schema-constrained answer returned by the localization VLM."""

    model_config = ConfigDict(extra="forbid")

    region: QuadLocRegion
    reason: str = Field(min_length=1, max_length=800)

    @field_validator("region", mode="before")
    @classmethod
    def normalize_region(cls, value: object) -> object:
        """Accept case-insensitive provider output."""
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        """Reject whitespace-only explanations from providers."""
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("reason must not be empty")
        return normalized


@dataclass(frozen=True, slots=True)
class QuadLocConfig:
    """Deterministic search, correction, and artifact settings."""

    max_depth: int = 5
    max_backtracks: int = 3
    region_expansion_ratio: float = 0.2
    overlay_opacity: float = 0.38
    artifact_root: str = "output/tools/quadloc"
    model_segmentation_prompt: str = "3D model"
    sam3_confidence_threshold: float = 0.5
    sam3_overlay_opacity: float = 0.45
    llm_role: str = "quadloc"
    nearest_point_chunk_size: int = 262_144

    def __post_init__(self) -> None:
        if not 1 <= self.max_depth <= 10:
            raise ValueError("max_depth must be between 1 and 10")
        if not 0 <= self.max_backtracks <= 20:
            raise ValueError("max_backtracks must be between 0 and 20")
        if not 0.0 <= self.region_expansion_ratio <= 1.0:
            raise ValueError(
                "region_expansion_ratio must be between 0 and 1"
            )
        if not 0.0 < self.overlay_opacity < 1.0:
            raise ValueError("overlay_opacity must be between 0 and 1")
        if not self.artifact_root.strip():
            raise ValueError("artifact_root must not be empty")
        if not self.model_segmentation_prompt.strip():
            raise ValueError("model_segmentation_prompt must not be empty")
        if not 0.05 <= self.sam3_confidence_threshold <= 0.95:
            raise ValueError(
                "sam3_confidence_threshold must be between 0.05 and 0.95"
            )
        if not 0.0 <= self.sam3_overlay_opacity <= 1.0:
            raise ValueError(
                "sam3_overlay_opacity must be between 0 and 1"
            )
        if not self.llm_role.strip():
            raise ValueError("llm_role must not be empty")
        if self.nearest_point_chunk_size < 1:
            raise ValueError("nearest_point_chunk_size must be positive")


@dataclass(frozen=True, slots=True)
class QuadLocBox:
    """Half-open rectangle in original screenshot pixel coordinates."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_payload(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class QuadLocResult:
    """Final screenshot coordinate plus inspectable search artifacts."""

    x: int
    y: int
    raw_x: int
    raw_y: int
    corrected: bool
    correction_distance: float
    depth_reached: int
    backtrack_count: int
    vlm_call_count: int
    artifact_dir: str
    trace_path: str
    visualization_path: str
    cleaned_mask_path: str
    sam3_overlay_path: str | None

    def as_payload(self) -> dict[str, JsonValue]:
        """Return a JSON-compatible LangGraph Tool result."""
        return {
            "coordinate": {"x": self.x, "y": self.y},
            "raw_vlm_coordinate": {"x": self.raw_x, "y": self.raw_y},
            "mask_correction": {
                "applied": self.corrected,
                "distance_pixels": self.correction_distance,
                "cleaned_mask_path": self.cleaned_mask_path,
            },
            "search": {
                "depth_reached": self.depth_reached,
                "backtrack_count": self.backtrack_count,
                "vlm_call_count": self.vlm_call_count,
            },
            "artifacts": {
                "directory": self.artifact_dir,
                "trace_path": self.trace_path,
                "final_visualization_path": self.visualization_path,
                "sam3_overlay_path": self.sam3_overlay_path,
            },
        }


class QuadLocCompletion(Protocol):
    """Structural completion result supplied by the workflow LLM adapter."""

    value: BaseModel
    metadata: dict[str, JsonValue]


class QuadLocVlm(Protocol):
    """Minimal multimodal structured-completion surface used by QuadLoc."""

    def complete(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: Sequence[str | Path],
        response_model: type[QuadLocDecision],
    ) -> QuadLocCompletion:
        """Return one quadrant decision."""
        ...


class QuadLocSegmentTool(Protocol):
    """Direct invocation surface of the existing SAM3 LangGraph Tool."""

    def invoke(self, input: dict[str, JsonValue]) -> object:
        """Segment the original screenshot and return a cleaned mask."""
        ...


class QuadLocPromptBuilder(Protocol):
    """English user-prompt builder kept in the centralized prompt module."""

    def __call__(
        self,
        *,
        location_description: str,
        depth: int,
        max_depth: int,
        crop_box: Mapping[str, int],
        available_regions: Sequence[str],
        rejected_regions: Sequence[str],
    ) -> str:
        """Build one recursive quadrant-classification prompt."""
        ...


@dataclass(slots=True)
class _SearchFrame:
    crop_box: QuadLocBox
    source_region_box: QuadLocBox | None = None
    rejected_regions: set[QuadLocRegion] = field(default_factory=set)
    selected_region: QuadLocRegion | None = None


_REGION_COLORS: dict[QuadLocRegion, tuple[int, int, int]] = {
    QuadLocRegion.RED: (235, 68, 82),
    QuadLocRegion.GREEN: (42, 184, 108),
    QuadLocRegion.BLUE: (61, 116, 238),
    QuadLocRegion.YELLOW: (244, 184, 46),
}
_SELECTABLE_REGIONS = tuple(_REGION_COLORS)


class QuadLocLocator:
    """Run bounded four-way visual localization and model-mask correction."""

    def __init__(
        self,
        *,
        llm: QuadLocVlm,
        segment_tool: QuadLocSegmentTool,
        system_prompt: str,
        prompt_builder: QuadLocPromptBuilder,
        config: QuadLocConfig | None = None,
        workdir: Path | None = None,
    ) -> None:
        self._llm = llm
        self._segment_tool = segment_tool
        self._system_prompt = system_prompt
        self._prompt_builder = prompt_builder
        self.config = config or QuadLocConfig()
        self._workdir = (workdir or Path.cwd()).resolve()

    def locate(
        self,
        *,
        image_path: str | Path,
        location_description: str,
        artifact_root: str | None = None,
    ) -> QuadLocResult:
        """Locate one described target in screenshot pixel coordinates."""
        image_file = _resolve_image_path(image_path)
        description = _normalize_description(location_description)
        image = _load_rgb_image(image_file)
        full_box = QuadLocBox(0, 0, image.width, image.height)
        if full_box.width < 2 or full_box.height < 2:
            raise QuadLocInputError(
                "QuadLoc requires an image at least 2 by 2 pixels"
            )
        run_dir = _create_run_dir(
            artifact_root or self.config.artifact_root,
            workdir=self._workdir,
            image_stem=image_file.stem,
        )
        frames = [_SearchFrame(crop_box=full_box)]
        trace: list[dict[str, JsonValue]] = []
        backtrack_count = 0
        vlm_call_count = 0
        raw_coordinate: tuple[int, int] | None = None
        depth_reached = 0

        while raw_coordinate is None:
            frame = frames[-1]
            depth = len(frames)
            depth_reached = max(depth_reached, depth)
            if frame.crop_box.width < 2 or frame.crop_box.height < 2:
                source = frame.source_region_box or frame.crop_box
                raw_coordinate = _box_center(source, full_box)
                trace.append(
                    {
                        "call_index": vlm_call_count,
                        "depth": depth,
                        "crop_box": frame.crop_box.as_payload(),
                        "action": "pixel_resolution_limit",
                        "raw_coordinate": _point_payload(raw_coordinate),
                    }
                )
                break

            available = [
                region
                for region in _SELECTABLE_REGIONS
                if region not in frame.rejected_regions
            ]
            if not available:
                raise QuadLocSearchError(
                    "QuadLoc exhausted all four regions before reaching "
                    "the requested depth"
                )
            quadrants = _quadrant_boxes(frame.crop_box)
            vlm_call_count += 1
            overlay_path = run_dir / (
                f"step-{vlm_call_count:02d}-depth-{depth:02d}-overlay.png"
            )
            _render_quadrant_overlay(
                image,
                crop_box=frame.crop_box,
                quadrants=quadrants,
                output_path=overlay_path,
                opacity=self.config.overlay_opacity,
            )
            user_prompt = self._prompt_builder(
                location_description=description,
                depth=depth,
                max_depth=self.config.max_depth,
                crop_box=frame.crop_box.as_payload(),
                available_regions=[item.value for item in available],
                rejected_regions=[
                    item.value
                    for item in _SELECTABLE_REGIONS
                    if item in frame.rejected_regions
                ],
            )
            decision, metadata = self._complete_decision(
                user_prompt=user_prompt,
                overlay_path=overlay_path,
            )
            entry: dict[str, JsonValue] = {
                "call_index": vlm_call_count,
                "depth": depth,
                "crop_box": frame.crop_box.as_payload(),
                "quadrants": {
                    region.value: box.as_payload()
                    for region, box in quadrants.items()
                },
                "available_regions": [item.value for item in available],
                "rejected_regions": [
                    item.value
                    for item in _SELECTABLE_REGIONS
                    if item in frame.rejected_regions
                ],
                "overlay_path": str(overlay_path),
                "decision": decision.model_dump(mode="json"),
                "llm_metadata": metadata,
            }

            if (
                decision.region is not QuadLocRegion.NONE
                and decision.region in available
            ):
                selected_box = quadrants[decision.region]
                frame.selected_region = decision.region
                entry["selected_box"] = selected_box.as_payload()
                if depth >= self.config.max_depth:
                    raw_coordinate = _box_center(selected_box, full_box)
                    entry["action"] = "finish"
                    entry["raw_coordinate"] = _point_payload(
                        raw_coordinate
                    )
                    trace.append(entry)
                    break
                expanded = _expand_box(
                    selected_box,
                    within=full_box,
                    ratio=self.config.region_expansion_ratio,
                )
                entry["action"] = "descend"
                entry["expanded_next_crop"] = expanded.as_payload()
                trace.append(entry)
                frames.append(
                    _SearchFrame(
                        crop_box=expanded,
                        source_region_box=selected_box,
                    )
                )
                continue

            entry["action"] = "backtrack"
            if decision.region in frame.rejected_regions:
                entry["decision_rejected"] = True
            trace.append(entry)
            if backtrack_count >= self.config.max_backtracks:
                raise QuadLocSearchError(
                    "QuadLoc reached the maximum of "
                    f"{self.config.max_backtracks} backtracks"
                )
            backtrack_count += 1
            if len(frames) == 1:
                entry["action"] = "retry_root"
                continue
            frames.pop()
            parent = frames[-1]
            if parent.selected_region is not None:
                parent.rejected_regions.add(parent.selected_region)
                parent.selected_region = None

        assert raw_coordinate is not None
        segment_payload = self._segment_original_model(
            image_path=image_file,
            output_dir=run_dir,
        )
        cleaned_mask_path = _cleaned_mask_path(segment_payload)
        corrected, distance = _correct_point_to_mask(
            raw_coordinate,
            mask_path=cleaned_mask_path,
            image_size=image.size,
            chunk_size=self.config.nearest_point_chunk_size,
        )
        final_visualization_path = run_dir / "quadloc-final.png"
        _render_final_visualization(
            image,
            mask_path=cleaned_mask_path,
            raw_point=raw_coordinate,
            corrected_point=corrected,
            output_path=final_visualization_path,
        )
        trace_path = run_dir / "quadloc-trace.json"
        _write_json(
            trace_path,
            {
                "algorithm": {
                    "name": "QuadLoc-four-way-recursive/v1",
                    "max_depth": self.config.max_depth,
                    "max_backtracks": self.config.max_backtracks,
                    "region_expansion_ratio": (
                        self.config.region_expansion_ratio
                    ),
                },
                "input": {
                    "image_path": str(image_file),
                    "location_description": description,
                    "image_size": {
                        "width": image.width,
                        "height": image.height,
                    },
                },
                "trace": trace,
                "result": {
                    "raw_coordinate": _point_payload(raw_coordinate),
                    "coordinate": _point_payload(corrected),
                    "mask_correction_applied": corrected != raw_coordinate,
                    "mask_correction_distance_pixels": distance,
                },
            },
        )
        sam3_overlay_path = _optional_result_path(
            segment_payload,
            "overlay_path",
        )
        return QuadLocResult(
            x=corrected[0],
            y=corrected[1],
            raw_x=raw_coordinate[0],
            raw_y=raw_coordinate[1],
            corrected=corrected != raw_coordinate,
            correction_distance=distance,
            depth_reached=depth_reached,
            backtrack_count=backtrack_count,
            vlm_call_count=vlm_call_count,
            artifact_dir=str(run_dir),
            trace_path=str(trace_path),
            visualization_path=str(final_visualization_path),
            cleaned_mask_path=str(cleaned_mask_path),
            sam3_overlay_path=sam3_overlay_path,
        )

    def _complete_decision(
        self,
        *,
        user_prompt: str,
        overlay_path: Path,
    ) -> tuple[QuadLocDecision, dict[str, JsonValue]]:
        try:
            completion = self._llm.complete(
                role=self.config.llm_role,
                system_prompt=self._system_prompt,
                user_prompt=user_prompt,
                image_paths=[overlay_path],
                response_model=QuadLocDecision,
            )
        except Exception as error:
            raise QuadLocVlmError(
                f"QuadLoc VLM request failed: {error}"
            ) from error
        if not isinstance(completion.value, QuadLocDecision):
            raise QuadLocVlmError(
                "QuadLoc VLM returned an unexpected structured output"
            )
        return completion.value, dict(completion.metadata)

    def _segment_original_model(
        self,
        *,
        image_path: Path,
        output_dir: Path,
    ) -> dict[str, JsonValue]:
        response = self._segment_tool.invoke(
            {
                "image_path": str(image_path),
                "prompt": self.config.model_segmentation_prompt,
                "confidence_threshold": (
                    self.config.sam3_confidence_threshold
                ),
                "overlay_opacity": self.config.sam3_overlay_opacity,
                "output_dir": str(output_dir),
            }
        )
        payload: object = response
        if isinstance(response, str):
            try:
                payload = json.loads(response)
            except json.JSONDecodeError as error:
                raise QuadLocSegmentationError(
                    "segment_with_sam3 returned non-JSON text"
                ) from error
        if not isinstance(payload, dict):
            raise QuadLocSegmentationError(
                "segment_with_sam3 returned an unexpected result"
            )
        if "sam3_error" in payload:
            sam3_error = payload["sam3_error"]
            if _is_empty_sam3_error(sam3_error):
                raise QuadLocNoModelMaskError(
                    "SAM3 produced no whole-model mask for QuadLoc"
                )
            raise QuadLocSegmentationError(
                "segment_with_sam3 failed: "
                + json.dumps(
                    sam3_error,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        result = payload.get("result")
        if not isinstance(result, dict):
            raise QuadLocSegmentationError(
                "segment_with_sam3 result is missing result"
            )
        return cast(dict[str, JsonValue], payload)


def _resolve_image_path(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)):
        raise QuadLocInputError("image_path must be a string or Path")
    raw = str(value).strip()
    if not raw:
        raise QuadLocInputError("image_path must not be empty")
    path = Path(os.path.expandvars(os.path.expanduser(raw)))
    if not path.is_absolute():
        raise QuadLocInputError("image_path must be absolute")
    resolved = path.resolve()
    if not resolved.is_file():
        raise QuadLocInputError(f"image_path is not a file: {resolved}")
    return resolved


def _normalize_description(value: str) -> str:
    if not isinstance(value, str):
        raise QuadLocInputError("location_description must be a string")
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise QuadLocInputError("location_description must not be empty")
    if len(normalized) > 512:
        raise QuadLocInputError(
            "location_description must contain at most 512 characters"
        )
    return normalized


def _load_rgb_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise QuadLocInputError(f"cannot read input screenshot: {path}") from error


def _quadrant_boxes(box: QuadLocBox) -> dict[QuadLocRegion, QuadLocBox]:
    mid_x = box.left + box.width // 2
    mid_y = box.top + box.height // 2
    return {
        QuadLocRegion.RED: QuadLocBox(
            box.left,
            box.top,
            mid_x,
            mid_y,
        ),
        QuadLocRegion.GREEN: QuadLocBox(
            mid_x,
            box.top,
            box.right,
            mid_y,
        ),
        QuadLocRegion.BLUE: QuadLocBox(
            box.left,
            mid_y,
            mid_x,
            box.bottom,
        ),
        QuadLocRegion.YELLOW: QuadLocBox(
            mid_x,
            mid_y,
            box.right,
            box.bottom,
        ),
    }


def _expand_box(
    box: QuadLocBox,
    *,
    within: QuadLocBox,
    ratio: float,
) -> QuadLocBox:
    # A 20% expansion scales width and height by 1.2, adding 10% per side.
    horizontal_padding = box.width * ratio / 2.0
    vertical_padding = box.height * ratio / 2.0
    return QuadLocBox(
        left=max(within.left, math.floor(box.left - horizontal_padding)),
        top=max(within.top, math.floor(box.top - vertical_padding)),
        right=min(within.right, math.ceil(box.right + horizontal_padding)),
        bottom=min(
            within.bottom,
            math.ceil(box.bottom + vertical_padding),
        ),
    )


def _box_center(
    box: QuadLocBox,
    within: QuadLocBox,
) -> tuple[int, int]:
    x = max(within.left, min(within.right - 1, (box.left + box.right) // 2))
    y = max(within.top, min(within.bottom - 1, (box.top + box.bottom) // 2))
    return x, y


def _render_quadrant_overlay(
    image: Image.Image,
    *,
    crop_box: QuadLocBox,
    quadrants: Mapping[QuadLocRegion, QuadLocBox],
    output_path: Path,
    opacity: float,
) -> None:
    crop = image.crop(
        (crop_box.left, crop_box.top, crop_box.right, crop_box.bottom)
    ).convert("RGBA")
    overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    alpha = round(255 * opacity)
    for region in _SELECTABLE_REGIONS:
        box = quadrants[region]
        local = (
            box.left - crop_box.left,
            box.top - crop_box.top,
            box.right - crop_box.left - 1,
            box.bottom - crop_box.top - 1,
        )
        color = _REGION_COLORS[region]
        draw.rectangle(
            local,
            fill=(*color, alpha),
            outline=(*color, 255),
            width=max(1, min(crop.size) // 300),
        )
        text_x = local[0] + max(4, crop.width // 100)
        text_y = local[1] + max(4, crop.height // 100)
        draw.text(
            (text_x, text_y),
            region.value,
            fill=(255, 255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 220),
        )
    rendered = Image.alpha_composite(crop, overlay).convert("RGB")
    _save_png(rendered, output_path)


def _cleaned_mask_path(payload: Mapping[str, JsonValue]) -> Path:
    result = payload.get("result")
    if not isinstance(result, dict):
        raise QuadLocSegmentationError(
            "segment_with_sam3 result is missing result"
        )
    raw_path = result.get("cleaned_mask_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise QuadLocSegmentationError(
            "segment_with_sam3 is missing cleaned_mask_path"
        )
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise QuadLocSegmentationError(
            f"cleaned SAM3 mask is not a file: {path}"
        )
    return path


def _optional_result_path(
    payload: Mapping[str, JsonValue],
    key: str,
) -> str | None:
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    value = result.get(key)
    return value if isinstance(value, str) else None


def _correct_point_to_mask(
    point: tuple[int, int],
    *,
    mask_path: Path,
    image_size: tuple[int, int],
    chunk_size: int,
) -> tuple[tuple[int, int], float]:
    try:
        with Image.open(mask_path) as source:
            mask = source.convert("L")
            if mask.size != image_size:
                mask = mask.resize(image_size, resample=Image.Resampling.NEAREST)
            mask_array = np.asarray(mask, dtype=np.uint8)
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
    ) as error:
        raise QuadLocSegmentationError(
            f"cannot read cleaned SAM3 mask: {mask_path}"
        ) from error
    x = max(0, min(image_size[0] - 1, int(point[0])))
    y = max(0, min(image_size[1] - 1, int(point[1])))
    if mask_array[y, x] > 0:
        return (x, y), 0.0
    foreground_y, foreground_x = np.nonzero(mask_array > 0)
    if foreground_x.size == 0:
        raise QuadLocNoModelMaskError(
            "cleaned SAM3 model mask contains no foreground"
        )
    best_index = -1
    best_distance_squared: int | None = None
    for start in range(0, foreground_x.size, chunk_size):
        stop = min(foreground_x.size, start + chunk_size)
        chunk_x = foreground_x[start:stop].astype(np.int64, copy=False)
        chunk_y = foreground_y[start:stop].astype(np.int64, copy=False)
        distances = (chunk_x - x) ** 2 + (chunk_y - y) ** 2
        local_index = int(np.argmin(distances))
        candidate_distance = int(distances[local_index])
        if (
            best_distance_squared is None
            or candidate_distance < best_distance_squared
        ):
            best_distance_squared = candidate_distance
            best_index = start + local_index
    assert best_distance_squared is not None and best_index >= 0
    corrected = (
        int(foreground_x[best_index]),
        int(foreground_y[best_index]),
    )
    return corrected, math.sqrt(best_distance_squared)


def _is_empty_sam3_error(value: JsonValue) -> bool:
    if not isinstance(value, dict):
        return False
    error_type = value.get("type")
    message = value.get("message")
    return (
        error_type == "mask_cleanup_error"
        and isinstance(message, str)
        and "no foreground" in message.casefold()
    )


def _render_final_visualization(
    image: Image.Image,
    *,
    mask_path: Path,
    raw_point: tuple[int, int],
    corrected_point: tuple[int, int],
    output_path: Path,
) -> None:
    try:
        with Image.open(mask_path) as source:
            mask = source.convert("L")
            if mask.size != image.size:
                mask = mask.resize(
                    image.size,
                    resample=Image.Resampling.NEAREST,
                )
    except OSError as error:
        raise QuadLocArtifactError(
            f"cannot render cleaned mask: {mask_path}"
        ) from error
    base = image.convert("RGBA")
    tint = Image.new("RGBA", image.size, (45, 205, 220, 72))
    binary_mask = mask.point(lambda value: 112 if value > 0 else 0)
    base.alpha_composite(
        Image.composite(
            tint,
            Image.new("RGBA", image.size),
            binary_mask,
        )
    )
    draw = ImageDraw.Draw(base)
    radius = max(5, min(image.size) // 140)
    if raw_point != corrected_point:
        draw.line(
            (raw_point, corrected_point),
            fill=(255, 145, 45, 255),
            width=max(2, radius // 3),
        )
    _draw_point(draw, raw_point, radius, fill=(255, 145, 45, 255))
    _draw_point(draw, corrected_point, radius, fill=(40, 230, 120, 255))
    _save_png(base.convert("RGB"), output_path)


def _draw_point(
    draw: ImageDraw.ImageDraw,
    point: tuple[int, int],
    radius: int,
    *,
    fill: tuple[int, int, int, int],
) -> None:
    x, y = point
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=fill,
        outline=(255, 255, 255, 255),
        width=max(1, radius // 3),
    )


def _create_run_dir(
    artifact_root: str,
    *,
    workdir: Path,
    image_stem: str,
) -> Path:
    root = Path(
        os.path.expandvars(os.path.expanduser(artifact_root.strip()))
    )
    if not root.is_absolute():
        root = workdir / root
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", image_stem).strip("-.")
    safe_stem = safe_stem[:64] or "image"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = root.resolve() / f"{safe_stem}-{timestamp}-{uuid4().hex[:8]}"
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise QuadLocArtifactError(
            f"cannot create QuadLoc artifact directory: {run_dir}"
        ) from error
    return run_dir


def _save_png(image: Image.Image, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    except OSError as error:
        raise QuadLocArtifactError(
            f"cannot write QuadLoc image artifact: {path}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, JsonValue]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as error:
        raise QuadLocArtifactError(
            f"cannot write QuadLoc trace artifact: {path}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _point_payload(point: tuple[int, int]) -> dict[str, int]:
    return {"x": point[0], "y": point[1]}
