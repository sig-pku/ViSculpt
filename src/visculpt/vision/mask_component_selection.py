"""VLM-assisted selection of one semantic mask component."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field

from visculpt.bridge import JsonValue

_MASK_THRESHOLD = 128
_DEFAULT_ARTIFACT_ROOT = Path("output/tools/select-mask-component")
_MAX_COMPONENTS = 256

MaskArray = NDArray[np.uint8]
CompletionOutput = TypeVar("CompletionOutput", bound=BaseModel)


class MaskComponentSelectionError(RuntimeError):
    """Base error for semantic component selection."""


class MaskComponentSelectionInputError(MaskComponentSelectionError):
    """Raised when input images or descriptions are invalid."""


class MaskComponentSelectionVlmError(MaskComponentSelectionError):
    """Raised when the VLM cannot select one numbered component."""


class MaskComponentSelectionArtifactError(MaskComponentSelectionError):
    """Raised when selection artifacts cannot be persisted."""


class MaskComponentDecision(BaseModel):
    """One numbered connected-component decision returned by the VLM."""

    model_config = ConfigDict(extra="forbid")

    component_number: int = Field(ge=1, le=_MAX_COMPONENTS)
    reason: str = Field(min_length=1, max_length=2_000)


class MaskComponentCompletion(Protocol):
    """Provider-neutral completion result used by the selector."""

    value: BaseModel
    metadata: dict[str, JsonValue]


class MaskComponentVlm(Protocol):
    """Minimal multimodal structured-completion surface."""

    def complete(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: Sequence[str | Path],
        response_model: type[CompletionOutput],
    ) -> MaskComponentCompletion:
        """Select exactly one numbered component."""


class MaskComponentPromptBuilder(Protocol):
    """English prompt builder kept in the centralized prompt module."""

    def __call__(
        self,
        *,
        part_description: str,
        component_count: int,
    ) -> str:
        """Build one component-selection prompt."""


@dataclass(frozen=True, slots=True)
class MaskComponent:
    """Deterministically numbered connected-component geometry."""

    number: int
    label: int
    area_pixels: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    marker: tuple[int, int]

    def as_payload(self) -> dict[str, JsonValue]:
        """Return JSON-safe component metadata."""
        left, top, right, bottom = self.bbox
        return {
            "number": self.number,
            "area_pixels": self.area_pixels,
            "bbox": {
                "left": left,
                "top": top,
                "right_exclusive": right,
                "bottom_exclusive": bottom,
                "width": right - left,
                "height": bottom - top,
            },
            "centroid": {
                "x": round(self.centroid[0], 6),
                "y": round(self.centroid[1], 6),
            },
            "marker": {"x": self.marker[0], "y": self.marker[1]},
        }


@dataclass(frozen=True, slots=True)
class MaskComponentSelectionResult:
    """Persisted single-component mask and review artifacts."""

    payload: dict[str, JsonValue]

    def as_payload(self) -> dict[str, JsonValue]:
        """Return a detached payload for Tool responses."""
        return dict(self.payload)


def foreground_component_count(mask_path: str | Path) -> int:
    """Count 8-connected foreground regions in one binary mask."""
    mask = _load_binary_mask(_resolve_file(mask_path, label="mask_path"))
    try:
        count, _, _, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    except cv2.error as error:
        raise MaskComponentSelectionInputError(
            f"Cannot analyze mask connected components: {error}"
        ) from error
    return max(0, int(count) - 1)


def select_mask_component(
    *,
    llm: MaskComponentVlm,
    image_path: str | Path,
    cleaned_mask_path: str | Path,
    part_description: str,
    system_prompt: str,
    prompt_builder: MaskComponentPromptBuilder,
    llm_role: str = "translator",
    overlay_opacity: float = 0.45,
    output_dir: str | Path | None = None,
    workdir: str | Path | None = None,
) -> MaskComponentSelectionResult:
    """Number mask components, ask the VLM, and persist one component."""
    image_file = _resolve_file(image_path, label="image_path")
    mask_file = _resolve_file(cleaned_mask_path, label="cleaned_mask_path")
    description = " ".join(part_description.strip().split())
    if not description:
        raise MaskComponentSelectionInputError(
            "part_description must not be empty"
        )
    role = llm_role.strip()
    if not role:
        raise MaskComponentSelectionInputError("llm_role must not be empty")
    if not math.isfinite(overlay_opacity) or not 0.0 <= overlay_opacity <= 1.0:
        raise MaskComponentSelectionInputError(
            "overlay_opacity must be between 0 and 1"
        )

    image = _load_rgb_image(image_file)
    mask = _load_binary_mask(mask_file)
    if mask.shape != (image.height, image.width):
        raise MaskComponentSelectionInputError(
            "cleaned mask dimensions do not match the source image"
        )
    labels, components = _analyze_components(mask)
    if not components:
        raise MaskComponentSelectionInputError(
            "cleaned mask contains no foreground component"
        )

    artifact_dir = _resolve_output_dir(
        output_dir,
        workdir=workdir,
    )
    numbered_overlay_path = artifact_dir / "numbered-components-overlay.png"
    _render_numbered_overlay(
        image=image,
        labels=labels,
        components=components,
        opacity=overlay_opacity,
        output_path=numbered_overlay_path,
    )

    completion_metadata: dict[str, JsonValue] | None = None
    if len(components) == 1:
        selected = components[0]
        strategy = "SINGLE_COMPONENT"
        reason = "The cleaned mask already contains exactly one component."
    else:
        try:
            completion = llm.complete(
                role=role,
                system_prompt=system_prompt,
                user_prompt=prompt_builder(
                    part_description=description,
                    component_count=len(components),
                ),
                image_paths=[numbered_overlay_path],
                response_model=MaskComponentDecision,
            )
        except Exception as error:
            raise MaskComponentSelectionVlmError(
                f"Mask component VLM request failed: {error}"
            ) from error
        if not isinstance(completion.value, MaskComponentDecision):
            raise MaskComponentSelectionVlmError(
                "Mask component VLM returned an unexpected response model"
            )
        if completion.value.component_number > len(components):
            raise MaskComponentSelectionVlmError(
                "Mask component VLM selected number "
                f"{completion.value.component_number}, but only "
                f"1-{len(components)} are available"
            )
        selected = components[completion.value.component_number - 1]
        strategy = "VLM_NUMBERED_OVERLAY"
        reason = completion.value.reason
        completion_metadata = dict(completion.metadata)

    selected_mask = np.where(
        labels == selected.label,
        255,
        0,
    ).astype(np.uint8)
    selected_mask_path = artifact_dir / "selected-component-mask.png"
    selected_overlay_path = artifact_dir / "selected-component-overlay.png"
    metadata_path = artifact_dir / "component-selection.json"
    _save_mask(selected_mask, selected_mask_path)
    _render_selected_overlay(
        image=image,
        selected_mask=selected_mask,
        opacity=overlay_opacity,
        output_path=selected_overlay_path,
    )
    payload: dict[str, JsonValue] = {
        "schema_version": "mask-component-selection/v1",
        "selection_applied": len(components) > 1,
        "part_description": description,
        "source_mask_path": str(mask_file),
        "component_count": len(components),
        "component_numbering": "TOP_TO_BOTTOM_THEN_LEFT_TO_RIGHT",
        "components": [item.as_payload() for item in components],
        "selected_component": selected.as_payload(),
        "selection": {
            "strategy": strategy,
            "reason": reason,
        },
        "selected_mask_path": str(selected_mask_path),
        "selected_overlay_path": str(selected_overlay_path),
        "numbered_overlay_path": str(numbered_overlay_path),
        "metadata_path": str(metadata_path),
        "llm": completion_metadata,
    }
    _write_json(metadata_path, payload)
    return MaskComponentSelectionResult(payload=payload)


def _analyze_components(
    mask: MaskArray,
) -> tuple[NDArray[np.int32], tuple[MaskComponent, ...]]:
    try:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )
    except cv2.error as error:
        raise MaskComponentSelectionInputError(
            f"Cannot analyze mask connected components: {error}"
        ) from error
    if count - 1 > _MAX_COMPONENTS:
        raise MaskComponentSelectionInputError(
            f"cleaned mask contains more than {_MAX_COMPONENTS} components"
        )
    raw: list[
        tuple[
            int,
            int,
            tuple[int, int, int, int],
            tuple[float, float],
            tuple[int, int],
        ]
    ] = []
    for label in range(1, count):
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        centroid = (
            float(centroids[label, 0]),
            float(centroids[label, 1]),
        )
        component_mask = np.where(labels == label, 255, 0).astype(np.uint8)
        marker = _interior_marker(component_mask, centroid=centroid)
        raw.append(
            (
                label,
                area,
                (left, top, left + width, top + height),
                centroid,
                marker,
            )
        )
    raw.sort(key=lambda item: (item[4][1], item[4][0], -item[1]))
    components = tuple(
        MaskComponent(
            number=number,
            label=label,
            area_pixels=area,
            bbox=bbox,
            centroid=centroid,
            marker=marker,
        )
        for number, (label, area, bbox, centroid, marker) in enumerate(
            raw,
            start=1,
        )
    )
    return labels, components


def _interior_marker(
    component: MaskArray,
    *,
    centroid: tuple[float, float],
) -> tuple[int, int]:
    x = int(round(centroid[0]))
    y = int(round(centroid[1]))
    if (
        0 <= y < component.shape[0]
        and 0 <= x < component.shape[1]
        and component[y, x] != 0
    ):
        return x, y
    padded = np.pad(component, 1, mode="constant", constant_values=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    marker_y, marker_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
    return int(marker_x), int(marker_y)


def _render_numbered_overlay(
    *,
    image: Image.Image,
    labels: NDArray[np.int32],
    components: Sequence[MaskComponent],
    opacity: float,
    output_path: Path,
) -> None:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    foreground = labels > 0
    color = np.asarray((46, 204, 113), dtype=np.float32)
    rendered = base.copy()
    rendered[foreground] = (
        base[foreground] * (1.0 - opacity) + color * opacity
    )
    overlay = Image.fromarray(np.clip(rendered, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(overlay)
    radius = max(12, min(38, round(min(image.size) * 0.025)))
    font = _badge_font(max(14, round(radius * 1.25)))
    for component in components:
        x, y = component.marker
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(18, 18, 18),
            outline=(255, 255, 255),
            width=max(2, radius // 7),
        )
        text = str(component.number)
        box = draw.textbbox((0, 0), text, font=font, stroke_width=1)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        draw.text(
            (x - text_width / 2, y - text_height / 2 - box[1]),
            text,
            font=font,
            fill=(255, 255, 255),
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
    _save_png(overlay, output_path)


def _render_selected_overlay(
    *,
    image: Image.Image,
    selected_mask: MaskArray,
    opacity: float,
    output_path: Path,
) -> None:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    foreground = selected_mask != 0
    color = np.asarray((46, 204, 113), dtype=np.float32)
    rendered = base.copy()
    rendered[foreground] = (
        base[foreground] * (1.0 - opacity) + color * opacity
    )
    _save_png(
        Image.fromarray(np.clip(rendered, 0, 255).astype(np.uint8)),
        output_path,
    )


def _badge_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=size)
    except OSError:
        return ImageFont.load_default(size=size)


def _resolve_file(value: str | Path, *, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise MaskComponentSelectionInputError(
            f"{label} must be a string or Path"
        )
    raw = str(value).strip()
    if not raw:
        raise MaskComponentSelectionInputError(f"{label} must not be empty")
    path = Path(os.path.expandvars(os.path.expanduser(raw)))
    if not path.is_absolute():
        raise MaskComponentSelectionInputError(f"{label} must be absolute")
    resolved = path.resolve()
    if not resolved.is_file():
        raise MaskComponentSelectionInputError(
            f"{label} is not a file: {resolved}"
        )
    return resolved


def _resolve_output_dir(
    value: str | Path | None,
    *,
    workdir: str | Path | None,
) -> Path:
    root = Path.cwd() if workdir is None else Path(workdir).expanduser()
    path = _DEFAULT_ARTIFACT_ROOT if value is None else Path(value).expanduser()
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise MaskComponentSelectionArtifactError(
            f"Cannot create component-selection output directory: {error}"
        ) from error
    return resolved


def _load_rgb_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as source:
            return source.convert("RGB")
    except (OSError, ValueError) as error:
        raise MaskComponentSelectionInputError(
            f"Cannot read source image {path}: {error}"
        ) from error


def _load_binary_mask(path: Path) -> MaskArray:
    try:
        with Image.open(path) as source:
            mask = np.asarray(source.convert("L"), dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise MaskComponentSelectionInputError(
            f"Cannot read cleaned mask {path}: {error}"
        ) from error
    return np.where(mask >= _MASK_THRESHOLD, 255, 0).astype(np.uint8)


def _save_mask(mask: MaskArray, path: Path) -> None:
    try:
        Image.fromarray(mask, mode="L").save(path, format="PNG")
    except OSError as error:
        raise MaskComponentSelectionArtifactError(
            f"Cannot save selected component mask: {error}"
        ) from error


def _save_png(image: Image.Image, path: Path) -> None:
    try:
        image.save(path, format="PNG")
    except OSError as error:
        raise MaskComponentSelectionArtifactError(
            f"Cannot save component overlay: {error}"
        ) from error


def _write_json(path: Path, payload: dict[str, JsonValue]) -> None:
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise MaskComponentSelectionArtifactError(
            f"Cannot save component-selection metadata: {error}"
        ) from error
