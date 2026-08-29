"""Generate font-centerline mouse trajectories inside a segmentation mask."""

from __future__ import annotations

import difflib
import math
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from visculpt.bridge import JsonValue

from .mask_trajectory_fit import (
    MaskTrajectoryFitConfig,
    MaskTrajectoryFitError,
    fit_svg_trajectories_to_mask,
)

type Point = tuple[float, float]
type Pixel = tuple[int, int]
type BinaryImage = NDArray[np.bool_]

_CANVAS_SIZE = 512.0
_POINT_EPSILON = 1e-9
_SCALE_TIERS = {"SMALL", "MEDIUM", "LARGE"}


class TextTrajectoryGenerationError(ValueError):
    """Raised when text cannot be converted into mask-contained strokes."""


@dataclass(frozen=True, slots=True)
class TextTrajectoryConfig:
    """Deterministic font-centerline and layout-search settings."""

    maximum_characters: int = 64
    maximum_layout_lanes: int = 8
    base_font_size: float = 100.0
    skeleton_raster_scale: float = 2.0
    centerline_sample_spacing: float = 2.0
    centerline_simplify_tolerance: float = 0.5
    skeleton_spur_prune_length: float = 2.0
    line_gap_ratio: float = 0.2
    canonical_canvas_margin: float = 8.0

    def __post_init__(self) -> None:
        """Validate bounded layout settings."""
        if not 1 <= self.maximum_characters <= 256:
            raise ValueError("maximum_characters must be between 1 and 256")
        if not 1 <= self.maximum_layout_lanes <= 32:
            raise ValueError(
                "maximum_layout_lanes must be between 1 and 32"
            )
        if (
            not math.isfinite(self.base_font_size)
            or not 16.0 <= self.base_font_size <= 512.0
        ):
            raise ValueError("base_font_size must be between 16 and 512")
        if (
            not math.isfinite(self.skeleton_raster_scale)
            or not 1.0 <= self.skeleton_raster_scale <= 8.0
        ):
            raise ValueError(
                "skeleton_raster_scale must be between 1 and 8"
            )
        if (
            not math.isfinite(self.centerline_sample_spacing)
            or not 0.25 <= self.centerline_sample_spacing <= 16.0
        ):
            raise ValueError(
                "centerline_sample_spacing must be between 0.25 and 16"
            )
        if (
            not math.isfinite(self.centerline_simplify_tolerance)
            or not 0.0 <= self.centerline_simplify_tolerance <= 4.0
        ):
            raise ValueError(
                "centerline_simplify_tolerance must be between 0 and 4"
            )
        if (
            not math.isfinite(self.skeleton_spur_prune_length)
            or not 0.0 <= self.skeleton_spur_prune_length <= 16.0
        ):
            raise ValueError(
                "skeleton_spur_prune_length must be between 0 and 16"
            )
        if (
            not math.isfinite(self.line_gap_ratio)
            or not 0.0 <= self.line_gap_ratio <= 1.0
        ):
            raise ValueError("line_gap_ratio must be between 0 and 1")
        if (
            not math.isfinite(self.canonical_canvas_margin)
            or not 0.0 <= self.canonical_canvas_margin < 128.0
        ):
            raise ValueError(
                "canonical_canvas_margin must be between 0 and 128"
            )


@dataclass(frozen=True, slots=True)
class TextMouseTrajectoryResult:
    """Selected layout and mask-contained mouse trajectories."""

    trajectory_plan: dict[str, JsonValue]

    def as_payload(self) -> dict[str, JsonValue]:
        """Return the JSON-compatible trajectory plan."""
        return self.trajectory_plan


@dataclass(frozen=True, slots=True)
class _ResolvedFont:
    requested_name: str
    family_name: str
    file_path: Path


@dataclass(frozen=True, slots=True)
class _TextPlacement:
    value: str
    baseline_x: float
    baseline_y: float


@dataclass(frozen=True, slots=True)
class _RawStroke:
    points: tuple[Point, ...]
    closed: bool


@dataclass(frozen=True, slots=True)
class _LayoutCandidate:
    orientation: str
    lanes: tuple[str, ...]
    trajectory_plan: dict[str, object]
    canonical_scale: float


@dataclass(frozen=True, slots=True)
class _FittedCandidate:
    layout: _LayoutCandidate
    fitted_plan: dict[str, JsonValue]
    effective_font_size_pixels: float
    center_offset_pixels: float


def generate_text_mouse_trajectories(
    *,
    text: str,
    font_name: str,
    mask_path: str | Path,
    scale_tier: str = "MEDIUM",
    config: TextTrajectoryConfig | None = None,
    fit_config: MaskTrajectoryFitConfig | None = None,
) -> TextMouseTrajectoryResult:
    """Lay out English text and fit its font centerlines inside a mask."""
    settings = config or TextTrajectoryConfig()
    normalized_text = _normalize_english_text(
        text,
        maximum_characters=settings.maximum_characters,
    )
    normalized_font_name = _normalize_font_name(font_name)
    normalized_scale_tier = _normalize_scale_tier(scale_tier)
    font = _resolve_font(normalized_font_name)
    _validate_font_characters(font, normalized_text)
    candidates = _layout_candidates(
        normalized_text,
        font=font,
        settings=settings,
    )
    fitted: list[_FittedCandidate] = []
    for candidate in candidates:
        try:
            result = fit_svg_trajectories_to_mask(
                mask_path=mask_path,
                trajectory_plan=candidate.trajectory_plan,
                scale_tier=normalized_scale_tier,
                config=fit_config,
            ).trajectory_plan
        except MaskTrajectoryFitError:
            continue
        transform = result.get("transform")
        if not isinstance(transform, dict):
            continue
        rotation = float(transform.get("rotation_degrees", math.nan))
        if not math.isclose(rotation, 0.0, abs_tol=1e-6):
            continue
        fitted_scale = float(transform.get("uniform_scale", math.nan))
        center_offset = float(
            transform.get("center_offset_pixels", math.nan)
        )
        if not math.isfinite(fitted_scale) or not math.isfinite(
            center_offset
        ):
            continue
        fitted.append(
            _FittedCandidate(
                layout=candidate,
                fitted_plan=result,
                effective_font_size_pixels=(
                    settings.base_font_size
                    * candidate.canonical_scale
                    * fitted_scale
                ),
                center_offset_pixels=center_offset,
            )
        )
    if not fitted:
        raise TextTrajectoryGenerationError(
            "No horizontal or vertical text layout can fit completely "
            "inside the selected mask component"
        )
    selected = min(fitted, key=_fitted_candidate_sort_key)
    return TextMouseTrajectoryResult(
        trajectory_plan=_result_payload(
            text=normalized_text,
            font=font,
            selected=selected,
            scale_tier=normalized_scale_tier,
            candidate_count=len(candidates),
            fitted_candidate_count=len(fitted),
        )
    )


def _normalize_english_text(value: str, *, maximum_characters: int) -> str:
    if not isinstance(value, str):
        raise TextTrajectoryGenerationError("text must be a string")
    stripped = value.strip()
    if not stripped:
        raise TextTrajectoryGenerationError("text must not be empty")
    if any(ord(character) < 32 or ord(character) > 126 for character in stripped):
        raise TextTrajectoryGenerationError(
            "text supports printable English ASCII characters only"
        )
    normalized = " ".join(stripped.split())
    if len(normalized) > maximum_characters:
        raise TextTrajectoryGenerationError(
            f"text must contain at most {maximum_characters} characters"
        )
    if not any(character != " " for character in normalized):
        raise TextTrajectoryGenerationError("text must contain a glyph")
    return normalized


def _normalize_font_name(value: str) -> str:
    if not isinstance(value, str):
        raise TextTrajectoryGenerationError("font_name must be a string")
    normalized = " ".join(value.strip().split())
    if not normalized or len(normalized) > 128:
        raise TextTrajectoryGenerationError(
            "font_name must contain between 1 and 128 characters"
        )
    if any(character in normalized for character in ("/", "\\", "\0")):
        raise TextTrajectoryGenerationError(
            "font_name must be a font family name, not a file path"
        )
    return normalized


def _normalize_scale_tier(value: str) -> str:
    if not isinstance(value, str):
        raise TextTrajectoryGenerationError("scale_tier must be a string")
    normalized = value.strip().upper()
    if normalized not in _SCALE_TIERS:
        raise TextTrajectoryGenerationError(
            "scale_tier must be SMALL, MEDIUM, or LARGE"
        )
    return normalized


@lru_cache(maxsize=1)
def _matplotlib_font_api() -> tuple[object, object]:
    # Cache fonts in a temporary directory without requiring home-directory writes.
    cache_dir = Path(tempfile.gettempdir()) / "agentic-geometry-editing-mpl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    return font_manager, FT2Font


@lru_cache(maxsize=64)
def _resolve_font(font_name: str) -> _ResolvedFont:
    font_manager, _ = _matplotlib_font_api()
    manager = font_manager.fontManager
    normalized = _font_key(font_name)
    matches = [
        entry
        for entry in manager.ttflist
        if _font_key(str(entry.name)) == normalized
    ]
    if not matches:
        available = sorted({str(entry.name) for entry in manager.ttflist})
        suggestions = difflib.get_close_matches(
            font_name,
            available,
            n=3,
            cutoff=0.55,
        )
        suffix = (
            f"; close matches: {', '.join(suggestions)}"
            if suggestions
            else ""
        )
        raise TextTrajectoryGenerationError(
            f"Font family '{font_name}' is not available{suffix}"
        )
    selected = min(matches, key=_font_entry_sort_key)
    path = Path(str(selected.fname)).expanduser().resolve()
    if not path.is_file():
        raise TextTrajectoryGenerationError(
            f"Resolved font file does not exist for '{font_name}'"
        )
    return _ResolvedFont(
        requested_name=font_name,
        family_name=str(selected.name),
        file_path=path,
    )


def _font_key(value: str) -> str:
    return " ".join(value.casefold().split())


def _font_entry_sort_key(entry: object) -> tuple[int, int, str]:
    style = str(getattr(entry, "style", "normal")).casefold()
    weight = getattr(entry, "weight", 400)
    return (
        0 if style == "normal" else 1,
        abs(_numeric_font_weight(weight) - 400),
        str(getattr(entry, "fname", "")),
    )


def _numeric_font_weight(value: object) -> int:
    if isinstance(value, bool):
        return 400
    if isinstance(value, (int, float)):
        return int(value)
    weights = {
        "ultralight": 100,
        "light": 200,
        "normal": 400,
        "regular": 400,
        "book": 400,
        "medium": 500,
        "semibold": 600,
        "bold": 700,
        "heavy": 800,
        "black": 900,
    }
    return weights.get(str(value).casefold(), 400)


def _validate_font_characters(font: _ResolvedFont, text: str) -> None:
    _, font_type = _matplotlib_font_api()
    try:
        charmap = font_type(str(font.file_path)).get_charmap()
    except RuntimeError as error:
        raise TextTrajectoryGenerationError(
            f"Cannot read font '{font.family_name}': {error}"
        ) from error
    missing = sorted(
        {
            character
            for character in text
            if character != " " and ord(character) not in charmap
        }
    )
    if missing:
        rendered = ", ".join(repr(character) for character in missing)
        raise TextTrajectoryGenerationError(
            f"Font '{font.family_name}' does not contain: {rendered}"
        )


def _layout_candidates(
    text: str,
    *,
    font: _ResolvedFont,
    settings: TextTrajectoryConfig,
) -> list[_LayoutCandidate]:
    compact = "".join(text.split())
    maximum_lanes = min(settings.maximum_layout_lanes, len(compact))
    candidates: list[_LayoutCandidate] = []
    for lane_count in range(1, maximum_lanes + 1):
        horizontal_lanes = _balanced_horizontal_lanes(text, lane_count)
        horizontal = _render_layout_centerlines(
            orientation="HORIZONTAL",
            lanes=horizontal_lanes,
            font=font,
            settings=settings,
        )
        candidates.append(
            _canonical_candidate(
                orientation="HORIZONTAL",
                lanes=horizontal_lanes,
                strokes=horizontal,
                settings=settings,
                font=font,
            )
        )
        vertical_lanes = _balanced_character_lanes(compact, lane_count)
        vertical = _render_layout_centerlines(
            orientation="VERTICAL",
            lanes=vertical_lanes,
            font=font,
            settings=settings,
        )
        candidates.append(
            _canonical_candidate(
                orientation="VERTICAL",
                lanes=vertical_lanes,
                strokes=vertical,
                settings=settings,
                font=font,
            )
        )
    return candidates


def _balanced_horizontal_lanes(text: str, lane_count: int) -> tuple[str, ...]:
    words = text.split()
    if lane_count == 1:
        return (text,)
    if len(words) >= lane_count:
        return _balanced_word_lanes(words, lane_count)
    return _balanced_character_lanes("".join(words), lane_count)


def _balanced_word_lanes(
    words: list[str],
    lane_count: int,
) -> tuple[str, ...]:
    lanes: list[str] = []
    start = 0
    for lane_index in range(lane_count - 1):
        remaining_lanes = lane_count - lane_index
        maximum_end = len(words) - (remaining_lanes - 1)
        remaining_length = len(" ".join(words[start:]))
        target = remaining_length / remaining_lanes
        end = min(
            range(start + 1, maximum_end + 1),
            key=lambda candidate: (
                abs(len(" ".join(words[start:candidate])) - target),
                candidate,
            ),
        )
        lanes.append(" ".join(words[start:end]))
        start = end
    lanes.append(" ".join(words[start:]))
    return tuple(lanes)


def _balanced_character_lanes(
    text: str,
    lane_count: int,
) -> tuple[str, ...]:
    base, remainder = divmod(len(text), lane_count)
    lengths = [
        base + int(index < remainder) for index in range(lane_count)
    ]
    lanes: list[str] = []
    offset = 0
    for length in lengths:
        lanes.append(text[offset : offset + length])
        offset += length
    return tuple(lanes)


def _render_layout_centerlines(
    *,
    orientation: str,
    lanes: tuple[str, ...],
    font: _ResolvedFont,
    settings: TextTrajectoryConfig,
) -> tuple[_RawStroke, ...]:
    raster_scale = settings.skeleton_raster_scale
    font_size = max(1, int(round(settings.base_font_size * raster_scale)))
    try:
        raster_font = ImageFont.truetype(str(font.file_path), font_size)
    except OSError as error:
        raise TextTrajectoryGenerationError(
            f"Cannot rasterize font '{font.family_name}': {error}"
        ) from error
    line_advance = font_size * (1.0 + settings.line_gap_ratio)
    if orientation == "HORIZONTAL":
        placements = _horizontal_text_placements(
            lanes,
            font=raster_font,
            line_advance=line_advance,
        )
    else:
        placements = _vertical_text_placements(
            lanes,
            font=raster_font,
            line_advance=line_advance,
            column_gap=font_size * settings.line_gap_ratio,
        )
    glyph_mask = _render_placements(placements, font=raster_font)
    skeleton = _zhang_suen_thinning(glyph_mask)
    adjacency = _skeleton_adjacency(skeleton)
    _prune_short_spurs(
        adjacency,
        maximum_length=(
            settings.skeleton_spur_prune_length * raster_scale
        ),
    )
    paths = _trace_skeleton_paths(adjacency)
    strokes = tuple(
        _pixel_path_to_raw_stroke(
            path,
            settings=settings,
        )
        for path in paths
        if path
    )
    if not strokes:
        raise TextTrajectoryGenerationError(
            f"Font '{font.family_name}' produced no centerline for "
            f"{''.join(lanes)!r}"
        )
    return strokes


def _horizontal_text_placements(
    lanes: tuple[str, ...],
    *,
    font: ImageFont.FreeTypeFont,
    line_advance: float,
) -> tuple[_TextPlacement, ...]:
    bboxes = [_font_bbox(font, lane) for lane in lanes]
    maximum_width = max(right - left for left, _, right, _ in bboxes)
    placements: list[_TextPlacement] = []
    for index, (lane, bbox) in enumerate(zip(lanes, bboxes)):
        left, _, right, _ = bbox
        width = right - left
        placements.append(
            _TextPlacement(
                value=lane,
                baseline_x=(maximum_width - width) / 2.0 - left,
                baseline_y=index * line_advance,
            )
        )
    return tuple(placements)


def _vertical_text_placements(
    lanes: tuple[str, ...],
    *,
    font: ImageFont.FreeTypeFont,
    line_advance: float,
    column_gap: float,
) -> tuple[_TextPlacement, ...]:
    placements: list[_TextPlacement] = []
    column_left = 0.0
    for lane in lanes:
        entries = [
            (character, _font_bbox(font, character))
            for character in lane
        ]
        column_width = max(
            right - left for _, (left, _, right, _) in entries
        )
        for row_index, (character, bbox) in enumerate(entries):
            left, _, right, _ = bbox
            width = right - left
            placements.append(
                _TextPlacement(
                    value=character,
                    baseline_x=(
                        column_left
                        + (column_width - width) / 2.0
                        - left
                    ),
                    baseline_y=row_index * line_advance,
                )
            )
        column_left += column_width + column_gap
    return tuple(placements)


def _font_bbox(
    font: ImageFont.FreeTypeFont,
    text: str,
) -> tuple[int, int, int, int]:
    bbox = font.getbbox(text, anchor="ls")
    if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise TextTrajectoryGenerationError(
            f"Font produced an empty glyph box for {text!r}"
        )
    return bbox


def _render_placements(
    placements: tuple[_TextPlacement, ...],
    *,
    font: ImageFont.FreeTypeFont,
) -> BinaryImage:
    bounds = []
    for placement in placements:
        left, top, right, bottom = _font_bbox(font, placement.value)
        bounds.append(
            (
                placement.baseline_x + left,
                placement.baseline_y + top,
                placement.baseline_x + right,
                placement.baseline_y + bottom,
            )
        )
    minimum_x = min(bound[0] for bound in bounds)
    minimum_y = min(bound[1] for bound in bounds)
    maximum_x = max(bound[2] for bound in bounds)
    maximum_y = max(bound[3] for bound in bounds)
    padding = 4
    width = max(1, int(math.ceil(maximum_x - minimum_x)) + 2 * padding + 1)
    height = max(1, int(math.ceil(maximum_y - minimum_y)) + 2 * padding + 1)
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    shift_x = padding - minimum_x
    shift_y = padding - minimum_y
    for placement in placements:
        draw.text(
            (
                placement.baseline_x + shift_x,
                placement.baseline_y + shift_y,
            ),
            placement.value,
            font=font,
            fill=255,
            anchor="ls",
        )
    return np.asarray(image, dtype=np.uint8) >= 128


def _zhang_suen_thinning(image: BinaryImage) -> BinaryImage:
    skeleton = image.astype(bool, copy=True)
    maximum_iterations = max(skeleton.shape)
    for _ in range(maximum_iterations):
        changed = False
        for subiteration in (0, 1):
            neighbors = _binary_neighbors(skeleton)
            neighbor_count = sum(
                neighbor.astype(np.uint8) for neighbor in neighbors
            )
            transitions = sum(
                (~first) & second
                for first, second in zip(
                    neighbors,
                    (*neighbors[1:], neighbors[0]),
                )
            )
            p2, p4, p6, p8 = (
                neighbors[0],
                neighbors[2],
                neighbors[4],
                neighbors[6],
            )
            if subiteration == 0:
                first_product = p2 & p4 & p6
                second_product = p4 & p6 & p8
            else:
                first_product = p2 & p4 & p8
                second_product = p2 & p6 & p8
            removable = (
                skeleton
                & (neighbor_count >= 2)
                & (neighbor_count <= 6)
                & (transitions == 1)
                & ~first_product
                & ~second_product
            )
            if np.any(removable):
                skeleton[removable] = False
                changed = True
        if not changed:
            return skeleton
    raise TextTrajectoryGenerationError(
        "Font skeletonization did not converge within the image bound"
    )


def _binary_neighbors(image: BinaryImage) -> tuple[BinaryImage, ...]:
    padded = np.pad(image, 1, mode="constant", constant_values=False)
    return (
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, 2:],
        padded[2:, 2:],
        padded[2:, 1:-1],
        padded[2:, :-2],
        padded[1:-1, :-2],
        padded[:-2, :-2],
    )


def _skeleton_adjacency(
    skeleton: BinaryImage,
) -> dict[Pixel, set[Pixel]]:
    nodes = {
        (int(x), int(y)) for y, x in np.argwhere(skeleton)
    }
    adjacency = {node: set() for node in nodes}
    offsets = (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )
    for x, y in nodes:
        for delta_x, delta_y in offsets:
            neighbor = (x + delta_x, y + delta_y)
            if neighbor not in nodes:
                continue
            if delta_x != 0 and delta_y != 0:
                if (x + delta_x, y) in nodes or (x, y + delta_y) in nodes:
                    continue
            adjacency[(x, y)].add(neighbor)
    return adjacency


def _prune_short_spurs(
    adjacency: dict[Pixel, set[Pixel]],
    *,
    maximum_length: float,
) -> None:
    if maximum_length <= 0.0:
        return
    while True:
        removed = False
        endpoints = sorted(
            (node for node, neighbors in adjacency.items() if len(neighbors) == 1),
            key=_pixel_sort_key,
        )
        for endpoint in endpoints:
            if endpoint not in adjacency or len(adjacency[endpoint]) != 1:
                continue
            path = _walk_to_graph_vertex(adjacency, endpoint)
            terminal = path[-1]
            length = sum(
                math.dist(first, second)
                for first, second in zip(path, path[1:])
            )
            if len(adjacency.get(terminal, ())) < 3 or length > maximum_length:
                continue
            _remove_graph_nodes(adjacency, path[:-1])
            removed = True
            break
        if not removed:
            return


def _walk_to_graph_vertex(
    adjacency: dict[Pixel, set[Pixel]],
    start: Pixel,
) -> list[Pixel]:
    path = [start]
    previous: Pixel | None = None
    current = start
    while True:
        candidates = sorted(
            (neighbor for neighbor in adjacency[current] if neighbor != previous),
            key=_pixel_sort_key,
        )
        if not candidates:
            return path
        following = candidates[0]
        path.append(following)
        previous, current = current, following
        if len(adjacency[current]) != 2:
            return path


def _remove_graph_nodes(
    adjacency: dict[Pixel, set[Pixel]],
    nodes: list[Pixel],
) -> None:
    for node in nodes:
        if node not in adjacency:
            continue
        for neighbor in tuple(adjacency[node]):
            adjacency[neighbor].discard(node)
        del adjacency[node]


def _trace_skeleton_paths(
    adjacency: dict[Pixel, set[Pixel]],
) -> list[list[Pixel]]:
    paths = [
        [node]
        for node, neighbors in sorted(
            adjacency.items(),
            key=lambda item: _pixel_sort_key(item[0]),
        )
        if not neighbors
    ]
    visited: set[tuple[Pixel, Pixel]] = set()
    graph_vertices = sorted(
        (
            node
            for node, neighbors in adjacency.items()
            if len(neighbors) != 2 and neighbors
        ),
        key=_pixel_sort_key,
    )
    for start in graph_vertices:
        for following in sorted(adjacency[start], key=_pixel_sort_key):
            edge = _edge_key(start, following)
            if edge in visited:
                continue
            paths.append(
                _walk_unvisited_edges(
                    adjacency,
                    start=start,
                    following=following,
                    visited=visited,
                )
            )
    all_edges = sorted(
        {
            _edge_key(node, neighbor)
            for node, neighbors in adjacency.items()
            for neighbor in neighbors
        },
        key=lambda edge: (
            _pixel_sort_key(edge[0]),
            _pixel_sort_key(edge[1]),
        ),
    )
    for start, following in all_edges:
        if _edge_key(start, following) in visited:
            continue
        paths.append(
            _walk_unvisited_edges(
                adjacency,
                start=start,
                following=following,
                visited=visited,
            )
        )
    return paths


def _walk_unvisited_edges(
    adjacency: dict[Pixel, set[Pixel]],
    *,
    start: Pixel,
    following: Pixel,
    visited: set[tuple[Pixel, Pixel]],
) -> list[Pixel]:
    path = [start, following]
    visited.add(_edge_key(start, following))
    previous = start
    current = following
    while current != start and len(adjacency[current]) == 2:
        candidates = sorted(
            (neighbor for neighbor in adjacency[current] if neighbor != previous),
            key=_pixel_sort_key,
        )
        if not candidates:
            break
        next_node = candidates[0]
        edge = _edge_key(current, next_node)
        if edge in visited:
            break
        visited.add(edge)
        path.append(next_node)
        previous, current = current, next_node
    return path


def _edge_key(first: Pixel, second: Pixel) -> tuple[Pixel, Pixel]:
    return tuple(sorted((first, second), key=_pixel_sort_key))


def _pixel_sort_key(pixel: Pixel) -> tuple[int, int]:
    return pixel[1], pixel[0]


def _pixel_path_to_raw_stroke(
    path: list[Pixel],
    *,
    settings: TextTrajectoryConfig,
) -> _RawStroke:
    closed = len(path) > 2 and path[0] == path[-1]
    simplified = _simplify_pixel_path(
        path,
        closed=closed,
        tolerance=(
            settings.centerline_simplify_tolerance
            * settings.skeleton_raster_scale
        ),
    )
    sampled = _densify_polyline(
        simplified,
        closed=closed,
        spacing=(
            settings.centerline_sample_spacing
            * settings.skeleton_raster_scale
        ),
    )
    scale = settings.skeleton_raster_scale
    return _RawStroke(
        points=tuple((x / scale, y / scale) for x, y in sampled),
        closed=closed,
    )


def _simplify_pixel_path(
    path: list[Pixel],
    *,
    closed: bool,
    tolerance: float,
) -> list[Point]:
    points = path[:-1] if closed else path
    if len(points) <= 2 or tolerance <= 0.0:
        simplified = [(float(x), float(y)) for x, y in points]
    else:
        coordinates = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
        approximation = cv2.approxPolyDP(
            coordinates,
            epsilon=tolerance,
            closed=closed,
        )
        simplified = [
            (float(point[0][0]), float(point[0][1]))
            for point in approximation
        ]
        if closed and len(simplified) < 3:
            simplified = [(float(x), float(y)) for x, y in points]
    if closed and simplified:
        simplified.append(simplified[0])
    return simplified


def _densify_polyline(
    points: list[Point],
    *,
    closed: bool,
    spacing: float,
) -> list[Point]:
    if len(points) <= 1:
        return points
    source = points[:-1] if closed and points[0] == points[-1] else points
    if len(source) <= 1:
        return source
    segments = [*source, source[0]] if closed else source
    result = [segments[0]]
    for start, end in zip(segments, segments[1:]):
        length = math.dist(start, end)
        steps = max(1, int(math.ceil(length / spacing)))
        for step in range(1, steps + 1):
            ratio = step / steps
            result.append(
                (
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                )
            )
    if closed:
        result[-1] = result[0]
    return result


def _canonical_candidate(
    *,
    orientation: str,
    lanes: tuple[str, ...],
    strokes: tuple[_RawStroke, ...],
    settings: TextTrajectoryConfig,
    font: _ResolvedFont,
) -> _LayoutCandidate:
    if not strokes:
        raise TextTrajectoryGenerationError("text layout has no centerlines")
    all_points = [point for stroke in strokes for point in stroke.points]
    minimum_x = min(point[0] for point in all_points)
    maximum_x = max(point[0] for point in all_points)
    minimum_y = min(point[1] for point in all_points)
    maximum_y = max(point[1] for point in all_points)
    span_x = maximum_x - minimum_x
    span_y = maximum_y - minimum_y
    available = _CANVAS_SIZE - 2.0 * settings.canonical_canvas_margin
    scale_bounds = [
        available / span
        for span in (span_x, span_y)
        if span > _POINT_EPSILON
    ]
    canonical_scale = min(scale_bounds) if scale_bounds else 1.0
    center_x = (minimum_x + maximum_x) / 2.0
    center_y = (minimum_y + maximum_y) / 2.0
    trajectories: list[dict[str, object]] = []
    orientation_slug = orientation.casefold()
    for index, stroke in enumerate(strokes, start=1):
        points = [
            {
                "x": _canonical_coordinate(
                    (point[0] - center_x) * canonical_scale
                    + _CANVAS_SIZE / 2.0
                ),
                "y": _canonical_coordinate(
                    (point[1] - center_y) * canonical_scale
                    + _CANVAS_SIZE / 2.0
                ),
            }
            for point in stroke.points
        ]
        trajectories.append(
            {
                "id": f"text-{orientation_slug}-{len(lanes):02d}-{index:04d}",
                "source": {
                    "kind": "font_centerline",
                    "font_family": font.family_name,
                    "orientation": orientation,
                    "skeletonization": "ZHANG_SUEN",
                    "graph_decomposition": (
                        "PATHS_BETWEEN_ENDPOINTS_JUNCTIONS_OR_CYCLES"
                    ),
                },
                "closed": stroke.closed,
                "points": points,
            }
        )
    return _LayoutCandidate(
        orientation=orientation,
        lanes=lanes,
        trajectory_plan={
            "format": "svg-mouse-trajectories/v1",
            "trajectories": trajectories,
        },
        canonical_scale=canonical_scale,
    )


def _canonical_coordinate(value: float) -> float:
    bounded = min(_CANVAS_SIZE, max(0.0, value))
    rounded = round(bounded, 6)
    return 0.0 if rounded == -0.0 else rounded


def _fitted_candidate_sort_key(
    candidate: _FittedCandidate,
) -> tuple[float, float, int, int, tuple[str, ...]]:
    return (
        -candidate.effective_font_size_pixels,
        candidate.center_offset_pixels,
        len(candidate.layout.lanes),
        0 if candidate.layout.orientation == "HORIZONTAL" else 1,
        candidate.layout.lanes,
    )


def _result_payload(
    *,
    text: str,
    font: _ResolvedFont,
    selected: _FittedCandidate,
    scale_tier: str,
    candidate_count: int,
    fitted_candidate_count: int,
) -> dict[str, JsonValue]:
    fit = selected.fitted_plan
    orientation = selected.layout.orientation
    lane_kind = "LINES" if orientation == "HORIZONTAL" else "COLUMNS"
    return {
        "format": "mask-fitted-text-mouse-trajectories/v1",
        "algorithm": {
            "name": "font_centerline_mask_layout_search",
            "deterministic": True,
            "trajectory_semantics": "FONT_STEM_CENTERLINE",
            "glyph_rasterization": "FREETYPE_FILLED_GLYPHS",
            "skeletonization": "ZHANG_SUEN",
            "spur_pruning": "SHORT_ENDPOINT_BRANCHES_ONLY",
            "graph_decomposition": (
                "PATHS_BETWEEN_ENDPOINTS_JUNCTIONS_OR_CYCLES"
            ),
            "objective_priority": [
                "SATISFY_SCALE_TIER_BOUNDARY_CLEARANCE",
                "MAXIMIZE_EFFECTIVE_FONT_SIZE_WITHIN_TIER",
                "MINIMIZE_DISTANCE_TO_MASK_CENTER",
                "MINIMIZE_LINE_OR_COLUMN_COUNT",
                "PREFER_HORIZONTAL_ON_EXACT_TIE",
            ],
            "complete_polyline_containment": True,
            "font_fallback_allowed": False,
        },
        "text": {
            "value": text,
            "language": "ENGLISH_ASCII",
            "requested_font_family": font.requested_name,
            "resolved_font_family": font.family_name,
            "resolved_font_file_name": font.file_path.name,
        },
        "layout": {
            "orientation": orientation,
            "flow": (
                "LEFT_TO_RIGHT_TOP_TO_BOTTOM"
                if orientation == "HORIZONTAL"
                else "TOP_TO_BOTTOM_LEFT_TO_RIGHT"
            ),
            "lane_kind": lane_kind,
            "lane_count": len(selected.layout.lanes),
            "lanes": list(selected.layout.lanes),
            "effective_font_size_pixels": round(
                selected.effective_font_size_pixels,
                6,
            ),
            "center_offset_pixels": round(
                selected.center_offset_pixels,
                6,
            ),
            "scale_tier": scale_tier,
            "candidate_count": candidate_count,
            "fitted_candidate_count": fitted_candidate_count,
        },
        "coordinate_system": fit["coordinate_system"],
        "gesture_contract": fit["gesture_contract"],
        "transform": fit["transform"],
        "sizing": fit["sizing"],
        "mask": fit["mask"],
        "containment": fit["containment"],
        "trajectories": fit["trajectories"],
        "summary": fit["summary"],
    }
