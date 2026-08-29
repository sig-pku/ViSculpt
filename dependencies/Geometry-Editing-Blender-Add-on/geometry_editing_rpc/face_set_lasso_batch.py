"""Validate transactional batches of Sculpt Face Set lasso paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NoReturn

from .face_set_lasso import (
    FaceSetLassoRequest,
    parse_face_set_lasso_request,
)
from .protocol import JsonRpcError

MAX_FACE_SET_LASSO_BATCH_PATHS = 32

_REQUEST_FIELDS = {
    "paths",
    "padding_pixels",
    "rollback_on_residual",
    "use_front_faces_only",
    "show_face_sets",
    "window_index",
    "area_index",
}
_PATH_FIELDS = {"label", "path"}


@dataclass(frozen=True, slots=True)
class LabeledFaceSetLasso:
    """One validated path and its stable diagnostic label."""

    label: str
    request: FaceSetLassoRequest


@dataclass(frozen=True, slots=True)
class FaceSetLassoBatchRequest:
    """Validated transactional Face Set lasso batch."""

    paths: tuple[LabeledFaceSetLasso, ...]
    padding_pixels: int
    rollback_on_residual: bool
    show_face_sets: bool
    window_index: int | None
    area_index: int | None


def parse_face_set_lasso_batch_request(
    params: dict[str, Any],
    *,
    region_width: int,
    region_height: int,
) -> FaceSetLassoBatchRequest:
    """Validate a bounded batch using the single-lasso parser."""
    unknown = sorted(set(params) - _REQUEST_FIELDS)
    if unknown:
        _invalid("request contains unknown parameters", unknown=unknown)
    raw_paths = params.get("paths")
    if not isinstance(raw_paths, list):
        _invalid("paths must be a JSON array")
    if not raw_paths:
        _invalid("paths must contain at least one lasso")
    if len(raw_paths) > MAX_FACE_SET_LASSO_BATCH_PATHS:
        _invalid(
            "paths exceeds the batch safety limit",
            maximum=MAX_FACE_SET_LASSO_BATCH_PATHS,
        )
    padding_pixels = _integer(
        params.get("padding_pixels"),
        "padding_pixels",
        minimum=1,
        maximum=192,
    )
    rollback_on_residual = _boolean(
        params.get("rollback_on_residual", True),
        "rollback_on_residual",
    )
    show_face_sets = _boolean(
        params.get("show_face_sets", False),
        "show_face_sets",
    )
    common = {
        "use_front_faces_only": params.get(
            "use_front_faces_only",
            False,
        ),
        "show_face_sets": show_face_sets,
    }
    if "window_index" in params:
        common["window_index"] = params["window_index"]
    if "area_index" in params:
        common["area_index"] = params["area_index"]

    paths: list[LabeledFaceSetLasso] = []
    labels: set[str] = set()
    for index, raw_item in enumerate(raw_paths):
        if not isinstance(raw_item, dict):
            _invalid("each paths item must be an object", path_index=index)
        item_unknown = sorted(set(raw_item) - _PATH_FIELDS)
        if item_unknown:
            _invalid(
                "paths item contains unknown fields",
                path_index=index,
                unknown=item_unknown,
            )
        label = raw_item.get("label")
        if not isinstance(label, str) or not label.strip():
            _invalid(
                "paths item label must be a non-empty string",
                path_index=index,
            )
        normalized_label = " ".join(label.split())
        if len(normalized_label) > 128 or any(
            ord(character) < 32 for character in normalized_label
        ):
            _invalid(
                "paths item label must be printable and at most 128 characters",
                path_index=index,
            )
        folded = normalized_label.casefold()
        if folded in labels:
            _invalid("paths item labels must be unique", label=normalized_label)
        labels.add(folded)
        request = parse_face_set_lasso_request(
            {**common, "path": raw_item.get("path")},
            region_width=region_width,
            region_height=region_height,
        )
        paths.append(
            LabeledFaceSetLasso(
                label=normalized_label,
                request=request,
            )
        )

    first = paths[0].request
    return FaceSetLassoBatchRequest(
        paths=tuple(paths),
        padding_pixels=padding_pixels,
        rollback_on_residual=rollback_on_residual,
        show_face_sets=show_face_sets,
        window_index=first.window_index,
        area_index=first.area_index,
    )


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        _invalid(f"{name} must be a boolean")
    return value


def _integer(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _invalid(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        _invalid(
            f"{name} is outside the supported range",
            minimum=minimum,
            maximum=maximum,
        )
    return value


def _invalid(reason: str, **data: Any) -> NoReturn:
    raise JsonRpcError(-32602, "Invalid params", {"reason": reason, **data})
