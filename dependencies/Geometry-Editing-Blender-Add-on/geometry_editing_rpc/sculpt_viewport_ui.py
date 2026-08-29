"""Validation contract for restorable Sculpt viewport UI snapshots."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .protocol import JsonRpcError

SCULPT_VIEWPORT_UI_SCHEMA_VERSION = "1.0"

SPACE_UI_PROPERTIES = (
    "show_region_asset_shelf",
    "show_region_toolbar",
    "show_region_ui",
    "show_region_header",
    "show_region_tool_header",
    "show_gizmo_navigate",
)

OVERLAY_UI_PROPERTIES = (
    "show_floor",
    "show_ortho_grid",
    "show_text",
    "show_stats",
)

SPACE_UI_REQUIRED_REGION_TYPES = {
    "show_region_asset_shelf": "ASSET_SHELF",
    "show_region_toolbar": "TOOLS",
    "show_region_ui": "UI",
    "show_region_header": "HEADER",
    "show_region_tool_header": "TOOL_HEADER",
}

DYNAMIC_REGION_TOGGLE_FALLBACKS = {
    ("space", "show_region_asset_shelf"): "ASSET_SHELF",
}


@dataclass
class StabilityTracker:
    """Require the same ready-state signature across consecutive checks."""

    required_consecutive: int = 2
    checks: int = 0
    consecutive: int = 0
    last_signature: Any = None

    def __post_init__(self) -> None:
        if self.required_consecutive < 1:
            raise ValueError("required_consecutive must be positive")

    def observe(self, signature: Any | None) -> bool:
        """Record one check and report whether the state is stable."""
        self.checks += 1
        if signature is None:
            self.consecutive = 0
            self.last_signature = None
            return False
        if signature == self.last_signature:
            self.consecutive += 1
        else:
            self.last_signature = signature
            self.consecutive = 1
        return self.consecutive >= self.required_consecutive


def apply_sculpt_viewport_ui_properties(
    owners: Mapping[str, Any],
    state: Mapping[str, Mapping[str, bool]],
    *,
    region_toggle: Callable[[str], None],
    blender_version: str,
) -> dict[str, str]:
    """Apply UI values once with bounded region fallbacks."""
    methods: dict[str, str] = {}
    for category in ("space", "overlay"):
        owner = owners.get(category)
        for name, expected in state[category].items():
            path = f"{category}.{name}"
            if owner is None or not hasattr(owner, name):
                raise _update_error(
                    "Sculpt viewport UI setting is unavailable",
                    property=path,
                    blender_version=blender_version,
                )
            current = bool(getattr(owner, name))
            if current == expected:
                methods[path] = "unchanged"
                continue

            fallback = DYNAMIC_REGION_TOGGLE_FALLBACKS.get(
                (category, name)
            )
            readonly = _is_property_readonly(owner, name)
            if readonly and fallback is None:
                raise _update_error(
                    "Sculpt viewport UI setting is read-only",
                    property=path,
                    blender_version=blender_version,
                )
            if readonly:
                _toggle_region(
                    region_toggle,
                    fallback,
                    property_path=path,
                    blender_version=blender_version,
                )
                methods[path] = "region_toggle"
                continue

            try:
                setattr(owner, name, expected)
                methods[path] = "rna_setattr"
            except (AttributeError, RuntimeError, TypeError, ValueError) as error:
                if fallback is None:
                    raise _update_error(
                        "Cannot update Sculpt viewport UI",
                        property=path,
                        blender_version=blender_version,
                        reason=str(error),
                    ) from error
                # The Asset Shelf may become dynamically read-only after checking.
                if bool(getattr(owner, name)) != expected:
                    _toggle_region(
                        region_toggle,
                        fallback,
                        property_path=path,
                        blender_version=blender_version,
                    )
                methods[path] = "region_toggle"

    return methods


def read_sculpt_viewport_ui_properties(
    owners: Mapping[str, Any],
    state: Mapping[str, Mapping[str, bool]],
) -> tuple[
    dict[str, dict[str, bool | None]],
    dict[str, dict[str, bool | None]],
]:
    """Read expected properties and return JSON-safe mismatches."""
    actual_state: dict[str, dict[str, bool | None]] = {
        "space": {},
        "overlay": {},
    }
    mismatches: dict[str, dict[str, bool | None]] = {}
    for category in ("space", "overlay"):
        owner = owners.get(category)
        for name, expected in state[category].items():
            actual = (
                bool(getattr(owner, name))
                if owner is not None and hasattr(owner, name)
                else None
            )
            actual_state[category][name] = actual
            if actual != expected:
                mismatches[f"{category}.{name}"] = {
                    "expected": expected,
                    "actual": actual,
                }
    return actual_state, mismatches


def parse_sculpt_viewport_ui_snapshot(value: Any) -> dict[str, Any]:
    """Return one strict, bounded viewport UI snapshot."""
    snapshot = _object(value, name="snapshot")
    _reject_unknown(
        snapshot,
        {
            "schema_version",
            "window_index",
            "area_index",
            "space",
            "overlay",
        },
        name="snapshot",
    )
    if snapshot.get("schema_version") != SCULPT_VIEWPORT_UI_SCHEMA_VERSION:
        raise _invalid(
            "snapshot.schema_version is unsupported",
            supported_values=[SCULPT_VIEWPORT_UI_SCHEMA_VERSION],
        )
    window_index = _non_negative_integer(
        snapshot.get("window_index"),
        name="snapshot.window_index",
    )
    area_index = _non_negative_integer(
        snapshot.get("area_index"),
        name="snapshot.area_index",
    )
    space = _boolean_properties(
        snapshot.get("space"),
        name="snapshot.space",
        allowed=SPACE_UI_PROPERTIES,
    )
    overlay = _boolean_properties(
        snapshot.get("overlay"),
        name="snapshot.overlay",
        allowed=OVERLAY_UI_PROPERTIES,
    )
    if not space and not overlay:
        raise _invalid("snapshot contains no restorable UI properties")
    return {
        "schema_version": SCULPT_VIEWPORT_UI_SCHEMA_VERSION,
        "window_index": window_index,
        "area_index": area_index,
        "space": space,
        "overlay": overlay,
    }


def _boolean_properties(
    value: Any,
    *,
    name: str,
    allowed: tuple[str, ...],
) -> dict[str, bool]:
    payload = _object(value, name=name)
    _reject_unknown(payload, set(allowed), name=name)
    result: dict[str, bool] = {}
    for key, item in payload.items():
        if not isinstance(item, bool):
            raise _invalid(f"{name}.{key} must be a boolean")
        result[key] = item
    return result


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _invalid(f"{name} must be an object")
    return value


def _non_negative_integer(value: Any, *, name: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise _invalid(f"{name} must be a non-negative integer")
    return value


def _reject_unknown(
    payload: dict[str, Any],
    allowed: set[str],
    *,
    name: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise _invalid(
            f"{name} contains unknown properties",
            unknown_properties=unknown,
        )


def _is_property_readonly(owner: Any, name: str) -> bool:
    checker = getattr(owner, "is_property_readonly", None)
    if not callable(checker):
        return False
    try:
        return bool(checker(name))
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise _update_error(
            "Cannot inspect Sculpt viewport UI setting",
            property=name,
            reason=str(error),
        ) from error


def _toggle_region(
    callback: Callable[[str], None],
    region_type: str | None,
    *,
    property_path: str,
    blender_version: str,
) -> None:
    if region_type is None:
        raise _update_error(
            "Sculpt viewport UI setting has no region fallback",
            property=property_path,
            blender_version=blender_version,
        )
    try:
        callback(region_type)
    except JsonRpcError:
        raise
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise _update_error(
            "Cannot toggle Sculpt viewport UI region",
            property=property_path,
            region_type=region_type,
            blender_version=blender_version,
            reason=str(error),
        ) from error


def _update_error(reason: str, **data: Any) -> JsonRpcError:
    return JsonRpcError(-32027, reason, data or None)


def _invalid(reason: str, **data: Any) -> JsonRpcError:
    return JsonRpcError(
        -32602,
        "Invalid params",
        {"reason": reason, **data},
    )
