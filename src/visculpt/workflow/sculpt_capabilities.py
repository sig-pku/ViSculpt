"""Runtime Blender Sculpt brush capability catalog and validation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from visculpt.bridge import JsonValue


class SculptCapabilityError(ValueError):
    """Raised when Blender capability data or a selected value is invalid."""


class SculptBrushCapability(BaseModel):
    """One exact local brush name and its context-sensitive directions."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    direction_values: list[str] = Field(max_length=64)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Keep the exact Blender asset name while rejecting control text."""
        name = value.strip()
        if not name or any(ord(character) < 32 for character in name):
            raise ValueError("brush name is invalid")
        return name

    @field_validator("direction_values")
    @classmethod
    def validate_directions(cls, values: list[str]) -> list[str]:
        """Normalize Blender enum identifiers without changing their order."""
        normalized: list[str] = []
        for value in values:
            direction = value.strip().upper()
            if (
                not direction
                or len(direction) > 64
                or any(ord(character) < 32 for character in direction)
            ):
                raise ValueError("brush direction is invalid")
            if direction not in normalized:
                normalized.append(direction)
        return normalized


class BlenderSculptCapabilities(BaseModel):
    """Compact, JSON-safe capability snapshot stored in LangGraph State."""

    model_config = ConfigDict(extra="forbid")

    blender_version: str = Field(min_length=1, max_length=128)
    blender_version_tuple: list[int] = Field(min_length=3, max_length=4)
    brush_count: int = Field(gt=0)
    brushes: list[SculptBrushCapability] = Field(min_length=1)
    inventory_scan: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_catalog(self) -> Self:
        """Require internally consistent and unambiguous exact brush names."""
        if self.brush_count != len(self.brushes):
            raise ValueError("brush_count does not match brushes")
        names = [brush.name for brush in self.brushes]
        if len(names) != len(set(names)):
            raise ValueError("brush names must be unique")
        return self

    def canonical_brush_name(self, candidate: str) -> str:
        """Resolve a model-selected name to the exact local asset spelling."""
        stripped = candidate.strip()
        exact = [
            brush.name
            for brush in self.brushes
            if brush.name == stripped
        ]
        if exact:
            return exact[0]
        folded = [
            brush.name
            for brush in self.brushes
            if brush.name.casefold() == stripped.casefold()
        ]
        if len(folded) == 1:
            return folded[0]
        raise SculptCapabilityError(
            f"Sculpt brush is not uniquely available locally: {candidate}"
        )

    def direction_values_for(self, brush_name: str) -> tuple[str, ...]:
        """Return the exact Direction choices for one canonical brush name."""
        canonical = self.canonical_brush_name(brush_name)
        for brush in self.brushes:
            if brush.name == canonical:
                return tuple(brush.direction_values)
        raise AssertionError("canonical brush disappeared from catalog")

    def normalize_direction(
        self,
        brush_name: str,
        selected_direction: str | None,
    ) -> str | None:
        """Map one selected Direction to the exact runtime-safe value."""
        allowed = self.direction_values_for(brush_name)
        if not allowed:
            return None
        if len(allowed) == 1:
            return allowed[0]
        if selected_direction is None:
            raise SculptCapabilityError(
                f"Sculpt brush {brush_name} requires one of: "
                + ", ".join(allowed)
            )
        normalized = selected_direction.strip().upper()
        if normalized not in allowed:
            raise SculptCapabilityError(
                f"Direction {selected_direction} is not valid for "
                f"{brush_name}; expected one of: " + ", ".join(allowed)
            )
        return normalized

    def brush_from_subtask_description(self, description: str) -> str:
        """Read the exact catalog brush from the required sentence prefix."""
        exact_matches = [
            brush.name
            for brush in self.brushes
            if description.startswith(f"Use the {brush.name} brush at ")
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]
        matches = [
            brush.name
            for brush in self.brushes
            if _subtask_brush_prefix(brush.name).match(description)
        ]
        if len(matches) > 1:
            raise SculptCapabilityError(
                "Subtask brush name is ambiguous in the local catalog"
            )
        if not matches:
            raise SculptCapabilityError(
                "Subtask description does not use an available Sculpt brush"
            )
        return matches[0]

    def canonicalize_subtask_description(self, description: str) -> str:
        """Restore exact local brush spelling in a valid subtask sentence."""
        canonical = self.brush_from_subtask_description(description)
        match = _subtask_brush_prefix(canonical).match(description)
        if match is None:
            raise SculptCapabilityError(
                "Subtask description has an invalid brush prefix"
            )
        prefix = f"Use the {canonical} brush at "
        return prefix + description[match.end():]


def _subtask_brush_prefix(brush_name: str) -> re.Pattern[str]:
    return re.compile(
        rf"^Use the {re.escape(brush_name)} brush at ",
        flags=re.IGNORECASE,
    )


def parse_get_state_sculpt_capabilities(
    response: JsonValue,
) -> BlenderSculptCapabilities:
    """Extract and validate the capability subset of a get_state envelope."""
    if not isinstance(response, Mapping):
        raise SculptCapabilityError("get_state response must be an object")
    error = response.get("error")
    if isinstance(error, Mapping):
        message = error.get("message")
        raise SculptCapabilityError(
            "get_state returned an RPC error"
            + (f": {message}" if isinstance(message, str) else "")
        )
    result = response.get("result")
    if not isinstance(result, Mapping):
        raise SculptCapabilityError("get_state response is missing result")
    blender = result.get("blender")
    sculpt = result.get("sculpt")
    if not isinstance(blender, Mapping) or not isinstance(sculpt, Mapping):
        raise SculptCapabilityError(
            "get_state result is missing blender or sculpt metadata"
        )
    version = blender.get("version")
    version_tuple = blender.get("version_tuple")
    details = sculpt.get("available_brush_details")
    if not isinstance(version, str) or not isinstance(version_tuple, list):
        raise SculptCapabilityError("get_state has invalid Blender version")
    if not isinstance(details, list) or not details:
        raise SculptCapabilityError(
            "get_state did not return any available Sculpt brushes"
        )

    brushes: list[SculptBrushCapability] = []
    for detail in details:
        if not isinstance(detail, Mapping):
            raise SculptCapabilityError("brush detail must be an object")
        name = detail.get("name")
        directions = detail.get("direction_values")
        if not isinstance(name, str) or not isinstance(directions, list):
            raise SculptCapabilityError(
                "brush detail is missing name or direction_values"
            )
        if not all(isinstance(value, str) for value in directions):
            raise SculptCapabilityError(
                f"brush {name} contains a non-string Direction value"
            )
        brushes.append(
            SculptBrushCapability(
                name=name,
                direction_values=directions,
            )
        )

    scan = sculpt.get("brush_inventory")
    inventory_scan = _inventory_scan_summary(scan)
    try:
        parsed_version_tuple = [int(value) for value in version_tuple]
    except (TypeError, ValueError) as error:
        raise SculptCapabilityError(
            "get_state has an invalid Blender version tuple"
        ) from error
    return BlenderSculptCapabilities(
        blender_version=version,
        blender_version_tuple=parsed_version_tuple,
        brush_count=len(brushes),
        brushes=brushes,
        inventory_scan=inventory_scan,
    )


def _inventory_scan_summary(value: object) -> dict[str, JsonValue]:
    """Keep scan health without persisting machine-specific asset paths."""
    if not isinstance(value, Mapping):
        return {}
    summary: dict[str, JsonValue] = {}
    complete = value.get("complete")
    if isinstance(complete, bool):
        summary["complete"] = complete
    for key in (
        "library_file_count",
        "scanned_file_count",
        "cached_file_count",
    ):
        count = value.get(key)
        if isinstance(count, int) and not isinstance(count, bool):
            summary[key] = count
    errors = value.get("errors")
    summary["error_count"] = len(errors) if isinstance(errors, list) else 0
    return summary
