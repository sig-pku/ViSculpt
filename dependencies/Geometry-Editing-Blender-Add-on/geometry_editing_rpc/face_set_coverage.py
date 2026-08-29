"""Deterministic topology checks for generated Sculpt Face Sets."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any


def analyze_face_set_coverage(
    *,
    baseline_values: Sequence[int],
    current_values: Sequence[int],
    centers: Sequence[float],
    polygon_loop_starts: Sequence[int],
    polygon_loop_totals: Sequence[int],
    loop_edges: Sequence[int],
    generated_face_set_ids: Sequence[int],
) -> dict[str, Any]:
    """Find unchanged face islands enclosed by generated Face Sets."""
    face_count = len(current_values)
    _validate_lengths(
        face_count=face_count,
        baseline_values=baseline_values,
        centers=centers,
        polygon_loop_starts=polygon_loop_starts,
        polygon_loop_totals=polygon_loop_totals,
    )
    normalized_ids = tuple(
        dict.fromkeys(abs(int(value)) for value in generated_face_set_ids)
    )
    unchanged = bytearray(
        abs(int(before)) == abs(int(after))
        for before, after in zip(
            baseline_values,
            current_values,
            strict=True,
        )
    )
    residual_indices: set[int] = set()
    face_sets: list[dict[str, Any]] = []
    empty_face_set_ids: list[int] = []

    for face_set_id in normalized_ids:
        selected = [
            index
            for index, value in enumerate(current_values)
            if abs(int(value)) == face_set_id
        ]
        if not selected:
            empty_face_set_ids.append(face_set_id)
            face_sets.append(
                {
                    "face_set_id": face_set_id,
                    "assigned_face_count": 0,
                    "candidate_original_face_count": 0,
                    "residual_original_face_count": 0,
                    "residual_component_count": 0,
                    "residual_component_sizes": [],
                    "bounds": None,
                }
            )
            continue

        lower, upper = _face_center_bounds(selected, centers)
        candidates = {
            index
            for index in range(face_count)
            if unchanged[index]
            and _center_inside_bounds(
                index,
                centers=centers,
                lower=lower,
                upper=upper,
            )
        }
        components, touches_unchanged_outside, touches_generated = (
            _candidate_components(
                candidates=candidates,
                unchanged=unchanged,
                polygon_loop_starts=polygon_loop_starts,
                polygon_loop_totals=polygon_loop_totals,
                loop_edges=loop_edges,
            )
        )
        residual_components = [
            members
            for root, members in components.items()
            if root not in touches_unchanged_outside
            and root in touches_generated
        ]
        current_residual = {
            index
            for members in residual_components
            for index in members
        }
        residual_indices.update(current_residual)
        face_sets.append(
            {
                "face_set_id": face_set_id,
                "assigned_face_count": len(selected),
                "candidate_original_face_count": len(candidates),
                "residual_original_face_count": len(current_residual),
                "residual_component_count": len(residual_components),
                "residual_component_sizes": sorted(
                    len(members) for members in residual_components
                ),
                "bounds": {
                    "min": lower,
                    "max": upper,
                    "space": "OBJECT_LOCAL_FACE_CENTERS",
                },
            }
        )

    return {
        "algorithm": "enclosed-unchanged-face-islands/v1",
        "complete": not residual_indices and not empty_face_set_ids,
        "residual_original_face_count": len(residual_indices),
        "empty_face_set_ids": empty_face_set_ids,
        "generated_face_set_ids": list(normalized_ids),
        "face_sets": face_sets,
    }


def _validate_lengths(
    *,
    face_count: int,
    baseline_values: Sequence[int],
    centers: Sequence[float],
    polygon_loop_starts: Sequence[int],
    polygon_loop_totals: Sequence[int],
) -> None:
    """Reject inconsistent mesh arrays before topology traversal."""
    if len(baseline_values) != face_count:
        raise ValueError("baseline and current Face Set lengths differ")
    if len(centers) != face_count * 3:
        raise ValueError("face center array must contain three values per face")
    if len(polygon_loop_starts) != face_count:
        raise ValueError("polygon loop_start length differs from face count")
    if len(polygon_loop_totals) != face_count:
        raise ValueError("polygon loop_total length differs from face count")


def _face_center_bounds(
    indices: Sequence[int],
    centers: Sequence[float],
) -> tuple[list[float], list[float]]:
    """Return an axis-aligned local-space box for selected face centers."""
    lower = [
        min(float(centers[index * 3 + axis]) for index in indices)
        for axis in range(3)
    ]
    upper = [
        max(float(centers[index * 3 + axis]) for index in indices)
        for axis in range(3)
    ]
    return lower, upper


def _center_inside_bounds(
    index: int,
    *,
    centers: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
) -> bool:
    """Return whether one face center lies in an inclusive AABB."""
    offset = index * 3
    return all(
        lower[axis] <= float(centers[offset + axis]) <= upper[axis]
        for axis in range(3)
    )


def _candidate_components(
    *,
    candidates: set[int],
    unchanged: Sequence[int],
    polygon_loop_starts: Sequence[int],
    polygon_loop_totals: Sequence[int],
    loop_edges: Sequence[int],
) -> tuple[dict[int, list[int]], set[int], set[int]]:
    """Classify candidate components by topology across their boundary."""
    if not candidates:
        return {}, set(), set()
    parents = {index: index for index in candidates}

    def find(index: int) -> int:
        root = index
        while root != parents[root]:
            root = parents[root]
        while index != root:
            next_index = parents[index]
            parents[index] = root
            index = next_index
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    edge_owners: dict[int, list[int]] = defaultdict(list)
    for index in candidates:
        start = int(polygon_loop_starts[index])
        stop = start + int(polygon_loop_totals[index])
        for loop_index in range(start, stop):
            edge_owners[int(loop_edges[loop_index])].append(index)
    for owners in edge_owners.values():
        for other in owners[1:]:
            union(owners[0], other)

    # Scan every edge to support non-manifold meshes with three or more faces.
    candidate_edges = {
        edge: owners[0] for edge, owners in edge_owners.items()
    }
    touches_unchanged_outside: set[int] = set()
    touches_generated: set[int] = set()
    for index in range(len(unchanged)):
        if index in candidates:
            continue
        start = int(polygon_loop_starts[index])
        stop = start + int(polygon_loop_totals[index])
        for loop_index in range(start, stop):
            owner = candidate_edges.get(int(loop_edges[loop_index]))
            if owner is None:
                continue
            root = find(owner)
            if unchanged[index]:
                touches_unchanged_outside.add(root)
            else:
                touches_generated.add(root)

    components: dict[int, list[int]] = defaultdict(list)
    for index in candidates:
        components[find(index)].append(index)
    return dict(components), touches_unchanged_outside, touches_generated
