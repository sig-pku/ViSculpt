"""Safe browser-facing references for files produced by one workflow run."""

from __future__ import annotations

import hashlib
import mimetypes
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field

from visculpt.bridge import JsonValue

WORKFLOW_ARTIFACT_SCHEMA_VERSION = "1.0"


class WorkflowArtifactKind(StrEnum):
    """Small, stable artifact vocabulary used by the Web client."""

    SCREENSHOT = "screenshot"
    MASK = "mask"
    OVERLAY = "overlay"
    TRAJECTORY = "trajectory"
    BLENDER_STATE = "blender_state"
    WORKFLOW_STATE = "workflow_state"
    METADATA = "metadata"
    OTHER = "other"


class WorkflowArtifact(BaseModel):
    """A path-independent artifact descriptor safe to send to browsers."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = WORKFLOW_ARTIFACT_SCHEMA_VERSION
    artifact_id: str = Field(min_length=16, max_length=64)
    run_id: str = Field(min_length=1, max_length=128)
    kind: WorkflowArtifactKind
    label: str = Field(min_length=1, max_length=256)
    relative_path: str = Field(min_length=1, max_length=2048)
    uri: str = Field(min_length=1, max_length=4096)
    media_type: str = Field(min_length=1, max_length=256)
    size_bytes: int = Field(ge=0)
    created_at: datetime
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


def create_artifact(
    *,
    run_id: str,
    workflow_dir: Path,
    path: Path,
    kind: WorkflowArtifactKind,
    label: str,
    metadata: dict[str, JsonValue] | None = None,
) -> WorkflowArtifact:
    """Create a safe reference after proving the file belongs to the run."""
    root = workflow_dir.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("artifact path must be inside workflow_dir") from error
    if not resolved.is_file():
        raise ValueError(f"artifact file does not exist: {resolved}")
    normalized = relative.as_posix()
    digest = hashlib.sha256(
        f"{run_id}\0{normalized}".encode("utf-8")
    ).hexdigest()[:24]
    media_type = mimetypes.guess_type(resolved.name)[0]
    if media_type is None:
        media_type = "application/octet-stream"
    return WorkflowArtifact(
        artifact_id=f"artifact-{digest}",
        run_id=run_id,
        kind=kind,
        label=label,
        relative_path=normalized,
        uri=(
            f"/workflow/artifacts/{quote(run_id, safe='')}/"
            f"{quote(normalized, safe='/')}"
        ),
        media_type=media_type,
        size_bytes=resolved.stat().st_size,
        created_at=datetime.now(timezone.utc),
        metadata={} if metadata is None else metadata,
    )


def merge_artifacts(
    existing: list[dict[str, JsonValue]],
    additions: list[WorkflowArtifact],
) -> list[dict[str, JsonValue]]:
    """Merge artifact descriptors by stable ID while preserving order."""
    merged = [dict(item) for item in existing]
    positions = {
        str(item.get("artifact_id")): index
        for index, item in enumerate(merged)
        if isinstance(item.get("artifact_id"), str)
    }
    for artifact in additions:
        payload = artifact.model_dump(mode="json")
        position = positions.get(artifact.artifact_id)
        if position is None:
            positions[artifact.artifact_id] = len(merged)
            merged.append(payload)
        else:
            merged[position] = payload
    return merged
