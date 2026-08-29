"""Safe deletion of run-owned Workflow artifacts."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .state import RUN_ID_PATTERN


class WorkflowArtifactDeletionError(RuntimeError):
    """Raised when a run directory cannot be deleted safely."""


@dataclass(frozen=True, slots=True)
class WorkflowArtifactDeletionResult:
    """Summary of one idempotent artifact deletion request."""

    deleted_run_ids: tuple[str, ...]
    missing_run_ids: tuple[str, ...]
    deleted_file_count: int
    deleted_directory_count: int
    deleted_bytes: int

    def as_payload(self) -> dict[str, object]:
        """Return a JSON-safe response payload."""
        return {
            "deleted_run_ids": list(self.deleted_run_ids),
            "missing_run_ids": list(self.missing_run_ids),
            "deleted_file_count": self.deleted_file_count,
            "deleted_directory_count": self.deleted_directory_count,
            "deleted_bytes": self.deleted_bytes,
        }


def delete_workflow_run_artifacts(
    artifact_root: Path,
    run_ids: Iterable[str],
) -> WorkflowArtifactDeletionResult:
    """Delete only exact ``run-<id>`` directories below the artifact root."""
    root = artifact_root.resolve()
    unique_run_ids = tuple(dict.fromkeys(run_ids))
    invalid = [
        run_id
        for run_id in unique_run_ids
        if RUN_ID_PATTERN.fullmatch(run_id) is None
    ]
    if invalid:
        raise WorkflowArtifactDeletionError(
            f"Invalid workflow run IDs: {invalid}"
        )

    deleted: list[str] = []
    missing: list[str] = []
    file_count = 0
    directory_count = 0
    byte_count = 0

    for run_id in unique_run_ids:
        target = root / f"run-{run_id}"
        if target.is_symlink():
            raise WorkflowArtifactDeletionError(
                f"Refusing to delete symlinked workflow directory: {target}"
            )
        try:
            target.resolve().relative_to(root)
        except ValueError as error:
            raise WorkflowArtifactDeletionError(
                f"Workflow directory escapes artifact root: {target}"
            ) from error
        if not target.exists():
            missing.append(run_id)
            continue
        if not target.is_dir():
            raise WorkflowArtifactDeletionError(
                f"Workflow artifact path is not a directory: {target}"
            )

        files, directories, size = _directory_inventory(target)
        try:
            shutil.rmtree(target)
        except OSError as error:
            raise WorkflowArtifactDeletionError(
                f"Could not delete workflow artifacts for {run_id}: {error}"
            ) from error
        if target.exists() or target.is_symlink():
            raise WorkflowArtifactDeletionError(
                f"Workflow artifacts still exist after deletion: {target}"
            )
        deleted.append(run_id)
        file_count += files
        directory_count += directories
        byte_count += size

    return WorkflowArtifactDeletionResult(
        deleted_run_ids=tuple(deleted),
        missing_run_ids=tuple(missing),
        deleted_file_count=file_count,
        deleted_directory_count=directory_count,
        deleted_bytes=byte_count,
    )


def _directory_inventory(root: Path) -> tuple[int, int, int]:
    """Count every entry without following symlinks outside the run."""
    file_count = 0
    directory_count = 1
    byte_count = 0
    for current, directory_names, file_names in os.walk(
        root,
        followlinks=False,
    ):
        current_path = Path(current)
        real_directories: list[str] = []
        for name in directory_names:
            path = current_path / name
            if path.is_symlink():
                file_count += 1
                byte_count += path.lstat().st_size
            else:
                directory_count += 1
                real_directories.append(name)
        directory_names[:] = real_directories
        for name in file_names:
            path = current_path / name
            file_count += 1
            try:
                byte_count += path.lstat().st_size
            except FileNotFoundError:
                # An active process may have just removed the temporary file.
                continue
    return file_count, directory_count, byte_count
