"""Local LangGraph development-checkpointer cleanup helpers."""

from __future__ import annotations

import pickle
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID


class ThreadPersistenceDeletionError(RuntimeError):
    """Raised when local checkpoint data cannot be fully purged."""


@dataclass(frozen=True, slots=True)
class ThreadPersistenceDeletionResult:
    """Result of one backend-aware checkpoint purge."""

    backend: str
    purged: bool
    deleted_storage_entries: int
    deleted_write_entries: int
    deleted_blob_entries: int
    persistence_files_checked: int

    def as_payload(self, thread_id: str) -> dict[str, object]:
        """Return a JSON-safe response payload."""
        return {
            "thread_id": thread_id,
            "backend": self.backend,
            "purged": self.purged,
            "deleted_storage_entries": self.deleted_storage_entries,
            "deleted_write_entries": self.deleted_write_entries,
            "deleted_blob_entries": self.deleted_blob_entries,
            "persistence_files_checked": self.persistence_files_checked,
        }


@dataclass(frozen=True, slots=True)
class ThreadDeletionConfirmationResult:
    """Durable confirmation after the Agent Server deleted one Thread."""

    backend: str
    confirmed: bool
    deleted_retry_counter_entries: int
    operations_file_checked: bool
    retry_counter_file_checked: bool
    checkpoints: ThreadPersistenceDeletionResult

    def as_payload(self, thread_id: str) -> dict[str, object]:
        """Return a JSON-safe response payload."""
        return {
            "thread_id": thread_id,
            "backend": self.backend,
            "confirmed": self.confirmed,
            "deleted_retry_counter_entries": (
                self.deleted_retry_counter_entries
            ),
            "operations_file_checked": self.operations_file_checked,
            "retry_counter_file_checked": self.retry_counter_file_checked,
            "checkpoints": self.checkpoints.as_payload(thread_id),
        }


def purge_local_thread_checkpoints(
    thread_id: str,
    *,
    project_root: Path,
) -> ThreadPersistenceDeletionResult:
    """Purge and immediately compact a local in-memory checkpointer Thread."""
    try:
        UUID(thread_id)
    except ValueError as error:
        raise ThreadPersistenceDeletionError(
            f"Invalid LangGraph thread ID: {thread_id}"
        ) from error

    checkpointer = _local_in_memory_checkpointer()
    if checkpointer is None:
        return ThreadPersistenceDeletionResult(
            backend="managed",
            purged=False,
            deleted_storage_entries=0,
            deleted_write_entries=0,
            deleted_blob_entries=0,
            persistence_files_checked=0,
        )

    storage = checkpointer.storage
    writes = checkpointer.writes
    blobs = checkpointer.blobs
    storage_count = int(thread_id in storage)
    write_count = _thread_key_count(writes, thread_id)
    blob_count = _thread_key_count(blobs, thread_id)

    checkpointer.delete_thread(thread_id)
    for mapping in (storage, writes, blobs):
        sync = getattr(mapping, "sync", None)
        if callable(sync):
            sync()

    residual_count = (
        int(thread_id in storage)
        + _thread_key_count(writes, thread_id)
        + _thread_key_count(blobs, thread_id)
    )
    if residual_count:
        raise ThreadPersistenceDeletionError(
            "LangGraph checkpointer still contains the deleted Thread"
        )

    persistence_files = tuple(
        (project_root / ".langgraph_api").glob(
            ".langgraph_checkpoint.*.pckl"
        )
    )
    encoded_thread_id = thread_id.encode("ascii")
    residual_files = [
        path.name
        for path in persistence_files
        if encoded_thread_id in path.read_bytes()
    ]
    if residual_files:
        raise ThreadPersistenceDeletionError(
            "Deleted Thread remains in local checkpoint files: "
            f"{residual_files}"
        )

    return ThreadPersistenceDeletionResult(
        backend="in_memory",
        purged=True,
        deleted_storage_entries=storage_count,
        deleted_write_entries=write_count,
        deleted_blob_entries=blob_count,
        persistence_files_checked=len(persistence_files),
    )


def confirm_local_thread_deletion(
    thread_id: str,
    *,
    agent_server_run_ids: list[str],
    project_root: Path,
) -> ThreadDeletionConfirmationResult:
    """Force local Agent Server stores to disk and verify full deletion."""
    _validated_uuid(thread_id, label="LangGraph thread ID")
    for run_id in agent_server_run_ids:
        _validated_uuid(run_id, label="Agent Server run ID")

    checkpoint_result = purge_local_thread_checkpoints(
        thread_id,
        project_root=project_root,
    )
    database = _local_in_memory_database()
    if database is None:
        return ThreadDeletionConfirmationResult(
            backend="managed",
            confirmed=True,
            deleted_retry_counter_entries=0,
            operations_file_checked=False,
            retry_counter_file_checked=False,
            checkpoints=checkpoint_result,
        )
    store, retry_counter = database

    residual_rows = _thread_database_rows(store, thread_id)
    if any(residual_rows.values()):
        raise ThreadPersistenceDeletionError(
            "Agent Server still contains the deleted Thread: "
            f"{residual_rows}"
        )

    run_ids = set(agent_server_run_ids)
    counters = retry_counter._counters
    locks = retry_counter._locks
    deleted_counters = 0
    for key in list(counters):
        if str(key) in run_ids:
            del counters[key]
            deleted_counters += 1
    for key in list(locks):
        if str(key) in run_ids:
            del locks[key]

    store.sync()
    counter_sync = getattr(counters, "sync", None)
    if callable(counter_sync):
        counter_sync()

    operations_path = project_root / ".langgraph_api" / ".langgraph_ops.pckl"
    if operations_path.is_file():
        persisted = pickle.loads(operations_path.read_bytes())
        if not isinstance(persisted, dict):
            raise ThreadPersistenceDeletionError(
                "LangGraph operations persistence has an invalid format"
            )
        persisted_rows = _thread_database_rows(persisted, thread_id)
        if any(persisted_rows.values()):
            raise ThreadPersistenceDeletionError(
                "Deleted Thread remains in Agent Server persistence: "
                f"{persisted_rows}"
            )
        operations_checked = True
    else:
        operations_checked = False

    retry_path = (
        project_root
        / ".langgraph_api"
        / ".langgraph_retry_counter.pckl"
    )
    if retry_path.is_file():
        persisted_counters = pickle.loads(retry_path.read_bytes())
        if not isinstance(persisted_counters, dict):
            raise ThreadPersistenceDeletionError(
                "LangGraph retry persistence has an invalid format"
            )
        residual_run_ids = [
            str(key)
            for key in persisted_counters
            if str(key) in run_ids
        ]
        if residual_run_ids:
            raise ThreadPersistenceDeletionError(
                "Deleted runs remain in retry persistence: "
                f"{residual_run_ids}"
            )
        retry_checked = True
    else:
        retry_checked = False

    return ThreadDeletionConfirmationResult(
        backend="in_memory",
        confirmed=True,
        deleted_retry_counter_entries=deleted_counters,
        operations_file_checked=operations_checked,
        retry_counter_file_checked=retry_checked,
        checkpoints=checkpoint_result,
    )


def _thread_key_count(
    mapping: MutableMapping[Any, Any],
    thread_id: str,
) -> int:
    """Count tuple keys owned by one Thread without inspecting values."""
    return sum(
        1
        for key in mapping
        if isinstance(key, tuple) and key and key[0] == thread_id
    )


def _thread_database_rows(
    store: MutableMapping[Any, Any],
    thread_id: str,
) -> dict[str, int]:
    """Count persisted rows that still belong to one Thread."""
    return {
        "threads": sum(
            1
            for row in store.get("threads", [])
            if str(row.get("thread_id")) == thread_id
        ),
        "runs": sum(
            1
            for row in store.get("runs", [])
            if str(row.get("thread_id")) == thread_id
        ),
        "crons": sum(
            1
            for row in store.get("crons", [])
            if str(row.get("thread_id")) == thread_id
        ),
    }


def _validated_uuid(value: str, *, label: str) -> None:
    """Validate one persistent Agent Server identifier."""
    try:
        UUID(value)
    except ValueError as error:
        raise ThreadPersistenceDeletionError(
            f"Invalid {label}: {value}"
        ) from error


def _local_in_memory_checkpointer() -> Any | None:
    """Return the local runtime checkpointer only for file-backed dev mode."""
    try:
        from langgraph_api import config as api_config
        from langgraph_api.feature_flags import IS_POSTGRES_BACKEND
        from langgraph_runtime_inmem.checkpoint import Checkpointer
    except ImportError:
        return None
    if IS_POSTGRES_BACKEND or api_config.USE_CUSTOM_CHECKPOINTER:
        return None
    return Checkpointer()


def _local_in_memory_database() -> tuple[Any, Any] | None:
    """Return local operation and retry stores only in dev in-memory mode."""
    try:
        from langgraph_api import config as api_config
        from langgraph_api.feature_flags import IS_POSTGRES_BACKEND
        from langgraph_runtime_inmem.database import (
            GLOBAL_RETRY_COUNTER,
            GLOBAL_STORE,
        )
    except ImportError:
        return None
    if IS_POSTGRES_BACKEND or api_config.USE_CUSTOM_CHECKPOINTER:
        return None
    return GLOBAL_STORE, GLOBAL_RETRY_COUNTER
