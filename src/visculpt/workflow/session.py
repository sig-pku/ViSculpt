"""Cross-platform file lease for the single mutable Blender session."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from uuid import uuid4

from visculpt.bridge import JsonValue

from .errors import WorkflowExecutionError

_PROCESS_LOCK = threading.Lock()
_FILESYSTEM_GUARD_TIMEOUT_SECONDS = 5.0
_FILESYSTEM_GUARD_STALE_SECONDS = 30.0


class BlenderSessionLease:
    """Prevent independent Agent Server threads from interleaving RPC work."""

    def __init__(self, path: Path, *, lease_seconds: float) -> None:
        self.path = path.resolve()
        self.lease_seconds = lease_seconds

    def acquire(self, run_id: str) -> dict[str, JsonValue]:
        """Atomically acquire or refresh the lease for one workflow run."""
        with self._locked():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            existing = self._read()
            if existing is None and self.path.exists():
                # Writes use atomic replacement. An unparseable leftover cannot
                # be a renewable valid lease, so remove it before exclusive create.
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
            if existing is not None:
                owner = existing.get("run_id")
                if owner == run_id:
                    return self._write(run_id, existing=existing)
                if not self._is_stale():
                    raise WorkflowExecutionError(
                        "Blender session is busy with workflow run "
                        f"{owner!r}; wait for it to finish or release the "
                        "stale lease from the workflow console"
                    )
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
            return self._create_exclusive(run_id)

    def heartbeat(self, run_id: str) -> dict[str, JsonValue]:
        """Refresh the lease and reject a run that no longer owns it."""
        with self._locked():
            existing = self._read()
            if existing is None or existing.get("run_id") != run_id:
                raise WorkflowExecutionError(
                    "The workflow no longer owns the Blender session lease"
                )
            return self._write(run_id, existing=existing)

    def remember_viewport_ui_snapshot(
        self,
        run_id: str,
        snapshot: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        """Persist the recovery snapshot beside the owning lease."""
        with self._locked():
            existing = self._read()
            if existing is None or existing.get("run_id") != run_id:
                raise WorkflowExecutionError(
                    "The workflow no longer owns the Blender session lease"
                )
            payload = dict(existing)
            payload["sculpt_viewport_ui_snapshot"] = snapshot
            return self._write(run_id, existing=payload)

    def remember_checkpoint(
        self,
        run_id: str,
        checkpoint_path: str,
    ) -> dict[str, JsonValue]:
        """Persist the active subtask rollback checkpoint for cancellation."""
        with self._locked():
            existing = self._read()
            if existing is None or existing.get("run_id") != run_id:
                raise WorkflowExecutionError(
                    "The workflow no longer owns the Blender session lease"
                )
            payload = dict(existing)
            payload["checkpoint_path"] = checkpoint_path
            return self._write(run_id, existing=payload)

    def clear_checkpoint(self, run_id: str) -> dict[str, JsonValue]:
        """Clear the committed subtask checkpoint from the active lease."""
        with self._locked():
            existing = self._read()
            if existing is None or existing.get("run_id") != run_id:
                raise WorkflowExecutionError(
                    "The workflow no longer owns the Blender session lease"
                )
            payload = dict(existing)
            payload.pop("checkpoint_path", None)
            return self._write(run_id, existing=payload)

    def release(self, run_id: str, *, force: bool = False) -> bool:
        """Release only the caller's lease unless an operator forces it."""
        with self._locked():
            existing = self._read()
            if existing is None:
                return False
            if not force and existing.get("run_id") != run_id:
                return False
            try:
                self.path.unlink()
            except FileNotFoundError:
                return False
            return True

    def status(self) -> dict[str, JsonValue]:
        """Return a public, secret-free lease status payload."""
        with self._locked():
            existing = self._read()
            return {
                "locked": existing is not None and not self._is_stale(),
                "stale": existing is not None and self._is_stale(),
                "lease": existing,
                "path": str(self.path),
                "lease_seconds": self.lease_seconds,
            }

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Serialize lease replacement across threads and server processes."""
        with _PROCESS_LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            guard = self.path.with_name(f".{self.path.name}.guard")
            deadline = time.monotonic() + _FILESYSTEM_GUARD_TIMEOUT_SECONDS
            while True:
                try:
                    guard.mkdir()
                    break
                except FileExistsError:
                    try:
                        age = time.time() - guard.stat().st_mtime
                    except OSError:
                        age = 0.0
                    if age > _FILESYSTEM_GUARD_STALE_SECONDS:
                        try:
                            guard.rmdir()
                        except OSError:
                            pass
                        continue
                    if time.monotonic() >= deadline:
                        raise WorkflowExecutionError(
                            "Timed out while locking the Blender session lease"
                        )
                    time.sleep(0.01)
            try:
                yield
            finally:
                try:
                    guard.rmdir()
                except OSError:
                    pass

    def _create_exclusive(self, run_id: str) -> dict[str, JsonValue]:
        now = datetime.now(timezone.utc).isoformat()
        payload: dict[str, JsonValue] = {
            "run_id": run_id,
            "acquired_at": now,
            "heartbeat_at": now,
            "pid": os.getpid(),
        }
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except FileExistsError as error:
            raise WorkflowExecutionError(
                "Another workflow acquired the Blender session concurrently"
            ) from error
        try:
            os.write(
                descriptor,
                json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            )
        finally:
            os.close(descriptor)
        return payload

    def _write(
        self,
        run_id: str,
        *,
        existing: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        payload = dict(existing)
        payload["run_id"] = run_id
        payload["heartbeat_at"] = datetime.now(timezone.utc).isoformat()
        payload["pid"] = os.getpid()
        temporary = self.path.with_name(
            f".{self.path.name}.{uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(payload, separators=(",", ":")),
                encoding="utf-8",
            )
            temporary.replace(self.path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return payload

    def _read(self) -> dict[str, JsonValue] | None:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        return cast(dict[str, JsonValue], value) if isinstance(value, dict) else None

    def _is_stale(self) -> bool:
        try:
            age = datetime.now(timezone.utc).timestamp() - self.path.stat().st_mtime
        except OSError:
            return False
        return age > self.lease_seconds


def session_lease_for_artifact_root(
    artifact_root: Path,
    *,
    lease_seconds: float,
) -> BlenderSessionLease:
    """Return the canonical lease used by both graph and custom HTTP routes."""
    return BlenderSessionLease(
        artifact_root / ".runtime" / "blender-session.json",
        lease_seconds=lease_seconds,
    )
