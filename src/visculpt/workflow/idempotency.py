"""Best-effort durable ledger for external Blender stroke side effects."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast
from uuid import uuid4

from visculpt.bridge import JsonValue


class AppliedOperationLedger:
    """Record completed operation IDs outside rewindable LangGraph State."""

    def __init__(self, workflow_dir: Path) -> None:
        self.path = (
            workflow_dir / ".runtime" / "applied-operations.json"
        ).resolve()

    def contains(self, operation_id: str) -> bool:
        return operation_id in self._read()

    def operation_ids(self) -> list[str]:
        return sorted(self._read())

    def record(self, operation_id: str) -> None:
        values = self._read()
        values.add(operation_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f".{self.path.name}.{uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "operation_ids": sorted(values),
                    },
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            temporary.replace(self.path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _read(self) -> set[str]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return set()
        if not isinstance(payload, dict):
            return set()
        values = payload.get("operation_ids")
        if not isinstance(values, list):
            return set()
        return {
            cast(str, value)
            for value in values
            if isinstance(value, str)
        }


def sculpt_operation_id(
    *,
    run_id: str,
    subtask_index: int,
    attempt: int,
    call_index: int,
    tool_input: dict[str, JsonValue],
) -> str:
    """Derive a deterministic ID for one external operator invocation."""
    canonical = json.dumps(
        tool_input,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return (
        f"{run_id}:subtask-{subtask_index}:attempt-{attempt}:"
        f"stroke-{call_index}:{digest}"
    )
