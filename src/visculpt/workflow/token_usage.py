"""Provider-normalized, persistent Token usage accounting."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Literal, cast
from urllib.parse import urlsplit
from uuid import uuid4

from visculpt.bridge import JsonValue

from .llm import LlmCallObservation

TOKEN_USAGE_SCHEMA_VERSION = "1.0"
TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
)
MetricSource = Literal["api", "official_derived", "unreported"]
UsageScope = Literal["workflow", "model_test"]


@dataclass(frozen=True, slots=True)
class TokenUsageContext:
    """Stable ownership metadata bound to one LLM adapter."""

    scope: UsageScope
    run_id: str | None = None
    thread_id: str | None = None
    workflow_title: str | None = None
    call_site: str | None = None


@dataclass(frozen=True, slots=True)
class NormalizedTokenUsage:
    """Five normalized metrics plus their provider evidence."""

    tokens: dict[str, int | None]
    sources: dict[str, MetricSource]
    actual_model: str
    request_id: str | None
    raw_usage: dict[str, JsonValue]


class TokenUsageStore:
    """Append-only SQLite ledger and deterministic aggregate queries."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self._schema_lock = RLock()
        self._schema_ready = False

    def record(
        self,
        observation: LlmCallObservation,
        context: TokenUsageContext,
    ) -> dict[str, JsonValue]:
        """Persist one logical model call and return a live update snapshot."""
        normalized = normalize_token_usage(observation)
        call_id = uuid4().hex
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO token_usage_calls (
                    call_id, scope, thread_id, run_id, workflow_title,
                    role, call_site, provider, base_url, endpoint_path,
                    requested_model, actual_model, request_id, outcome,
                    started_at, completed_at,
                    input_tokens, output_tokens, total_tokens,
                    cached_input_tokens, reasoning_tokens,
                    metric_sources, raw_usage
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    call_id,
                    context.scope,
                    context.thread_id,
                    context.run_id,
                    context.workflow_title,
                    observation.role,
                    context.call_site or observation.role,
                    observation.provider,
                    observation.base_url,
                    observation.endpoint_path,
                    observation.requested_model,
                    normalized.actual_model,
                    normalized.request_id,
                    observation.outcome,
                    observation.started_at.isoformat(),
                    observation.completed_at.isoformat(),
                    normalized.tokens["input_tokens"],
                    normalized.tokens["output_tokens"],
                    normalized.tokens["total_tokens"],
                    normalized.tokens["cached_input_tokens"],
                    normalized.tokens["reasoning_tokens"],
                    json.dumps(normalized.sources, sort_keys=True),
                    json.dumps(normalized.raw_usage, sort_keys=True),
                ),
            )

        workflow = (
            self.workflow_summary(
                context.run_id,
                thread_id=context.thread_id,
                title=context.workflow_title,
            )
            if context.scope == "workflow" and context.run_id is not None
            else None
        )
        model_key = _model_key(
            observation.provider,
            observation.base_url,
            normalized.actual_model,
        )
        return cast(
            dict[str, JsonValue],
            {
                "schema_version": TOKEN_USAGE_SCHEMA_VERSION,
                "call_id": call_id,
                "scope": context.scope,
                "outcome": observation.outcome,
                "role": observation.role,
                "call_site": context.call_site or observation.role,
                "model_key": model_key,
                "completed_at": observation.completed_at.isoformat(),
                "workflow_summary": workflow,
                "global_aggregate": self.aggregate(scope="workflow"),
                "global_model": self.model_summary(
                    model_key,
                    scope="workflow",
                ),
            },
        )

    def aggregate(
        self,
        *,
        scope: UsageScope = "workflow",
    ) -> dict[str, JsonValue]:
        """Return one aggregate for all retained calls in a scope."""
        rows = self._select_rows("scope = ?", (scope,))
        return _aggregate_rows(rows)

    def overview(self) -> dict[str, JsonValue]:
        """Return the complete Settings Usage snapshot."""
        workflow_rows = self._select_rows("scope = ?", ("workflow",))
        test_rows = self._select_rows("scope = ?", ("model_test",))
        workflows = []
        by_run = _group_rows(workflow_rows, lambda row: str(row["run_id"]))
        for run_id, rows in by_run.items():
            if not run_id or run_id == "None":
                continue
            workflows.append(self._workflow_summary_from_rows(run_id, rows))
        workflows.sort(
            key=lambda item: str(item.get("last_called_at") or ""),
            reverse=True,
        )
        return cast(
            dict[str, JsonValue],
            {
                "schema_version": TOKEN_USAGE_SCHEMA_VERSION,
                "aggregate": _aggregate_rows(workflow_rows),
                "by_model": _model_breakdown(workflow_rows),
                "workflows": workflows,
                "model_tests": {
                    "aggregate": _aggregate_rows(test_rows),
                    "by_model": _model_breakdown(test_rows),
                },
            },
        )

    def workflow_summary(
        self,
        run_id: str,
        *,
        thread_id: str | None = None,
        title: str | None = None,
    ) -> dict[str, JsonValue]:
        """Return totals and role/model breakdowns for one Workflow run."""
        rows = self._select_rows(
            "scope = ? AND run_id = ?",
            ("workflow", run_id),
        )
        if rows:
            return self._workflow_summary_from_rows(run_id, rows)
        return cast(
            dict[str, JsonValue],
            {
                "run_id": run_id,
                "thread_id": thread_id,
                "title": title,
                "aggregate": _aggregate_rows([]),
                "by_role": [],
                "by_model": [],
                "first_called_at": None,
                "last_called_at": None,
            },
        )

    def model_summary(
        self,
        model_key: str,
        *,
        scope: UsageScope,
    ) -> dict[str, JsonValue] | None:
        """Return the aggregate row for one provider/model key."""
        rows = self._select_rows("scope = ?", (scope,))
        return next(
            (
                item
                for item in _model_breakdown(rows)
                if item.get("key") == model_key
            ),
            None,
        )

    def delete_thread(self, thread_id: str) -> int:
        """Delete all locally retained usage owned by one LangGraph Thread."""
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM token_usage_calls WHERE thread_id = ?",
                (thread_id,),
            )
            return max(0, cursor.rowcount)

    def delete_runs(self, run_ids: Sequence[str]) -> int:
        """Delete usage by semantic Workflow run ID."""
        unique = tuple(dict.fromkeys(run_ids))
        if not unique:
            return 0
        placeholders = ",".join("?" for _ in unique)
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM token_usage_calls "
                f"WHERE run_id IN ({placeholders})",
                unique,
            )
            return max(0, cursor.rowcount)

    def _workflow_summary_from_rows(
        self,
        run_id: str,
        rows: list[sqlite3.Row],
    ) -> dict[str, JsonValue]:
        first = rows[0]
        timestamps = [str(row["completed_at"]) for row in rows]
        return cast(
            dict[str, JsonValue],
            {
                "run_id": run_id,
                "thread_id": first["thread_id"],
                "title": first["workflow_title"],
                "aggregate": _aggregate_rows(rows),
                "by_role": _role_breakdown(rows),
                "by_model": _model_breakdown(rows),
                "first_called_at": min(timestamps),
                "last_called_at": max(timestamps),
            },
        )

    def _select_rows(
        self,
        where: str,
        arguments: Sequence[object],
    ) -> list[sqlite3.Row]:
        with self._connection() as connection:
            cursor = connection.execute(
                f"SELECT * FROM token_usage_calls WHERE {where} "
                "ORDER BY id ASC",
                tuple(arguments),
            )
            return list(cursor.fetchall())

    def _connect(self) -> sqlite3.Connection:
        self._ensure_schema()
        connection = sqlite3.connect(
            self.database_path,
            timeout=10.0,
            isolation_level="DEFERRED",
        )
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Commit writes and deterministically close one short connection."""
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.database_path, timeout=10.0)
            try:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("PRAGMA foreign_keys = ON")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS token_usage_calls (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        call_id TEXT NOT NULL UNIQUE,
                        scope TEXT NOT NULL,
                        thread_id TEXT,
                        run_id TEXT,
                        workflow_title TEXT,
                        role TEXT NOT NULL,
                        call_site TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        base_url TEXT NOT NULL,
                        endpoint_path TEXT NOT NULL,
                        requested_model TEXT NOT NULL,
                        actual_model TEXT NOT NULL,
                        request_id TEXT,
                        outcome TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        completed_at TEXT NOT NULL,
                        input_tokens INTEGER,
                        output_tokens INTEGER,
                        total_tokens INTEGER,
                        cached_input_tokens INTEGER,
                        reasoning_tokens INTEGER,
                        metric_sources TEXT NOT NULL,
                        raw_usage TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_token_usage_scope
                        ON token_usage_calls(scope);
                    CREATE INDEX IF NOT EXISTS idx_token_usage_thread
                        ON token_usage_calls(thread_id);
                    CREATE INDEX IF NOT EXISTS idx_token_usage_run
                        ON token_usage_calls(run_id);
                    CREATE INDEX IF NOT EXISTS idx_token_usage_model
                        ON token_usage_calls(
                            provider, base_url, actual_model
                        );
                    """
                )
                connection.commit()
            finally:
                connection.close()
            self._schema_ready = True


class TokenUsageRecorder:
    """Translate LLM observations into durable records and live events."""

    def __init__(
        self,
        store: TokenUsageStore,
        context: TokenUsageContext,
        *,
        on_update: Callable[[dict[str, JsonValue]], None] | None = None,
    ) -> None:
        self._store = store
        self._context = context
        self._on_update = on_update

    def __call__(self, observation: LlmCallObservation) -> None:
        update = self._store.record(observation, self._context)
        if self._on_update is not None:
            self._on_update(update)


def default_token_usage_database_path(project_root: Path) -> Path:
    """Keep local accounting beside the existing Agent Server stores."""
    return project_root.resolve() / ".langgraph_api" / "token_usage.sqlite3"


def normalize_token_usage(
    observation: LlmCallObservation,
) -> NormalizedTokenUsage:
    """Normalize one latest provider response without tokenizer estimates."""
    response = observation.response or {}
    raw_usage = _mapping(response.get("usage"))
    if not raw_usage:
        raw_usage = _mapping(response.get("usageMetadata"))
    if not raw_usage:
        raw_usage = _mapping(response.get("stats"))

    if _mapping(response.get("usageMetadata")):
        tokens, sources = _normalize_gemini_usage(raw_usage)
    elif _mapping(response.get("stats")):
        tokens, sources = _normalize_lm_studio_usage(raw_usage)
    elif observation.provider == "anthropic":
        tokens, sources = _normalize_anthropic_usage(raw_usage)
    else:
        tokens, sources = _normalize_openai_usage(raw_usage)

    actual_model = _first_string(
        response.get("model"),
        response.get("modelVersion"),
        response.get("model_instance_id"),
    ) or observation.requested_model
    request_id = _first_string(
        response.get("id"),
        response.get("response_id"),
        response.get("request_id"),
    )
    return NormalizedTokenUsage(
        tokens=tokens,
        sources=sources,
        actual_model=actual_model,
        request_id=request_id,
        raw_usage=raw_usage,
    )


def _normalize_openai_usage(
    usage: Mapping[str, JsonValue],
) -> tuple[dict[str, int | None], dict[str, MetricSource]]:
    input_tokens = _integer(usage.get("prompt_tokens"))
    if input_tokens is None:
        input_tokens = _integer(usage.get("input_tokens"))
    output_tokens = _integer(usage.get("completion_tokens"))
    if output_tokens is None:
        output_tokens = _integer(usage.get("output_tokens"))
    total_tokens = _integer(usage.get("total_tokens"))
    prompt_details = _mapping(usage.get("prompt_tokens_details"))
    if not prompt_details:
        prompt_details = _mapping(usage.get("input_tokens_details"))
    completion_details = _mapping(usage.get("completion_tokens_details"))
    if not completion_details:
        completion_details = _mapping(usage.get("output_tokens_details"))
    cached = _integer(prompt_details.get("cached_tokens"))
    reasoning = _integer(completion_details.get("reasoning_tokens"))
    tokens, sources = _direct_metrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached,
        reasoning_tokens=reasoning,
    )
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        tokens["total_tokens"] = input_tokens + output_tokens
        sources["total_tokens"] = "official_derived"
    return tokens, sources


def _normalize_anthropic_usage(
    usage: Mapping[str, JsonValue],
) -> tuple[dict[str, int | None], dict[str, MetricSource]]:
    ordinary = _integer(usage.get("input_tokens"))
    created = _integer(usage.get("cache_creation_input_tokens"))
    cached = _integer(usage.get("cache_read_input_tokens"))
    has_cache_breakdown = (
        "cache_creation_input_tokens" in usage
        or "cache_read_input_tokens" in usage
    )
    input_tokens = ordinary
    input_source: MetricSource = "api" if ordinary is not None else "unreported"
    if has_cache_breakdown and ordinary is not None:
        input_tokens = ordinary + (created or 0) + (cached or 0)
        input_source = "official_derived"
    output_tokens = _integer(usage.get("output_tokens"))
    details = _mapping(usage.get("output_tokens_details"))
    reasoning = _integer(details.get("thinking_tokens"))
    if reasoning is None:
        reasoning = _integer(details.get("reasoning_tokens"))
    total_tokens = _integer(usage.get("total_tokens"))
    tokens, sources = _direct_metrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_input_tokens=cached,
        reasoning_tokens=reasoning,
    )
    sources["input_tokens"] = input_source
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        tokens["total_tokens"] = input_tokens + output_tokens
        sources["total_tokens"] = "official_derived"
    return tokens, sources


def _normalize_gemini_usage(
    usage: Mapping[str, JsonValue],
) -> tuple[dict[str, int | None], dict[str, MetricSource]]:
    input_tokens = _integer(usage.get("promptTokenCount"))
    candidates = _integer(usage.get("candidatesTokenCount"))
    reasoning = _integer(usage.get("thoughtsTokenCount"))
    output_tokens = candidates
    output_source: MetricSource = "api" if candidates is not None else "unreported"
    if candidates is not None and reasoning is not None:
        output_tokens = candidates + reasoning
        output_source = "official_derived"
    tokens, sources = _direct_metrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=_integer(usage.get("totalTokenCount")),
        cached_input_tokens=_integer(
            usage.get("cachedContentTokenCount")
        ),
        reasoning_tokens=reasoning,
    )
    sources["output_tokens"] = output_source
    if (
        tokens["total_tokens"] is None
        and input_tokens is not None
        and output_tokens is not None
    ):
        tokens["total_tokens"] = input_tokens + output_tokens
        sources["total_tokens"] = "official_derived"
    return tokens, sources


def _normalize_lm_studio_usage(
    usage: Mapping[str, JsonValue],
) -> tuple[dict[str, int | None], dict[str, MetricSource]]:
    input_tokens = _integer(usage.get("input_tokens"))
    output_tokens = _integer(usage.get("total_output_tokens"))
    tokens, sources = _direct_metrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=None,
        cached_input_tokens=None,
        reasoning_tokens=_integer(usage.get("reasoning_output_tokens")),
    )
    if input_tokens is not None and output_tokens is not None:
        tokens["total_tokens"] = input_tokens + output_tokens
        sources["total_tokens"] = "official_derived"
    return tokens, sources


def _direct_metrics(
    **values: int | None,
) -> tuple[dict[str, int | None], dict[str, MetricSource]]:
    tokens = {field: values.get(field) for field in TOKEN_FIELDS}
    sources: dict[str, MetricSource] = {
        field: "api" if tokens[field] is not None else "unreported"
        for field in TOKEN_FIELDS
    }
    return tokens, sources


def _aggregate_rows(rows: Sequence[sqlite3.Row]) -> dict[str, JsonValue]:
    call_count = len(rows)
    tokens: dict[str, JsonValue] = {}
    coverage: dict[str, JsonValue] = {}
    for field in TOKEN_FIELDS:
        reported = [int(row[field]) for row in rows if row[field] is not None]
        tokens[field] = sum(reported) if reported else None
        coverage[field] = {
            "reported_calls": len(reported),
            "total_calls": call_count,
            "partial": 0 < len(reported) < call_count,
        }
    return {
        "call_count": call_count,
        "tokens": tokens,
        "coverage": coverage,
    }


def _role_breakdown(rows: Sequence[sqlite3.Row]) -> list[dict[str, JsonValue]]:
    groups = _group_rows(rows, lambda row: str(row["role"]))
    return [
        {
            "key": role,
            "label": _role_label(role),
            "aggregate": _aggregate_rows(group),
        }
        for role, group in groups.items()
    ]


def _model_breakdown(rows: Sequence[sqlite3.Row]) -> list[dict[str, JsonValue]]:
    groups = _group_rows(
        rows,
        lambda row: _model_key(
            str(row["provider"]),
            str(row["base_url"]),
            str(row["actual_model"]),
        ),
    )
    result: list[dict[str, JsonValue]] = []
    for key, group in groups.items():
        first = group[0]
        result.append(
            {
                "key": key,
                "provider": str(first["provider"]),
                "provider_label": _provider_label(
                    str(first["provider"]),
                    str(first["base_url"]),
                ),
                "base_url": str(first["base_url"]),
                "model": str(first["actual_model"]),
                "aggregate": _aggregate_rows(group),
            }
        )
    result.sort(
        key=lambda item: int(
            cast(dict[str, JsonValue], item["aggregate"])["call_count"]
        ),
        reverse=True,
    )
    return result


def _group_rows(
    rows: Sequence[sqlite3.Row],
    key: Callable[[sqlite3.Row], str],
) -> dict[str, list[sqlite3.Row]]:
    groups: defaultdict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return dict(groups)


def _model_key(provider: str, base_url: str, model: str) -> str:
    return "|".join((provider, base_url.rstrip("/").casefold(), model))


def _provider_label(provider: str, base_url: str) -> str:
    host = (urlsplit(base_url).hostname or "").casefold()
    if host in {"127.0.0.1", "localhost"}:
        return "LM Studio"
    if "generativelanguage.googleapis.com" in host:
        return "Gemini"
    if "dashscope.aliyuncs.com" in host:
        return "Qwen"
    if host == "api.x.ai":
        return "xAI"
    return (
        "Anthropic compatible"
        if provider == "anthropic"
        else "OpenAI compatible"
    )


def _role_label(role: str) -> str:
    return {
        "decomposer": "Decomposer",
        "translator": "Translator",
        "view_selector": "View Selector",
        "quadloc": "QuadLoc",
        "svg_pattern_generator": "SVG Pattern Generator",
        "grader": "Grader",
        "retry_planner": "Retry Planner",
    }.get(role, role.replace("_", " ").title())


def _mapping(value: JsonValue | object) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): cast(JsonValue, item)
        for key, item in value.items()
    }


def _integer(value: JsonValue | object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    integer = int(value)
    return integer if integer >= 0 and integer == value else None


def _first_string(*values: object) -> str | None:
    return next(
        (
            value.strip()
            for value in values
            if isinstance(value, str) and value.strip()
        ),
        None,
    )
