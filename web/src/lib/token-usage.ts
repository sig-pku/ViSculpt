import type { JsonValue, WorkflowEvent } from "@/lib/workflow-types";

export const TOKEN_FIELDS = [
  "input_tokens",
  "output_tokens",
  "total_tokens",
  "cached_input_tokens",
  "reasoning_tokens",
] as const;

export type TokenField = (typeof TOKEN_FIELDS)[number];

export interface TokenMetrics {
  input_tokens: number | null;
  output_tokens: number | null;
  total_tokens: number | null;
  cached_input_tokens: number | null;
  reasoning_tokens: number | null;
}

export interface TokenMetricCoverage {
  reported_calls: number;
  total_calls: number;
  partial: boolean;
}

export interface TokenUsageAggregate {
  call_count: number;
  tokens: TokenMetrics;
  coverage: Record<TokenField, TokenMetricCoverage>;
}

export interface RoleTokenUsage {
  key: string;
  label: string;
  aggregate: TokenUsageAggregate;
}

export interface ModelTokenUsage {
  key: string;
  provider: string;
  provider_label: string;
  base_url: string;
  model: string;
  aggregate: TokenUsageAggregate;
}

export interface WorkflowTokenUsageSummary {
  run_id: string;
  thread_id: string | null;
  title: string | null;
  aggregate: TokenUsageAggregate;
  by_role: RoleTokenUsage[];
  by_model: ModelTokenUsage[];
  first_called_at: string | null;
  last_called_at: string | null;
}

export interface TokenUsageOverview {
  schema_version: "1.0";
  aggregate: TokenUsageAggregate;
  by_model: ModelTokenUsage[];
  workflows: WorkflowTokenUsageSummary[];
  model_tests: {
    aggregate: TokenUsageAggregate;
    by_model: ModelTokenUsage[];
  };
}

export interface UsageUpdatedPayload {
  kind: "usage.updated";
  call_id: string;
  outcome: "success" | "invalid_response" | "transport_error";
  role: string;
  call_site: string;
  model_key: string;
  workflow_summary: WorkflowTokenUsageSummary;
  global_aggregate: TokenUsageAggregate;
  global_model: ModelTokenUsage | null;
}

export function formatTokenCount(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  const absolute = Math.abs(value);
  if (absolute < 1_000) return value.toLocaleString("en-US");
  const divisor = absolute >= 1_000_000 ? 1_000_000 : 1_000;
  const suffix = divisor === 1_000_000 ? "M" : "K";
  const scaled = value / divisor;
  const decimals = Math.abs(scaled) >= 100 ? 0 : Math.abs(scaled) >= 10 ? 1 : 2;
  return `${scaled.toFixed(decimals).replace(/\.0+$|(?<=\.[0-9])0+$/, "")}${suffix}`;
}

export function exactTokenCount(value: number | null | undefined): string {
  return value === null || value === undefined
    ? "Not reported by the provider"
    : value.toLocaleString("en-US");
}

export function usagePayload(
  event: WorkflowEvent,
): UsageUpdatedPayload | null {
  return event.payload.kind === "usage.updated"
    ? (event.payload as UsageUpdatedPayload)
    : null;
}

export function mergeUsageUpdate(
  current: TokenUsageOverview | null,
  update: UsageUpdatedPayload,
): TokenUsageOverview {
  if (!current) {
    return {
      schema_version: "1.0",
      aggregate: update.global_aggregate,
      by_model: update.global_model ? [update.global_model] : [],
      workflows: [update.workflow_summary],
      model_tests: {
        aggregate: emptyTokenUsageAggregate(),
        by_model: [],
      },
    };
  }
  const workflows = [
    update.workflow_summary,
    ...current.workflows.filter(
      (item) => item.run_id !== update.workflow_summary.run_id,
    ),
  ];
  const byModel = update.global_model
    ? [
        update.global_model,
        ...current.by_model.filter(
          (item) => item.key !== update.global_model?.key,
        ),
      ]
    : current.by_model;
  return {
    ...current,
    aggregate: update.global_aggregate,
    by_model: byModel,
    workflows,
  };
}

function emptyTokenUsageAggregate(): TokenUsageAggregate {
  const coverage = Object.fromEntries(
    TOKEN_FIELDS.map((field) => [
      field,
      { reported_calls: 0, total_calls: 0, partial: false },
    ]),
  ) as Record<TokenField, TokenMetricCoverage>;
  return {
    call_count: 0,
    tokens: {
      input_tokens: null,
      output_tokens: null,
      total_tokens: null,
      cached_input_tokens: null,
      reasoning_tokens: null,
    },
    coverage,
  };
}

export function isWorkflowTokenUsageSummary(
  value: JsonValue | unknown,
): value is WorkflowTokenUsageSummary {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const candidate = value as Partial<WorkflowTokenUsageSummary>;
  return (
    typeof candidate.run_id === "string" &&
    Boolean(candidate.aggregate) &&
    Array.isArray(candidate.by_role) &&
    Array.isArray(candidate.by_model)
  );
}

export function isTokenUsageOverview(
  value: unknown,
): value is TokenUsageOverview {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const candidate = value as Partial<TokenUsageOverview>;
  return (
    candidate.schema_version === "1.0" &&
    Boolean(candidate.aggregate) &&
    Array.isArray(candidate.by_model) &&
    Array.isArray(candidate.workflows) &&
    Boolean(candidate.model_tests)
  );
}
