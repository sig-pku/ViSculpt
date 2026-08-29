export type LlmProvider = "openai_compatible" | "anthropic";
export type LlmEffort =
  | "none"
  | "minimal"
  | "low"
  | "medium"
  | "high"
  | "xhigh"
  | "max";
export type LlmRole =
  | "decomposer"
  | "translator"
  | "view_selector"
  | "quadloc"
  | "svg_pattern_generator"
  | "grader"
  | "retry_planner";
export type LlmTestStatus = "pending" | "success" | "failure";

export type RuntimeLlmModels = Record<LlmRole, string>;

export interface RuntimeLlmSettings {
  base_url: string;
  provider: LlmProvider;
  endpoint_path: string;
  effort: LlmEffort | null;
  models: RuntimeLlmModels;
}

export interface RuntimeLlmPreset {
  id: string;
  label: string;
  base_url: string;
  openai_endpoint_path: string;
  anthropic_endpoint_path: string | null;
}

export interface LlmTestSnapshot {
  status: LlmTestStatus;
  fingerprint: string | null;
  tested_at: string | null;
  error: string | null;
  result: {
    tested_model_count?: number;
    results?: unknown[];
  } | null;
}

export interface LlmSettingsResponse {
  settings: RuntimeLlmSettings;
  presets?: RuntimeLlmPreset[];
  test: LlmTestSnapshot;
}

export const PENDING_LLM_TEST: LlmTestSnapshot = {
  status: "pending",
  fingerprint: null,
  tested_at: null,
  error: null,
  result: null,
};

export const LLM_ROLES: ReadonlyArray<{
  id: LlmRole;
  label: string;
}> = [
  { id: "decomposer", label: "Decomposer" },
  { id: "translator", label: "Translator" },
  { id: "view_selector", label: "View Selector" },
  { id: "quadloc", label: "QuadLoc" },
  { id: "svg_pattern_generator", label: "SVG Pattern Generator" },
  { id: "grader", label: "Grader" },
  { id: "retry_planner", label: "Retry Planner" },
];

export const OPENAI_COMPATIBLE_EFFORTS: readonly LlmEffort[] = [
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
];

export const ANTHROPIC_COMPATIBLE_EFFORTS: readonly LlmEffort[] = [
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
];

export function effortOptionsFor(provider: LlmProvider) {
  return provider === "anthropic"
    ? ANTHROPIC_COMPATIBLE_EFFORTS
    : OPENAI_COMPATIBLE_EFFORTS;
}

export function normalizeEffortForProvider(
  provider: LlmProvider,
  effort: LlmEffort | null,
): LlmEffort | null {
  if (effort === null) return null;
  return effortOptionsFor(provider).includes(effort) ? effort : null;
}

function normalizedBaseUrl(value: string) {
  return value.trim().replace(/\/+$/, "").toLowerCase();
}

export function matchingLlmPreset(
  baseUrl: string,
  presets: RuntimeLlmPreset[],
) {
  const normalized = normalizedBaseUrl(baseUrl);
  return presets.find(
    (preset) => normalizedBaseUrl(preset.base_url) === normalized,
  );
}

export function endpointPathFor(
  baseUrl: string,
  provider: LlmProvider,
  presets: RuntimeLlmPreset[],
) {
  const preset = matchingLlmPreset(baseUrl, presets);
  if (provider === "anthropic") {
    return preset?.anthropic_endpoint_path ?? "/v1/messages";
  }
  return preset?.openai_endpoint_path ?? "/v1/chat/completions";
}

export function supportsAnthropic(
  baseUrl: string,
  presets: RuntimeLlmPreset[],
) {
  const preset = matchingLlmPreset(baseUrl, presets);
  return preset ? preset.anthropic_endpoint_path !== null : true;
}

export function llmSettingsFingerprint(settings: RuntimeLlmSettings) {
  return JSON.stringify({
    base_url: settings.base_url,
    provider: settings.provider,
    endpoint_path: settings.endpoint_path,
    effort: settings.effort,
    models: LLM_ROLES.map(({ id }) => [id, settings.models[id]]),
  });
}
