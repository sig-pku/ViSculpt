"use client";

import {
  BarChart3,
  Box,
  Check,
  CheckCircle2,
  Clock3,
  Cpu,
  Gauge,
  Link2,
  LoaderCircle,
  Monitor,
  Moon,
  RefreshCw,
  Route,
  Settings2,
  Sparkles,
  Sun,
  Type,
  X,
  XCircle,
} from "lucide-react";
import { useEffect, useLayoutEffect, useState } from "react";

import {
  readBrowserSetting,
  writeBrowserSetting,
} from "@/lib/browser-storage";
import {
  effortOptionsFor,
  endpointPathFor,
  LLM_ROLES,
  matchingLlmPreset,
  normalizeEffortForProvider,
  supportsAnthropic,
  type LlmEffort,
  type LlmProvider,
  type LlmRole,
  type LlmTestSnapshot,
  type RuntimeLlmPreset,
  type RuntimeLlmSettings,
} from "@/lib/llm-settings";
import type { RuntimeServiceSettings } from "@/lib/service-settings";
import {
  exactTokenCount,
  formatTokenCount,
  type TokenField,
  type TokenUsageAggregate,
  type TokenUsageOverview,
} from "@/lib/token-usage";
import type { ServiceHealth } from "@/lib/workflow-types";

type ThemePreference = "system" | "light" | "dark";
type SizePreference = "small" | "medium" | "large";
type PreferenceSection = "general" | "models" | "usage" | "blender" | "sam3";

type Props = {
  compact?: boolean;
  llmSettings: RuntimeLlmSettings | null;
  llmPresets: RuntimeLlmPreset[];
  llmTest: LlmTestSnapshot;
  llmSettingsLoading: boolean;
  llmTesting: boolean;
  serviceSettings: RuntimeServiceSettings | null;
  serviceHealth: ServiceHealth | null;
  serviceHealthError: string | null;
  serviceSettingsLoading: boolean;
  serviceSettingsSaving: boolean;
  serviceSettingsDisabled: boolean;
  usage?: TokenUsageOverview | null;
  usageLoading?: boolean;
  usageError?: string | null;
  onLlmSettingsChange: (settings: RuntimeLlmSettings) => void;
  onTestLlm: () => void;
  onServiceSettingsChange: (settings: RuntimeServiceSettings) => void;
  onRefreshUsage?: () => void;
};

const THEME_KEY = "geometry-agent.theme";
const SIZE_KEY = "geometry-agent.size";
const USAGE_METRICS: Array<{
  field: TokenField;
  label: string;
}> = [
  { field: "input_tokens", label: "Input" },
  { field: "output_tokens", label: "Output" },
  { field: "total_tokens", label: "Total" },
  { field: "cached_input_tokens", label: "Cached input" },
  { field: "reasoning_tokens", label: "Reasoning" },
];

function TokenValue({
  aggregate,
  field,
}: {
  aggregate: TokenUsageAggregate;
  field: TokenField;
}) {
  const value = aggregate.tokens[field];
  const coverage = aggregate.coverage[field];
  const partial = coverage?.partial === true;
  const title = `${exactTokenCount(value)}${partial ? ` · ${coverage.reported_calls}/${coverage.total_calls} calls reported` : ""}`;
  return (
    <span className="usage-token-value" title={title}>
      {formatTokenCount(value)}{partial ? <sup>*</sup> : null}
    </span>
  );
}

function isTheme(value: string | null): value is ThemePreference {
  return value === "system" || value === "light" || value === "dark";
}

function isSize(value: string | null): value is SizePreference {
  return value === "small" || value === "medium" || value === "large";
}

function applyPreferences(theme: ThemePreference, size: SizePreference) {
  const systemTheme =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-color-scheme: light)").matches
      ? "light"
      : "dark";
  document.documentElement.dataset.theme =
    theme === "system" ? systemTheme : theme;
  document.documentElement.dataset.themeMode = theme;
  document.documentElement.dataset.uiSize = size;
}

export function PreferencesMenu({
  compact = false,
  llmSettings,
  llmPresets,
  llmTest,
  llmSettingsLoading,
  llmTesting,
  serviceSettings,
  serviceHealth,
  serviceHealthError,
  serviceSettingsLoading,
  serviceSettingsSaving,
  serviceSettingsDisabled,
  usage = null,
  usageLoading = false,
  usageError = null,
  onLlmSettingsChange,
  onTestLlm,
  onServiceSettingsChange,
  onRefreshUsage = () => undefined,
}: Props) {
  const [open, setOpen] = useState(false);
  const [section, setSection] = useState<PreferenceSection>("general");
  const [theme, setTheme] = useState<ThemePreference>("system");
  const [size, setSize] = useState<SizePreference>("medium");

  useLayoutEffect(() => {
    const storedTheme = readBrowserSetting(THEME_KEY);
    const storedSize = readBrowserSetting(SIZE_KEY);
    const nextTheme = isTheme(storedTheme) ? storedTheme : "system";
    const nextSize = isSize(storedSize) ? storedSize : "medium";
    applyPreferences(nextTheme, nextSize);
    const timer = window.setTimeout(() => {
      setTheme(nextTheme);
      setSize(nextSize);
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (theme !== "system" || typeof window.matchMedia !== "function") return;
    const media = window.matchMedia("(prefers-color-scheme: light)");
    const applySystemTheme = () => applyPreferences("system", size);
    media.addEventListener("change", applySystemTheme);
    return () => media.removeEventListener("change", applySystemTheme);
  }, [size, theme]);

  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const closeFromKeyboard = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", closeFromKeyboard);
    return () => {
      document.removeEventListener("keydown", closeFromKeyboard);
      document.body.style.overflow = previousOverflow;
    };
  }, [open]);

  function updateTheme(nextTheme: ThemePreference) {
    setTheme(nextTheme);
    writeBrowserSetting(THEME_KEY, nextTheme);
    applyPreferences(nextTheme, size);
  }

  function updateSize(nextSize: SizePreference) {
    setSize(nextSize);
    writeBrowserSetting(SIZE_KEY, nextSize);
    applyPreferences(theme, nextSize);
  }

  function updateBaseUrl(baseUrl: string) {
    if (!llmSettings) return;
    const preset = matchingLlmPreset(baseUrl, llmPresets);
    const provider =
      llmSettings.provider === "anthropic" &&
      preset?.anthropic_endpoint_path === null
        ? "openai_compatible"
        : llmSettings.provider;
    onLlmSettingsChange({
      ...llmSettings,
      base_url: baseUrl,
      provider,
      endpoint_path: endpointPathFor(baseUrl, provider, llmPresets),
      effort: normalizeEffortForProvider(provider, llmSettings.effort),
    });
  }

  function updateProvider(provider: LlmProvider) {
    if (!llmSettings) return;
    onLlmSettingsChange({
      ...llmSettings,
      provider,
      endpoint_path: endpointPathFor(
        llmSettings.base_url,
        provider,
        llmPresets,
      ),
      effort: normalizeEffortForProvider(provider, llmSettings.effort),
    });
  }

  function updateEffort(effort: string) {
    if (!llmSettings) return;
    onLlmSettingsChange({
      ...llmSettings,
      effort: effort === "" ? null : (effort as LlmEffort),
    });
  }

  function updateModel(role: LlmRole, modelId: string) {
    if (!llmSettings) return;
    onLlmSettingsChange({
      ...llmSettings,
      models: { ...llmSettings.models, [role]: modelId },
    });
  }

  function synchronizeModel(role: LlmRole) {
    if (!llmSettings) return;
    const modelId = llmSettings.models[role];
    onLlmSettingsChange({
      ...llmSettings,
      models: Object.fromEntries(
        LLM_ROLES.map(({ id }) => [id, modelId]),
      ) as RuntimeLlmSettings["models"],
    });
  }

  const anthropicAllowed = llmSettings
    ? supportsAnthropic(llmSettings.base_url, llmPresets)
    : false;
  const selectedBaseUrlPreset = llmSettings
    ? matchingLlmPreset(llmSettings.base_url, llmPresets)
    : undefined;
  const effortLabel =
    llmSettings?.provider === "anthropic"
      ? "Effort"
      : "Reasoning Effort";
  const effortOptions = llmSettings
    ? effortOptionsFor(llmSettings.provider)
    : [];
  const testIcon =
    llmTest.status === "success" ? (
      <CheckCircle2 size={14} />
    ) : llmTest.status === "failure" ? (
      <XCircle size={14} />
    ) : (
      <Clock3 size={14} />
    );
  const sectionTitle =
    section === "general"
      ? "General"
      : section === "models"
        ? "Models"
        : section === "usage"
          ? "Usage"
        : section === "blender"
          ? "Blender RPC"
          : "SAM3";

  function renderServiceSettings(
    service: "blender_rpc" | "sam3",
    label: "Blender RPC" | "SAM3",
  ) {
    const urlKey = service === "blender_rpc" ? "blender_rpc_url" : "sam3_url";
    const state = serviceSettingsSaving || serviceSettingsLoading
      ? "checking"
      : serviceHealth?.services[service]?.ready === true
        ? "ready"
        : "offline";
    const stateLabel =
      state === "checking" ? "Checking" : state === "ready" ? "Ready" : "Unavailable";
    const detail =
      serviceHealth?.services[service]?.error ??
      serviceHealthError ??
      (serviceSettingsDisabled
        ? "Server URL is locked while a workflow is using Blender."
        : state === "ready"
          ? `${label} responded successfully.`
          : "Edit the URL to run a fresh connection check.");

    return (
      <div className="preference-service">
        <div className={`service-settings-status service-settings-status--${state}`} role="status">
          <span className="service-settings-status__dot" aria-hidden="true" />
          <div>
            <strong>{stateLabel}</strong>
            <span>{detail}</span>
          </div>
        </div>
        <div className="llm-field service-url-field">
          <label htmlFor={`service-url-${service}`}>
            <Link2 size={13} /> Server URL
          </label>
          <input
            id={`service-url-${service}`}
            aria-label={`${label} Server URL`}
            value={serviceSettings?.[urlKey] ?? ""}
            placeholder={serviceSettingsLoading ? "Loading…" : "http://127.0.0.1:…"}
            disabled={!serviceSettings || serviceSettingsDisabled}
            onChange={(event) => {
              if (!serviceSettings) return;
              onServiceSettingsChange({
                ...serviceSettings,
                [urlKey]: event.target.value,
              });
            }}
            autoComplete="off"
            spellCheck={false}
          />
        </div>
      </div>
    );
  }

  return (
    <div
      className={`preferences-root ${compact ? "preferences-root--compact" : ""}`}
    >
      {open ? (
        <div className="preferences-modal-layer">
          <button
            className="preferences-backdrop"
            type="button"
            aria-label="Close settings background"
            onClick={() => setOpen(false)}
          />
          <section
            className="preferences-popover"
            role="dialog"
            aria-modal="true"
            aria-label="Settings"
          >
            <aside className="preferences-navigation">
              <button
                className="preferences-close"
                type="button"
                aria-label="Close settings"
                title="Close settings"
                onClick={() => setOpen(false)}
              >
                <X size={19} />
              </button>
              <nav aria-label="Settings categories">
                <button
                  type="button"
                  className={section === "general" ? "active" : ""}
                  aria-current={section === "general" ? "page" : undefined}
                  onClick={() => setSection("general")}
                >
                  <Settings2 size={16} />
                  General
                </button>
                <button
                  type="button"
                  className={section === "models" ? "active" : ""}
                  aria-current={section === "models" ? "page" : undefined}
                  onClick={() => setSection("models")}
                >
                  <Cpu size={16} />
                  Models
                </button>
                <button
                  type="button"
                  className={section === "usage" ? "active" : ""}
                  aria-current={section === "usage" ? "page" : undefined}
                  onClick={() => setSection("usage")}
                >
                  <BarChart3 size={16} />
                  Usage
                </button>
                <button
                  type="button"
                  className={section === "blender" ? "active" : ""}
                  aria-current={section === "blender" ? "page" : undefined}
                  onClick={() => setSection("blender")}
                >
                  <Box size={16} />
                  Blender RPC
                </button>
                <button
                  type="button"
                  className={section === "sam3" ? "active" : ""}
                  aria-current={section === "sam3" ? "page" : undefined}
                  onClick={() => setSection("sam3")}
                >
                  <Sparkles size={16} />
                  SAM3
                </button>
              </nav>
            </aside>

            <div className="preferences-content">
              <div className="preferences-content-heading">
                <h2>{sectionTitle}</h2>
              </div>

              {section === "general" ? (
                <div className="preference-interface-grid">
                  <div className="preference-group">
                    <div className="preference-label">
                      <Sun size={14} />
                      <span>Theme</span>
                    </div>
                    <div
                      className="preference-options"
                      role="group"
                      aria-label="Theme"
                    >
                      {(["system", "light", "dark"] as const).map((option) => (
                        <button
                          type="button"
                          key={option}
                          aria-pressed={theme === option}
                          onClick={() => updateTheme(option)}
                        >
                          {option === "system" ? (
                            <Monitor size={13} />
                          ) : option === "light" ? (
                            <Sun size={13} />
                          ) : (
                            <Moon size={13} />
                          )}
                          <span>
                            {option[0].toUpperCase() + option.slice(1)}
                          </span>
                          {theme === option ? <Check size={12} /> : null}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="preference-group">
                    <div className="preference-label">
                      <Type size={14} />
                      <span>Size</span>
                    </div>
                    <div
                      className="preference-options"
                      role="group"
                      aria-label="Size"
                    >
                      {(["small", "medium", "large"] as const).map((option) => (
                        <button
                          type="button"
                          key={option}
                          aria-pressed={size === option}
                          onClick={() => updateSize(option)}
                        >
                          <span>{option[0].toUpperCase() + option.slice(1)}</span>
                          {size === option ? <Check size={12} /> : null}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              ) : section === "models" ? (
                <div className="preference-models">
                  <div className="llm-field">
                    <label htmlFor="llm-base-url">
                      <Link2 size={13} /> Base URL
                    </label>
                    <div className="llm-base-url-row">
                      <input
                        id="llm-base-url"
                        aria-label="Base URL"
                        value={llmSettings?.base_url ?? ""}
                        placeholder={llmSettingsLoading ? "Loading…" : "https://…"}
                        disabled={!llmSettings}
                        onChange={(event) => updateBaseUrl(event.target.value)}
                        autoComplete="off"
                        spellCheck={false}
                      />
                      <select
                        aria-label="Base URL preset"
                        title="Choose a Base URL preset"
                        value={selectedBaseUrlPreset?.id ?? ""}
                        disabled={!llmSettings || llmPresets.length === 0}
                        onChange={(event) => {
                          const preset = llmPresets.find(
                            (item) => item.id === event.target.value,
                          );
                          if (preset) updateBaseUrl(preset.base_url);
                        }}
                      >
                        <option value="">Custom</option>
                        {llmPresets.map((preset) => (
                          <option key={preset.id} value={preset.id}>
                            {preset.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div className="llm-field">
                    <label htmlFor="llm-endpoint-path">
                      <Route size={13} /> Endpoint
                    </label>
                    <div className="llm-endpoint-row">
                      <input
                        id="llm-endpoint-path"
                        aria-label="Endpoint path"
                        value={llmSettings?.endpoint_path ?? ""}
                        disabled={!llmSettings}
                        onChange={(event) => {
                          if (!llmSettings) return;
                          onLlmSettingsChange({
                            ...llmSettings,
                            endpoint_path: event.target.value,
                          });
                        }}
                        autoComplete="off"
                        spellCheck={false}
                      />
                      <select
                        aria-label="Endpoint compatibility"
                        value={llmSettings?.provider ?? "openai_compatible"}
                        disabled={!llmSettings}
                        onChange={(event) =>
                          updateProvider(event.target.value as LlmProvider)
                        }
                      >
                        <option value="openai_compatible">OpenAI Compatible</option>
                        <option value="anthropic" disabled={!anthropicAllowed}>
                          Anthropic Compatible
                        </option>
                      </select>
                    </div>
                  </div>

                  <div className="llm-field">
                    <label htmlFor="llm-effort">
                      <Gauge size={13} /> {effortLabel}
                    </label>
                    <select
                      id="llm-effort"
                      className="llm-effort-select"
                      aria-label={effortLabel}
                      value={llmSettings?.effort ?? ""}
                      disabled={!llmSettings}
                      onChange={(event) => updateEffort(event.target.value)}
                    >
                      <option value="">Not set</option>
                      {effortOptions.map((effort) => (
                        <option key={effort} value={effort}>
                          {effort}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="llm-model-list">
                    {LLM_ROLES.map(({ id, label }) => (
                      <div className="llm-model-field" key={id}>
                        <label htmlFor={`llm-model-${id}`}>{label}</label>
                        <div>
                          <input
                            id={`llm-model-${id}`}
                            aria-label={`${label} model ID`}
                            value={llmSettings?.models[id] ?? ""}
                            disabled={!llmSettings}
                            onChange={(event) => updateModel(id, event.target.value)}
                            autoComplete="off"
                            spellCheck={false}
                          />
                          <button
                            type="button"
                            aria-label={`Synchronize ${label} model ID`}
                            title="Use this Model ID for every workflow role"
                            disabled={!llmSettings}
                            onClick={() => synchronizeModel(id)}
                          >
                            <RefreshCw size={13} />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="llm-test-panel">
                    <div
                      className={`llm-test-status llm-test-status--${llmTest.status}`}
                      role="status"
                      aria-live="polite"
                    >
                      {testIcon}
                      <div>
                        <strong>
                          {llmTesting
                            ? "Testing"
                            : llmTest.status === "success"
                              ? "Connection verified"
                              : llmTest.status === "failure"
                                ? "Connection failed"
                                : "Waiting for test"}
                        </strong>
                        <span>
                          {llmTest.error ??
                            (llmTest.status === "success"
                              ? `${llmTest.result?.tested_model_count ?? 0} distinct model${llmTest.result?.tested_model_count === 1 ? "" : "s"} verified`
                              : "Run is locked until this configuration succeeds.")}
                        </span>
                      </div>
                    </div>
                    <button
                      className="llm-test-button"
                      type="button"
                      disabled={!llmSettings || llmTesting || llmSettingsLoading}
                      onClick={onTestLlm}
                    >
                      {llmTesting ? (
                        <LoaderCircle className="spin" size={14} />
                      ) : (
                        <RefreshCw size={14} />
                      )}
                      Test configuration
                    </button>
                  </div>
                </div>
              ) : section === "usage" ? (
                <div className="preference-usage">
                  <div className="usage-toolbar">
                    <span>
                      {usage
                        ? `${usage.aggregate.call_count} workflow calls`
                        : "Token accounting"}
                    </span>
                    <button
                      type="button"
                      onClick={onRefreshUsage}
                      disabled={usageLoading}
                    >
                      <RefreshCw
                        size={13}
                        className={usageLoading ? "spin" : ""}
                      />
                      Refresh
                    </button>
                  </div>

                  {usageError ? (
                    <div className="usage-error" role="alert">
                      {usageError}
                    </div>
                  ) : null}

                  {usage ? (
                    <>
                      <div className="usage-summary-grid">
                        {USAGE_METRICS.map(({ field, label }) => (
                          <div className="usage-summary-card" key={field}>
                            <span>{label}</span>
                            <strong>
                              <TokenValue
                                aggregate={usage.aggregate}
                                field={field}
                              />
                            </strong>
                          </div>
                        ))}
                      </div>

                      <section className="usage-section">
                        <div className="usage-section-heading">
                          <h3>By model</h3>
                          <span>{usage.by_model.length} models</span>
                        </div>
                        {usage.by_model.length ? (
                          <div className="usage-table-scroll">
                            <table className="usage-table">
                              <thead>
                                <tr>
                                  <th>Provider / model</th>
                                  <th>Calls</th>
                                  {USAGE_METRICS.map(({ field, label }) => (
                                    <th key={field}>{label}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {usage.by_model.map((model) => (
                                  <tr key={model.key}>
                                    <td>
                                      <span>{model.provider_label}</span>
                                      <small>{model.model}</small>
                                    </td>
                                    <td>{model.aggregate.call_count}</td>
                                    {USAGE_METRICS.map(({ field }) => (
                                      <td key={field}>
                                        <TokenValue
                                          aggregate={model.aggregate}
                                          field={field}
                                        />
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        ) : (
                          <p className="usage-empty">No workflow usage yet.</p>
                        )}
                      </section>

                      <section className="usage-section">
                        <div className="usage-section-heading">
                          <h3>Workflows</h3>
                          <span>{usage.workflows.length} runs</span>
                        </div>
                        <div className="usage-workflow-list">
                          {usage.workflows.map((workflow) => (
                            <details
                              className="usage-workflow"
                              key={workflow.run_id}
                            >
                              <summary>
                                <span>
                                  {workflow.title || `Workflow ${workflow.run_id.slice(0, 8)}`}
                                </span>
                                <small>
                                  {workflow.aggregate.call_count} calls · {formatTokenCount(workflow.aggregate.tokens.total_tokens)} total
                                </small>
                              </summary>
                              <div className="usage-workflow-detail">
                                <h4>By role</h4>
                                <div className="usage-table-scroll">
                                  <table className="usage-table usage-table--compact">
                                    <thead>
                                      <tr>
                                        <th>Role</th>
                                        <th>Calls</th>
                                        {USAGE_METRICS.map(({ field, label }) => (
                                          <th key={field}>{label}</th>
                                        ))}
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {workflow.by_role.map((role) => (
                                        <tr key={role.key}>
                                          <td>{role.label}</td>
                                          <td>{role.aggregate.call_count}</td>
                                          {USAGE_METRICS.map(({ field }) => (
                                            <td key={field}>
                                              <TokenValue
                                                aggregate={role.aggregate}
                                                field={field}
                                              />
                                            </td>
                                          ))}
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                                <h4>By model</h4>
                                <div className="usage-table-scroll">
                                  <table className="usage-table usage-table--compact">
                                    <thead>
                                      <tr>
                                        <th>Provider / model</th>
                                        <th>Calls</th>
                                        {USAGE_METRICS.map(({ field, label }) => (
                                          <th key={field}>{label}</th>
                                        ))}
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {workflow.by_model.map((model) => (
                                        <tr key={model.key}>
                                          <td>
                                            <span>{model.provider_label}</span>
                                            <small>{model.model}</small>
                                          </td>
                                          <td>{model.aggregate.call_count}</td>
                                          {USAGE_METRICS.map(({ field }) => (
                                            <td key={field}>
                                              <TokenValue
                                                aggregate={model.aggregate}
                                                field={field}
                                              />
                                            </td>
                                          ))}
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                              </div>
                            </details>
                          ))}
                        </div>
                      </section>

                      <div className="usage-model-tests">
                        <span>Model API tests</span>
                        <strong>
                          {formatTokenCount(
                            usage.model_tests.aggregate.tokens.total_tokens,
                          )}
                        </strong>
                        <small>
                          {usage.model_tests.aggregate.call_count} calls · excluded from Workflow total
                        </small>
                      </div>
                    </>
                  ) : (
                    <div className="usage-loading">
                      {usageLoading ? "Loading usage…" : "Usage is unavailable."}
                    </div>
                  )}
                </div>
              ) : section === "blender" ? (
                renderServiceSettings("blender_rpc", "Blender RPC")
              ) : (
                renderServiceSettings("sam3", "SAM3")
              )}
            </div>
          </section>
        </div>
      ) : null}

      <button
        className="preferences-button"
        type="button"
        aria-label="Open settings"
        title="Settings"
        aria-expanded={open}
        onClick={() => {
          if (open) {
            setOpen(false);
          } else {
            setSection("general");
            setOpen(true);
          }
        }}
      >
        <Settings2 size={17} />
        <span>Settings</span>
      </button>
    </div>
  );
}
