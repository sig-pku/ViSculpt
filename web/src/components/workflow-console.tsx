"use client";

import { useChannel, useStream } from "@langchain/react";
import type { RunCompletedInfo } from "@langchain/langgraph-sdk/stream";
import { AlertCircle, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";

import { ApprovalCard } from "@/components/approval-card";
import { ArtifactPanel } from "@/components/artifact-panel";
import { InstructionComposer } from "@/components/instruction-composer";
import { ManualMaskDialog } from "@/components/manual-mask-dialog";
import { ManualViewSelectionDialog } from "@/components/manual-view-selection-dialog";
import { PreferencesMenu } from "@/components/preferences-menu";
import { SubtaskPanel } from "@/components/subtask-panel";
import {
  ThreadSidebar,
  type ThreadListItem,
} from "@/components/thread-sidebar";
import { WorkflowOverview } from "@/components/workflow-overview";
import { WorkflowTimeline } from "@/components/workflow-timeline";
import {
  readBrowserSetting,
  removeBrowserSetting,
  writeBrowserSetting,
} from "@/lib/browser-storage";
import {
  llmSettingsFingerprint,
  PENDING_LLM_TEST,
  type LlmSettingsResponse,
  type LlmTestSnapshot,
  type RuntimeLlmPreset,
  type RuntimeLlmSettings,
} from "@/lib/llm-settings";
import {
  type RuntimeServiceSettings,
  type ServiceSettingsResponse,
} from "@/lib/service-settings";
import {
  isTokenUsageOverview,
  isWorkflowTokenUsageSummary,
  mergeUsageUpdate,
  usagePayload,
  type TokenUsageOverview,
  type WorkflowTokenUsageSummary,
} from "@/lib/token-usage";
import { cancelActiveWorkflowRuns } from "@/lib/workflow-cancellation";
import {
  collectAgentServerRunIds,
  collectWorkflowDeletionPlan,
} from "@/lib/workflow-deletion";
import {
  findWorkflowInterrupt,
  findWorkflowInterruptInState,
  interruptResponseTarget,
  type PendingWorkflowInterrupt,
} from "@/lib/workflow-interrupts";
import {
  EMPTY_WORKFLOW_STATE,
  isManualMaskRequest,
  isManualViewSelectionRequest,
  isSculptExecutionApproval,
  mergeWorkflowEvents,
  type JsonValue,
  type ManualMaskDecision,
  type ServiceHealth,
  type WorkflowInterrupt,
  type WorkflowState,
} from "@/lib/workflow-types";

const ACTIVE_THREAD_KEY = "geometry-agent.active-thread";
const WORKFLOW_EVENT_CHANNELS = ["custom"] as const;
const LLM_SETTINGS_SAVE_DELAY_MS = 450;
const SERVICE_SETTINGS_SAVE_DELAY_MS = 320;
const LLM_SETTINGS_SAVE_ERROR_PREFIX = "Could not save LLM settings: ";
const INTERRUPT_RECONCILIATION_DELAYS_MS = [0, 100, 300, 750] as const;

type WorkspaceViewTransition = {
  finished: Promise<void>;
  skipTransition: () => void;
};

type CancellationCleanupResponse = {
  blender_state_restored: boolean | null;
  blender_state_restore_response: JsonValue;
  viewport_ui_restored: boolean | null;
  viewport_ui_restore_response: WorkflowState["sculpt_viewport_ui_restore_response"];
  lease_released: boolean;
  cleanup_error: string | null;
};

type WorkflowSessionResponse = {
  lease?: { run_id?: unknown } | null;
};

type WorkflowArtifactDeletionResponse = {
  deleted_run_ids: string[];
  missing_run_ids: string[];
  deleted_file_count: number;
  deleted_directory_count: number;
  deleted_bytes: number;
};

type LocalCheckpointDeletionResponse = {
  backend: "in_memory" | "managed";
  purged: boolean;
};

type ThreadDeletionConfirmationResponse = {
  backend: "in_memory" | "managed";
  confirmed: boolean;
  deleted_usage_calls?: number;
};

type InterruptReconciliationRequest = {
  threadId: string;
  sequence: number;
  required: boolean;
};

const ACTIVE_WORKFLOW_STATUSES = new Set([
  "ready",
  "planning",
  "executing",
  "finishing",
]);

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === "string") return error;
  try {
    return JSON.stringify(error);
  } catch {
    return "An unknown workflow error occurred.";
  }
}

async function httpErrorMessage(response: Response, label: string) {
  const detail = (await response.text()).trim();
  return `${label} returned HTTP ${response.status}${detail ? `: ${detail}` : ""}`;
}

function createDraftThreadId() {
  const randomId =
    typeof window === "undefined" ? undefined : window.crypto?.randomUUID?.();
  const fallback = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  return `draft:${randomId ?? fallback}`;
}

export function WorkflowConsole() {
  const apiUrl = useMemo(() => {
    const origin =
      typeof window === "undefined"
        ? "http://127.0.0.1:3000"
        : window.location.origin;
    return new URL("/api/langgraph", origin).toString();
  }, []);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [instruction, setInstruction] = useState("");
  const [threads, setThreads] = useState<ThreadListItem[]>([]);
  const [draftThread, setDraftThread] = useState<ThreadListItem | null>(null);
  const [threadsLoading, setThreadsLoading] = useState(true);
  const [threadRefresh, setThreadRefresh] = useState(0);
  const [health, setHealth] = useState<ServiceHealth | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [responding, setResponding] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [cancelledThreadId, setCancelledThreadId] = useState<string | null>(null);
  const [workspaceActive, setWorkspaceActive] = useState(false);
  const [workspaceTransitioning, setWorkspaceTransitioning] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [artifactsCollapsed, setArtifactsCollapsed] = useState(false);
  const [deletingThreadId, setDeletingThreadId] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const [reconciledInterrupt, setReconciledInterrupt] =
    useState<PendingWorkflowInterrupt | null>(null);
  const [interruptReconciliation, setInterruptReconciliation] =
    useState<InterruptReconciliationRequest | null>(null);
  const [llmSettings, setLlmSettings] =
    useState<RuntimeLlmSettings | null>(null);
  const [llmPresets, setLlmPresets] = useState<RuntimeLlmPreset[]>([]);
  const [llmTest, setLlmTest] =
    useState<LlmTestSnapshot>(PENDING_LLM_TEST);
  const [llmSettingsLoading, setLlmSettingsLoading] = useState(true);
  const [llmTesting, setLlmTesting] = useState(false);
  const [serviceSettings, setServiceSettings] =
    useState<RuntimeServiceSettings | null>(null);
  const [serviceSettingsLoading, setServiceSettingsLoading] = useState(true);
  const [serviceSettingsSaving, setServiceSettingsSaving] = useState(false);
  const [usage, setUsage] = useState<TokenUsageOverview | null>(null);
  const [usageLoading, setUsageLoading] = useState(true);
  const [usageError, setUsageError] = useState<string | null>(null);
  const [threadMetadataUpdate, setThreadMetadataUpdate] = useState<{
    threadId: string;
    title: string;
  } | null>(null);
  const pendingTitle = useRef<string | null>(null);
  const currentThreadId = useRef<string | null>(null);
  const draftThreadRef = useRef<ThreadListItem | null>(null);
  const workspaceTransitionRef = useRef<WorkspaceViewTransition | null>(null);
  const workspaceTransitionFrame = useRef<number | null>(null);
  const llmSettingsRef = useRef<RuntimeLlmSettings | null>(null);
  const llmSettingsEdited = useRef(false);
  const llmRequestGeneration = useRef(0);
  const llmSettingsSaveTimer = useRef<number | null>(null);
  const llmSettingsSaveQueue = useRef<Promise<void>>(Promise.resolve());
  const serviceSettingsRef = useRef<RuntimeServiceSettings | null>(null);
  const serviceSettingsEdited = useRef(false);
  const serviceSettingsPending = useRef(false);
  const serviceRequestGeneration = useRef(0);
  const serviceSettingsSaveTimer = useRef<number | null>(null);
  const usageRefreshTimer = useRef<number | null>(null);
  const lastUsageEventId = useRef<string | null>(null);
  const interruptReconciliationSequence = useRef(0);
  const interruptReconciliationGeneration = useRef(0);
  const pendingInterruptWithoutThreadId = useRef(false);

  const clearReconciledInterrupt = useCallback(() => {
    pendingInterruptWithoutThreadId.current = false;
    interruptReconciliationGeneration.current += 1;
    setReconciledInterrupt(null);
    setInterruptReconciliation(null);
  }, []);

  const queueInterruptReconciliation = useCallback(
    (targetThreadId: string | null, required: boolean) => {
      if (!targetThreadId) return;
      interruptReconciliationSequence.current += 1;
      setInterruptReconciliation({
        threadId: targetThreadId,
        sequence: interruptReconciliationSequence.current,
        required,
      });
    },
    [],
  );

  const updateDraftThread = useCallback((draft: ThreadListItem | null) => {
    draftThreadRef.current = draft;
    setDraftThread(draft);
  }, []);

  const transitionWorkspace = useCallback((active: boolean) => {
    if (workspaceTransitionFrame.current !== null) {
      window.cancelAnimationFrame(workspaceTransitionFrame.current);
      workspaceTransitionFrame.current = null;
    }
    const update = () => flushSync(() => {
      setWorkspaceTransitioning(true);
      setWorkspaceActive(active);
    });
    const transitionDocument = document as Document & {
      startViewTransition?: (
        callback: () => void,
      ) => WorkspaceViewTransition;
    };
    const reduceMotion = window.matchMedia?.(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    if (!transitionDocument.startViewTransition || reduceMotion) {
      update();
      workspaceTransitionFrame.current = window.requestAnimationFrame(() => {
        workspaceTransitionFrame.current = null;
        setWorkspaceTransitioning(false);
      });
      return;
    }

    // Use one layout transition to prevent overlapping snapshot and CSS flicker.
    workspaceTransitionRef.current?.skipTransition();
    const transition = transitionDocument.startViewTransition(update);
    workspaceTransitionRef.current = transition;
    void transition.finished.finally(() => {
      if (workspaceTransitionRef.current === transition) {
        workspaceTransitionRef.current = null;
        setWorkspaceTransitioning(false);
      }
    });
  }, []);

  const handleThreadId = useCallback((nextThreadId: string) => {
    const draft = draftThreadRef.current;
    const title = pendingTitle.current ?? (
      typeof draft?.metadata.title === "string"
        ? draft.metadata.title
        : "New task"
    );
    if (draft || pendingTitle.current) {
      setThreads((items) => [
        {
          thread_id: nextThreadId,
          metadata: { title, client: "workflow-console" },
          status: "busy",
        },
        ...items.filter((item) => item.thread_id !== nextThreadId),
      ]);
      updateDraftThread(null);
    }
    currentThreadId.current = nextThreadId;
    setThreadId(nextThreadId);
    writeBrowserSetting(ACTIVE_THREAD_KEY, nextThreadId);
    if (pendingInterruptWithoutThreadId.current) {
      pendingInterruptWithoutThreadId.current = false;
      queueInterruptReconciliation(nextThreadId, true);
    }
  }, [queueInterruptReconciliation, updateDraftThread]);

  const handleRunCreated = useCallback(() => {
    const title = pendingTitle.current;
    const createdThreadId = currentThreadId.current;
    if (title && createdThreadId) {
      // The first run created the server thread, so title updates cannot race it.
      pendingTitle.current = null;
      setThreadMetadataUpdate({ threadId: createdThreadId, title });
    }
  }, []);

  const handleRunCompleted = useCallback((info: RunCompletedInfo) => {
    setThreadsLoading(true);
    setThreadRefresh((value) => value + 1);
    setResponding(false);
    if (info.reason === "interrupt") {
      const targetThreadId = currentThreadId.current;
      if (targetThreadId) {
        queueInterruptReconciliation(targetThreadId, true);
      } else {
        pendingInterruptWithoutThreadId.current = true;
      }
    } else {
      clearReconciledInterrupt();
    }
  }, [clearReconciledInterrupt, queueInterruptReconciliation]);

  const stream = useStream<WorkflowState, WorkflowInterrupt>({
    assistantId: "sculpt_agent",
    apiUrl,
    threadId,
    initialValues: EMPTY_WORKFLOW_STATE,
    messagesKey: "messages",
    optimistic: false,
    onThreadId: handleThreadId,
    onCreated: handleRunCreated,
    onCompleted: handleRunCompleted,
  });

  const customEvents = useChannel(
    stream,
    WORKFLOW_EVENT_CHANNELS,
    undefined,
    { bufferSize: 512, replay: true },
  );
  const state = stream.values;
  const displayedThreadId = stream.threadId ?? threadId;
  const liveInterrupt = useMemo(
    () => findWorkflowInterrupt(stream.interrupts),
    [stream.interrupts],
  );
  const activeInterrupt = liveInterrupt ?? reconciledInterrupt;
  const displayedState = useMemo<WorkflowState>(
    () =>
      cancelledThreadId !== null && cancelledThreadId === displayedThreadId
        ? { ...state, workflow_status: "cancelled" }
        : state,
    [cancelledThreadId, displayedThreadId, state],
  );
  const events = useMemo(
    () => mergeWorkflowEvents(state.events, customEvents),
    [customEvents, state.events],
  );
  const approval = isSculptExecutionApproval(activeInterrupt?.value)
    ? activeInterrupt.value
    : null;
  const manualViewSelection = isManualViewSelectionRequest(
    activeInterrupt?.value,
  )
    ? activeInterrupt.value
    : null;
  const manualMask = isManualMaskRequest(activeInterrupt?.value)
    ? activeInterrupt.value
    : null;
  const workflowCanStop =
    stream.isLoading ||
    ACTIVE_WORKFLOW_STATUSES.has(displayedState.workflow_status ?? "created");
  const liveInterruptIdentity = liveInterrupt
    ? `${liveInterrupt.value.type}:${liveInterrupt.id ?? (
        "approval_id" in liveInterrupt.value
          ? liveInterrupt.value.approval_id
          : liveInterrupt.value.intervention_id
      )}`
    : null;

  useEffect(() => {
    if (!liveInterruptIdentity) return;
    interruptReconciliationGeneration.current += 1;
    setReconciledInterrupt(null);
    setInterruptReconciliation(null);
  }, [liveInterruptIdentity]);

  useEffect(() => {
    if (!interruptReconciliation || liveInterrupt) return;

    const request = interruptReconciliation;
    const generation = interruptReconciliationGeneration.current + 1;
    interruptReconciliationGeneration.current = generation;
    const abortController = new AbortController();
    const timers = new Set<number>();
    let disposed = false;

    const isCurrentRequest = () =>
      !disposed &&
      !abortController.signal.aborted &&
      interruptReconciliationGeneration.current === generation &&
      currentThreadId.current === request.threadId;

    const wait = (delay: number) =>
      new Promise<void>((resolve) => {
        if (delay <= 0) {
          resolve();
          return;
        }
        const timer = window.setTimeout(() => {
          timers.delete(timer);
          resolve();
        }, delay);
        timers.add(timer);
      });

    void (async () => {
      let lastError: unknown = null;
      for (const delay of INTERRUPT_RECONCILIATION_DELAYS_MS) {
        await wait(delay);
        if (!isCurrentRequest()) return;
        try {
          const snapshot =
            await stream.client.threads.getState<WorkflowState>(
              request.threadId,
              undefined,
              { signal: abortController.signal },
            );
          if (!isCurrentRequest()) return;
          lastError = null;
          const pending = findWorkflowInterruptInState(snapshot);
          if (pending) {
            setReconciledInterrupt(pending);
            setInterruptReconciliation((current) =>
              current?.sequence === request.sequence ? null : current,
            );
            return;
          }
        } catch (error) {
          if (!isCurrentRequest()) return;
          lastError = error;
        }
      }

      if (!isCurrentRequest()) return;
      setInterruptReconciliation((current) =>
        current?.sequence === request.sequence ? null : current,
      );
      if (request.required) {
        setLocalError(
          lastError
            ? `Workflow paused for user input, but its pending prompt could not be loaded: ${errorMessage(lastError)}`
            : "Workflow paused for user input, but no supported pending prompt was found in its saved state.",
        );
      }
    })();

    return () => {
      disposed = true;
      abortController.abort();
      for (const timer of timers) window.clearTimeout(timer);
      timers.clear();
    };
  }, [interruptReconciliation, liveInterrupt, stream.client]);

  const refreshUsage = useCallback(async () => {
    setUsageLoading(true);
    try {
      const response = await fetch("/api/langgraph/workflow/usage", {
        cache: "no-store",
      });
      if (!response.ok) {
        throw new Error(`Usage endpoint returned ${response.status}`);
      }
      const payload: unknown = await response.json();
      if (!isTokenUsageOverview(payload)) {
        throw new Error("Usage endpoint returned an invalid payload");
      }
      setUsage(payload);
      setUsageError(null);
    } catch (error) {
      setUsageError(errorMessage(error));
    } finally {
      setUsageLoading(false);
    }
  }, []);

  const latestUsageUpdate = (() => {
    for (let index = events.length - 1; index >= 0; index -= 1) {
      const payload = usagePayload(events[index]);
      if (payload) return { eventId: events[index].event_id, payload };
    }
    return null;
  })();

  const currentWorkflowUsage: WorkflowTokenUsageSummary | null = (() => {
    const runId = displayedState.run_id;
    if (!runId) return null;
    if (
      latestUsageUpdate?.payload.workflow_summary.run_id === runId
    ) {
      return latestUsageUpdate.payload.workflow_summary;
    }
    if (isWorkflowTokenUsageSummary(displayedState.token_usage)) {
      return displayedState.token_usage;
    }
    return usage?.workflows?.find((item) => item.run_id === runId) ?? null;
  })();

  const refreshHealth = useCallback(async () => {
    if (serviceSettingsPending.current) return;
    try {
      const response = await fetch(
        "/api/langgraph/workflow/services/health",
        { cache: "no-store" },
      );
      if (!response.ok) {
        throw new Error(`Health endpoint returned ${response.status}`);
      }
      setHealth((await response.json()) as ServiceHealth);
      setHealthError(null);
    } catch (error) {
      setHealthError(errorMessage(error));
    }
  }, []);

  const refreshServiceSettings = useCallback(async () => {
    try {
      const response = await fetch(
        "/api/langgraph/workflow/services/settings",
        { cache: "no-store" },
      );
      if (!response.ok) {
        throw new Error(`Service settings endpoint returned ${response.status}`);
      }
      const payload = (await response.json()) as ServiceSettingsResponse;
      if (!serviceSettingsEdited.current) {
        serviceSettingsRef.current = payload.settings;
        setServiceSettings(payload.settings);
      }
      setServiceSettingsLoading(false);
    } catch (error) {
      setServiceSettingsLoading(false);
      setHealthError(errorMessage(error));
    }
  }, []);

  const persistServiceSettings = useCallback(
    async (settings: RuntimeServiceSettings, generation: number) => {
      try {
        const response = await fetch(
          "/api/langgraph/workflow/services/settings",
          {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(settings),
          },
        );
        if (!response.ok) {
          throw new Error(
            `HTTP ${response.status}: ${await response.text()}`,
          );
        }
        const payload = (await response.json()) as ServiceSettingsResponse;
        if (generation !== serviceRequestGeneration.current) return;
        serviceSettingsRef.current = payload.settings;
        setServiceSettings(payload.settings);
        if (payload.health) setHealth(payload.health);
        setHealthError(null);
      } catch (error) {
        if (generation === serviceRequestGeneration.current) {
          setHealth(null);
          setHealthError(errorMessage(error));
        }
      } finally {
        if (generation === serviceRequestGeneration.current) {
          serviceSettingsPending.current = false;
          setServiceSettingsSaving(false);
        }
      }
    },
    [],
  );

  const updateServiceSettings = useCallback(
    (settings: RuntimeServiceSettings) => {
      serviceSettingsEdited.current = true;
      serviceSettingsPending.current = true;
      serviceSettingsRef.current = settings;
      const generation = serviceRequestGeneration.current + 1;
      serviceRequestGeneration.current = generation;
      setServiceSettings(settings);
      setServiceSettingsSaving(true);
      setHealth(null);
      setHealthError(null);
      if (serviceSettingsSaveTimer.current !== null) {
        window.clearTimeout(serviceSettingsSaveTimer.current);
      }
      if (usageRefreshTimer.current !== null) {
        window.clearTimeout(usageRefreshTimer.current);
      }
      serviceSettingsSaveTimer.current = window.setTimeout(() => {
        serviceSettingsSaveTimer.current = null;
        void persistServiceSettings(settings, generation);
      }, SERVICE_SETTINGS_SAVE_DELAY_MS);
    },
    [persistServiceSettings],
  );

  const refreshLlmSettings = useCallback(async () => {
    try {
      const response = await fetch(
        "/api/langgraph/workflow/llm/settings",
        { cache: "no-store" },
      );
      if (!response.ok) {
        throw new Error(`LLM settings endpoint returned ${response.status}`);
      }
      const payload = (await response.json()) as LlmSettingsResponse;
      setLlmPresets(payload.presets ?? []);
      if (!llmSettingsEdited.current) {
        llmSettingsRef.current = payload.settings;
        setLlmSettings(payload.settings);
        setLlmTest(payload.test);
      }
      setLlmSettingsLoading(false);
    } catch (error) {
      setLlmSettingsLoading(false);
      setLocalError(errorMessage(error));
    }
  }, []);

  const persistLlmSettings = useCallback(
    (settings: RuntimeLlmSettings): Promise<string | null> => {
      const request = llmSettingsSaveQueue.current.then(async () => {
        const response = await fetch(
          "/api/langgraph/workflow/llm/settings",
          {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(settings),
          },
        );
        if (!response.ok) {
          throw new Error(
            `HTTP ${response.status}: ${await response.text()}`,
          );
        }
      });
      llmSettingsSaveQueue.current = request.catch(() => undefined);
      return request.then(
        () => {
          setLocalError((current) =>
            current?.startsWith(LLM_SETTINGS_SAVE_ERROR_PREFIX)
              ? null
              : current,
          );
          return null;
        },
        (error) => {
          const message = `${LLM_SETTINGS_SAVE_ERROR_PREFIX}${errorMessage(error)}`;
          setLocalError(message);
          return message;
        },
      );
    },
    [],
  );

  const updateLlmSettings = useCallback(
    (settings: RuntimeLlmSettings) => {
      llmSettingsEdited.current = true;
      llmRequestGeneration.current += 1;
      llmSettingsRef.current = settings;
      setLlmSettings(settings);
      setLlmTesting(false);
      setLlmTest({ ...PENDING_LLM_TEST });
      if (llmSettingsSaveTimer.current !== null) {
        window.clearTimeout(llmSettingsSaveTimer.current);
      }
      llmSettingsSaveTimer.current = window.setTimeout(() => {
        llmSettingsSaveTimer.current = null;
        void persistLlmSettings(settings);
      }, LLM_SETTINGS_SAVE_DELAY_MS);
    },
    [persistLlmSettings],
  );

  const testLlmSettings = useCallback(async () => {
    const candidate = llmSettingsRef.current;
    if (!candidate) return;
    const candidateFingerprint = llmSettingsFingerprint(candidate);
    const generation = llmRequestGeneration.current + 1;
    llmRequestGeneration.current = generation;
    setLlmTesting(true);
    setLlmTest({ ...PENDING_LLM_TEST });
    try {
      if (llmSettingsSaveTimer.current !== null) {
        window.clearTimeout(llmSettingsSaveTimer.current);
        llmSettingsSaveTimer.current = null;
      }
      const saveError = await persistLlmSettings(candidate);
      const currentAfterSave = llmSettingsRef.current;
      if (
        generation !== llmRequestGeneration.current ||
        !currentAfterSave ||
        llmSettingsFingerprint(currentAfterSave) !== candidateFingerprint
      ) {
        return;
      }
      if (saveError) {
        setLlmTest({
          ...PENDING_LLM_TEST,
          status: "failure",
          error: saveError,
        });
        return;
      }
      const response = await fetch("/api/langgraph/workflow/llm/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(candidate),
      });
      if (!response.ok) {
        throw new Error(
          `LLM test returned ${response.status}: ${await response.text()}`,
        );
      }
      const payload = (await response.json()) as LlmSettingsResponse;
      const current = llmSettingsRef.current;
      if (
        generation === llmRequestGeneration.current &&
        current &&
        llmSettingsFingerprint(current) === candidateFingerprint
      ) {
        setLlmTest(payload.test);
        void refreshUsage();
      }
    } catch (error) {
      if (generation === llmRequestGeneration.current) {
        setLlmTest({
          ...PENDING_LLM_TEST,
          status: "failure",
          error: errorMessage(error),
        });
      }
    } finally {
      if (generation === llmRequestGeneration.current) {
        setLlmTesting(false);
      }
    }
  }, [persistLlmSettings, refreshUsage]);

  useEffect(() => {
    return () => {
      if (llmSettingsSaveTimer.current !== null) {
        window.clearTimeout(llmSettingsSaveTimer.current);
      }
      if (serviceSettingsSaveTimer.current !== null) {
        window.clearTimeout(serviceSettingsSaveTimer.current);
      }
      if (workspaceTransitionFrame.current !== null) {
        window.cancelAnimationFrame(workspaceTransitionFrame.current);
      }
    };
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const storedThreadId = readBrowserSetting(ACTIVE_THREAD_KEY);
      if (storedThreadId) {
        currentThreadId.current = storedThreadId;
        setThreadId(storedThreadId);
        setWorkspaceActive(true);
        queueInterruptReconciliation(storedThreadId, false);
      }
    }, 0);
    return () => window.clearTimeout(timer);
  }, [queueInterruptReconciliation]);

  useEffect(() => {
    // Avoid model contention between health probes and active SAM 3 inference.
    if (workflowCanStop) return;
    const initialTimer = window.setTimeout(() => void refreshHealth(), 0);
    const pollTimer = window.setInterval(() => void refreshHealth(), 30_000);
    return () => {
      window.clearTimeout(initialTimer);
      window.clearInterval(pollTimer);
    };
  }, [refreshHealth, workflowCanStop]);

  useEffect(() => {
    const timer = window.setTimeout(() => void refreshLlmSettings(), 0);
    return () => window.clearTimeout(timer);
  }, [refreshLlmSettings]);

  useEffect(() => {
    const timer = window.setTimeout(() => void refreshServiceSettings(), 0);
    return () => window.clearTimeout(timer);
  }, [refreshServiceSettings]);

  useEffect(() => {
    const timer = window.setTimeout(() => void refreshUsage(), 0);
    return () => window.clearTimeout(timer);
  }, [refreshUsage]);

  useEffect(() => {
    if (
      !latestUsageUpdate ||
      lastUsageEventId.current === latestUsageUpdate.eventId
    ) {
      return;
    }
    lastUsageEventId.current = latestUsageUpdate.eventId;
    setUsage((current) =>
      mergeUsageUpdate(current, latestUsageUpdate.payload),
    );
    if (usageRefreshTimer.current !== null) {
      window.clearTimeout(usageRefreshTimer.current);
    }
    usageRefreshTimer.current = window.setTimeout(() => {
      usageRefreshTimer.current = null;
      void refreshUsage();
    }, 80);
  }, [latestUsageUpdate, refreshUsage]);

  useEffect(() => {
    if (
      llmSettingsLoading ||
      llmSettingsEdited.current ||
      llmTest.status !== "pending"
    ) {
      return;
    }
    const timer = window.setInterval(
      () => void refreshLlmSettings(),
      1_500,
    );
    return () => window.clearInterval(timer);
  }, [llmSettingsLoading, llmTest.status, refreshLlmSettings]);

  useEffect(() => {
    let cancelled = false;
    void stream.client.threads
      .search({
        limit: 24,
        sortBy: "updated_at",
        sortOrder: "desc",
        select: ["thread_id", "updated_at", "metadata", "status"],
      })
      .then(async (items) => {
        const restoredItems = await Promise.all(
          (items as ThreadListItem[]).map(async (item) => {
            const savedTitle = item.metadata.title;
            if (typeof savedTitle === "string" && savedTitle.trim()) {
              return item;
            }
            try {
              const snapshot =
                await stream.client.threads.getState<WorkflowState>(
                  item.thread_id,
                );
              const instruction = snapshot.values.user_instruction;
              const title =
                typeof instruction === "string" ? instruction.trim() : "";
              if (!title) return item;
              const restored = {
                ...item,
                metadata: {
                  ...item.metadata,
                  title,
                  client: "workflow-console",
                },
              };
              try {
                await stream.client.threads.update(item.thread_id, {
                  metadata: restored.metadata,
                });
              } catch {
                // Prefer the state instruction even when persistence repair fails.
              }
              return restored;
            } catch {
              return item;
            }
          }),
        );
        if (!cancelled) setThreads(restoredItems);
      })
      .catch((error) => {
        if (!cancelled) setLocalError(errorMessage(error));
      })
      .finally(() => {
        if (!cancelled) setThreadsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [stream.client, threadRefresh]);

  useEffect(() => {
    if (!threadMetadataUpdate) return;
    let cancelled = false;
    void stream.client.threads
      .update(threadMetadataUpdate.threadId, {
        metadata: {
          title: threadMetadataUpdate.title,
          client: "workflow-console",
        },
      })
      .then(() => {
        if (!cancelled) {
          setThreads((items) =>
            items.map((item) =>
              item.thread_id === threadMetadataUpdate.threadId
                ? {
                    ...item,
                    metadata: {
                      ...item.metadata,
                      title: threadMetadataUpdate.title,
                    },
                  }
                : item,
            ),
          );
          setThreadsLoading(true);
          setThreadRefresh((value) => value + 1);
        }
      })
      .catch((error) => {
        if (!cancelled) setLocalError(errorMessage(error));
      })
      .finally(() => {
        if (!cancelled) setThreadMetadataUpdate(null);
      });
    return () => {
      cancelled = true;
    };
  }, [stream.client, threadMetadataUpdate]);

  async function submitInstruction() {
    const value = instruction.trim();
    const runLlmSettings = llmSettingsRef.current;
    if (
      !value ||
      workspaceActive ||
      stream.isLoading ||
      llmTest.status !== "success" ||
      !runLlmSettings
    ) {
      return;
    }
    clearReconciledInterrupt();
    setLocalError(null);
    setCancelledThreadId(null);
    if (!currentThreadId.current) {
      pendingTitle.current = value;
      const draft = draftThreadRef.current ?? {
        thread_id: createDraftThreadId(),
        metadata: { title: "New task" },
        isDraft: true,
      };
      updateDraftThread({
        ...draft,
        metadata: { ...draft.metadata, title: value },
      });
    }
    transitionWorkspace(true);
    try {
      await stream.submit(
        { user_instruction: value },
        {
          multitaskStrategy: "reject",
          metadata: {
            source: "workflow-console",
            title: value,
            client: "workflow-console",
          },
          config: {
            configurable: {
              llm: {
                ...runLlmSettings,
                models: { ...runLlmSettings.models },
              },
            },
          },
        },
      );
    } catch (error) {
      setLocalError(errorMessage(error));
    }
  }

  async function stopRun() {
    if (stopping) return;
    clearReconciledInterrupt();
    setStopping(true);
    setLocalError(null);
    try {
      const activeThreadId = stream.threadId ?? currentThreadId.current;
      if (!activeThreadId) {
        throw new Error("Cannot stop a workflow before its thread is created");
      }

      const stateBeforeCancellation = stream.client.threads
        .getState<WorkflowState>(activeThreadId)
        .then((snapshot) => snapshot.values)
        .catch(() => null);
      await cancelActiveWorkflowRuns(stream.client.runs, activeThreadId);
      const beforeState = await stateBeforeCancellation;
      let afterState: WorkflowState | null = null;
      let stateReadError: unknown;
      try {
        afterState = (
          await stream.client.threads.getState<WorkflowState>(activeThreadId)
        ).values;
      } catch (error) {
        stateReadError = error;
      }
      await stream.stop({ cancel: false });

      const warnings: string[] = [];
      const cleanupState = afterState?.run_id ? afterState : beforeState;
      if (!cleanupState?.run_id && stateReadError) {
        warnings.push(
          `could not read the current workflow state: ${errorMessage(stateReadError)}`,
        );
      }
      let cleanup: CancellationCleanupResponse | null = null;
      if (cleanupState?.run_id) {
        const cleanupResponse = await fetch(
          `/api/langgraph/workflow/session/${cleanupState.run_id}/cancel-cleanup`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              sculpt_viewport_ui_snapshot:
                cleanupState.sculpt_viewport_ui_snapshot ?? null,
              checkpoint_path: cleanupState.checkpoint_path ?? null,
            }),
          },
        );
        if (cleanupResponse.ok) {
          cleanup = (await cleanupResponse.json()) as CancellationCleanupResponse;
          if (cleanup.cleanup_error) warnings.push(cleanup.cleanup_error);
        } else {
          warnings.push(
            `cancellation cleanup returned ${cleanupResponse.status}`,
          );
        }
      }

      try {
        await stream.client.threads.updateState<Partial<WorkflowState>>(
          activeThreadId,
          {
            asNode: "finalize",
            values: {
              workflow_status: "cancelled",
              blender_session_lease: null,
              sculpt_viewport_ui_restore_response:
                cleanup?.viewport_ui_restore_response ?? null,
            },
          },
        );
      } catch (error) {
        warnings.push(`could not persist cancelled state: ${errorMessage(error)}`);
      }

      setCancelledThreadId(activeThreadId);
      if (warnings.length > 0) {
        setLocalError(`Workflow stopped, but ${warnings.join("; ")}`);
      }
      setThreadRefresh((value) => value + 1);
      setThreadsLoading(true);
    } catch (error) {
      setLocalError(errorMessage(error));
    } finally {
      setStopping(false);
    }
  }

  async function respondToApproval(decision: "approve" | "reject") {
    const responseTarget = interruptResponseTarget(activeInterrupt);
    clearReconciledInterrupt();
    setResponding(true);
    setLocalError(null);
    try {
      await stream.respond({ decision }, responseTarget);
    } catch (error) {
      setResponding(false);
      setLocalError(errorMessage(error));
      queueInterruptReconciliation(currentThreadId.current, true);
    }
  }

  async function respondToManualViewSelection(
    decision: { decision: "select"; view: string } | { decision: "skip" },
  ) {
    const responseTarget = interruptResponseTarget(activeInterrupt);
    clearReconciledInterrupt();
    setResponding(true);
    setLocalError(null);
    try {
      await stream.respond(decision, responseTarget);
    } catch (error) {
      setResponding(false);
      setLocalError(errorMessage(error));
      queueInterruptReconciliation(currentThreadId.current, true);
    }
  }

  async function respondToManualMask(decision: ManualMaskDecision) {
    const responseTarget = interruptResponseTarget(activeInterrupt);
    clearReconciledInterrupt();
    setResponding(true);
    setLocalError(null);
    try {
      await stream.respond(decision, responseTarget);
    } catch (error) {
      setResponding(false);
      setLocalError(errorMessage(error));
      queueInterruptReconciliation(currentThreadId.current, true);
    }
  }

  function createNewThread() {
    if (workflowCanStop) return;
    clearReconciledInterrupt();
    updateDraftThread(null);
    currentThreadId.current = null;
    setThreadId(null);
    setInstruction("");
    setLocalError(null);
    setCancelledThreadId(null);
    pendingTitle.current = null;
    removeBrowserSetting(ACTIVE_THREAD_KEY);
    transitionWorkspace(false);
  }

  function selectThread(nextThreadId: string) {
    if (workflowCanStop) return;
    if (nextThreadId === draftThreadRef.current?.thread_id) {
      if (!workspaceActive) transitionWorkspace(true);
      return;
    }
    if (nextThreadId === threadId) {
      queueInterruptReconciliation(nextThreadId, false);
      return;
    }
    clearReconciledInterrupt();
    currentThreadId.current = nextThreadId;
    setThreadId(nextThreadId);
    setInstruction("");
    setLocalError(null);
    setCancelledThreadId(null);
    writeBrowserSetting(ACTIVE_THREAD_KEY, nextThreadId);
    queueInterruptReconciliation(nextThreadId, false);
    // Replace thread data in place to avoid duplicate shared-element animation.
    if (!workspaceActive) transitionWorkspace(true);
  }

  async function deleteThread(targetThreadId: string) {
    if (deletingThreadId) return;
    setLocalError(null);
    if (targetThreadId === draftThreadRef.current?.thread_id) {
      updateDraftThread(null);
      if (!currentThreadId.current) transitionWorkspace(false);
      return;
    }
    setDeletingThreadId(targetThreadId);
    try {
      const deletingDisplayedThread = targetThreadId === displayedThreadId;
      const initialPlan = await collectWorkflowDeletionPlan(
        stream.client.threads,
        targetThreadId,
        deletingDisplayedThread ? state : null,
      );

      // Stop active runs before deletion so they cannot recreate state or assets.
      await cancelActiveWorkflowRuns(stream.client.runs, targetThreadId);
      if (deletingDisplayedThread) {
        await stream.stop({ cancel: false });
      }

      // Re-read settled history to include final writes from the race window.
      const settledPlan = await collectWorkflowDeletionPlan(
        stream.client.threads,
        targetThreadId,
        deletingDisplayedThread ? state : null,
      );
      const plan = {
        runIds: [
          ...new Set([...initialPlan.runIds, ...settledPlan.runIds]),
        ],
        viewportSnapshots: {
          ...settledPlan.viewportSnapshots,
          ...initialPlan.viewportSnapshots,
        },
      };
      const agentServerRunIds = await collectAgentServerRunIds(
        stream.client.runs,
        targetThreadId,
      );

      const sessionResponse = await fetch(
        "/api/langgraph/workflow/session",
        { cache: "no-store" },
      );
      if (!sessionResponse.ok) {
        throw new Error(
          await httpErrorMessage(sessionResponse, "Session inspection"),
        );
      }
      const session = (await sessionResponse.json()) as WorkflowSessionResponse;
      const leaseRunId = session.lease?.run_id;
      if (
        typeof leaseRunId === "string" &&
        plan.runIds.includes(leaseRunId)
      ) {
        const cleanupResponse = await fetch(
          `/api/langgraph/workflow/session/${leaseRunId}/cancel-cleanup`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              sculpt_viewport_ui_snapshot:
                plan.viewportSnapshots[leaseRunId] ?? null,
            }),
          },
        );
        if (!cleanupResponse.ok) {
          throw new Error(
            await httpErrorMessage(
              cleanupResponse,
              "Workflow cancellation cleanup",
            ),
          );
        }
        const cleanup =
          (await cleanupResponse.json()) as CancellationCleanupResponse;
        if (cleanup.cleanup_error) {
          throw new Error(cleanup.cleanup_error);
        }
      }

      const artifactResponse = await fetch(
        "/api/langgraph/workflow/artifacts",
        {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ run_ids: plan.runIds }),
        },
      );
      if (!artifactResponse.ok) {
        throw new Error(
          await httpErrorMessage(artifactResponse, "Artifact deletion"),
        );
      }
      const artifactDeletion =
        (await artifactResponse.json()) as WorkflowArtifactDeletionResponse;
      const acknowledgedRunIds = new Set([
        ...artifactDeletion.deleted_run_ids,
        ...artifactDeletion.missing_run_ids,
      ]);
      const unacknowledged = plan.runIds.filter(
        (runId) => !acknowledgedRunIds.has(runId),
      );
      if (unacknowledged.length > 0) {
        throw new Error(
          `Artifact deletion did not account for Workflow runs: ${unacknowledged.join(", ")}`,
        );
      }

      const usageResponse = await fetch(
        "/api/langgraph/workflow/usage",
        {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ run_ids: plan.runIds }),
        },
      );
      if (!usageResponse.ok) {
        throw new Error(
          await httpErrorMessage(usageResponse, "Token usage deletion"),
        );
      }

      const checkpointResponse = await fetch(
        `/api/langgraph/workflow/threads/${targetThreadId}/local-checkpoints`,
        { method: "DELETE" },
      );
      if (!checkpointResponse.ok) {
        throw new Error(
          await httpErrorMessage(
            checkpointResponse,
            "Local checkpoint deletion",
          ),
        );
      }
      const checkpointDeletion =
        (await checkpointResponse.json()) as LocalCheckpointDeletionResponse;
      if (
        checkpointDeletion.backend === "in_memory" &&
        !checkpointDeletion.purged
      ) {
        throw new Error("Local checkpoint deletion was not confirmed");
      }

      await stream.client.threads.delete(targetThreadId);
      const confirmationResponse = await fetch(
        `/api/langgraph/workflow/threads/${targetThreadId}/deletion-confirmation`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            agent_server_run_ids: agentServerRunIds,
          }),
        },
      );
      if (!confirmationResponse.ok) {
        throw new Error(
          await httpErrorMessage(
            confirmationResponse,
            "Thread deletion confirmation",
          ),
        );
      }
      const confirmation =
        (await confirmationResponse.json()) as ThreadDeletionConfirmationResponse;
      if (!confirmation.confirmed) {
        throw new Error("Thread deletion was not durably confirmed");
      }
      void refreshUsage();
      setThreads((items) =>
        items.filter((item) => item.thread_id !== targetThreadId),
      );
      if (targetThreadId === currentThreadId.current) {
        clearReconciledInterrupt();
        currentThreadId.current = null;
        setThreadId(null);
        setInstruction("");
        setCancelledThreadId(null);
        removeBrowserSetting(ACTIVE_THREAD_KEY);
        transitionWorkspace(false);
      }
    } catch (error) {
      setLocalError(errorMessage(error));
    } finally {
      setDeletingThreadId(null);
    }
  }

  const visibleError = localError || (
    !stopping &&
    deletingThreadId !== displayedThreadId &&
    displayedState.workflow_status !== "cancelled" &&
    stream.error
      ? errorMessage(stream.error)
      : null
  );
  const sidebarThreads = draftThread
    ? [draftThread, ...threads]
    : threads;
  const activeSidebarThreadId =
    displayedThreadId ?? draftThread?.thread_id ?? null;
  const displayedInstruction =
    workspaceActive &&
    typeof displayedState.user_instruction === "string" &&
    displayedState.user_instruction.trim()
      ? displayedState.user_instruction
      : instruction;

  return (
    <main className="app-shell">
      <div
        className={`workspace-grid ${workspaceActive ? "workspace-grid--active" : "workspace-grid--welcome"} ${workspaceTransitioning ? "workspace-grid--transitioning" : ""} ${sidebarCollapsed ? "workspace-grid--sidebar-collapsed" : ""} ${workspaceActive && artifactsCollapsed ? "workspace-grid--artifacts-collapsed" : ""}`}
      >
        <ThreadSidebar
          activeThreadId={activeSidebarThreadId}
          threads={sidebarThreads}
          loading={threadsLoading}
          disabled={workflowCanStop}
          deletingThreadId={deletingThreadId}
          collapsed={sidebarCollapsed}
          usage={currentWorkflowUsage}
          onSelect={selectThread}
          onDelete={(targetThreadId) => void deleteThread(targetThreadId)}
          onNew={createNewThread}
          onRefresh={() => {
            setThreadsLoading(true);
            setThreadRefresh((value) => value + 1);
          }}
          onCollapsedChange={setSidebarCollapsed}
        />

        <div
          className={`main-column ${workspaceActive ? "main-column--active" : "main-column--welcome"}`}
        >
          <InstructionComposer
            mode={workspaceActive ? "active" : "welcome"}
            value={displayedInstruction}
            disabled={stream.isThreadLoading}
            locked={workspaceActive}
            isLoading={workflowCanStop}
            isStopping={stopping}
            servicesReady={health?.ready ?? null}
            llmTestStatus={llmTest.status}
            onChange={setInstruction}
            onSubmit={() => void submitInstruction()}
            onStop={() => void stopRun()}
          />

          {workspaceActive ? (
            <div className="workflow-content workspace-enter">
              {visibleError ? (
                <div className="error-banner" role="alert">
                  <AlertCircle size={17} />
                  <span>{visibleError}</span>
                  <button type="button" onClick={() => setLocalError(null)} aria-label="Dismiss error">
                    <X size={14} />
                  </button>
                </div>
              ) : null}

              {approval ? (
                <ApprovalCard
                  approval={approval}
                  responding={responding}
                  onDecision={(decision) => void respondToApproval(decision)}
                />
              ) : null}

              {manualViewSelection ? (
                <ManualViewSelectionDialog
                  key={manualViewSelection.intervention_id}
                  request={manualViewSelection}
                  responding={responding}
                  onDecision={(decision) =>
                    void respondToManualViewSelection(decision)
                  }
                />
              ) : null}

              {manualMask ? (
                <ManualMaskDialog
                  key={`${manualMask.intervention_id}:${manualMask.stage}:${manualMask.revision}`}
                  request={manualMask}
                  responding={responding}
                  onDecision={(decision) =>
                    void respondToManualMask(decision)
                  }
                />
              ) : null}

              <WorkflowOverview state={displayedState} />
              <div className="workflow-panels">
                <WorkflowTimeline events={events} />
                <SubtaskPanel state={displayedState} />
              </div>
            </div>
          ) : visibleError ? (
            <div className="error-banner welcome-error" role="alert">
              <AlertCircle size={17} />
              <span>{visibleError}</span>
              <button type="button" onClick={() => setLocalError(null)} aria-label="Dismiss error">
                <X size={14} />
              </button>
            </div>
          ) : null}
        </div>

        {workspaceActive ? (
          <ArtifactPanel
            state={displayedState}
            collapsed={artifactsCollapsed}
            onCollapsedChange={setArtifactsCollapsed}
          />
        ) : null}
      </div>
      <PreferencesMenu
        compact={sidebarCollapsed}
        llmSettings={llmSettings}
        llmPresets={llmPresets}
        llmTest={llmTest}
        llmSettingsLoading={llmSettingsLoading}
        llmTesting={llmTesting}
        serviceSettings={serviceSettings}
        serviceHealth={health}
        serviceHealthError={healthError}
        serviceSettingsLoading={serviceSettingsLoading}
        serviceSettingsSaving={serviceSettingsSaving}
        serviceSettingsDisabled={workflowCanStop}
        usage={usage}
        usageLoading={usageLoading}
        usageError={usageError}
        onLlmSettingsChange={updateLlmSettings}
        onTestLlm={() => void testLlmSettings()}
        onServiceSettingsChange={updateServiceSettings}
        onRefreshUsage={() => void refreshUsage()}
      />
    </main>
  );
}
