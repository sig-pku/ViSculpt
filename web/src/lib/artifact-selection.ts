import type {
  GraderResult,
  JsonValue,
  WorkflowArtifact,
  WorkflowState,
} from "@/lib/workflow-types";

export interface OperationEvidence {
  subtaskIndex: number | null;
  attempt: number | null;
  selectedView: string | null;
  operationMethod: "Smear" | "Drag" | "Draw" | null;
  fullBefore: WorkflowArtifact | null;
  fullAfter: WorkflowArtifact | null;
  roiBefore: WorkflowArtifact | null;
  roiAfter: WorkflowArtifact | null;
  segmentationOverlay: WorkflowArtifact | null;
  poseSubpartOverlays: WorkflowArtifact[];
  trajectories: WorkflowArtifact[];
  graderResult: GraderResult | null;
}

type ResultRecord = Record<string, JsonValue>;

function numericField(record: ResultRecord | undefined, key: string) {
  const value = record?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringField(record: ResultRecord | undefined, key: string) {
  const value = record?.[key];
  return typeof value === "string" && value.trim() ? value : null;
}

function resultForSubtask(
  state: WorkflowState,
  subtaskIndex: number,
): ResultRecord | undefined {
  return (state.subtask_results ?? [])
    .filter((value) => numericField(value, "subtask_index") === subtaskIndex)
    .at(-1);
}

/** Pick the operation that is active now, or the most recently finished one. */
export function selectDefaultArtifactSubtaskIndex(
  state: WorkflowState,
): number | null {
  const total = state.subtasks?.length ?? 0;
  if (total === 0) return null;
  const current = state.current_subtask_index;
  const latestResult = (state.subtask_results ?? []).at(-1);
  if (
    typeof current === "number" &&
    current >= 0 &&
    current < total &&
    ((state.current_attempt ?? 0) > 0 || latestResult === undefined)
  ) {
    return current;
  }
  const resultIndex = numericField(latestResult, "subtask_index");
  if (resultIndex !== null && resultIndex >= 0 && resultIndex < total) {
    return resultIndex;
  }
  if (typeof current === "number" && current >= 0 && current < total) {
    return current;
  }
  return total - 1;
}

function artifactAttempt(
  artifact: WorkflowArtifact,
  subtaskIndex: number,
) {
  const prefix = `subtasks/subtask-${String(subtaskIndex + 1).padStart(3, "0")}/attempt-`;
  if (!artifact.relative_path.startsWith(prefix)) return null;
  const remainder = artifact.relative_path.slice(prefix.length);
  const match = /^(\d+)\//.exec(remainder);
  return match ? Number(match[1]) : null;
}

function attemptForSubtask(
  state: WorkflowState,
  subtaskIndex: number,
  result: ResultRecord | undefined,
) {
  const currentAttempt = state.current_attempt ?? 0;
  if (state.current_subtask_index === subtaskIndex && currentAttempt > 0) {
    return currentAttempt;
  }
  const resultAttempt = numericField(result, "attempts");
  if (resultAttempt !== null && resultAttempt > 0) return resultAttempt;
  const attempts = (state.artifacts ?? [])
    .map((artifact) => artifactAttempt(artifact, subtaskIndex))
    .filter((value): value is number => value !== null);
  return attempts.length ? Math.max(...attempts) : null;
}

function lastMatching(
  artifacts: WorkflowArtifact[],
  predicate: (artifact: WorkflowArtifact) => boolean,
) {
  return artifacts.filter(predicate).at(-1) ?? null;
}

function operationArtifacts(
  state: WorkflowState,
  subtaskIndex: number,
  attempt: number | null,
) {
  if (attempt === null) return [];
  const scope = `subtasks/subtask-${String(subtaskIndex + 1).padStart(3, "0")}/attempt-${String(attempt).padStart(2, "0")}/`;
  return (state.artifacts ?? []).filter(
    (artifact) =>
      artifact.media_type.startsWith("image/") &&
      artifact.relative_path.startsWith(scope),
  );
}

function operationMethod(
  state: WorkflowState,
  subtaskIndex: number,
  result: ResultRecord | undefined,
): OperationEvidence["operationMethod"] {
  const candidate =
    state.subtasks?.[subtaskIndex]?.operation_method ??
    stringField(result, "operation_method");
  return candidate === "Smear" || candidate === "Drag" || candidate === "Draw"
    ? candidate
    : null;
}

function selectedView(
  state: WorkflowState,
  subtaskIndex: number,
  result: ResultRecord | undefined,
) {
  const resultView =
    stringField(result, "selected_view") ??
    stringField(result, "last_selected_view");
  if (
    state.current_subtask_index === subtaskIndex &&
    (state.current_attempt ?? 0) > 0
  ) {
    return state.selected_view ?? resultView;
  }
  return resultView;
}

function poseSubpartOverlays(
  state: WorkflowState,
  subtaskIndex: number,
  attempt: number | null,
) {
  const candidates = (state.artifacts ?? []).filter((artifact) => {
    const metadataIndex = artifact.metadata.subtask_index;
    return (
      artifact.kind === "overlay" &&
      artifact.metadata.visualization === "pose_subpart_mask_overlay" &&
      metadataIndex === subtaskIndex
    );
  });
  if (!candidates.length) return [];
  const exact =
    attempt === null
      ? []
      : candidates.filter(
          (artifact) => artifactAttempt(artifact, subtaskIndex) === attempt,
        );
  const selected = exact.length
    ? exact
    : (() => {
        const latestAttempt = Math.max(
          ...candidates.map(
            (artifact) => artifactAttempt(artifact, subtaskIndex) ?? 0,
          ),
        );
        return candidates.filter(
          (artifact) =>
            (artifactAttempt(artifact, subtaskIndex) ?? 0) === latestAttempt,
        );
      })();
  return selected.sort((left, right) => {
    const leftOrder = left.metadata.subpart_order;
    const rightOrder = right.metadata.subpart_order;
    return (
      (typeof leftOrder === "number" ? leftOrder : 0) -
      (typeof rightOrder === "number" ? rightOrder : 0)
    );
  });
}

function graderResult(
  state: WorkflowState,
  subtaskIndex: number,
  result: ResultRecord | undefined,
): GraderResult | null {
  if (
    state.current_subtask_index === subtaskIndex &&
    state.grader_result
  ) {
    return state.grader_result;
  }
  const value = result?.grader ?? result?.last_grader;
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as GraderResult)
    : null;
}

/** Select evidence for exactly one subtask and its latest relevant attempt. */
export function selectOperationEvidence(
  state: WorkflowState,
  requestedSubtaskIndex = selectDefaultArtifactSubtaskIndex(state),
): OperationEvidence {
  const total = state.subtasks?.length ?? 0;
  const subtaskIndex =
    requestedSubtaskIndex !== null &&
    requestedSubtaskIndex >= 0 &&
    requestedSubtaskIndex < total
      ? requestedSubtaskIndex
      : selectDefaultArtifactSubtaskIndex(state);
  if (subtaskIndex === null) {
    return {
      subtaskIndex: null,
      attempt: null,
      selectedView: null,
      operationMethod: null,
      fullBefore: null,
      fullAfter: null,
      roiBefore: null,
      roiAfter: null,
      segmentationOverlay: null,
      poseSubpartOverlays: [],
      trajectories: [],
      graderResult: null,
    };
  }

  const result = resultForSubtask(state, subtaskIndex);
  const attempt = attemptForSubtask(state, subtaskIndex, result);
  const artifacts = operationArtifacts(state, subtaskIndex, attempt);
  const method = operationMethod(state, subtaskIndex, result);

  const fullBefore =
    lastMatching(
      artifacts,
      (artifact) =>
        artifact.kind === "screenshot" && /\bfull before\b/i.test(artifact.label),
    ) ??
    lastMatching(
      artifacts,
      (artifact) =>
        artifact.kind === "screenshot" && /\bbaseline A\b/i.test(artifact.label),
    );
  const fullAfter =
    lastMatching(
      artifacts,
      (artifact) =>
        artifact.kind === "screenshot" && /\bfull after\b/i.test(artifact.label),
    ) ??
    lastMatching(
      artifacts,
      (artifact) =>
        artifact.kind === "screenshot" &&
        /\battempt\s+\d+\s+(?:focused\s+)?after\b/i.test(artifact.label),
    );
  const roiBefore = lastMatching(
    artifacts,
    (artifact) =>
      artifact.kind === "screenshot" &&
      /target-region before crop/i.test(artifact.label),
  );
  const roiAfter = lastMatching(
    artifacts,
    (artifact) =>
      artifact.kind === "screenshot" &&
      /target-region after crop/i.test(artifact.label),
  );
  const segmentationOverlay =
    lastMatching(
      artifacts,
      (artifact) =>
        artifact.kind === "overlay" &&
        /focused noise-cleaned sculpt mask overlay/i.test(artifact.label),
    ) ??
    lastMatching(
      artifacts,
      (artifact) =>
        artifact.kind === "overlay" &&
        /VLM-selected Sculpt mask overlay/i.test(artifact.label),
    ) ??
    lastMatching(
      artifacts,
      (artifact) =>
        artifact.kind === "overlay" &&
        /\/segment\//.test(artifact.relative_path) &&
        /Noise-cleaned (?:Sculpt|SAM3) mask overlay/i.test(artifact.label),
    );
  const trajectories = artifacts.filter((artifact) => {
    if (artifact.kind !== "trajectory") return false;
    if (method === "Drag") {
      return artifact.metadata.visualization === "drag_trajectory";
    }
    if (method === "Draw") {
      return artifact.metadata.visualization === "draw_trajectories";
    }
    return artifact.metadata.visualization === "mouse_trajectories";
  });

  return {
    subtaskIndex,
    attempt,
    selectedView: selectedView(state, subtaskIndex, result),
    operationMethod: method,
    fullBefore,
    fullAfter,
    roiBefore,
    roiAfter,
    segmentationOverlay,
    poseSubpartOverlays: poseSubpartOverlays(
      state,
      subtaskIndex,
      attempt,
    ),
    trajectories,
    graderResult: graderResult(state, subtaskIndex, result),
  };
}
