import type {
  ModelTokenUsage,
  TokenUsageAggregate,
  WorkflowTokenUsageSummary,
} from "@/lib/token-usage";

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue =
  | JsonPrimitive
  | JsonValue[]
  | { [key: string]: JsonValue };

export type WorkflowStatus =
  | "created"
  | "ready"
  | "planning"
  | "executing"
  | "finishing"
  | "completed"
  | "partially_completed"
  | "failed"
  | "rejected"
  | "cancelled";

export type WorkflowEventStatus =
  | "created"
  | "running"
  | "waiting"
  | "success"
  | "error"
  | "skipped"
  | "retrying"
  | "passed"
  | "rejected"
  | "completed";

export type WorkflowEventType =
  | "workflow.status"
  | "node.status"
  | "service.status"
  | "progress.update"
  | "artifact.created"
  | "subtask.status"
  | "approval.status"
  | "intervention.status"
  | "effect.visibility"
  | "usage.updated"
  | "error.raised";

export type WorkflowEventPayload =
  | {
      kind: "workflow.status";
      workflow_status: string;
    }
  | {
      kind: "node.status";
      phase: "started" | "completed";
    }
  | {
      kind: "service.status";
      service: "blender_rpc" | "sam3";
      ready: boolean;
    }
  | {
      kind: "progress.update";
      label: string;
      current: number;
      total: number;
      unit: "view" | "stroke" | "step" | "artifact";
    }
  | {
      kind: "artifact.created";
      artifact_ids: string[];
    }
  | {
      kind: "subtask.status";
      action: "prepared" | "executed" | "graded" | "advanced";
      operation_method: "Smear" | "Drag" | "Draw" | null;
    }
  | {
      kind: "approval.status";
      approval_id: string;
      decision: "pending" | "approve" | "reject";
    }
  | {
      kind: "intervention.status";
      intervention_id: string;
      intervention_type: "manual_view_selection" | "manual_mask";
      decision:
        | "pending"
        | "selected"
        | "painted"
        | "redraw"
        | "confirmed"
        | "skipped";
      stage: "paint" | "review" | null;
      selected_view: string | null;
    }
  | {
      kind: "effect.visibility";
      verdict: "NO_EFFECT" | "TOO_SUBTLE" | "VISIBLE" | "INCONCLUSIVE";
      target_mean_abs_diff: number;
      target_changed_fraction: number;
    }
  | {
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
  | {
      kind: "error.raised";
      code: string;
      retryable: boolean;
    };

export interface WorkflowEvent {
  schema_version: "2.0";
  event_id: string;
  sequence: number;
  timestamp: string;
  run_id: string;
  type: WorkflowEventType;
  status: WorkflowEventStatus;
  node: string;
  message: string;
  subtask_index: number | null;
  attempt: number | null;
  payload: WorkflowEventPayload;
}

export type WorkflowArtifactKind =
  | "screenshot"
  | "mask"
  | "overlay"
  | "trajectory"
  | "blender_state"
  | "workflow_state"
  | "metadata"
  | "other";

export interface WorkflowArtifact {
  schema_version: "1.0";
  artifact_id: string;
  run_id: string;
  kind: WorkflowArtifactKind;
  label: string;
  relative_path: string;
  uri: string;
  media_type: string;
  size_bytes: number;
  created_at: string;
  metadata: Record<string, JsonValue>;
}

export interface SculptIntent {
  operation_location?: string;
  part_to_be_changed?: string;
  sculpt_brush?: string;
  brush_scale?: "LOCAL" | "REGIONAL" | "BROAD";
  brush_strength?: number;
  brush_direction?: string | null;
  draw_pattern_description?: string | null;
  draw_text?: string | null;
  draw_scale_tier?: "SMALL" | "MEDIUM" | "LARGE" | null;
  effect_intensity?: "SUBTLE" | "MEDIUM_VISIBLE" | "STRONG";
}

export interface ResolvedSculptSettings {
  sculpt_brush?: string;
  brush_size?: number;
  brush_strength?: number;
  brush_direction?: string | null;
  dyntopo_enabled?: boolean;
  dyntopo_detail_size?: number;
  use_unified_size?: boolean;
  use_unified_strength?: boolean;
  use_size_pressure?: boolean;
  use_strength_pressure?: boolean;
}

export interface ResolvedSculptPlan {
  settings?: ResolvedSculptSettings;
  stroke_policy?: {
    pass_count?: number;
    target_coverage?: number;
    dose_multiplier?: number;
  };
  resolution_context?: Record<string, JsonValue>;
}

export interface WorkflowSubtask {
  description?: string;
  operation_method?: "Smear" | "Drag" | "Draw";
}

export interface WorkflowTranslation {
  subtask_index?: number;
  intent?: SculptIntent;
}

export interface ViewSegmentationAssessment {
  valid?: boolean;
  invalid_reason?: string | null;
  instance_count?: number;
  max_confidence?: number | null;
  foreground_pixels?: number;
  foreground_ratio?: number;
  mask_width?: number;
  mask_height?: number;
}

export interface ViewSegmentationResult {
  view?: string;
  segmentation_prompt?: string;
  screenshot_path?: string | null;
  assessment?: ViewSegmentationAssessment;
  segmentation?: Record<string, JsonValue>;
}

export interface GraderResult {
  effect_appropriateness?:
    | "TOO_WEAK"
    | "APPROPRIATE"
    | "EXCESSIVE_FOR_INSTRUCTION"
    | "WRONG_EFFECT"
    | "WRONG_REGION"
    | "INCONCLUSIVE";
  effect_magnitude?: "SUBTLE" | "MODERATE" | "LARGE" | "DRAMATIC";
  visual_evidence?: string[];
  instruction_compliance?: number;
  visual_quality?: number;
  geometric_plausibility?: number;
  total_score?: number;
  score_qualifies?: boolean;
  analysis?: string;
}

export interface MinimumEffectResult {
  verdict?: "NO_EFFECT" | "TOO_SUBTLE" | "VISIBLE" | "INCONCLUSIVE";
  metrics?: {
    noise_threshold?: number;
    target_mean_abs_diff?: number;
    target_changed_fraction?: number;
    evaluation_pixel_count?: number;
  };
  requirements?: {
    minimum_mean_abs_diff?: number;
    minimum_changed_fraction?: number;
  };
  reason_codes?: string[];
  artifact_paths?: Record<string, string>;
}

export interface AcceptanceDecision {
  accepted?: boolean;
  action?: "advance" | "retry" | "restore";
  reason_codes?: string[];
}

export interface WorkflowState {
  run_id?: string;
  user_instruction?: string;
  messages?: unknown[];
  workflow_status?: WorkflowStatus | string;
  llm_config?: Record<string, JsonValue>;
  llm_config_overridden?: boolean;
  token_usage?: WorkflowTokenUsageSummary | Record<string, never>;
  event_schema_version?: string;
  event_sequence?: number;
  service_status?: Record<string, JsonValue>;
  blender_session_lease?: Record<string, JsonValue> | null;
  sculpt_viewport_ui_snapshot?: Record<string, JsonValue> | null;
  sculpt_viewport_ui_restore_response?: JsonValue;
  sculpt_brush_capabilities?: {
    blender_version?: string;
    blender_version_tuple?: number[];
    brush_count?: number;
    brushes?: Array<{
      name?: string;
      direction_values?: string[];
    }>;
    inventory_scan?: Record<string, JsonValue>;
  } | null;
  artifacts?: WorkflowArtifact[];
  subtasks?: WorkflowSubtask[];
  translations?: WorkflowTranslation[];
  current_subtask_index?: number;
  current_attempt?: number;
  view_segmentation_results?: Record<string, ViewSegmentationResult>;
  valid_views?: string[];
  rejected_drag_anchor_views?: string[];
  selected_view?: string | null;
  view_selection_reason?: string | null;
  view_selection_intervention?: Record<string, JsonValue> | null;
  manual_mask_request?: Record<string, JsonValue> | null;
  manual_mask_intervention?: Record<string, JsonValue> | null;
  manual_mask_result?: Record<string, JsonValue> | null;
  manual_mask_overrides?: Record<string, JsonValue>;
  execution_preparation?: Record<string, JsonValue> | null;
  segmentation_result?: Record<string, JsonValue> | null;
  draw_pattern_result?: Record<string, JsonValue> | null;
  draw_trajectory_result?: Record<string, JsonValue> | null;
  draw_fitted_trajectory_result?: Record<string, JsonValue> | null;
  resolved_sculpt_plan?: ResolvedSculptPlan | null;
  stroke_plan_result?: Record<string, JsonValue> | null;
  baseline_screenshot_a?: Record<string, JsonValue> | null;
  baseline_screenshot_b?: Record<string, JsonValue> | null;
  minimum_effect_result?: MinimumEffectResult | null;
  grader_result?: GraderResult | null;
  acceptance_decision?: AcceptanceDecision | null;
  retry_feedback?: Record<string, JsonValue> | null;
  execution_error?: string | null;
  execution_decision?: string | null;
  current_approval_id?: string | null;
  subtask_results?: Array<Record<string, JsonValue>>;
  attempt_history?: Array<Record<string, JsonValue>>;
  events?: WorkflowEvent[];
  [key: string]: unknown;
}

export interface SculptExecutionApproval {
  schema_version: "2.0";
  type: "sculpt.execution.approval";
  approval_id: string;
  run_id: string;
  subtask_index: number;
  attempt: number;
  question: string;
  selected_view: string;
  subtask: WorkflowSubtask;
  intent: SculptIntent;
  resolved_sculpt_plan: ResolvedSculptPlan;
  before_artifact?: WorkflowArtifact;
  allowed_decisions: Array<"approve" | "reject">;
}

export interface ManualViewSelectionOption {
  view: string;
  screenshot_artifact: WorkflowArtifact;
  assessment?: ViewSegmentationAssessment | null;
}

export interface ManualViewSelectionRequest {
  schema_version: "2.0";
  type: "sculpt.view_selection.required";
  intervention_id: string;
  run_id: string;
  subtask_index: number;
  attempt: number;
  question: string;
  subtask: WorkflowSubtask;
  views: ManualViewSelectionOption[];
  allowed_decisions: Array<"select" | "skip">;
}

export interface ManualMaskPoint {
  x: number;
  y: number;
}

export interface ManualMaskStroke {
  brush_size: number;
  points: ManualMaskPoint[];
}

interface ManualMaskRequestBase {
  schema_version: "2.0";
  type: "sculpt.manual_mask.required";
  intervention_id: string;
  run_id: string;
  subtask_index: number;
  attempt: number;
  call_site: string;
  part_description: string;
  source_artifact: WorkflowArtifact;
  revision: number;
}

export interface ManualMaskPaintRequest extends ManualMaskRequestBase {
  stage: "paint";
  image_width: number;
  image_height: number;
  brush: {
    minimum_size: number;
    maximum_size: number;
    default_size: number;
  };
  allowed_decisions: Array<"finish" | "skip">;
}

export interface ManualMaskReviewRequest extends ManualMaskRequestBase {
  stage: "review";
  mask_artifact: WorkflowArtifact;
  overlay_artifact: WorkflowArtifact;
  intersection_foreground_pixels: number;
  confirm_allowed: boolean;
  allowed_decisions: Array<"confirm" | "redraw" | "skip">;
}

export type ManualMaskRequest =
  | ManualMaskPaintRequest
  | ManualMaskReviewRequest;

export type ManualMaskDecision =
  | { decision: "skip" }
  | {
      decision: "finish";
      image_width: number;
      image_height: number;
      strokes: ManualMaskStroke[];
    }
  | { decision: "confirm" }
  | { decision: "redraw" };

export type WorkflowInterrupt =
  | SculptExecutionApproval
  | ManualViewSelectionRequest
  | ManualMaskRequest;

export interface ServiceHealth {
  ready: boolean;
  services: {
    blender_rpc?: { ready?: boolean; error?: string };
    sam3?: { ready?: boolean; error?: string };
  };
}

export const EMPTY_WORKFLOW_STATE: WorkflowState = {
  workflow_status: "created",
  artifacts: [],
  subtasks: [],
  translations: [],
  view_segmentation_results: {},
  valid_views: [],
  rejected_drag_anchor_views: [],
  subtask_results: [],
  attempt_history: [],
  events: [],
};

export function isWorkflowEvent(value: unknown): value is WorkflowEvent {
  if (!value || typeof value !== "object") return false;
  const event = value as Partial<WorkflowEvent>;
  return (
    event.schema_version === "2.0" &&
    typeof event.event_id === "string" &&
    typeof event.sequence === "number" &&
    typeof event.type === "string" &&
    typeof event.message === "string" &&
    Boolean(event.payload)
  );
}

export function isSculptExecutionApproval(
  value: unknown,
): value is SculptExecutionApproval {
  if (!value || typeof value !== "object") return false;
  const payload = value as Partial<SculptExecutionApproval>;
  return (
    payload.schema_version === "2.0" &&
    payload.type === "sculpt.execution.approval" &&
    typeof payload.approval_id === "string"
  );
}

export function isManualViewSelectionRequest(
  value: unknown,
): value is ManualViewSelectionRequest {
  if (!value || typeof value !== "object") return false;
  const payload = value as Partial<ManualViewSelectionRequest>;
  return (
    payload.schema_version === "2.0" &&
    payload.type === "sculpt.view_selection.required" &&
    typeof payload.intervention_id === "string" &&
    Array.isArray(payload.views) &&
    payload.views.length > 0 &&
    payload.views.every(
      (item) =>
        Boolean(item) &&
        typeof item === "object" &&
        typeof item.view === "string" &&
        Boolean(item.screenshot_artifact) &&
        typeof item.screenshot_artifact === "object" &&
        typeof item.screenshot_artifact.uri === "string",
    )
  );
}

export function isManualMaskRequest(
  value: unknown,
): value is ManualMaskRequest {
  if (!value || typeof value !== "object") return false;
  const payload = value as Partial<ManualMaskRequest>;
  if (
    payload.schema_version !== "2.0" ||
    payload.type !== "sculpt.manual_mask.required" ||
    typeof payload.intervention_id !== "string" ||
    (payload.stage !== "paint" && payload.stage !== "review") ||
    typeof payload.run_id !== "string" ||
    typeof payload.call_site !== "string" ||
    typeof payload.part_description !== "string" ||
    typeof payload.revision !== "number" ||
    !payload.source_artifact ||
    typeof payload.source_artifact.uri !== "string"
  ) {
    return false;
  }
  if (payload.stage === "paint") {
    const paint = payload as Partial<ManualMaskPaintRequest>;
    return (
      typeof paint.image_width === "number" &&
      paint.image_width > 0 &&
      typeof paint.image_height === "number" &&
      paint.image_height > 0 &&
      Boolean(paint.brush) &&
      typeof paint.brush?.minimum_size === "number" &&
      typeof paint.brush?.maximum_size === "number" &&
      typeof paint.brush?.default_size === "number" &&
      Array.isArray(paint.allowed_decisions)
    );
  }
  const review = payload as Partial<ManualMaskReviewRequest>;
  return (
    Boolean(review.mask_artifact) &&
    typeof review.mask_artifact?.uri === "string" &&
    Boolean(review.overlay_artifact) &&
    typeof review.overlay_artifact?.uri === "string" &&
    typeof review.intersection_foreground_pixels === "number" &&
    typeof review.confirm_allowed === "boolean" &&
    Array.isArray(review.allowed_decisions)
  );
}

export function extractWorkflowEvent(value: unknown): WorkflowEvent | null {
  if (isWorkflowEvent(value)) return value;
  if (!value || typeof value !== "object") return null;
  const event = value as {
    method?: unknown;
    params?: { data?: { name?: unknown; payload?: unknown } };
  };
  if (
    event.method === "custom" &&
    event.params?.data?.name === "workflow.event" &&
    isWorkflowEvent(event.params.data.payload)
  ) {
    return event.params.data.payload;
  }
  return null;
}

export function mergeWorkflowEvents(
  persisted: WorkflowEvent[] | undefined,
  transient: unknown[],
): WorkflowEvent[] {
  const byId = new Map<string, WorkflowEvent>();
  for (const event of persisted ?? []) byId.set(event.event_id, event);
  for (const value of transient) {
    const event = extractWorkflowEvent(value);
    if (event) byId.set(event.event_id, event);
  }
  return [...byId.values()].sort((left, right) =>
    left.sequence === right.sequence
      ? left.timestamp.localeCompare(right.timestamp)
      : left.sequence - right.sequence,
  );
}

export function artifactUrl(artifact: WorkflowArtifact): string {
  return `/api/langgraph${artifact.uri}`;
}
