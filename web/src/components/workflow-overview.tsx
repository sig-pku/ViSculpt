import { Eye, Gauge, Layers3, RotateCcw } from "lucide-react";

import type { WorkflowState } from "@/lib/workflow-types";

type Props = {
  state: WorkflowState;
};

function currentIntent(state: WorkflowState) {
  const index = state.current_subtask_index ?? 0;
  return state.translations?.find((item) => item.subtask_index === index)?.intent;
}

function workflowStatusLabel(status: string | undefined) {
  const normalized = (status || "idle").replaceAll("_", " ").toLowerCase();
  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
}

export function WorkflowOverview({ state }: Props) {
  const intent = currentIntent(state);
  const settings = state.resolved_sculpt_plan?.settings;
  const total = state.subtasks?.length ?? 0;
  const current = Math.min((state.current_subtask_index ?? 0) + 1, total || 1);
  const grader = state.grader_result;
  const progress = total ? Math.min(100, ((state.current_subtask_index ?? 0) / total) * 100) : 0;
  const workflowStatus = state.workflow_status || "idle";

  return (
    <section className="overview-section">
      <div className="overview-heading">
        <h2>Workflow State</h2>
        <span className={`workflow-state workflow-state--${workflowStatus}`}>
          {workflowStatusLabel(workflowStatus)}
        </span>
      </div>

      <div className="progress-track" aria-label={`${Math.round(progress)}% complete`}>
        <span style={{ width: `${progress}%` }} />
      </div>

      <div className="metric-grid">
        <article className="metric-card">
          <Layers3 size={15} />
          <span>Subtask</span>
          <strong>{total ? `${current} / ${total}` : "—"}</strong>
        </article>
        <article className="metric-card">
          <RotateCcw size={15} />
          <span>Attempt</span>
          <strong>{state.current_attempt || "—"}</strong>
        </article>
        <article className="metric-card">
          <Eye size={15} />
          <span>Selected view</span>
          <strong>{state.selected_view || "—"}</strong>
        </article>
        <article className="metric-card">
          <Gauge size={15} />
          <span>Grader score</span>
          <strong>{grader?.total_score !== undefined ? `${grader.total_score} / 15` : "—"}</strong>
        </article>
      </div>

      <div className="parameter-strip">
        <div>
          <span>OPERATION LOCATION</span>
          <strong>{intent?.operation_location || "Not translated"}</strong>
        </div>
        <div>
          <span>PART TO BE CHANGED</span>
          <strong>{intent?.part_to_be_changed || "Not translated"}</strong>
        </div>
        <div>
          <span>BRUSH</span>
          <strong>{intent?.sculpt_brush || "—"}</strong>
        </div>
        <div>
          <span>RESOLVED SIZE / STRENGTH</span>
          <strong>
            {settings?.brush_size ? `${settings.brush_size}px` : intent?.brush_scale || "—"}
            {settings?.brush_strength !== undefined
              ? ` · ${settings.brush_strength}`
              : intent?.brush_strength !== undefined
                ? ` · intent ${intent.brush_strength}`
              : ""}
          </strong>
        </div>
        <div>
          <span>EFFECT GATE</span>
          <strong>{state.minimum_effect_result?.verdict || "Pending"}</strong>
        </div>
      </div>
    </section>
  );
}
