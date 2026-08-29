"use client";

import {
  AlertTriangle,
  Check,
  ChevronDown,
  Circle,
  Clock3,
  LoaderCircle,
  RotateCcw,
} from "lucide-react";
import { useMemo, useState } from "react";

import type { WorkflowEvent } from "@/lib/workflow-types";
import { buildWorkflowTimeline } from "@/lib/workflow-timeline";

type Props = {
  events: WorkflowEvent[];
};

const NODE_LABELS: Record<string, string> = {
  initialize: "Initialize",
  initial_check: "Preflight",
  decomposer: "Decomposer",
  translator: "Translator",
  prepare_subtask: "Prepare subtask",
  save_checkpoint: "Save Blender state",
  view_selector: "View Selector",
  manual_view_selection: "Manual view selection",
  prepare_manual_mask: "Prepare manual mask",
  manual_mask_input: "Paint manual mask",
  manual_mask_review: "Review manual mask",
  prepare_execution_dispatch: "Resume preparation",
  prepare_drag_context: "Capture Drag context",
  prepare_drag_changed_part: "Changed part and Face Sets",
  prepare_drag_quadloc: "QuadLoc",
  prepare_execution: "Prepare execution",
  approve_execution: "Human approval",
  executor: "Executor",
  minimum_effect: "Minimum Effect Gate",
  grader: "Visual Grader",
  acceptance_gate: "Acceptance Gate",
  retry_planner: "Retry Planner",
  restore_checkpoint: "Restore Blender state",
  advance_subtask: "Advance subtask",
  finalize: "Finalize",
};

function EventIcon({ event }: { event: WorkflowEvent }) {
  if (event.status === "error") return <AlertTriangle size={14} />;
  if (event.status === "retrying") return <RotateCcw size={14} />;
  if (event.status === "waiting") return <Clock3 size={14} />;
  if (event.status === "running") return <LoaderCircle size={14} className="spin" />;
  if (["success", "passed", "completed"].includes(event.status)) return <Check size={14} />;
  return <Circle size={11} />;
}

export function WorkflowTimeline({ events }: Props) {
  const timelineItems = buildWorkflowTimeline(events);
  const [expanded, setExpanded] = useState(false);
  const currentItem = useMemo(() => {
    const active = timelineItems.filter((item) =>
      ["running", "waiting", "retrying"].includes(item.event.status),
    );
    return active.at(-1) ?? timelineItems.at(-1) ?? null;
  }, [timelineItems]);
  const visibleItems = expanded
    ? timelineItems
    : currentItem
      ? [currentItem]
      : [];
  const hiddenCount = Math.max(0, timelineItems.length - 1);

  return (
    <section className="panel timeline-panel">
      <div className="panel-heading">
        <h2>Execution Timeline</h2>
        {hiddenCount ? (
          <button
            className="timeline-toggle"
            type="button"
            aria-expanded={expanded}
            onClick={() => setExpanded((value) => !value)}
          >
            {expanded ? "Hide history" : `Show history · ${hiddenCount}`}
            <ChevronDown
              size={14}
              className={expanded ? "timeline-toggle__icon--expanded" : ""}
            />
          </button>
        ) : null}
      </div>
      <div
        className={`timeline-list ${expanded ? "timeline-list--expanded" : "timeline-list--current"}`}
        aria-live="polite"
      >
        {timelineItems.length === 0 ? (
          <div className="panel-empty">
            <Circle size={18} />
            <p>Node events will appear here as the workflow runs.</p>
          </div>
        ) : null}
        {visibleItems.map((item) => {
          const event = item.event;
          const progress = event.payload.kind === "progress.update" ? event.payload : null;
          return (
            <article
              className={`timeline-event timeline-event--${event.status}`}
              key={item.id}
            >
              <div className="timeline-event__icon"><EventIcon event={event} /></div>
              <div className="timeline-event__body">
                <div className="timeline-event__meta">
                  <strong>{NODE_LABELS[event.node] || event.node}</strong>
                  {event.subtask_index !== null ? (
                    <span>subtask {event.subtask_index + 1}</span>
                  ) : null}
                  {event.attempt ? <span>attempt {event.attempt}</span> : null}
                </div>
                <p>{item.message}</p>
                {progress ? (
                  <div className="event-progress">
                    <span style={{ width: `${(progress.current / progress.total) * 100}%` }} />
                  </div>
                ) : null}
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
