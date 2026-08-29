import type {
  WorkflowEvent,
  WorkflowEventPayload,
  WorkflowEventStatus,
} from "@/lib/workflow-types";

const SERVICE_LABELS: Record<string, string> = {
  blender_rpc: "Blender RPC",
  sam3: "SAM3",
};

export interface WorkflowTimelineItem {
  id: string;
  event: WorkflowEvent;
  message: string;
  sourceCount: number;
}

function eventScope(event: WorkflowEvent): string {
  return [
    event.node,
    event.subtask_index ?? "workflow",
    event.attempt ?? "none",
  ].join(":");
}

function nodePhase(event: WorkflowEvent): "started" | "completed" | null {
  return event.payload.kind === "node.status" ? event.payload.phase : null;
}

function isSemanticEvent(event: WorkflowEvent): boolean {
  return [
    "workflow.status",
    "subtask.status",
    "approval.status",
    "intervention.status",
    "effect.visibility",
    "error.raised",
  ].includes(event.type);
}

function progressKey(event: WorkflowEvent): string | null {
  if (event.payload.kind !== "progress.update") return null;
  return [
    eventScope(event),
    event.payload.label,
    event.payload.unit,
  ].join(":");
}

function progressMessage(
  payload: Extract<WorkflowEventPayload, { kind: "progress.update" }>,
): string {
  if (payload.current === payload.total) {
    if (payload.unit === "view") {
      if (payload.label === "SAM3 view prefilter") {
        return `Evaluated ${payload.total} standard views with SAM3`;
      }
      return `Captured ${payload.total} standard views`;
    }
    if (payload.unit === "stroke") {
      return `Completed ${payload.total} sculpt ${payload.total === 1 ? "stroke" : "strokes"}`;
    }
    return `${payload.label} completed`;
  }
  return `${payload.label} (${payload.current}/${payload.total})`;
}

function serviceSummary(events: WorkflowEvent[]): WorkflowTimelineItem {
  const latestByService = new Map<string, WorkflowEvent>();
  for (const event of events) {
    if (event.payload.kind === "service.status") {
      latestByService.set(event.payload.service, event);
    }
  }
  const latest = events.at(-1)!;
  const failed = [...latestByService.values()].filter(
    (event) =>
      event.payload.kind === "service.status" && !event.payload.ready,
  );
  const labels = [...latestByService.keys()].map(
    (service) => SERVICE_LABELS[service] ?? service,
  );
  const status: WorkflowEventStatus = failed.length ? "error" : "success";
  const message = failed.length
    ? `${labels.join(" and ")} service check failed`
    : `${labels.join(" and ")} ${labels.length === 1 ? "is" : "are"} ready`;

  return {
    id: `services:${eventScope(latest)}`,
    event: { ...latest, status },
    message,
    sourceCount: events.length,
  };
}

/**
 * Convert the canonical event stream into a compact, chronological UI model.
 * The transport schema remains unchanged; only repetitive presentation events
 * are collapsed here.
 */
export function buildWorkflowTimeline(
  events: WorkflowEvent[],
): WorkflowTimelineItem[] {
  const ordered = [...events].sort((left, right) =>
    left.sequence === right.sequence
      ? left.timestamp.localeCompare(right.timestamp)
      : left.sequence - right.sequence,
  );
  const completedNodeScopes = new Set(
    ordered
      .filter((event) => nodePhase(event) === "completed")
      .map(eventScope),
  );
  const semanticScopes = new Set(
    ordered.filter(isSemanticEvent).map(eventScope),
  );
  const progressGroups = new Map<string, WorkflowEvent[]>();
  const serviceGroups = new Map<string, WorkflowEvent[]>();

  for (const event of ordered) {
    const key = progressKey(event);
    if (key) {
      progressGroups.set(key, [...(progressGroups.get(key) ?? []), event]);
    }
    if (event.payload.kind === "service.status") {
      const scope = eventScope(event);
      serviceGroups.set(scope, [...(serviceGroups.get(scope) ?? []), event]);
    }
  }

  const items: WorkflowTimelineItem[] = [];
  for (const event of ordered) {
    const scope = eventScope(event);
    const phase = nodePhase(event);

    // Artifacts and accounting have dedicated panels.
    if (["artifact.created", "usage.updated"].includes(event.type)) continue;

    // Once a node has a result, omit its transient "started" row. Nodes with a
    // semantic result (approval, grading, execution, and so on) use that row.
    if (
      phase === "started" &&
      (completedNodeScopes.has(scope) || semanticScopes.has(scope))
    ) {
      continue;
    }
    if (phase === "completed" && semanticScopes.has(scope)) continue;

    if (event.payload.kind === "service.status") {
      const group = serviceGroups.get(scope) ?? [event];
      if (group.at(-1)?.event_id === event.event_id) {
        items.push(serviceSummary(group));
      }
      continue;
    }

    if (event.payload.kind === "progress.update") {
      const key = progressKey(event)!;
      const group = progressGroups.get(key) ?? [event];
      if (group.at(-1)?.event_id === event.event_id) {
        items.push({
          id: `progress:${key}`,
          event,
          message: progressMessage(event.payload),
          sourceCount: group.length,
        });
      }
      continue;
    }

    items.push({
      id: event.event_id,
      event,
      message: event.message,
      sourceCount: 1,
    });
  }

  return items;
}
