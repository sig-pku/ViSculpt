import type { Client, Config, ThreadState } from "@langchain/langgraph-sdk";

import type { JsonValue, WorkflowState } from "@/lib/workflow-types";

const RUN_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const HISTORY_PAGE_SIZE = 100;
const MAX_HISTORY_PAGES = 10_000;

type WorkflowThreadsClient = Client<WorkflowState>["threads"];
type WorkflowRunsClient = Client<WorkflowState>["runs"];
type ViewportSnapshot = Record<string, JsonValue>;

export type WorkflowDeletionPlan = {
  runIds: string[];
  viewportSnapshots: Record<string, ViewportSnapshot>;
  checkpointCount: number;
};

/** Collect every server run in a thread for persistent retry cleanup. */
export async function collectAgentServerRunIds(
  client: Pick<WorkflowRunsClient, "list">,
  threadId: string,
): Promise<string[]> {
  const runIds = new Set<string>();
  for (let pageIndex = 0; pageIndex < MAX_HISTORY_PAGES; pageIndex += 1) {
    const page = await client.list(threadId, {
      limit: HISTORY_PAGE_SIZE,
      offset: pageIndex * HISTORY_PAGE_SIZE,
    });
    for (const run of page) {
      if (!UUID_PATTERN.test(run.run_id)) {
        throw new Error(
          "Invalid Agent Server run ID prevents complete Workflow deletion",
        );
      }
      runIds.add(run.run_id);
    }
    if (page.length < HISTORY_PAGE_SIZE) return [...runIds];
  }
  throw new Error(
    `Agent Server run history exceeded ${MAX_HISTORY_PAGES * HISTORY_PAGE_SIZE} entries; deletion was stopped to avoid missing local data`,
  );
}

/** Collect local run-directory identifiers from the full thread history. */
export async function collectWorkflowDeletionPlan(
  client: Pick<WorkflowThreadsClient, "getState" | "getHistory">,
  threadId: string,
  seedValues?: WorkflowState | null,
): Promise<WorkflowDeletionPlan> {
  const runIds = new Set<string>();
  const viewportSnapshots: Record<string, ViewportSnapshot> = {};
  let checkpointCount = 0;

  const inspect = (values: WorkflowState) => {
    const primaryRunId = persistedRunId(values.run_id, "state.run_id");
    if (primaryRunId) runIds.add(primaryRunId);

    const workflowDirRunId = runIdFromWorkflowDirectory(
      values.workflow_dir,
    );
    if (workflowDirRunId) runIds.add(workflowDirRunId);

    for (const artifact of values.artifacts ?? []) {
      const artifactRunId = persistedRunId(
        artifact.run_id,
        "artifact.run_id",
      );
      if (artifactRunId) runIds.add(artifactRunId);
    }
    for (const event of values.events ?? []) {
      const eventRunId = persistedRunId(event.run_id, "event.run_id");
      if (eventRunId) runIds.add(eventRunId);
    }

    const snapshot = values.sculpt_viewport_ui_snapshot;
    if (
      primaryRunId &&
      snapshot &&
      typeof snapshot === "object" &&
      !Array.isArray(snapshot) &&
      !viewportSnapshots[primaryRunId]
    ) {
      viewportSnapshots[primaryRunId] = snapshot;
    }
  };

  let current: ThreadState<WorkflowState> | null = null;
  try {
    current = await client.getState<WorkflowState>(threadId);
  } catch {
    // Empty history also confirms a newly created thread without checkpoints.
  }
  if (current) inspect(current.values);
  if (seedValues) inspect(seedValues);

  let before: Config | undefined;
  const seenCursors = new Set<string>();
  for (let pageIndex = 0; pageIndex < MAX_HISTORY_PAGES; pageIndex += 1) {
    const page = await client.getHistory<WorkflowState>(threadId, {
      limit: HISTORY_PAGE_SIZE,
      before,
    });
    checkpointCount += page.length;
    page.forEach((state) => inspect(state.values));
    if (page.length < HISTORY_PAGE_SIZE) {
      return {
        runIds: [...runIds],
        viewportSnapshots,
        checkpointCount,
      };
    }
    before = historyCursor(page.at(-1), seenCursors);
  }

  throw new Error(
    `Workflow history exceeded ${MAX_HISTORY_PAGES * HISTORY_PAGE_SIZE} checkpoints; deletion was stopped to avoid missing local data`,
  );
}

function historyCursor(
  state: ThreadState<WorkflowState> | undefined,
  seenCursors: Set<string>,
): Config {
  const checkpoint = state?.checkpoint;
  const checkpointId = checkpoint?.checkpoint_id;
  if (!checkpoint || !checkpointId) {
    throw new Error(
      "Workflow history pagination did not return a checkpoint cursor",
    );
  }
  const cursorKey = [
    checkpoint.thread_id,
    checkpoint.checkpoint_ns,
    checkpointId,
  ].join(":");
  if (seenCursors.has(cursorKey)) {
    throw new Error(
      "Workflow history pagination repeated a checkpoint cursor",
    );
  }
  seenCursors.add(cursorKey);
  return { configurable: { ...checkpoint } };
}

function persistedRunId(value: unknown, source: string): string | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== "string" || !RUN_ID_PATTERN.test(value)) {
    throw new Error(`Invalid ${source} prevents complete Workflow deletion`);
  }
  return value;
}

function runIdFromWorkflowDirectory(value: unknown): string | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== "string") {
    throw new Error(
      "Invalid state.workflow_dir prevents complete Workflow deletion",
    );
  }
  const basename = value.split(/[\\/]/).at(-1) ?? "";
  if (!basename.startsWith("run-")) {
    throw new Error(
      "Invalid state.workflow_dir prevents complete Workflow deletion",
    );
  }
  return persistedRunId(
    basename.slice("run-".length),
    "state.workflow_dir",
  );
}
