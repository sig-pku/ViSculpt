export type ActiveWorkflowRunStatus = "pending" | "running";

type WorkflowRun = {
  run_id: string;
  status: string;
};

export type WorkflowRunsClient = {
  list: (
    threadId: string,
    options?: { limit?: number },
  ) => Promise<WorkflowRun[]>;
  cancel: (
    threadId: string,
    runId: string,
    wait?: boolean,
    action?: "interrupt" | "rollback",
  ) => Promise<void>;
};

type CancellationOptions = {
  maxRounds?: number;
  requiredEmptyRounds?: number;
  settleDelayMs?: number;
  sleep?: (delayMs: number) => Promise<void>;
};

export type WorkflowCancellationResult = {
  cancelledRunIds: string[];
};

const ACTIVE_STATUSES = new Set<ActiveWorkflowRunStatus>([
  "pending",
  "running",
]);

function defaultSleep(delayMs: number) {
  return new Promise<void>((resolve) => window.setTimeout(resolve, delayMs));
}

function activeRuns(runs: WorkflowRun[]) {
  return runs.filter((run) =>
    ACTIVE_STATUSES.has(run.status as ActiveWorkflowRunStatus),
  );
}

/** Cancel every server run in a thread, including new recovery runs. */
export async function cancelActiveWorkflowRuns(
  client: WorkflowRunsClient,
  threadId: string,
  options: CancellationOptions = {},
): Promise<WorkflowCancellationResult> {
  const maxRounds = options.maxRounds ?? 6;
  const requiredEmptyRounds = options.requiredEmptyRounds ?? 2;
  const settleDelayMs = options.settleDelayMs ?? 75;
  const sleep = options.sleep ?? defaultSleep;
  const cancelledRunIds = new Set<string>();
  let emptyRounds = 0;
  let lastCancellationError: unknown;

  for (let round = 0; round < maxRounds; round += 1) {
    const active = activeRuns(await client.list(threadId, { limit: 100 }));
    if (active.length === 0) {
      emptyRounds += 1;
      if (emptyRounds >= requiredEmptyRounds) {
        return { cancelledRunIds: [...cancelledRunIds] };
      }
    } else {
      emptyRounds = 0;
      for (const run of active) {
        try {
          await client.cancel(threadId, run.run_id, true, "interrupt");
          cancelledRunIds.add(run.run_id);
        } catch (error) {
          // A run may finish concurrently; trust the subsequent server query.
          lastCancellationError = error;
        }
      }
    }

    if (round + 1 < maxRounds) await sleep(settleDelayMs);
  }

  const remaining = activeRuns(await client.list(threadId, { limit: 100 }));
  if (remaining.length > 0) {
    const suffix = lastCancellationError instanceof Error
      ? `: ${lastCancellationError.message}`
      : "";
    throw new Error(
      `Agent Server still has active runs ${remaining
        .map((run) => run.run_id)
        .join(", ")}${suffix}`,
    );
  }
  return { cancelledRunIds: [...cancelledRunIds] };
}
