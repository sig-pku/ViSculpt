import type { Interrupt, ThreadState } from "@langchain/langgraph-sdk";

import {
  isManualMaskRequest,
  isManualViewSelectionRequest,
  isSculptExecutionApproval,
  type WorkflowInterrupt,
  type WorkflowState,
} from "@/lib/workflow-types";

export type PendingWorkflowInterrupt = {
  id?: string;
  value: WorkflowInterrupt;
  namespace?: string[];
};

function isWorkflowInterrupt(value: unknown): value is WorkflowInterrupt {
  return (
    isSculptExecutionApproval(value) ||
    isManualViewSelectionRequest(value) ||
    isManualMaskRequest(value)
  );
}

function normalizeInterrupt(
  interrupt: Interrupt<unknown>,
): PendingWorkflowInterrupt | null {
  if (!isWorkflowInterrupt(interrupt.value)) return null;

  const legacyNamespace = Array.isArray(interrupt.ns)
    ? interrupt.ns
    : undefined;
  const namespace = Array.isArray(interrupt.namespace)
    ? interrupt.namespace
    : legacyNamespace;

  return {
    ...(typeof interrupt.id === "string" && interrupt.id
      ? { id: interrupt.id }
      : {}),
    value: interrupt.value,
    ...(namespace ? { namespace: [...namespace] } : {}),
  };
}

export function findWorkflowInterrupt(
  interrupts: readonly Interrupt<unknown>[],
): PendingWorkflowInterrupt | null {
  for (const interrupt of interrupts) {
    const normalized = normalizeInterrupt(interrupt);
    if (normalized) return normalized;
  }
  return null;
}

export function findWorkflowInterruptInState(
  snapshot: ThreadState<WorkflowState>,
): PendingWorkflowInterrupt | null {
  const pendingStates: Array<ThreadState<WorkflowState>> = [snapshot];
  const visited = new Set<ThreadState<WorkflowState>>();

  while (pendingStates.length > 0) {
    const current = pendingStates.shift();
    if (!current || visited.has(current)) continue;
    visited.add(current);

    for (const task of current.tasks) {
      const interrupt = findWorkflowInterrupt(task.interrupts);
      if (interrupt) return interrupt;
      if (task.state) {
        pendingStates.push(task.state as ThreadState<WorkflowState>);
      }
    }
  }

  return null;
}

export function interruptResponseTarget(
  interrupt: PendingWorkflowInterrupt | null,
): { interruptId?: string; namespace?: string[] } | undefined {
  if (!interrupt) return undefined;
  if (!interrupt.id && interrupt.namespace === undefined) return undefined;
  return {
    ...(interrupt.id ? { interruptId: interrupt.id } : {}),
    ...(interrupt.namespace !== undefined
      ? { namespace: [...interrupt.namespace] }
      : {}),
  };
}
