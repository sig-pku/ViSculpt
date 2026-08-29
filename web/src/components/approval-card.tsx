import { Check, MousePointer2, ShieldAlert, X } from "lucide-react";

import type { SculptExecutionApproval } from "@/lib/workflow-types";

type Props = {
  approval: SculptExecutionApproval;
  responding: boolean;
  onDecision: (decision: "approve" | "reject") => void;
};

export function ApprovalCard({ approval, responding, onDecision }: Props) {
  const intent = approval.intent;
  const settings = approval.resolved_sculpt_plan.settings;
  return (
    <section className="approval-card" aria-live="polite">
      <div className="approval-card__icon">
        <ShieldAlert size={19} />
      </div>
      <div className="approval-card__content">
        <div className="approval-card__heading">
          <div>
            <span className="section-kicker section-kicker--amber">USER APPROVAL</span>
            <h2>Review the Sculpt operation</h2>
          </div>
          <span className="approval-card__attempt">
            Attempt {approval.attempt} · {approval.selected_view}
          </span>
        </div>
        <p>{approval.subtask.description}</p>
        <div className="approval-parameters">
          <span>
            <MousePointer2 size={13} /> Operation: {intent.operation_location}
          </span>
          <span>Changed part: {intent.part_to_be_changed}</span>
          <span>{settings?.sculpt_brush ?? intent.sculpt_brush} brush</span>
          <span>{settings?.brush_size ?? "—"}px</span>
          <span>{settings?.brush_strength ?? intent.brush_strength} strength</span>
          <span>{settings?.brush_direction ?? intent.brush_direction}</span>
        </div>
      </div>
      <div className="approval-card__actions">
        <button
          className="reject-button"
          type="button"
          onClick={() => onDecision("reject")}
          disabled={responding}
        >
          <X size={15} /> Reject
        </button>
        <button
          className="approve-button"
          type="button"
          onClick={() => onDecision("approve")}
          disabled={responding}
        >
          <Check size={15} /> {responding ? "Responding…" : "Approve stroke"}
        </button>
      </div>
    </section>
  );
}
