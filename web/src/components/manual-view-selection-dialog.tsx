"use client";

import { Check, Eye, SkipForward } from "lucide-react";
import Image from "next/image";
import { useEffect, useState } from "react";

import { ModalPortal } from "@/components/modal-portal";
import {
  artifactUrl,
  type ManualViewSelectionRequest,
} from "@/lib/workflow-types";

type Props = {
  request: ManualViewSelectionRequest;
  responding: boolean;
  onDecision: (
    decision: { decision: "select"; view: string } | { decision: "skip" },
  ) => void;
};

export function ManualViewSelectionDialog({
  request,
  responding,
  onDecision,
}: Props) {
  const [selectedView, setSelectedView] = useState<string | null>(null);
  const description = request.subtask.description ?? "Current sculpt subtask";

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, []);

  return (
    <ModalPortal>
      <div className="manual-view-layer">
      <div className="manual-view-backdrop" aria-hidden="true" />
      <section
        className="manual-view-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="manual-view-title"
        aria-describedby="manual-view-description"
      >
        <header className="manual-view-dialog__header">
          <div className="manual-view-dialog__icon">
            <Eye size={19} />
          </div>
          <div>
            <h2 id="manual-view-title">Choose an operation view</h2>
            <p>
              SAM3 could not validate the target in any standard view. Select
              the view that best exposes the requested part, or skip this
              subtask.
            </p>
          </div>
        </header>

        <div className="manual-view-subtask">
          <span>Subtask {request.subtask_index + 1}</span>
          <p id="manual-view-description">{description}</p>
        </div>

        <div className="manual-view-grid" aria-label="Standard operation views">
          {request.views.map((option) => {
            const selected = option.view === selectedView;
            return (
              <button
                className={`manual-view-option ${selected ? "manual-view-option--selected" : ""}`}
                type="button"
                key={option.view}
                aria-pressed={selected}
                aria-label={`Select ${option.view} view`}
                onClick={() => setSelectedView(option.view)}
                disabled={responding}
              >
                <span className="manual-view-option__image">
                  <Image
                    src={artifactUrl(option.screenshot_artifact)}
                    alt={`${option.view} Blender view`}
                    fill
                    sizes="(max-width: 760px) 45vw, 260px"
                    unoptimized
                  />
                  {selected ? (
                    <span className="manual-view-option__check" aria-hidden="true">
                      <Check size={14} />
                    </span>
                  ) : null}
                </span>
                <span className="manual-view-option__label">{option.view}</span>
              </button>
            );
          })}
        </div>

        <footer className="manual-view-dialog__actions">
          <button
            className="manual-view-skip"
            type="button"
            onClick={() => onDecision({ decision: "skip" })}
            disabled={responding}
          >
            <SkipForward size={15} /> Skip subtask
          </button>
          <button
            className="manual-view-continue"
            type="button"
            onClick={() => {
              if (selectedView) {
                onDecision({ decision: "select", view: selectedView });
              }
            }}
            disabled={responding || selectedView === null}
          >
            <Check size={15} />
            {responding
              ? "Continuing…"
              : selectedView
                ? `Continue with ${selectedView}`
                : "Select a view to continue"}
          </button>
        </footer>
      </section>
      </div>
    </ModalPortal>
  );
}
