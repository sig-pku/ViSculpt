import { ArrowUp, Square } from "lucide-react";

import type { LlmTestStatus } from "@/lib/llm-settings";

type Props = {
  mode: "welcome" | "active";
  value: string;
  disabled: boolean;
  locked: boolean;
  isLoading: boolean;
  isStopping: boolean;
  servicesReady: boolean | null;
  llmTestStatus: LlmTestStatus;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
};

export function InstructionComposer({
  mode,
  value,
  disabled,
  locked,
  isLoading,
  isStopping,
  servicesReady,
  llmTestStatus,
  onChange,
  onSubmit,
  onStop,
}: Props) {
  return (
    <section className={`composer-shell composer-shell--${mode}`}>
      <div className="composer-copy">
        <h1>Describe the geometric change.</h1>
      </div>
      <div className={`composer ${locked ? "composer--locked" : ""}`}>
        <textarea
          value={value}
          onChange={(event) => {
            if (!disabled && !locked) onChange(event.target.value);
          }}
          onKeyDown={(event) => {
            if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
              event.preventDefault();
              if (!disabled && !locked) onSubmit();
            }
          }}
          placeholder="e.g. Smooth the left arm while preserving the silhouette."
          aria-label="Sculpt instruction"
          rows={3}
          disabled={disabled}
          readOnly={locked}
        />
        <div className="composer__footer">
          <span>
            {locked
              ? isLoading
                ? "Workflow is running"
                : "Workflow finished · start a new task to run again"
              : servicesReady === null
              ? "Checking local services"
              : servicesReady === false
              ? "Local services are not ready"
              : llmTestStatus === "failure"
                ? "LLM configuration test failed"
                : llmTestStatus === "pending"
                  ? "Test the LLM configuration in Settings"
                  : (
                      <>
                        <span className="shortcut-hint shortcut-hint--command">
                          ⌘ Enter to run
                        </span>
                        <span className="shortcut-hint shortcut-hint--control">
                          Ctrl Enter to run
                        </span>
                      </>
                    )}
          </span>
          {isLoading ? (
            <button
              className="stop-button"
              type="button"
              onClick={onStop}
              disabled={isStopping}
            >
              <Square size={13} fill="currentColor" />
              {isStopping ? "Stopping…" : "Stop"}
            </button>
          ) : (
            <button
              className="submit-button"
              type="button"
              onClick={onSubmit}
              disabled={
                disabled ||
                locked ||
                !value.trim() ||
                servicesReady !== true ||
                llmTestStatus !== "success"
              }
            >
              Run workflow <ArrowUp size={15} />
            </button>
          )}
        </div>
      </div>
    </section>
  );
}
