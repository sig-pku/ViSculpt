import {
  CheckCircle2,
  CircleDashed,
  Settings2,
  SkipForward,
  WandSparkles,
} from "lucide-react";

import type {
  ResolvedSculptPlan,
  WorkflowState,
} from "@/lib/workflow-types";

export function SubtaskPanel({ state }: { state: WorkflowState }) {
  const subtasks = state.subtasks ?? [];
  const translations = state.translations ?? [];
  const current = state.current_subtask_index ?? 0;
  const results = state.subtask_results ?? [];

  return (
    <section className="panel subtask-panel">
      <div className="panel-heading">
        <h2>Subtasks &amp; Sculpt Intent</h2>
      </div>
      <div className="subtask-list">
        {subtasks.length === 0 ? (
          <div className="panel-empty">
            <WandSparkles size={18} />
            <p>Decomposer and Translator outputs will appear here.</p>
          </div>
        ) : null}
        {subtasks.map((subtask, index) => {
          const intent = translations.find(
            (translation) => translation.subtask_index === index,
          )?.intent;
          const result = results.find((item) => item.subtask_index === index);
          const savedResolved = result?.resolved_sculpt_plan;
          const resolved = (
            index === current
              ? state.resolved_sculpt_plan
              : savedResolved && typeof savedResolved === "object" && !Array.isArray(savedResolved)
                ? savedResolved
                : null
          ) as ResolvedSculptPlan | null | undefined;
          const settings = resolved?.settings;
          const status = typeof result?.status === "string" ? result.status : null;
          const complete = Boolean(status);
          const skipped = status?.startsWith("skipped") || status === "rejected_by_user";
          return (
            <article
              className={`subtask-item ${index === current ? "subtask-item--active" : ""}`}
              key={`${index}-${subtask.description}`}
            >
              <div className="subtask-item__index">
                {skipped ? (
                  <SkipForward size={14} />
                ) : complete ? (
                  <CheckCircle2 size={14} />
                ) : (
                  <CircleDashed size={14} />
                )}
              </div>
              <div className="subtask-item__content">
                <div className="subtask-item__meta">
                  <span>#{index + 1}</span>
                  <span>{subtask.operation_method}</span>
                  {status ? <span>{status.replaceAll("_", " ")}</span> : null}
                </div>
                <p>{subtask.description}</p>
                {intent ? (
                  <div
                    className="translator-parameters"
                    aria-label={`Translator intent for subtask ${index + 1}`}
                  >
                    <div className="translator-parameters__heading">
                      <Settings2 size={12} />
                      <span>Translator intent</span>
                    </div>
                    <dl className="translator-parameters__grid">
                      <div>
                        <dt>Operation location</dt>
                        <dd>{intent.operation_location ?? "—"}</dd>
                      </div>
                      <div>
                        <dt>Part to be changed</dt>
                        <dd>{intent.part_to_be_changed ?? "—"}</dd>
                      </div>
                      <div>
                        <dt>Brush</dt>
                        <dd>{intent.sculpt_brush ?? "—"}</dd>
                      </div>
                      <div>
                        <dt>Scale</dt>
                        <dd>{intent.brush_scale ?? "—"}</dd>
                      </div>
                      <div>
                        <dt>Strength</dt>
                        <dd>
                          {intent.brush_strength !== undefined
                            ? intent.brush_strength
                            : "—"}
                        </dd>
                      </div>
                      <div>
                        <dt>Direction</dt>
                        <dd>{intent.brush_direction ?? "—"}</dd>
                      </div>
                      {intent.draw_pattern_description ? (
                        <div>
                          <dt>Pattern</dt>
                          <dd>{intent.draw_pattern_description}</dd>
                        </div>
                      ) : null}
                      {intent.draw_text ? (
                        <div>
                          <dt>Text</dt>
                          <dd>{intent.draw_text}</dd>
                        </div>
                      ) : null}
                      {intent.draw_scale_tier ? (
                        <div>
                          <dt>Draw scale</dt>
                          <dd>{intent.draw_scale_tier}</dd>
                        </div>
                      ) : null}
                      <div>
                        <dt>Effect</dt>
                        <dd>{intent.effect_intensity ?? "—"}</dd>
                      </div>
                    </dl>
                  </div>
                ) : (
                  <div className="translator-parameters--pending">
                    Translator intent pending
                  </div>
                )}
                {settings ? (
                  <div className="translator-parameters translator-parameters--resolved">
                    <div className="translator-parameters__heading">
                      <Settings2 size={12} />
                      <span>Resolved attempt settings</span>
                    </div>
                    <dl className="translator-parameters__grid">
                      <div>
                        <dt>Size</dt>
                        <dd>{settings.brush_size !== undefined ? `${settings.brush_size} px` : "—"}</dd>
                      </div>
                      <div><dt>Strength</dt><dd>{settings.brush_strength ?? "—"}</dd></div>
                      <div><dt>Passes</dt><dd>{resolved?.stroke_policy?.pass_count ?? 1}</dd></div>
                      <div><dt>Direction</dt><dd>{settings.brush_direction ?? "—"}</dd></div>
                      <div>
                        <dt>Dyntopo</dt>
                        <dd>
                          {settings.dyntopo_enabled === undefined
                            ? "—"
                            : settings.dyntopo_enabled
                              ? "Enabled"
                              : "Disabled"}
                        </dd>
                      </div>
                      <div>
                        <dt>Detail size</dt>
                        <dd>
                          {settings.dyntopo_detail_size !== undefined
                            ? `${settings.dyntopo_detail_size} px`
                            : "—"}
                        </dd>
                      </div>
                    </dl>
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
