"use client";

import Image from "next/image";
import { useEffect, useRef, useState } from "react";
import {
  ImageIcon,
  Images,
  Layers2,
  PanelRightClose,
  PanelRightOpen,
  ScanSearch,
  SlidersHorizontal,
  X,
} from "lucide-react";

import {
  selectDefaultArtifactSubtaskIndex,
  selectOperationEvidence,
} from "@/lib/artifact-selection";
import {
  artifactUrl,
  type WorkflowArtifact,
  type WorkflowState,
} from "@/lib/workflow-types";

type Tab = "views" | "segmentation" | "result";

type Props = {
  state: WorkflowState;
  collapsed?: boolean;
  onCollapsedChange?: (collapsed: boolean) => void;
};

function formatViewName(view: string | null) {
  if (!view) return null;
  return view.charAt(0).toUpperCase() + view.slice(1).toLowerCase();
}

function ArtifactImageCard({
  artifact,
  eyebrow,
}: {
  artifact: WorkflowArtifact;
  eyebrow: string;
}) {
  return (
    <a
      className="artifact-card"
      href={artifactUrl(artifact)}
      target="_blank"
      rel="noreferrer"
    >
      <div className="artifact-card__image">
        <Image
          src={artifactUrl(artifact)}
          alt={artifact.label}
          width={960}
          height={720}
          sizes="(max-width: 1100px) 50vw, 340px"
          unoptimized
        />
      </div>
      <div className="artifact-card__caption">
        <span>{eyebrow}</span>
        <strong>{artifact.label}</strong>
      </div>
    </a>
  );
}

function BeforeAfterCompare({
  before,
  after,
  title,
}: {
  before: WorkflowArtifact;
  after: WorkflowArtifact;
  title: string;
}) {
  const [position, setPosition] = useState(50);
  return (
    <section className="compare-section">
      <div className="evidence-section-heading">
        <span>{title}</span>
        <small>Drag to compare</small>
      </div>
      <div className="compare-shell">
        <div className="compare-stage">
          <Image
            src={artifactUrl(before)}
            alt={before.label}
            width={1200}
            height={900}
            unoptimized
          />
          <div
            className="compare-stage__after"
            style={{ clipPath: `inset(0 0 0 ${position}%)` }}
          >
            <Image
              src={artifactUrl(after)}
              alt={after.label}
              width={1200}
              height={900}
              unoptimized
            />
          </div>
          <span
            className="compare-stage__divider"
            style={{ left: `${position}%` }}
          >
            <span />
          </span>
          <span className="compare-label compare-label--before">Before</span>
          <span className="compare-label compare-label--after">After</span>
        </div>
        <label className="compare-slider">
          <SlidersHorizontal size={14} />
          <input
            type="range"
            min="0"
            max="100"
            value={position}
            onChange={(event) => setPosition(Number(event.target.value))}
            aria-label={`${title} before and after comparison`}
          />
        </label>
      </div>
    </section>
  );
}

function EmptyArtifacts({ label }: { label: string }) {
  return (
    <div className="artifact-empty">
      <ImageIcon size={24} />
      <p>{label}</p>
      <span>Evidence appears as soon as the operation reaches this stage.</span>
    </div>
  );
}

function SubtaskSelector({
  state,
  value,
  onChange,
}: {
  state: WorkflowState;
  value: number | null;
  onChange: (subtaskIndex: number) => void;
}) {
  const subtasks = state.subtasks ?? [];
  return (
    <label className="artifact-subtask-selector">
      <span>Subtask</span>
      <select
        aria-label="Displayed subtask"
        value={value ?? ""}
        disabled={subtasks.length === 0}
        onChange={(event) => onChange(Number(event.target.value))}
      >
        {subtasks.length === 0 ? <option value="">None</option> : null}
        {subtasks.map((subtask, index) => (
          <option key={index} value={index}>
            Subtask {index + 1}: {subtask.operation_method ?? "Pending"}
          </option>
        ))}
      </select>
    </label>
  );
}

function EvidencePair({
  before,
  after,
  title,
}: {
  before: WorkflowArtifact | null;
  after: WorkflowArtifact | null;
  title: string;
}) {
  if (!before && !after) return null;
  return (
    <section className="evidence-section">
      <div className="evidence-section-heading">
        <span>{title}</span>
        <small>{before && after ? "Before · After" : "In progress"}</small>
      </div>
      <div className="artifact-grid">
        {before ? <ArtifactImageCard artifact={before} eyebrow="Before" /> : null}
        {after ? <ArtifactImageCard artifact={after} eyebrow="After" /> : null}
      </div>
    </section>
  );
}

export function ArtifactPanel({
  state,
  collapsed = false,
  onCollapsedChange = () => undefined,
}: Props) {
  const [tab, setTab] = useState<Tab>("views");
  const [flyoutOpen, setFlyoutOpen] = useState(false);
  const automaticSubtaskIndex = selectDefaultArtifactSubtaskIndex(state);
  const [selectedSubtaskIndex, setSelectedSubtaskIndex] = useState<number | null>(
    automaticSubtaskIndex,
  );
  const rootRef = useRef<HTMLElement>(null);
  const selectionPinned = useRef(false);
  const previousRunId = useRef(state.run_id);

  useEffect(() => {
    const runChanged = previousRunId.current !== state.run_id;
    previousRunId.current = state.run_id;
    const selectionInvalid =
      selectedSubtaskIndex !== null &&
      selectedSubtaskIndex >= (state.subtasks?.length ?? 0);
    if (runChanged || selectionInvalid) {
      selectionPinned.current = false;
      setSelectedSubtaskIndex(automaticSubtaskIndex);
      return;
    }
    if (!selectionPinned.current) {
      setSelectedSubtaskIndex(automaticSubtaskIndex);
    }
  }, [
    automaticSubtaskIndex,
    selectedSubtaskIndex,
    state.run_id,
    state.subtasks?.length,
  ]);

  useEffect(() => {
    if (!flyoutOpen) return;
    const closeFromPointer = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        setFlyoutOpen(false);
      }
    };
    const closeFromKeyboard = (event: KeyboardEvent) => {
      if (event.key === "Escape") setFlyoutOpen(false);
    };
    document.addEventListener("pointerdown", closeFromPointer);
    document.addEventListener("keydown", closeFromKeyboard);
    return () => {
      document.removeEventListener("pointerdown", closeFromPointer);
      document.removeEventListener("keydown", closeFromKeyboard);
    };
  }, [flyoutOpen]);

  const evidence = selectOperationEvidence(state, selectedSubtaskIndex);
  const isDrag = evidence.operationMethod === "Drag";
  const selectedView = formatViewName(evidence.selectedView);
  const segmentationOverlay = evidence.segmentationOverlay;
  const poseSubpartOverlays = evidence.poseSubpartOverlays;
  const trajectories = [...evidence.trajectories].sort((left, right) => {
    const rank = (artifact: WorkflowArtifact) =>
      artifact.metadata.background === "overlay" ? 0 : 1;
    return rank(left) - rank(right);
  });
  const viewCount = [
    evidence.fullBefore,
    evidence.fullAfter,
    evidence.roiBefore,
    evidence.roiAfter,
  ].filter(Boolean).length;

  function selectTab(nextTab: Tab) {
    const shouldClose = collapsed && flyoutOpen && tab === nextTab;
    setTab(nextTab);
    if (collapsed) setFlyoutOpen(!shouldClose);
  }

  const subtaskSelector = (
    <SubtaskSelector
      state={state}
      value={evidence.subtaskIndex}
      onChange={(subtaskIndex) => {
        selectionPinned.current = true;
        setSelectedSubtaskIndex(subtaskIndex);
      }}
    />
  );

  const tabs = (
    <div
      className={`artifact-tabs ${collapsed ? "artifact-tabs--rail" : ""}`}
      role="tablist"
      aria-label="Visual results"
    >
      <button
        type="button"
        role="tab"
        aria-selected={tab === "views" && (!collapsed || flyoutOpen)}
        className={tab === "views" && (!collapsed || flyoutOpen) ? "active" : ""}
        onClick={() => selectTab("views")}
      >
        <Images size={19} /> <span>Views</span>
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={tab === "segmentation" && (!collapsed || flyoutOpen)}
        className={tab === "segmentation" && (!collapsed || flyoutOpen) ? "active" : ""}
        onClick={() => selectTab("segmentation")}
      >
        <ScanSearch size={19} /> <span>Segment</span>
      </button>
      <button
        type="button"
        role="tab"
        aria-selected={tab === "result" && (!collapsed || flyoutOpen)}
        className={tab === "result" && (!collapsed || flyoutOpen) ? "active" : ""}
        onClick={() => selectTab("result")}
      >
        <Layers2 size={19} /> <span>Compare</span>
      </button>
    </div>
  );

  const artifactBody = (
    <div className="artifact-panel__body">
      {tab === "views" ? (
        viewCount ? (
          <div className="evidence-stack">
            <EvidencePair
              before={evidence.fullBefore}
              after={evidence.fullAfter}
              title={`Full view${selectedView ? ` · ${selectedView}` : ""}`}
            />
            <EvidencePair
              before={evidence.roiBefore}
              after={evidence.roiAfter}
              title="Target region"
            />
          </div>
        ) : (
          <EmptyArtifacts label="Selected-view evidence is not ready yet." />
        )
      ) : null}

      {tab === "segmentation" ? (
        segmentationOverlay || poseSubpartOverlays.length || trajectories.length ? (
          <div className="evidence-stack artifact-tab-enter">
            {segmentationOverlay ? (
              <section className="evidence-section">
                <div className="evidence-section-heading">
                  <span>Noise-cleaned mask overlay</span>
                  <small>{selectedView ?? "Selected view"}</small>
                </div>
                <div className="artifact-grid artifact-grid--single">
                  <ArtifactImageCard
                    artifact={segmentationOverlay}
                    eyebrow={selectedView ? `${selectedView} view` : "Selected view"}
                  />
                </div>
              </section>
            ) : null}
            {poseSubpartOverlays.length ? (
              <section className="evidence-section">
                <div className="evidence-section-heading">
                  <span>Pose subpart masks</span>
                  <small>{poseSubpartOverlays.length} subparts</small>
                </div>
                <div className="artifact-grid artifact-grid--single">
                  {poseSubpartOverlays.map((artifact) => (
                    <ArtifactImageCard
                      artifact={artifact}
                      eyebrow="Kinematic subpart"
                      key={artifact.artifact_id}
                    />
                  ))}
                </div>
              </section>
            ) : null}
            {trajectories.length ? (
              <section className="evidence-section">
                <div className="evidence-section-heading">
                  <span>{isDrag ? "Drag trajectory" : "Mouse trajectories"}</span>
                  <small>
                    {isDrag
                      ? "Start point · direction · end point"
                      : "One color per gesture"}
                  </small>
                </div>
                <div className="artifact-grid artifact-grid--single">
                  {trajectories.map((artifact) => (
                    <ArtifactImageCard
                      artifact={artifact}
                      eyebrow={isDrag ? "Drag gesture overlay" : "Mouse-down to mouse-up"}
                      key={artifact.artifact_id}
                    />
                  ))}
                </div>
              </section>
            ) : null}
          </div>
        ) : (
          <EmptyArtifacts label="The selected view has no SAM3 result yet." />
        )
      ) : null}

      {tab === "result" ? (
        evidence.fullBefore && evidence.fullAfter ? (
          <div className="evidence-stack artifact-tab-enter">
            <BeforeAfterCompare
              before={evidence.fullBefore}
              after={evidence.fullAfter}
              title={`Full view${selectedView ? ` · ${selectedView}` : ""}`}
            />
            {evidence.roiBefore && evidence.roiAfter ? (
              <BeforeAfterCompare
                before={evidence.roiBefore}
                after={evidence.roiAfter}
                title="Target region"
              />
            ) : null}
            {evidence.graderResult ? (
              <div className="grader-card">
                <div className="grader-card__scores">
                  <span>
                    <small>Instruction</small>
                    <strong>{evidence.graderResult.instruction_compliance ?? "—"}</strong>
                  </span>
                  <span>
                    <small>Visual</small>
                    <strong>{evidence.graderResult.visual_quality ?? "—"}</strong>
                  </span>
                  <span>
                    <small>Geometry</small>
                    <strong>{evidence.graderResult.geometric_plausibility ?? "—"}</strong>
                  </span>
                </div>
                <p>{evidence.graderResult.analysis}</p>
              </div>
            ) : null}
          </div>
        ) : (
          <EmptyArtifacts label="Full before/after evidence is not complete." />
        )
      ) : null}
    </div>
  );

  if (collapsed) {
    return (
      <aside
        className="artifact-panel artifact-panel--collapsed workspace-enter"
        ref={rootRef}
      >
        <button
          className="artifact-rail-button"
          type="button"
          aria-label="Open Visual Results"
          title="Open Visual Results"
          onClick={() => {
            setFlyoutOpen(false);
            onCollapsedChange(false);
          }}
        >
          <PanelRightOpen size={19} />
        </button>
        {tabs}
        {flyoutOpen ? (
          <section
            className="artifact-flyout panel-flyout-enter"
            aria-label={`${tab === "views" ? "Views" : tab === "segmentation" ? "Segment" : "Compare"} results`}
          >
            <div className="artifact-panel__heading">
              <h2>
                {tab === "views"
                  ? "Views"
                  : tab === "segmentation"
                    ? "Segment"
                    : "Compare"}
              </h2>
              {subtaskSelector}
              <div className="artifact-panel__actions">
                <button
                  className="icon-button"
                  type="button"
                  aria-label="Close visual preview"
                  title="Close visual preview"
                  onClick={() => setFlyoutOpen(false)}
                >
                  <X size={15} />
                </button>
              </div>
            </div>
            {artifactBody}
          </section>
        ) : null}
      </aside>
    );
  }

  return (
    <aside className="artifact-panel artifact-panel--expanded workspace-enter" ref={rootRef}>
      <div className="artifact-panel__heading">
        <h2>Visual Results</h2>
        {subtaskSelector}
        <div className="artifact-panel__actions">
          <button
            className="artifact-collapse-button"
            type="button"
            aria-label="Close Visual Results"
            title="Close Visual Results"
            onClick={() => onCollapsedChange(true)}
          >
            <PanelRightClose size={18} />
          </button>
        </div>
      </div>
      {tabs}
      {artifactBody}
    </aside>
  );
}
