"use client";

import {
  Box,
  History,
  LoaderCircle,
  MessageSquarePlus,
  MessageSquareText,
  PanelLeftClose,
  PanelLeftOpen,
  RotateCw,
  Trash2,
  X,
} from "lucide-react";
import {
  type CSSProperties,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

import {
  formatTokenCount,
  type WorkflowTokenUsageSummary,
} from "@/lib/token-usage";

export interface ThreadListItem {
  thread_id: string;
  metadata: Record<string, unknown>;
  updated_at?: string;
  status?: "idle" | "busy" | "interrupted" | "error";
  isDraft?: boolean;
}

type Props = {
  activeThreadId: string | null;
  threads: ThreadListItem[];
  loading: boolean;
  disabled: boolean;
  deletingThreadId: string | null;
  collapsed: boolean;
  usage?: WorkflowTokenUsageSummary | null;
  onSelect: (threadId: string) => void;
  onDelete: (threadId: string) => void;
  onNew: () => void;
  onRefresh: () => void;
  onCollapsedChange: (collapsed: boolean) => void;
};

function threadTitle(thread: ThreadListItem) {
  const title = thread.metadata.title;
  return typeof title === "string" && title.trim()
    ? title
    : `Workflow ${thread.thread_id.slice(0, 8)}`;
}

const THREAD_MARQUEE_GAP_PIXELS = 30;
const THREAD_MARQUEE_SPEED_PIXELS_PER_SECOND = 40;

export function threadMarqueeDurationSeconds(textWidth: number) {
  const safeWidth = Number.isFinite(textWidth) ? Math.max(0, textWidth) : 0;
  return (
    (safeWidth + THREAD_MARQUEE_GAP_PIXELS) /
    THREAD_MARQUEE_SPEED_PIXELS_PER_SECOND
  );
}

function ScrollingTitle({ title }: { title: string }) {
  const containerRef = useRef<HTMLSpanElement>(null);
  const textRef = useRef<HTMLSpanElement>(null);
  const [overflowing, setOverflowing] = useState(false);
  const [marqueeDuration, setMarqueeDuration] = useState(8);

  useLayoutEffect(() => {
    const measure = () => {
      const container = containerRef.current;
      const text = textRef.current;
      if (!container || !text) return;
      const nextOverflowing = text.scrollWidth > container.clientWidth + 1;
      setOverflowing(nextOverflowing);
      if (nextOverflowing) {
        setMarqueeDuration(threadMarqueeDurationSeconds(text.scrollWidth));
      }
    };
    measure();
    const observer =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(measure);
    if (containerRef.current) observer?.observe(containerRef.current);
    if (textRef.current) observer?.observe(textRef.current);
    window.addEventListener("resize", measure);
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [title]);

  return (
    <span
      className={`thread-title ${overflowing ? "thread-title--overflow" : ""}`}
      ref={containerRef}
    >
      <span
        className="thread-title__track"
        style={
          overflowing
            ? ({
                "--thread-marquee-duration": `${marqueeDuration}s`,
              } as CSSProperties)
            : undefined
        }
      >
        <span className="thread-title__text" ref={textRef}>
          {title}
        </span>
        {overflowing ? (
          <span className="thread-title__text" aria-hidden="true">
            {title}
          </span>
        ) : null}
      </span>
    </span>
  );
}

export function ThreadSidebar({
  activeThreadId,
  threads,
  loading,
  disabled,
  deletingThreadId,
  collapsed,
  usage = null,
  onSelect,
  onDelete,
  onNew,
  onRefresh,
  onCollapsedChange,
}: Props) {
  const [recentsOpen, setRecentsOpen] = useState(false);
  const rootRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (!recentsOpen) return;
    const closeFromPointer = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        setRecentsOpen(false);
      }
    };
    const closeFromKeyboard = (event: KeyboardEvent) => {
      if (event.key === "Escape") setRecentsOpen(false);
    };
    document.addEventListener("pointerdown", closeFromPointer);
    document.addEventListener("keydown", closeFromKeyboard);
    return () => {
      document.removeEventListener("pointerdown", closeFromPointer);
      document.removeEventListener("keydown", closeFromKeyboard);
    };
  }, [recentsOpen]);

  const threadList = (
    <nav className="thread-list" aria-label="Workflow threads">
      {threads.length === 0 && !loading ? (
        <div className="sidebar-empty">
          <MessageSquareText size={20} />
          <p>No saved threads yet.</p>
          <span>Start a new task to create one.</span>
        </div>
      ) : null}
      {threads.map((thread) => {
        const title = threadTitle(thread);
        const active = thread.thread_id === activeThreadId;
        const deleting = deletingThreadId === thread.thread_id;
        return (
          <div
            className={`thread-row ${active ? "thread-row--active" : ""}`}
            key={thread.thread_id}
          >
            <button
              type="button"
              className="thread-card"
              onClick={() => {
                setRecentsOpen(false);
                onSelect(thread.thread_id);
              }}
              disabled={disabled && !active}
              aria-current={active ? "page" : undefined}
              aria-label={`Open thread: ${title}`}
            >
              <ScrollingTitle title={title} />
            </button>
            <button
              className="thread-delete"
              type="button"
              onClick={() => onDelete(thread.thread_id)}
              disabled={deletingThreadId !== null || (disabled && !active)}
              aria-label={`Delete thread: ${title}`}
              title="Delete thread"
            >
              {deleting ? (
                <LoaderCircle size={13} className="spin" />
              ) : (
                <Trash2 size={13} />
              )}
            </button>
          </div>
        );
      })}
    </nav>
  );

  if (collapsed) {
    return (
      <aside className="sidebar sidebar--collapsed" ref={rootRef}>
        <div className="sidebar-rail" aria-label="Collapsed sidebar">
          <button
            className="sidebar-rail-button"
            type="button"
            aria-label="Open sidebar"
            title="Open sidebar"
            onClick={() => {
              setRecentsOpen(false);
              onCollapsedChange(false);
            }}
          >
            <Box
              className="sidebar-rail-brand-icon"
              size={19}
              strokeWidth={1.8}
            />
            <PanelLeftOpen className="sidebar-rail-open-icon" size={19} />
          </button>
          <button
            className="sidebar-rail-button"
            type="button"
            aria-label="New chat"
            title="New chat"
            disabled={disabled}
            onClick={() => {
              setRecentsOpen(false);
              onNew();
            }}
          >
            <MessageSquarePlus size={19} />
          </button>
          <button
            className={`sidebar-rail-button ${recentsOpen ? "active" : ""}`}
            type="button"
            aria-label="Recents"
            title="Recents"
            aria-expanded={recentsOpen}
            aria-controls="sidebar-recents-flyout"
            onClick={() => setRecentsOpen((value) => !value)}
          >
            <History size={19} />
          </button>
        </div>

        {recentsOpen ? (
          <section
            className="sidebar-recents-flyout panel-flyout-enter"
            id="sidebar-recents-flyout"
            aria-label="Recents"
          >
            <div className="sidebar-flyout-heading">
              <h2>Recents</h2>
              <div>
                <button
                  className="icon-button"
                  type="button"
                  onClick={onRefresh}
                  aria-label="Refresh threads"
                  title="Refresh threads"
                >
                  <RotateCw size={14} className={loading ? "spin" : ""} />
                </button>
                <button
                  className="icon-button"
                  type="button"
                  onClick={() => setRecentsOpen(false)}
                  aria-label="Close Recents"
                  title="Close Recents"
                >
                  <X size={15} />
                </button>
              </div>
            </div>
            {threadList}
          </section>
        ) : null}
      </aside>
    );
  }

  return (
    <aside className="sidebar sidebar--expanded" ref={rootRef}>
      <div className="sidebar-brand-row">
        <div className="sidebar-brand-lockup">
          <span className="sidebar-brand-copy">
            <span>ViSculpt</span>
          </span>
        </div>
        <button
          className="sidebar-control-button"
          type="button"
          aria-label="Close sidebar"
          title="Close sidebar"
          onClick={() => onCollapsedChange(true)}
        >
          <PanelLeftClose size={18} />
        </button>
      </div>

      <div className="sidebar-primary-actions">
        <button
          className="new-task-button"
          type="button"
          onClick={onNew}
          disabled={disabled}
        >
          <MessageSquarePlus size={18} />
          New task
        </button>
      </div>

      <div className="sidebar-heading">
        <span>Recents</span>
        <button
          className="icon-button"
          type="button"
          onClick={onRefresh}
          aria-label="Refresh threads"
          title="Refresh threads"
        >
          <RotateCw size={14} className={loading ? "spin" : ""} />
        </button>
      </div>
      {threadList}
      {usage && usage.aggregate.call_count > 0 ? (
        <div className="sidebar-token-usage" role="status" aria-live="polite">
          <div className="sidebar-token-usage__heading">
            <span>Token usage</span>
            <small>{usage.aggregate.call_count} calls</small>
          </div>
          <div className="sidebar-token-usage__metrics">
            <span>
              <small>Input</small>
              {formatTokenCount(usage.aggregate.tokens.input_tokens)}
            </span>
            <span>
              <small>Output</small>
              {formatTokenCount(usage.aggregate.tokens.output_tokens)}
            </span>
            <span>
              <small>Total</small>
              {formatTokenCount(usage.aggregate.tokens.total_tokens)}
            </span>
          </div>
        </div>
      ) : null}
    </aside>
  );
}
