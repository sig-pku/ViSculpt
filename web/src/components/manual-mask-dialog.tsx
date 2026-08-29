"use client";

import {
  Check,
  MousePointer2,
  Paintbrush,
  Redo2,
  RotateCcw,
  SkipForward,
  Undo2,
} from "lucide-react";
import NextImage from "next/image";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";

import { ModalPortal } from "@/components/modal-portal";
import {
  artifactUrl,
  type ManualMaskDecision,
  type ManualMaskPaintRequest,
  type ManualMaskRequest,
  type ManualMaskStroke,
} from "@/lib/workflow-types";

type Props = {
  request: ManualMaskRequest;
  responding: boolean;
  onDecision: (decision: ManualMaskDecision) => void;
};

type Transform = { scale: number; x: number; y: number };
type StrokeHistory = {
  past: ManualMaskStroke[][];
  present: ManualMaskStroke[];
  future: ManualMaskStroke[][];
};
type PointerMode =
  | { kind: "paint"; pointerId: number }
  | {
      kind: "pan";
      pointerId: number;
      startX: number;
      startY: number;
      originX: number;
      originY: number;
    }
  | null;

const HISTORY_LIMIT = 5;
const POINT_DISTANCE_THRESHOLD = 1.5;
const MIN_ZOOM = 0.2;
const MAX_ZOOM = 12;
const MAX_BRUSH_SIZE = 300;

export function ManualMaskDialog({ request, responding, onDecision }: Props) {
  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, []);

  return (
    <ModalPortal>
      <div className="manual-mask-layer">
        <div className="manual-mask-backdrop" aria-hidden="true" />
        {request.stage === "paint" ? (
          <ManualMaskPainter
            request={request}
            responding={responding}
            onDecision={onDecision}
          />
        ) : (
          <ManualMaskReview
            request={request}
            responding={responding}
            onDecision={onDecision}
          />
        )}
      </div>
    </ModalPortal>
  );
}

function ManualMaskPainter({
  request,
  responding,
  onDecision,
}: {
  request: ManualMaskPaintRequest;
  responding: boolean;
  onDecision: (decision: ManualMaskDecision) => void;
}) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const pointerModeRef = useRef<PointerMode>(null);
  const activeStrokeRef = useRef<ManualMaskStroke | null>(null);
  const cursorClientRef = useRef<{ x: number; y: number } | null>(null);
  const fAdjustRef = useRef<{
    originX: number;
    initialSize: number;
    initialScale: number;
  } | null>(null);
  const [imageReady, setImageReady] = useState(false);
  const [imageError, setImageError] = useState<string | null>(null);
  const [transform, setTransform] = useState<Transform>({
    scale: 1,
    x: 0,
    y: 0,
  });
  const transformRef = useRef(transform);
  const minimumBrushSize = Math.max(
    1,
    Math.min(MAX_BRUSH_SIZE, Math.round(request.brush.minimum_size)),
  );
  const maximumBrushSize = Math.max(
    minimumBrushSize,
    Math.min(MAX_BRUSH_SIZE, Math.round(request.brush.maximum_size)),
  );
  const [brushSize, setBrushSize] = useState(() =>
    Math.max(
      minimumBrushSize,
      Math.min(maximumBrushSize, Math.round(request.brush.default_size)),
    ),
  );
  const brushSizeRef = useRef(brushSize);
  const [history, setHistory] = useState<StrokeHistory>({
    past: [],
    present: [],
    future: [],
  });
  const [activeStroke, setActiveStroke] = useState<ManualMaskStroke | null>(
    null,
  );
  const [cursor, setCursor] = useState<{
    x: number;
    y: number;
    visible: boolean;
  }>({ x: 0, y: 0, visible: false });
  const [fAdjusting, setFAdjusting] = useState(false);

  useEffect(() => {
    transformRef.current = transform;
  }, [transform]);
  useEffect(() => {
    brushSizeRef.current = brushSize;
  }, [brushSize]);

  const clampBrush = useCallback(
    (value: number) =>
      Math.max(
        minimumBrushSize,
        Math.min(maximumBrushSize, Math.round(value)),
      ),
    [maximumBrushSize, minimumBrushSize],
  );

  const fitImage = useCallback(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const bounds = viewport.getBoundingClientRect();
    if (bounds.width <= 0 || bounds.height <= 0) return;
    const scale = Math.min(
      bounds.width / request.image_width,
      bounds.height / request.image_height,
    );
    setTransform({
      scale,
      x: (bounds.width - request.image_width * scale) / 2,
      y: (bounds.height - request.image_height * scale) / 2,
    });
  }, [request.image_height, request.image_width]);

  useEffect(() => {
    const image = new window.Image();
    image.decoding = "async";
    image.onload = () => {
      if (
        image.naturalWidth !== request.image_width ||
        image.naturalHeight !== request.image_height
      ) {
        setImageError("The screenshot dimensions no longer match this request.");
        return;
      }
      imageRef.current = image;
      setImageReady(true);
      requestAnimationFrame(fitImage);
    };
    image.onerror = () => {
      setImageError("The source screenshot could not be loaded.");
    };
    image.src = artifactUrl(request.source_artifact);
    return () => {
      image.onload = null;
      image.onerror = null;
      imageRef.current = null;
    };
  }, [fitImage, request]);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const observer = new ResizeObserver(() => fitImage());
    observer.observe(viewport);
    return () => observer.disconnect();
  }, [fitImage]);

  const visibleStrokes = useMemo(
    () =>
      activeStroke
        ? [...history.present, activeStroke]
        : history.present,
    [activeStroke, history.present],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image || !imageReady) return;
    canvas.width = request.image_width;
    canvas.height = request.image_height;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    context.save();
    context.globalAlpha = 0.52;
    context.strokeStyle = "#35d58a";
    context.fillStyle = "#35d58a";
    context.lineCap = "round";
    context.lineJoin = "round";
    for (const stroke of visibleStrokes) {
      const points = stroke.points;
      if (!points.length) continue;
      context.lineWidth = stroke.brush_size * 2;
      if (points.length === 1) {
        context.beginPath();
        context.arc(
          points[0].x,
          points[0].y,
          stroke.brush_size,
          0,
          Math.PI * 2,
        );
        context.fill();
        continue;
      }
      context.beginPath();
      context.moveTo(points[0].x, points[0].y);
      for (const point of points.slice(1)) context.lineTo(point.x, point.y);
      context.stroke();
    }
    context.restore();
  }, [imageReady, request.image_height, request.image_width, visibleStrokes]);

  const updateCursor = useCallback((clientX: number, clientY: number) => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    const bounds = viewport.getBoundingClientRect();
    cursorClientRef.current = { x: clientX, y: clientY };
    setCursor({
      x: clientX - bounds.left,
      y: clientY - bounds.top,
      visible:
        clientX >= bounds.left &&
        clientX <= bounds.right &&
        clientY >= bounds.top &&
        clientY <= bounds.bottom,
    });
  }, []);

  const imagePoint = useCallback((clientX: number, clientY: number) => {
    const viewport = viewportRef.current;
    if (!viewport) return null;
    const bounds = viewport.getBoundingClientRect();
    const current = transformRef.current;
    const x = (clientX - bounds.left - current.x) / current.scale;
    const y = (clientY - bounds.top - current.y) / current.scale;
    if (
      x < 0 ||
      x > request.image_width - 1 ||
      y < 0 ||
      y > request.image_height - 1
    ) {
      return null;
    }
    return { x, y };
  }, [request.image_height, request.image_width]);

  const commitStroke = useCallback((stroke: ManualMaskStroke) => {
    setHistory((current) => ({
      past: [...current.past, current.present].slice(-HISTORY_LIMIT),
      present: [...current.present, stroke],
      future: [],
    }));
  }, []);

  const finishPointer = useCallback((pointerId: number) => {
    const mode = pointerModeRef.current;
    if (!mode || mode.pointerId !== pointerId) return;
    if (mode.kind === "paint" && activeStrokeRef.current) {
      commitStroke(activeStrokeRef.current);
    }
    activeStrokeRef.current = null;
    pointerModeRef.current = null;
    setActiveStroke(null);
  }, [commitStroke]);

  function handlePointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (responding || event.button !== 0) return;
    updateCursor(event.clientX, event.clientY);
    if (fAdjustRef.current) {
      fAdjustRef.current = null;
      setFAdjusting(false);
      event.preventDefault();
      return;
    }
    if (event.shiftKey) {
      event.currentTarget.setPointerCapture(event.pointerId);
      const current = transformRef.current;
      pointerModeRef.current = {
        kind: "pan",
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        originX: current.x,
        originY: current.y,
      };
      event.preventDefault();
      return;
    }
    const point = imagePoint(event.clientX, event.clientY);
    if (!point) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const stroke: ManualMaskStroke = {
      brush_size: brushSizeRef.current,
      points: [point],
    };
    activeStrokeRef.current = stroke;
    pointerModeRef.current = { kind: "paint", pointerId: event.pointerId };
    setActiveStroke(stroke);
    event.preventDefault();
  }

  function handlePointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    if (fAdjustRef.current) {
      event.preventDefault();
      return;
    }
    updateCursor(event.clientX, event.clientY);
    const mode = pointerModeRef.current;
    if (!mode || mode.pointerId !== event.pointerId) return;
    if (mode.kind === "pan") {
      setTransform((current) => ({
        ...current,
        x: mode.originX + event.clientX - mode.startX,
        y: mode.originY + event.clientY - mode.startY,
      }));
      event.preventDefault();
      return;
    }
    const point = imagePoint(event.clientX, event.clientY);
    const stroke = activeStrokeRef.current;
    if (!point || !stroke) return;
    const previous = stroke.points[stroke.points.length - 1];
    if (
      Math.hypot(point.x - previous.x, point.y - previous.y) <
      POINT_DISTANCE_THRESHOLD
    ) {
      return;
    }
    const updated = { ...stroke, points: [...stroke.points, point] };
    activeStrokeRef.current = updated;
    setActiveStroke(updated);
    event.preventDefault();
  }

  const handleWheel = useCallback((event: WheelEvent) => {
    event.preventDefault();
    event.stopPropagation();
    const viewport = viewportRef.current;
    if (!viewport) return;
    const bounds = viewport.getBoundingClientRect();
    const localX = event.clientX - bounds.left;
    const localY = event.clientY - bounds.top;
    const current = transformRef.current;
    const imageX = (localX - current.x) / current.scale;
    const imageY = (localY - current.y) / current.scale;
    const nextScale = Math.max(
      MIN_ZOOM,
      Math.min(MAX_ZOOM, current.scale * Math.exp(-event.deltaY * 0.0015)),
    );
    setTransform({
      scale: nextScale,
      x: localX - imageX * nextScale,
      y: localY - imageY * nextScale,
    });
  }, []);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    viewport.addEventListener("wheel", handleWheel, { passive: false });
    return () => viewport.removeEventListener("wheel", handleWheel);
  }, [handleWheel]);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      const target = event.target;
      if (
        target instanceof Element &&
        target.matches("input, textarea, select, [contenteditable='true']")
      ) {
        return;
      }
      if (
        event.key.toLowerCase() === "f" &&
        !event.repeat &&
        !fAdjustRef.current &&
        cursorClientRef.current
      ) {
        event.preventDefault();
        fAdjustRef.current = {
          originX: cursorClientRef.current.x,
          initialSize: brushSizeRef.current,
          initialScale: transformRef.current.scale,
        };
        setFAdjusting(true);
      } else if (event.key === "Escape" && fAdjustRef.current) {
        setBrushSize(fAdjustRef.current.initialSize);
        fAdjustRef.current = null;
        setFAdjusting(false);
      }
    }
    function handlePointerMove(event: PointerEvent) {
      const adjustment = fAdjustRef.current;
      if (!adjustment) return;
      setBrushSize(
        clampBrush(
          adjustment.initialSize +
            (event.clientX - adjustment.originX) /
              adjustment.initialScale,
        ),
      );
    }
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("pointermove", handlePointerMove);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("pointermove", handlePointerMove);
    };
  }, [clampBrush]);

  function handlePointerLeave() {
    if (!fAdjustRef.current) {
      setCursor((value) => ({ ...value, visible: false }));
    }
  }

  const undo = () => {
    setHistory((current) => {
      if (!current.past.length) return current;
      return {
        past: current.past.slice(0, -1),
        present: current.past[current.past.length - 1],
        future: [current.present, ...current.future].slice(0, HISTORY_LIMIT),
      };
    });
  };
  const redo = () => {
    setHistory((current) => {
      if (!current.future.length) return current;
      return {
        past: [...current.past, current.present].slice(-HISTORY_LIMIT),
        present: current.future[0],
        future: current.future.slice(1),
      };
    });
  };

  return (
    <section
      className="manual-mask-dialog"
      role="dialog"
      aria-modal="true"
      aria-labelledby="manual-mask-title"
    >
      <header className="manual-mask-header">
        <span className="manual-mask-header__icon"><Paintbrush size={19} /></span>
        <div>
          <h2 id="manual-mask-title">Paint the missing segmentation mask</h2>
          <p>
            SAM3 could not segment <strong>{request.part_description}</strong>.
            Paint the intended region; it will be clipped to the model surface.
          </p>
        </div>
      </header>

      <div className="manual-mask-editor">
        <div
          ref={viewportRef}
          className={`manual-mask-viewport ${fAdjusting ? "manual-mask-viewport--adjusting" : ""}`}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={(event) => finishPointer(event.pointerId)}
          onPointerCancel={(event) => finishPointer(event.pointerId)}
          onPointerEnter={(event) => updateCursor(event.clientX, event.clientY)}
          onPointerLeave={handlePointerLeave}
        >
          <canvas
            ref={canvasRef}
            className="manual-mask-canvas"
            style={{
              width: request.image_width,
              height: request.image_height,
              transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`,
            }}
            aria-label={`Paint mask for ${request.part_description}`}
          />
          {cursor.visible ? (
            <span
              className="manual-mask-brush-cursor"
              style={{
                left: cursor.x,
                top: cursor.y,
                width: Math.max(2, brushSize * transform.scale * 2),
                height: Math.max(2, brushSize * transform.scale * 2),
              }}
              aria-hidden="true"
            />
          ) : null}
          {!imageReady ? (
            <span className="manual-mask-loading">
              {imageError ?? "Loading image…"}
            </span>
          ) : null}
          {fAdjusting ? (
            <span className="manual-mask-f-hint">Move horizontally, then click to confirm</span>
          ) : null}
        </div>

        <aside className="manual-mask-controls">
          <div className="manual-mask-size-row">
            <label htmlFor="manual-mask-size">Size</label>
            <input
              id="manual-mask-size"
              type="range"
              min={minimumBrushSize}
              max={maximumBrushSize}
              value={brushSize}
              onChange={(event) => setBrushSize(clampBrush(Number(event.target.value)))}
              disabled={responding}
            />
            <input
              className="manual-mask-size-number"
              aria-label="Brush size in pixels"
              type="number"
              min={minimumBrushSize}
              max={maximumBrushSize}
              value={brushSize}
              onChange={(event) => setBrushSize(clampBrush(Number(event.target.value)))}
              disabled={responding}
            />
            <span>px</span>
          </div>
          <div className="manual-mask-control-actions">
            <button type="button" onClick={undo} disabled={!history.past.length || responding}>
              <Undo2 size={15} /> Undo
            </button>
            <button type="button" onClick={redo} disabled={!history.future.length || responding}>
              <Redo2 size={15} /> Redo
            </button>
            <button type="button" onClick={fitImage} disabled={responding}>
              <RotateCcw size={15} /> Reset view
            </button>
          </div>
          <div className="manual-mask-shortcuts">
            <span><MousePointer2 size={14} /> Drag to paint</span>
            <span><kbd>Shift</kbd> + drag to pan</span>
            <span>Wheel to zoom</span>
            <span><kbd>F</kbd> to resize</span>
          </div>
        </aside>
      </div>

      <footer className="manual-mask-actions">
        <button
          className="manual-mask-secondary"
          type="button"
          onClick={() => onDecision({ decision: "skip" })}
          disabled={responding}
        >
          <SkipForward size={15} /> Skip subtask
        </button>
        <button
          className="manual-mask-primary"
          type="button"
          disabled={!history.present.length || responding || !imageReady}
          onClick={() =>
            onDecision({
              decision: "finish",
              image_width: request.image_width,
              image_height: request.image_height,
              strokes: history.present,
            })
          }
        >
          <Check size={15} /> {responding ? "Processing…" : "Finish"}
        </button>
      </footer>
    </section>
  );
}

function ManualMaskReview({
  request,
  responding,
  onDecision,
}: {
  request: Extract<ManualMaskRequest, { stage: "review" }>;
  responding: boolean;
  onDecision: (decision: ManualMaskDecision) => void;
}) {
  return (
    <section
      className="manual-mask-dialog manual-mask-dialog--review"
      role="dialog"
      aria-modal="true"
      aria-labelledby="manual-mask-review-title"
    >
      <header className="manual-mask-header">
        <span className="manual-mask-header__icon"><Check size={19} /></span>
        <div>
          <h2 id="manual-mask-review-title">Review the clipped mask</h2>
          <p>
            The painted region has been intersected with the complete model mask
            for <strong>{request.part_description}</strong>.
          </p>
        </div>
      </header>
      <div className="manual-mask-review-image">
        <NextImage
          src={artifactUrl(request.overlay_artifact)}
          alt={`Manual mask preview for ${request.part_description}`}
          fill
          sizes="min(86vw, 1100px)"
          unoptimized
        />
      </div>
      {!request.confirm_allowed ? (
        <p className="manual-mask-empty-warning" role="alert">
          The painted region does not overlap the model. Redraw the mask or skip
          this subtask.
        </p>
      ) : null}
      <footer className="manual-mask-actions">
        <button
          className="manual-mask-secondary"
          type="button"
          onClick={() => onDecision({ decision: "skip" })}
          disabled={responding}
        >
          <SkipForward size={15} /> Skip subtask
        </button>
        <span className="manual-mask-actions__spacer" />
        <button
          className="manual-mask-secondary"
          type="button"
          onClick={() => onDecision({ decision: "redraw" })}
          disabled={responding}
        >
          <Paintbrush size={15} /> Redraw
        </button>
        <button
          className="manual-mask-primary"
          type="button"
          onClick={() => onDecision({ decision: "confirm" })}
          disabled={responding || !request.confirm_allowed}
        >
          <Check size={15} /> {responding ? "Continuing…" : "Confirm mask"}
        </button>
      </footer>
    </section>
  );
}
