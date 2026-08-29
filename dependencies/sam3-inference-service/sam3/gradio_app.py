"""Gradio UI and named HTTP API for local SAM 3 inference."""

from __future__ import annotations

import argparse
import tempfile
import threading
from collections import deque
from pathlib import Path
from uuid import uuid4

import gradio as gr
import numpy as np
from PIL import Image

from sam3.inference import Sam3TextSegmenter, render_overlay


class Sam3InferenceService:
    """Lazily load the model and serialize access to its mutable processor state."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        device: str = "auto",
        allow_download: bool = True,
        compile_model: bool = False,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.allow_download = allow_download
        self.compile_model = compile_model
        self._segmenter: Sam3TextSegmenter | None = None
        self._lock = threading.Lock()
        # Keep the directory alive so Gradio can serve returned mask files.
        self._artifact_directory = tempfile.TemporaryDirectory(
            prefix="sam3-instance-masks-"
        )
        self._instance_mask_artifacts: deque[Path] = deque()
        self._max_instance_mask_artifacts = 32

    def _load(self) -> Sam3TextSegmenter:
        if self._segmenter is None:
            self._segmenter = Sam3TextSegmenter(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                allow_download=self.allow_download,
                compile_model=self.compile_model,
            )
        return self._segmenter

    def segment(
        self,
        image: Image.Image | None,
        prompt: str,
        confidence_threshold: float,
        overlay_opacity: float,
    ) -> tuple[Image.Image, Image.Image, str, dict]:
        if image is None:
            raise gr.Error("Upload an image before running segmentation.")
        if not prompt or not prompt.strip():
            raise gr.Error("Enter a non-empty English text prompt.")

        try:
            with self._lock:
                segmenter = self._load()
                result = segmenter.segment(
                    image,
                    prompt,
                    confidence_threshold=confidence_threshold,
                )
            overlay = render_overlay(image, result, opacity=overlay_opacity)
            semantic_mask = Image.fromarray(
                result.semantic_mask.astype(np.uint8) * 255,
                mode="L",
            )
            instance_masks_path = (
                Path(self._artifact_directory.name)
                / f"instance-masks-{uuid4().hex}.npz"
            )
            np.savez_compressed(
                instance_masks_path,
                masks=result.masks.astype(np.bool_, copy=False),
            )
            self._instance_mask_artifacts.append(instance_masks_path)
            while (
                len(self._instance_mask_artifacts)
                > self._max_instance_mask_artifacts
            ):
                self._instance_mask_artifacts.popleft().unlink(
                    missing_ok=True
                )
            metadata = result.to_metadata()
            metadata.update(
                {
                    "protocol_version": "sam3-gradio-segmentation/v2",
                    "prompt": prompt.strip(),
                    "device": str(segmenter.device),
                    "precision": segmenter.precision,
                    "checkpoint": str(segmenter.checkpoint_path),
                }
            )
            return (
                overlay,
                semantic_mask,
                str(instance_masks_path),
                metadata,
            )
        except gr.Error:
            raise
        except Exception as error:
            raise gr.Error(f"SAM 3 inference failed: {error}") from error


def build_demo(
    checkpoint_path: str | Path | None = None,
    *,
    device: str = "auto",
    allow_download: bool = True,
    compile_model: bool = False,
    service: Sam3InferenceService | None = None,
) -> gr.Blocks:
    service = service or Sam3InferenceService(
        checkpoint_path=checkpoint_path,
        device=device,
        allow_download=allow_download,
        compile_model=compile_model,
    )

    with gr.Blocks(title="SAM 3 Text-Prompt Segmentation") as demo:
        gr.Markdown(
            "# SAM 3 Text-Prompt Segmentation\n"
            "Upload an image and enter a short English concept such as "
            "`red car`. The service uses original SAM 3 and prefers a local "
            "`checkpoint/sam3.pt`."
        )
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Input image",
                    type="pil",
                    image_mode="RGB",
                    sources=["upload", "clipboard"],
                )
                prompt = gr.Textbox(
                    label="Text prompt",
                    placeholder="For example: person wearing a red shirt",
                    lines=1,
                )
                confidence = gr.Slider(
                    minimum=0.05,
                    maximum=0.95,
                    value=0.5,
                    step=0.05,
                    label="Confidence threshold",
                )
                opacity = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.45,
                    step=0.05,
                    label="Mask opacity",
                )
                run_button = gr.Button("Segment", variant="primary")
                gr.ClearButton([input_image, prompt])
            with gr.Column():
                overlay = gr.Image(
                    label="Segmentation overlay",
                    type="pil",
                    interactive=False,
                )
                semantic_mask = gr.Image(
                    label="Semantic union mask",
                    type="pil",
                    image_mode="L",
                    interactive=False,
                )
                instance_masks = gr.File(
                    label="Per-instance masks (NPZ)",
                    interactive=False,
                    visible=False,
                )
                metadata = gr.JSON(label="Instances and runtime metadata")

        run_button.click(
            fn=service.segment,
            inputs=[input_image, prompt, confidence, opacity],
            outputs=[overlay, semantic_mask, instance_masks, metadata],
            api_name="segment",
            api_description=(
                "Use original SAM 3 to segment all instances matching a "
                "text prompt and return both the semantic union and "
                "per-instance masks."
            ),
            concurrency_limit=1,
            concurrency_id="sam3-inference",
        )

    return demo.queue(default_concurrency_limit=1, max_size=8)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda, cuda:<index>, mps, or cpu",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail instead of downloading from facebook/sam3 when no local checkpoint exists.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        dest="compile_model",
        help="Enable torch.compile optimizations (CUDA only).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    demo = build_demo(
        checkpoint_path=args.checkpoint,
        device=args.device,
        allow_download=not args.no_download,
        compile_model=args.compile_model,
    )
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()


__all__ = ["Sam3InferenceService", "build_demo", "main"]
