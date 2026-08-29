"""Readiness probes for external services used by the graph."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

from gradio_client import Client
from PIL import Image, ImageDraw

from visculpt.bridge import (
    BlenderRpcBridgeError,
    BlenderRpcClient,
    JsonValue,
)
from visculpt.vision import (
    Sam3GradioClient,
    Sam3GradioConfig,
)

from .errors import WorkflowExecutionError

REQUIRED_BLENDER_METHODS = frozenset(
    {
        "activate_sculpt_brush",
        "enter_sculpt_mode",
        "get_state",
        "get_screenshot",
        "load_blend_file",
        "restore_sculpt_viewport_ui",
        "focus_viewport_roi",
        "restore_viewport_state",
        "save_blend_file",
        "sculpt_brush_stroke",
        "sculpt_face_set_lasso_batch",
        "set_sculpt_settings",
        "set_dyntopo",
        "set_sculpt_brush",
        "set_use_size_pressure",
        "set_use_strength_pressure",
        "set_use_unified_size",
        "set_use_unified_strength",
        "set_view",
    }
)


class GradioProbeClient(Protocol):
    """Small surface needed to inspect a running Gradio application."""

    def view_api(
        self,
        all_endpoints: bool | None = None,
        print_info: bool = True,
        return_format: str | None = None,
    ) -> dict[str, object] | str | None:
        """Return endpoint metadata."""

    def close(self) -> None:
        """Release background resources."""


type GradioProbeFactory = Callable[..., GradioProbeClient]


def check_blender_rpc_ready(
    client: BlenderRpcClient,
) -> dict[str, JsonValue]:
    """Verify RPC reachability and every method needed by this workflow."""
    try:
        ping = client.send(_rpc_request("ping"))
        ping_result = _rpc_result(ping, "ping")
        if ping_result.get("pong") is not True:
            raise WorkflowExecutionError(
                "Blender RPC ping response did not contain pong=true"
            )
        discovery = client.send(_rpc_request("rpc.discover"))
        discovery_result = _rpc_result(discovery, "rpc.discover")
    except BlenderRpcBridgeError as error:
        raise WorkflowExecutionError(
            f"Blender RPC Server is not ready: {error}"
        ) from error
    methods = discovery_result.get("methods")
    if not isinstance(methods, dict):
        raise WorkflowExecutionError(
            "Blender rpc.discover response is missing methods"
        )
    missing = sorted(REQUIRED_BLENDER_METHODS.difference(methods))
    if missing:
        raise WorkflowExecutionError(
            "Blender RPC Server is missing required methods: "
            + ", ".join(missing)
        )
    return {
        "ready": True,
        "pong": True,
        "required_methods": sorted(REQUIRED_BLENDER_METHODS),
    }


def check_sam3_ready(
    config: Sam3GradioConfig,
    *,
    client_factory: GradioProbeFactory = Client,
    inference_probe: Callable[[], Mapping[str, object]] | None = None,
) -> dict[str, JsonValue]:
    """Verify endpoint discovery and one complete lightweight inference."""
    client: GradioProbeClient | None = None
    try:
        client = client_factory(
            config.service_url,
            verbose=False,
            analytics_enabled=False,
            download_files=False,
            httpx_kwargs={"timeout": config.timeout},
        )
        api = client.view_api(print_info=False, return_format="dict")
    except Exception as error:
        raise WorkflowExecutionError(
            "SAM3 Gradio Service is not ready at "
            f"{config.service_url}: {error}"
        ) from error
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
    if not _contains_segment_endpoint(api):
        raise WorkflowExecutionError(
            "SAM3 Gradio Service does not expose the named /segment endpoint"
        )
    try:
        inference = dict(
            inference_probe()
            if inference_probe is not None
            else _run_sam3_inference_probe(config)
        )
    except Exception as error:
        raise WorkflowExecutionError(
            "SAM3 endpoint is discoverable but a real inference failed: "
            f"{error}"
        ) from error
    if inference.get("verified") is not True:
        raise WorkflowExecutionError(
            "SAM3 inference probe did not verify persisted outputs"
        )
    return {
        "ready": True,
        "service_url": config.service_url,
        "api_name": "/segment",
        "inference_verified": True,
    }


def _run_sam3_inference_probe(
    config: Sam3GradioConfig,
) -> dict[str, object]:
    """Exercise model loading and output persistence with a tiny image."""
    with tempfile.TemporaryDirectory(
        prefix="agentic-geometry-sam3-health-"
    ) as temporary:
        root = Path(temporary)
        image_path = root / "probe.png"
        image = Image.new("RGB", (64, 64), "black")
        draw = ImageDraw.Draw(image)
        draw.rectangle((18, 18, 46, 46), fill="white")
        image.save(image_path)
        result = Sam3GradioClient(config).segment(
            image_path=image_path,
            prompt="white square",
            confidence_threshold=0.5,
            overlay_opacity=0.45,
            output_dir=root / "result",
        )
        outputs = (
            Path(result.overlay_path),
            Path(result.mask_path),
            Path(result.instance_masks_path),
        )
        verified = all(path.exists() for path in outputs)
        return {
            "verified": verified,
            "instance_count": result.metadata.get("instance_count"),
        }


def _rpc_request(method: str) -> dict[str, JsonValue]:
    return {
        "jsonrpc": "2.0",
        "id": f"workflow-readiness-{uuid4().hex}",
        "method": method,
        "params": {},
    }


def _rpc_result(value: JsonValue, label: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise WorkflowExecutionError(
            f"Blender RPC {label} returned a non-object response"
        )
    error = value.get("error")
    if error is not None:
        raise WorkflowExecutionError(
            f"Blender RPC {label} failed: "
            f"{json.dumps(error, ensure_ascii=False)}"
        )
    result = value.get("result")
    if not isinstance(result, dict):
        raise WorkflowExecutionError(
            f"Blender RPC {label} response is missing an object result"
        )
    return cast(dict[str, JsonValue], result)


def _contains_segment_endpoint(value: object) -> bool:
    if isinstance(value, str):
        return value == "/segment" or "/segment" in value
    if isinstance(value, Mapping):
        return any(
            _contains_segment_endpoint(key)
            or _contains_segment_endpoint(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_segment_endpoint(item) for item in value)
    return False
