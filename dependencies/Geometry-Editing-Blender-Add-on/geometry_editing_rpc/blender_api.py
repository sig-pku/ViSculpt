"""Blender-facing JSON-RPC method implementations."""

from __future__ import annotations

import base64
import gzip
import json
import math
import os
import sys
import tempfile
from array import array
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bpy
import imbuf
from bpy_extras import view3d_utils

from .deferred import DeferredMainThreadCall
from .brush_quality import (
    BRUSH_QUALITY_PARAMETER_RNA_NAMES,
    BRUSH_QUALITY_PARAMETERS,
    parse_brush_quality_parameters,
)
from .face_set_lasso import (
    MAX_FACE_SET_LASSO_POINTS,
    parse_face_set_lasso_request,
)
from .face_set_coverage import analyze_face_set_coverage
from .face_set_lasso_batch import (
    MAX_FACE_SET_LASSO_BATCH_PATHS,
    parse_face_set_lasso_batch_request,
)
from .image_utils import (
    CropGeometry,
    PngCropError,
    calculate_region_crop,
    png_dimensions,
    select_imbuf_load_api,
)
from .pose_brush import (
    POSE_BRUSH_PARAMETER_DEFAULTS,
    POSE_BRUSH_PARAMETER_RNA_NAMES,
    POSE_BRUSH_PARAMETERS,
    parse_pose_brush_parameters,
)
from .protocol import JsonRpcError
from .sculpt_stroke import (
    MAX_SCULPT_STROKE_ELEMENTS,
    anchor_sculpt_drag_stroke,
    parse_sculpt_brush_stroke_request,
    prepare_sculpt_paint_curve_points,
    resolve_sculpt_stroke_execution_mode,
    resolve_sculpt_stroke_location_mode,
)
from .sculpt_viewport_ui import (
    OVERLAY_UI_PROPERTIES,
    SCULPT_VIEWPORT_UI_SCHEMA_VERSION,
    SPACE_UI_REQUIRED_REGION_TYPES,
    SPACE_UI_PROPERTIES,
    StabilityTracker,
    apply_sculpt_viewport_ui_properties,
    parse_sculpt_viewport_ui_snapshot,
    read_sculpt_viewport_ui_properties,
)
from .server import API_VERSION, SERVICE_NAME

_AXIS_VIEWS = {"FRONT", "BACK", "LEFT", "RIGHT", "TOP", "BOTTOM"}
_VIEW_PROJECTIONS = {
    "PERSPECTIVE": "PERSP",
    "ORTHOGRAPHIC": "ORTHO",
}
_VIEW_PROJECTION_ALIASES = {
    "PERSP": "PERSPECTIVE",
    "ORTHO": "ORTHOGRAPHIC",
}
_ESSENTIAL_SCULPT_BRUSH_PREFIX = (
    "brushes/essentials_brushes-mesh_sculpt.blend/Brush/"
)
_LASSO_FACE_SET_TOOL_ID = "builtin.lasso_face_set"
_LASSO_FACE_SET_OPERATOR_ID = "sculpt.face_set_lasso_gesture"
_VIEWPORT_UI_SETTLE_INTERVAL = 0.05
_WORKSPACE_STABILITY_MAX_CHECKS = 40
_VIEWPORT_STABILITY_MAX_CHECKS = 20
_VIEWPORT_UI_VERIFY_ATTEMPTS = 3
_VIEWPORT_UI_SETTLE_MAX_CHECKS = 20
# Blender 5.1 region animation takes about 0.22 seconds; allow scheduling margin.
_VIEWPORT_UI_REGION_MIN_SETTLE_CHECKS = 8
_SCULPT_OPERATION_LEDGER_PROPERTY = (
    "_agentic_geometry_applied_sculpt_operations_v1"
)
_MAX_SCULPT_OPERATION_LEDGER_ENTRIES = 4096
_SCULPT_DIRECTION_VALUES_BY_TYPE = {
    "SMOOTH": ("SMOOTH", "ENHANCE_DETAILS"),
    "PINCH": ("MAGNIFY", "PINCH"),
    "INFLATE": ("INFLATE", "DEFLATE"),
    # Blender 4.2-4.4 use separate legacy plane brush types.
    "FLATTEN": ("CONTRAST", "FLATTEN"),
    "FILL": ("FILL", "DEEPEN"),
    "SCRAPE": ("SCRAPE", "PEAKS"),
}


class BlenderRpcApi:
    """Expose bounded, JSON-serializable Blender operations."""

    def __init__(self) -> None:
        self._sculpt_replay_paint_curve: Any | None = None
        self._sculpt_brush_library_cache: dict[
            str,
            tuple[
                tuple[int, int],
                tuple[tuple[str, tuple[str, ...]], ...],
                str | None,
            ],
        ] = {}
        self._methods: dict[str, Callable[[dict[str, Any]], Any]] = {
            "ping": self.ping,
            "get_state": self.get_state,
            "save_blend_file": self.save_blend_file,
            "load_blend_file": self.load_blend_file,
            "set_view": self.set_view,
            "focus_viewport_roi": self.focus_viewport_roi,
            "restore_viewport_state": self.restore_viewport_state,
            "get_screenshot": self.get_screenshot,
            "enter_sculpt_mode": self.enter_sculpt_mode,
            "restore_sculpt_viewport_ui": (
                self.restore_sculpt_viewport_ui
            ),
            "activate_lasso_face_set_tool": (
                self.activate_lasso_face_set_tool
            ),
            "sculpt_face_set_lasso": self.sculpt_face_set_lasso,
            "sculpt_face_set_lasso_batch": (
                self.sculpt_face_set_lasso_batch
            ),
            "activate_sculpt_brush": self.activate_sculpt_brush,
            "set_sculpt_settings": self.set_sculpt_settings_transaction,
            "set_sculpt_brush": self.set_sculpt_brush,
            "set_use_unified_size": self.set_use_unified_size,
            "set_use_unified_strength": self.set_use_unified_strength,
            "set_use_size_pressure": self.set_use_size_pressure,
            "set_use_strength_pressure": self.set_use_strength_pressure,
            "set_dyntopo": self.set_dyntopo,
            "sculpt_brush_stroke": self.sculpt_brush_stroke,
            "rpc.discover": self.discover,
        }
        self._aliases = {
            "blender.get_state": "get_state",
            "blender.save_blend_file": "save_blend_file",
            "blender.load_blend_file": "load_blend_file",
            "blender.set_view": "set_view",
            "blender.focus_viewport_roi": "focus_viewport_roi",
            "blender.restore_viewport_state": "restore_viewport_state",
            "blender.get_screenshot": "get_screenshot",
            "sculpt.enter": "enter_sculpt_mode",
            "sculpt.restore_viewport_ui": (
                "restore_sculpt_viewport_ui"
            ),
            "sculpt.activate_lasso_face_set_tool": (
                "activate_lasso_face_set_tool"
            ),
            "sculpt.face_set_lasso": "sculpt_face_set_lasso",
            "sculpt.face_set_lasso_batch": (
                "sculpt_face_set_lasso_batch"
            ),
            "sculpt.activate_brush": "activate_sculpt_brush",
            "sculpt.set_settings": "set_sculpt_settings",
            "sculpt.set_brush": "set_sculpt_brush",
            "sculpt.set_unified_size": "set_use_unified_size",
            "sculpt.set_unified_strength": "set_use_unified_strength",
            "sculpt.set_size_pressure": "set_use_size_pressure",
            "sculpt.set_strength_pressure": "set_use_strength_pressure",
            "sculpt.set_dyntopo": "set_dyntopo",
            "sculpt.brush_stroke": "sculpt_brush_stroke",
        }

    def call(self, method: str, params: Any) -> Any:
        """Resolve and execute an RPC method on Blender's main thread."""
        canonical_method = self._aliases.get(method, method)
        handler = self._methods.get(canonical_method)
        if handler is None:
            raise JsonRpcError(
                -32601,
                "Method not found",
                {"method": method},
            )
        if not isinstance(params, dict):
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {"reason": "Named parameters must be a JSON object"},
            )
        return handler(params)

    def ping(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return service liveness and Blender identity."""
        _reject_unknown(params, set())
        return {
            "pong": True,
            "service": SERVICE_NAME,
            "api_version": API_VERSION,
            "blender_version": bpy.app.version_string,
            "time_utc": datetime.now(timezone.utc).isoformat(),
        }

    def discover(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return a compact, machine-readable method catalog."""
        _reject_unknown(params, set())
        return {
            "service": SERVICE_NAME,
            "api_version": API_VERSION,
            "transport": {
                "protocol": "JSON-RPC 2.0",
                "http_path": "/rpc",
                "web_client_path": "/",
            },
            "methods": {
                "ping": {"params": {}},
                "get_state": {
                    "behavior": (
                        "Include every locally available Sculpt brush in "
                        "sculpt.available_brushes, each brush's Direction "
                        "choices in sculpt.available_brush_details, and "
                        "currently loaded ones in sculpt.loaded_brushes"
                    ),
                    "params": {
                        "include_objects": "boolean, default false",
                        "object_limit": "integer, default 200, max 10000",
                        "include_viewports": "boolean, default true",
                    },
                },
                "save_blend_file": {
                    "behavior": (
                        "Save a complete .blend snapshot without changing "
                        "the current project filepath"
                    ),
                    "params": {
                        "filepath": "absolute .blend path",
                        "overwrite": "boolean, default false",
                        "compress": "boolean, default false",
                    },
                },
                "load_blend_file": {
                    "behavior": (
                        "Replace the current Blender state from a .blend file; "
                        "embedded scripts are not executed"
                    ),
                    "params": {
                        "filepath": "absolute existing .blend path",
                        "load_ui": "boolean, default true",
                    },
                },
                "set_view": {
                    "behavior": (
                        "Temporarily switch each target window to Object Mode, "
                        "then restore its previous mode and enabled Dyntopo"
                    ),
                    "params": {
                        "view": (
                            "FRONT | BACK | LEFT | RIGHT | TOP | BOTTOM | "
                            "CAMERA"
                        ),
                        "window_index": "integer, optional",
                        "area_index": "integer, optional",
                        "all_viewports": "boolean, default false",
                        "align_active": "boolean, default false",
                        "frame": (
                            "KEEP | SELECTED | ALL, default ALL; "
                            "axis views only"
                        ),
                        "projection": (
                            "PERSPECTIVE | ORTHOGRAPHIC, default "
                            "PERSPECTIVE; axis views only"
                        ),
                    },
                },
                "focus_viewport_roi": {
                    "behavior": (
                        "In an orthographic VIEW_3D region, center and zoom "
                        "to one screenshot-space ROI, then return exact "
                        "before/focused viewport snapshots and the applied "
                        "old-image to focused-image affine transform"
                    ),
                    "params": {
                        "roi": (
                            "object with x_min, y_min, x_max, y_max in "
                            "top-left screenshot pixels"
                        ),
                        "image_width": "positive integer",
                        "image_height": "positive integer",
                        "margin_ratio": "number in [0, 2], default 0.2",
                        "maximum_zoom_factor": (
                            "number in [1, 100], default 12"
                        ),
                        "window_index": "integer, optional",
                        "area_index": "integer, optional",
                    },
                },
                "restore_viewport_state": {
                    "behavior": (
                        "Apply one versioned viewport snapshot returned by "
                        "focus_viewport_roi and verify the resulting state"
                    ),
                    "params": {
                        "snapshot": "viewport-state/v1 object, required",
                        "require_region_match": (
                            "boolean, default true"
                        ),
                    },
                },
                "get_screenshot": {
                    "params": {
                        "output": "base64 | file, default base64",
                        "filepath": "absolute PNG path, required for file",
                        "window_index": "integer, optional",
                        "area_index": "VIEW_3D area integer, optional",
                        "redraw": "boolean, default true",
                    },
                },
                "enter_sculpt_mode": {
                    "behavior": (
                        "Enter Sculpt Mode and optionally hide the Asset "
                        "Shelf, Toolbar, Sidebar, main Header, Tool Settings, grids, "
                        "general info, statistics, and Navigate gizmo "
                        "only after the final Sculpting Screen, Regions, "
                        "and delayed UI readbacks are stable"
                    ),
                    "params": {
                        "window_index": "integer, optional",
                        "area_index": "integer, optional",
                        "hide_viewport_ui": "boolean, default false",
                    },
                },
                "restore_sculpt_viewport_ui": {
                    "behavior": (
                        "Restore only the allowlisted VIEW_3D UI values "
                        "captured by enter_sculpt_mode, with delayed "
                        "verification and bounded retry"
                    ),
                    "params": {
                        "snapshot": (
                            "sculpt-viewport-ui snapshot object returned "
                            "by enter_sculpt_mode, required"
                        ),
                    },
                },
                "activate_lasso_face_set_tool": {
                    "behavior": (
                        "Activate Blender's interactive Lasso Face Set tool "
                        "in an existing Sculpt Mode VIEW_3D context"
                    ),
                    "params": {
                        "window_index": "integer, optional",
                        "area_index": "integer, optional",
                    },
                },
                "sculpt_face_set_lasso": {
                    "behavior": (
                        "Replay a validated VIEW_3D region-local mouse path "
                        "with bpy.ops.sculpt.face_set_lasso_gesture"
                    ),
                    "params": {
                        "path": (
                            "OperatorMousePath array with loc [x, y] and "
                            "optional time, required"
                        ),
                        "use_front_faces_only": (
                            "boolean, default false"
                        ),
                        "show_face_sets": (
                            "boolean, default false; set the target "
                            "VIEW_3D Sculpt Face Sets overlay after drawing"
                        ),
                        "window_index": "integer, optional",
                        "area_index": "integer, optional",
                        "maximum_points": MAX_FACE_SET_LASSO_POINTS,
                    },
                },
                "sculpt_face_set_lasso_batch": {
                    "behavior": (
                        "Apply labeled Lasso Face Set paths as one "
                        "transaction, detect enclosed unchanged face "
                        "islands, and optionally roll back incomplete "
                        "coverage"
                    ),
                    "params": {
                        "paths": (
                            "array of {label, path} objects, required"
                        ),
                        "padding_pixels": (
                            "integer 1..192 used for the attempt trace"
                        ),
                        "rollback_on_residual": (
                            "boolean, default true"
                        ),
                        "use_front_faces_only": (
                            "boolean, default false"
                        ),
                        "show_face_sets": (
                            "boolean, default false"
                        ),
                        "window_index": "integer, optional",
                        "area_index": "integer, optional",
                        "maximum_paths": (
                            MAX_FACE_SET_LASSO_BATCH_PATHS
                        ),
                        "maximum_points_per_path": (
                            MAX_FACE_SET_LASSO_POINTS
                        ),
                    },
                },
                "activate_sculpt_brush": {
                    "params": {
                        "brush": "Sculpt brush asset name",
                        "window_index": "integer, optional",
                        "area_index": "integer, optional",
                    },
                },
                "set_sculpt_settings": {
                    "behavior": (
                        "Apply brush activation, Unified/Pressure toggles, "
                        "active Brush properties, and Dyntopo as one "
                        "compensating transaction"
                    ),
                    "params": {
                        "sculpt_brush": "Sculpt brush asset name",
                        "brush": "set_sculpt_brush parameter object",
                        "use_unified_size": "boolean",
                        "use_unified_strength": "boolean",
                        "use_size_pressure": "boolean",
                        "use_strength_pressure": "boolean",
                        "dyntopo": "{enabled, detail_size} object",
                    },
                },
                "set_sculpt_brush": {
                    "behavior": (
                        "Update only the active Sculpt Brush datablock; "
                        "validate direction against that brush's "
                        "direction_values; accept Pose-only settings when "
                        "the active brush type is POSE; omitted settings "
                        "remain unchanged; Unified Paint Settings remain "
                        "unchanged"
                    ),
                    "params": {
                        "size": "integer in [1, 10000], optional",
                        "strength": "number in [0, 10], optional",
                        "direction": (
                            "string, optional; allowed values come from "
                            "get_state sculpt.brush.direction_values"
                        ),
                        "stroke_method": (
                            "DOTS | DRAG_DOT | SPACE | AIRBRUSH | "
                            "ANCHORED | LINE | CURVE, optional"
                        ),
                        "spacing": "integer in [1, 1000], optional",
                        "use_space_attenuation": "boolean, optional",
                        "auto_smooth_factor": (
                            "number in [0, 1], optional"
                        ),
                        "deformation_target": (
                            "GEOMETRY | CLOTH_SIM, Pose-only optional; "
                            "Web request default GEOMETRY"
                        ),
                        "rotation_origins": (
                            "TOPOLOGY | FACE_SETS | FACE_SETS_FK, "
                            "Pose-only optional; Web request default "
                            "FACE_SETS"
                        ),
                        "pose_origin_offset": (
                            "number in [0, 2], Pose-only optional; "
                            "Web request default 0"
                        ),
                        "smooth_iterations": (
                            "integer in [0, 100], Pose-only optional; "
                            "Web request default 100"
                        ),
                        "pose_ik_segments": (
                            "integer in [1, 20], Pose-only optional; "
                            "Web request default 1"
                        ),
                        "connected_only": (
                            "boolean, Pose-only optional; Web request "
                            "default false"
                        ),
                        "max_element_distance": (
                            "number in [0, 10], Pose-only optional; "
                            "Web request default 0.1"
                        ),
                        "window_index": "integer, optional",
                        "area_index": "integer, optional",
                    },
                },
                "set_use_unified_size": {
                    "behavior": (
                        "Enable or disable Sculpt Use Unified Size using "
                        "the version-appropriate settings owner"
                    ),
                    "params": {
                        "enabled": "boolean",
                    },
                },
                "set_use_unified_strength": {
                    "behavior": (
                        "Enable or disable Sculpt Use Unified Strength using "
                        "the version-appropriate settings owner"
                    ),
                    "params": {
                        "enabled": "boolean",
                    },
                },
                "set_use_size_pressure": {
                    "behavior": (
                        "Enable or disable Use Size Pressure on the active "
                        "Sculpt brush; missing Sculpt context and brush are "
                        "prepared automatically when possible"
                    ),
                    "params": {
                        "enabled": "boolean",
                        "window_index": "integer, optional",
                        "area_index": "integer, optional",
                    },
                },
                "set_use_strength_pressure": {
                    "behavior": (
                        "Enable or disable Use Strength Pressure on the "
                        "active Sculpt brush; missing Sculpt context and "
                        "brush are prepared automatically when possible"
                    ),
                    "params": {
                        "enabled": "boolean",
                        "window_index": "integer, optional",
                        "area_index": "integer, optional",
                    },
                },
                "set_dyntopo": {
                    "params": {
                        "enabled": "boolean",
                        "detail_size": "number in [0.5, 40], optional",
                        "window_index": "integer, optional",
                        "area_index": "integer, optional",
                    },
                },
                "sculpt_brush_stroke": {
                    "params": {
                        "operation_id": (
                            "printable string up to 256 chars, optional; "
                            "persisted in the .blend for idempotent replay"
                        ),
                        "stroke": (
                            "OperatorStrokeElement array from "
                            "segment_with_sam3, required"
                        ),
                        "mode": "NORMAL | INVERT, default NORMAL",
                        "brush_toggle": (
                            "None | SMOOTH | ERASE | MASK, default None"
                        ),
                        "pen_flip": "boolean, default false",
                        "execution_mode": (
                            "AUTO | DIRECT_DABS | PAINT_CURVE | "
                            "ANCHORED_DRAG, default AUTO"
                        ),
                        "location_mode": (
                            "AUTO | SURFACE_RAYCAST | ANCHORED_DRAG, "
                            "default AUTO"
                        ),
                        "ignore_background_click": (
                            "must be true, default true"
                        ),
                        "window_index": "integer, optional",
                        "area_index": "integer, optional",
                        "maximum_elements": MAX_SCULPT_STROKE_ELEMENTS,
                    },
                },
            },
            "aliases": dict(self._aliases),
        }

    def get_state(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return a bounded snapshot of Blender's current state."""
        _reject_unknown(
            params,
            {"include_objects", "object_limit", "include_viewports"},
        )
        include_objects = _boolean(params, "include_objects", default=False)
        include_viewports = _boolean(
            params,
            "include_viewports",
            default=True,
        )
        object_limit = _integer(
            params,
            "object_limit",
            default=200,
            minimum=1,
            maximum=10_000,
        )

        context = bpy.context
        scene = context.scene
        view_layer = context.view_layer
        active_object = (
            view_layer.objects.active if view_layer is not None else None
        )
        selected_objects = list(getattr(context, "selected_objects", ()) or ())

        state: dict[str, Any] = {
            "blender": {
                "version": bpy.app.version_string,
                "version_tuple": list(bpy.app.version),
                "version_cycle": bpy.app.version_cycle,
                "binary_path": bpy.app.binary_path,
                "background": bpy.app.background,
                "build_branch": _text(bpy.app.build_branch),
                "build_commit_date": _text(bpy.app.build_commit_date),
                "python_version": sys.version.split()[0],
            },
            "file": {
                "path": bpy.data.filepath,
                "is_saved": bool(bpy.data.filepath),
                "is_dirty": bpy.data.is_dirty,
            },
            "context": {
                "mode": getattr(context, "mode", "UNKNOWN"),
                "workspace": _name(getattr(context, "workspace", None)),
                "screen": _name(getattr(context, "screen", None)),
                "view_layer": _name(view_layer),
                "active_collection": _name(
                    getattr(context, "collection", None)
                ),
                "active_layer_collection": _name(
                    getattr(view_layer, "active_layer_collection", None)
                    if view_layer is not None
                    else None
                ),
            },
            "scene": self._scene_summary(scene),
            "active_object": self._object_summary(active_object, view_layer),
            "selected_objects": [
                self._object_summary(item, view_layer)
                for item in selected_objects
            ],
            "sculpt": self._sculpt_state_summary(context, active_object),
            "data_counts": {
                "scenes": len(bpy.data.scenes),
                "objects": len(bpy.data.objects),
                "collections": len(bpy.data.collections),
                "meshes": len(bpy.data.meshes),
                "curves": len(bpy.data.curves),
                "materials": len(bpy.data.materials),
                "images": len(bpy.data.images),
                "cameras": len(bpy.data.cameras),
                "lights": len(bpy.data.lights),
            },
            "available_scenes": [item.name for item in bpy.data.scenes],
        }

        if include_viewports:
            state["viewports"] = self._viewport_summaries()

        if include_objects:
            scene_objects = list(scene.objects) if scene is not None else []
            state["objects"] = [
                self._object_summary(item, view_layer)
                for item in scene_objects[:object_limit]
            ]
            state["objects_truncated"] = len(scene_objects) > object_limit

        return state

    def save_blend_file(self, params: dict[str, Any]) -> dict[str, Any]:
        """Save the current Blender state as a .blend snapshot."""
        _reject_unknown(params, {"filepath", "overwrite", "compress"})
        filepath = _absolute_blend_path(
            _required_string(params, "filepath")
        )
        overwrite = _boolean(params, "overwrite", default=False)
        compress = _boolean(params, "compress", default=False)

        existed = filepath.exists()
        if existed and not filepath.is_file():
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {"reason": "filepath must identify a regular file"},
            )
        if existed and not overwrite:
            raise JsonRpcError(
                -32030,
                "The Blender snapshot already exists",
                {
                    "filepath": str(filepath),
                    "reason": "Set overwrite to true to replace it",
                },
            )

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise JsonRpcError(
                -32030,
                "Cannot create the Blender snapshot directory",
                {
                    "filepath": str(filepath),
                    "reason": str(error),
                },
            ) from error

        current_filepath = bpy.data.filepath
        self._call_operator(
            bpy.ops.wm.save_as_mainfile,
            {
                "filepath": str(filepath),
                "check_existing": False,
                "compress": compress,
                "relative_remap": True,
                "copy": True,
            },
            error_message="Cannot save the Blender snapshot",
            error_code=-32030,
        )
        try:
            size_bytes = filepath.stat().st_size
        except OSError as error:
            raise JsonRpcError(
                -32030,
                "Blender did not create a readable snapshot",
                {
                    "filepath": str(filepath),
                    "reason": str(error),
                },
            ) from error

        return {
            "filepath": str(filepath),
            "size_bytes": size_bytes,
            "overwritten": existed,
            "compressed": compress,
            "saved_as_copy": True,
            "current_file_path": bpy.data.filepath,
            "current_file_path_unchanged": (
                bpy.data.filepath == current_filepath
            ),
        }

    def load_blend_file(self, params: dict[str, Any]) -> dict[str, Any]:
        """Replace the current Blender state from a local .blend file."""
        _reject_unknown(params, {"filepath", "load_ui"})
        filepath = _absolute_blend_path(
            _required_string(params, "filepath")
        )
        _validate_blend_file(filepath)
        load_ui = _boolean(params, "load_ui", default=True)

        previous = {
            "filepath": bpy.data.filepath,
            "is_dirty": bpy.data.is_dirty,
            "scene": _name(getattr(bpy.context, "scene", None)),
            "mode": getattr(bpy.context, "mode", "UNKNOWN"),
            "workspace": _name(getattr(bpy.context, "workspace", None)),
        }
        # Main-file loading invalidates every cached Blender RNA pointer.
        self._sculpt_replay_paint_curve = None
        self._call_operator(
            bpy.ops.wm.open_mainfile,
            {
                "filepath": str(filepath),
                "load_ui": load_ui,
                "use_scripts": False,
                "display_file_selector": False,
            },
            error_message="Cannot load the Blender snapshot",
            error_code=-32031,
        )

        loaded_filepath = bpy.data.filepath
        if not loaded_filepath or not _paths_equal(
            Path(loaded_filepath),
            filepath,
        ):
            raise JsonRpcError(
                -32031,
                "Blender did not load the requested snapshot",
                {
                    "requested_filepath": str(filepath),
                    "actual_filepath": loaded_filepath,
                },
            )

        context = bpy.context
        return {
            "filepath": loaded_filepath,
            "loaded": True,
            "load_ui": load_ui,
            "scripts_executed": False,
            "previous": previous,
            "restored": {
                "is_dirty": bpy.data.is_dirty,
                "scene": _name(getattr(context, "scene", None)),
                "mode": getattr(context, "mode", "UNKNOWN"),
                "workspace": _name(getattr(context, "workspace", None)),
                "active_object": _name(
                    getattr(context, "active_object", None)
                ),
            },
        }

    def set_view(self, params: dict[str, Any]) -> dict[str, Any]:
        """Switch one or more 3D viewports to a standard view."""
        _reject_unknown(
            params,
            {
                "view",
                "window_index",
                "area_index",
                "all_viewports",
                "align_active",
                "frame",
                "projection",
            },
        )
        view = _required_string(params, "view").upper()
        if view not in _AXIS_VIEWS | {"CAMERA"}:
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {
                    "reason": "Unsupported standard view",
                    "supported": sorted(_AXIS_VIEWS | {"CAMERA"}),
                },
            )
        window_index = _optional_integer(params, "window_index", minimum=0)
        area_index = _optional_integer(params, "area_index", minimum=0)
        all_viewports = _boolean(
            params,
            "all_viewports",
            default=False,
        )
        align_active = _boolean(params, "align_active", default=False)
        frame = _string(params, "frame", default="ALL").upper()
        if frame not in {"KEEP", "SELECTED", "ALL"}:
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {"reason": "frame must be KEEP, SELECTED, or ALL"},
            )
        projection = _string(
            params,
            "projection",
            default="PERSPECTIVE",
        ).upper()
        projection = _VIEW_PROJECTION_ALIASES.get(projection, projection)
        if projection not in _VIEW_PROJECTIONS:
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {
                    "reason": (
                        "projection must be PERSPECTIVE or ORTHOGRAPHIC"
                    ),
                    "supported": sorted(_VIEW_PROJECTIONS),
                },
            )

        targets = self._resolve_viewports(
            window_index=window_index,
            area_index=area_index,
            all_viewports=all_viewports,
        )
        smooth_view = getattr(bpy.context.preferences.view, "smooth_view", None)
        if smooth_view is not None:
            bpy.context.preferences.view.smooth_view = 0
        changed: list[dict[str, Any]] = []
        try:
            with self._temporary_object_mode(targets):
                for target in targets:
                    with bpy.context.temp_override(
                        window=target["window"],
                        area=target["area"],
                        region=target["region"],
                    ):
                        if view == "CAMERA":
                            self._call_operator(
                                bpy.ops.view3d.view_camera,
                                {},
                                error_message="Camera view is unavailable",
                            )
                        else:
                            self._call_operator(
                                bpy.ops.view3d.view_axis,
                                {
                                    "type": view,
                                    "align_active": align_active,
                                    "relative": False,
                                },
                                error_message="Standard view is unavailable",
                            )
                            self._set_view_projection(
                                target,
                                _VIEW_PROJECTIONS[projection],
                            )

                            if frame == "SELECTED":
                                self._call_operator(
                                    bpy.ops.view3d.view_selected,
                                    {"use_all_regions": False},
                                    error_message=(
                                        "Cannot frame the current selection"
                                    ),
                                )
                            elif frame == "ALL":
                                self._call_operator(
                                    bpy.ops.view3d.view_all,
                                    {"center": False},
                                    error_message="Cannot frame all objects",
                                )

                    target["area"].tag_redraw()
                    changed.append(self._viewport_summary(target))
        finally:
            if smooth_view is not None:
                bpy.context.preferences.view.smooth_view = smooth_view

        return {
            "view": view,
            "frame": frame,
            "frame_applied": view != "CAMERA" and frame != "KEEP",
            "projection": projection,
            "projection_applied": view != "CAMERA",
            "mode_restored": True,
            "changed_count": len(changed),
            "viewports": changed,
        }

    def focus_viewport_roi(self, params: dict[str, Any]) -> dict[str, Any]:
        """Center and zoom one orthographic viewport to a screenshot ROI."""
        _reject_unknown(
            params,
            {
                "roi",
                "image_width",
                "image_height",
                "margin_ratio",
                "maximum_zoom_factor",
                "window_index",
                "area_index",
            },
        )
        if "image_width" not in params or "image_height" not in params:
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {"reason": "image_width and image_height are required"},
            )
        image_width = _integer(
            params,
            "image_width",
            default=0,
            minimum=2,
            maximum=100_000,
        )
        image_height = _integer(
            params,
            "image_height",
            default=0,
            minimum=2,
            maximum=100_000,
        )
        margin_ratio = _optional_number(
            params,
            "margin_ratio",
            minimum=0.0,
            maximum=2.0,
        )
        if margin_ratio is None:
            margin_ratio = 0.2
        maximum_zoom_factor = _optional_number(
            params,
            "maximum_zoom_factor",
            minimum=1.0,
            maximum=100.0,
        )
        if maximum_zoom_factor is None:
            maximum_zoom_factor = 12.0
        roi = _parse_focus_roi(
            params.get("roi"),
            image_width=image_width,
            image_height=image_height,
        )
        window_index = _optional_integer(
            params,
            "window_index",
            minimum=0,
        )
        area_index = _optional_integer(
            params,
            "area_index",
            minimum=0,
        )
        target = self._resolve_viewports(
            window_index=window_index,
            area_index=area_index,
            all_viewports=False,
        )[0]
        region = target["region"]
        region_3d = target["space"].region_3d
        if str(region_3d.view_perspective) != "ORTHO":
            raise JsonRpcError(
                -32013,
                "ROI focus requires Orthographic Projection",
                {
                    "actual_projection": str(
                        region_3d.view_perspective
                    )
                },
            )
        scale_x = image_width / region.width
        scale_y = image_height / region.height
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (scale_x, scale_y)
        ):
            raise JsonRpcError(
                -32013,
                "Screenshot and viewport dimensions are inconsistent",
            )

        expanded_roi = _expanded_focus_roi(
            roi,
            image_width=image_width,
            image_height=image_height,
            margin_ratio=margin_ratio,
            maximum_zoom_factor=maximum_zoom_factor,
        )
        border = _image_roi_to_region_border(
            expanded_roi,
            image_width=image_width,
            image_height=image_height,
            region_width=region.width,
            region_height=region.height,
        )
        before = _viewport_state_snapshot(
            target,
            image_width=image_width,
            image_height=image_height,
        )
        sample_points = _viewport_world_samples(
            region=region,
            region_3d=region_3d,
            image_width=image_width,
            image_height=image_height,
        )
        smooth_view = getattr(
            bpy.context.preferences.view,
            "smooth_view",
            None,
        )
        if smooth_view is not None:
            bpy.context.preferences.view.smooth_view = 0
        try:
            with bpy.context.temp_override(
                window=target["window"],
                area=target["area"],
                region=region,
            ):
                self._call_operator(
                    bpy.ops.view3d.zoom_border,
                    {
                        **border,
                        "wait_for_input": False,
                        "zoom_out": False,
                    },
                    error_message="Cannot focus the 3D viewport ROI",
                    error_code=-32013,
                )
            region_3d.update()
            target["area"].tag_redraw()
            if str(region_3d.view_perspective) != "ORTHO":
                raise JsonRpcError(
                    -32013,
                    "ROI focus changed the viewport projection",
                )
            focused = _viewport_state_snapshot(
                target,
                image_width=image_width,
                image_height=image_height,
            )
            before_rotation = before["view_rotation"]
            focused_rotation = focused["view_rotation"]
            direct_rotation_error = max(
                abs(left - right)
                for left, right in zip(
                    before_rotation,
                    focused_rotation,
                    strict=True,
                )
            )
            negated_rotation_error = max(
                abs(left + right)
                for left, right in zip(
                    before_rotation,
                    focused_rotation,
                    strict=True,
                )
            )
            if min(
                direct_rotation_error,
                negated_rotation_error,
            ) > 1e-5:
                raise JsonRpcError(
                    -32013,
                    "ROI focus changed the viewport rotation",
                )
            affine = _viewport_image_affine(
                region=region,
                region_3d=region_3d,
                sample_points=sample_points,
                image_width=image_width,
                image_height=image_height,
            )
        except (JsonRpcError, RuntimeError, TypeError, ValueError) as error:
            try:
                _apply_viewport_state_snapshot(
                    target,
                    before,
                    require_region_match=False,
                )
            except (JsonRpcError, RuntimeError, TypeError, ValueError):
                pass
            if isinstance(error, JsonRpcError):
                raise
            raise JsonRpcError(
                -32013,
                "Cannot focus the 3D viewport ROI",
                {"reason": str(error)},
            ) from error
        finally:
            if smooth_view is not None:
                bpy.context.preferences.view.smooth_view = smooth_view

        transformed_roi = _transform_roi(roi, affine["matrix"])
        return {
            "schema_version": "viewport-roi-focus/v1",
            "projection": "ORTHOGRAPHIC",
            "roi": roi,
            "expanded_roi": expanded_roi,
            "region_border": border,
            "margin_ratio": margin_ratio,
            "maximum_zoom_factor": maximum_zoom_factor,
            "before": before,
            "focused": focused,
            "image_transform": {
                **affine,
                "source": {
                    "width": image_width,
                    "height": image_height,
                    "origin": "TOP_LEFT",
                },
                "target": {
                    "width": image_width,
                    "height": image_height,
                    "origin": "TOP_LEFT",
                },
                "transformed_roi": transformed_roi,
            },
        }

    def restore_viewport_state(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply and verify a viewport snapshot returned by ROI focus."""
        _reject_unknown(params, {"snapshot", "require_region_match"})
        snapshot = params.get("snapshot")
        if not isinstance(snapshot, dict):
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {"reason": "snapshot must be an object"},
            )
        require_region_match = _boolean(
            params,
            "require_region_match",
            default=True,
        )
        parsed = _parse_viewport_state_snapshot(snapshot)
        target = self._resolve_viewports(
            window_index=parsed["window_index"],
            area_index=parsed["area_index"],
            all_viewports=False,
        )[0]
        _apply_viewport_state_snapshot(
            target,
            parsed,
            require_region_match=require_region_match,
        )
        actual = _viewport_state_snapshot(
            target,
            image_width=parsed["image"]["width"],
            image_height=parsed["image"]["height"],
        )
        mismatches = _viewport_state_mismatches(parsed, actual)
        if mismatches:
            raise JsonRpcError(
                -32014,
                "Blender did not restore the viewport state",
                {"mismatches": mismatches},
            )
        return {
            "schema_version": "viewport-state-restore/v1",
            "status": "restored",
            "snapshot": actual,
        }

    def get_screenshot(self, params: dict[str, Any]) -> dict[str, Any]:
        """Capture and crop a VIEW_3D WINDOW region as PNG."""
        _reject_unknown(
            params,
            {
                "output",
                "filepath",
                "window_index",
                "area_index",
                "redraw",
            },
        )

        output = _string(params, "output", default="base64").lower()
        if output not in {"base64", "file"}:
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {"reason": "output must be base64 or file"},
            )
        window_index = _optional_integer(params, "window_index", minimum=0)
        area_index = _optional_integer(params, "area_index", minimum=0)
        redraw = _boolean(params, "redraw", default=True)
        target = self._resolve_viewports(
            window_index=window_index,
            area_index=area_index,
            all_viewports=False,
        )[0]
        window = target["window"]
        region = target["region"]

        if output == "file":
            filepath_value = _required_string(params, "filepath")
            filepath = _absolute_png_path(filepath_value)
            filepath.parent.mkdir(parents=True, exist_ok=True)
        else:
            if "filepath" in params:
                raise JsonRpcError(
                    -32602,
                    "Invalid params",
                    {"reason": "filepath is only valid when output is file"},
                )
            filepath = None

        descriptor, temporary_path = tempfile.mkstemp(
            prefix="blender_rpc_window_",
            suffix=".png",
        )
        os.close(descriptor)
        capture_path = Path(temporary_path)

        remove_cropped_after_read = output == "base64"
        if remove_cropped_after_read:
            descriptor, temporary_path = tempfile.mkstemp(
                prefix="blender_rpc_view3d_",
                suffix=".png",
            )
            os.close(descriptor)
            cropped_path = Path(temporary_path)
        else:
            assert filepath is not None
            cropped_path = filepath

        try:
            with bpy.context.temp_override(window=window):
                if redraw and bpy.ops.wm.redraw_timer.poll():
                    bpy.ops.wm.redraw_timer(
                        type="DRAW_WIN_SWAP",
                        iterations=1,
                    )
                self._call_operator(
                    bpy.ops.screen.screenshot,
                    {
                        "filepath": str(capture_path),
                        "check_existing": False,
                    },
                    error_message="Blender window screenshot is unavailable",
                )

            if not capture_path.is_file() or capture_path.stat().st_size == 0:
                raise JsonRpcError(
                    -32012,
                    "Screenshot capture did not produce a PNG file",
                )

            try:
                source_png = capture_path.read_bytes()
                source_width, source_height = png_dimensions(source_png)
                areas = list(window.screen.areas)
                coordinate_max_x = max(
                    area.x + area.width for area in areas
                )
                coordinate_max_y = max(
                    area.y + area.height for area in areas
                )
                crop_geometry = calculate_region_crop(
                    image_width=source_width,
                    image_height=source_height,
                    window_width=window.width,
                    window_height=window.height,
                    coordinate_max_x=coordinate_max_x,
                    coordinate_max_y=coordinate_max_y,
                    region_x=region.x,
                    region_y=region.y,
                    region_width=region.width,
                    region_height=region.height,
                )
                image_load_api = _crop_png_with_imbuf(
                    source_path=capture_path,
                    source_png=source_png,
                    output_path=cropped_path,
                    source_width=source_width,
                    source_height=source_height,
                    geometry=crop_geometry,
                )
                if (
                    not cropped_path.is_file()
                    or cropped_path.stat().st_size == 0
                ):
                    raise PngCropError(
                        "ImBuf did not produce a cropped PNG file"
                    )
                size_bytes = cropped_path.stat().st_size
                if output == "base64":
                    image_bytes = cropped_path.read_bytes()
                    png_header = image_bytes
                else:
                    image_bytes = b""
                    with cropped_path.open("rb") as image_file:
                        png_header = image_file.read(24)
                width, height = png_dimensions(png_header)
                if (width, height) != (
                    crop_geometry.width,
                    crop_geometry.height,
                ):
                    raise PngCropError(
                        "ImBuf output dimensions do not match the crop"
                    )
            except (OSError, PngCropError) as error:
                raise JsonRpcError(
                    -32012,
                    "Cannot crop the VIEW_3D WINDOW region",
                    {"reason": str(error)},
                ) from error

            result: dict[str, Any] = {
                "mime_type": "image/png",
                "width": width,
                "height": height,
                "size_bytes": size_bytes,
                "capture_scope": "VIEW_3D_WINDOW",
                "image_backend": "imbuf",
                "image_load_api": image_load_api,
                "window_index": target["window_index"],
                "area_index": target["area_index"],
                "source_window": {
                    "width": source_width,
                    "height": source_height,
                },
                "region": {
                    "x": region.x,
                    "y": region.y,
                    "width": region.width,
                    "height": region.height,
                },
                "crop_box": {
                    "left": crop_geometry.left,
                    "top": crop_geometry.top,
                    "right": crop_geometry.right,
                    "bottom": crop_geometry.bottom,
                },
                "coordinate_scale": {
                    "x": crop_geometry.scale_x,
                    "y": crop_geometry.scale_y,
                },
                "window_pixel_scale": {
                    "x": source_width / window.width,
                    "y": source_height / window.height,
                },
            }
            if output == "base64":
                result.update(
                    {
                        "encoding": "base64",
                        "data": base64.b64encode(image_bytes).decode("ascii"),
                    }
                )
            else:
                assert filepath is not None
                result.update(
                    {
                        "encoding": "file",
                        "filepath": str(filepath),
                    }
                )
            return result
        finally:
            try:
                capture_path.unlink(missing_ok=True)
            except OSError:
                pass
            if remove_cropped_after_read:
                try:
                    cropped_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def enter_sculpt_mode(
        self,
        params: dict[str, Any],
    ) -> DeferredMainThreadCall:
        """Schedule a stable Sculpting workspace and Sculpt Mode switch."""
        _reject_unknown(
            params,
            {"window_index", "area_index", "hide_viewport_ui"},
        )
        window_index = _optional_integer(params, "window_index", minimum=0)
        area_index = _optional_integer(params, "area_index", minimum=0)
        hide_viewport_ui = _boolean(
            params,
            "hide_viewport_ui",
            default=False,
        )
        if bpy.app.background:
            raise JsonRpcError(
                -32010,
                "3D viewports are unavailable in background mode",
            )
        window, resolved_window_index = self._resolve_window(window_index)
        view_layer = window.view_layer
        active_object = (
            view_layer.objects.active if view_layer is not None else None
        )
        if active_object is None or active_object.type != "MESH":
            raise JsonRpcError(
                -32020,
                "A mesh active object is required for Sculpt Mode",
                {
                    "active_object": _name(active_object),
                    "active_object_type": getattr(
                        active_object,
                        "type",
                        None,
                    ),
                },
            )

        workspace = bpy.data.workspaces.get("Sculpting")
        if workspace is None:
            raise JsonRpcError(
                -32021,
                "The Sculpting workspace is unavailable",
            )
        return DeferredMainThreadCall(
            steps=self._enter_sculpt_mode_transaction(
                window=window,
                window_index=resolved_window_index,
                area_index=area_index,
                workspace=workspace,
                hide_viewport_ui=hide_viewport_ui,
            ),
            label="enter_sculpt_mode",
        )

    def restore_sculpt_viewport_ui(
        self,
        params: dict[str, Any],
    ) -> DeferredMainThreadCall:
        """Schedule a verified restoration of one viewport UI snapshot."""
        _reject_unknown(params, {"snapshot"})
        if "snapshot" not in params:
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {"reason": "snapshot is required"},
            )
        snapshot = parse_sculpt_viewport_ui_snapshot(params["snapshot"])
        if bpy.app.background:
            raise JsonRpcError(
                -32010,
                "3D viewports are unavailable in background mode",
            )
        window, resolved_window_index = self._resolve_window(
            snapshot["window_index"]
        )
        return DeferredMainThreadCall(
            steps=self._restore_sculpt_viewport_ui_transaction(
                window=window,
                window_index=resolved_window_index,
                snapshot=snapshot,
            ),
            label="restore_sculpt_viewport_ui",
        )

    def activate_lasso_face_set_tool(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Activate the interactive Lasso Face Set Sculpt tool."""
        _reject_unknown(
            params,
            {"window_index", "area_index"},
        )
        target = self._resolve_viewports(
            window_index=_optional_integer(
                params,
                "window_index",
                minimum=0,
            ),
            area_index=_optional_integer(params, "area_index", minimum=0),
            all_viewports=False,
        )[0]

        with bpy.context.temp_override(
            window=target["window"],
            area=target["area"],
            region=target["region"],
        ):
            sculpt_object, _, _ = self._require_sculpt_context()
            tool_operator = self._operator_or_none("wm", "tool_set_by_id")
            if tool_operator is None:
                raise JsonRpcError(
                    -32022,
                    "Sculpt tool activation is unavailable",
                    {"tool_id": _LASSO_FACE_SET_TOOL_ID},
                )
            operator_result = self._call_operator(
                tool_operator,
                {
                    "name": _LASSO_FACE_SET_TOOL_ID,
                    "space_type": "VIEW_3D",
                },
                error_message="Cannot activate the Lasso Face Set tool",
                error_code=-32022,
            )

            try:
                active_tool = (
                    bpy.context.workspace.tools.from_space_view3d_mode(
                        bpy.context.mode,
                        create=False,
                    )
                )
                if active_tool is not None:
                    active_tool.refresh_from_context()
                active_tool_id = getattr(active_tool, "idname", None)
            except (AttributeError, RuntimeError, TypeError) as error:
                raise JsonRpcError(
                    -32022,
                    "Cannot verify the active Sculpt tool",
                    {
                        "tool_id": _LASSO_FACE_SET_TOOL_ID,
                        "reason": str(error),
                    },
                ) from error
            if active_tool_id != _LASSO_FACE_SET_TOOL_ID:
                raise JsonRpcError(
                    -32022,
                    "Blender did not activate the Lasso Face Set tool",
                    {
                        "expected_tool_id": _LASSO_FACE_SET_TOOL_ID,
                        "active_tool_id": active_tool_id,
                    },
                )
            active_summary = self._object_summary(
                sculpt_object,
                bpy.context.view_layer,
            )

        target["area"].tag_redraw()
        return {
            "mode": "SCULPT",
            "active_object": active_summary,
            "tool": {
                "idname": active_tool_id,
                "label": "Lasso Face Set",
                "gesture_operator": _LASSO_FACE_SET_OPERATOR_ID,
            },
            "operator_status": sorted(
                str(item) for item in operator_result
            ),
            "viewport": self._viewport_summary(target),
        }

    def activate_sculpt_brush(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Activate a named built-in Sculpt brush."""
        _reject_unknown(
            params,
            {"brush", "window_index", "area_index"},
        )
        brush_name = _required_string(params, "brush")
        if len(brush_name) > 128 or any(ord(char) < 32 for char in brush_name):
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {"reason": "brush contains unsupported characters"},
            )
        target = self._resolve_viewports(
            window_index=_optional_integer(
                params,
                "window_index",
                minimum=0,
            ),
            area_index=_optional_integer(params, "area_index", minimum=0),
            all_viewports=False,
        )[0]

        asset_identifier: str | None = None
        source: str
        with bpy.context.temp_override(
            window=target["window"],
            area=target["area"],
            region=target["region"],
        ):
            _, sculpt_settings, _ = self._require_sculpt_context()
            asset_operator = self._operator_or_none(
                "brush",
                "asset_activate",
            )
            if asset_operator is not None:
                source = "ESSENTIALS"
                asset_identifier = (
                    f"{_ESSENTIAL_SCULPT_BRUSH_PREFIX}{brush_name}"
                )
                try:
                    if not asset_operator.poll():
                        raise JsonRpcError(
                            -32022,
                            "Sculpt brush activation is unavailable",
                            {"brush": brush_name, "retryable": True},
                        )
                    operator_result = asset_operator(
                        asset_library_type="ESSENTIALS",
                        relative_asset_identifier=asset_identifier,
                    )
                except JsonRpcError:
                    raise
                except (RuntimeError, TypeError, ValueError) as error:
                    raise JsonRpcError(
                        -32022,
                        "Cannot activate the Sculpt brush",
                        {
                            "brush": brush_name,
                            "reason": str(error),
                            "retryable": False,
                        },
                    ) from error

                active_brush = sculpt_settings.brush
                active_matches = (
                    active_brush is not None
                    and active_brush.name.casefold() == brush_name.casefold()
                )
                if "FINISHED" not in operator_result and not active_matches:
                    raise JsonRpcError(
                        -32022,
                        "Sculpt brush asset is not ready",
                        {
                            "brush": brush_name,
                            "operator_status": sorted(
                                str(item) for item in operator_result
                            ),
                            "retryable": True,
                        },
                    )
            else:
                source = "LEGACY"
                self._activate_legacy_sculpt_brush(brush_name)

            active_brush = sculpt_settings.brush
            if active_brush is None:
                raise JsonRpcError(
                    -32022,
                    "Blender has no active Sculpt brush",
                    {"brush": brush_name, "retryable": True},
                )
            brush_summary = self._brush_summary(
                active_brush,
                bpy.context.tool_settings,
            )

        target["area"].tag_redraw()
        return {
            "source": source,
            "requested_brush": brush_name,
            "asset_identifier": asset_identifier,
            "brush": brush_summary,
            "viewport": self._viewport_summary(target),
        }

    def set_sculpt_settings_transaction(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply the complete Sculpt settings bundle transactionally."""
        allowed = {
            "sculpt_brush",
            "brush",
            "use_unified_size",
            "use_unified_strength",
            "use_size_pressure",
            "use_strength_pressure",
            "dyntopo",
        }
        _reject_unknown(params, allowed)
        if set(params) != allowed:
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {
                    "reason": "Every transactional Sculpt setting is required",
                    "missing": sorted(allowed - set(params)),
                },
            )
        brush_name = _required_string(params, "sculpt_brush")
        brush_params = params.get("brush")
        dyntopo_params = params.get("dyntopo")
        if not isinstance(brush_params, dict) or not brush_params:
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {"reason": "brush must be a non-empty object"},
            )
        if not isinstance(dyntopo_params, dict):
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {"reason": "dyntopo must be an object"},
            )
        toggles = {
            name: _boolean(params, name, default=False)
            for name in (
                "use_unified_size",
                "use_unified_strength",
                "use_size_pressure",
                "use_strength_pressure",
            )
        }
        before = self._sculpt_state_summary(
            bpy.context,
            getattr(bpy.context.view_layer.objects, "active", None),
        )
        original_brush = before.get("brush")
        completed: list[str] = []
        responses: list[dict[str, Any]] = []
        target_brush_before: dict[str, Any] | None = None
        try:
            activation = self.activate_sculpt_brush({"brush": brush_name})
            completed.append("activate_sculpt_brush")
            responses.append(
                {"method": "activate_sculpt_brush", "result": activation}
            )
            activated_summary = activation.get("brush")
            if not isinstance(activated_summary, dict):
                raise JsonRpcError(
                    -32023,
                    "Activated Sculpt brush has no readable settings",
                )
            target_brush_before = dict(activated_summary)
            effective_brush_params = dict(brush_params)
            if activated_summary.get("sculpt_brush_type") != "POSE":
                for name in POSE_BRUSH_PARAMETERS:
                    effective_brush_params.pop(name, None)

            steps = (
                ("set_use_unified_size", {"enabled": toggles["use_unified_size"]}),
                (
                    "set_use_unified_strength",
                    {"enabled": toggles["use_unified_strength"]},
                ),
                (
                    "set_use_size_pressure",
                    {"enabled": toggles["use_size_pressure"]},
                ),
                (
                    "set_use_strength_pressure",
                    {"enabled": toggles["use_strength_pressure"]},
                ),
                ("set_sculpt_brush", effective_brush_params),
                ("set_dyntopo", dict(dyntopo_params)),
            )
            for method, step_params in steps:
                result = self._methods[method](step_params)
                completed.append(method)
                responses.append({"method": method, "result": result})
        except Exception as error:
            rollback_errors = self._rollback_sculpt_settings_transaction(
                before=before,
                original_brush=original_brush,
                target_brush_before=target_brush_before,
            )
            data = {
                "transaction": "sculpt-settings/v1",
                "completed_methods": completed,
                "rollback": {
                    "succeeded": not rollback_errors,
                    "errors": rollback_errors,
                },
            }
            if isinstance(error, JsonRpcError):
                if isinstance(error.data, dict):
                    data["cause"] = error.data
                raise JsonRpcError(error.code, error.message, data) from error
            raise JsonRpcError(
                -32023,
                "Cannot apply Sculpt settings transaction",
                {**data, "reason": str(error)},
            ) from error

        return {
            "transaction": "sculpt-settings/v1",
            "status": "committed",
            "completed_methods": completed,
            "steps": responses,
            "sculpt": self._sculpt_state_summary(
                bpy.context,
                getattr(bpy.context.view_layer.objects, "active", None),
            ),
        }

    def _rollback_sculpt_settings_transaction(
        self,
        *,
        before: dict[str, Any],
        original_brush: Any,
        target_brush_before: dict[str, Any] | None,
    ) -> list[str]:
        """Best-effort compensation for a failed settings transaction."""
        errors: list[str] = []

        def restore(label: str, callback: Callable[[], Any]) -> None:
            try:
                callback()
            except Exception as error:
                errors.append(f"{label}: {error}")

        if target_brush_before is not None:
            restore(
                "target brush",
                lambda: self.set_sculpt_brush(
                    self._brush_restore_params(target_brush_before)
                ),
            )
            for method, key in (
                ("set_use_size_pressure", "use_size_pressure"),
                ("set_use_strength_pressure", "use_strength_pressure"),
            ):
                value = target_brush_before.get(key)
                if isinstance(value, bool):
                    restore(
                        key,
                        lambda method=method, value=value: self._methods[
                            method
                        ]({"enabled": value}),
                    )

        unified_size = before.get("unified_size")
        if isinstance(unified_size, dict) and isinstance(
            unified_size.get("enabled"), bool
        ):
            restore(
                "unified size",
                lambda: self.set_use_unified_size(
                    {"enabled": unified_size["enabled"]}
                ),
            )
        unified_strength = before.get("unified_strength")
        if isinstance(unified_strength, dict) and isinstance(
            unified_strength.get("enabled"), bool
        ):
            restore(
                "unified strength",
                lambda: self.set_use_unified_strength(
                    {"enabled": unified_strength["enabled"]}
                ),
            )
        dyntopo = before.get("dyntopo")
        if isinstance(dyntopo, dict) and isinstance(
            dyntopo.get("enabled"), bool
        ):
            restore(
                "dyntopo",
                lambda: self.set_dyntopo(
                    {
                        "enabled": dyntopo["enabled"],
                        "detail_size": dyntopo.get("detail_size", 12.0),
                    }
                ),
            )
        if isinstance(original_brush, dict):
            name = original_brush.get("name")
            if isinstance(name, str) and name:
                restore(
                    "active brush",
                    lambda: self.activate_sculpt_brush({"brush": name}),
                )
        return errors

    @staticmethod
    def _brush_restore_params(summary: dict[str, Any]) -> dict[str, Any]:
        """Convert a brush summary back into bounded setter parameters."""
        params: dict[str, Any] = {
            "size": summary["size"],
            "strength": summary["strength"],
        }
        if summary.get("direction_supported") is True:
            params["direction"] = summary["direction"]
        for name in BRUSH_QUALITY_PARAMETERS:
            key = "spacing" if name == "spacing" else name
            value = summary.get(key)
            if value is not None:
                params[name] = value
        pose = summary.get("pose_settings")
        if isinstance(pose, dict):
            for name in POSE_BRUSH_PARAMETERS:
                value = pose.get(name)
                if value is not None:
                    params[name] = value
        return params

    def set_sculpt_brush(self, params: dict[str, Any]) -> dict[str, Any]:
        """Update bounded settings on the active Sculpt brush."""
        _reject_unknown(
            params,
            {
                "size",
                "strength",
                "direction",
                *BRUSH_QUALITY_PARAMETERS,
                *POSE_BRUSH_PARAMETERS,
                "window_index",
                "area_index",
            },
        )
        brush_parameters = {
            "size",
            "strength",
            "direction",
            *BRUSH_QUALITY_PARAMETERS,
            *POSE_BRUSH_PARAMETERS,
        }
        if not brush_parameters & params.keys():
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {
                    "reason": (
                        "At least one Brush setting is required"
                    )
                },
            )

        size = (
            _integer(
                params,
                "size",
                default=1,
                minimum=1,
                maximum=10_000,
            )
            if "size" in params
            else None
        )
        strength = _optional_number(
            params,
            "strength",
            minimum=0.0,
            maximum=10.0,
        )
        direction = (
            _required_string(params, "direction").upper()
            if "direction" in params
            else None
        )
        quality_updates = parse_brush_quality_parameters(params)
        pose_updates = parse_pose_brush_parameters(params)
        target = self._resolve_viewports(
            window_index=_optional_integer(
                params,
                "window_index",
                minimum=0,
            ),
            area_index=_optional_integer(params, "area_index", minimum=0),
            all_viewports=False,
        )[0]
        with bpy.context.temp_override(
            window=target["window"],
            area=target["area"],
            region=target["region"],
        ):
            _, _, brush = self._require_sculpt_context()
            if brush is None:
                raise JsonRpcError(
                    -32023,
                    "Blender has no active Sculpt brush",
                )
            brush_type = self._sculpt_brush_type(brush)
            if pose_updates and brush_type != "POSE":
                raise JsonRpcError(
                    -32602,
                    "Invalid params",
                    {
                        "reason": (
                            "Pose settings require the active Sculpt "
                            "brush type to be POSE"
                        ),
                        "brush": str(brush.name),
                        "sculpt_brush_type": brush_type or None,
                        "pose_parameters": sorted(pose_updates),
                    },
                )
            self._validate_pose_brush_runtime(brush, pose_updates)
            self._validate_brush_quality_runtime(brush, quality_updates)
            direction_values = self._sculpt_brush_direction_values(brush)
            if direction is not None and direction not in direction_values:
                raise JsonRpcError(
                    -32602,
                    "Invalid params",
                    {
                        "reason": (
                            "direction is not supported by the active "
                            "Sculpt brush"
                        ),
                        "brush": str(brush.name),
                        "direction": direction,
                        "supported_values": list(direction_values),
                    },
                )
            unified_settings, settings_owner = (
                self._find_sculpt_unified_settings(
                    bpy.context.tool_settings,
                )
            )
            unified_before = self._unified_settings_summary(
                unified_settings,
                settings_owner,
            )
            brush_before = self._brush_summary(
                brush,
                bpy.context.tool_settings,
            )
            try:
                if size is not None:
                    brush.size = size
                if strength is not None:
                    brush.strength = strength
                if direction is not None:
                    brush.direction = direction
                for parameter, value in quality_updates.items():
                    setattr(
                        brush,
                        BRUSH_QUALITY_PARAMETER_RNA_NAMES[parameter],
                        value,
                    )
                for parameter, value in pose_updates.items():
                    setattr(
                        brush,
                        POSE_BRUSH_PARAMETER_RNA_NAMES[parameter],
                        value,
                    )
            except (AttributeError, RuntimeError, TypeError, ValueError) as error:
                raise JsonRpcError(
                    -32023,
                    "Cannot update the active Sculpt brush",
                    {"reason": str(error)},
                ) from error
            brush_summary = self._brush_summary(
                brush,
                bpy.context.tool_settings,
            )
            unified_after = self._unified_settings_summary(
                unified_settings,
                settings_owner,
            )
            if unified_after != unified_before:
                raise JsonRpcError(
                    -32023,
                    "Updating the active Brush changed Unified settings",
                    {
                        "before": unified_before,
                        "after": unified_after,
                    },
                )

        target["area"].tag_redraw()
        return {
            "update_scope": "ACTIVE_BRUSH",
            "updated_parameters": sorted(
                {
                    name
                    for name, value in (
                        ("size", size),
                        ("strength", strength),
                        ("direction", direction),
                    )
                    if value is not None
                }
                | quality_updates.keys()
                | pose_updates.keys()
            ),
            "previous_brush": brush_before,
            "brush": brush_summary,
            "unified_settings": unified_after,
            "unified_settings_unchanged": True,
            "viewport": self._viewport_summary(target),
        }

    def set_use_unified_size(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Enable or disable Sculpt's version-appropriate unified size."""
        _reject_unknown(params, {"enabled"})
        if "enabled" not in params:
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {"reason": "enabled is required"},
            )
        enabled = _boolean(params, "enabled", default=False)
        tool_settings = getattr(bpy.context, "tool_settings", None)
        unified_settings, settings_owner = (
            self._find_sculpt_unified_settings(tool_settings)
        )
        if unified_settings is None or settings_owner is None:
            raise JsonRpcError(
                -32023,
                "Sculpt unified size settings are unavailable",
                {"blender_version": bpy.app.version_string},
            )

        previous = bool(unified_settings.use_unified_size)
        try:
            unified_settings.use_unified_size = enabled
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise JsonRpcError(
                -32023,
                "Cannot update Sculpt Use Unified Size",
                {"reason": str(error)},
            ) from error
        applied = bool(unified_settings.use_unified_size)
        if applied != enabled:
            raise JsonRpcError(
                -32023,
                "Blender did not apply Sculpt Use Unified Size",
                {"requested": enabled, "actual": applied},
            )

        sculpt_settings = getattr(tool_settings, "sculpt", None)
        brush = getattr(sculpt_settings, "brush", None)
        self._tag_view3d_redraws()
        return {
            "enabled": applied,
            "previous": previous,
            "settings_owner": settings_owner,
            "rna_path": (
                "tool_settings.sculpt.unified_paint_settings"
                if settings_owner == "SCULPT_SETTINGS"
                else "tool_settings.unified_paint_settings"
            ),
            "prerequisites_completed": [],
            "brush": self._brush_summary(brush, tool_settings),
        }

    def set_use_size_pressure(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Enable or disable size pressure on the active Sculpt brush."""
        _reject_unknown(
            params,
            {"enabled", "window_index", "area_index"},
        )
        if "enabled" not in params:
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {"reason": "enabled is required"},
            )
        enabled = _boolean(params, "enabled", default=False)
        window_index = _optional_integer(
            params,
            "window_index",
            minimum=0,
        )
        area_index = _optional_integer(params, "area_index", minimum=0)
        brush, prerequisites_completed, viewport = (
            self._ensure_active_sculpt_brush(
                window_index=window_index,
                area_index=area_index,
            )
        )
        supported = self._brush_supports_size_pressure(brush)
        if enabled and not supported:
            raise JsonRpcError(
                -32023,
                "The active Sculpt brush does not support size pressure",
                {
                    "brush": brush.name,
                    "requested": enabled,
                    "size_pressure_supported": False,
                },
            )

        property_name = self._size_pressure_property_name(brush)
        if property_name is None:
            raise JsonRpcError(
                -32023,
                "This Blender version exposes no size pressure setting",
                {"blender_version": bpy.app.version_string},
            )
        previous = bool(getattr(brush, property_name))
        try:
            setattr(brush, property_name, enabled)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise JsonRpcError(
                -32023,
                "Cannot update Sculpt Use Size Pressure",
                {"brush": brush.name, "reason": str(error)},
            ) from error
        applied = bool(getattr(brush, property_name))
        if applied != enabled:
            raise JsonRpcError(
                -32023,
                "Blender did not apply Sculpt Use Size Pressure",
                {
                    "brush": brush.name,
                    "requested": enabled,
                    "actual": applied,
                },
            )

        self._tag_view3d_redraws()
        result = {
            "enabled": applied,
            "previous": previous,
            "size_pressure_supported": supported,
            "rna_property": property_name,
            "prerequisites_completed": prerequisites_completed,
            "brush": self._brush_summary(
                brush,
                bpy.context.tool_settings,
            ),
        }
        if viewport is not None:
            result["viewport"] = viewport
        return result

    def set_use_unified_strength(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Enable or disable version-appropriate unified strength."""
        _reject_unknown(params, {"enabled"})
        if "enabled" not in params:
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {"reason": "enabled is required"},
            )
        enabled = _boolean(params, "enabled", default=False)
        tool_settings = getattr(bpy.context, "tool_settings", None)
        unified_settings, settings_owner = (
            self._find_sculpt_unified_settings(tool_settings)
        )
        if unified_settings is None or settings_owner is None:
            raise JsonRpcError(
                -32023,
                "Sculpt unified strength settings are unavailable",
                {"blender_version": bpy.app.version_string},
            )

        previous = bool(unified_settings.use_unified_strength)
        try:
            unified_settings.use_unified_strength = enabled
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise JsonRpcError(
                -32023,
                "Cannot update Sculpt Use Unified Strength",
                {"reason": str(error)},
            ) from error
        applied = bool(unified_settings.use_unified_strength)
        if applied != enabled:
            raise JsonRpcError(
                -32023,
                "Blender did not apply Sculpt Use Unified Strength",
                {"requested": enabled, "actual": applied},
            )

        sculpt_settings = getattr(tool_settings, "sculpt", None)
        brush = getattr(sculpt_settings, "brush", None)
        self._tag_view3d_redraws()
        return {
            "enabled": applied,
            "previous": previous,
            "settings_owner": settings_owner,
            "rna_path": (
                "tool_settings.sculpt.unified_paint_settings"
                if settings_owner == "SCULPT_SETTINGS"
                else "tool_settings.unified_paint_settings"
            ),
            "prerequisites_completed": [],
            "brush": self._brush_summary(brush, tool_settings),
        }

    def set_use_strength_pressure(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Enable or disable strength pressure on the active Sculpt brush."""
        _reject_unknown(
            params,
            {"enabled", "window_index", "area_index"},
        )
        if "enabled" not in params:
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {"reason": "enabled is required"},
            )
        enabled = _boolean(params, "enabled", default=False)
        window_index = _optional_integer(
            params,
            "window_index",
            minimum=0,
        )
        area_index = _optional_integer(params, "area_index", minimum=0)
        brush, prerequisites_completed, viewport = (
            self._ensure_active_sculpt_brush(
                window_index=window_index,
                area_index=area_index,
            )
        )
        supported = self._brush_supports_strength_pressure(brush)
        if enabled and not supported:
            raise JsonRpcError(
                -32023,
                "The active Sculpt brush does not support strength pressure",
                {
                    "brush": brush.name,
                    "requested": enabled,
                    "strength_pressure_supported": False,
                },
            )

        property_name = self._strength_pressure_property_name(brush)
        if property_name is None:
            raise JsonRpcError(
                -32023,
                "This Blender version exposes no strength pressure setting",
                {"blender_version": bpy.app.version_string},
            )
        previous = bool(getattr(brush, property_name))
        try:
            setattr(brush, property_name, enabled)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise JsonRpcError(
                -32023,
                "Cannot update Sculpt Use Strength Pressure",
                {"brush": brush.name, "reason": str(error)},
            ) from error
        applied = bool(getattr(brush, property_name))
        if applied != enabled:
            raise JsonRpcError(
                -32023,
                "Blender did not apply Sculpt Use Strength Pressure",
                {
                    "brush": brush.name,
                    "requested": enabled,
                    "actual": applied,
                },
            )

        self._tag_view3d_redraws()
        result = {
            "enabled": applied,
            "previous": previous,
            "strength_pressure_supported": supported,
            "rna_property": property_name,
            "prerequisites_completed": prerequisites_completed,
            "brush": self._brush_summary(
                brush,
                bpy.context.tool_settings,
            ),
        }
        if viewport is not None:
            result["viewport"] = viewport
        return result

    def set_dyntopo(self, params: dict[str, Any]) -> dict[str, Any]:
        """Enable or disable Dyntopo and optionally set relative detail size."""
        _reject_unknown(
            params,
            {
                "enabled",
                "detail_size",
                "window_index",
                "area_index",
            },
        )
        if "enabled" not in params:
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {"reason": "enabled is required"},
            )
        enabled = _boolean(params, "enabled", default=False)
        detail_size = _optional_number(
            params,
            "detail_size",
            minimum=0.5,
            maximum=40.0,
        )
        target = self._resolve_viewports(
            window_index=_optional_integer(
                params,
                "window_index",
                minimum=0,
            ),
            area_index=_optional_integer(params, "area_index", minimum=0),
            all_viewports=False,
        )[0]

        with bpy.context.temp_override(
            window=target["window"],
            area=target["area"],
            region=target["region"],
        ):
            sculpt_object, sculpt_settings, _ = (
                self._require_sculpt_context()
            )
            current_enabled = bool(
                sculpt_object.use_dynamic_topology_sculpting
            )
            if current_enabled != enabled:
                self._call_operator(
                    bpy.ops.sculpt.dynamic_topology_toggle,
                    {},
                    error_message="Cannot update Dyntopo",
                    error_code=-32024,
                )
            final_enabled = bool(
                sculpt_object.use_dynamic_topology_sculpting
            )
            if final_enabled != enabled:
                raise JsonRpcError(
                    -32024,
                    "Blender did not apply the requested Dyntopo state",
                    {
                        "requested": enabled,
                        "actual": final_enabled,
                    },
                )
            if detail_size is not None:
                try:
                    sculpt_settings.detail_type_method = "RELATIVE"
                    sculpt_settings.detail_size = detail_size
                except (
                    AttributeError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:
                    raise JsonRpcError(
                        -32024,
                        "Cannot update Dyntopo detail size",
                        {"reason": str(error)},
                    ) from error
            dyntopo_summary = self._dyntopo_summary(
                sculpt_object,
                sculpt_settings,
            )

        target["area"].tag_redraw()
        return {
            "object": sculpt_object.name,
            **dyntopo_summary,
            "viewport": self._viewport_summary(target),
        }

    def sculpt_face_set_lasso(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a Face Set from one validated lasso mouse path."""
        allowed = {
            "path",
            "use_front_faces_only",
            "show_face_sets",
            "window_index",
            "area_index",
        }
        _reject_unknown(params, allowed)
        target = self._resolve_viewports(
            window_index=_optional_integer(
                params,
                "window_index",
                minimum=0,
            ),
            area_index=_optional_integer(
                params,
                "area_index",
                minimum=0,
            ),
            all_viewports=False,
        )[0]
        request = parse_face_set_lasso_request(
            params,
            region_width=target["region"].width,
            region_height=target["region"].height,
        )

        with bpy.context.temp_override(
            window=target["window"],
            area=target["area"],
            region=target["region"],
        ):
            sculpt_object, _, _ = self._require_sculpt_context()
            operator = self._operator_or_none(
                "sculpt",
                "face_set_lasso_gesture",
            )
            if operator is None:
                raise JsonRpcError(
                    -32026,
                    "Lasso Face Set is unavailable in this Blender version",
                    {
                        "blender_version": bpy.app.version_string,
                        "operator": (
                            "bpy.ops.sculpt.face_set_lasso_gesture"
                        ),
                    },
                )
            operator_kwargs, path_item_fields = (
                self._face_set_lasso_operator_kwargs(
                    operator,
                    request.operator_kwargs,
                )
            )
            operator_result = self._call_operator(
                operator,
                operator_kwargs,
                error_message="Cannot execute the Lasso Face Set gesture",
                error_code=-32026,
            )
            face_set_attribute = self._face_set_attribute_summary(
                sculpt_object
            )
            face_sets_overlay = self._set_sculpt_face_sets_overlay(
                target,
                request.show_face_sets,
            )

        target["area"].tag_redraw()
        return {
            "operator": "bpy.ops.sculpt.face_set_lasso_gesture",
            "operator_status": sorted(
                str(item) for item in operator_result
            ),
            "path_point_count": request.point_count,
            "operator_mouse_path_fields": path_item_fields,
            "use_front_faces_only": bool(
                request.operator_kwargs["use_front_faces_only"]
            ),
            "show_face_sets": face_sets_overlay["enabled"],
            "face_sets_overlay": face_sets_overlay,
            "mouse_bounds": {
                "min": list(request.mouse_min),
                "max": list(request.mouse_max),
            },
            "coordinate_system": {
                "space": "VIEW_3D_WINDOW_REGION",
                "origin": "BOTTOM_LEFT",
                "units": "BLENDER_UI_PIXELS",
            },
            "object": sculpt_object.name,
            "face_set_attribute": face_set_attribute,
            "viewport": self._viewport_summary(target),
        }

    def sculpt_face_set_lasso_batch(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply and verify one rollback-safe Face Set lasso batch."""
        target = self._resolve_viewports(
            window_index=_optional_integer(
                params,
                "window_index",
                minimum=0,
            ),
            area_index=_optional_integer(
                params,
                "area_index",
                minimum=0,
            ),
            all_viewports=False,
        )[0]
        request = parse_face_set_lasso_batch_request(
            params,
            region_width=target["region"].width,
            region_height=target["region"].height,
        )

        with bpy.context.temp_override(
            window=target["window"],
            area=target["area"],
            region=target["region"],
        ):
            sculpt_object, _, _ = self._require_sculpt_context()
            operator = self._operator_or_none(
                "sculpt",
                "face_set_lasso_gesture",
            )
            if operator is None:
                raise JsonRpcError(
                    -32026,
                    "Lasso Face Set is unavailable in this Blender version",
                    {
                        "blender_version": bpy.app.version_string,
                        "operator": (
                            "bpy.ops.sculpt.face_set_lasso_gesture"
                        ),
                    },
                )

            snapshot = self._face_set_attribute_snapshot(sculpt_object)
            transaction_started = False
            try:
                self._ensure_face_set_attribute(sculpt_object)
                transaction_started = True
                baseline_values = self._face_set_attribute_values(
                    sculpt_object
                )
                previous_values = baseline_values
                operations: list[dict[str, Any]] = []
                generated_ids: list[int] = []
                for item in request.paths:
                    operator_kwargs, path_item_fields = (
                        self._face_set_lasso_operator_kwargs(
                            operator,
                            item.request.operator_kwargs,
                        )
                    )
                    expected_id = max(
                        abs(int(value)) for value in previous_values
                    ) + 1
                    operator_result = self._call_operator(
                        operator,
                        operator_kwargs,
                        error_message=(
                            "Cannot execute the Lasso Face Set batch"
                        ),
                        error_code=-32026,
                    )
                    current_values = self._face_set_attribute_values(
                        sculpt_object
                    )
                    assigned_id, changed_count = (
                        self._assigned_face_set_id(
                            previous_values,
                            current_values,
                            fallback_id=expected_id,
                        )
                    )
                    generated_ids.append(assigned_id)
                    operations.append(
                        {
                            "label": item.label,
                            "operator_status": sorted(
                                str(value) for value in operator_result
                            ),
                            "path_point_count": item.request.point_count,
                            "operator_mouse_path_fields": (
                                path_item_fields
                            ),
                            "mouse_bounds": {
                                "min": list(item.request.mouse_min),
                                "max": list(item.request.mouse_max),
                            },
                            "assigned_face_set_id": assigned_id,
                            "changed_face_count": changed_count,
                        }
                    )
                    previous_values = current_values

                mesh_arrays = self._face_set_mesh_arrays(sculpt_object)
                coverage = analyze_face_set_coverage(
                    baseline_values=baseline_values,
                    current_values=previous_values,
                    generated_face_set_ids=generated_ids,
                    **mesh_arrays,
                )
                rolled_back = bool(
                    not coverage["complete"]
                    and request.rollback_on_residual
                )
                if rolled_back:
                    self._restore_face_set_attribute_snapshot(
                        sculpt_object,
                        snapshot,
                    )
                face_set_attribute = self._face_set_attribute_summary(
                    sculpt_object
                )
            except Exception as error:
                if transaction_started:
                    try:
                        self._restore_face_set_attribute_snapshot(
                            sculpt_object,
                            snapshot,
                        )
                    except Exception as rollback_error:
                        raise JsonRpcError(
                            -32026,
                            "Face Set lasso batch failed and rollback failed",
                            {
                                "reason": str(error),
                                "rollback_reason": str(rollback_error),
                            },
                        ) from rollback_error
                raise

            face_sets_overlay = self._set_sculpt_face_sets_overlay(
                target,
                request.show_face_sets,
            )

        target["area"].tag_redraw()
        return {
            "operator": "bpy.ops.sculpt.face_set_lasso_gesture",
            "transaction": "sculpt-face-set-lasso-batch/v1",
            "padding_pixels": request.padding_pixels,
            "path_count": len(request.paths),
            "operations": operations,
            "coverage": coverage,
            "rollback": {
                "requested_on_residual": (
                    request.rollback_on_residual
                ),
                "performed": rolled_back,
                "restored": rolled_back,
            },
            "use_front_faces_only": bool(
                request.paths[0].request.operator_kwargs[
                    "use_front_faces_only"
                ]
            ),
            "show_face_sets": face_sets_overlay["enabled"],
            "face_sets_overlay": face_sets_overlay,
            "coordinate_system": {
                "space": "VIEW_3D_WINDOW_REGION",
                "origin": "BOTTOM_LEFT",
                "units": "BLENDER_UI_PIXELS",
            },
            "object": sculpt_object.name,
            "face_set_attribute": face_set_attribute,
            "viewport": self._viewport_summary(target),
        }

    def sculpt_brush_stroke(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Replay one validated screen-space Sculpt mouse gesture."""
        allowed = {
            "operation_id",
            "stroke",
            "mode",
            "brush_toggle",
            "pen_flip",
            "execution_mode",
            "location_mode",
            "ignore_background_click",
            "window_index",
            "area_index",
        }
        _reject_unknown(params, allowed)
        target = self._resolve_viewports(
            window_index=_optional_integer(
                params,
                "window_index",
                minimum=0,
            ),
            area_index=_optional_integer(
                params,
                "area_index",
                minimum=0,
            ),
            all_viewports=False,
        )[0]
        request = parse_sculpt_brush_stroke_request(
            params,
            region_width=target["region"].width,
            region_height=target["region"].height,
        )
        if (
            request.operation_id is not None
            and self._sculpt_operation_was_applied(
                bpy.context.scene,
                request.operation_id,
            )
        ):
            return {
                "operator": "bpy.ops.sculpt.brush_stroke",
                "operator_status": [],
                "stroke_element_count": request.element_count,
                "operation_id": request.operation_id,
                "idempotency": {"status": "already_applied"},
                "viewport": self._viewport_summary(target),
            }

        with bpy.context.temp_override(
            window=target["window"],
            area=target["area"],
            region=target["region"],
        ):
            sculpt_object, _, brush = self._require_sculpt_context()
            if brush is None:
                raise JsonRpcError(
                    -32025,
                    "Blender has no active Sculpt brush",
                )
            brush_type = self._sculpt_brush_type(brush)
            cloth_deform_type = getattr(
                brush,
                "cloth_deform_type",
                None,
            )
            execution_mode = resolve_sculpt_stroke_execution_mode(
                request.execution_mode,
                sculpt_brush_type=brush_type,
                brush_toggle=request.operator_kwargs["brush_toggle"],
                cloth_deform_type=cloth_deform_type,
            )
            location_mode = resolve_sculpt_stroke_location_mode(
                (
                    "ANCHORED_DRAG"
                    if execution_mode == "ANCHORED_DRAG"
                    else request.location_mode
                ),
                sculpt_brush_type=brush_type,
                brush_toggle=request.operator_kwargs["brush_toggle"],
                cloth_deform_type=cloth_deform_type,
            )
            if execution_mode != "ANCHORED_DRAG":
                location_mode = "SURFACE_RAYCAST"
            operator_kwargs = dict(request.operator_kwargs)
            initial_location: list[float] | None = None
            override_location: bool | None = None
            driver_operator = "bpy.ops.sculpt.brush_stroke"
            paint_curve_point_count: int | None = None
            if execution_mode == "ANCHORED_DRAG":
                prepared_stroke, initial_location = (
                    self._prepare_anchored_drag_stroke(
                        target,
                        sculpt_object,
                        request.operator_kwargs["stroke"],
                    )
                )
                operator_kwargs["stroke"] = prepared_stroke
                operator_kwargs["override_location"] = False
                override_location = False
            elif execution_mode == "PAINT_CURVE":
                _, initial_location = self._prepare_anchored_drag_stroke(
                    target,
                    sculpt_object,
                    request.operator_kwargs["stroke"],
                )
                (
                    operator_result,
                    driver_operator,
                    paint_curve_point_count,
                    override_location,
                ) = self._execute_sculpt_paint_curve(
                    brush=brush,
                    operator_kwargs=operator_kwargs,
                )
            else:
                operator_kwargs["override_location"] = True
                override_location = True
            if execution_mode != "PAINT_CURVE":
                operator_result = self._call_operator(
                    bpy.ops.sculpt.brush_stroke,
                    operator_kwargs,
                    error_message="Cannot execute the Sculpt brush stroke",
                    error_code=-32025,
                )
            brush_summary = self._brush_summary(
                brush,
                bpy.context.tool_settings,
            )

        target["area"].tag_redraw()
        if request.operation_id is not None:
            self._record_sculpt_operation(
                bpy.context.scene,
                request.operation_id,
            )
        result = {
            "operator": "bpy.ops.sculpt.brush_stroke",
            "driver_operator": driver_operator,
            "operator_status": sorted(
                str(item) for item in operator_result
            ),
            "stroke_element_count": request.element_count,
            "execution_mode": {
                "requested": request.execution_mode,
                "resolved": execution_mode,
            },
            "location_mode": {
                "requested": request.location_mode,
                "resolved": location_mode,
                "override_location": override_location,
            },
            "initial_location": initial_location,
            "mouse_bounds": {
                "min": list(request.mouse_min),
                "max": list(request.mouse_max),
            },
            "object": sculpt_object.name,
            "brush": brush_summary,
            "viewport": self._viewport_summary(target),
            "operation_id": request.operation_id,
            "idempotency": {"status": "applied"},
        }
        if paint_curve_point_count is not None:
            result["paint_curve_point_count"] = paint_curve_point_count
        return result

    @staticmethod
    def _sculpt_operation_ids(scene: Any) -> list[str]:
        """Read the checkpoint-rewindable Sculpt operation ledger."""
        raw = scene.get(_SCULPT_OPERATION_LEDGER_PROPERTY, "[]")
        if not isinstance(raw, str):
            return []
        try:
            values = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return []
        if not isinstance(values, list):
            return []
        return [
            value
            for value in values
            if isinstance(value, str) and value
        ][-_MAX_SCULPT_OPERATION_LEDGER_ENTRIES:]

    @classmethod
    def _sculpt_operation_was_applied(
        cls,
        scene: Any,
        operation_id: str,
    ) -> bool:
        return operation_id in cls._sculpt_operation_ids(scene)

    @classmethod
    def _record_sculpt_operation(
        cls,
        scene: Any,
        operation_id: str,
    ) -> None:
        values = cls._sculpt_operation_ids(scene)
        if operation_id not in values:
            values.append(operation_id)
        try:
            scene[_SCULPT_OPERATION_LEDGER_PROPERTY] = json.dumps(
                values[-_MAX_SCULPT_OPERATION_LEDGER_ENTRIES:],
                separators=(",", ":"),
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise JsonRpcError(
                -32025,
                "Cannot persist Sculpt operation idempotency state",
                {"operation_id": operation_id, "reason": str(error)},
            ) from error

    def _execute_sculpt_paint_curve(
        self,
        *,
        brush: Any,
        operator_kwargs: dict[str, Any],
    ) -> tuple[set[str], str, int, bool | None]:
        """Use Blender's Paint Curve path for native spacing and overlap."""
        stroke = operator_kwargs["stroke"]
        curve_points = prepare_sculpt_paint_curve_points(stroke)
        if len(curve_points) <= 1:
            single_kwargs = dict(operator_kwargs)
            single_kwargs["stroke"] = [stroke[0]]
            single_kwargs["override_location"] = True
            result = self._call_operator(
                bpy.ops.sculpt.brush_stroke,
                single_kwargs,
                error_message="Cannot execute the Sculpt brush dab",
                error_code=-32025,
            )
            return result, "bpy.ops.sculpt.brush_stroke", 1, True

        original_curve = getattr(brush, "paint_curve", None)
        original_stroke_method = str(brush.stroke_method)
        replay_curve = self._valid_sculpt_replay_paint_curve()
        temporary_curve_active = False
        try:
            brush.stroke_method = "CURVE"
            if replay_curve is None:
                self._call_operator(
                    bpy.ops.paintcurve.new,
                    {},
                    error_message="Cannot create a temporary Paint Curve",
                    error_code=-32025,
                )
                replay_curve = brush.paint_curve
                if replay_curve is None:
                    raise JsonRpcError(
                        -32025,
                        "Blender did not create a temporary Paint Curve",
                    )
                self._sculpt_replay_paint_curve = replay_curve
            else:
                brush.paint_curve = replay_curve
            temporary_curve_active = True

            self._clear_sculpt_replay_paint_curve()
            for x, y in curve_points:
                self._call_operator(
                    bpy.ops.paintcurve.add_point,
                    {"location": [x, y]},
                    error_message="Cannot add a temporary Paint Curve point",
                    error_code=-32025,
                )
            result = self._call_operator(
                bpy.ops.paintcurve.draw,
                {},
                error_message="Cannot draw the Sculpt Paint Curve",
                error_code=-32025,
            )
            return result, "bpy.ops.paintcurve.draw", len(curve_points), None
        finally:
            if temporary_curve_active:
                try:
                    self._clear_sculpt_replay_paint_curve()
                except (
                    AttributeError,
                    ReferenceError,
                    RuntimeError,
                    TypeError,
                ):
                    self._sculpt_replay_paint_curve = None
            try:
                brush.paint_curve = original_curve
                brush.stroke_method = original_stroke_method
            except (
                AttributeError,
                ReferenceError,
                RuntimeError,
                TypeError,
            ) as error:
                raise JsonRpcError(
                    -32025,
                    "Cannot restore the Sculpt brush after Paint Curve replay",
                    {"reason": str(error)},
                ) from error

    def _valid_sculpt_replay_paint_curve(self) -> Any | None:
        """Return the cached temporary Paint Curve if its RNA is still live."""
        replay_curve = self._sculpt_replay_paint_curve
        if replay_curve is None:
            return None
        try:
            replay_curve.as_pointer()
        except (AttributeError, ReferenceError, RuntimeError):
            self._sculpt_replay_paint_curve = None
            return None
        return replay_curve

    @staticmethod
    def _clear_sculpt_replay_paint_curve() -> None:
        """Select and remove every point from the active temporary curve."""
        for _ in range(2):
            bpy.ops.paintcurve.select(
                location=(0, 0),
                toggle=True,
                extend=False,
            )
        bpy.ops.paintcurve.delete_point()

    @staticmethod
    def _prepare_anchored_drag_stroke(
        target: dict[str, Any],
        sculpt_object: Any,
        stroke: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[float]]:
        """Anchor a screen-space Drag to the first visible Sculpt hit."""
        first_mouse = stroke[0]["mouse_event"]
        region = target["region"]
        region_3d = target["space"].region_3d
        try:
            ray_origin = view3d_utils.region_2d_to_origin_3d(
                region,
                region_3d,
                first_mouse,
            )
            ray_direction = view3d_utils.region_2d_to_vector_3d(
                region,
                region_3d,
                first_mouse,
            )
            scene = bpy.context.scene
            depsgraph = bpy.context.evaluated_depsgraph_get()
            (
                hit,
                world_location,
                _world_normal,
                _face_index,
                hit_object,
                _hit_matrix,
            ) = scene.ray_cast(
                depsgraph,
                ray_origin,
                ray_direction,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise JsonRpcError(
                -32025,
                "Cannot resolve the anchored Sculpt Drag start",
                {
                    "reason_code": "INITIAL_HIT_RAYCAST_FAILED",
                    "mouse_event": list(first_mouse),
                    "reason": str(error),
                },
            ) from error

        hit_original = getattr(hit_object, "original", hit_object)
        sculpt_original = getattr(
            sculpt_object,
            "original",
            sculpt_object,
        )
        same_object = bool(
            hit
            and hit_object is not None
            and (
                hit_object == sculpt_object
                or hit_original == sculpt_original
            )
        )
        if not same_object:
            raise JsonRpcError(
                -32025,
                "Anchored Sculpt Drag must start on the active Sculpt object",
                {
                    "reason_code": "INITIAL_HIT_MISSED",
                    "mouse_event": list(first_mouse),
                    "active_object": getattr(
                        sculpt_object,
                        "name",
                        None,
                    ),
                    "hit_object": getattr(hit_object, "name", None),
                },
            )

        object_location = (
            sculpt_object.matrix_world.inverted_safe() @ world_location
        )
        initial_location = [float(value) for value in object_location]
        prepared_stroke = anchor_sculpt_drag_stroke(
            stroke,
            initial_location,
        )
        return prepared_stroke, initial_location

    @staticmethod
    def _operator_or_none(category: str, name: str) -> Any | None:
        try:
            operator = getattr(getattr(bpy.ops, category), name)
            operator.get_rna_type()
        except (AttributeError, KeyError, RuntimeError):
            return None
        return operator

    @staticmethod
    def _find_sculpt_unified_settings(
        tool_settings: Any,
    ) -> tuple[Any | None, str | None]:
        """Locate Sculpt unified settings across the Blender 5.0 migration."""
        if tool_settings is None:
            return None, None
        sculpt_settings = getattr(tool_settings, "sculpt", None)
        scoped_settings = getattr(
            sculpt_settings,
            "unified_paint_settings",
            None,
        )
        if scoped_settings is not None:
            return scoped_settings, "SCULPT_SETTINGS"
        legacy_settings = getattr(
            tool_settings,
            "unified_paint_settings",
            None,
        )
        if legacy_settings is not None:
            return legacy_settings, "TOOL_SETTINGS"
        return None, None

    @staticmethod
    def _unified_settings_summary(
        unified_settings: Any,
        settings_owner: str | None,
    ) -> dict[str, Any] | None:
        if unified_settings is None:
            return None
        return {
            "settings_owner": settings_owner,
            "use_unified_size": bool(
                unified_settings.use_unified_size
            ),
            "size": int(unified_settings.size),
            "use_unified_strength": bool(
                unified_settings.use_unified_strength
            ),
            "strength": float(unified_settings.strength),
        }

    @staticmethod
    def _size_pressure_property_name(brush: Any) -> str | None:
        if hasattr(brush, "use_pressure_size"):
            return "use_pressure_size"
        if hasattr(brush, "use_size_pressure"):
            return "use_size_pressure"
        return None

    @staticmethod
    def _brush_supports_size_pressure(brush: Any) -> bool:
        capabilities = getattr(brush, "sculpt_capabilities", None)
        supported = getattr(capabilities, "has_size_pressure", None)
        return True if supported is None else bool(supported)

    @staticmethod
    def _strength_pressure_property_name(brush: Any) -> str | None:
        if hasattr(brush, "use_pressure_strength"):
            return "use_pressure_strength"
        if hasattr(brush, "use_strength_pressure"):
            return "use_strength_pressure"
        return None

    @staticmethod
    def _brush_supports_strength_pressure(brush: Any) -> bool:
        capabilities = getattr(brush, "sculpt_capabilities", None)
        supported = getattr(capabilities, "has_strength_pressure", None)
        return True if supported is None else bool(supported)

    def _ensure_active_sculpt_brush(
        self,
        *,
        window_index: int | None,
        area_index: int | None,
    ) -> tuple[Any, list[str], dict[str, Any] | None]:
        tool_settings = getattr(bpy.context, "tool_settings", None)
        sculpt_settings = getattr(tool_settings, "sculpt", None)
        if sculpt_settings is None:
            raise JsonRpcError(-32020, "Sculpt settings are unavailable")
        brush = getattr(sculpt_settings, "brush", None)
        if brush is not None:
            return brush, [], None
        if bpy.app.background:
            raise JsonRpcError(
                -32010,
                "A 3D viewport is required to activate a Sculpt brush",
            )

        previous_mode = bpy.context.mode
        previous_workspace = _name(getattr(bpy.context, "workspace", None))
        enter_params: dict[str, Any] = {}
        if window_index is not None:
            enter_params["window_index"] = window_index
        if area_index is not None:
            enter_params["area_index"] = area_index
        entered = self.enter_sculpt_mode(enter_params)

        completed: list[str] = []
        if previous_workspace != entered["workspace"]:
            completed.append("switched_to_sculpting_workspace")
        if previous_mode != "SCULPT":
            completed.append("entered_sculpt_mode")

        viewport = entered["viewport"]
        activated = self.activate_sculpt_brush(
            {
                "brush": "Draw",
                "window_index": viewport["window_index"],
                "area_index": viewport["area_index"],
            }
        )
        completed.append("activated_default_draw_brush")
        brush = getattr(
            getattr(bpy.context.tool_settings, "sculpt", None),
            "brush",
            None,
        )
        if brush is None:
            raise JsonRpcError(
                -32022,
                "Blender has no active Sculpt brush after preparation",
                {"retryable": True},
            )
        return brush, completed, activated["viewport"]

    @staticmethod
    def _tag_view3d_redraws() -> None:
        if bpy.app.background:
            return
        window_manager = getattr(bpy.context, "window_manager", None)
        windows = (
            list(window_manager.windows)
            if window_manager is not None
            else []
        )
        for window in windows:
            for area in window.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()

    def _activate_legacy_sculpt_brush(self, brush_name: str) -> None:
        """Use Blender's pre-asset brush selector when it is available."""
        operator = self._operator_or_none("paint", "brush_select")
        if operator is None:
            raise JsonRpcError(
                -32022,
                "This Blender version has no supported brush selector",
                {"brush": brush_name},
            )

        normalized = "".join(
            char if char.isalnum() else "_" for char in brush_name.upper()
        )
        normalized = "_".join(
            part for part in normalized.split("_") if part
        )
        aliases = {
            "DRAW_SHARP": "DRAW",
            "SCRAPE_FILL": "SCRAPE",
        }
        sculpt_tool = aliases.get(normalized, normalized)
        try:
            property_definition = operator.get_rna_type().properties[
                "sculpt_tool"
            ]
            supported = {
                item.identifier
                for item in property_definition.enum_items
            }
        except (AttributeError, KeyError, TypeError):
            supported = set()
        if supported and sculpt_tool not in supported:
            raise JsonRpcError(
                -32602,
                "Unsupported Sculpt brush",
                {
                    "brush": brush_name,
                    "supported": sorted(supported),
                },
            )
        self._call_operator(
            operator,
            {
                "paint_mode": "SCULPT",
                "sculpt_tool": sculpt_tool,
                "toggle": False,
                "create_missing": True,
            },
            error_message="Cannot activate the Sculpt brush",
            error_code=-32022,
        )

    @staticmethod
    def _require_sculpt_context() -> tuple[Any, Any, Any | None]:
        context = bpy.context
        sculpt_object = getattr(context, "sculpt_object", None)
        if (
            context.mode != "SCULPT"
            or sculpt_object is None
            or sculpt_object.type != "MESH"
        ):
            active_object = (
                context.view_layer.objects.active
                if context.view_layer is not None
                else None
            )
            raise JsonRpcError(
                -32020,
                "Sculpt Mode with an active mesh is required",
                {
                    "mode": context.mode,
                    "active_object": _name(active_object),
                    "active_object_type": getattr(
                        active_object,
                        "type",
                        None,
                    ),
                },
            )
        sculpt_settings = getattr(context.tool_settings, "sculpt", None)
        if sculpt_settings is None:
            raise JsonRpcError(
                -32020,
                "Sculpt settings are unavailable",
            )
        return sculpt_object, sculpt_settings, sculpt_settings.brush

    def _sculpt_state_summary(
        self,
        context: Any,
        active_object: Any,
    ) -> dict[str, Any]:
        sculpt_settings = getattr(
            getattr(context, "tool_settings", None),
            "sculpt",
            None,
        )
        brush = (
            getattr(sculpt_settings, "brush", None)
            if sculpt_settings is not None
            else None
        )
        dyntopo = (
            self._dyntopo_summary(active_object, sculpt_settings)
            if (
                active_object is not None
                and active_object.type == "MESH"
                and sculpt_settings is not None
            )
            else None
        )
        unified_settings, settings_owner = (
            self._find_sculpt_unified_settings(context.tool_settings)
        )
        unified_summary = self._unified_settings_summary(
            unified_settings,
            settings_owner,
        )
        brush_inventory = self._sculpt_brush_inventory(brush)
        return {
            "active": context.mode == "SCULPT",
            "brush": self._brush_summary(brush, context.tool_settings),
            "available_brushes": brush_inventory["available_brushes"],
            "available_brush_count": brush_inventory[
                "available_brush_count"
            ],
            "available_brush_details": brush_inventory[
                "available_brush_details"
            ],
            "loaded_brushes": brush_inventory["loaded_brushes"],
            "loaded_brush_count": brush_inventory["loaded_brush_count"],
            "brush_inventory": brush_inventory["scan"],
            "unified_size": {
                "enabled": bool(unified_settings.use_unified_size),
                "size": int(unified_settings.size),
                "settings_owner": settings_owner,
            }
            if unified_settings is not None
            else None,
            "unified_strength": {
                "enabled": bool(unified_settings.use_unified_strength),
                "strength": float(unified_settings.strength),
                "settings_owner": settings_owner,
            }
            if unified_settings is not None
            else None,
            "unified_settings": unified_summary,
            "dyntopo": dyntopo,
        }

    @classmethod
    def _loaded_sculpt_brush_details(
        cls,
        active_brush: Any,
    ) -> list[dict[str, Any]]:
        """Return loaded brushes usable in Sculpt Mode."""
        brush_assets_required = bpy.app.version >= (4, 3, 0)
        details = []
        for brush in bpy.data.brushes:
            if not bool(getattr(brush, "use_paint_sculpt", False)):
                continue
            is_active = active_brush is not None and brush == active_brush
            is_asset = getattr(brush, "asset_data", None) is not None
            if brush_assets_required and not (is_active or is_asset):
                continue
            direction_values = cls._sculpt_brush_direction_values(brush)
            details.append(
                {
                    "name": str(brush.name),
                    "direction_supported": bool(direction_values),
                    "direction_values": list(direction_values),
                }
            )
        return sorted(
            details,
            key=lambda detail: (
                detail["name"].casefold(),
                detail["name"],
            ),
        )

    def _sculpt_brush_inventory(
        self,
        active_brush: Any,
    ) -> dict[str, Any]:
        """Collect loaded and locally available Sculpt brush names."""
        loaded_details = self._loaded_sculpt_brush_details(active_brush)
        loaded_brushes = [detail["name"] for detail in loaded_details]
        available_directions = {
            detail["name"]: list(detail["direction_values"])
            for detail in loaded_details
        }
        errors: list[dict[str, str]] = []
        library_files = self._asset_library_blend_files(errors)
        current_file = (
            self._normalized_path(Path(bpy.data.filepath))
            if bpy.data.filepath
            else None
        )
        scanned_files = 0
        cached_files = 0
        active_cache_keys: set[str] = set()

        for library_file in library_files:
            cache_key = self._normalized_path(library_file)
            active_cache_keys.add(cache_key)
            if current_file is not None and cache_key == current_file:
                continue
            try:
                details, cached, scan_error = (
                    self._sculpt_brush_details_from_library(library_file)
                )
            except OSError as error:
                errors.append(
                    {
                        "path": str(library_file),
                        "reason": str(error),
                    }
                )
                continue
            scanned_files += 1
            if cached:
                cached_files += 1
            for name, direction_values in details:
                merged_values = available_directions.setdefault(name, [])
                for direction_value in direction_values:
                    if direction_value not in merged_values:
                        merged_values.append(direction_value)
            if scan_error is not None:
                errors.append(
                    {
                        "path": str(library_file),
                        "reason": scan_error,
                    }
                )

        stale_cache_keys = (
            set(self._sculpt_brush_library_cache) - active_cache_keys
        )
        for cache_key in stale_cache_keys:
            del self._sculpt_brush_library_cache[cache_key]

        available_brushes = sorted(
            available_directions,
            key=lambda name: (name.casefold(), name),
        )
        available_brush_details = [
            {
                "name": name,
                "direction_supported": bool(available_directions[name]),
                "direction_values": list(available_directions[name]),
            }
            for name in available_brushes
        ]
        return {
            "available_brushes": available_brushes,
            "available_brush_count": len(available_brushes),
            "available_brush_details": available_brush_details,
            "loaded_brushes": loaded_brushes,
            "loaded_brush_count": len(loaded_brushes),
            "scan": {
                "complete": not errors,
                "library_file_count": len(library_files),
                "scanned_file_count": scanned_files,
                "cached_file_count": cached_files,
                "errors": errors,
            },
        }

    def _sculpt_brush_details_from_library(
        self,
        library_file: Path,
    ) -> tuple[
        tuple[tuple[str, tuple[str, ...]], ...],
        bool,
        str | None,
    ]:
        """Inspect Brush assets in one blend file without retaining IDs."""
        stat = library_file.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        cache_key = self._normalized_path(library_file)
        cached = self._sculpt_brush_library_cache.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1], True, cached[2]

        before_ids = set(bpy.data.user_map().keys())
        details: tuple[tuple[str, tuple[str, ...]], ...] = ()
        scan_error: str | None = None
        source_names: tuple[str, ...] = ()
        loaded_brushes: list[Any] = []
        try:
            with bpy.data.libraries.load(
                str(library_file),
                link=True,
                relative=False,
                assets_only=True,
            ) as (data_from, data_to):
                source_names = tuple(str(name) for name in data_from.brushes)
                # Blender replaces data_to list entries with loaded IDs in place.
                data_to.brushes = list(source_names)
            loaded_brushes = list(data_to.brushes)
            details = tuple(
                (
                    source_name,
                    self._sculpt_brush_direction_values(loaded_brush),
                )
                for source_name, loaded_brush in zip(
                    source_names,
                    loaded_brushes,
                )
                if (
                    loaded_brush is not None
                    and bool(
                        getattr(
                            loaded_brush,
                            "use_paint_sculpt",
                            False,
                        )
                    )
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            scan_error = str(error)
        finally:
            new_ids = set(bpy.data.user_map().keys()) - before_ids
            if new_ids:
                bpy.data.batch_remove(new_ids)

        details = tuple(
            sorted(
                set(details),
                key=lambda detail: (
                    detail[0].casefold(),
                    detail[0],
                    detail[1],
                ),
            )
        )
        self._sculpt_brush_library_cache[cache_key] = (
            signature,
            details,
            scan_error,
        )
        return details, False, scan_error

    @staticmethod
    def _asset_library_blend_files(
        errors: list[dict[str, str]],
    ) -> list[Path]:
        """Find blend files in Essentials and enabled user libraries."""
        roots: list[Path] = []
        essentials_path = bpy.utils.system_resource(
            "DATAFILES",
            path="assets",
        )
        if essentials_path:
            roots.append(Path(essentials_path))

        preferences = getattr(bpy.context, "preferences", None)
        filepaths = getattr(preferences, "filepaths", None)
        asset_libraries = getattr(filepaths, "asset_libraries", ()) or ()
        for asset_library in asset_libraries:
            if not bool(getattr(asset_library, "enabled", True)):
                continue
            raw_path = str(getattr(asset_library, "path", "") or "")
            if not raw_path:
                continue
            roots.append(Path(bpy.path.abspath(raw_path)).expanduser())

        files: dict[str, Path] = {}
        seen_roots: set[str] = set()
        for root in roots:
            root_key = BlenderRpcApi._normalized_path(root)
            if root_key in seen_roots:
                continue
            seen_roots.add(root_key)
            if not root.is_dir():
                errors.append(
                    {
                        "path": str(root),
                        "reason": "Asset library directory is unavailable",
                    }
                )
                continue

            walk_errors: list[OSError] = []
            for directory, _, filenames in os.walk(
                root,
                followlinks=False,
                onerror=walk_errors.append,
            ):
                for filename in filenames:
                    if not filename.casefold().endswith(".blend"):
                        continue
                    path = Path(directory) / filename
                    files[BlenderRpcApi._normalized_path(path)] = path
            errors.extend(
                {
                    "path": str(root),
                    "reason": str(error),
                }
                for error in walk_errors
            )
        return [files[key] for key in sorted(files)]

    @staticmethod
    def _normalized_path(path: Path) -> str:
        return os.path.normcase(os.path.realpath(os.fspath(path)))

    @staticmethod
    def _sculpt_brush_type(brush: Any) -> str:
        """Return the current or legacy Sculpt brush type identifier."""
        brush_type = getattr(brush, "sculpt_brush_type", None)
        if brush_type is None:
            brush_type = getattr(brush, "sculpt_tool", None)
        return str(brush_type or "").upper()

    @staticmethod
    def _brush_enum_values(
        brush: Any,
        property_name: str,
    ) -> tuple[str, ...]:
        brush_rna = getattr(brush, "bl_rna", None)
        properties = getattr(brush_rna, "properties", None)
        enum_property = (
            properties.get(property_name)
            if properties is not None
            else None
        )
        if enum_property is None:
            return ()
        try:
            return tuple(
                str(item.identifier)
                for item in enum_property.enum_items
                if item.identifier
            )
        except (AttributeError, ReferenceError, TypeError):
            return ()

    @classmethod
    def _validate_pose_brush_runtime(
        cls,
        brush: Any,
        updates: dict[str, Any],
    ) -> None:
        """Verify that this Blender exposes every requested Pose property."""
        for parameter, value in updates.items():
            property_name = POSE_BRUSH_PARAMETER_RNA_NAMES[parameter]
            if not hasattr(brush, property_name):
                raise JsonRpcError(
                    -32023,
                    "Pose Brush setting is unavailable",
                    {
                        "parameter": parameter,
                        "rna_property": property_name,
                        "blender_version": bpy.app.version_string,
                    },
                )
            if parameter not in {
                "deformation_target",
                "rotation_origins",
            }:
                continue
            supported_values = cls._brush_enum_values(
                brush,
                property_name,
            )
            if value not in supported_values:
                raise JsonRpcError(
                    -32602,
                    "Invalid params",
                    {
                        "reason": (
                            f"{parameter} is not supported by this "
                            "Blender version"
                        ),
                        "parameter": parameter,
                        "value": value,
                        "supported_values": list(supported_values),
                        "blender_version": bpy.app.version_string,
                    },
                )

    @classmethod
    def _validate_brush_quality_runtime(
        cls,
        brush: Any,
        updates: dict[str, Any],
    ) -> None:
        """Verify that Blender exposes every requested quality property."""
        for parameter, value in updates.items():
            property_name = BRUSH_QUALITY_PARAMETER_RNA_NAMES[parameter]
            if not hasattr(brush, property_name):
                raise JsonRpcError(
                    -32023,
                    "Sculpt Brush quality setting is unavailable",
                    {
                        "parameter": parameter,
                        "rna_property": property_name,
                        "blender_version": bpy.app.version_string,
                    },
                )
            if parameter != "stroke_method":
                continue
            supported_values = cls._brush_enum_values(
                brush,
                property_name,
            )
            if value not in supported_values:
                raise JsonRpcError(
                    -32602,
                    "Invalid params",
                    {
                        "reason": (
                            "stroke_method is not supported by this "
                            "Blender version"
                        ),
                        "parameter": parameter,
                        "value": value,
                        "supported_values": list(supported_values),
                        "blender_version": bpy.app.version_string,
                    },
                )

    @classmethod
    def _pose_brush_settings_summary(
        cls,
        brush: Any,
    ) -> dict[str, Any] | None:
        if cls._sculpt_brush_type(brush) != "POSE":
            return None
        return {
            "deformation_target": str(brush.deform_target),
            "deformation_target_values": list(
                cls._brush_enum_values(brush, "deform_target")
            ),
            "rotation_origins": str(brush.pose_origin_type),
            "rotation_origins_values": list(
                cls._brush_enum_values(brush, "pose_origin_type")
            ),
            "pose_origin_offset": float(brush.pose_offset),
            "smooth_iterations": int(brush.pose_smooth_iterations),
            "pose_ik_segments": int(brush.pose_ik_segments),
            "connected_only": bool(brush.use_connected_only),
            "max_element_distance": float(
                brush.disconnected_distance_max
            ),
            "request_defaults": dict(POSE_BRUSH_PARAMETER_DEFAULTS),
        }

    @staticmethod
    def _sculpt_brush_direction_values(brush: Any) -> tuple[str, ...]:
        """Return Blender's context-sensitive Direction identifiers."""
        capabilities = getattr(brush, "sculpt_capabilities", None)
        has_direction = getattr(capabilities, "has_direction", None)
        if has_direction is False:
            return ()

        brush_type = BlenderRpcApi._sculpt_brush_type(brush)
        if brush_type == "MASK":
            mask_tool = str(getattr(brush, "mask_tool", "") or "").upper()
            return ("ADD", "SUBTRACT") if mask_tool == "DRAW" else ()
        specialized_values = _SCULPT_DIRECTION_VALUES_BY_TYPE.get(
            brush_type
        )
        if specialized_values is not None:
            return specialized_values

        # Static RNA applies only to regular direction brushes; the mapping above
        # handles dynamic item callbacks for specialized brush types.
        brush_rna = getattr(brush, "bl_rna", None)
        properties = getattr(brush_rna, "properties", None)
        direction_property = (
            properties.get("direction") if properties is not None else None
        )
        if direction_property is None:
            return ()
        try:
            return tuple(
                str(item.identifier)
                for item in direction_property.enum_items
                if item.identifier
            )
        except (AttributeError, ReferenceError, TypeError):
            return ()

    @staticmethod
    def _brush_summary(
        brush: Any,
        tool_settings: Any | None = None,
    ) -> dict[str, Any] | None:
        if brush is None:
            return None
        unified_settings, settings_owner = (
            BlenderRpcApi._find_sculpt_unified_settings(tool_settings)
        )
        use_unified_size = bool(
            unified_settings is not None
            and unified_settings.use_unified_size
        )
        use_unified_strength = bool(
            unified_settings is not None
            and unified_settings.use_unified_strength
        )
        effective_size = (
            unified_settings.size
            if use_unified_size
            else brush.size
        )
        effective_strength = (
            unified_settings.strength
            if use_unified_strength
            else brush.strength
        )
        size_pressure_property = (
            BlenderRpcApi._size_pressure_property_name(brush)
        )
        strength_pressure_property = (
            BlenderRpcApi._strength_pressure_property_name(brush)
        )
        direction_values = (
            BlenderRpcApi._sculpt_brush_direction_values(brush)
        )
        brush_type = BlenderRpcApi._sculpt_brush_type(brush)
        return {
            "name": brush.name,
            "sculpt_brush_type": brush_type or None,
            "size": int(brush.size),
            "strength": float(brush.strength),
            "direction": str(brush.direction),
            "direction_supported": bool(direction_values),
            "direction_values": list(direction_values),
            "stroke_method": (
                str(brush.stroke_method)
                if hasattr(brush, "stroke_method")
                else None
            ),
            "spacing": (
                int(brush.spacing)
                if hasattr(brush, "spacing")
                else None
            ),
            "use_space_attenuation": (
                bool(brush.use_space_attenuation)
                if hasattr(brush, "use_space_attenuation")
                else None
            ),
            "auto_smooth_factor": (
                float(brush.auto_smooth_factor)
                if hasattr(brush, "auto_smooth_factor")
                else None
            ),
            "effective_size": int(effective_size),
            "effective_strength": float(effective_strength),
            "use_unified_size": use_unified_size,
            "use_unified_strength": use_unified_strength,
            "unified_size_settings_owner": settings_owner,
            "use_size_pressure": (
                bool(getattr(brush, size_pressure_property))
                if size_pressure_property is not None
                else None
            ),
            "size_pressure_supported": (
                BlenderRpcApi._brush_supports_size_pressure(brush)
            ),
            "use_strength_pressure": (
                bool(getattr(brush, strength_pressure_property))
                if strength_pressure_property is not None
                else None
            ),
            "strength_pressure_supported": (
                BlenderRpcApi._brush_supports_strength_pressure(brush)
            ),
            "pose_settings": (
                BlenderRpcApi._pose_brush_settings_summary(brush)
            ),
        }

    @staticmethod
    def _dyntopo_summary(
        sculpt_object: Any,
        sculpt_settings: Any,
    ) -> dict[str, Any]:
        return {
            "enabled": bool(
                sculpt_object.use_dynamic_topology_sculpting
            ),
            "detail_size": float(sculpt_settings.detail_size),
            "detail_method": str(sculpt_settings.detail_type_method),
        }

    @staticmethod
    def _face_set_attribute(sculpt_object: Any) -> Any | None:
        """Return the Sculpt Face Set attribute across Blender versions."""
        mesh = getattr(sculpt_object, "data", None)
        attributes = getattr(mesh, "attributes", None)
        if attributes is None:
            return None
        attribute = attributes.get(".sculpt_face_set")
        if attribute is None:
            attribute = attributes.get("sculpt_face_set")
        return attribute

    @staticmethod
    def _face_set_attribute_snapshot(
        sculpt_object: Any,
    ) -> dict[str, Any]:
        """Capture the complete Face Set attribute without serializing it."""
        attribute = BlenderRpcApi._face_set_attribute(sculpt_object)
        if attribute is None:
            return {"present": False, "name": ".sculpt_face_set"}
        values = array("i", [0]) * len(attribute.data)
        attribute.data.foreach_get("value", values)
        return {
            "present": True,
            "name": str(attribute.name),
            "values": values,
        }

    @staticmethod
    def _ensure_face_set_attribute(sculpt_object: Any) -> Any:
        """Create a deterministic baseline attribute when none exists."""
        attribute = BlenderRpcApi._face_set_attribute(sculpt_object)
        if attribute is not None:
            return attribute
        mesh = getattr(sculpt_object, "data", None)
        attributes = getattr(mesh, "attributes", None)
        if attributes is None:
            raise JsonRpcError(
                -32026,
                "The Sculpt mesh does not support Face Set attributes",
            )
        try:
            with BlenderRpcApi._face_set_attribute_edit_context(
                sculpt_object
            ):
                attribute = attributes.new(
                    name=".sculpt_face_set",
                    type="INT",
                    domain="FACE",
                )
                values = array("i", [1]) * len(attribute.data)
                attribute.data.foreach_set("value", values)
                BlenderRpcApi._refresh_face_set_attribute(sculpt_object)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise JsonRpcError(
                -32026,
                "Cannot initialize the Sculpt Face Set attribute",
                {"reason": str(error)},
            ) from error
        return attribute

    @staticmethod
    def _face_set_attribute_values(sculpt_object: Any) -> array[int]:
        """Read all current Face Set IDs with one RNA bulk operation."""
        attribute = BlenderRpcApi._face_set_attribute(sculpt_object)
        if attribute is None:
            raise JsonRpcError(
                -32026,
                "Blender did not create a Sculpt Face Set attribute",
            )
        values = array("i", [0]) * len(attribute.data)
        try:
            attribute.data.foreach_get("value", values)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise JsonRpcError(
                -32026,
                "Cannot read the Sculpt Face Set attribute",
                {"reason": str(error)},
            ) from error
        return values

    @staticmethod
    def _restore_face_set_attribute_snapshot(
        sculpt_object: Any,
        snapshot: dict[str, Any],
    ) -> None:
        """Restore an exact Face Set snapshot after a failed attempt."""
        mesh = getattr(sculpt_object, "data", None)
        attributes = getattr(mesh, "attributes", None)
        if attributes is None:
            raise RuntimeError("Sculpt mesh has no attribute collection")
        with BlenderRpcApi._face_set_attribute_edit_context(sculpt_object):
            if not snapshot["present"]:
                for name in (".sculpt_face_set", "sculpt_face_set"):
                    attribute = attributes.get(name)
                    if attribute is not None:
                        attributes.remove(attribute)
            else:
                name = str(snapshot["name"])
                attribute = attributes.get(name)
                if attribute is None:
                    attribute = attributes.new(
                        name=name,
                        type="INT",
                        domain="FACE",
                    )
                values = snapshot["values"]
                if len(attribute.data) != len(values):
                    raise RuntimeError(
                        "Face count changed during Face Set transaction"
                    )
                attribute.data.foreach_set("value", values)
            BlenderRpcApi._refresh_face_set_attribute(sculpt_object)

    @staticmethod
    @contextmanager
    def _face_set_attribute_edit_context(
        sculpt_object: Any,
    ) -> Iterator[None]:
        """Edit mesh attributes in Object Mode and rebuild Sculpt PBVH."""
        restore_sculpt_mode = str(sculpt_object.mode) == "SCULPT"
        if restore_sculpt_mode:
            if not bpy.ops.object.mode_set.poll():
                raise RuntimeError(
                    "Cannot leave Sculpt Mode for Face Set rollback"
                )
            result = bpy.ops.object.mode_set(mode="OBJECT")
            if "FINISHED" not in result:
                raise RuntimeError(
                    "Blender did not enter Object Mode for Face Set rollback"
                )
        try:
            yield
        finally:
            if restore_sculpt_mode:
                if not bpy.ops.object.mode_set.poll():
                    raise RuntimeError(
                        "Cannot restore Sculpt Mode after Face Set rollback"
                    )
                result = bpy.ops.object.mode_set(mode="SCULPT")
                if "FINISHED" not in result:
                    raise RuntimeError(
                        "Blender did not restore Sculpt Mode after Face Set "
                        "rollback"
                    )

    @staticmethod
    def _refresh_face_set_attribute(sculpt_object: Any) -> None:
        """Invalidate mesh data after direct transactional restoration."""
        mesh = getattr(sculpt_object, "data", None)
        if mesh is not None:
            mesh.update()
        try:
            sculpt_object.update_tag(refresh={"DATA"})
        except TypeError:
            sculpt_object.update_tag()
        view_layer = getattr(bpy.context, "view_layer", None)
        if view_layer is not None:
            view_layer.update()

    @staticmethod
    def _assigned_face_set_id(
        before: array[int],
        after: array[int],
        *,
        fallback_id: int,
    ) -> tuple[int, int]:
        """Infer the one Face Set ID assigned by a lasso operator."""
        if len(before) != len(after):
            raise JsonRpcError(
                -32026,
                "Face count changed during Face Set lasso batch",
            )
        counts: dict[int, int] = {}
        changed_count = 0
        for previous, current in zip(before, after, strict=True):
            if abs(int(previous)) == abs(int(current)):
                continue
            changed_count += 1
            face_set_id = abs(int(current))
            counts[face_set_id] = counts.get(face_set_id, 0) + 1
        assigned = (
            max(counts, key=lambda value: (counts[value], value))
            if counts
            else fallback_id
        )
        return assigned, changed_count

    @staticmethod
    def _face_set_mesh_arrays(sculpt_object: Any) -> dict[str, Any]:
        """Bulk-read mesh arrays needed by deterministic coverage checks."""
        mesh = getattr(sculpt_object, "data", None)
        if mesh is None:
            raise JsonRpcError(-32026, "Active Sculpt object has no mesh")
        face_count = len(mesh.polygons)
        centers = array("f", [0.0]) * (face_count * 3)
        loop_starts = array("i", [0]) * face_count
        loop_totals = array("i", [0]) * face_count
        loop_edges = array("i", [0]) * len(mesh.loops)
        try:
            mesh.polygons.foreach_get("center", centers)
            mesh.polygons.foreach_get("loop_start", loop_starts)
            mesh.polygons.foreach_get("loop_total", loop_totals)
            mesh.loops.foreach_get("edge_index", loop_edges)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise JsonRpcError(
                -32026,
                "Cannot inspect mesh topology for Face Set coverage",
                {"reason": str(error)},
            ) from error
        return {
            "centers": centers,
            "polygon_loop_starts": loop_starts,
            "polygon_loop_totals": loop_totals,
            "loop_edges": loop_edges,
        }

    @staticmethod
    def _face_set_attribute_summary(
        sculpt_object: Any,
    ) -> dict[str, Any]:
        attribute = BlenderRpcApi._face_set_attribute(sculpt_object)
        if attribute is None:
            return {"present": False}
        return {
            "present": True,
            "name": attribute.name,
            "domain": str(attribute.domain),
            "data_type": str(attribute.data_type),
            "element_count": len(attribute.data),
        }

    @staticmethod
    def _set_sculpt_face_sets_overlay(
        target: dict[str, Any],
        enabled: bool,
    ) -> dict[str, Any]:
        """Set only the Face Sets child toggle in the target 3D Viewport."""
        overlay = getattr(target["space"], "overlay", None)
        if overlay is None or not hasattr(overlay, "show_sculpt_face_sets"):
            raise JsonRpcError(
                -32026,
                "Sculpt Face Sets overlay is unavailable",
                {
                    "blender_version": bpy.app.version_string,
                    "rna_path": (
                        "SpaceView3D.overlay.show_sculpt_face_sets"
                    ),
                },
            )
        previous_enabled = bool(overlay.show_sculpt_face_sets)
        try:
            overlay.show_sculpt_face_sets = enabled
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise JsonRpcError(
                -32026,
                "Cannot update the Sculpt Face Sets overlay",
                {
                    "enabled": enabled,
                    "reason": str(error),
                },
            ) from error
        actual = bool(overlay.show_sculpt_face_sets)
        if actual != enabled:
            raise JsonRpcError(
                -32026,
                "Blender did not update the Sculpt Face Sets overlay",
                {
                    "requested": enabled,
                    "actual": actual,
                },
            )
        return {
            "enabled": actual,
            "previous_enabled": previous_enabled,
            "show_overlays": bool(overlay.show_overlays),
            "opacity": float(overlay.sculpt_mode_face_sets_opacity),
            "rna_path": "SpaceView3D.overlay.show_sculpt_face_sets",
        }

    @staticmethod
    def _face_set_lasso_operator_kwargs(
        operator: Any,
        validated_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """Adapt OperatorMousePath items to the runtime RNA schema."""
        try:
            path_property = operator.get_rna_type().properties["path"]
            supported_fields = {
                item.identifier
                for item in path_property.fixed_type.properties
            }
        except (AttributeError, KeyError, RuntimeError, TypeError) as error:
            raise JsonRpcError(
                -32026,
                "Cannot inspect the Lasso Face Set path schema",
                {"reason": str(error)},
            ) from error
        if "loc" not in supported_fields:
            raise JsonRpcError(
                -32026,
                "The Lasso Face Set path schema has no loc field",
                {"supported_fields": sorted(supported_fields)},
            )

        ordered_fields = [
            field
            for field in ("name", "loc", "time")
            if field in supported_fields
        ]
        operator_path = []
        for index, point in enumerate(validated_kwargs["path"]):
            operator_point = {field: point[field] for field in ordered_fields}
            if "name" in ordered_fields and not operator_point["name"]:
                operator_point["name"] = f"face-set-lasso-{index:06d}"
            operator_path.append(operator_point)
        return (
            {
                "path": operator_path,
                "use_front_faces_only": validated_kwargs[
                    "use_front_faces_only"
                ],
            },
            ordered_fields,
        )

    def _scene_summary(self, scene: Any) -> dict[str, Any] | None:
        if scene is None:
            return None
        render = scene.render
        unit_settings = scene.unit_settings
        cursor = scene.cursor
        return {
            "name": scene.name,
            "frame_current": scene.frame_current,
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "camera": _name(scene.camera),
            "world": _name(scene.world),
            "object_count": len(scene.objects),
            "root_collection": scene.collection.name,
            "cursor": {
                "location": _float_list(cursor.location),
                "rotation": _float_list(cursor.rotation_euler),
            },
            "render": {
                "engine": render.engine,
                "resolution_x": render.resolution_x,
                "resolution_y": render.resolution_y,
                "resolution_percentage": render.resolution_percentage,
                "fps": render.fps,
            },
            "units": {
                "system": unit_settings.system,
                "scale_length": unit_settings.scale_length,
                "length_unit": unit_settings.length_unit,
            },
        }

    def _object_summary(
        self,
        obj: Any,
        view_layer: Any,
    ) -> dict[str, Any] | None:
        if obj is None:
            return None
        try:
            is_visible = obj.visible_get(view_layer=view_layer)
        except (RuntimeError, TypeError):
            is_visible = obj.visible_get()
        try:
            is_selected = obj.select_get(view_layer=view_layer)
        except (RuntimeError, TypeError):
            is_selected = obj.select_get()
        return {
            "name": obj.name,
            "type": obj.type,
            "mode": obj.mode,
            "data_name": _name(obj.data),
            "parent": _name(obj.parent),
            "location": _float_list(obj.location),
            "rotation_mode": obj.rotation_mode,
            "rotation": _float_list(obj.rotation_euler),
            "scale": _float_list(obj.scale),
            "dimensions": _float_list(obj.dimensions),
            "selected": is_selected,
            "visible": is_visible,
            "hide_viewport": obj.hide_viewport,
            "hide_render": obj.hide_render,
        }

    def _resolve_window(self, requested_index: int | None) -> tuple[Any, int]:
        window_manager = bpy.context.window_manager
        windows = list(window_manager.windows) if window_manager is not None else []
        if not windows:
            raise JsonRpcError(
                -32010,
                "No Blender window is available",
            )

        if requested_index is None:
            active_window = bpy.context.window
            if active_window in windows:
                return active_window, windows.index(active_window)
            return windows[0], 0
        if requested_index >= len(windows):
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {
                    "reason": "window_index is out of range",
                    "window_count": len(windows),
                },
            )
        return windows[requested_index], requested_index

    def _resolve_viewports(
        self,
        *,
        window_index: int | None,
        area_index: int | None,
        all_viewports: bool,
    ) -> list[dict[str, Any]]:
        if bpy.app.background:
            raise JsonRpcError(
                -32010,
                "3D viewports are unavailable in background mode",
            )
        if all_viewports and area_index is not None:
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {
                    "reason": (
                        "area_index cannot be combined with all_viewports"
                    )
                },
            )

        window_manager = bpy.context.window_manager
        windows = list(window_manager.windows) if window_manager is not None else []
        if not windows:
            raise JsonRpcError(-32010, "No Blender window is available")

        if all_viewports and window_index is None:
            indexed_windows = list(enumerate(windows))
        else:
            window, resolved_index = self._resolve_window(window_index)
            indexed_windows = [(resolved_index, window)]

        targets: list[dict[str, Any]] = []
        for resolved_window_index, window in indexed_windows:
            areas = list(window.screen.areas)
            if area_index is not None:
                if area_index >= len(areas) or areas[area_index].type != "VIEW_3D":
                    raise JsonRpcError(
                        -32602,
                        "Invalid params",
                        {
                            "reason": (
                                "area_index does not identify a 3D viewport"
                            ),
                            "area_count": len(areas),
                        },
                    )
                candidates = [(area_index, areas[area_index])]
            else:
                candidates = [
                    (index, area)
                    for index, area in enumerate(areas)
                    if area.type == "VIEW_3D"
                ]
                if not all_viewports and candidates:
                    active_area = bpy.context.area
                    active_match = next(
                        (
                            candidate
                            for candidate in candidates
                            if candidate[1] == active_area
                        ),
                        None,
                    )
                    candidates = [active_match or candidates[0]]

            for resolved_area_index, area in candidates:
                region = next(
                    (item for item in area.regions if item.type == "WINDOW"),
                    None,
                )
                space = area.spaces.active
                if region is None or space.type != "VIEW_3D":
                    continue
                targets.append(
                    {
                        "window": window,
                        "window_index": resolved_window_index,
                        "area": area,
                        "area_index": resolved_area_index,
                        "region": region,
                        "space": space,
                    }
                )

        if not targets:
            raise JsonRpcError(
                -32010,
                "No 3D viewport is available",
            )
        return targets

    def _viewport_summaries(self) -> list[dict[str, Any]]:
        if bpy.app.background:
            return []
        try:
            targets = self._resolve_viewports(
                window_index=None,
                area_index=None,
                all_viewports=True,
            )
        except JsonRpcError:
            return []
        return [self._viewport_summary(target) for target in targets]

    @staticmethod
    def _set_view_projection(
        target: dict[str, Any],
        projection: str,
    ) -> None:
        """Set a viewport projection deterministically instead of toggling."""
        region_3d = target["space"].region_3d
        try:
            region_3d.view_perspective = projection
            region_3d.update()
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise JsonRpcError(
                -32011,
                "Cannot set the 3D viewport projection",
                {
                    "projection": projection,
                    "reason": str(error),
                },
            ) from error
        actual = str(region_3d.view_perspective)
        if actual != projection:
            raise JsonRpcError(
                -32011,
                "Blender did not apply the 3D viewport projection",
                {
                    "requested": projection,
                    "actual": actual,
                },
            )

    @staticmethod
    def _viewport_summary(target: dict[str, Any]) -> dict[str, Any]:
        space = target["space"]
        region_3d = space.region_3d
        return {
            "window_index": target["window_index"],
            "area_index": target["area_index"],
            "area_width": target["area"].width,
            "area_height": target["area"].height,
            "view_perspective": region_3d.view_perspective,
            "view_location": _float_list(region_3d.view_location),
            "view_rotation": _float_list(region_3d.view_rotation),
            "view_distance": region_3d.view_distance,
            "shading": space.shading.type,
            "camera": _name(space.camera),
        }

    def _enter_sculpt_mode_transaction(
        self,
        *,
        window: Any,
        window_index: int,
        area_index: int | None,
        workspace: Any,
        hide_viewport_ui: bool,
    ) -> Generator[float, None, dict[str, Any]]:
        """Complete workspace, mode, and UI changes across timer ticks."""
        try:
            window.workspace = workspace
        except (AttributeError, RuntimeError, TypeError) as error:
            raise JsonRpcError(
                -32021,
                "Cannot switch to the Sculpting workspace",
                {"reason": str(error)},
            ) from error

        workspace_checks = yield from self._wait_for_workspace_stable(
            window,
            workspace,
        )
        target = self._resolve_window_viewport(
            window=window,
            window_index=window_index,
            area_index=area_index,
        )
        locator = self._viewport_locator(target)
        self._enter_sculpt_mode_in_target(target)
        target, viewport_checks = yield from self._wait_for_viewport_stable(
            locator,
            require_sculpt=True,
            phase="sculpt_mode",
            error_code=-32021,
        )
        with bpy.context.temp_override(
            window=target["window"],
            area=target["area"],
            region=target["region"],
        ):
            active_object = bpy.context.sculpt_object
            if active_object is None:
                raise JsonRpcError(
                    -32021,
                    "Blender did not enter Sculpt Mode",
                )
            active_summary = self._object_summary(
                active_object,
                bpy.context.view_layer,
            )

        viewport_ui: dict[str, Any] = {
            "hide_requested": hide_viewport_ui,
            "snapshot": None,
            "hidden_state": None,
            "application_methods": {},
            "verification_attempts": 0,
            "unsupported_properties": [],
        }
        if hide_viewport_ui:
            snapshot, unsupported = self._sculpt_viewport_ui_snapshot(target)
            hidden = {
                "space": {
                    name: False for name in snapshot["space"]
                },
                "overlay": {
                    name: False for name in snapshot["overlay"]
                },
            }
            try:
                (
                    hidden_methods,
                    hidden_state,
                    verification_attempts,
                ) = yield from self._apply_viewport_ui_until_stable(
                    locator,
                    hidden,
                    phase="hide",
                )
            except JsonRpcError as error:
                rollback = yield from self._rollback_viewport_ui(
                    locator,
                    snapshot,
                )
                raise self._viewport_ui_error_with_rollback(
                    error,
                    rollback,
                ) from error
            viewport_ui = {
                "hide_requested": True,
                "snapshot": snapshot,
                "hidden_state": hidden_state,
                "application_methods": hidden_methods,
                "verification_attempts": verification_attempts,
                "unsupported_properties": unsupported,
            }

        target = self._resolve_viewport_locator(locator)
        target["area"].tag_redraw()
        return {
            "workspace": workspace.name,
            "workspace_switch_deferred": False,
            "mode": active_object.mode,
            "active_object": active_summary,
            "viewport": self._viewport_summary(target),
            "viewport_ui": viewport_ui,
            "stabilization": {
                "workspace_checks": workspace_checks,
                "viewport_checks": viewport_checks,
                "settle_interval_seconds": _VIEWPORT_UI_SETTLE_INTERVAL,
            },
        }

    def _restore_sculpt_viewport_ui_transaction(
        self,
        *,
        window: Any,
        window_index: int,
        snapshot: dict[str, Any],
    ) -> Generator[float, None, dict[str, Any]]:
        """Restore and verify one snapshot after the target is stable."""
        target = self._resolve_window_viewport(
            window=window,
            window_index=window_index,
            area_index=snapshot["area_index"],
        )
        locator = self._viewport_locator(target)
        target, viewport_checks = yield from self._wait_for_viewport_stable(
            locator,
            require_sculpt=False,
            phase="restore_target",
            error_code=-32027,
        )
        previous, unsupported = self._sculpt_viewport_ui_snapshot(target)
        try:
            (
                application_methods,
                restored_state,
                verification_attempts,
            ) = yield from self._apply_viewport_ui_until_stable(
                locator,
                snapshot,
                phase="restore",
            )
        except JsonRpcError as error:
            rollback = yield from self._rollback_viewport_ui(
                locator,
                previous,
            )
            raise self._viewport_ui_error_with_rollback(
                error,
                rollback,
            ) from error

        target = self._resolve_viewport_locator(locator)
        target["area"].tag_redraw()
        return {
            "restored": True,
            "snapshot_schema_version": SCULPT_VIEWPORT_UI_SCHEMA_VERSION,
            "previous_state": {
                "space": previous["space"],
                "overlay": previous["overlay"],
            },
            "restored_state": restored_state,
            "application_methods": application_methods,
            "verification_attempts": verification_attempts,
            "unsupported_properties": unsupported,
            "viewport": self._viewport_summary(target),
            "stabilization": {
                "viewport_checks": viewport_checks,
                "settle_interval_seconds": _VIEWPORT_UI_SETTLE_INTERVAL,
            },
        }

    def _wait_for_workspace_stable(
        self,
        window: Any,
        workspace: Any,
    ) -> Generator[float, None, int]:
        """Wait until Blender has applied the requested workspace Screen."""
        tracker = StabilityTracker(required_consecutive=2)
        for _ in range(_WORKSPACE_STABILITY_MAX_CHECKS):
            yield _VIEWPORT_UI_SETTLE_INTERVAL
            signature = self._workspace_stability_signature(
                window,
                workspace,
            )
            if tracker.observe(signature):
                return tracker.checks
        raise JsonRpcError(
            -32021,
            "Sculpting workspace did not stabilize",
            {
                "checks": tracker.checks,
                "workspace": workspace.name,
                "actual_workspace": _name(getattr(window, "workspace", None)),
            },
        )

    def _wait_for_viewport_stable(
        self,
        locator: dict[str, Any],
        *,
        require_sculpt: bool,
        phase: str,
        error_code: int,
    ) -> Generator[float, None, tuple[dict[str, Any], int]]:
        """Wait for one final Area and its Region collection to settle."""
        tracker = StabilityTracker(required_consecutive=2)
        last_error: JsonRpcError | None = None
        latest_target: dict[str, Any] | None = None
        for _ in range(_VIEWPORT_STABILITY_MAX_CHECKS):
            yield _VIEWPORT_UI_SETTLE_INTERVAL
            try:
                target = self._resolve_viewport_locator(locator)
            except JsonRpcError as error:
                last_error = error
                signature = None
            else:
                active_object = target["window"].view_layer.objects.active
                mode_ready = (
                    not require_sculpt
                    or (
                        active_object is not None
                        and active_object.type == "MESH"
                        and active_object.mode == "SCULPT"
                    )
                )
                signature = (
                    self._viewport_stability_signature(target)
                    if mode_ready
                    else None
                )
                latest_target = target
            if tracker.observe(signature):
                assert latest_target is not None
                return latest_target, tracker.checks
        data: dict[str, Any] = {
            "phase": phase,
            "checks": tracker.checks,
            "require_sculpt": require_sculpt,
        }
        if last_error is not None:
            data["last_error"] = {
                "code": last_error.code,
                "message": last_error.message,
                "data": last_error.data,
            }
        raise JsonRpcError(
            error_code,
            "Sculpt viewport did not stabilize",
            data,
        )

    def _apply_viewport_ui_until_stable(
        self,
        locator: dict[str, Any],
        state: dict[str, Any],
        *,
        phase: str,
    ) -> Generator[
        float,
        None,
        tuple[dict[str, str], dict[str, dict[str, bool | None]], int],
    ]:
        """Apply once, then wait without restarting Region animations."""
        history: list[dict[str, Any]] = []
        latest_methods: dict[str, str] = {}
        latest_actual: dict[str, dict[str, bool | None]] = {
            "space": {},
            "overlay": {},
        }
        for attempt in range(1, _VIEWPORT_UI_VERIFY_ATTEMPTS + 1):
            target = self._resolve_viewport_locator(locator)
            try:
                latest_methods = self._apply_sculpt_viewport_ui_state(
                    target,
                    state,
                )
            except JsonRpcError as error:
                history.append(
                    {
                        "attempt": attempt,
                        "application_error": {
                            "code": error.code,
                            "message": error.message,
                            "data": error.data,
                        },
                    }
                )
                if attempt < _VIEWPORT_UI_VERIFY_ATTEMPTS:
                    yield _VIEWPORT_UI_SETTLE_INTERVAL
                    continue
                raise JsonRpcError(
                    -32027,
                    "Cannot apply stable Sculpt viewport UI settings",
                    {"phase": phase, "history": history},
                ) from error

            target["area"].tag_redraw()
            changed_region = any(
                method != "unchanged"
                and path.startswith("space.")
                and path.removeprefix("space.")
                in SPACE_UI_REQUIRED_REGION_TYPES
                for path, method in latest_methods.items()
            )
            minimum_settle_checks = (
                _VIEWPORT_UI_REGION_MIN_SETTLE_CHECKS
                if changed_region
                else 2
            )
            tracker = StabilityTracker(required_consecutive=2)
            for settle_check in range(
                1,
                _VIEWPORT_UI_SETTLE_MAX_CHECKS + 1,
            ):
                yield _VIEWPORT_UI_SETTLE_INTERVAL
                target = self._resolve_viewport_locator(locator)
                owners = {
                    "space": target["space"],
                    "overlay": getattr(target["space"], "overlay", None),
                }
                latest_actual, mismatches = (
                    read_sculpt_viewport_ui_properties(owners, state)
                )
                history.append(
                    {
                        "attempt": attempt,
                        "settle_check": settle_check,
                        "mismatches": mismatches,
                    }
                )
                signature = None if mismatches else ("matched",)
                if (
                    tracker.observe(signature)
                    and settle_check >= minimum_settle_checks
                ):
                    return latest_methods, latest_actual, attempt

        raise JsonRpcError(
            -32027,
            "Sculpt viewport UI settings did not remain stable",
            {
                "phase": phase,
                "attempts": _VIEWPORT_UI_VERIFY_ATTEMPTS,
                "settle_checks_per_attempt": (
                    _VIEWPORT_UI_SETTLE_MAX_CHECKS
                ),
                "history": history,
            },
        )

    def _rollback_viewport_ui(
        self,
        locator: dict[str, Any],
        state: dict[str, Any],
    ) -> Generator[float, None, dict[str, Any]]:
        """Best-effort verified rollback used by failed UI transactions."""
        try:
            _, actual, attempts = yield from (
                self._apply_viewport_ui_until_stable(
                    locator,
                    state,
                    phase="rollback",
                )
            )
        except JsonRpcError as error:
            return {
                "succeeded": False,
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "data": error.data,
                },
            }
        return {
            "succeeded": True,
            "attempts": attempts,
            "state": actual,
        }

    @staticmethod
    def _viewport_ui_error_with_rollback(
        error: JsonRpcError,
        rollback: dict[str, Any],
    ) -> JsonRpcError:
        data = (
            dict(error.data)
            if isinstance(error.data, dict)
            else {"original_data": error.data}
        )
        data["rollback"] = rollback
        return JsonRpcError(error.code, error.message, data)

    def _enter_sculpt_mode_in_target(self, target: dict[str, Any]) -> None:
        """Enter Sculpt Mode using one stable target context."""
        with bpy.context.temp_override(
            window=target["window"],
            area=target["area"],
            region=target["region"],
        ):
            active_object = bpy.context.view_layer.objects.active
            if active_object is None or active_object.type != "MESH":
                raise JsonRpcError(
                    -32020,
                    "A mesh active object is required for Sculpt Mode",
                )
            if not active_object.select_get():
                active_object.select_set(True)
            if active_object.mode != "SCULPT":
                if active_object.mode != "OBJECT":
                    self._call_operator(
                        bpy.ops.object.mode_set,
                        {"mode": "OBJECT"},
                        error_message="Cannot leave the current object mode",
                        error_code=-32021,
                    )
                self._call_operator(
                    bpy.ops.object.mode_set,
                    {"mode": "SCULPT"},
                    error_message="Cannot enter Sculpt Mode",
                    error_code=-32021,
                )
            if bpy.context.sculpt_object is None:
                raise JsonRpcError(
                    -32021,
                    "Blender did not enter Sculpt Mode",
                )

    def _resolve_window_viewport(
        self,
        *,
        window: Any,
        window_index: int,
        area_index: int | None,
    ) -> dict[str, Any]:
        """Resolve the final Screen's explicit or largest 3D Viewport."""
        window_manager = bpy.context.window_manager
        if window_manager is None or window not in list(window_manager.windows):
            raise JsonRpcError(-32010, "Blender window is unavailable")
        areas = list(window.screen.areas)
        if area_index is not None:
            if area_index >= len(areas) or areas[area_index].type != "VIEW_3D":
                raise JsonRpcError(
                    -32602,
                    "Invalid params",
                    {
                        "reason": (
                            "area_index does not identify a 3D viewport "
                            "in the target screen"
                        ),
                        "area_count": len(areas),
                    },
                )
            candidates = [(area_index, areas[area_index])]
        else:
            candidates = [
                (index, area)
                for index, area in enumerate(areas)
                if area.type == "VIEW_3D"
            ]
            candidates.sort(
                key=lambda item: (
                    item[1].width * item[1].height,
                    -item[0],
                ),
                reverse=True,
            )

        for resolved_area_index, area in candidates:
            region = next(
                (item for item in area.regions if item.type == "WINDOW"),
                None,
            )
            space = area.spaces.active
            if region is not None and space.type == "VIEW_3D":
                return {
                    "window": window,
                    "window_index": window_index,
                    "area": area,
                    "area_index": resolved_area_index,
                    "region": region,
                    "space": space,
                }
        raise JsonRpcError(-32010, "No 3D viewport is available")

    @staticmethod
    def _viewport_locator(target: dict[str, Any]) -> dict[str, Any]:
        return {
            "window": target["window"],
            "window_index": target["window_index"],
            "screen": target["window"].screen,
            "area": target["area"],
            "area_index": target["area_index"],
        }

    def _resolve_viewport_locator(
        self,
        locator: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve fresh Region and Space references for one stable Area."""
        window = locator["window"]
        window_manager = bpy.context.window_manager
        if window_manager is None or window not in list(window_manager.windows):
            raise JsonRpcError(-32027, "Sculpt viewport window disappeared")
        if window.screen != locator["screen"]:
            raise JsonRpcError(
                -32027,
                "Sculpt viewport screen changed during the transaction",
            )
        areas = list(window.screen.areas)
        area = locator["area"]
        if area not in areas or area.type != "VIEW_3D":
            raise JsonRpcError(
                -32027,
                "Sculpt viewport area changed during the transaction",
            )
        region = next(
            (item for item in area.regions if item.type == "WINDOW"),
            None,
        )
        space = area.spaces.active
        if region is None or space.type != "VIEW_3D":
            raise JsonRpcError(
                -32027,
                "Sculpt viewport Region or Space is unavailable",
            )
        return {
            "window": window,
            "window_index": locator["window_index"],
            "area": area,
            "area_index": areas.index(area),
            "region": region,
            "space": space,
        }

    def _workspace_stability_signature(
        self,
        window: Any,
        workspace: Any,
    ) -> tuple[Any, ...] | None:
        window_manager = bpy.context.window_manager
        if window_manager is None or window not in list(window_manager.windows):
            return None
        if window.workspace != workspace or window.screen is None:
            return None
        return (
            self._rna_pointer(workspace),
            self._rna_pointer(window.screen),
            tuple(
                (
                    self._rna_pointer(area),
                    area.type,
                    area.width,
                    area.height,
                    tuple(
                        (region.type, region.width, region.height)
                        for region in area.regions
                    ),
                )
                for area in window.screen.areas
            ),
        )

    def _viewport_stability_signature(
        self,
        target: dict[str, Any],
    ) -> tuple[Any, ...]:
        return (
            self._rna_pointer(target["window"].screen),
            self._rna_pointer(target["area"]),
            self._rna_pointer(target["space"]),
            tuple(
                (region.type, region.width, region.height)
                for region in target["area"].regions
            ),
        )

    @staticmethod
    def _rna_pointer(value: Any) -> int:
        pointer = getattr(value, "as_pointer", None)
        return int(pointer()) if callable(pointer) else id(value)

    @staticmethod
    def _sculpt_viewport_ui_snapshot(
        target: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        """Capture only the UI values managed by the Sculpt workflow."""
        space = target["space"]
        overlay = getattr(space, "overlay", None)
        unsupported: list[str] = []
        space_values: dict[str, bool] = {}
        overlay_values: dict[str, bool] = {}
        region_types = {region.type for region in target["area"].regions}
        for name in SPACE_UI_PROPERTIES:
            required_region = SPACE_UI_REQUIRED_REGION_TYPES.get(name)
            if (
                required_region is not None
                and required_region not in region_types
            ):
                unsupported.append(f"space.{name}")
            elif hasattr(space, name):
                space_values[name] = bool(getattr(space, name))
            else:
                unsupported.append(f"space.{name}")
        for name in OVERLAY_UI_PROPERTIES:
            if overlay is not None and hasattr(overlay, name):
                overlay_values[name] = bool(getattr(overlay, name))
            else:
                unsupported.append(f"overlay.{name}")
        return (
            {
                "schema_version": SCULPT_VIEWPORT_UI_SCHEMA_VERSION,
                "window_index": target["window_index"],
                "area_index": target["area_index"],
                "space": space_values,
                "overlay": overlay_values,
            },
            unsupported,
        )

    def _apply_sculpt_viewport_ui_state(
        self,
        target: dict[str, Any],
        state: dict[str, Any],
    ) -> dict[str, str]:
        """Apply and verify one previously validated allowlisted state."""
        owners = {
            "space": target["space"],
            "overlay": getattr(target["space"], "overlay", None),
        }

        def toggle_region(region_type: str) -> None:
            self._call_operator(
                bpy.ops.screen.region_toggle,
                {"region_type": region_type},
                error_message=(
                    f"Cannot toggle Sculpt viewport region {region_type}"
                ),
                error_code=-32027,
            )

        with bpy.context.temp_override(
            window=target["window"],
            area=target["area"],
            region=target["region"],
        ):
            return apply_sculpt_viewport_ui_properties(
                owners,
                state,
                region_toggle=toggle_region,
                blender_version=bpy.app.version_string,
            )

    @contextmanager
    def _temporary_object_mode(
        self,
        targets: list[dict[str, Any]],
    ) -> Iterator[None]:
        mode_states: list[dict[str, Any]] = []
        seen_windows: set[int] = set()
        try:
            for target in targets:
                window_key = id(target["window"])
                if window_key in seen_windows:
                    continue
                seen_windows.add(window_key)

                with bpy.context.temp_override(
                    window=target["window"],
                    area=target["area"],
                    region=target["region"],
                ):
                    view_layer = bpy.context.view_layer
                    active_object = (
                        view_layer.objects.active
                        if view_layer is not None
                        else None
                    )
                    mode = getattr(active_object, "mode", "OBJECT")
                    sculpt_settings = getattr(
                        getattr(bpy.context, "tool_settings", None),
                        "sculpt",
                        None,
                    )
                    dyntopo_enabled = bool(
                        mode == "SCULPT"
                        and getattr(
                            active_object,
                            "use_dynamic_topology_sculpting",
                            False,
                        )
                    )
                    state = {
                        "target": target,
                        "active_object": active_object,
                        "mode": mode,
                        "switched": False,
                        "dyntopo_enabled": dyntopo_enabled,
                        "detail_size": (
                            float(sculpt_settings.detail_size)
                            if dyntopo_enabled and sculpt_settings is not None
                            else None
                        ),
                        "detail_method": (
                            str(sculpt_settings.detail_type_method)
                            if dyntopo_enabled and sculpt_settings is not None
                            else None
                        ),
                    }
                    mode_states.append(state)
                    if active_object is None or mode == "OBJECT":
                        continue

                    self._call_operator(
                        bpy.ops.object.mode_set,
                        {"mode": "OBJECT"},
                        error_message=(
                            "Cannot switch to Object Mode before changing "
                            "the view"
                        ),
                    )
                    state["switched"] = True
                    if active_object.mode != "OBJECT":
                        raise JsonRpcError(
                            -32011,
                            "Cannot switch to Object Mode before changing "
                            "the view",
                            {
                                "expected_mode": "OBJECT",
                                "actual_mode": active_object.mode,
                            },
                        )

            yield
        finally:
            restore_error: JsonRpcError | None = None
            for state in reversed(mode_states):
                if not state["switched"]:
                    continue
                target = state["target"]
                active_object = state["active_object"]
                mode = state["mode"]
                try:
                    with bpy.context.temp_override(
                        window=target["window"],
                        area=target["area"],
                        region=target["region"],
                    ):
                        view_layer = bpy.context.view_layer
                        if view_layer is None:
                            raise JsonRpcError(
                                -32011,
                                "Cannot restore the previous mode after "
                                "changing the view",
                                {"expected_mode": mode},
                            )
                        if view_layer.objects.active != active_object:
                            view_layer.objects.active = active_object
                        self._call_operator(
                            bpy.ops.object.mode_set,
                            {"mode": mode},
                            error_message=(
                                "Cannot restore the previous mode after "
                                "changing the view"
                            ),
                        )
                        if active_object.mode != mode:
                            raise JsonRpcError(
                                -32011,
                                "Cannot restore the previous mode after "
                                "changing the view",
                                {
                                    "expected_mode": mode,
                                    "actual_mode": active_object.mode,
                                },
                            )
                        if state["dyntopo_enabled"]:
                            if not bool(
                                active_object.use_dynamic_topology_sculpting
                            ):
                                self._call_operator(
                                    bpy.ops.sculpt.dynamic_topology_toggle,
                                    {},
                                    error_message=(
                                        "Cannot restore Dyntopo after changing "
                                        "the view"
                                    ),
                                )
                            if not bool(
                                active_object.use_dynamic_topology_sculpting
                            ):
                                raise JsonRpcError(
                                    -32011,
                                    "Cannot restore Dyntopo after changing "
                                    "the view",
                                )
                            sculpt_settings = getattr(
                                bpy.context.tool_settings,
                                "sculpt",
                                None,
                            )
                            if sculpt_settings is None:
                                raise JsonRpcError(
                                    -32011,
                                    "Cannot restore Dyntopo settings after "
                                    "changing the view",
                                )
                            sculpt_settings.detail_size = state["detail_size"]
                            sculpt_settings.detail_type_method = state[
                                "detail_method"
                            ]
                except (
                    AttributeError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:
                    candidate = JsonRpcError(
                        -32011,
                        "Cannot restore the previous mode after changing "
                        "the view",
                        {
                            "expected_mode": mode,
                            "reason": str(error),
                        },
                    )
                    if restore_error is None:
                        restore_error = candidate
                except JsonRpcError as error:
                    if restore_error is None:
                        restore_error = error
            if restore_error is not None:
                raise restore_error

    @staticmethod
    def _call_operator(
        operator: Any,
        kwargs: dict[str, Any],
        *,
        error_message: str,
        error_code: int = -32011,
    ) -> Any:
        try:
            if not operator.poll():
                raise JsonRpcError(error_code, error_message)
            result = operator(**kwargs)
        except JsonRpcError:
            raise
        except (RuntimeError, TypeError, ValueError) as error:
            raise JsonRpcError(
                error_code,
                error_message,
                {"reason": str(error)},
            ) from error
        if "FINISHED" not in result:
            raise JsonRpcError(
                error_code,
                error_message,
                {"operator_status": sorted(str(item) for item in result)},
            )
        return result


def _parse_focus_roi(
    value: Any,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    if not isinstance(value, dict):
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {"reason": "roi must be an object"},
        )
    _reject_unknown(value, {"x_min", "y_min", "x_max", "y_max"})
    parsed: dict[str, float] = {}
    for name in ("x_min", "y_min", "x_max", "y_max"):
        item = value.get(name)
        if (
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
        ):
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {"reason": f"roi.{name} must be a finite number"},
            )
        parsed[name] = float(item)
    if not (
        0.0 <= parsed["x_min"] < parsed["x_max"] <= image_width
        and 0.0 <= parsed["y_min"] < parsed["y_max"] <= image_height
    ):
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {
                "reason": "roi must be a non-empty rectangle inside image",
                "image_width": image_width,
                "image_height": image_height,
            },
        )
    return parsed


def _expanded_focus_roi(
    roi: dict[str, float],
    *,
    image_width: int,
    image_height: int,
    margin_ratio: float,
    maximum_zoom_factor: float,
) -> dict[str, float]:
    center_x = (roi["x_min"] + roi["x_max"]) * 0.5
    center_y = (roi["y_min"] + roi["y_max"]) * 0.5
    width = max(
        (roi["x_max"] - roi["x_min"]) * (1.0 + 2.0 * margin_ratio),
        image_width / maximum_zoom_factor,
    )
    height = max(
        (roi["y_max"] - roi["y_min"]) * (1.0 + 2.0 * margin_ratio),
        image_height / maximum_zoom_factor,
    )
    x_min, x_max = _bounded_interval(center_x, width, float(image_width))
    y_min, y_max = _bounded_interval(center_y, height, float(image_height))
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
    }


def _bounded_interval(
    center: float,
    length: float,
    limit: float,
) -> tuple[float, float]:
    bounded_length = min(max(length, 2.0), limit)
    start = center - bounded_length * 0.5
    end = center + bounded_length * 0.5
    if start < 0.0:
        end -= start
        start = 0.0
    if end > limit:
        start -= end - limit
        end = limit
    return max(0.0, start), min(limit, end)


def _image_roi_to_region_border(
    roi: dict[str, float],
    *,
    image_width: int,
    image_height: int,
    region_width: int,
    region_height: int,
) -> dict[str, int]:
    scale_x = image_width / region_width
    scale_y = image_height / region_height
    x_min = math.floor(roi["x_min"] / scale_x)
    x_max = math.ceil(roi["x_max"] / scale_x)
    y_min = math.floor((image_height - roi["y_max"]) / scale_y)
    y_max = math.ceil((image_height - roi["y_min"]) / scale_y)
    x_min = min(max(x_min, 0), region_width - 2)
    x_max = min(max(x_max, x_min + 1), region_width - 1)
    y_min = min(max(y_min, 0), region_height - 2)
    y_max = min(max(y_max, y_min + 1), region_height - 1)
    return {
        "xmin": x_min,
        "xmax": x_max,
        "ymin": y_min,
        "ymax": y_max,
    }


def _viewport_state_snapshot(
    target: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    region = target["region"]
    space = target["space"]
    region_3d = space.region_3d
    return {
        "schema_version": "viewport-state/v1",
        "window_index": int(target["window_index"]),
        "area_index": int(target["area_index"]),
        "region": {
            "width": int(region.width),
            "height": int(region.height),
        },
        "image": {
            "width": int(image_width),
            "height": int(image_height),
            "origin": "TOP_LEFT",
        },
        "coordinate_scale": {
            "x": float(image_width / region.width),
            "y": float(image_height / region.height),
        },
        "view_perspective": str(region_3d.view_perspective),
        "view_location": _float_list(region_3d.view_location),
        "view_rotation": _float_list(region_3d.view_rotation),
        "view_distance": float(region_3d.view_distance),
        "view_camera_offset": _float_list(
            region_3d.view_camera_offset
        ),
        "view_camera_zoom": float(region_3d.view_camera_zoom),
        "lens": float(space.lens),
        "perspective_matrix": [
            [float(item) for item in row]
            for row in region_3d.perspective_matrix
        ],
    }


def _parse_viewport_state_snapshot(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != "viewport-state/v1":
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {"reason": "snapshot schema_version must be viewport-state/v1"},
        )
    parsed = dict(value)
    for name in ("window_index", "area_index"):
        item = parsed.get(name)
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {"reason": f"snapshot.{name} must be non-negative integer"},
            )
    region = _snapshot_dimensions(parsed.get("region"), "snapshot.region")
    image = _snapshot_dimensions(parsed.get("image"), "snapshot.image")
    if parsed["image"].get("origin") != "TOP_LEFT":
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {"reason": "snapshot.image.origin must be TOP_LEFT"},
        )
    projection = parsed.get("view_perspective")
    if projection not in {"ORTHO", "PERSP", "CAMERA"}:
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {"reason": "snapshot has invalid view_perspective"},
        )
    parsed["region"] = region
    parsed["image"] = {**image, "origin": "TOP_LEFT"}
    parsed["view_location"] = _snapshot_number_list(
        parsed.get("view_location"),
        "snapshot.view_location",
        length=3,
    )
    parsed["view_rotation"] = _snapshot_number_list(
        parsed.get("view_rotation"),
        "snapshot.view_rotation",
        length=4,
    )
    parsed["view_camera_offset"] = _snapshot_number_list(
        parsed.get("view_camera_offset"),
        "snapshot.view_camera_offset",
        length=2,
    )
    for name in ("view_distance", "view_camera_zoom", "lens"):
        item = parsed.get(name)
        if (
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
        ):
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {"reason": f"snapshot.{name} must be a finite number"},
            )
        parsed[name] = float(item)
    if parsed["view_distance"] < 0.0 or parsed["lens"] <= 0.0:
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {"reason": "snapshot distance and lens are invalid"},
        )
    return parsed


def _snapshot_dimensions(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {"reason": f"{label} must be an object"},
        )
    result: dict[str, int] = {}
    for name in ("width", "height"):
        item = value.get(name)
        if not isinstance(item, int) or isinstance(item, bool) or item < 2:
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {"reason": f"{label}.{name} must be an integer >= 2"},
            )
        result[name] = item
    return result


def _snapshot_number_list(
    value: Any,
    label: str,
    *,
    length: int,
) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {"reason": f"{label} must contain {length} numbers"},
        )
    result: list[float] = []
    for item in value:
        if (
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
        ):
            raise JsonRpcError(
                -32602,
                "Invalid params",
                {"reason": f"{label} must contain finite numbers"},
            )
        result.append(float(item))
    return result


def _apply_viewport_state_snapshot(
    target: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    require_region_match: bool,
) -> None:
    parsed = _parse_viewport_state_snapshot(snapshot)
    region = target["region"]
    expected_region = parsed["region"]
    if require_region_match and (
        region.width != expected_region["width"]
        or region.height != expected_region["height"]
    ):
        raise JsonRpcError(
            -32014,
            "Viewport dimensions changed before state restoration",
            {
                "expected": expected_region,
                "actual": {
                    "width": int(region.width),
                    "height": int(region.height),
                },
            },
        )
    space = target["space"]
    region_3d = space.region_3d
    try:
        space.lens = parsed["lens"]
        region_3d.view_rotation = parsed["view_rotation"]
        region_3d.view_location = parsed["view_location"]
        region_3d.view_distance = parsed["view_distance"]
        region_3d.view_camera_offset = parsed["view_camera_offset"]
        region_3d.view_camera_zoom = parsed["view_camera_zoom"]
        region_3d.view_perspective = parsed["view_perspective"]
        region_3d.update()
        target["area"].tag_redraw()
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise JsonRpcError(
            -32014,
            "Cannot apply the viewport state",
            {"reason": str(error)},
        ) from error


def _viewport_state_mismatches(
    expected: dict[str, Any],
    actual: dict[str, Any],
) -> dict[str, Any]:
    mismatches: dict[str, Any] = {}
    if expected["view_perspective"] != actual["view_perspective"]:
        mismatches["view_perspective"] = {
            "expected": expected["view_perspective"],
            "actual": actual["view_perspective"],
        }
    for name, tolerance in (
        ("view_location", 1e-5),
        ("view_rotation", 1e-5),
        ("view_camera_offset", 1e-5),
    ):
        expected_values = expected[name]
        actual_values = actual[name]
        direct = max(
            abs(left - right)
            for left, right in zip(expected_values, actual_values, strict=True)
        )
        if name == "view_rotation":
            negated = max(
                abs(left + right)
                for left, right in zip(
                    expected_values,
                    actual_values,
                    strict=True,
                )
            )
            direct = min(direct, negated)
        if direct > tolerance:
            mismatches[name] = {
                "expected": expected_values,
                "actual": actual_values,
            }
    for name in ("view_distance", "view_camera_zoom", "lens"):
        tolerance = max(1e-5, abs(expected[name]) * 1e-6)
        if abs(expected[name] - actual[name]) > tolerance:
            mismatches[name] = {
                "expected": expected[name],
                "actual": actual[name],
            }
    return mismatches


def _viewport_world_samples(
    *,
    region: Any,
    region_3d: Any,
    image_width: int,
    image_height: int,
) -> list[Any]:
    image_points = (
        (0.0, 0.0),
        (float(image_width - 1), 0.0),
        (0.0, float(image_height - 1)),
        (float(image_width - 1), float(image_height - 1)),
    )
    depth = region_3d.view_location.copy()
    return [
        view3d_utils.region_2d_to_location_3d(
            region,
            region_3d,
            _image_to_region_coordinate(
                point,
                image_width=image_width,
                image_height=image_height,
                region_width=region.width,
                region_height=region.height,
            ),
            depth,
        )
        for point in image_points
    ]


def _viewport_image_affine(
    *,
    region: Any,
    region_3d: Any,
    sample_points: list[Any],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    projected: list[tuple[float, float]] = []
    for world_point in sample_points:
        region_point = view3d_utils.location_3d_to_region_2d(
            region,
            region_3d,
            world_point,
            default=None,
        )
        if region_point is None:
            raise JsonRpcError(
                -32013,
                "Cannot project the focused viewport transform",
            )
        projected.append(
            _region_to_image_coordinate(
                (float(region_point[0]), float(region_point[1])),
                image_width=image_width,
                image_height=image_height,
                region_width=region.width,
                region_height=region.height,
            )
        )
    span_x = float(image_width - 1)
    span_y = float(image_height - 1)
    q0, qx, qy, qxy = projected
    matrix = [
        [
            (qx[0] - q0[0]) / span_x,
            (qy[0] - q0[0]) / span_y,
            q0[0],
        ],
        [
            (qx[1] - q0[1]) / span_x,
            (qy[1] - q0[1]) / span_y,
            q0[1],
        ],
    ]
    predicted = _transform_point(
        (span_x, span_y),
        matrix,
    )
    residual = math.dist(predicted, qxy)
    determinant = (
        matrix[0][0] * matrix[1][1]
        - matrix[0][1] * matrix[1][0]
    )
    if (
        not math.isfinite(residual)
        or residual > 0.75
        or not math.isfinite(determinant)
        or abs(determinant) <= 1e-12
    ):
        raise JsonRpcError(
            -32013,
            "Focused viewport is not a stable affine image transform",
            {
                "residual_pixels": residual,
                "determinant": determinant,
            },
        )
    inverse = _inverse_affine(matrix)
    scale_x = math.hypot(matrix[0][0], matrix[1][0])
    scale_y = math.hypot(matrix[0][1], matrix[1][1])
    return {
        "kind": "AFFINE_2D",
        "matrix": [
            [round(value, 12) for value in row]
            for row in matrix
        ],
        "inverse_matrix": [
            [round(value, 12) for value in row]
            for row in inverse
        ],
        "scale": {
            "x": round(scale_x, 9),
            "y": round(scale_y, 9),
            "geometric_mean": round(
                math.sqrt(abs(determinant)),
                9,
            ),
        },
        "determinant": round(determinant, 12),
        "affine_residual_pixels": round(residual, 9),
    }


def _image_to_region_coordinate(
    point: tuple[float, float],
    *,
    image_width: int,
    image_height: int,
    region_width: int,
    region_height: int,
) -> tuple[float, float]:
    scale_x = image_width / region_width
    scale_y = image_height / region_height
    mouse_x = (point[0] + 0.5) / scale_x - 0.5
    logical_top_y = (point[1] + 0.5) / scale_y - 0.5
    mouse_y = region_height - 1.0 - logical_top_y
    return mouse_x, mouse_y


def _region_to_image_coordinate(
    point: tuple[float, float],
    *,
    image_width: int,
    image_height: int,
    region_width: int,
    region_height: int,
) -> tuple[float, float]:
    scale_x = image_width / region_width
    scale_y = image_height / region_height
    image_x = (point[0] + 0.5) * scale_x - 0.5
    logical_top_y = region_height - 1.0 - point[1]
    image_y = (logical_top_y + 0.5) * scale_y - 0.5
    return image_x, image_y


def _inverse_affine(matrix: list[list[float]]) -> list[list[float]]:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    determinant = a * e - b * d
    return [
        [
            e / determinant,
            -b / determinant,
            (b * f - e * c) / determinant,
        ],
        [
            -d / determinant,
            a / determinant,
            (d * c - a * f) / determinant,
        ],
    ]


def _transform_point(
    point: tuple[float, float],
    matrix: list[list[float]],
) -> tuple[float, float]:
    return (
        matrix[0][0] * point[0]
        + matrix[0][1] * point[1]
        + matrix[0][2],
        matrix[1][0] * point[0]
        + matrix[1][1] * point[1]
        + matrix[1][2],
    )


def _transform_roi(
    roi: dict[str, float],
    matrix: list[list[float]],
) -> dict[str, float]:
    points = [
        _transform_point(point, matrix)
        for point in (
            (roi["x_min"], roi["y_min"]),
            (roi["x_max"], roi["y_min"]),
            (roi["x_min"], roi["y_max"]),
            (roi["x_max"], roi["y_max"]),
        )
    ]
    return {
        "x_min": round(min(point[0] for point in points), 6),
        "y_min": round(min(point[1] for point in points), 6),
        "x_max": round(max(point[0] for point in points), 6),
        "y_max": round(max(point[1] for point in points), 6),
    }


def _reject_unknown(params: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {"unknown_parameters": unknown},
        )


def _required_string(params: dict[str, Any], name: str) -> str:
    value = params.get(name)
    if not isinstance(value, str) or not value.strip():
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {"reason": f"{name} must be a non-empty string"},
        )
    return value.strip()


def _string(params: dict[str, Any], name: str, *, default: str) -> str:
    if name not in params:
        return default
    value = params[name]
    if not isinstance(value, str):
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {"reason": f"{name} must be a string"},
        )
    return value


def _boolean(params: dict[str, Any], name: str, *, default: bool) -> bool:
    if name not in params:
        return default
    value = params[name]
    if not isinstance(value, bool):
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {"reason": f"{name} must be a boolean"},
        )
    return value


def _integer(
    params: dict[str, Any],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if name not in params:
        return default
    value = params[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {"reason": f"{name} must be an integer"},
        )
    if not minimum <= value <= maximum:
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {
                "reason": f"{name} is out of range",
                "minimum": minimum,
                "maximum": maximum,
            },
        )
    return value


def _optional_integer(
    params: dict[str, Any],
    name: str,
    *,
    minimum: int,
) -> int | None:
    if name not in params:
        return None
    value = params[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {"reason": f"{name} must be an integer"},
        )
    if value < minimum:
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {"reason": f"{name} must be at least {minimum}"},
        )
    return value


def _optional_number(
    params: dict[str, Any],
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    if name not in params:
        return None
    value = params[name]
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not minimum <= float(value) <= maximum
    ):
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {
                "reason": f"{name} must be a number in range",
                "minimum": minimum,
                "maximum": maximum,
            },
        )
    return float(value)


def _name(value: Any) -> str | None:
    return getattr(value, "name", None) if value is not None else None


def _float_list(value: Any) -> list[float]:
    return [float(item) for item in value]


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _crop_png_with_imbuf(
    *,
    source_path: Path,
    source_png: bytes,
    output_path: Path,
    source_width: int,
    source_height: int,
    geometry: CropGeometry,
) -> str:
    """Crop a screenshot with the ImBuf API available in this Blender."""
    load_from_buffer = getattr(imbuf, "load_from_buffer", None)
    load_api = select_imbuf_load_api(
        bpy.app.version,
        has_load_from_buffer=callable(load_from_buffer),
    )
    image: Any | None = None
    try:
        if load_api == "load_from_buffer":
            image = load_from_buffer(source_png)
        else:
            image = imbuf.load(str(source_path))
        if image is None:
            raise PngCropError("ImBuf could not load the window screenshot")
        if tuple(image.size) != (source_width, source_height):
            raise PngCropError(
                "ImBuf source dimensions do not match the screenshot"
            )

        minimum = (
            geometry.left,
            source_height - geometry.bottom,
        )
        maximum = (
            geometry.right - 1,
            source_height - geometry.top - 1,
        )
        image.crop(minimum, maximum)
        if tuple(image.size) != (geometry.width, geometry.height):
            raise PngCropError(
                "ImBuf crop dimensions do not match the requested region"
            )
        imbuf.write(image, filepath=str(output_path))
    except PngCropError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise PngCropError(
            f"ImBuf {load_api} crop failed: {error}"
        ) from error
    finally:
        if image is not None:
            image.free()
    return load_api


def _absolute_png_path(value: str) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if not expanded.is_absolute():
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {"reason": "filepath must be absolute"},
        )
    if expanded.suffix.lower() != ".png":
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {"reason": "filepath must end in .png"},
        )
    return expanded.resolve()


def _absolute_blend_path(value: str) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if not expanded.is_absolute():
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {"reason": "filepath must be absolute"},
        )
    if expanded.suffix.lower() != ".blend":
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {"reason": "filepath must end in .blend"},
        )
    return expanded.resolve()


def _validate_blend_file(filepath: Path) -> None:
    if not filepath.is_file():
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {"reason": "filepath must identify an existing .blend file"},
        )
    try:
        with filepath.open("rb") as stream:
            header = stream.read(7)
        if header.startswith(b"\x1f\x8b"):
            with gzip.open(filepath, "rb") as stream:
                header = stream.read(7)
    except (EOFError, OSError) as error:
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {
                "reason": "filepath is not a readable .blend file",
                "detail": str(error),
            },
        ) from error
    if header != b"BLENDER" and not header.startswith(b"\x28\xb5\x2f\xfd"):
        raise JsonRpcError(
            -32602,
            "Invalid params",
            {"reason": "filepath does not contain Blender file data"},
        )


def _paths_equal(first: Path, second: Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError:
        return os.path.normcase(str(first.resolve())) == os.path.normcase(
            str(second.resolve())
        )
