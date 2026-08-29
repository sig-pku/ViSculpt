"""Blender registration, preferences, and UI."""

from typing import Any

import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, StringProperty

from .runtime import runtime

ADDON_ID = __package__


def _preferences(context: Any) -> Any | None:
    addons = context.preferences.addons
    addon = addons.get(ADDON_ID)
    if addon is not None:
        return addon.preferences

    short_name = ADDON_ID.rsplit(".", maxsplit=1)[-1]
    for item in addons:
        if item.module == short_name or item.module.endswith(f".{short_name}"):
            return item.preferences
    return None


def _start_from_preferences(context: Any) -> dict[str, Any]:
    preferences = _preferences(context)
    if preferences is None:
        raise RuntimeError("Geometry Editing RPC preferences are unavailable")
    return runtime.start(
        port=preferences.port,
        access_token=preferences.access_token,
        request_timeout=preferences.request_timeout,
    )


class GeometryEditingRpcPreferences(bpy.types.AddonPreferences):
    """Persistent Add-on settings."""

    bl_idname = ADDON_ID

    auto_start: BoolProperty(
        name="Start RPC server automatically",
        description="Start the loopback server after enabling the Add-on",
        default=True,
    )
    port: IntProperty(
        name="Port",
        description="TCP port on 127.0.0.1",
        default=8765,
        min=1024,
        max=65535,
    )
    access_token: StringProperty(
        name="Access token",
        description="Optional bearer token required by POST /rpc",
        default="",
        subtype="PASSWORD",
    )
    request_timeout: FloatProperty(
        name="Request timeout",
        description="Maximum seconds to wait for Blender's main thread",
        default=30.0,
        min=1.0,
        max=300.0,
        unit="TIME",
    )

    def draw(self, context: Any) -> None:
        _draw_settings(self.layout, context)


class GEOMETRY_EDITING_RPC_OT_start(bpy.types.Operator):
    """Start the local RPC server."""

    bl_idname = "geometry_editing_rpc.start"
    bl_label = "Start RPC Server"
    bl_options = {"REGISTER"}

    def execute(self, context: Any) -> set[str]:
        try:
            status = _start_from_preferences(context)
        except Exception as error:  # noqa: BLE001
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"RPC server listening on {status['endpoint']}")
        return {"FINISHED"}


class GEOMETRY_EDITING_RPC_OT_stop(bpy.types.Operator):
    """Stop the local RPC server."""

    bl_idname = "geometry_editing_rpc.stop"
    bl_label = "Stop RPC Server"
    bl_options = {"REGISTER"}

    def execute(self, context: Any) -> set[str]:
        runtime.stop()
        self.report({"INFO"}, "RPC server stopped")
        return {"FINISHED"}


class GEOMETRY_EDITING_RPC_OT_restart(bpy.types.Operator):
    """Restart the local RPC server with current settings."""

    bl_idname = "geometry_editing_rpc.restart"
    bl_label = "Restart RPC Server"
    bl_options = {"REGISTER"}

    def execute(self, context: Any) -> set[str]:
        preferences = _preferences(context)
        if preferences is None:
            self.report({"ERROR"}, "Add-on preferences are unavailable")
            return {"CANCELLED"}
        try:
            status = runtime.restart(
                port=preferences.port,
                access_token=preferences.access_token,
                request_timeout=preferences.request_timeout,
            )
        except Exception as error:  # noqa: BLE001
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, f"RPC server listening on {status['endpoint']}")
        return {"FINISHED"}


class VIEW3D_PT_geometry_editing_rpc(bpy.types.Panel):
    """Show RPC controls in the 3D Viewport sidebar."""

    bl_label = "Geometry Editing RPC"
    bl_idname = "VIEW3D_PT_geometry_editing_rpc"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "RPC"

    def draw(self, context: Any) -> None:
        layout = self.layout
        status = runtime.status()

        status_box = layout.box()
        status_box.label(
            text="Running" if status["running"] else "Stopped",
            icon="CHECKMARK" if status["running"] else "PAUSE",
        )
        if status["endpoint"]:
            status_box.label(text=status["endpoint"])
        row = layout.row(align=True)
        if status["running"]:
            row.operator(
                GEOMETRY_EDITING_RPC_OT_stop.bl_idname,
                icon="PAUSE",
            )
            row.operator(
                GEOMETRY_EDITING_RPC_OT_restart.bl_idname,
                icon="FILE_REFRESH",
            )
        else:
            row.operator(
                GEOMETRY_EDITING_RPC_OT_start.bl_idname,
                icon="PLAY",
            )

        _draw_settings(layout, context)


def _draw_settings(layout: Any, context: Any) -> None:
    preferences = _preferences(context)
    if preferences is None:
        layout.label(text="Preferences unavailable", icon="ERROR")
        return
    column = layout.column(align=True)
    column.prop(preferences, "auto_start")
    column.prop(preferences, "port")
    column.prop(preferences, "request_timeout")
    column.prop(preferences, "access_token")
    if not preferences.access_token:
        layout.label(
            text="No access token (loopback only)",
            icon="INFO",
        )
    if runtime.status()["running"]:
        layout.label(text="Restart to apply changed settings", icon="INFO")


def _auto_start() -> None:
    preferences = _preferences(bpy.context)
    if preferences is None or not preferences.auto_start:
        return None
    if runtime.status()["running"]:
        return None
    try:
        status = _start_from_preferences(bpy.context)
        print(f"[Geometry Editing RPC] Listening on {status['endpoint']}")
    except Exception as error:  # noqa: BLE001
        print(f"[Geometry Editing RPC] Failed to start: {error}")
    return None


_CLASSES = (
    GeometryEditingRpcPreferences,
    GEOMETRY_EDITING_RPC_OT_start,
    GEOMETRY_EDITING_RPC_OT_stop,
    GEOMETRY_EDITING_RPC_OT_restart,
    VIEW3D_PT_geometry_editing_rpc,
)


def register() -> None:
    """Register classes and schedule optional server startup."""
    for class_type in _CLASSES:
        bpy.utils.register_class(class_type)
    if not bpy.app.timers.is_registered(_auto_start):
        bpy.app.timers.register(_auto_start, first_interval=0.1)


def unregister() -> None:
    """Stop the service and unregister Blender classes."""
    if bpy.app.timers.is_registered(_auto_start):
        bpy.app.timers.unregister(_auto_start)
    runtime.stop()
    for class_type in reversed(_CLASSES):
        bpy.utils.unregister_class(class_type)
