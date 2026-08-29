"""Geometry Editing RPC Blender Add-on."""

bl_info = {
    "name": "Geometry Editing RPC",
    "author": "Geometry Editing Project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > Sidebar > RPC",
    "description": "Expose a local JSON-RPC API for ViSculpt",
    "category": "System",
}


def register() -> None:
    """Register the Blender Add-on."""
    from . import addon

    addon.register()


def unregister() -> None:
    """Unregister the Blender Add-on."""
    from . import addon

    addon.unregister()


if __name__ == "__main__":
    register()
