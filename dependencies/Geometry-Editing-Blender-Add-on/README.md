# Geometry Editing RPC Blender Add-on

Geometry Editing RPC is the Blender companion extension used by ViSculpt. It
runs a loopback-only JSON-RPC server inside Blender and exposes the controlled
operations required by the agent workflow.

Its capabilities include Blender state snapshots, standard orthographic views,
viewport screenshots and focusing, Sculpt Mode setup, brush configuration,
Face Set creation, and deterministic sculpt-stroke execution. Blender API work
is dispatched to the main thread to preserve Blender's execution constraints.

The release package is built and installed from the ViSculpt repository with
`uv run visculpt install-addon`. Users enable the extension manually in
Blender Preferences.

The extension declares `GPL-3.0-or-later` in its Blender manifest.
