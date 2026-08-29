"""Build an installable Blender extension archive."""

from __future__ import annotations

from pathlib import Path
import tomllib
from zipfile import ZIP_DEFLATED, ZipFile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "geometry_editing_rpc"
MANIFEST_PATH = PACKAGE_ROOT / "blender_manifest.toml"
PACKAGE_FILES = (
    "__init__.py",
    "addon.py",
    "blender_api.py",
    "brush_quality.py",
    "dispatcher.py",
    "deferred.py",
    "face_set_coverage.py",
    "face_set_lasso.py",
    "face_set_lasso_batch.py",
    "image_utils.py",
    "pose_brush.py",
    "protocol.py",
    "runtime.py",
    "server.py",
    "sculpt_stroke.py",
    "sculpt_viewport_ui.py",
    "blender_manifest.toml",
)


def extension_version() -> str:
    """Read the archive version from Blender's authoritative manifest."""
    with MANIFEST_PATH.open("rb") as manifest_file:
        version = tomllib.load(manifest_file).get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("blender_manifest.toml must define version")
    return version.strip()


def output_path() -> Path:
    """Return the versioned extension archive path."""
    return (
        REPOSITORY_ROOT
        / "dist"
        / f"geometry_editing_rpc-{extension_version()}.zip"
    )


def main() -> None:
    archive_path = output_path()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for relative_path in PACKAGE_FILES:
            archive.write(
                PACKAGE_ROOT / relative_path,
                arcname=relative_path,
            )
    print(archive_path)


if __name__ == "__main__":
    main()
