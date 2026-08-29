"""PNG metadata and Blender UI coordinate conversion helpers."""

from __future__ import annotations

import struct
from dataclasses import dataclass

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_IMBUF_BUFFER_API_MIN_VERSION = (4, 5, 0)


class PngCropError(ValueError):
    """Raised when a screenshot cannot be cropped as a valid PNG."""


@dataclass(frozen=True, slots=True)
class CropGeometry:
    """A top-left-origin crop rectangle and its coordinate scale."""

    left: int
    top: int
    right: int
    bottom: int
    scale_x: float
    scale_y: float

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def png_dimensions(data: bytes) -> tuple[int, int]:
    """Return dimensions from a PNG byte string."""
    if len(data) < 24 or data[:8] != _PNG_SIGNATURE:
        raise PngCropError("Screenshot output is not a valid PNG")
    return struct.unpack(">II", data[16:24])


def select_imbuf_load_api(
    blender_version: tuple[int, int, int],
    *,
    has_load_from_buffer: bool,
) -> str:
    """Select the compatible ImBuf loader for a Blender version."""
    if (
        blender_version >= _IMBUF_BUFFER_API_MIN_VERSION
        and has_load_from_buffer
    ):
        return "load_from_buffer"
    return "load"


def calculate_region_crop(
    *,
    image_width: int,
    image_height: int,
    window_width: int,
    window_height: int,
    coordinate_max_x: int,
    coordinate_max_y: int,
    region_x: int,
    region_y: int,
    region_width: int,
    region_height: int,
) -> CropGeometry:
    """Map a bottom-left Blender region rectangle to PNG pixels."""
    values = {
        "image_width": image_width,
        "image_height": image_height,
        "window_width": window_width,
        "window_height": window_height,
        "region_width": region_width,
        "region_height": region_height,
    }
    invalid = [name for name, value in values.items() if value <= 0]
    if invalid:
        raise PngCropError(
            f"Screenshot geometry must be positive: {', '.join(invalid)}"
        )

    scale_x = _coordinate_scale(
        image_size=image_width,
        window_size=window_width,
        coordinate_max=coordinate_max_x,
    )
    scale_y = _coordinate_scale(
        image_size=image_height,
        window_size=window_height,
        coordinate_max=coordinate_max_y,
    )

    left = round(region_x * scale_x)
    right = round((region_x + region_width) * scale_x)
    lower = round(region_y * scale_y)
    upper = round((region_y + region_height) * scale_y)
    left = min(max(left, 0), image_width)
    right = min(max(right, 0), image_width)
    lower = min(max(lower, 0), image_height)
    upper = min(max(upper, 0), image_height)
    top = image_height - upper
    bottom = image_height - lower
    if right <= left or bottom <= top:
        raise PngCropError("VIEW_3D WINDOW region is outside the screenshot")
    return CropGeometry(
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        scale_x=scale_x,
        scale_y=scale_y,
    )


def _coordinate_scale(
    *,
    image_size: int,
    window_size: int,
    coordinate_max: int,
) -> float:
    """Detect logical or framebuffer-pixel Blender region coordinates."""
    tolerance = max(8, round(window_size * 0.01))
    if coordinate_max <= window_size + tolerance:
        return image_size / window_size
    return 1.0
