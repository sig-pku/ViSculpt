"""LangGraph Tool that generates a validated monochrome SVG pattern."""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypeVar
from xml.etree.ElementTree import Element

from defusedxml import ElementTree as DefusedElementTree
from defusedxml.common import DefusedXmlException
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from visculpt.bridge import JsonValue

_SVG_NAMESPACE = "http://www.w3.org/2000/svg"
_SVG_CANVAS_SIZE = 512
_MAX_SVG_CHARACTERS = 131_072
_MAX_GRAPHIC_ELEMENTS = 16
_MAX_PATH_COMMANDS = 32
_MAX_PATH_NUMBERS = 96
_MAX_POLY_POINTS = 32
_GRAPHIC_TAGS = {
    "circle",
    "ellipse",
    "line",
    "path",
    "polygon",
    "polyline",
    "rect",
}
_ALLOWED_TAGS = {"svg", "g", *_GRAPHIC_TAGS}
_TRANSFORMABLE_TAGS = {"g", *_GRAPHIC_TAGS}
_TRANSFORM_ARITIES = {
    "matrix": {6},
    "rotate": {1, 3},
    "scale": {1, 2},
    "skewX": {1},
    "skewY": {1},
    "translate": {1, 2},
}
_PRESENTATION_ATTRIBUTES = {
    "fill",
    "fill-rule",
    "stroke",
    "stroke-dasharray",
    "stroke-dashoffset",
    "stroke-linecap",
    "stroke-linejoin",
    "stroke-miterlimit",
    "stroke-width",
}
_GEOMETRY_ATTRIBUTES = {
    "circle": {"cx", "cy", "r", "pathLength"},
    "ellipse": {"cx", "cy", "rx", "ry", "pathLength"},
    "g": set(),
    "line": {"x1", "x2", "y1", "y2", "pathLength"},
    "path": {"d", "pathLength"},
    "polygon": {"points", "pathLength"},
    "polyline": {"points", "pathLength"},
    "rect": {
        "height",
        "pathLength",
        "rx",
        "ry",
        "width",
        "x",
        "y",
    },
    "svg": {
        "height",
        "preserveAspectRatio",
        "version",
        "viewBox",
        "width",
    },
}
_FILLABLE_TAGS = {
    "circle",
    "ellipse",
    "path",
    "polygon",
    "polyline",
    "rect",
}

CompletionOutput = TypeVar("CompletionOutput", bound=BaseModel)


class SvgPatternCompletion(Protocol):
    """Structural completion result used without importing workflow modules."""

    value: BaseModel
    metadata: dict[str, JsonValue]


class SvgPatternLlm(Protocol):
    """Provider-neutral structured LLM surface required by this Tool."""

    def complete(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: Sequence[str | Path],
        response_model: type[CompletionOutput],
    ) -> SvgPatternCompletion:
        """Return one schema-validated completion."""


class GenerateSvgPatternInput(BaseModel):
    """Natural-language pattern description accepted by the Tool."""

    model_config = ConfigDict(extra="forbid")

    pattern_description: str = Field(
        min_length=1,
        max_length=2_048,
        description="Text description of the 2D pattern to generate.",
    )

    @field_validator("pattern_description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        """Normalize whitespace while retaining the user's wording."""
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("value must not be empty")
        return normalized


class SvgPatternLlmOutput(BaseModel):
    """Structured schema requested from the configured LLM."""

    model_config = ConfigDict(extra="forbid")

    svg: str = Field(
        min_length=1,
        max_length=_MAX_SVG_CHARACTERS,
        description="Complete standalone SVG markup without Markdown fences.",
    )


class SvgPatternValidationError(ValueError):
    """Raised when generated markup violates the monochrome SVG contract."""


def create_generate_svg_pattern_tool(
    *,
    llm: SvgPatternLlm,
    llm_role: str = "svg_pattern_generator",
) -> StructuredTool:
    """Create a sync/async LangGraph Tool for text-to-SVG generation."""
    role = llm_role.strip()
    if not role:
        raise ValueError("llm_role must not be empty")

    # Delay import to avoid a tools/workflow package initialization cycle.
    from visculpt.workflow.prompts import (
        SVG_PATTERN_SYSTEM_PROMPT,
        svg_pattern_user_prompt,
    )

    def generate(pattern_description: str) -> dict[str, JsonValue]:
        try:
            completion = llm.complete(
                role=role,
                system_prompt=SVG_PATTERN_SYSTEM_PROMPT,
                user_prompt=svg_pattern_user_prompt(
                    pattern_description=pattern_description,
                ),
                image_paths=[],
                response_model=SvgPatternLlmOutput,
            )
        except Exception as error:
            raise ToolException(
                _tool_error("llm_error", str(error))
            ) from error
        if not isinstance(completion.value, SvgPatternLlmOutput):
            raise ToolException(
                _tool_error(
                    "llm_error",
                    "The LLM returned an unexpected response model",
                )
            )
        try:
            svg = validate_svg_pattern(completion.value.svg)
        except SvgPatternValidationError as error:
            raise ToolException(
                _tool_error("invalid_svg", str(error))
            ) from error
        return {
            "input": {"pattern_description": pattern_description},
            "result": {
                "svg": svg,
                "media_type": "image/svg+xml",
                "width": _SVG_CANVAS_SIZE,
                "height": _SVG_CANVAS_SIZE,
                "view_box": [0, 0, _SVG_CANVAS_SIZE, _SVG_CANVAS_SIZE],
                "color_contract": {
                    "background": "#ffffff",
                    "pattern": "#000000",
                },
                "rendering_contract": {
                    "style": "SIMPLE_STRUCTURAL_LINE_ART",
                    "visible_black_stroke_required": True,
                    "solid_fill_policy": "SMALL_ESSENTIAL_ACCENTS_ONLY",
                    "detail_policy": "MINIMIZE_SHORT_DECORATIVE_STROKES",
                    "maximum_graphic_elements": _MAX_GRAPHIC_ELEMENTS,
                    "maximum_path_commands": _MAX_PATH_COMMANDS,
                    "maximum_poly_points": _MAX_POLY_POINTS,
                },
                "llm": completion.metadata,
            },
        }

    async def agenerate(
        pattern_description: str,
    ) -> dict[str, JsonValue]:
        return await asyncio.to_thread(generate, pattern_description)

    return StructuredTool.from_function(
        func=generate,
        coroutine=agenerate,
        name="generate_svg_pattern",
        description=(
            "Use the configured LLM to generate a safety-validated 512x512 SVG "
            "line drawing from text. Express the subject with a few simple, long "
            "black structural lines instead of dense short decorative details."
        ),
        args_schema=GenerateSvgPatternInput,
        infer_schema=False,
        handle_tool_error=lambda error: str(error),
        handle_validation_error=lambda _: _tool_error(
            "invalid_tool_input",
            "pattern_description must be a non-empty text value",
        ),
    )


def validate_svg_pattern(svg: str) -> str:
    """Validate and return standalone black-on-white SVG markup."""
    markup = _strip_svg_fence(svg)
    if not markup:
        raise SvgPatternValidationError("Generated SVG is empty")
    if len(markup) > _MAX_SVG_CHARACTERS:
        raise SvgPatternValidationError(
            "Generated SVG exceeds the maximum allowed length"
        )
    try:
        root = DefusedElementTree.fromstring(markup)
    except (DefusedXmlException, SyntaxError, ValueError) as error:
        raise SvgPatternValidationError(
            f"Generated SVG is not safe, well-formed XML: {error}"
        ) from error
    if _qualified_tag(root) != ("svg", _SVG_NAMESPACE):
        raise SvgPatternValidationError(
            "Root element must be svg in the standard SVG namespace"
        )
    _validate_root(root)
    children = list(root)
    if len(children) < 2:
        raise SvgPatternValidationError(
            "SVG must contain a white background and a black pattern"
        )
    _validate_background(children[0])

    filled_graphic_count = 0
    stroked_graphic_count = 0
    graphic_element_count = 0
    inherited = {
        "fill": "black",
        "stroke": "none",
        "stroke-width": "1",
    }
    for child in children[1:]:
        child_fills, child_strokes, child_graphics = (
            _validate_pattern_element(
                child,
                inherited_paint=inherited,
            )
        )
        filled_graphic_count += child_fills
        stroked_graphic_count += child_strokes
        graphic_element_count += child_graphics
    if filled_graphic_count + stroked_graphic_count == 0:
        raise SvgPatternValidationError(
            "SVG does not contain a visible black pattern"
        )
    if stroked_graphic_count == 0:
        raise SvgPatternValidationError(
            "Line-art SVG must contain at least one visible black stroke"
        )
    if filled_graphic_count > stroked_graphic_count:
        raise SvgPatternValidationError(
            "Line-art SVG must not contain more filled graphic elements "
            "than stroked graphic elements"
        )
    if graphic_element_count > _MAX_GRAPHIC_ELEMENTS:
        raise SvgPatternValidationError(
            "Structural line-art SVG must contain at most "
            f"{_MAX_GRAPHIC_ELEMENTS} graphic elements"
        )
    return markup


def _validate_root(root: Element) -> None:
    _validate_attributes(root, tag="svg")
    if _number(root.get("width")) != _SVG_CANVAS_SIZE:
        raise SvgPatternValidationError("SVG width must be 512")
    if _number(root.get("height")) != _SVG_CANVAS_SIZE:
        raise SvgPatternValidationError("SVG height must be 512")
    view_box = root.get("viewBox", "")
    try:
        values = [
            float(value)
            for value in re.split(r"[\s,]+", view_box.strip())
            if value
        ]
    except ValueError as error:
        raise SvgPatternValidationError(
            "SVG viewBox must contain numeric values"
        ) from error
    if values != [0.0, 0.0, 512.0, 512.0]:
        raise SvgPatternValidationError(
            "SVG viewBox must be exactly 0 0 512 512"
        )
    if any(name in root.attrib for name in _PRESENTATION_ATTRIBUTES):
        raise SvgPatternValidationError(
            "The SVG root must not override pattern paint"
        )
    _validate_opacity(root)
    _require_whitespace_only(root.text, label="SVG root text")


def _validate_background(element: Element) -> None:
    tag, namespace = _qualified_tag(element)
    if tag != "rect" or namespace != _SVG_NAMESPACE:
        raise SvgPatternValidationError(
            "The first SVG child must be a full-canvas white rect"
        )
    _validate_attributes(element, tag=tag)
    if _number(element.get("x", "0")) != 0.0:
        raise SvgPatternValidationError("Background rect x must be 0")
    if _number(element.get("y", "0")) != 0.0:
        raise SvgPatternValidationError("Background rect y must be 0")
    if not _length_is_canvas(element.get("width")):
        raise SvgPatternValidationError("Background rect width must be 512")
    if not _length_is_canvas(element.get("height")):
        raise SvgPatternValidationError("Background rect height must be 512")
    if element.get("transform") is not None or any(
        name in element.attrib for name in {"rx", "ry"}
    ):
        raise SvgPatternValidationError(
            "Background rect must be square and untransformed"
        )
    if _paint(element.get("fill")) != "white":
        raise SvgPatternValidationError(
            "Background rect fill must be pure white"
        )
    if _paint(element.get("stroke", "none")) != "none":
        raise SvgPatternValidationError(
            "Background rect must not have a stroke"
        )
    _validate_opacity(element)
    _require_whitespace_only(element.text, label="Background rect text")
    _require_whitespace_only(element.tail, label="Background rect tail")
    if list(element):
        raise SvgPatternValidationError(
            "Background rect must not contain child elements"
        )


def _validate_pattern_element(
    element: Element,
    *,
    inherited_paint: Mapping[str, str],
) -> tuple[int, int, int]:
    tag, namespace = _qualified_tag(element)
    if namespace != _SVG_NAMESPACE or tag not in _ALLOWED_TAGS - {"svg"}:
        raise SvgPatternValidationError(
            f"Unsupported SVG element: {tag or element.tag}"
        )
    _validate_attributes(element, tag=tag)
    _validate_opacity(element)
    _validate_geometry_complexity(element, tag=tag)
    _require_whitespace_only(element.text, label=f"{tag} text")
    _require_whitespace_only(element.tail, label=f"{tag} tail")

    fill = _paint(element.get("fill", inherited_paint["fill"]))
    stroke = _paint(element.get("stroke", inherited_paint["stroke"]))
    if fill not in {"black", "none"}:
        raise SvgPatternValidationError(
            f"Pattern element {tag} must use only black or no fill"
        )
    if stroke not in {"black", "none"}:
        raise SvgPatternValidationError(
            f"Pattern element {tag} must use only black or no stroke"
        )
    raw_stroke_width = element.get(
        "stroke-width",
        inherited_paint["stroke-width"],
    )
    stroke_width = _number(raw_stroke_width)
    if stroke_width < 0.0:
        raise SvgPatternValidationError(
            "stroke-width must not be negative"
        )

    filled = 0
    stroked = 0
    graphics = 0
    if tag in _GRAPHIC_TAGS:
        graphics = 1
        has_fill = tag in _FILLABLE_TAGS and fill == "black"
        has_stroke = stroke == "black" and stroke_width > 0.0
        filled = int(has_fill)
        stroked = int(has_stroke)
    for child in element:
        child_fills, child_strokes, child_graphics = (
            _validate_pattern_element(
                child,
                inherited_paint={
                    "fill": fill,
                    "stroke": stroke,
                    "stroke-width": raw_stroke_width,
                },
            )
        )
        filled += child_fills
        stroked += child_strokes
        graphics += child_graphics
    return filled, stroked, graphics


def _validate_geometry_complexity(element: Element, *, tag: str) -> None:
    if tag == "path":
        path_data = element.get("d", "")
        command_count = len(
            re.findall(r"[AaCcHhLlMmQqSsTtVvZz]", path_data)
        )
        number_count = len(
            re.findall(
                r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
                path_data,
            )
        )
        if command_count == 0:
            raise SvgPatternValidationError(
                "Path elements must contain drawing commands"
            )
        if command_count > _MAX_PATH_COMMANDS:
            raise SvgPatternValidationError(
                "Structural line-art paths must contain at most "
                f"{_MAX_PATH_COMMANDS} commands"
            )
        if number_count > _MAX_PATH_NUMBERS:
            raise SvgPatternValidationError(
                "Structural line-art paths contain too many coordinates"
            )
    if tag in {"polygon", "polyline"}:
        values = re.findall(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            element.get("points", ""),
        )
        if len(values) < 4 or len(values) % 2 != 0:
            raise SvgPatternValidationError(
                f"{tag} points must contain coordinate pairs"
            )
        point_count = len(values) // 2
        if point_count > _MAX_POLY_POINTS:
            raise SvgPatternValidationError(
                f"Structural line-art {tag} must contain at most "
                f"{_MAX_POLY_POINTS} points"
            )


def _validate_attributes(element: Element, *, tag: str) -> None:
    allowed = _GEOMETRY_ATTRIBUTES[tag] | _PRESENTATION_ATTRIBUTES | {
        "opacity",
        "fill-opacity",
        "stroke-opacity",
    }
    if tag in _TRANSFORMABLE_TAGS:
        allowed.add("transform")
    for name, value in element.attrib.items():
        if name.startswith("{") or name not in allowed:
            raise SvgPatternValidationError(
                f"Unsupported attribute {name!r} on {tag}"
            )
        lowered = value.casefold()
        if any(
            token in lowered
            for token in ("url(", "javascript:", "data:")
        ):
            raise SvgPatternValidationError(
                f"Unsafe attribute value on {tag}.{name}"
            )
        if name == "transform":
            _validate_transform(value, tag=tag)


def _validate_transform(value: str, *, tag: str) -> None:
    expression = value.strip()
    if not expression:
        raise SvgPatternValidationError(
            f"Transform on {tag} must not be empty"
        )
    function_pattern = re.compile(
        r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^()]*)\)"
    )
    cursor = 0
    function_count = 0
    for match in function_pattern.finditer(expression):
        separator = expression[cursor : match.start()]
        if separator.strip(" \t\r\n,"):
            raise SvgPatternValidationError(
                f"Unsupported transform syntax on {tag}"
            )
        function_name = match.group(1)
        raw_arguments = [
            item
            for item in re.split(r"[\s,]+", match.group(2).strip())
            if item
        ]
        if len(raw_arguments) not in _TRANSFORM_ARITIES[function_name]:
            raise SvgPatternValidationError(
                f"Transform {function_name} on {tag} has the wrong "
                "number of arguments"
            )
        for argument in raw_arguments:
            if re.fullmatch(
                r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
                argument,
            ) is None:
                raise SvgPatternValidationError(
                    f"Transform {function_name} on {tag} must use only "
                    "finite numeric arguments"
                )
            if not math.isfinite(float(argument)):
                raise SvgPatternValidationError(
                    f"Transform {function_name} on {tag} must use only "
                    "finite numeric arguments"
                )
        cursor = match.end()
        function_count += 1
    if (
        function_count == 0
        or expression[cursor:].strip(" \t\r\n,")
    ):
        raise SvgPatternValidationError(
            f"Unsupported transform syntax on {tag}"
        )


def _validate_opacity(element: Element) -> None:
    for name in ("opacity", "fill-opacity", "stroke-opacity"):
        if name not in element.attrib:
            continue
        if not math.isclose(
            _number(element.get(name)),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise SvgPatternValidationError(
                f"{name} must be 1 to preserve pure black and white output"
            )


def _qualified_tag(element: Element) -> tuple[str, str | None]:
    tag = element.tag
    if not isinstance(tag, str):
        return "", None
    if tag.startswith("{") and "}" in tag:
        namespace, local = tag[1:].split("}", maxsplit=1)
        return local, namespace
    return tag, None


def _paint(value: str | None) -> str:
    if value is None:
        return "none"
    compact = re.sub(r"\s+", "", value).casefold()
    if compact in {"black", "#000", "#000000", "rgb(0,0,0)"}:
        return "black"
    if compact in {
        "white",
        "#fff",
        "#ffffff",
        "rgb(255,255,255)",
    }:
        return "white"
    if compact == "none":
        return "none"
    return compact


def _number(value: str | None) -> float:
    if value is None:
        raise SvgPatternValidationError(
            "Required numeric SVG value is missing"
        )
    normalized = value.strip()
    if normalized.casefold().endswith("px"):
        normalized = normalized[:-2].strip()
    try:
        number = float(normalized)
    except ValueError as error:
        raise SvgPatternValidationError(
            f"Invalid numeric SVG value: {value!r}"
        ) from error
    if not math.isfinite(number):
        raise SvgPatternValidationError(
            f"SVG numeric value must be finite: {value!r}"
        )
    return number


def _length_is_canvas(value: str | None) -> bool:
    if value is None:
        return False
    if value.strip() == "100%":
        return True
    return _number(value) == float(_SVG_CANVAS_SIZE)


def _require_whitespace_only(value: str | None, *, label: str) -> None:
    if value is not None and value.strip():
        raise SvgPatternValidationError(
            f"{label} must not contain free text"
        )


def _strip_svg_fence(value: str) -> str:
    stripped = value.strip()
    match = re.fullmatch(
        r"```(?:svg|xml)?\s*(.*?)\s*```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match is not None else stripped


def _tool_error(error_type: str, message: str) -> str:
    return json.dumps(
        {
            "generate_svg_pattern_error": {
                "type": error_type,
                "message": message,
            }
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
