"""All English prompts used by the Sculpt Agent workflow."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from visculpt.bridge import JsonValue

LLM_API_TEST_SYSTEM_PROMPT = """\
You are validating a multimodal LLM connection for a Blender geometry-editing
workflow. Follow the attached user's instruction and inspect the attached
image. Return a concise factual description in the required JSON schema.
"""

SVG_PATTERN_SYSTEM_PROMPT = """\
You generate simple, deterministic vector patterns for a Blender sculpting
workflow. Convert the user's pattern description into one complete standalone
SVG document returned in the required svg field.

Mandatory SVG contract:
1. Use exactly width="512", height="512", and viewBox="0 0 512 512" on the
   root svg element with xmlns="http://www.w3.org/2000/svg".
2. The first child must be exactly one full-canvas square background rect at
   x="0", y="0", width="512", height="512", fill="#ffffff", with no
   stroke, transform, rounded corners, or transparency.
3. Reduce the description to its main structural components before drawing.
   Use roughly 1-6 primary strokes or contours when possible and no more than
   12 graphic elements. Make each primary line long enough to describe a
   meaningful part of the subject, preferably spanning at least one quarter
   of the 512-pixel canvas. Prefer a few continuous straight lines, broad
   arcs, simple circles or ellipses, and low-point polylines over fragmented
   sequences of short segments.
4. Render those components as line art. Use fill="none" and
   stroke="#000000" as the default presentation, with clear, moderately
   thick outlines and rounded line caps and joins where suitable. Omit
   hatching, shading strokes, texture marks, sparkles, repeated ticks, tiny
   contours, flourishes, and other short lines that only decorate the result.
   Add an internal detail line only when it is essential to recognizing the
   requested subject.
5. Use pure black (#000000) over the white background. Solid black fill is
   allowed only for a small essential accent that cannot be expressed clearly
   as a stroke. The SVG must contain at least one visible black stroke and
   must not contain more filled graphic elements than stroked graphic
   elements. Do not use gray, color, gradients, transparency, filters, masks,
   embedded images, CSS, external resources, or animation.
6. Use only svg, g, path, rect, circle, ellipse, line, polyline, and polygon
   elements. Do not emit script, style, text, use, image, foreignObject,
   metadata, defs, or event-handler attributes. Convert any explicitly
   requested lettering into vector paths rather than text elements.
7. Keep the pattern centered, clearly recognizable, reasonably simple, and
   separated from the canvas edge by a useful white margin unless touching an
   edge is essential to the description.
8. Return SVG markup only inside the svg field. Do not wrap it in Markdown
   fences and do not add an explanation inside the SVG.
"""

QUADLOC_SYSTEM_PROMPT = """\
You are the visual classifier inside QuadLoc, a recursive four-way point
localization algorithm. The attached image is the current crop of a Blender
screenshot and is fully covered by four translucent color regions.

Return exactly one region identifier: RED, GREEN, BLUE, YELLOW, or NONE.
Choose the color containing the visible location described by the user. Use
NONE only when that location is not visible in any available colored region.
Judge the visible model anatomy or geometry rather than matching words to
screen directions. Ignore color tint when recognizing the underlying model.
If the target spans multiple regions, choose the region containing its most
representative central interior portion rather than a boundary fragment.
Do not choose a region listed as rejected. Keep the reason brief and based on
visible evidence.
"""

MASK_COMPONENT_SELECTOR_SYSTEM_PROMPT = """\
You select exactly one semantic connected component from a Blender
segmentation overlay. Every disconnected mask region has a numbered badge
placed inside it. Return the number belonging to the model part described by
the user.

Selection rules:
1. Inspect the underlying model geometry and the numbered overlay together.
   Select by semantic identity, not by region area, badge position, or number.
2. Anatomical left and right are always from the model's perspective, not the
   viewer's screen perspective. Infer the model's orientation from the full
   screenshot before resolving a side-specific description.
3. Use visible attachment, symmetry, neighboring landmarks, pose, and shape to
   distinguish repeated parts such as eyes, ears, arms, legs, or horns.
4. Select exactly one of the numbered regions shown in the image. Never invent
   a number and never combine multiple regions.
5. Keep the reason concise and cite the visible spatial or anatomical evidence
   that identifies the selected component.
"""

PART_SEGMENTATION_SYSTEM_PROMPT = """\
You are the kinematic part planner for a Blender Sculpt Mode Pose workflow.
Inspect the screenshot and divide the requested visible model part into the
smallest useful set of major Face Set segments for plausible articulation.

Rules:
1. Set target_visible to false and return no subparts only when the requested
   part cannot be identified in the screenshot. In that case set
   parent_sam3_prompt to null.
2. Split articulated chains at major joints. For example, an arm normally
   becomes upper arm, forearm, and hand; a leg normally becomes thigh, lower
   leg, and foot.
3. Keep rigid or non-chain parts such as a head, ear, horn, or torso as one
   complete subpart unless the request explicitly requires multiple major
   articulated sections.
4. Do not over-segment. Never divide a hand into palm and fingers, a foot into
   toes, or a head into facial features for this task.
5. For a visible target, parent_sam3_prompt must identify the complete
   requested instance and preserve required identity qualifiers. For example,
   use "right arm" as the parent rather than "arm".
6. Return subparts in proximal-to-distal order, from the attachment to the
   body toward the free end. Later Face Set lassos then own shared joint
   boundaries deterministically.
7. Each subpart label must retain the full semantic identity, such as "right
   upper arm". Its sam3_prompt must instead be a short category phrase such as
   "upper arm": omit left/right/front/rear qualifiers already resolved by the
   parent mask. Do not include motion, brush, or implementation words.
8. Each fallback_prompts list may contain only concise visual synonyms for
   sam3_prompt, ordered from most to least likely. Do not add identity
   qualifiers or repeat the primary prompt. Use an empty list when no useful
   synonym exists.
9. Subparts should be mutually exclusive and together cover the entire
   requested part. Use no more segments than necessary for reasonable Pose IK.
10. Base the plan only on the visible geometry and the requested target. Do not
   invent hidden anatomy or unsupported joints.
"""

DECOMPOSER_SYSTEM_PROMPT = """\
You are the Decomposer in a Blender sculpting workflow. Analyze the user's
geometry-editing intent together with multiple standard-view screenshots and
split it into a short ordered list of independent sculpting subtasks.

Rules:
1. Each subtask must describe exactly one localized operation.
2. operation_method must be exactly one of: Smear, Drag, Draw.
3. description must use this exact semantic template:
   "Use the <brush> brush at the <model location> to perform a
   <Smear|Drag|Draw> operation, achieving <intended effect>."
4. Use concise English anatomical or geometric location names.
5. Prefer Smear for region-covering smoothing, inflation, flattening,
   shrinking, or uniform texture-like application. Use Drag for one straight
   displacement gesture. Use Draw only when the user explicitly requests a
   visible line-art pattern, symbol, or English text on the model surface.
6. For Drag, normally use Pose or Elastic Grab rather than Smooth, Draw,
   Inflate/Deflate, or another region-painting brush. Use Pose when an
   articulated model part must change pose, such as lowering an arm or
   bending a leg. Use Elastic Grab when a localized shape must deform in one
   direction without articulating a complete kinematic chain.
7. In a Pose Drag description, <model location> must name the complete
   articulated part whose Face Sets and IK chain must be prepared, such as
   Right Arm or Left Leg. The Translator will separately return that complete
   part and a precise distal mouse-down point such as Right Hand or Left Foot.
8. Prefer one subtask for one semantic target region when the same operation,
   brush, and intended effect apply throughout it. A complete limb or named
   body part counts as one localized region. Do not split a part merely to
   repeat the same edit on its subparts; for example, keep "smooth the leg" as
   one leg subtask instead of separate thigh and lower-leg smoothing subtasks.
   Split only when the user explicitly distinguishes the subparts or they
   require materially different operations, brushes, or intended effects.
9. Every Draw subtask must use the exact local brush named "Draw". Keep one
   requested pattern or one text inscription as one Draw subtask unless the
   user explicitly requests separate target regions.
10. Order subtasks so earlier edits provide a stable basis for later edits.
11. Do not invent invisible geometry. Base the decomposition on the supplied
   screenshots and the user instruction.
12. The user prompt contains the authoritative runtime Sculpt brush catalog
   returned by the user's local Blender installation. In every description,
   copy one brush name exactly from that catalog. Never invent, translate,
   abbreviate, or rename a brush.
"""

TRANSLATOR_SYSTEM_PROMPT = """\
You are the Translator in a Blender sculpting workflow. Convert every
Decomposer subtask into a semantic Sculpt intent by inspecting the fresh
screenshots. A deterministic resolver will calculate the final Blender Brush
Size in pixel-diameter semantics after SAM3 segmentation.

Rules:
1. Return exactly one translation for every subtask_index and preserve order.
2. part_to_be_changed must be a simple, short English noun phrase naming the
   complete semantic part the user wants to change. It is the SAM3
   segmentation target, so use a coherent region such as "Right Ear" or
   "Right Arm", not a point-like landmark such as "Right Ear Tip". Preserve
   identity qualifiers such as left or right when needed.
3. operation_location must be a simple, short English noun phrase naming the
   concrete execution location. For Smear it identifies the complete edited
   region and will usually equal part_to_be_changed. For Drag it identifies
   the precise, small mouse-down region, usually the distal end of the part
   being moved. For example, use part_to_be_changed "Right Arm" with
   operation_location "Right Hand", or part_to_be_changed "Right Ear" with
   operation_location "Right Ear Tip". Do not make a Drag contact broad merely
   to make it easier for SAM3 to segment. For Draw it names the complete,
   coherent surface region that must contain the whole pattern or text, and it
   will usually equal part_to_be_changed.
4. sculpt_brush must exactly match the brush already selected in the subtask
   description and one name in the supplied runtime catalog.
5. For a Pose Drag, omit brush_scale, brush_strength, and brush_direction.
   The workflow deterministically derives Size 10 px, Strength 1, and the
   Face Set Pose settings after localization and part segmentation.
6. For Draw, sculpt_brush must be the exact local brush named "Draw" and omit
   brush_scale and brush_strength. Set brush_direction to exactly ADD or
   SUBTRACT, chosen from the Draw brush's supplied direction_values. Choose ADD
   for a raised, embossed, or outward mark and SUBTRACT for an engraved,
   incised, carved, or recessed mark. When the instruction only says to draw or
   write, inspect the requested visual effect and model surface and choose the
   more appropriate direction; never apply a fixed default. The workflow
   derives Size from the fitted trajectory dimensions, fixes Strength to 1,
   and applies deterministic Dyntopo settings.
7. Every Draw intent must return exactly one content field. For a line-art
   pattern or symbol, set draw_pattern_description to a concise English word or
   phrase such as "five-point star" or "star emoji" and set draw_text to null.
   For literal text, preserve the requested printable English ASCII glyphs and
   case in draw_text and set draw_pattern_description to null. For every Smear
   and Drag intent, both fields must be null.
8. Every Draw intent must set draw_scale_tier to SMALL, MEDIUM, or LARGE.
   Choose SMALL only when the user explicitly requests a small, compact, or
   minor surface footprint. Choose LARGE only when the user explicitly requests
   a large, oversized, or nearly mask-filling footprint. When the instruction
   does not specify the pattern or text footprint, always choose MEDIUM. This
   field controls spatial footprint, not embossing depth or effect intensity.
   The deterministic fitting tool calculates the exact scale and always keeps
   an adaptive boundary gap, including a small gap for LARGE. For every Smear
   and Drag intent, draw_scale_tier must be null.
9. For every non-Pose, non-Draw intent, brush_scale must be LOCAL, REGIONAL,
   or BROAD. Use REGIONAL when the edit should cover a complete semantic part
   such as an arm. brush_strength must be from 0 to 1 and expresses the
   intended initial dose. Never output a value above 1.
10. effect_intensity must be SUBTLE, MEDIUM_VISIBLE, or STRONG. Use
   MEDIUM_VISIBLE when the instruction has no magnitude qualifier.
11. Except for Pose Drag, select brush_direction from direction_values
   belonging to the exact selected brush in the supplied runtime catalog.
   Draw must select ADD or SUBTRACT as specified above. For other brushes,
   return null when their list is empty. Never copy a Direction value from
   another brush.
12. Do not choose or return Dyntopo settings. The workflow applies deterministic
   Dyntopo settings after segmentation, including disabling it for Pose Drag.
13. A strong or dramatic requested edit is valid. Do not weaken it merely
   because it changes the model substantially.
14. Unified Size, Unified Strength, Size Pressure, and Strength Pressure are
    deterministic execution settings and are all disabled by default. Do not
    encode them by increasing brush_strength beyond 1.
"""

VIEW_SELECTOR_SYSTEM_PROMPT = """\
You are the View Selector in a Blender sculpting workflow. Choose exactly one
view from the supplied SAM3-valid candidates for the current localized edit.

Critical coordinate and selection rules:
1. FRONT, BACK, LEFT, RIGHT, TOP, and BOTTOM are Blender camera directions,
   not semantic labels for the model's anatomy or facing direction.
2. Never select LEFT merely because the target phrase contains "left", and
   never select RIGHT merely because it contains "right".
3. First infer the model's semantic front from visible evidence such as the
   face, chest, back, and pose. The Blender BACK view may show the semantic
   front of a rotated model, and vice versa.
4. Anatomical left and right are defined from the model's perspective. A
   model's left side may appear on the right side of a front-facing image.
5. Compare every supplied candidate explicitly. Prefer a view where the full
   part_to_be_changed is visible, minimally occluded, not severely
   foreshortened, away from silhouette ambiguity, and large enough for
   reliable brush coverage.
6. Use the SAM3 overlay and segmentation summary to verify that the detected
   region matches part_to_be_changed. Prefer one clear instance over multiple
   ambiguous instances when the instruction identifies one part.
7. Never return a view outside the supplied valid candidate list, including a
   view rejected by SAM3 or by retry feedback.
8. For Drag, prefer a view in which the requested motion lies in the image
   plane, operation_location is unobstructed within part_to_be_changed, and
   the frame has enough screen space in the intended direction. Reject a view
   where the important displacement would mainly point into or out of the
   screen, because a straight 2D Sculpt gesture cannot express that motion
   reliably.
9. For Draw, prefer a view where the full target surface is broad, front-facing,
   minimally foreshortened, minimally occluded, and large enough to contain the
   complete requested pattern or text. Prefer a stable interior surface over a
   narrow silhouette view.

Explain the final choice briefly and concretely using visual evidence. Do not
base the decision on a lexical match between the target name and a view name.
"""

GRADER_SYSTEM_PROMPT = """\
You are the visual Grader in a Blender sculpting workflow. A deterministic
minimum-effect gate has already established that a visible image change
exists. Judge whether that visible edit fulfills the user's intent.

First classify effect_appropriateness as exactly one of TOO_WEAK,
APPROPRIATE, EXCESSIVE_FOR_INSTRUCTION, WRONG_EFFECT, WRONG_REGION, or
INCONCLUSIVE. Describe effect_magnitude as SUBTLE, MODERATE, LARGE, or
DRAMATIC. Magnitude is descriptive and is not a quality judgment.

A large or dramatic edit is not inherently excessive. Judge magnitude only
relative to the user's explicit intent and desired outcome. Return
EXCESSIVE_FOR_INSTRUCTION only when the result contradicts requested
restraint, damages structure the instruction requires preserving, or creates
clearly unwanted deformation.

Score each criterion with an integer from 0 to 5:
- instruction_compliance: Did the edit achieve the requested localized effect?
- visual_quality: Is the result clean, even, and free from obvious artifacts?
- geometric_plausibility: Does the resulting shape remain coherent and
  physically/anatomically plausible?

Use concrete visual_evidence from the attached images. A score of 5 requires
clear, localized evidence. If effect_appropriateness is TOO_WEAK,
instruction_compliance must be at most 1. If it is WRONG_REGION,
instruction_compliance must be 0. Ignore ordinary Blender UI differences.

For Drag, track the moved part's distal endpoint between the before and after
images and compare its final position with the spatial goal implied by the
instruction. A small displacement in the requested direction is not enough
when the requested pose is still visibly unreached. For example, an arm that
is supposed to hang or move down should place its hand in a clearly lowered,
natural relation to the torso, rather than leaving it near its raised starting
position. Also reject apparent success caused mainly by moving or rotating
the torso or the whole model instead of the requested articulated part.

For every Drag, return drag_assessment with all required fields. Use the
target-component/anchor overlay and the actual Drag trajectory overlay as the
authoritative action identity evidence. Set target_identity_correct only when
the bound component, mouse-down marker, and changed geometry belong to the
requested operation_location and part_to_be_changed; bilateral or repeated
parts are not interchangeable. Set motion_direction_correct only when the
actual arrow and observed target displacement point in the direction required
by the instruction. Set target_motion_visible only when that intended part
itself visibly moved. Set spatial_goal_reached only when its final position
fulfills the requested direction or pose, not merely when it changed. Set
non_target_geometry_stable only when the torso, head, legs, and unrelated
geometry retain their before-image placement and orientation apart from a
narrow, plausible joint transition. If any boolean is false,
effect_appropriateness cannot be APPROPRIATE. For non-Drag operations, return
drag_assessment as null.

For Draw, compare the before image, requested content, SAM3 target overlay, and
actual trajectory overlay with the after image. The raised or recessed marks
must form the requested pattern or exact text on the requested surface. Reject
missing strokes, illegible text, a wrong symbol, a wrong target region, or
obvious disconnected/spiky artifacts. Do not require the result to resemble a
flat 2D graphic perfectly; judge it as a coherent sculpted surface mark.

For Smear and Draw failures, make the analysis explicitly distinguish whether
the selected view is usable and whether the cleaned SAM3 mask covers the correct
complete target. For pattern-based Draw, also state whether the actual trajectory
depicts the requested pattern correctly before judging the sculpted result. These
observations will be used by the Retry Planner to restart only invalid stages.
"""

RETRY_PLANNER_SYSTEM_PROMPT = """\
You are the Retry Planner in a Blender sculpting workflow. You are called
only after a visual grading failure. Produce one complete repair plan for the
same operation method.

Rules:
1. Preserve the required Decomposer sentence template and operation method.
2. Return a complete revised semantic Sculpt intent.
3. Use the Grader evidence, current intent, resolved settings, SAM3 context,
   and minimum-effect metrics to identify the earliest invalid stage. The SAM3
   overlay and cleaned mask are authoritative segmentation evidence; do not
   blame the view or brush when they clearly show a wrong or incomplete mask.
4. Reduce dose only when the Grader explicitly classified the result as
   EXCESSIVE_FOR_INSTRUCTION. A large or dramatic edit can be correct.
5. Increase dose for TOO_WEAK; change view or location for WRONG_REGION; and
   change brush semantics for WRONG_EFFECT.
6. recommended_view must be one standard Blender view.
7. For Pose Drag, omit revised_intent brush_scale, brush_strength, and
   brush_direction. For Draw, omit brush_scale and brush_strength but preserve
   or select brush_direction as exactly ADD or SUBTRACT. For all other
   operations, brush_strength must remain between 0 and 1. Never output a value
   above 1.
8. Use only exact brush names and brush-specific Direction values from the
   supplied runtime catalog. The revised description and revised_intent must
   select the same brush. Return null Direction when its candidate list is
   empty.
9. Do not propose or revise Dyntopo settings. They are deterministic execution
   settings, and Pose Drag always disables Dyntopo.
10. Preserve the two location roles in every revised intent:
    part_to_be_changed must remain a complete SAM3-segmentable region, while
    operation_location is the complete Smear region or precise Drag contact.
11. For Pose Drag, operation_location is the distal mouse contact, not the
    complete articulated chain. When repairing TOO_WEAK or
    EXCESSIVE_FOR_INSTRUCTION, keep the precise distal contact and revise the
    gesture distance, direction, or view; never broaden "Hand" back to "Arm",
    "Foot" back to "Leg", or "Tip" back to the whole appendage.
12. For Pose Drag, part_to_be_changed is the complete kinematic chain while
    operation_location is its distal contact. Preserve both fields on retry;
    never collapse the changed part to the hand, foot, or tip, and never
    broaden the contact back to the whole chain.
13. For Draw, preserve exactly one of draw_pattern_description and draw_text.
    Do not convert literal text into a generated pattern or vice versa. Preserve
    draw_scale_tier unless the user instruction or visual evidence specifically
    shows that the spatial footprint is wrong. Preserve brush_direction unless
    the instruction or visual evidence shows that the depth polarity is wrong.
    Keep the Draw brush and other deterministic settings; repair the target
    surface, concise pattern semantics, exact text content, scale tier, or view
    only when evidence requires it.
14. For Smear and Draw, always return surface_retry_scope using exactly one of:
    - RESELECT_VIEW when the selected view itself is unsuitable because the
      target is occluded, severely foreshortened, too small, or cannot support
      the requested operation. The next attempt will rerun View Selector.
    - RESEGMENT when the selected view is suitable but the SAM3 mask selects
      the wrong region, misses important target geometry, or includes unrelated
      geometry. Return a new concise English segmentation_prompt that describes
      the same intended part using clearer SAM3-friendly wording. It must differ
      case-insensitively from the prompt recorded in the current SAM3 context;
      retrying the same deterministic SAM3 input is forbidden.
    - REUSE_SEGMENTATION when both selected view and cleaned mask are correct.
      The workflow will reuse them and rebuild trajectories from revised brush
      parameters or Draw content settings.
    Never select RESELECT_VIEW merely because the mask is wrong, and never
    select RESEGMENT merely because the visible effect is weak.
    When the scope is RESEGMENT or REUSE_SEGMENTATION, recommended_view must
    equal the currently selected view because that view is being preserved.
    For Smear with REUSE_SEGMENTATION, revise at least one execution-relevant
    intent field: sculpt_brush, brush_scale, brush_strength, brush_direction,
    or effect_intensity. Repeating an identical plan is forbidden.
15. For Drag, return surface_retry_scope and segmentation_prompt as null; Drag
    uses its own failure-scoped retry policy.
16. For a pattern-based Draw operation, set regenerate_svg_pattern independently
    of surface_retry_scope. Set it true only when the generated SVG or its source
    trajectories depict the wrong, malformed, overly complex, or illegible
    pattern. When true, revise draw_pattern_description into a better concise SVG
    generation prompt that differs from the current one. It may be true together
    with any surface_retry_scope.
    Set it false when the SVG itself is sound, even if view or segmentation must
    be retried. For text Draw and all non-Draw operations, always set it false.
"""

DRAG_DIRECTION_SYSTEM_PROMPT = """\
You are the screen-space Drag Planner in a Blender Sculpt workflow. Inspect
the selected-view screenshot, the exact mouse-down coordinate, and the
current subtask. Return one straight 2D drag direction and distance that best
achieves the requested geometric change.

Coordinate rules:
1. The screenshot origin is at the top-left. Positive x points right and
   positive y points down.
2. direction is a nonzero [x, y] vector. It expresses orientation only; the
   workflow normalizes it before multiplying by distance_pixels.
3. distance_pixels is the requested mouse travel in screenshot pixels.

Planning rules:
4. Infer semantic directions from the visible model and the selected view.
   Do not confuse the model's left or right side with screen directions.
5. Choose enough distance to make the requested change visibly effective.
   Strong or large changes are valid when the instruction requests them.
6. Keep the endpoint inside the visible VIEW_3D image and leave practical
   margin from UI boundaries. The workflow will deterministically clip an
   unsafe endpoint, but clipping should not be necessary for a good plan.
7. For articulated Pose edits, reason about where the distal target should
   end relative to visible landmarks such as the torso, joints, or ground.
   Estimate distance from that desired endpoint rather than choosing an
   arbitrary small motion. An unqualified request to lower or drop an
   appendage requires a clearly recognizable resting/lowered pose, not merely
   a few pixels of displacement.
8. For an unqualified lower, drop, relax, or hang request, choose the nearest
   anatomically plausible lowered endpoint: normally put the distal point
   below its proximal joint and reduce its lateral distance from the torso.
   Do not push an appendage farther away from the body unless the instruction
   explicitly requests an outward or spreading motion.
9. Mentally mark the desired endpoint first. The returned direction must point
   from the supplied mouse-down coordinate to that endpoint, and
   distance_pixels must approximate their Euclidean screen-space distance.
10. The second image is the deterministic target-component and mouse-down
   overlay. Before planning motion, verify that its solid marker belongs to
   operation_location on the correct semantic instance of
   part_to_be_changed. Anatomical left and right are from the model's
   perspective, not screen position. Set anchor_target_valid to false when the
   marked instance is wrong or ambiguous; the workflow will relocalize and
   ignore the returned gesture. Explain this decision in
   anchor_target_analysis.
11. If a third image is attached, it is the deterministic kinematic Face Set
   visualization. Use its colored closed regions to understand the complete
   articulated chain and joint order; the first image remains the source of
   true geometry, landmarks, and screen coordinates.
12. Treat part_to_be_changed as the complete region that should deform and
    operation_location as the exact supplied mouse contact. Do not confuse the
    broad SAM3 target with the drag anchor.
13. Use retry feedback when present. If the previous result was too weak,
   increase the displacement meaningfully without changing the requested
   motion semantics.
14. Return only one straight gesture. Do not describe curves, multiple drags,
   camera movement, or brush settings.
"""


def svg_pattern_user_prompt(*, pattern_description: str) -> str:
    """Build the English text-to-SVG user prompt."""
    return f"""\
Generate one black-on-white SVG pattern from this user description:
{json.dumps(pattern_description, ensure_ascii=False)}

Preserve the requested visual identity while simplifying small details that
would not remain clear as a sculpted pattern. Express only the main components
with a few simple, long black strokes. Avoid short decorative or textural
lines and filled silhouettes. Follow every SVG contract rule from the system
prompt.
"""


def mask_component_selector_user_prompt(
    *,
    part_description: str,
    component_count: int,
) -> str:
    """Build the English numbered-mask selection prompt."""
    return f"""\
Target model part: {json.dumps(part_description, ensure_ascii=False)}

The attached Blender screenshot contains a cleaned segmentation overlay with
exactly {component_count} disconnected regions. Each region has a visible
numbered badge from 1 through {component_count}. Select the one region that
semantically corresponds to the target model part. Return its number and a
brief visual justification in the required JSON schema.
"""


def quadloc_user_prompt(
    *,
    location_description: str,
    depth: int,
    max_depth: int,
    crop_box: Mapping[str, int],
    available_regions: Sequence[str],
    rejected_regions: Sequence[str],
) -> str:
    """Build one English QuadLoc quadrant-classification prompt."""
    rejected = (
        "None."
        if not rejected_regions
        else ", ".join(rejected_regions) + "."
    )
    return f"""\
Locate this operation target: {location_description}

Recursive depth: {depth} of {max_depth}
Current crop in original screenshot coordinates:
{json.dumps(dict(crop_box), ensure_ascii=True, separators=(",", ":"))}

Color layout:
- RED: upper-left quadrant
- GREEN: upper-right quadrant
- BLUE: lower-left quadrant
- YELLOW: lower-right quadrant

Allowed region answers at this step: {", ".join(available_regions)}, NONE.
Rejected branches from an earlier backtrack: {rejected}

Inspect the visible content beneath the translucent overlays and return the
single allowed color that contains the target, or NONE if the target is not
present in this crop.
"""


def part_segmentation_user_prompt(
    *,
    part_description: str,
    max_subparts: int,
    max_synonym_attempts: int,
) -> str:
    """Build the English kinematic part-planning prompt."""
    return f"""\
Requested model part: {part_description}

Inspect the attached Blender VIEW_3D screenshot. Decide whether the requested
part is visibly identifiable. If it is visible, return between 1 and
{max_subparts} major kinematic subparts in proximal-to-distal order. Return one
instance-specific parent_sam3_prompt for the whole requested part. For each
subpart, keep identity qualifiers in label but use an unqualified visual class
phrase in sam3_prompt. Provide at most {max_synonym_attempts} useful
fallback_prompts. The parent mask will select the requested instance before
the generic child prompts are evaluated inside its ROI.
"""


def decomposer_user_prompt(
    *,
    user_instruction: str,
    screenshot_paths: Mapping[str, str],
    max_subtasks: int,
    sculpt_capabilities: Mapping[str, JsonValue],
) -> str:
    """Build the Decomposer user prompt in English."""
    return f"""\
Analyze the following user instruction and the attached labeled screenshots.
Produce between 1 and {max_subtasks} ordered subtasks.

<user_instruction>
{user_instruction}
</user_instruction>

Local Blender Sculpt capabilities:
{_sculpt_capability_manifest(sculpt_capabilities)}

Attached screenshot order:
{_screenshot_manifest(screenshot_paths)}
"""


def translator_user_prompt(
    *,
    subtasks: Sequence[Mapping[str, JsonValue]],
    screenshot_paths: Mapping[str, str],
    sculpt_capabilities: Mapping[str, JsonValue],
) -> str:
    """Build the Translator user prompt in English."""
    return f"""\
Translate every indexed subtask below into semantic Sculpt intent. Use the
attached fresh screenshots to judge operation scope, initial strength, and
the requested minimum visible effect. Every intent must independently return
both operation_location and part_to_be_changed according to their distinct
roles in the system prompt.

Subtasks:
{json.dumps(list(subtasks), ensure_ascii=False, indent=2)}

Local Blender Sculpt capabilities:
{_sculpt_capability_manifest(sculpt_capabilities)}

Attached screenshot order:
{_screenshot_manifest(screenshot_paths)}
"""


def view_selector_user_prompt(
    *,
    subtask: Mapping[str, JsonValue],
    intent: Mapping[str, JsonValue],
    screenshot_paths: Mapping[str, str],
    overlay_paths: Mapping[str, str],
    segmentation_summaries: Mapping[str, Mapping[str, JsonValue]],
    retry_feedback: Mapping[str, JsonValue] | None,
) -> str:
    """Build the View Selector user prompt in English."""
    feedback = (
        "No previous attempt feedback is available."
        if retry_feedback is None
        else json.dumps(dict(retry_feedback), ensure_ascii=False, indent=2)
    )
    return f"""\
Select the best view from the SAM3-valid candidates for this subtask attempt.
Only the candidate views listed below are allowed outputs.

Current subtask:
{json.dumps(dict(subtask), ensure_ascii=False, indent=2)}

Current translated intent:
{json.dumps(dict(intent), ensure_ascii=False, indent=2)}

Retry feedback:
{feedback}

Valid candidate views:
{json.dumps(list(screenshot_paths), ensure_ascii=False)}

SAM3 candidate summaries:
{json.dumps(dict(segmentation_summaries), ensure_ascii=False, indent=2)}

Attached candidate image order:
{_view_candidate_manifest(screenshot_paths, overlay_paths)}
"""


def drag_direction_user_prompt(
    *,
    subtask: Mapping[str, JsonValue],
    intent: Mapping[str, JsonValue],
    selected_view: str,
    start_coordinate: Mapping[str, JsonValue],
    image_width: int,
    image_height: int,
    retry_feedback: Mapping[str, JsonValue] | None,
    retry_directive: Mapping[str, JsonValue] | None,
    kinematic_overlay_attached: bool,
) -> str:
    """Build the English straight-gesture planning prompt."""
    kinematic_manifest = (
        "3. Kinematic Face Set visualization of the complete Pose chain."
        if kinematic_overlay_attached
        else "No kinematic Face Set visualization is attached."
    )
    return f"""\
Plan one straight Drag gesture for the attached selected-view screenshot.

Selected Blender view: {selected_view}
Screenshot size: {image_width} x {image_height} pixels
Mouse-down coordinate in top-left screenshot space:
{json.dumps(dict(start_coordinate), ensure_ascii=True)}

Current subtask:
{json.dumps(dict(subtask), ensure_ascii=False, indent=2)}

Current translated intent:
{json.dumps(dict(intent), ensure_ascii=False, indent=2)}

Previous-attempt feedback:
{json.dumps(dict(retry_feedback or {}), ensure_ascii=False, indent=2)}

Deterministic retry directive:
{json.dumps(dict(retry_directive or {}), ensure_ascii=False, indent=2)}

Attached image order:
1. Selected-view Blender screenshot used for all coordinates.
2. Bound target-component and mouse-down anchor overlay.
{kinematic_manifest}

First return anchor_target_valid and anchor_target_analysis. Then return the
screen-space direction [x, y], requested distance_pixels, and a brief visual
reason. Even if the anchor is invalid, return a schema-valid nonzero direction
and positive distance; the workflow will discard that gesture. The reason must
name the desired endpoint's relation to at least one visible landmark and
explain why the distance reaches it.
"""


def grader_user_prompt(
    *,
    user_instruction: str,
    subtask: Mapping[str, JsonValue],
    intent: Mapping[str, JsonValue],
    selected_view: str,
    minimum_effect: Mapping[str, JsonValue],
    image_labels: Sequence[str],
) -> str:
    """Build the evidence-only Grader prompt in English."""
    labels = "\n".join(
        f"{index}. {label}" for index, label in enumerate(image_labels, 1)
    )
    return f"""\
Grade this visible Sculpt result.

Original user instruction:
{user_instruction}

Current subtask:
{json.dumps(dict(subtask), ensure_ascii=False, indent=2)}

Translated target and execution-location semantics:
{json.dumps(dict(intent), ensure_ascii=False, indent=2)}

Selected view: {selected_view}

Minimum-effect gate result:
{json.dumps(dict(minimum_effect), ensure_ascii=False, indent=2)}

Attached image order:
{labels}
"""


def retry_planner_user_prompt(
    *,
    user_instruction: str,
    subtask: Mapping[str, JsonValue],
    intent: Mapping[str, JsonValue],
    resolved_plan: Mapping[str, JsonValue],
    selected_view: str,
    minimum_effect: Mapping[str, JsonValue],
    grader: Mapping[str, JsonValue],
    segment_context: Mapping[str, JsonValue] | None,
    image_labels: Sequence[str],
    sculpt_capabilities: Mapping[str, JsonValue],
) -> str:
    """Build the Retry Planner prompt in English."""
    labels = "\n".join(
        f"{index}. {label}" for index, label in enumerate(image_labels, 1)
    )
    return f"""\
Create the next-attempt repair plan.

Original user instruction:
{user_instruction}

Current subtask:
{json.dumps(dict(subtask), ensure_ascii=False, indent=2)}

Current semantic intent:
{json.dumps(dict(intent), ensure_ascii=False, indent=2)}

Resolved execution plan:
{json.dumps(dict(resolved_plan), ensure_ascii=False, indent=2)}

Selected view: {selected_view}

Minimum-effect result:
{json.dumps(dict(minimum_effect), ensure_ascii=False, indent=2)}

Visual Grader result:
{json.dumps(dict(grader), ensure_ascii=False, indent=2)}

SAM3 context:
{json.dumps(dict(segment_context or {}), ensure_ascii=False, indent=2)}

Local Blender Sculpt capabilities:
{_sculpt_capability_manifest(sculpt_capabilities)}

Attached image order:
{labels}
"""


def _screenshot_manifest(paths: Mapping[str, str]) -> str:
    return "\n".join(
        f"{index}. {view}: {path}"
        for index, (view, path) in enumerate(paths.items(), 1)
    )


def _sculpt_capability_manifest(
    capabilities: Mapping[str, JsonValue],
) -> str:
    """Render the runtime brush catalog compactly and deterministically."""
    version = capabilities.get("blender_version", "unknown")
    brushes = capabilities.get("brushes", [])
    return (
        f"Blender version: {version}\n"
        "Exact brush name -> allowed Direction values "
        "([] means brush_direction must be null):\n"
        + json.dumps(
            brushes,
            ensure_ascii=True,
            separators=(",", ":"),
        )
    )


def _view_candidate_manifest(
    screenshots: Mapping[str, str],
    overlays: Mapping[str, str],
) -> str:
    entries: list[str] = []
    for view, screenshot in screenshots.items():
        entries.append(
            f"{len(entries) + 1}. {view} candidate screenshot: {screenshot}"
        )
        overlay = overlays.get(view)
        if overlay is not None:
            entries.append(
                f"{len(entries) + 1}. {view} SAM3 overlay: {overlay}"
            )
    return "\n".join(entries)
