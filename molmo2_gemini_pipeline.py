"""PointBench-native Gemini rewrite and Molmo2+Gemini-judge pipelines."""

from __future__ import annotations

import ast
from contextlib import nullcontext
import io
import json
import os
import re
import socket
import tempfile
import threading
import time
import traceback
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import torch
from loguru import logger
from runtime_warnings import suppress_known_runtime_warnings

suppress_known_runtime_warnings()

from google import genai
from google.genai import types
from PIL import Image, ImageDraw


STRICT_JUDGE_FIELDS = [
    "rewrite_faithful",
    "point_count_correct",
    "all_points_clearly_inside_target",
    "spatial_relation_correct",
    "high_confidence",
]

MOLMO2_SINGLE_POINT_CONSTRAINT_PROMPT = """
Additional hard constraint for point count:
- Unless the instruction explicitly requires counting or multiple target instances, return exactly one point.
- If the instruction and the image do not clearly require multiple objects, output only one point.
- When uncertain between one point and multiple points, choose exactly one point.
- Return multiple points only for clear counting or explicit multi-instance requests supported by the image.
""".strip()

MULTI_COUNT_WORDS = ("two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
MULTI_COUNT_WORD_PATTERN = "|".join(MULTI_COUNT_WORDS)
DIRECT_TARGET_STOP_WORDS = (
    " that ", " who ", " which ", " where ", " used ", " capable ", " able ", " designed ",
    " relies ", " to ", " toward ", " into ", " between ", " behind ", " below ", " above ",
    " under ", " over ", " in front ", " next to ", " near ", " close to ", " at ", " on ",
    " in ", " from ", " with ", " for ", " by ", " past ", " until ", " of ",
)
SINGULAR_S_WORDS = {"lens"}
NON_PLURAL_ENDINGS = ("ss", "us", "is")

_GEMINI_CLIENTS: Dict[Tuple[str, str], Any] = {}
_GEMINI_CLIENTS_LOCK = threading.Lock()
_GEMINI_NETWORK_RETRIES = 3
_GEMINI_REQUEST_TIMEOUT_MS = 100_000
_MOLMO2_MODEL = None
_MOLMO2_PROCESSOR = None
_MOLMO2_MODEL_PATH = None
_MOLMO2_DEVICE = None
_MOLMO2_MODEL_LOCK = threading.Lock()
MOLMO2_MODEL_ALIASES = {
    "1-1": "allenai/Molmo2-8B",
    "1-2": "allenai/Molmo2-4B",
    "Molmo2-8B": "allenai/Molmo2-8B",
    "Molmo2-4B": "allenai/Molmo2-4B",
}
GEMINI_BOX_PROMPT_VERSION = 2
INTERNAL_HOSTED_GEMINI_BOX_SOURCE_NAME = "internal_hosted_api_box_center"
STAGE_CACHE_VERSION = 1
REWRITE_PROMPT_MODE = "legacy_reference_overlay"
REWRITE_SOURCE_PIPELINE_NAME = "transform_gemini_twolines"
PIPELINE_STAGE_ORDER = (
    "rewrite",
    "gemini_box",
    "molmo2_local",
    "gemini_judge",
    "gemini_fallback",
    "fallback_judge",
    "finalize",
)
_REFERENCE_MARKER_RADIUS_RATIO = 0.06
_REFERENCE_MARKER_MIN_RADIUS = 12
_GRID_LINE_WIDTH_RATIO = 0.002
_GRID_LINE_MIN_WIDTH = 1
_GRID_SPACING_RATIO = 0.12
_GRID_MIN_SPACING = 36
_DISTANCE_QUERY_PATTERN = re.compile(
    r"\b(?:closest|nearest|farthest|furthest|closer|farther|further|near)\b|\bnext to\b|\badjacent\b",
    re.IGNORECASE,
)


def _parse_json_payload(text: Any) -> Any:
    """Parse JSON or Python-literal payload from free-form model output."""
    if text is None:
        return None

    cleaned = str(text).strip().replace("```json", "```")
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()

    if not cleaned:
        return None

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    match = re.search(r"\{.*\}|\[.*\]", cleaned, flags=re.DOTALL)
    if not match:
        return None

    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        try:
            return ast.literal_eval(candidate)
        except Exception:
            return None


def _normalize_final_text_no_markdown(value: Any) -> str:
    """Normalize outputs for logging and downstream parsing."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value).strip()


def _get_original_points_context(image_path: str, image_points_map: Dict[str, Any]) -> Tuple[str, List[List[float]]]:
    """Build steerable reference point text plus pixel coordinates."""
    image_filename = os.path.basename(image_path)
    original_points = image_points_map.get(image_filename)
    if not original_points:
        return "", []

    with Image.open(image_path) as img:
        img_width, img_height = img.size

    original_points_in_pixels = []
    original_points_in_image = []
    for point in original_points:
        pixel_x = point["x"] * img_width / 100
        pixel_y = point["y"] * img_height / 100
        original_points_in_pixels.append(f"[{pixel_x:.1f}, {pixel_y:.1f}]")
        original_points_in_image.append([pixel_x, pixel_y])

    if not original_points_in_pixels:
        return "", []

    original_points_str = ", ".join(original_points_in_pixels)
    original_points_info = (
        f"\nThe image contains an existing original point at pixel coordinates: {original_points_str}."
        "\nThe query refers to this existing point."
    )
    return original_points_info, original_points_in_image


def _safe_path_part(value: Any) -> str:
    normalized = unicodedata.normalize("NFC", str(value).strip())
    normalized = normalized.replace("/", "_").replace("\\", "_")
    return re.sub(r"[^0-9A-Za-z._-]+", "_", normalized).strip("_")


def _get_gemini_client(api_key: str, base_url: str):
    cache_key = (api_key, base_url)
    with _GEMINI_CLIENTS_LOCK:
        cached_client = _GEMINI_CLIENTS.get(cache_key)
        if cached_client is not None:
            return cached_client

        client_kwargs = {"api_key": api_key}
        http_options_kwargs = {"timeout": _GEMINI_REQUEST_TIMEOUT_MS}
        if base_url:
            http_options_kwargs["base_url"] = base_url
        # google-genai uses milliseconds here; 100_000 means a 100-second request timeout.
        client_kwargs["http_options"] = types.HttpOptions(**http_options_kwargs)
        client = genai.Client(**client_kwargs)
        _GEMINI_CLIENTS[cache_key] = client
        return client


def _call_gemini(api_key: str, base_url: str, model_name: str, prompt_text: str, image_paths: List[str]) -> str:
    """Call Gemini with one prompt and optional images."""
    if not api_key or not model_name:
        return ""

    contents: List[Any] = [prompt_text]
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        file_extension = os.path.splitext(image_path)[1].lower()
        if file_extension == ".png":
            mime_type = "image/png"
        elif file_extension in {".jpg", ".jpeg"}:
            mime_type = "image/jpeg"
        elif file_extension == ".webp":
            mime_type = "image/webp"
        elif file_extension == ".gif":
            mime_type = "image/gif"
        else:
            mime_type = "image/jpeg"
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

    client = _get_gemini_client(api_key, base_url)
    for attempt in range(_GEMINI_NETWORK_RETRIES + 1):
        try:
            response = client.models.generate_content(model=model_name, contents=contents)
            return (response.text or "").strip()
        except Exception as error:
            if attempt >= _GEMINI_NETWORK_RETRIES or not _is_network_error(error):
                raise
            # Gemini API occasionally closes HTTP streams mid-flight; retry only transient network failures.
            logger.warning(
                f"Gemini network error, retry {attempt + 1}/{_GEMINI_NETWORK_RETRIES}: "
                f"{type(error).__name__}: {error}"
            )
            time.sleep(2 ** attempt)
    return ""


def _legacy_steerable_extra(category: str) -> str:
    """Keep the original legacy steerable rule block unchanged."""
    if category != "steerable":
        return ""
    return """
Steerable-specific rule:
- If the category is steerable, the image may contain a large red reference point already drawn on the image.
- Treat that large red point as the existing reference point mentioned in the instruction.
- Use the red reference point together with the image content to infer the intended final target.
- The rewritten instruction must name the actual target in the image, not the reference point itself.
- Do not keep phrases such as "existing point", "current point", "reference point", or "red point" in the final rewritten instruction unless the raw instruction is literally asking to point at that point itself.
""".strip()


def _build_legacy_rewrite_prompt(raw_input: str, category: str, original_points_info: str) -> Tuple[str, str]:
    """Use the exact legacy rewrite prompt from the original evaluate project."""
    system_prompt = f"""
You are an expert instruction rewriter for visual grounding and point-based segmentation models such as SAM-style models and GroundingDINO.

You will be given an image and a raw user instruction.

Your job is to understand what object, object part, object group, or image region the raw instruction refers to in the image, then rewrite it into a shorter, more explicit, and more visually grounded instruction.

Goals:
- preserve the original intent
- use the image to resolve what the instruction refers to
- rewrite the instruction into visually concrete language
- prefer explicit object names, attributes, locations, regions, and relations
- remove unnecessary conversational wording
- convert abstract functional descriptions into likely visible object names whenever possible
- make the result easier for detection, grounding, and segmentation models to understand

Rules:
- Output only one rewritten instruction.
- Use plain English.
- Keep it concise, ideally one sentence.
- Always begin with: "Point to ..."
- Do not explain your reasoning.
- Do not mention hidden category labels.
- Do not invent invisible facts.
- Use the image as the primary evidence for deciding the final target.
- Prefer visible nouns over human-intent descriptions.
- Preserve plurality when the original refers to multiple objects.
- Preserve spatial relations such as left, right, above, below, nearest, farthest, middle, front, and back.
- If the original describes movement, rewrite toward the final target object or final target region whenever possible.
- The rewritten instruction must already contain the final simple target and should not require another round of reasoning or navigation.
- Do not keep intermediate navigation language such as "existing point", "current point", "reference point", "red point", "move", "moving toward", "reach", or "until" unless the final target itself is literally that point or region.
- If the raw instruction is anchored to a reference point, use the image to resolve the final referred target, then name that final target directly.
- If the original describes function or affordance, rewrite to the most likely visible object class.
- If the original asks for identity evidence such as brand, country, or name, rewrite to the visible evidence such as logo, text, sign, flag, or license plate.
- If the original is about direction of movement, target the object, arrow, lane marking, wheel orientation, or front-facing side that visually indicates motion.
- If the original refers to free space or an empty region, describe that region using nearby visible anchors.
- If the raw instruction is vague but the image makes the target clear, rewrite to the visually clear target.
- If multiple interpretations are possible, choose the one best supported by the image.

Rewriting heuristics:
- Functional description -> concrete object name.
  - "the tool people use to eat soup" -> "Point to the spoon."
  - "the object used to open doors" -> "Point to the door handle." or "Point to the key."
  - "the tool people use to write" -> "Point to the pen." or "Point to the pencil."
- Movement instruction -> final grounded target.
  - "move down until you reach the search button" -> "Point to the search button."
  - "the button to the right of the existing point" -> "Point to the submit button." if the image shows that the final target is the submit button.
- Counting instruction -> explicit plural target.
  - "counting all the monkeys in the image" -> "Point to all the monkeys."
- Identity / evidence instruction -> visible evidence.
  - "the part of the train that indicates the country it might be from" -> "Point to the flag or country marking on the train."

Normalization preferences:
- Prefer standard visual labels such as: car, bus, bicycle, chair, table, window, door handle, spoon, fork, knife, pen, pencil, button, search bar, sign, logo, license plate, arrow, wheel, seat, mirror, handle.
- Prefer singular or plural to match the original meaning.
- Prefer the final resolved object name over any reference-point wording.
- Avoid mentioning reference points unless the target itself is that point.
{_legacy_steerable_extra(category)}
""".strip()
    user_prompt = f"""
Rewrite the following raw instruction into a visually grounded instruction.

Category:
{category or "unknown"}

{original_points_info}

Raw instruction:
{raw_input}

Return only the rewritten instruction.
""".strip()
    return system_prompt, user_prompt


def _build_legacy_reference_overlay_prompt(raw_input: str, category: str, original_points_info: str) -> Tuple[str, str]:
    """Match transform_gemini_twolines legacy_reference_overlay prompt behavior."""
    system_prompt, user_prompt = _build_legacy_rewrite_prompt(raw_input, category, original_points_info)
    overlay_guidance = """
Reference-overlay rules:
- Some images may contain one large red reference point.
- For distance-oriented or adjacency-oriented queries, the image may also include a faint grid centered on that same reference point, and the grid's center row and center column indicate the anchor alignment.
- Treat the red point and faint grid only as visual aids for resolving the final target.
- Use these aids to compare direction and relative distance for relations such as nearest, closest, farthest, closer, near, next to, and adjacent.
- Do not rewrite the target as the grid or reference point unless the raw instruction literally asks for them.
- If the visual aids make one final object or region unambiguous, rewrite directly to that final target.
""".strip()
    return f"{system_prompt}\n\n{overlay_guidance}", user_prompt


def _build_rewrite_prompt(
    raw_input: str,
    category: str,
    original_points_info: str,
    prompt_mode: str = REWRITE_PROMPT_MODE,
) -> Tuple[str, str]:
    if prompt_mode == "legacy":
        return _build_legacy_rewrite_prompt(raw_input, category, original_points_info)
    if prompt_mode == "legacy_reference_overlay":
        return _build_legacy_reference_overlay_prompt(raw_input, category, original_points_info)
    raise ValueError(f"Unknown rewrite prompt mode: {prompt_mode}")


def _has_reference_info(item: Dict[str, Any]) -> bool:
    return bool(str(item.get("original_points_info", "")).strip())


def _is_distance_query(query_text: str) -> bool:
    return bool(_DISTANCE_QUERY_PATTERN.search(query_text or ""))


def _reference_marker_radius(image_w: int, image_h: int) -> int:
    return max(_REFERENCE_MARKER_MIN_RADIUS, int(round(min(image_w, image_h) * _REFERENCE_MARKER_RADIUS_RATIO)))


def _draw_reference_marker(draw: ImageDraw.ImageDraw, point: Tuple[float, float], radius: int) -> None:
    cx, cy = int(round(point[0])), int(round(point[1]))
    outline_radius = radius + max(3, radius // 5)
    draw.ellipse([cx - outline_radius, cy - outline_radius, cx + outline_radius, cy + outline_radius], fill=(255, 255, 255))
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(255, 0, 0))


def _clamp_point(point: Tuple[float, float], width: int, height: int) -> Tuple[int, int]:
    return (
        int(round(min(max(float(point[0]), 0.0), max(width - 1, 0)))),
        int(round(min(max(float(point[1]), 0.0), max(height - 1, 0)))),
    )


def _draw_reference_grid(
    draw: ImageDraw.ImageDraw,
    point: Tuple[int, int],
    width: int,
    height: int,
) -> None:
    cx, cy = point
    spacing = max(_GRID_MIN_SPACING, int(round(min(width, height) * _GRID_SPACING_RATIO)))
    grid_width = max(_GRID_LINE_MIN_WIDTH, int(round(min(width, height) * _GRID_LINE_WIDTH_RATIO)))
    center_line_width = max(grid_width + 1, 2)

    # The centered guide lines act as the anchor alignment cue.
    draw.line([(0, cy), (width - 1, cy)], fill=(255, 120, 120, 140), width=center_line_width)
    draw.line([(cx, 0), (cx, height - 1)], fill=(255, 120, 120, 140), width=center_line_width)

    # Expand the grid symmetrically from the reference point to visualize relative distance.
    for offset in range(spacing, max(width, height) + spacing, spacing):
        left_x = cx - offset
        right_x = cx + offset
        top_y = cy - offset
        bottom_y = cy + offset
        if left_x >= 0:
            draw.line([(left_x, 0), (left_x, height - 1)], fill=(255, 210, 210, 95), width=grid_width)
        if right_x < width:
            draw.line([(right_x, 0), (right_x, height - 1)], fill=(255, 210, 210, 95), width=grid_width)
        if top_y >= 0:
            draw.line([(0, top_y), (width - 1, top_y)], fill=(255, 210, 210, 95), width=grid_width)
        if bottom_y < height:
            draw.line([(0, bottom_y), (width - 1, bottom_y)], fill=(255, 210, 210, 95), width=grid_width)


def _build_rewrite_overlay_meta(
    image_path: str,
    original_points_in_image: List[List[float]],
    query_text: str,
) -> Dict[str, Any]:
    with Image.open(image_path) as source_image:
        width, height = source_image.size
    clamped_points = [_clamp_point((point_x, point_y), width, height) for point_x, point_y in original_points_in_image]
    return {
        "reference_points_count": len(clamped_points),
        "reference_points_in_image": clamped_points,
        "distance_query": _is_distance_query(query_text),
        "crosshair_enabled": len(clamped_points) == 1 and not _is_distance_query(query_text),
        "grid_enabled": len(clamped_points) == 1 and _is_distance_query(query_text),
        "image_size": {"width": width, "height": height},
    }


def _build_rewrite_overlay_output_paths(visualizations_dir: str, item: Dict[str, Any]) -> Tuple[str, str, str]:
    category = item.get("category", "unknown") or "unknown"
    image_name = unicodedata.normalize("NFC", item.get("image_filename") or os.path.basename(item["image_path"]))
    image_stem, _ = os.path.splitext(image_name)
    save_dir = unicodedata.normalize("NFC", os.path.join(visualizations_dir, "rewrite_overlay", category))
    overlay_path = unicodedata.normalize("NFC", os.path.join(save_dir, f"{image_stem}_reference_overlay.png"))
    meta_path = unicodedata.normalize("NFC", os.path.join(save_dir, f"{image_stem}_reference_overlay.json"))
    return save_dir, overlay_path, meta_path


def _save_rewrite_overlay_artifacts(
    visualizations_dir: str,
    item: Dict[str, Any],
    reference_image: Image.Image,
    overlay_meta: Dict[str, Any],
    prompt_mode: str,
    model_name: str,
) -> Tuple[str, str]:
    save_dir, overlay_path, meta_path = _build_rewrite_overlay_output_paths(visualizations_dir, item)
    os.makedirs(save_dir, exist_ok=True)
    reference_image.save(overlay_path, format="PNG")
    with open(meta_path, "w", encoding="utf-8") as meta_file:
        json.dump(
            {
                "pipeline": item.get("pipeline_name", "molmo2_guidance_dualquery_refpoint_hybrid_gemini_judge"),
                "rewrite_source_pipeline": REWRITE_SOURCE_PIPELINE_NAME,
                "rewrite_prompt_mode": prompt_mode,
                "planner_model_name": model_name,
                "image_filename": item.get("image_filename"),
                "image_path": item.get("image_path"),
                "category": item.get("category", ""),
                "raw_user_input": item.get("user_input", ""),
                "original_points_info": item.get("original_points_info", ""),
                **overlay_meta,
                "overlay_image_path": overlay_path,
            },
            meta_file,
            ensure_ascii=False,
            indent=2,
        )
    return overlay_path, meta_path


def _build_reference_marker_image(
    image_path: str,
    original_points_in_image: List[List[float]],
    query_text: str,
) -> Optional[Image.Image]:
    if not original_points_in_image:
        return None
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGBA")
    image_w, image_h = image.size
    radius = _reference_marker_radius(image_w, image_h)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    clamped_points = [_clamp_point((point_x, point_y), image_w, image_h) for point_x, point_y in original_points_in_image]

    # Only add the distance grid for single-anchor queries to avoid clutter.
    if len(clamped_points) == 1 and _is_distance_query(query_text):
        _draw_reference_grid(draw, clamped_points[0], image_w, image_h)

    for point in clamped_points:
        _draw_reference_marker(draw, point, radius)
    return Image.alpha_composite(image, overlay).convert("RGB")


def _build_reference_marker_image_path(
    image_path: str,
    original_points_in_image: List[List[float]],
    query_text: str,
    visualizations_dir: str = "",
    item: Optional[Dict[str, Any]] = None,
    prompt_mode: str = REWRITE_PROMPT_MODE,
    model_name: str = "",
) -> Optional[str]:
    reference_image = _build_reference_marker_image(image_path, original_points_in_image, query_text)
    if reference_image is None:
        return None
    if visualizations_dir and item is not None:
        overlay_meta = _build_rewrite_overlay_meta(image_path, original_points_in_image, query_text)
        overlay_path, meta_path = _save_rewrite_overlay_artifacts(
            visualizations_dir,
            item,
            reference_image,
            overlay_meta,
            prompt_mode,
            model_name,
        )
        item["rewrite_overlay_path"] = overlay_path
        item["rewrite_overlay_meta_path"] = meta_path
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    reference_image.save(temp_path, format="PNG")
    return temp_path


def call_transform_gemini(
    image_path: str,
    object_name: str,
    model_name: str = "",
    category: Optional[str] = None,
    item_ctx: Optional[Dict[str, Any]] = None,
    runtime_options: Optional[Dict[str, Any]] = None,
) -> str:
    """PointBench-native rewrite aligned with transform_gemini_twolines."""
    del object_name
    options = runtime_options or {}
    item = item_ctx if item_ctx is not None else {}
    item["image_path"] = image_path
    item["category"] = category or item.get("category", "")
    item["user_input"] = str(item.get("user_input") or "").strip()
    item["pipeline_name"] = "molmo2_guidance_dualquery_refpoint_hybrid_gemini_judge"

    image_points_map = options.get("image_points_map", {})
    original_points_info, original_points_in_image = _get_original_points_context(image_path, image_points_map)
    item["original_points_info"] = original_points_info
    item["original_points_in_image"] = original_points_in_image
    item["rewrite_overlay_path"] = ""
    item["rewrite_overlay_meta_path"] = ""

    api_key = str(options.get("api_key") or os.getenv("API_KEY", "")).strip()
    base_url = str(options.get("base_url") or os.getenv("API_BASE_URL", "")).strip()
    resolved_model_name = str(model_name or options.get("enhance_model") or os.getenv("API_MODEL_NAME") or os.getenv("SA2VA_PLANNER_MODEL", "gemini-3.5-flash")).strip()
    prompt_mode = str(options.get("rewrite_prompt_mode") or REWRITE_PROMPT_MODE).strip() or REWRITE_PROMPT_MODE
    visualizations_dir = str(options.get("visualizations_dir") or "").strip()

    temp_reference_path: Optional[str] = None
    try:
        if _has_reference_info(item):
            temp_reference_path = _build_reference_marker_image_path(
                image_path,
                item.get("original_points_in_image", []) or [],
                item["user_input"],
                visualizations_dir=visualizations_dir,
                item=item,
                prompt_mode=prompt_mode,
                model_name=resolved_model_name,
            )
        resolved_image_path = temp_reference_path or image_path
        system_prompt, user_prompt = _build_rewrite_prompt(
            item["user_input"],
            item.get("category", ""),
            item.get("original_points_info", ""),
            prompt_mode=prompt_mode,
        )
        combined_prompt = f"{system_prompt}\n\n{user_prompt}".strip()
        content = _call_gemini(api_key, base_url, resolved_model_name, combined_prompt, [resolved_image_path])
        return content.replace("```json", "").replace("```", "").strip()
    finally:
        if temp_reference_path:
            try:
                os.remove(temp_reference_path)
            except OSError:
                pass


def _build_pointing_prompt(question_text: str) -> str:
    return f"pointing: {str(question_text).strip()}".strip()


def _build_dualquery_guidance_prompt(
    raw_user_input: str,
    enhanced_user_input: str,
    question_field: str,
    guidance_points: List[List[int]],
    category: str,
) -> str:
    category_name = str(category or "").strip().lower()
    if guidance_points:
        guidance_text = (
            f"An upstream helper proposed {len(guidance_points)} numbered guidance point(s) for this query. "
            "These helper points are not visualized in the image for this ablation, and you must verify the target directly from the image."
        )
        task_instruction = "- Use the enhanced query and the helper guidance-point proposal as grounding hints."
        if category_name == "counting":
            point_count_instruction = (
                "- Return one point for each valid visible target instance that truly matches the query.\n"
                "- You may use the helper proposal count as a weak prior, but do not trust it blindly if the image disagrees."
            )
        else:
            point_count_instruction = (
                "- Return exactly one point.\n"
                "- Use the helper proposal only to narrow the candidate target, then place the final point on the visible main body of the real target object.\n"
                f"{MOLMO2_SINGLE_POINT_CONSTRAINT_PROMPT}"
            )
    else:
        guidance_text = (
            "No helper guidance points are available for this sample, so solve the task only from the image plus the original/enhanced text."
        )
        task_instruction = "- Use the enhanced query as an extra grounding hint."
        if category_name == "counting":
            point_count_instruction = "- Return one point for each valid visible target instance that matches the query."
        else:
            point_count_instruction = (
                "- Return exactly one point.\n"
                f"{MOLMO2_SINGLE_POINT_CONSTRAINT_PROMPT}"
            )

    prompt_body = f"""
Original user query:
{raw_user_input}

Enhanced query ({question_field}):
{enhanced_user_input}

{guidance_text}

Instructions:
- Use the original user query as the final task.
{task_instruction}
- Every returned point must lie on the visible surface of the real target object, preferably near the object's main visible body rather than the edge.
- Do not place points on empty background or heavily occluded tiny fragments.
{point_count_instruction}
    """.strip()
    return _build_pointing_prompt(prompt_body)


def _build_refpoint_box_explained_nolabel_prompt(
    raw_user_input: str,
    enhanced_user_input: str,
    question_field: str,
    pixel_boxes: List[List[int]],
    category: str,
    guidance_points: List[List[int]],
) -> str:
    if not pixel_boxes:
        return _build_dualquery_guidance_prompt(
            raw_user_input,
            enhanced_user_input,
            question_field,
            guidance_points,
            category,
        )

    box_count = len(pixel_boxes)
    category_name = str(category or "").strip().lower()
    guidance_text = (
        f"The image contains {box_count} red box(es). "
        "Each red box is a candidate target region proposed by an upstream helper for the enhanced query. "
        "These red boxes are visual grounding hints only, and you must still verify the target directly from the image."
    )
    task_instruction = "- Use the enhanced query and the red boxes as grounding hints."
    box_validity_instruction = "- If a red box does not actually contain a valid target, ignore it."

    if category_name == "counting":
        point_count_instruction = (
            "- Return one point for each valid visible target instance that truly matches the query.\n"
            "- When a matching red box is available, place the point on the real target inside that box.\n"
            "- Do not return points for boxes that miss the intended target."
        )
    else:
        point_count_instruction = (
            "- Return exactly one point.\n"
            "- If one red box best matches the final target, place the point on the visible main body of the real target inside that box.\n"
            "- Do not place the point on the box border or empty background.\n"
            f"{MOLMO2_SINGLE_POINT_CONSTRAINT_PROMPT}"
        )

    prompt_body = f"""
Original user query:
{raw_user_input}

Enhanced query ({question_field}):
{enhanced_user_input}

{guidance_text}

Instructions:
- Use the original user query as the final task.
{task_instruction}
- Every returned point must lie on the visible surface of the real target object, preferably near the object's main visible body rather than the edge.
- Do not place points on empty background or heavily occluded tiny fragments.
{box_validity_instruction}
{point_count_instruction}
    """.strip()
    return _build_pointing_prompt(prompt_body)


def _build_nolabel_box_overlay_image(image_path: str, pixel_boxes: List[List[int]]) -> Image.Image:
    with Image.open(image_path) as source_image:
        canvas = source_image.convert("RGB")

    if not pixel_boxes:
        return canvas

    draw = ImageDraw.Draw(canvas)
    line_width = max(2, min(canvas.size) // 220)
    for box in pixel_boxes:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 64, 64), width=line_width)
    return canvas


def _save_hybrid_box_debug_artifacts(
    overlay_image: Image.Image,
    returned_points: List[List[int]],
    save_dir: str,
    image_stem: str,
) -> Dict[str, str]:
    os.makedirs(save_dir, exist_ok=True)
    input_visualization_path = unicodedata.normalize(
        "NFC",
        os.path.join(save_dir, f"{image_stem}_dualquery_box_explained_nolabel_input.png"),
    )
    point_visualization_path = unicodedata.normalize(
        "NFC",
        os.path.join(save_dir, f"{image_stem}_dualquery_box_explained_nolabel_points.png"),
    )

    overlay_image.save(input_visualization_path)
    canvas = overlay_image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for point in returned_points:
        _draw_marker(draw, point, marker_color=(255, 0, 0), use_cross=True)
    canvas.save(point_visualization_path)
    return {
        "input_visualization_path": input_visualization_path,
        "point_visualization_path": point_visualization_path,
    }


def _save_debug_meta(debug_meta_path: str, debug_meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(debug_meta_path), exist_ok=True)
    with open(debug_meta_path, "w", encoding="utf-8") as meta_file:
        json.dump(debug_meta, meta_file, ensure_ascii=False, indent=2)


def _load_debug_meta(debug_meta_path: str) -> Dict[str, Any]:
    if not debug_meta_path or not os.path.exists(debug_meta_path):
        return {}
    try:
        with open(debug_meta_path, "r", encoding="utf-8") as meta_file:
            payload = json.load(meta_file)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _default_completed_stages() -> Dict[str, bool]:
    return {stage_name: False for stage_name in PIPELINE_STAGE_ORDER}


def _coerce_point_pairs(point_values: Any, use_int: bool = True) -> List[List[int]]:
    if not isinstance(point_values, list):
        return []

    normalized_points: List[List[int]] = []
    for point_value in point_values:
        if not isinstance(point_value, (list, tuple)) or len(point_value) < 2:
            continue
        try:
            point_x = float(point_value[0])
            point_y = float(point_value[1])
        except (TypeError, ValueError):
            continue
        if use_int:
            normalized_points.append([int(round(point_x)), int(round(point_y))])
        else:
            normalized_points.append([point_x, point_y])
    return normalized_points


def _coerce_box_list(box_values: Any, use_int: bool = True) -> List[List[int]]:
    if not isinstance(box_values, list):
        return []

    normalized_boxes: List[List[int]] = []
    for box_value in box_values:
        if not isinstance(box_value, (list, tuple)) or len(box_value) < 4:
            continue
        try:
            coords = [float(box_value[index]) for index in range(4)]
        except (TypeError, ValueError):
            continue
        if use_int:
            normalized_boxes.append([int(round(coord)) for coord in coords])
        else:
            normalized_boxes.append(coords)
    return normalized_boxes


def _coerce_final_point_payload(point_values: Any) -> Optional[List[Dict[str, List[float]]]]:
    if point_values is None:
        return None
    if not isinstance(point_values, list):
        return []

    normalized_points: List[Dict[str, List[float]]] = []
    for point_value in point_values:
        coords = point_value.get("point") if isinstance(point_value, dict) else point_value
        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            continue
        try:
            point_x = float(coords[0])
            point_y = float(coords[1])
        except (TypeError, ValueError):
            continue
        normalized_points.append({"point": [point_x, point_y]})
    return normalized_points


def _path_exists(path_value: str) -> bool:
    return bool(path_value) and os.path.exists(path_value)


def _normalize_model_root_for_signature(model_root: str) -> str:
    model_root_text = str(model_root or "").strip()
    if not model_root_text:
        return ""
    return str(Path(model_root_text).expanduser())


def _build_stage_cache_signature(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pipeline": "molmo2_guidance_dualquery_refpoint_hybrid_gemini_judge",
        "question_field": state["question_field"],
        "planner_model_name": state["planner_model_name"],
        "rewrite_model_name": state["rewrite_model_name"],
        "resolved_model_name": state["resolved_model_name"],
        "model_root": _normalize_model_root_for_signature(state["model_root"]),
        "max_tokens": state["max_tokens"],
    }


def _get_completed_stages(debug_meta: Dict[str, Any]) -> Dict[str, bool]:
    completed_stages = _default_completed_stages()
    cached_stages = debug_meta.get("completed_stages")
    if isinstance(cached_stages, dict):
        for stage_name in PIPELINE_STAGE_ORDER:
            completed_stages[stage_name] = bool(cached_stages.get(stage_name))
    return completed_stages


def _stage_cache_signature_matches(state: Dict[str, Any], cached_meta: Dict[str, Any]) -> bool:
    cached_signature = cached_meta.get("stage_cache_signature")
    current_signature = _build_stage_cache_signature(state)
    if cached_signature == current_signature:
        return True

    completed_stages = _get_completed_stages(cached_meta)
    rewrite_only = completed_stages["rewrite"] and not any(
        completed_stages[stage_name] for stage_name in PIPELINE_STAGE_ORDER if stage_name != "rewrite"
    )
    if not rewrite_only or not isinstance(cached_signature, dict):
        return False

    return (
        cached_signature.get("pipeline") == current_signature["pipeline"]
        and cached_signature.get("question_field") == current_signature["question_field"]
    )


def _rebuild_helper_guidance(debug_meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "box_visualization_path": str(debug_meta.get("helper_box_visualization_path") or ""),
        "point_visualization_path": str(debug_meta.get("helper_point_visualization_path") or ""),
        "guidance_points_xy": _coerce_point_pairs(debug_meta.get("guidance_points_xy"), use_int=True),
        "matched_box_centers_xy": _coerce_point_pairs(debug_meta.get("matched_box_centers_xy"), use_int=True),
        "pixel_boxes_xyxy": _coerce_box_list(debug_meta.get("pixel_boxes_xyxy"), use_int=True),
        "helper_box_prompt": str(debug_meta.get("helper_box_prompt") or ""),
        "helper_box_raw_response": str(debug_meta.get("helper_box_raw_response") or ""),
        "helper_box_parsed_response": debug_meta.get("helper_box_parsed_response"),
        "helper_box_reasoning_trace": debug_meta.get("helper_box_reasoning_trace", {}) or {},
        "helper_box_target_resolution": debug_meta.get("helper_box_target_resolution", {}) or {},
        "helper_box_reason": str(debug_meta.get("helper_box_reason") or ""),
        "normalized_boxes_yxyx_0_1000": _coerce_box_list(debug_meta.get("normalized_boxes_yxyx_0_1000"), use_int=False),
        "normalized_points_yx_0_1000": _coerce_point_pairs(debug_meta.get("normalized_points_yx_0_1000"), use_int=False),
    }


def _restore_stage_cache(state: Dict[str, Any]) -> Dict[str, Any]:
    # Stage cache lives in the per-sample justify_meta.json so reruns of the same
    # command can resume from the last finished stage instead of redoing API work.
    if not state["runtime_options"].get("resume", False):
        return state

    cached_meta = _load_debug_meta(state["debug_meta_path"])
    if not cached_meta:
        return state
    if cached_meta.get("stage_cache_version") != STAGE_CACHE_VERSION:
        return state
    if not _stage_cache_signature_matches(state, cached_meta):
        return state

    state["debug_meta"].update(cached_meta)
    state["debug_meta"]["completed_stages"] = _get_completed_stages(state["debug_meta"])

    state["enhanced_user_input"] = str(state["debug_meta"].get("enhanced_user_input") or "")
    if state["enhanced_user_input"]:
        state["item"]["enhanced_query"] = state["enhanced_user_input"]
        state["item"][state["question_field"]] = state["enhanced_user_input"]

    state["helper_guidance"] = _rebuild_helper_guidance(state["debug_meta"])
    state["local_route"] = str(state["debug_meta"].get("local_route") or "")
    state["prompt_text"] = str(state["debug_meta"].get("local_prompt_text") or "")
    state["molmo2_response"] = str(state["debug_meta"].get("local_response_text") or "")
    state["candidate_points"] = _coerce_point_pairs(state["debug_meta"].get("local_points"), use_int=True)
    state["local_box_artifacts"] = {
        "input_visualization_path": str(state["debug_meta"].get("local_input_image_path") or ""),
        "point_visualization_path": str(state["debug_meta"].get("local_box_point_visualization_path") or ""),
    }
    state["local_point_only_path"] = str(state["debug_meta"].get("local_point_only_image_path") or "")
    state["judge_point_only_path"] = str(state["debug_meta"].get("point_only_image_path") or "")
    state["judge_prompt"] = str(state["debug_meta"].get("judge_prompt") or "")
    state["judge_raw"] = str(state["debug_meta"].get("judge_raw") or "")
    state["judge_result"] = state["debug_meta"].get("judge_parsed") if isinstance(state["debug_meta"].get("judge_parsed"), dict) else None
    state["strict_accept"] = bool(state["debug_meta"].get("judge_strict_accept"))
    state["fallback_prompt"] = str(state["debug_meta"].get("gemini_fallback_prompt") or "")
    state["fallback_raw"] = str(state["debug_meta"].get("gemini_fallback_raw") or "")
    state["fallback_points"] = _coerce_point_pairs(state["debug_meta"].get("gemini_fallback_points"), use_int=True)
    state["fallback_viz_path"] = str(state["debug_meta"].get("gemini_first_point_only_image_path") or "")
    state["fallback_judge_raw"] = str(state["debug_meta"].get("gemini_fallback_final_judge_raw") or "")
    fallback_judge_parsed = state["debug_meta"].get("gemini_fallback_final_judge_parsed")
    state["fallback_judge_result"] = fallback_judge_parsed if isinstance(fallback_judge_parsed, dict) else None
    state["fallback_strict_accept"] = bool(state["debug_meta"].get("gemini_fallback_accept"))
    state["final_decision"] = str(state["debug_meta"].get("final_decision") or "")

    completed_stages = _get_completed_stages(state["debug_meta"])
    if completed_stages["finalize"]:
        state["final_points"] = _coerce_final_point_payload(state["debug_meta"].get("final_response"))
    elif completed_stages["molmo2_local"] and not state["original_points_in_image"]:
        state["final_points"] = _coerce_final_point_payload(state["debug_meta"].get("final_response"))
    elif completed_stages["gemini_judge"] and state["final_decision"] == "judge_accept_keep_molmo2":
        state["final_points"] = _coerce_final_point_payload(state["debug_meta"].get("final_response"))
    elif completed_stages["gemini_fallback"] and state["final_decision"] == "fallback_empty_keep_molmo2":
        state["final_points"] = _coerce_final_point_payload(state["debug_meta"].get("final_response"))
    elif completed_stages["fallback_judge"] and state["final_decision"] in {"fallback_accept_use_gemini", "fallback_reject_keep_molmo2"}:
        state["final_points"] = _coerce_final_point_payload(state["debug_meta"].get("final_response"))

    state["pipeline_error"] = ""
    return state


def _save_stage_cache(state: Dict[str, Any], completed_stage: Optional[str] = None) -> Dict[str, Any]:
    debug_meta = state.get("debug_meta")
    if not isinstance(debug_meta, dict):
        return state

    completed_stages = _get_completed_stages(debug_meta)
    if completed_stage:
        completed_stages[completed_stage] = True
    debug_meta["completed_stages"] = completed_stages
    debug_meta["stage_cache_version"] = STAGE_CACHE_VERSION
    debug_meta["stage_cache_signature"] = _build_stage_cache_signature(state)
    debug_meta["enhanced_user_input"] = state.get("enhanced_user_input", "")
    debug_meta["local_route"] = state.get("local_route", "")
    debug_meta["local_prompt_text"] = state.get("prompt_text", "")
    debug_meta["local_response_text"] = state.get("molmo2_response", "")
    debug_meta["local_points"] = state.get("candidate_points", [])
    debug_meta["judge_prompt"] = state.get("judge_prompt", "")
    debug_meta["judge_raw"] = state.get("judge_raw", "")
    debug_meta["judge_parsed"] = state.get("judge_result")
    debug_meta["judge_strict_accept"] = state.get("strict_accept")
    debug_meta["gemini_fallback_prompt"] = state.get("fallback_prompt", "")
    debug_meta["gemini_fallback_raw"] = state.get("fallback_raw", "")
    debug_meta["gemini_fallback_points"] = state.get("fallback_points", [])
    debug_meta["gemini_fallback_final_judge_raw"] = state.get("fallback_judge_raw", "")
    debug_meta["gemini_fallback_final_judge_parsed"] = state.get("fallback_judge_result")
    debug_meta["gemini_fallback_accept"] = state.get("fallback_strict_accept")
    debug_meta["final_decision"] = state.get("final_decision", "")
    debug_meta["final_response"] = state.get("final_points", []) if state.get("final_points") is not None else []
    debug_meta["pipeline_error"] = state.get("pipeline_error", "")
    debug_meta["pipeline_error_reason"] = debug_meta.get("pipeline_error_reason", "") if state.get("pipeline_error") else ""
    debug_meta["pipeline_error_traceback"] = (
        debug_meta.get("pipeline_error_traceback", "") if state.get("pipeline_error") else ""
    )
    _save_debug_meta(state["debug_meta_path"], debug_meta)
    return state


def _invalidate_stage_cache_from(state: Dict[str, Any], stage_name: str) -> Dict[str, Any]:
    debug_meta = state.get("debug_meta")
    if not isinstance(debug_meta, dict):
        return state

    clear_from_current = False
    completed_stages = _get_completed_stages(debug_meta)
    for cached_stage_name in PIPELINE_STAGE_ORDER:
        if cached_stage_name == stage_name:
            clear_from_current = True
        if clear_from_current:
            completed_stages[cached_stage_name] = False
    debug_meta["completed_stages"] = completed_stages
    state["final_points"] = None
    state["final_decision"] = ""
    state["pipeline_error"] = ""
    debug_meta["final_decision"] = ""
    debug_meta["final_response"] = []
    debug_meta["pipeline_error"] = ""
    debug_meta["pipeline_error_reason"] = ""
    debug_meta["pipeline_error_traceback"] = ""
    return state


def _stage_cache_ready(state: Dict[str, Any], stage_name: str) -> bool:
    completed_stages = _get_completed_stages(state.get("debug_meta", {}))
    if not completed_stages.get(stage_name):
        return False

    if stage_name == "rewrite":
        return bool(state.get("enhanced_user_input"))
    if stage_name == "gemini_box":
        return (
            bool(state["debug_meta"].get("helper_box_prompt"))
            or bool(state["debug_meta"].get("helper_box_raw_response"))
            or bool(state["debug_meta"].get("helper_box_reason"))
        )
    if stage_name == "molmo2_local":
        if not state["original_points_in_image"]:
            return bool(state.get("molmo2_response")) and state.get("final_points") is not None
        return bool(state.get("molmo2_response")) and _path_exists(state.get("judge_point_only_path", ""))
    if stage_name == "gemini_judge":
        return (
            state.get("strict_accept", False)
            or isinstance(state.get("judge_result"), dict)
            or bool(state.get("judge_raw"))
            or bool(state["debug_meta"].get("judge_hard_reject_reason"))
        )
    if stage_name == "gemini_fallback":
        return (
            state.get("final_decision") == "fallback_empty_keep_molmo2"
            or bool(state.get("fallback_raw"))
            or (bool(state.get("fallback_points")) and _path_exists(state.get("fallback_viz_path", "")))
        )
    if stage_name == "fallback_judge":
        return (
            state.get("final_decision") in {"fallback_accept_use_gemini", "fallback_reject_keep_molmo2"}
            or isinstance(state.get("fallback_judge_result"), dict)
            or bool(state.get("fallback_judge_raw"))
            or state["debug_meta"].get("gemini_fallback_accept") is not None
        )
    if stage_name == "finalize":
        return state.get("final_points") is not None and bool(state.get("final_decision"))
    return False


def _begin_stage(state: Dict[str, Any], stage_name: str) -> bool:
    if state.get("pipeline_error"):
        return False
    if _stage_cache_ready(state, stage_name):
        return False
    _invalidate_stage_cache_from(state, stage_name)
    return True


def _points_to_float_payload(points: List[List[int]]) -> List[Dict[str, List[float]]]:
    return [{"point": [float(point_x), float(point_y)]} for point_x, point_y in points]


def _clamp_0_1000(value: float) -> float:
    return max(0.0, min(float(value), 1000.0))


def _normalize_gemini_box(box_value: Any) -> Optional[List[float]]:
    if isinstance(box_value, dict):
        for field_name in ("bbox_2d", "bbox", "box_2d", "box", "coordinates"):
            if field_name in box_value:
                return _normalize_gemini_box(box_value[field_name])
        return None

    if not isinstance(box_value, (list, tuple)) or len(box_value) != 4:
        return None

    try:
        top, left, bottom, right = [_clamp_0_1000(float(coord)) for coord in box_value]
    except (TypeError, ValueError):
        return None

    y1, y2 = sorted((top, bottom))
    x1, x2 = sorted((left, right))
    if y2 <= y1 or x2 <= x1:
        return None
    return [y1, x1, y2, x2]


def _extract_gemini_boxes_from_payload(payload: Any) -> List[List[float]]:
    if isinstance(payload, dict):
        direct_box = _normalize_gemini_box(payload)
        if direct_box is not None:
            return [direct_box]

        for field_name in ("boxes", "final_boxes", "predicted_boxes", "candidate_boxes"):
            nested_value = payload.get(field_name)
            if isinstance(nested_value, list):
                return _extract_gemini_boxes_from_payload(nested_value)

        for field_name in ("final_answer", "decision", "result"):
            nested_value = payload.get(field_name)
            extracted_boxes = _extract_gemini_boxes_from_payload(nested_value)
            if extracted_boxes:
                return extracted_boxes
        return []

    if not isinstance(payload, list):
        return []

    extracted_boxes: List[List[float]] = []
    for item_value in payload:
        normalized_box = _normalize_gemini_box(item_value)
        if normalized_box is not None:
            extracted_boxes.append(normalized_box)
    return extracted_boxes


def _extract_gemini_boxes_from_text(cleaned_text: str) -> List[List[float]]:
    box_matches = re.findall(
        r"(?:bbox_2d|bbox|box_2d|box)\s*\"?\s*[:=]\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]",
        cleaned_text,
        flags=re.IGNORECASE,
    )
    extracted_boxes: List[List[float]] = []
    for top, left, bottom, right in box_matches:
        normalized_box = _normalize_gemini_box([top, left, bottom, right])
        if normalized_box is not None:
            extracted_boxes.append(normalized_box)
    return extracted_boxes


def _normalized_box_to_pixel_xyxy(box: List[float], image_w: int, image_h: int) -> List[int]:
    y1, x1, y2, x2 = box
    pixel_box = [
        int(round((float(x1) / 1000.0) * image_w)),
        int(round((float(y1) / 1000.0) * image_h)),
        int(round((float(x2) / 1000.0) * image_w)),
        int(round((float(y2) / 1000.0) * image_h)),
    ]
    pixel_box[0] = max(0, min(pixel_box[0], max(image_w - 1, 0)))
    pixel_box[1] = max(0, min(pixel_box[1], max(image_h - 1, 0)))
    pixel_box[2] = max(0, min(pixel_box[2], max(image_w - 1, 0)))
    pixel_box[3] = max(0, min(pixel_box[3], max(image_h - 1, 0)))
    return pixel_box


def _normalize_gemini_box_response(
    text: str,
    category_name: str,
    image_w: int,
    image_h: int,
) -> Tuple[Any, List[List[float]], List[List[float]], List[List[int]], List[List[int]]]:
    cleaned_text = _normalize_final_text_no_markdown(text)
    parsed_payload = _parse_json_payload(cleaned_text)
    normalized_boxes = _extract_gemini_boxes_from_payload(parsed_payload)
    if not normalized_boxes:
        normalized_boxes = _extract_gemini_boxes_from_text(cleaned_text)
    if category_name != "counting":
        normalized_boxes = normalized_boxes[:1]

    normalized_points = [
        [
            (box[0] + box[2]) / 2.0,
            (box[1] + box[3]) / 2.0,
        ]
        for box in normalized_boxes
    ]
    pixel_points = [_recover_gemini_point(point[0], point[1], image_w, image_h) for point in normalized_points]
    pixel_boxes = [_normalized_box_to_pixel_xyxy(box, image_w, image_h) for box in normalized_boxes]
    return parsed_payload, normalized_boxes, normalized_points, pixel_boxes, pixel_points


def _draw_gemini_box_visualization(
    base_image: Image.Image,
    pixel_boxes: List[List[int]],
    pixel_points: List[List[int]],
    original_points: List[List[int]],
    title: str,
) -> Image.Image:
    canvas = base_image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.size[0], 34], fill=(255, 255, 255))
    draw.text((8, 7), title, fill=(0, 0, 0))

    for point in original_points:
        _draw_marker(draw, point, marker_color=(0, 0, 255), use_cross=False)

    for index, box in enumerate(pixel_boxes, start=1):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=(255, 64, 64), width=3)
        label_right = min(canvas.size[0] - 1, x1 + 38)
        label_top = max(34, y1 - 22)
        draw.rectangle([x1, label_top, label_right, label_top + 18], fill=(255, 255, 255))
        draw.text((x1 + 4, label_top + 1), f"#{index}", fill=(255, 64, 64))

    for point in pixel_points:
        _draw_marker(draw, point, marker_color=(255, 0, 0), use_cross=True)
    return canvas


def _save_gemini_box_visualizations(
    save_dir: str,
    image_stem: str,
    base_image: Image.Image,
    pixel_boxes: List[List[int]],
    pixel_points: List[List[int]],
    original_points: List[List[int]],
) -> Dict[str, str]:
    os.makedirs(save_dir, exist_ok=True)
    box_only_path = unicodedata.normalize("NFC", os.path.join(save_dir, f"{image_stem}_gemini_boxes.jpg"))
    point_only_path = unicodedata.normalize("NFC", os.path.join(save_dir, f"{image_stem}_gemini_box_centers.jpg"))

    _draw_gemini_box_visualization(
        base_image,
        pixel_boxes,
        pixel_points,
        original_points,
        title="GEMINI BOXES + BOX CENTERS",
    ).save(box_only_path)
    _save_points_visualization(
        base_image,
        pixel_points,
        original_points,
        save_dir,
        os.path.basename(point_only_path),
    )
    return {
        "box_visualization_path": box_only_path,
        "point_visualization_path": point_only_path,
    }


def _build_gemini_box_grounding_prompt(
    raw_user_input: str,
    enhanced_user_input: str,
    question_field: str,
    category_name: str,
    original_points_info: str,
) -> str:
    reference_rule = ""
    if original_points_info:
        reference_rule = (
            "- The image may contain blue reference point(s) mentioned by the query.\n"
            "- Treat those reference point(s) only as anchors for relation understanding, never as automatic answer boxes.\n"
        )

    if category_name == "counting":
        count_rule = "- For counting tasks, return one box per clearly visible intended target instance.\n"
    else:
        count_rule = (
            "- For non-counting tasks, return exactly one final box.\n"
            "- If several objects could match, choose the single object that best fits the image, common sense, and the most natural reading of the original user input.\n"
        )

    return f"""You are a box-based visual grounding analyst.
You will receive one image, the original user query, and an auxiliary enhanced query.

Your task is to reason first and localize second.
The original user query is the primary instruction.
The enhanced query is only a helper and must be ignored if it changes the target, relation, or counting scope.

Reason in this exact order:
Phase A. Scene Scan: summarize the main scene, the major objects, and any obvious anchors.
Phase B. Instruction Resolution: read the original user query first, then the enhanced query, and decide what the target should mean in this image.
Phase C. Candidate Screening: compare plausible candidates and explain which ones should be downweighted or rejected.
Phase D. Final Box Decision: choose the final target set and output tight boxes around the visible main body of each selected target.
Phase E. Center Justification: explain why the center of each final box would land inside the intended target body.

Selection policy:
- Prioritize the main subject or the clearest intended target.
- Downweight tiny, edge-cut, heavily blurred, weakly visible, or background-only instances unless the instruction explicitly asks for them.
- Do not over-focus on faint boundary objects or marginal edge clutter when a clearer main-body target better satisfies the query.
- Boxes should cover the visible target body, not a large surrounding region, empty context, shadow, or motion blur tail.
- When several objects are plausible in a non-counting task, choose only the single best-supported one.
{count_rule}{reference_rule}
Output rules:
- Return JSON only.
- Do not return markdown.
- Do not return point coordinates in the final answer.
- Use Gemini normalized box coordinates in `[y1, x1, y2, x2]` format, with each value in the 0-1000 range.
- Keep every reasoning field concise but concrete.

Task category: {category_name}
Original user input: {raw_user_input}
Enhanced text input ({question_field}): {enhanced_user_input}

Return exactly this JSON schema:
{{
  "reasoning_trace": {{
    "phase_a_scene_scan": "brief scene scan",
    "phase_b_instruction_resolution": "how the instruction is resolved",
    "phase_c_candidate_screening": "which candidates are kept or rejected and why",
    "phase_d_final_box_decision": "why the final box set is chosen",
    "phase_e_center_justification": "why each box center lands inside the target"
  }},
  "target_resolution": {{
    "primary_source": "original_user_input or enhanced_text_input",
    "task_mode": "single or counting",
    "final_target_summary": "concise target summary",
    "ignore_marginal_instances": true or false
  }},
  "final_answer": {{
    "boxes": [
      {{
        "bbox_2d": [y1, x1, y2, x2],
        "object_name": "brief object name",
        "why_this_box": "brief reason"
      }}
    ]
  }},
  "reason": "brief final grounding reason"
}}"""


def _run_internal_gemini_box_grounding(
    image_path: str,
    image: Image.Image,
    image_w: int,
    image_h: int,
    item: Dict[str, Any],
    question_field: str,
    planner_model_name: str,
    api_key: str,
    base_url: str,
    category_dir: str,
    image_stem: str,
) -> Tuple[Dict[str, Any], bool]:
    raw_user_input = str(item.get("user_input") or "").strip()
    enhanced_user_input = str(item.get(question_field) or "").strip()
    category_name = str(item.get("category") or "").strip()
    original_points = item.get("original_points_in_image", []) or []
    prompt_text = _build_gemini_box_grounding_prompt(
        raw_user_input,
        enhanced_user_input,
        question_field,
        category_name,
        str(item.get("original_points_info") or "").strip(),
    )

    try:
        raw_response = _call_gemini(api_key, base_url, planner_model_name, prompt_text, [image_path])
    except Exception as error:
        if _is_network_error(error):
            logger.exception(
                f"Gemini helper box grounding network error for image={item.get('image_filename') or image_path}"
            )
            return {}, True
        logger.exception(f"Gemini helper box grounding failed for image={item.get('image_filename') or image_path}")
        raw_response = ""

    parsed_payload, normalized_boxes, normalized_points, pixel_boxes, pixel_points = _normalize_gemini_box_response(
        raw_response,
        category_name,
        image_w,
        image_h,
    )

    visualizations = {"box_visualization_path": "", "point_visualization_path": ""}
    if pixel_boxes or pixel_points:
        visualizations = _save_gemini_box_visualizations(
            category_dir,
            image_stem,
            image,
            pixel_boxes,
            pixel_points,
            original_points,
        )

    return {
        "source_field": INTERNAL_HOSTED_GEMINI_BOX_SOURCE_NAME,
        "guidance_points_xy": pixel_points,
        "matched_box_centers_xy": pixel_points,
        "pixel_boxes_xyxy": pixel_boxes,
        "normalized_boxes_yxyx_0_1000": normalized_boxes,
        "normalized_points_yx_0_1000": normalized_points,
        "helper_box_prompt": prompt_text,
        "helper_box_raw_response": raw_response,
        "helper_box_parsed_response": parsed_payload,
        "helper_box_reasoning_trace": parsed_payload.get("reasoning_trace", {}) if isinstance(parsed_payload, dict) else {},
        "helper_box_target_resolution": parsed_payload.get("target_resolution", {}) if isinstance(parsed_payload, dict) else {},
        "helper_box_reason": parsed_payload.get("reason", "") if isinstance(parsed_payload, dict) else "",
        "box_visualization_path": visualizations["box_visualization_path"],
        "point_visualization_path": visualizations["point_visualization_path"],
    }, False


def _normalize_query_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _extract_direct_target_phrase(query_text: str) -> str:
    match = re.search(r"\bpoint to\b\s+(.*)", query_text)
    if not match:
        return query_text

    phrase = match.group(1).strip(" .")
    first_stop_idx = None
    for stop_word in DIRECT_TARGET_STOP_WORDS:
        stop_idx = phrase.find(stop_word)
        if stop_idx > 0 and (first_stop_idx is None or stop_idx < first_stop_idx):
            first_stop_idx = stop_idx
    if first_stop_idx is not None:
        phrase = phrase[:first_stop_idx].strip(" .")
    return re.sub(r"^(?:the\s+)", "", phrase).strip(" .")


def _has_explicit_multi_target(query_text: str) -> bool:
    if re.search(r"\bpoint to\s+(?:counting\s+)?(?:all|every|each|both)\b", query_text):
        return True
    if re.search(rf"\bpoint to\s+(?:the\s+)?(?:first\s+)?(?:{MULTI_COUNT_WORD_PATTERN})\b", query_text):
        return True
    if re.search(r"\bpoint to\s+(?:the\s+)?(?:first\s+)?(?:[2-9]|10)\b(?!-)", query_text):
        return True
    return bool(re.search(r"\bpoint to\s+(?:the\s+)?[a-z-]+s\s+of\s+all\b", query_text))


def _direct_target_looks_plural(target_phrase: str) -> bool:
    if target_phrase.startswith(("what ", "where ", "which ", "moving ", "move ", "slightly move ", "directly ")):
        return False
    if "all caps" in target_phrase or re.search(r"\bone of\b", target_phrase):
        return False
    if re.match(rf"^(?:one|1|first)\b(?!\s+(?:{MULTI_COUNT_WORD_PATTERN}))", target_phrase):
        return False

    words = re.findall(r"[a-z]+", target_phrase)
    if not words:
        return False
    last_word = words[-1]
    if last_word in SINGULAR_S_WORDS or len(last_word) <= 2:
        return False
    if last_word.endswith(NON_PLURAL_ENDINGS):
        return False
    return last_word.endswith("s")


def _query_requests_multiple(raw_user_input: str, enhanced_user_input: str) -> bool:
    raw_query_text = _normalize_query_text(raw_user_input)
    enhanced_query_text = _normalize_query_text(enhanced_user_input)
    if _has_explicit_multi_target(raw_query_text):
        return True
    if _direct_target_looks_plural(_extract_direct_target_phrase(raw_query_text)):
        return True
    return _has_explicit_multi_target(enhanced_query_text)


def _load_molmo2_official_model(model_name: str, model_root: str):
    global _MOLMO2_MODEL, _MOLMO2_PROCESSOR, _MOLMO2_MODEL_PATH, _MOLMO2_DEVICE

    from transformers import AutoModelForImageTextToText, AutoProcessor

    resolved_name = MOLMO2_MODEL_ALIASES.get(str(model_name or "").strip(), str(model_name or "").strip())
    if not resolved_name:
        resolved_name = "allenai/Molmo2-4B"

    if model_root:
        root_path = Path(model_root).expanduser()
        if root_path.is_dir():
            repo_relative_path = Path(*resolved_name.split("/"))
            expected_path = root_path / repo_relative_path
            if expected_path.exists():
                resolved_path = str(expected_path.resolve())
            elif (root_path / "config.json").exists():
                resolved_path = str(root_path.resolve())
            else:
                raise FileNotFoundError(
                    "Molmo2 local weights not found at "
                    f"{expected_path}. Leave --model_root empty to load {resolved_name} "
                    "from HuggingFace, or place the model under "
                    f"{expected_path}."
                )
        elif root_path.exists():
            resolved_path = str(root_path.resolve())
        else:
            raise FileNotFoundError(
                f"Provided --model_root does not exist: {root_path}. "
                f"Leave --model_root empty to load {resolved_name} from HuggingFace."
            )
    else:
        # When --model_root is empty, transformers will resolve the HuggingFace repo id
        # and download/cache the weights in the user's HF cache directory.
        resolved_path = resolved_name

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with _MOLMO2_MODEL_LOCK:
        if (
            _MOLMO2_MODEL is None
            or _MOLMO2_PROCESSOR is None
            or _MOLMO2_MODEL_PATH != resolved_path
            or _MOLMO2_DEVICE != device
        ):
            processor = AutoProcessor.from_pretrained(
                resolved_path,
                trust_remote_code=True,
                padding_side="left",
            )
            if hasattr(processor, "image_processor"):
                processor.image_processor.max_crops = 24
                processor.image_processor.max_images = 20

            model = AutoModelForImageTextToText.from_pretrained(
                resolved_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
            ).to(device)
            model.eval()

            _MOLMO2_PROCESSOR = processor
            _MOLMO2_MODEL = model
            _MOLMO2_MODEL_PATH = resolved_path
            _MOLMO2_DEVICE = device

    return _MOLMO2_PROCESSOR, _MOLMO2_MODEL


def _build_molmo2_prompt(question_text: str) -> str:
    return f"pointing: {str(question_text).strip()}\n\n{MOLMO2_SINGLE_POINT_CONSTRAINT_PROMPT}".strip()


def _decode_generation(processor: Any, generated_ids: Any, prompt_len: int) -> str:
    generated_tokens = generated_ids[:, prompt_len:]
    if hasattr(processor, "post_process_image_text_to_text"):
        return processor.post_process_image_text_to_text(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    return processor.decode(generated_tokens[0], skip_special_tokens=True)


def _extract_molmo2_points(response_text: str, image_w: int, image_h: int) -> List[List[int]]:
    coord_regex = re.compile(r'<(?:points|tracks).*? coords="([0-9\t:;, .]+)"\s*/?>')
    frame_regex = re.compile(r"(?:^|\t|:|,|;)([0-9.]+) ([0-9. ]+)")
    points_regex = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")

    extracted_points: List[List[int]] = []
    for coord_match in coord_regex.finditer(response_text):
        coord_text = coord_match.group(1)
        for frame_match in frame_regex.finditer(coord_text):
            points_text = frame_match.group(2)
            for point_match in points_regex.finditer(points_text):
                pixel_x = float(point_match.group(2)) / 1000 * image_w
                pixel_y = float(point_match.group(3)) / 1000 * image_h
                if 0 <= pixel_x <= image_w and 0 <= pixel_y <= image_h:
                    extracted_points.append([int(pixel_x), int(pixel_y)])

    if extracted_points:
        return extracted_points

    parsed = _parse_json_payload(response_text)
    if isinstance(parsed, list):
        recovered: List[List[int]] = []
        for item in parsed:
            if isinstance(item, dict):
                point = item.get("point")
            else:
                point = item
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            try:
                recovered.append([int(float(point[0])), int(float(point[1]))])
            except Exception:
                continue
        return recovered
    return []


def _run_molmo2_item(
    image_path: str,
    prompt_text: str,
    model_name: str,
    model_root: str,
    max_new_tokens: int,
    image_override: Optional[Image.Image] = None,
) -> str:
    processor, model = _load_molmo2_official_model(model_name, model_root)
    if image_override is None:
        with Image.open(image_path) as source_image:
            image = source_image.convert("RGB")
    else:
        image = image_override.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image", "image": image},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    )
    image.close()

    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    prompt_len = inputs["input_ids"].size(1)

    if str(model.device).startswith("cuda"):
        autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast_context = nullcontext()

    with torch.inference_mode(), autocast_context:
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=min(int(max_new_tokens), 256),
            do_sample=False,
        )

    response_text = _decode_generation(processor, generated_ids, prompt_len).strip()
    with Image.open(image_path) as check_image:
        image_w, image_h = check_image.size
    points = _extract_molmo2_points(response_text, image_w, image_h)
    if points:
        return json.dumps([{"point": point} for point in points], ensure_ascii=False)
    return response_text


def _draw_marker(draw: ImageDraw.ImageDraw, point: List[float], marker_color: Tuple[int, int, int], use_cross: bool = True) -> None:
    cx, cy = int(point[0]), int(point[1])
    outline_color = (255, 255, 255)
    outer_radius = 6
    inner_radius = 5
    cross_len = 4
    draw.ellipse([cx - outer_radius - 1, cy - outer_radius - 1, cx + outer_radius + 1, cy + outer_radius + 1], fill=outline_color)
    draw.ellipse([cx - outer_radius, cy - outer_radius, cx + outer_radius, cy + outer_radius], fill=marker_color)
    draw.ellipse([cx - inner_radius, cy - inner_radius, cx + inner_radius, cy + inner_radius], fill=outline_color)
    if use_cross:
        draw.line([cx - cross_len, cy, cx + cross_len, cy], fill=marker_color, width=2)
        draw.line([cx, cy - cross_len, cx, cy + cross_len], fill=marker_color, width=2)


def _save_points_visualization(base_image: Image.Image, points: List[List[int]], original_points: List[List[int]], save_dir: str, filename: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    save_path = unicodedata.normalize("NFC", os.path.join(save_dir, filename))
    canvas = base_image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for point in original_points:
        _draw_marker(draw, point, marker_color=(0, 0, 255), use_cross=False)
    for point in points:
        _draw_marker(draw, point, marker_color=(255, 0, 0), use_cross=True)
    canvas.save(save_path)
    return save_path


def _draw_hollow_marker(draw: ImageDraw.ImageDraw, cx: int, cy: int, radius: int, marker_color: Tuple[int, int, int], use_cross: bool) -> None:
    white = (255, 255, 255)
    line_width = max(3, radius // 4)
    outline_width = line_width + 2
    cross_len = max(radius + 4, radius * 2)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=white, width=outline_width)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=marker_color, width=line_width)
    draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=marker_color)
    if use_cross:
        draw.line([cx - cross_len, cy, cx + cross_len, cy], fill=white, width=outline_width)
        draw.line([cx, cy - cross_len, cx, cy + cross_len], fill=white, width=outline_width)
        draw.line([cx - cross_len, cy, cx + cross_len, cy], fill=marker_color, width=line_width)
        draw.line([cx, cy - cross_len, cx, cy + cross_len], fill=marker_color, width=line_width)


def _build_judge_visualization(base_image: Image.Image, points: List[List[int]], original_points: List[List[int]]) -> Image.Image:
    base = base_image.convert("RGB")
    image_w, image_h = base.size
    label_h = 28
    gap = 12
    resample = Image.Resampling.LANCZOS

    def draw_marker_on_panel(draw: ImageDraw.ImageDraw, point: List[float], scale_x: float, scale_y: float, marker_color: Tuple[int, int, int], use_cross: bool, radius: int) -> None:
        cx = int(point[0] * scale_x)
        cy = int(point[1] * scale_y)
        _draw_hollow_marker(draw, cx, cy, radius, marker_color, use_cross)

    full_max_side = 1280
    full_scale = min(1.0, full_max_side / max(image_w, image_h))
    full_w = max(1, int(image_w * full_scale))
    full_h = max(1, int(image_h * full_scale))
    full_view = base.resize((full_w, full_h), resample)
    full_draw = ImageDraw.Draw(full_view)
    full_radius = max(8, min(24, min(full_w, full_h) // 55))
    for point in original_points:
        draw_marker_on_panel(full_draw, point, full_scale, full_scale, (0, 0, 255), False, full_radius)
    for point in points:
        draw_marker_on_panel(full_draw, point, full_scale, full_scale, (255, 0, 0), True, full_radius)

    full_panel = Image.new("RGB", (full_w, full_h + label_h), (255, 255, 255))
    ImageDraw.Draw(full_panel).text((8, 7), "FULL IMAGE: red=candidate, blue=reference", fill=(0, 0, 0))
    full_panel.paste(full_view, (0, label_h))

    crop_panels = []
    crop_size = 384
    crop_half = max(96, min(512, min(image_w, image_h) // 6))
    for idx, point in enumerate(points[:8]):
        cx, cy = int(point[0]), int(point[1])
        left = max(0, cx - crop_half)
        top = max(0, cy - crop_half)
        right = min(image_w, cx + crop_half)
        bottom = min(image_h, cy + crop_half)
        crop = base.crop((left, top, right, bottom)).resize((crop_size, crop_size), resample)
        crop_draw = ImageDraw.Draw(crop)
        local_x = int((cx - left) * crop_size / max(1, right - left))
        local_y = int((cy - top) * crop_size / max(1, bottom - top))
        _draw_hollow_marker(crop_draw, local_x, local_y, 20, (255, 0, 0), True)

        crop_panel = Image.new("RGB", (crop_size, crop_size + label_h), (255, 255, 255))
        ImageDraw.Draw(crop_panel).text((8, 7), f"ZOOM RED POINT {idx + 1}", fill=(0, 0, 0))
        crop_panel.paste(crop, (0, label_h))
        crop_panels.append(crop_panel)

    if not crop_panels:
        return full_panel

    crop_column_h = sum(panel.height for panel in crop_panels) + gap * (len(crop_panels) - 1)
    canvas_w = full_panel.width + gap + crop_size
    canvas_h = max(full_panel.height, crop_column_h)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    canvas.paste(full_panel, (0, 0))
    y_offset = 0
    for panel in crop_panels:
        canvas.paste(panel, (full_panel.width + gap, y_offset))
        y_offset += panel.height + gap
    return canvas


def _recover_gemini_point(norm_y: float, norm_x: float, orig_w: int, orig_h: int) -> List[int]:
    pixel_x = int((float(norm_x) / 1000.0) * orig_w)
    pixel_y = int((float(norm_y) / 1000.0) * orig_h)
    return [pixel_x, pixel_y]


def _extract_gemini_point_response(text: str, orig_w: int, orig_h: int, category_name: str) -> List[List[int]]:
    parsed = _parse_json_payload(text)
    if isinstance(parsed, list):
        converted_points: List[List[int]] = []
        for item_value in parsed:
            if isinstance(item_value, dict):
                point = item_value.get("point")
            elif isinstance(item_value, (list, tuple)) and len(item_value) == 2:
                point = item_value
            else:
                continue
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            try:
                converted_points.append(_recover_gemini_point(float(point[0]), float(point[1]), orig_w, orig_h))
            except Exception:
                continue
        if category_name != "counting" and len(converted_points) > 1:
            converted_points = [converted_points[0]]
        return converted_points

    coords = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", text or "")
    converted_points = []
    for y_str, x_str in coords:
        try:
            converted_points.append(_recover_gemini_point(float(y_str), float(x_str), orig_w, orig_h))
        except Exception:
            continue
    if category_name != "counting" and len(converted_points) > 1:
        converted_points = [converted_points[0]]
    return converted_points


def _get_hard_reject_reason(candidate_points: List[List[int]], image_w: int, image_h: int) -> str:
    if not candidate_points:
        return "hard_rule_reject:no_candidate_points"
    for idx, point in enumerate(candidate_points):
        if point is None or len(point) < 2:
            return f"hard_rule_reject:invalid_point_at_{idx}"
        x, y = int(point[0]), int(point[1])
        if x < 0 or y < 0 or x >= image_w or y >= image_h:
            return f"hard_rule_reject:point_{idx}_outside_image_bounds"
    return ""


def _make_rejected_judge_result(reason: str) -> Dict[str, Any]:
    skipped_summary = f"Skipped because the sample was hard-rejected before Gemini verification: {reason}."
    return {
        "verification_trace": {
            "step_1_global_observation": skipped_summary,
            "step_2_instruction_deconstruction": skipped_summary,
            "step_3_expected_target_derivation": skipped_summary,
            "step_4_candidate_point_analysis": f"Candidate point analysis could not proceed because of the hard rejection: {reason}.",
            "step_5_final_match_verification": f"Final decision: reject before Gemini verification because {reason}.",
        },
        "rewrite_faithful": False,
        "point_count_correct": False,
        "all_points_clearly_inside_target": False,
        "spatial_relation_correct": False,
        "high_confidence": False,
        "accept": False,
        "reason": reason,
    }


def _get_strict_judge_accept(judge_result: Any, hard_reject_reason: str) -> bool:
    if hard_reject_reason or not isinstance(judge_result, dict):
        return False
    if judge_result.get("accept") is not True:
        return False
    return all(judge_result.get(field) is True for field in STRICT_JUDGE_FIELDS)


def _build_molmo2_judge_prompt(raw_user_input: str, enhanced_user_input: str, question_field: str) -> str:
    return f'''You are a strict and conservative verifier for point grounding results.

Your goal is to maximize precision:
- Accept only when the red point result is clearly correct.
- If there is any uncertainty, ambiguity, missing target, extra target, boundary issue, or rewrite mismatch, reject.
- It is better to reject a correct result than to accept an incorrect result.

You will receive one visualization image:
- this is the original image without segmentation mask overlay;
- red points are the candidate point results;
- blue points, if present, are reference/context points;
- blue reference points are not prediction targets unless the textual instruction explicitly requires the same location.

Use the text inputs explicitly in your reasoning:
- Treat the original user input as the source of truth.
- Use the enhanced text only as auxiliary context, and ignore it if it conflicts with the original user input.
- Do not jump from the red points to a verdict. First infer the expected target from the text and the full image, then compare the red points to that expected target.

Perform and return this exact 5-step "System 2" verification trace:
Step 1. Global Observation: inspect the whole scene, identify the major objects, and locate all plausible targets and anchors.
Step 2. Instruction Deconstruction: parse the original user input first, then the enhanced text, and extract the exact target semantics, count constraints, and spatial relations.
Step 3. Expected Target Derivation: before judging any red point, determine which object or set of objects should be selected in this image, and verify whether blue points are anchors or targets in this query.
Step 4. Candidate Point Analysis: inspect every red point one by one and state what exact object/body region it falls on, whether it is clearly inside the target boundary, and whether it satisfies the required relation to anchors or referenced objects.
Step 5. Final Match Verification: compare the full red-point set against the expected target set from Steps 1-3, then decide whether the candidate set is strictly correct.

Strict accept criteria:
1. The candidate must satisfy the original user input, not merely a plausible rewrite.
2. Infer from the original user input whether the task asks for one target or multiple targets.
3. If the instruction asks for one specific target, there must be exactly one red point.
4. If the instruction asks for all/every/both/multiple targets, all clearly visible target instances must be pointed to.
5. The center of every red marker must be clearly inside the visible target object/body.
6. Reject if a red point is only near the target, on boundary/background/shadow, or on an adjacent object.
7. For spatial or steerable tasks, verify the exact relation to the blue reference point or referenced object.
8. Reject if any red point lands on a plausible object that is not the final target derived from the text and scene.
9. If confidence is not high, reject.

Rules:
- Return JSON only.
- Do not generate new points.
- Return the full 5-step verification trace in the JSON.
- Do not leave any verification step blank.
- Keep each step concise, concrete, and grounded in the actual image and text.
- Set "accept" to true only when all strict checks pass.

Original user input: {raw_user_input}
Enhanced text input ({question_field}): {enhanced_user_input}

Return exactly this JSON schema:
{{
  "verification_trace": {{
    "step_1_global_observation": "concise scene-level observation",
    "step_2_instruction_deconstruction": "concise parsing of the textual instruction",
    "step_3_expected_target_derivation": "which target should be selected before checking red points",
    "step_4_candidate_point_analysis": "what each red point falls on and whether it is valid",
    "step_5_final_match_verification": "final comparison between expected target set and candidate points"
  }},
  "rewrite_faithful": true or false,
  "point_count_correct": true or false,
  "all_points_clearly_inside_target": true or false,
  "spatial_relation_correct": true or false,
  "high_confidence": true or false,
  "accept": true or false,
  "reason": "brief reason"
}}'''


def _build_fallback_grounding_prompt(raw_user_input: str, enhanced_user_input: str, question_field: str) -> str:
    return f'''You are a visual point grounding model.
You will receive the original image and must produce a corrected grounding answer.

The previous candidate was rejected by a strict verifier.
Use the original user input as the source of truth.
Use the enhanced text only as auxiliary context, and ignore it if it conflicts with the original input.
Do not repeat the previous mistake.

Rules:
- Return JSON only.
- Infer from the original user input whether one or multiple targets are required.
- If one target is required, return exactly one point.
- If multiple targets are explicitly required, return one point per clearly visible target instance.
- Use Gemini coordinate format: [y, x] normalized to 0-1000.
- Prefer this schema: [{{"point": [y, x]}}].
- Do not provide explanation text.

Original user input: {raw_user_input}
Enhanced text input ({question_field}): {enhanced_user_input}
'''


def _is_network_error(error: BaseException) -> bool:
    network_type_names = {
        "APIConnectionError",
        "ConnectError",
        "ConnectTimeout",
        "PoolTimeout",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "WriteError",
        "WriteTimeout",
    }
    network_message_tokens = (
        "connection aborted",
        "connection refused",
        "connection reset",
        "dns",
        "name resolution",
        "network is unreachable",
        "remote protocol error",
        "server disconnected",
        "temporarily unavailable",
        "timed out",
        "timeout",
    )

    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, (TimeoutError, ConnectionError, socket.timeout, socket.gaierror, httpx.RequestError)):
            return True
        if type(current).__name__ in network_type_names:
            return True
        if any(token in str(current).lower() for token in network_message_tokens):
            return True
        current = current.__cause__ or current.__context__
    return False


def _build_initial_debug_meta(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pipeline": "molmo2_guidance_dualquery_refpoint_hybrid_gemini_judge",
        "stage_cache_version": STAGE_CACHE_VERSION,
        "stage_cache_signature": _build_stage_cache_signature(state),
        "completed_stages": _default_completed_stages(),
        "image_path": state["image_path"],
        "image_filename": state["image_filename"],
        "category": state["category"],
        "question_field": state["question_field"],
        "raw_user_input": state["raw_user_input"],
        "rewrite_source_pipeline": REWRITE_SOURCE_PIPELINE_NAME,
        "rewrite_prompt_mode": REWRITE_PROMPT_MODE,
        "rewrite_model_name": state["rewrite_model_name"],
        "rewrite_overlay_image_path": "",
        "rewrite_overlay_meta_path": "",
        "enhanced_user_input": "",
        "local_route": "",
        "local_prompt_source_pipeline": "molmo2_guidance_dualquery_refpoint_hybrid",
        "judge_requires_reference_points": True,
        "molmo2_guidance_source_field": INTERNAL_HOSTED_GEMINI_BOX_SOURCE_NAME,
        "molmo2_with_box_source_field": INTERNAL_HOSTED_GEMINI_BOX_SOURCE_NAME,
        "helper_sidecar_path": "",
        "helper_visualization_path": "",
        "guidance_points_xy": [],
        "matched_box_centers_xy": [],
        "pixel_boxes_xyxy": [],
        "helper_box_prompt_version": GEMINI_BOX_PROMPT_VERSION,
        "helper_box_prompt": "",
        "helper_box_raw_response": "",
        "helper_box_parsed_response": None,
        "helper_box_reasoning_trace": {},
        "helper_box_target_resolution": {},
        "helper_box_reason": "",
        "normalized_boxes_yxyx_0_1000": [],
        "normalized_points_yx_0_1000": [],
        "helper_box_visualization_path": "",
        "helper_point_visualization_path": "",
        "original_points_in_image": state["original_points_in_image"],
        "local_prompt_text": "",
        "local_input_image_path": "",
        "local_box_point_visualization_path": "",
        "local_point_only_image_path": "",
        "point_only_image_path": "",
        "local_response_text": "",
        "local_points": [],
        "judge_skipped": False,
        "judge_skip_reason": "",
        "judge_prompt": "",
        "judge_hard_reject_reason": "",
        "judge_raw": "",
        "judge_parsed": None,
        "judge_model_accept": None,
        "judge_accept": None,
        "judge_strict_accept": None,
        "judge_reason": "",
        "judge_verification_trace": {},
        "gemini_fallback_prompt": "",
        "gemini_fallback_raw": "",
        "gemini_fallback_points": [],
        "gemini_fallback_response_text": "",
        "gemini_first_point_only_image_path": "",
        "gemini_fallback_final_judge_raw": "",
        "gemini_fallback_final_judge_parsed": None,
        "gemini_fallback_final_judge_verification_trace": {},
        "gemini_fallback_accept": None,
        "pipeline_error": "",
        "pipeline_error_reason": "",
        "pipeline_error_traceback": "",
        "final_decision": "",
        "final_response": [],
    }


def _mark_pipeline_error(
    state: Dict[str, Any],
    error_code: str,
    reason: str = "",
    traceback_text: str = "",
) -> Dict[str, Any]:
    state["pipeline_error"] = error_code
    debug_meta = state.get("debug_meta")
    if isinstance(debug_meta, dict):
        debug_meta["pipeline_error"] = error_code
        debug_meta["pipeline_error_reason"] = reason or error_code
        debug_meta["pipeline_error_traceback"] = traceback_text
    return _save_stage_cache(state)


def build_molmo2_gemini_pipeline_state(
    image_path: str,
    model_name: str = "allenai/Molmo2-4B",
    category: Optional[str] = None,
    item_ctx: Optional[Dict[str, Any]] = None,
    runtime_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    options = runtime_options or {}
    item = item_ctx if item_ctx is not None else {}
    item["image_path"] = image_path
    item["image_filename"] = item.get("image_filename") or os.path.basename(image_path)
    item["category"] = category or item.get("category", "")
    item["user_input"] = str(item.get("user_input") or "").strip()

    image_points_map = options.get("image_points_map", {})
    original_points_info, original_points_in_image = _get_original_points_context(image_path, image_points_map)
    item["original_points_info"] = original_points_info
    item["original_points_in_image"] = original_points_in_image

    question_field = str(options.get("query_field", "enhanced_query")).strip() or "enhanced_query"
    image_name = unicodedata.normalize("NFC", item["image_filename"])
    image_stem = image_name.replace("/", "_").replace("\\", "_")
    visualizations_dir = str(
        options.get("visualizations_dir", "visualizations/molmo2_guidance_dualquery_refpoint_hybrid_gemini_judge")
    ).strip() or "visualizations/molmo2_guidance_dualquery_refpoint_hybrid_gemini_judge"
    category_dir = unicodedata.normalize("NFC", os.path.join(visualizations_dir, "predmask", item.get("category", "")))
    os.makedirs(category_dir, exist_ok=True)
    debug_meta_path = unicodedata.normalize("NFC", os.path.join(category_dir, f"{image_stem}_justify_meta.json"))

    raw_user_input = item["user_input"]
    api_key = str(options.get("api_key") or os.getenv("API_KEY", "")).strip()
    base_url = str(options.get("base_url") or os.getenv("API_BASE_URL", "")).strip()
    planner_model_name = str(
        options.get("enhance_model")
        or options.get("planner_model_name")
        or os.getenv("API_MODEL_NAME")
        or os.getenv("SA2VA_PLANNER_MODEL", "gemini-3.5-flash")
    ).strip()
    rewrite_model_name = str(
        options.get("rewrite_model")
        or os.getenv("API_REWRITE_MODEL_NAME")
        or "gemini-3.5-flash"
    ).strip()
    resolved_model_name = MOLMO2_MODEL_ALIASES.get(str(model_name or "").strip(), str(model_name or "").strip()) or "allenai/Molmo2-4B"
    max_tokens = int(options.get("max_tokens", 256))
    model_root = str(options.get("model_root", "")).strip()

    with Image.open(image_path) as source_image:
        image_w, image_h = source_image.size

    state = {
        "item": item,
        "runtime_options": options,
        "image_path": image_path,
        "image_filename": item["image_filename"],
        "category": item.get("category", ""),
        "question_field": question_field,
        "image_stem": image_stem,
        "category_dir": category_dir,
        "debug_meta_path": debug_meta_path,
        "raw_user_input": raw_user_input,
        "api_key": api_key,
        "base_url": base_url,
        "planner_model_name": planner_model_name,
        "rewrite_model_name": rewrite_model_name,
        "resolved_model_name": resolved_model_name,
        "max_tokens": max_tokens,
        "model_root": model_root,
        "image_points_map": image_points_map,
        "original_points_info": original_points_info,
        "original_points_in_image": original_points_in_image,
        "image_w": image_w,
        "image_h": image_h,
        "enhanced_user_input": "",
        "helper_guidance": {},
        "local_route": "",
        "prompt_text": "",
        "molmo2_response": "",
        "candidate_points": [],
        "local_box_artifacts": {
            "input_visualization_path": "",
            "point_visualization_path": "",
        },
        "local_point_only_path": "",
        "judge_point_only_path": "",
        "judge_prompt": "",
        "judge_raw": "",
        "judge_result": None,
        "strict_accept": False,
        "fallback_prompt": "",
        "fallback_raw": "",
        "fallback_points": [],
        "fallback_viz_path": "",
        "fallback_judge_raw": "",
        "fallback_judge_result": None,
        "fallback_strict_accept": False,
        "final_points": None,
        "final_decision": "",
        "pipeline_error": "",
    }
    state["debug_meta"] = _build_initial_debug_meta(state)
    return _restore_stage_cache(state)


def run_molmo2_gemini_rewrite_stage(state: Dict[str, Any]) -> Dict[str, Any]:
    if not _begin_stage(state, "rewrite"):
        return state

    try:
        enhanced_user_input = call_transform_gemini(
            image_path=state["image_path"],
            object_name=state["raw_user_input"],
            model_name=state["rewrite_model_name"],
            category=state["category"],
            item_ctx=state["item"],
            runtime_options={
                "image_points_map": state["image_points_map"],
                "api_key": state["api_key"],
                "base_url": state["base_url"],
                "enhance_model": state["rewrite_model_name"],
                "visualizations_dir": state["runtime_options"].get("visualizations_dir", ""),
                "rewrite_prompt_mode": REWRITE_PROMPT_MODE,
            },
        ).strip()
    except Exception:
        traceback_text = traceback.format_exc()
        logger.exception(
            f"Gemini rewrite failed for image={state['image_filename']}, category={state['category']}"
        )
        return _mark_pipeline_error(
            state,
            "rewrite_error",
            "Gemini rewrite failed.",
            traceback_text,
        )
    if not enhanced_user_input:
        return _mark_pipeline_error(state, "rewrite_empty", "Gemini rewrite returned empty text.")

    state["enhanced_user_input"] = enhanced_user_input
    state["item"]["enhanced_query"] = enhanced_user_input
    state["item"][state["question_field"]] = enhanced_user_input
    state["debug_meta"]["enhanced_user_input"] = enhanced_user_input
    state["debug_meta"]["rewrite_source_pipeline"] = REWRITE_SOURCE_PIPELINE_NAME
    state["debug_meta"]["rewrite_prompt_mode"] = REWRITE_PROMPT_MODE
    state["debug_meta"]["rewrite_model_name"] = state["rewrite_model_name"]
    state["debug_meta"]["rewrite_overlay_image_path"] = str(state["item"].get("rewrite_overlay_path") or "")
    state["debug_meta"]["rewrite_overlay_meta_path"] = str(state["item"].get("rewrite_overlay_meta_path") or "")
    return _save_stage_cache(state, "rewrite")


def run_molmo2_gemini_box_grounding_stage(state: Dict[str, Any]) -> Dict[str, Any]:
    if not _begin_stage(state, "gemini_box"):
        return state

    with Image.open(state["image_path"]) as source_image:
        image = source_image.convert("RGB")
    try:
        helper_guidance, helper_network_error = _run_internal_gemini_box_grounding(
            state["image_path"],
            image,
            state["image_w"],
            state["image_h"],
            state["item"],
            state["question_field"],
            state["planner_model_name"],
            state["api_key"],
            state["base_url"],
            state["category_dir"],
            state["image_stem"],
        )
    finally:
        image.close()

    if helper_network_error:
        return _mark_pipeline_error(
            state,
            "helper_network_error",
            "Gemini helper box grounding hit a network error.",
        )

    state["helper_guidance"] = helper_guidance
    debug_meta = state["debug_meta"]
    debug_meta["helper_visualization_path"] = helper_guidance.get("box_visualization_path", "")
    debug_meta["guidance_points_xy"] = helper_guidance.get("guidance_points_xy", []) or []
    debug_meta["matched_box_centers_xy"] = helper_guidance.get("matched_box_centers_xy", []) or []
    debug_meta["pixel_boxes_xyxy"] = helper_guidance.get("pixel_boxes_xyxy", []) or []
    debug_meta["helper_box_prompt"] = helper_guidance.get("helper_box_prompt", "")
    debug_meta["helper_box_raw_response"] = helper_guidance.get("helper_box_raw_response", "")
    debug_meta["helper_box_parsed_response"] = helper_guidance.get("helper_box_parsed_response")
    debug_meta["helper_box_reasoning_trace"] = helper_guidance.get("helper_box_reasoning_trace", {})
    debug_meta["helper_box_target_resolution"] = helper_guidance.get("helper_box_target_resolution", {})
    debug_meta["helper_box_reason"] = helper_guidance.get("helper_box_reason", "")
    debug_meta["normalized_boxes_yxyx_0_1000"] = helper_guidance.get("normalized_boxes_yxyx_0_1000", []) or []
    debug_meta["normalized_points_yx_0_1000"] = helper_guidance.get("normalized_points_yx_0_1000", []) or []
    debug_meta["helper_box_visualization_path"] = helper_guidance.get("box_visualization_path", "")
    debug_meta["helper_point_visualization_path"] = helper_guidance.get("point_visualization_path", "")
    return _save_stage_cache(state, "gemini_box")


def run_molmo2_gemini_local_stage(state: Dict[str, Any]) -> Dict[str, Any]:
    if not _begin_stage(state, "molmo2_local"):
        return state

    with Image.open(state["image_path"]) as source_image:
        image = source_image.convert("RGB")

    local_input_image: Optional[Image.Image] = None
    judge_viz: Optional[Image.Image] = None
    try:
        if state["original_points_in_image"]:
            state["local_route"] = "refpoint_nolabel_box"
            state["prompt_text"] = _build_refpoint_box_explained_nolabel_prompt(
                state["raw_user_input"],
                state["enhanced_user_input"],
                state["question_field"],
                state["helper_guidance"].get("pixel_boxes_xyxy", []) or [],
                state["category"],
                state["helper_guidance"].get("guidance_points_xy", []) or [],
            )
            local_input_image = _build_nolabel_box_overlay_image(
                state["image_path"],
                state["helper_guidance"].get("pixel_boxes_xyxy", []) or [],
            )
        else:
            state["local_route"] = "dualquery_prompt"
            state["prompt_text"] = _build_dualquery_guidance_prompt(
                state["raw_user_input"],
                state["enhanced_user_input"],
                state["question_field"],
                state["helper_guidance"].get("guidance_points_xy", []) or [],
                state["category"],
            )
            local_input_image = image.copy()

        try:
            state["molmo2_response"] = _run_molmo2_item(
                state["image_path"],
                state["prompt_text"],
                state["resolved_model_name"],
                state["model_root"],
                state["max_tokens"],
                image_override=local_input_image,
            )
        except Exception as error:
            traceback_text = traceback.format_exc()
            logger.exception(
                f"Molmo2 local stage failed for image={state['image_filename']}, "
                f"model={state['resolved_model_name']}, model_root={state['model_root'] or '<huggingface-auto-download>'}"
            )
            return _mark_pipeline_error(
                state,
                "local_molmo_error",
                f"Molmo2 local stage failed: {error}",
                traceback_text,
            )

        candidate_points = _extract_molmo2_points(state["molmo2_response"], state["image_w"], state["image_h"])
        if state["category"] != "counting" and len(candidate_points) > 1 and not _query_requests_multiple(state["raw_user_input"], state["enhanced_user_input"]):
            candidate_points = candidate_points[:1]
        state["candidate_points"] = candidate_points

        if state["local_route"] == "refpoint_nolabel_box":
            state["local_box_artifacts"] = _save_hybrid_box_debug_artifacts(
                local_input_image,
                candidate_points,
                state["category_dir"],
                state["image_stem"],
            )

        state["local_point_only_path"] = _save_points_visualization(
            image,
            candidate_points,
            state["original_points_in_image"],
            state["category_dir"],
            f"{state['image_stem']}_points_only.jpg",
        )
        judge_viz = _build_judge_visualization(image, candidate_points, state["original_points_in_image"])
        state["judge_point_only_path"] = unicodedata.normalize("NFC", os.path.join(state["category_dir"], f"{state['image_stem']}_judge_points_only.jpg"))
        judge_viz.save(state["judge_point_only_path"])

        debug_meta = state["debug_meta"]
        debug_meta["enhanced_user_input"] = state["enhanced_user_input"]
        debug_meta["local_route"] = state["local_route"]
        debug_meta["local_prompt_text"] = state["prompt_text"]
        debug_meta["local_input_image_path"] = state["local_box_artifacts"]["input_visualization_path"]
        debug_meta["local_box_point_visualization_path"] = state["local_box_artifacts"]["point_visualization_path"]
        debug_meta["local_point_only_image_path"] = state["local_point_only_path"]
        debug_meta["point_only_image_path"] = state["judge_point_only_path"]
        debug_meta["local_response_text"] = state["molmo2_response"]
        debug_meta["local_points"] = candidate_points

        if not state["original_points_in_image"]:
            state["final_points"] = _points_to_float_payload(candidate_points)
            state["final_decision"] = "skip_gemini_keep_molmo2_no_reference_point"
            debug_meta["judge_skipped"] = True
            debug_meta["judge_skip_reason"] = "no_reference_points"
            debug_meta["judge_reason"] = "Skipped Gemini judge because no reference points were provided for this sample."
            debug_meta["final_decision"] = state["final_decision"]
            debug_meta["final_response"] = state["final_points"]
    finally:
        if judge_viz is not None:
            judge_viz.close()
        if local_input_image is not None:
            local_input_image.close()
        image.close()

    return _save_stage_cache(state, "molmo2_local")


def run_molmo2_gemini_judge_stage(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("pipeline_error") or not state["original_points_in_image"]:
        return state
    if not _begin_stage(state, "gemini_judge"):
        return state
    if state.get("final_points") is not None:
        return state

    state["judge_prompt"] = _build_molmo2_judge_prompt(
        state["raw_user_input"],
        state["enhanced_user_input"],
        state["question_field"],
    )
    hard_reject_reason = _get_hard_reject_reason(state["candidate_points"], state["image_w"], state["image_h"])
    if hard_reject_reason:
        state["judge_raw"] = ""
        state["judge_result"] = _make_rejected_judge_result(hard_reject_reason)
    else:
        try:
            state["judge_raw"] = _call_gemini(
                state["api_key"],
                state["base_url"],
                state["planner_model_name"],
                state["judge_prompt"],
                [state["judge_point_only_path"]],
            )
            state["judge_result"] = _parse_json_payload(state["judge_raw"])
        except Exception as error:
            traceback_text = traceback.format_exc()
            if _is_network_error(error):
                logger.exception(f"Gemini judge network error for image={state['image_filename']}")
                return _mark_pipeline_error(
                    state,
                    "judge_network_error",
                    "Gemini judge hit a network error.",
                    traceback_text,
                )
            logger.exception(f"Gemini judge failed for image={state['image_filename']}")
            return _mark_pipeline_error(
                state,
                "judge_error",
                f"Gemini judge failed: {error}",
                traceback_text,
            )
        if not isinstance(state["judge_result"], dict):
            state["judge_result"] = None

    state["strict_accept"] = _get_strict_judge_accept(state["judge_result"], hard_reject_reason)
    debug_meta = state["debug_meta"]
    debug_meta["judge_skipped"] = False
    debug_meta["judge_prompt"] = state["judge_prompt"]
    debug_meta["judge_hard_reject_reason"] = hard_reject_reason
    debug_meta["judge_raw"] = state["judge_raw"]
    debug_meta["judge_parsed"] = state["judge_result"]
    debug_meta["judge_model_accept"] = state["judge_result"].get("accept") if isinstance(state["judge_result"], dict) else None
    debug_meta["judge_accept"] = state["judge_result"].get("accept") if isinstance(state["judge_result"], dict) else None
    debug_meta["judge_strict_accept"] = state["strict_accept"]
    debug_meta["judge_reason"] = state["judge_result"].get("reason", "") if isinstance(state["judge_result"], dict) else ""
    debug_meta["judge_verification_trace"] = state["judge_result"].get("verification_trace", {}) if isinstance(state["judge_result"], dict) else {}

    if state["strict_accept"]:
        state["final_points"] = _points_to_float_payload(state["candidate_points"])
        state["final_decision"] = "judge_accept_keep_molmo2"
        debug_meta["final_decision"] = state["final_decision"]
        debug_meta["final_response"] = state["final_points"]
    return _save_stage_cache(state, "gemini_judge")


def run_molmo2_gemini_fallback_stage(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("pipeline_error") or not state["original_points_in_image"]:
        return state
    if not _begin_stage(state, "gemini_fallback"):
        return state
    if state.get("final_points") is not None:
        return state

    state["fallback_prompt"] = _build_fallback_grounding_prompt(
        state["raw_user_input"],
        state["enhanced_user_input"],
        state["question_field"],
    )
    try:
        state["fallback_raw"] = _call_gemini(
            state["api_key"],
            state["base_url"],
            state["planner_model_name"],
            state["fallback_prompt"],
            [state["image_path"]],
        )
    except Exception as error:
        traceback_text = traceback.format_exc()
        if _is_network_error(error):
            logger.exception(f"Gemini fallback grounding network error for image={state['image_filename']}")
            return _mark_pipeline_error(
                state,
                "fallback_network_error",
                "Gemini fallback grounding hit a network error.",
                traceback_text,
            )
        logger.exception(f"Gemini fallback grounding failed for image={state['image_filename']}")
        return _mark_pipeline_error(
            state,
            "fallback_error",
            f"Gemini fallback grounding failed: {error}",
            traceback_text,
        )

    fallback_parse_category = "counting" if _query_requests_multiple(state["raw_user_input"], state["enhanced_user_input"]) else ""
    state["fallback_points"] = _extract_gemini_point_response(
        state["fallback_raw"],
        state["image_w"],
        state["image_h"],
        fallback_parse_category,
    )
    debug_meta = state["debug_meta"]
    debug_meta["gemini_fallback_prompt"] = state["fallback_prompt"]
    debug_meta["gemini_fallback_raw"] = state["fallback_raw"]
    debug_meta["gemini_fallback_points"] = state["fallback_points"]
    debug_meta["gemini_fallback_response_text"] = _normalize_final_text_no_markdown(state["fallback_points"])

    if not state["fallback_points"]:
        state["final_points"] = _points_to_float_payload(state["candidate_points"])
        state["final_decision"] = "fallback_empty_keep_molmo2"
        debug_meta["final_decision"] = state["final_decision"]
        debug_meta["final_response"] = state["final_points"]
        return _save_stage_cache(state, "gemini_fallback")

    with Image.open(state["image_path"]) as source_image:
        image = source_image.convert("RGB")
    try:
        state["fallback_viz_path"] = _save_points_visualization(
            image,
            state["fallback_points"],
            state["original_points_in_image"],
            state["category_dir"],
            f"{state['image_stem']}_gemini_first_points_only.jpg",
        )
    finally:
        image.close()
    debug_meta["gemini_first_point_only_image_path"] = state["fallback_viz_path"]
    return _save_stage_cache(state, "gemini_fallback")


def run_molmo2_gemini_fallback_judge_stage(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("pipeline_error") or not state["fallback_points"]:
        return state
    if not _begin_stage(state, "fallback_judge"):
        return state
    if state.get("final_points") is not None:
        return state

    fallback_hard_reject_reason = _get_hard_reject_reason(state["fallback_points"], state["image_w"], state["image_h"])
    if fallback_hard_reject_reason:
        state["fallback_judge_raw"] = ""
        state["fallback_judge_result"] = _make_rejected_judge_result(fallback_hard_reject_reason)
    else:
        try:
            state["fallback_judge_raw"] = _call_gemini(
                state["api_key"],
                state["base_url"],
                state["planner_model_name"],
                state["judge_prompt"],
                [state["fallback_viz_path"]],
            )
            state["fallback_judge_result"] = _parse_json_payload(state["fallback_judge_raw"])
        except Exception as error:
            traceback_text = traceback.format_exc()
            if _is_network_error(error):
                logger.exception(f"Gemini fallback judge network error for image={state['image_filename']}")
                return _mark_pipeline_error(
                    state,
                    "fallback_judge_network_error",
                    "Gemini fallback judge hit a network error.",
                    traceback_text,
                )
            logger.exception(f"Gemini fallback judge failed for image={state['image_filename']}")
            return _mark_pipeline_error(
                state,
                "fallback_judge_error",
                f"Gemini fallback judge failed: {error}",
                traceback_text,
            )
        if not isinstance(state["fallback_judge_result"], dict):
            state["fallback_judge_result"] = None

    state["fallback_strict_accept"] = _get_strict_judge_accept(state["fallback_judge_result"], fallback_hard_reject_reason)
    debug_meta = state["debug_meta"]
    debug_meta["gemini_fallback_final_judge_raw"] = state["fallback_judge_raw"]
    debug_meta["gemini_fallback_final_judge_parsed"] = state["fallback_judge_result"]
    debug_meta["gemini_fallback_final_judge_verification_trace"] = (
        state["fallback_judge_result"].get("verification_trace", {}) if isinstance(state["fallback_judge_result"], dict) else {}
    )
    debug_meta["gemini_fallback_accept"] = state["fallback_strict_accept"]

    if state["fallback_strict_accept"]:
        state["final_points"] = _points_to_float_payload(state["fallback_points"])
        state["final_decision"] = "fallback_accept_use_gemini"
    else:
        state["final_points"] = _points_to_float_payload(state["candidate_points"])
        state["final_decision"] = "fallback_reject_keep_molmo2"
    debug_meta["final_decision"] = state["final_decision"]
    debug_meta["final_response"] = state["final_points"]
    return _save_stage_cache(state, "fallback_judge")


def finalize_molmo2_gemini_pipeline_state(state: Dict[str, Any]) -> Dict[str, Any]:
    if state.get("final_points") is None:
        if state.get("candidate_points"):
            state["final_points"] = _points_to_float_payload(state["candidate_points"])
            state["final_decision"] = state.get("final_decision") or "pipeline_incomplete_keep_molmo2"
        else:
            state["final_points"] = []
            state["final_decision"] = state.get("final_decision") or "pipeline_error_no_points"

    debug_meta = state.get("debug_meta", {})
    if isinstance(debug_meta, dict):
        debug_meta["enhanced_user_input"] = state.get("enhanced_user_input", "")
        debug_meta["final_decision"] = state.get("final_decision", "")
        debug_meta["final_response"] = state.get("final_points", [])
        if state.get("pipeline_error"):
            debug_meta["pipeline_error"] = state["pipeline_error"]
            debug_meta["pipeline_error_reason"] = debug_meta.get("pipeline_error_reason") or state["pipeline_error"]
            debug_meta["pipeline_error_traceback"] = debug_meta.get("pipeline_error_traceback", "")
    return _save_stage_cache(state, "finalize")


def run_molmo2_gemini_pipeline_state(state: Dict[str, Any]) -> Dict[str, Any]:
    state = run_molmo2_gemini_rewrite_stage(state)
    state = run_molmo2_gemini_box_grounding_stage(state)
    state = run_molmo2_gemini_local_stage(state)
    state = run_molmo2_gemini_judge_stage(state)
    state = run_molmo2_gemini_fallback_stage(state)
    state = run_molmo2_gemini_fallback_judge_stage(state)
    return finalize_molmo2_gemini_pipeline_state(state)


def call_molmo2_guidance_dualquery_refpoint_hybrid_gemini_judge(
    image_path: str,
    object_name: str,
    model_name: str = "allenai/Molmo2-4B",
    category: Optional[str] = None,
    item_ctx: Optional[Dict[str, Any]] = None,
    runtime_options: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, List[float]]]:
    """Rewrite first, then run the refpoint-hybrid Molmo2 route, and only judge reference-point samples."""
    del object_name
    state = build_molmo2_gemini_pipeline_state(
        image_path=image_path,
        model_name=model_name,
        category=category,
        item_ctx=item_ctx,
        runtime_options=runtime_options,
    )
    state = run_molmo2_gemini_pipeline_state(state)
    return state.get("final_points") or []


"""
Example command

Only one Molmo2 + Gemini hybrid pipeline is exposed at the moment:

- `molmo2_guidance_dualquery_refpoint_hybrid_gemini_judge`
  - Flow: `raw user_input -> transform_gemini rewrite -> Gemini box/center grounding -> refpoint-hybrid local Molmo2 -> Gemini judge/fallback only for reference-point samples`

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python model_evaluator.py \
  --type molmo2_guidance_dualquery_refpoint_hybrid_gemini_judge \
  --model 1-2 \
  --model_root /path/to/models \
  --query_field enhanced_query \
  --enhance_model gemini-3.5-flash \
  --rewrite_model gemini-3.5-flash \
  --max_tokens 256 \
  --suffix molmo2_guidance_dualquery_refpoint_hybrid_gemini_judge_exp \
  --start 0 \
  --end -1
```

Argument descriptions and supported values

- `--type`
  - Purpose: Selects the pipeline type.
  - Currently supported value: `molmo2_guidance_dualquery_refpoint_hybrid_gemini_judge`
  - Meaning: After query rewriting, Gemini first produces the box/center helper, then the pipeline routes to the local refpoint-hybrid Molmo2 path, and only reference-point samples continue to the Gemini judge/fallback stage.

- `--model`
  - Purpose: Main model name.
  - Supported values:
    - `1-1`: maps to `allenai/Molmo2-8B`
    - `1-2`: maps to `allenai/Molmo2-4B`
    - You can also pass the full HuggingFace model name directly, for example `allenai/Molmo2-4B`

- `--model_root`
  - Purpose: Root directory for local Molmo2 weights.
  - Optional values:
    - Empty string: load `--model` directly as a HuggingFace repo id; on the first run `transformers` will download and cache it automatically
    - Local directory path, for example `/path/to/models`
  - When a directory is provided, the code resolves it as `<model_root>/<huggingface_repo_id>`, for example `<model_root>/allenai/Molmo2-4B`
  - You can also pass the concrete model directory itself, for example `/path/to/models/allenai/Molmo2-4B`

- `--query_field`
  - Purpose: Chooses which field in the data JSON receives the rewritten query.
  - Common values:
    - `enhanced_query`: recommended; stores the enhanced query separately
    - `user_input`: not recommended; overwrites the original query
  - The current pipeline always generates `enhanced_query` first and mirrors the rewritten text into this field as well.

- `--enhance_model`
  - Purpose: Gemini model name.
  - In the current pipeline it is used for:
    - box/center helper grounding
    - judge
    - fallback grounding
  - Common values:
    - `gemini-3.5-flash`: current default
    - Any other Gemini model name supported by your API endpoint

- `--rewrite_model`
  - Purpose: Gemini model name used only for the transform_gemini_twolines-style rewrite stage.
  - Common values:
    - `gemini-3.5-flash`: current default
    - Any other Gemini model name supported by your API endpoint

- `--max_tokens`
  - Purpose: Maximum number of new tokens generated by Molmo2.
  - Common values:
    - `256`: recommended default
    - You can also use smaller or larger integers such as `128` or `512`

- Concurrency notes
  - This pipeline does not require a separate `--workers` argument
  - It automatically chooses the worker process count as `min(number of visible GPUs, number of pending samples)`

- `--suffix`
  - Purpose: Suffix for the result directory and `res_*.json`, used to avoid overwriting existing experiment outputs.
  - Optional values: any string
  - Examples:
    - `molmo2_guidance_dualquery_refpoint_hybrid_gemini_judge_exp`
    - `molmo2_refpoint_hybrid_pb`

- `--start`
  - Purpose: Dataset index to start from.
  - Supported values: any non-negative integer
  - Examples: `0`, `100`

- `--end`
  - Purpose: Dataset index to stop at.
  - Supported values:
    - `-1`: run until the end of the dataset
    - Any positive integer, for example `200`

- `--resume` / `--no-resume`
  - Purpose: Controls whether the run continues from existing results.
  - Optional values:
    - `--resume`: continue from existing results; this is the default behavior
    - `--no-resume`: ignore existing results and start from scratch

Environment variables

- `CUDA_VISIBLE_DEVICES`
  - Purpose: Selects which GPU Molmo2 should use.
  - Examples:
    - `CUDA_VISIBLE_DEVICES=0`
    - `CUDA_VISIBLE_DEVICES=1`

- `API_KEY`
  - Purpose: Gemini API key.

- `API_BASE_URL`
  - Purpose: Gemini API base URL.
  - This can be left empty for the official endpoint; for third-party compatible endpoints, set the corresponding URL.

- `API_MODEL_NAME`
  - Optional.
  - If `--enhance_model` is not passed explicitly, this can provide the default Gemini model name.

Notes

- `transform_gemini` is still kept in this file as the internal function `call_transform_gemini(...)`.
- Internally the rewrite stage follows `transform_gemini_twolines` with the original project's `legacy_reference_overlay` prompt plus the reference-point overlay/grid visualization rules.
- The current default is: `rewrite_model=gemini-3.5-flash`, `enhance_model=gemini-3.5-flash`.
- In the refpoint-hybrid version, Gemini helper box grounding is now executed on demand inside PointBench for each sample instead of relying on an externally precomputed `hosted_api_box_center` directory.
- If the Gemini helper does not produce a box for the current sample, the local prompt automatically falls back to the dualquery prompt branch instead of failing immediately because of the missing box.
"""
