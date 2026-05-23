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
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import torch
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
        if base_url:
            client_kwargs["http_options"] = types.HttpOptions(base_url=base_url)
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
    response = client.models.generate_content(model=model_name, contents=contents)
    return (response.text or "").strip()


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


def _build_rewrite_prompt(raw_input: str, category: str, original_points_info: str) -> Tuple[str, str]:
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


def _reference_marker_radius(image_w: int, image_h: int) -> int:
    return max(12, int(round(min(image_w, image_h) * 0.06)))


def _draw_reference_marker(draw: ImageDraw.ImageDraw, point: Tuple[float, float], radius: int) -> None:
    cx, cy = int(round(point[0])), int(round(point[1]))
    outline_radius = radius + max(3, radius // 5)
    draw.ellipse([cx - outline_radius, cy - outline_radius, cx + outline_radius, cy + outline_radius], fill=(255, 255, 255))
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(255, 0, 0))


def _build_reference_marker_image(image_path: str, original_points_in_image: List[List[float]]) -> Optional[Image.Image]:
    if not original_points_in_image:
        return None
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
    image_w, image_h = image.size
    radius = _reference_marker_radius(image_w, image_h)
    draw = ImageDraw.Draw(image)
    for point_x, point_y in original_points_in_image:
        clamped_point = (
            min(max(float(point_x), 0.0), max(image_w - 1, 0)),
            min(max(float(point_y), 0.0), max(image_h - 1, 0)),
        )
        _draw_reference_marker(draw, clamped_point, radius)
    return image


def _build_reference_marker_image_path(image_path: str, original_points_in_image: List[List[float]]) -> Optional[str]:
    reference_image = _build_reference_marker_image(image_path, original_points_in_image)
    if reference_image is None:
        return None
    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    reference_image.save(temp_path, format="JPEG")
    return temp_path


def call_transform_gemini(
    image_path: str,
    object_name: str,
    model_name: str = "",
    category: Optional[str] = None,
    item_ctx: Optional[Dict[str, Any]] = None,
    runtime_options: Optional[Dict[str, Any]] = None,
) -> str:
    """PointBench-native version of transform_gemini."""
    del object_name
    options = runtime_options or {}
    item = item_ctx if item_ctx is not None else {}
    item["image_path"] = image_path
    item["category"] = category or item.get("category", "")
    item["user_input"] = str(item.get("user_input") or "").strip()

    image_points_map = options.get("image_points_map", {})
    original_points_info, original_points_in_image = _get_original_points_context(image_path, image_points_map)
    item["original_points_info"] = original_points_info
    item["original_points_in_image"] = original_points_in_image

    api_key = str(options.get("api_key") or os.getenv("API_KEY", "")).strip()
    base_url = str(options.get("base_url") or os.getenv("API_BASE_URL", "")).strip()
    resolved_model_name = str(model_name or options.get("enhance_model") or os.getenv("API_MODEL_NAME") or os.getenv("SA2VA_PLANNER_MODEL", "gemini-3.1-pro-preview")).strip()

    temp_reference_path: Optional[str] = None
    try:
        if str(item.get("original_points_info", "")).strip():
            temp_reference_path = _build_reference_marker_image_path(image_path, item.get("original_points_in_image", []) or [])
        resolved_image_path = temp_reference_path or image_path
        system_prompt, user_prompt = _build_rewrite_prompt(
            item["user_input"],
            item.get("category", ""),
            item.get("original_points_info", ""),
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
            return {}, True
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
            resolved_path = str((root_path / resolved_name.split("/")[-1]).resolve())
            if not Path(resolved_path).exists():
                raise FileNotFoundError(
                    "Molmo2 local weights not found at "
                    f"{resolved_path}. Leave --model_root empty to load {resolved_name} "
                    "from HuggingFace, or place the model under "
                    f"{root_path}/{resolved_name.split('/')[-1]}."
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
    image_stem = os.path.splitext(image_name)[0]
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
        or os.getenv("SA2VA_PLANNER_MODEL", "gemini-3.1-pro-preview")
    ).strip()

    enhanced_user_input = call_transform_gemini(
        image_path=image_path,
        object_name=raw_user_input,
        model_name=planner_model_name,
        category=item.get("category", ""),
        item_ctx=item,
        runtime_options={
            "image_points_map": image_points_map,
            "api_key": api_key,
            "base_url": base_url,
            "enhance_model": planner_model_name,
        },
    ).strip()
    if not enhanced_user_input:
        return []

    item["enhanced_query"] = enhanced_user_input
    item[question_field] = enhanced_user_input

    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
        image_w, image_h = image.size

    resolved_model_name = MOLMO2_MODEL_ALIASES.get(str(model_name or "").strip(), str(model_name or "").strip()) or "allenai/Molmo2-4B"
    max_tokens = int(options.get("max_tokens", 256))
    model_root = str(options.get("model_root", "")).strip()

    # PointBench 这里把 hosted Gemini box-center helper 直接内嵌到同一条 pipeline，
    # 不再依赖外部预先跑好的 source_field 目录。
    helper_guidance, helper_network_error = _run_internal_gemini_box_grounding(
        image_path,
        image,
        image_w,
        image_h,
        item,
        question_field,
        planner_model_name,
        api_key,
        base_url,
        category_dir,
        image_stem,
    )
    if helper_network_error:
        return []

    # 这里沿用原仓库 hybrid 的路由规则：有 reference points 时优先走 no-label box 分支，
    # 但如果当前样本这轮 Gemini helper 没给出 box，prompt 会自动回退成 dualquery 版本。
    if item.get("original_points_in_image", []) or []:
        local_route = "refpoint_nolabel_box"
        prompt_text = _build_refpoint_box_explained_nolabel_prompt(
            raw_user_input,
            enhanced_user_input,
            question_field,
            helper_guidance.get("pixel_boxes_xyxy", []) or [],
            item.get("category", ""),
            helper_guidance.get("guidance_points_xy", []) or [],
        )
        local_input_image = _build_nolabel_box_overlay_image(
            image_path,
            helper_guidance.get("pixel_boxes_xyxy", []) or [],
        )
    else:
        local_route = "dualquery_prompt"
        prompt_text = _build_dualquery_guidance_prompt(
            raw_user_input,
            enhanced_user_input,
            question_field,
            helper_guidance.get("guidance_points_xy", []) or [],
            item.get("category", ""),
        )
        local_input_image = image.copy()

    try:
        molmo2_response = _run_molmo2_item(
            image_path,
            prompt_text,
            resolved_model_name,
            model_root,
            max_tokens,
            image_override=local_input_image,
        )
    except Exception:
        return []

    candidate_points = _extract_molmo2_points(molmo2_response, image_w, image_h)
    if item.get("category") != "counting" and len(candidate_points) > 1 and not _query_requests_multiple(raw_user_input, enhanced_user_input):
        candidate_points = candidate_points[:1]

    if local_route == "refpoint_nolabel_box":
        local_box_artifacts = _save_hybrid_box_debug_artifacts(
            local_input_image,
            candidate_points,
            category_dir,
            image_stem,
        )
    else:
        local_box_artifacts = {
            "input_visualization_path": "",
            "point_visualization_path": "",
        }
    local_point_only_path = _save_points_visualization(
        image,
        candidate_points,
        item.get("original_points_in_image", []) or [],
        category_dir,
        f"{image_stem}_points_only.jpg",
    )
    judge_viz = _build_judge_visualization(image, candidate_points, item.get("original_points_in_image", []) or [])
    judge_point_only_path = unicodedata.normalize("NFC", os.path.join(category_dir, f"{image_stem}_judge_points_only.jpg"))
    judge_viz.save(judge_point_only_path)

    debug_meta: Dict[str, Any] = {
        "pipeline": "molmo2_guidance_dualquery_refpoint_hybrid_gemini_judge",
        "image_path": image_path,
        "image_filename": item["image_filename"],
        "category": item.get("category", ""),
        "question_field": question_field,
        "raw_user_input": raw_user_input,
        "enhanced_user_input": enhanced_user_input,
        "local_route": local_route,
        "local_prompt_source_pipeline": "molmo2_guidance_dualquery_refpoint_hybrid",
        "judge_requires_reference_points": True,
        "molmo2_guidance_source_field": INTERNAL_HOSTED_GEMINI_BOX_SOURCE_NAME,
        "molmo2_with_box_source_field": INTERNAL_HOSTED_GEMINI_BOX_SOURCE_NAME,
        "helper_sidecar_path": "",
        "helper_visualization_path": helper_guidance.get("box_visualization_path", ""),
        "guidance_points_xy": helper_guidance.get("guidance_points_xy", []) or [],
        "matched_box_centers_xy": helper_guidance.get("matched_box_centers_xy", []) or [],
        "pixel_boxes_xyxy": helper_guidance.get("pixel_boxes_xyxy", []) or [],
        "helper_box_prompt_version": GEMINI_BOX_PROMPT_VERSION,
        "helper_box_prompt": helper_guidance.get("helper_box_prompt", ""),
        "helper_box_raw_response": helper_guidance.get("helper_box_raw_response", ""),
        "helper_box_parsed_response": helper_guidance.get("helper_box_parsed_response"),
        "helper_box_reasoning_trace": helper_guidance.get("helper_box_reasoning_trace", {}),
        "helper_box_target_resolution": helper_guidance.get("helper_box_target_resolution", {}),
        "helper_box_reason": helper_guidance.get("helper_box_reason", ""),
        "normalized_boxes_yxyx_0_1000": helper_guidance.get("normalized_boxes_yxyx_0_1000", []) or [],
        "normalized_points_yx_0_1000": helper_guidance.get("normalized_points_yx_0_1000", []) or [],
        "helper_box_visualization_path": helper_guidance.get("box_visualization_path", ""),
        "helper_point_visualization_path": helper_guidance.get("point_visualization_path", ""),
        "original_points_in_image": item.get("original_points_in_image", []) or [],
        "local_prompt_text": prompt_text,
        "local_input_image_path": local_box_artifacts["input_visualization_path"],
        "local_box_point_visualization_path": local_box_artifacts["point_visualization_path"],
        "local_point_only_image_path": local_point_only_path,
        "point_only_image_path": judge_point_only_path,
        "local_response_text": molmo2_response,
        "local_points": candidate_points,
    }

    if not (item.get("original_points_in_image", []) or []):
        final_points = _points_to_float_payload(candidate_points)
        debug_meta.update(
            {
                "judge_skipped": True,
                "judge_skip_reason": "no_reference_points",
                "judge_raw": "",
                "judge_parsed": None,
                "judge_model_accept": None,
                "judge_accept": None,
                "judge_strict_accept": None,
                "judge_reason": "Skipped Gemini judge because no reference points were provided for this sample.",
                "judge_verification_trace": {},
                "gemini_fallback_prompt": "",
                "gemini_fallback_raw": "",
                "gemini_fallback_points": [],
                "gemini_fallback_response_text": "",
                "gemini_first_point_only_image_path": "",
                "gemini_fallback_final_judge_raw": "",
                "gemini_fallback_final_judge_parsed": None,
                "gemini_fallback_final_judge_verification_trace": {},
                "final_decision": "skip_gemini_keep_molmo2_no_reference_point",
                "final_response": final_points,
            }
        )
        _save_debug_meta(debug_meta_path, debug_meta)
        return final_points

    judge_prompt = _build_molmo2_judge_prompt(raw_user_input, enhanced_user_input, question_field)
    hard_reject_reason = _get_hard_reject_reason(candidate_points, image_w, image_h)
    if hard_reject_reason:
        judge_raw = ""
        judge_result = _make_rejected_judge_result(hard_reject_reason)
    else:
        try:
            judge_raw = _call_gemini(api_key, base_url, planner_model_name, judge_prompt, [judge_point_only_path])
            judge_result = _parse_json_payload(judge_raw)
        except Exception as error:
            if _is_network_error(error):
                return []
            return []
        if not isinstance(judge_result, dict):
            judge_result = None

    strict_accept = _get_strict_judge_accept(judge_result, hard_reject_reason)
    debug_meta.update(
        {
            "judge_skipped": False,
            "judge_prompt": judge_prompt,
            "judge_hard_reject_reason": hard_reject_reason,
            "judge_raw": judge_raw,
            "judge_parsed": judge_result,
            "judge_model_accept": judge_result.get("accept") if isinstance(judge_result, dict) else None,
            "judge_accept": judge_result.get("accept") if isinstance(judge_result, dict) else None,
            "judge_strict_accept": strict_accept,
            "judge_reason": judge_result.get("reason", "") if isinstance(judge_result, dict) else "",
            "judge_verification_trace": judge_result.get("verification_trace", {}) if isinstance(judge_result, dict) else {},
        }
    )

    if strict_accept:
        final_points = _points_to_float_payload(candidate_points)
        debug_meta["final_decision"] = "judge_accept_keep_molmo2"
        debug_meta["final_response"] = final_points
        _save_debug_meta(debug_meta_path, debug_meta)
        return final_points

    fallback_prompt = _build_fallback_grounding_prompt(raw_user_input, enhanced_user_input, question_field)
    try:
        fallback_raw = _call_gemini(api_key, base_url, planner_model_name, fallback_prompt, [image_path])
    except Exception as error:
        if _is_network_error(error):
            return []
        return []

    fallback_parse_category = "counting" if _query_requests_multiple(raw_user_input, enhanced_user_input) else ""
    fallback_points = _extract_gemini_point_response(fallback_raw, image_w, image_h, fallback_parse_category)
    debug_meta["gemini_fallback_prompt"] = fallback_prompt
    debug_meta["gemini_fallback_raw"] = fallback_raw
    debug_meta["gemini_fallback_points"] = fallback_points
    debug_meta["gemini_fallback_response_text"] = _normalize_final_text_no_markdown(fallback_points)

    if not fallback_points:
        final_points = _points_to_float_payload(candidate_points)
        debug_meta["final_decision"] = "fallback_empty_keep_molmo2"
        debug_meta["final_response"] = final_points
        _save_debug_meta(debug_meta_path, debug_meta)
        return final_points

    fallback_viz_path = _save_points_visualization(
        image,
        fallback_points,
        item.get("original_points_in_image", []) or [],
        category_dir,
        f"{image_stem}_gemini_first_points_only.jpg",
    )
    debug_meta["gemini_first_point_only_image_path"] = fallback_viz_path

    fallback_hard_reject_reason = _get_hard_reject_reason(fallback_points, image_w, image_h)
    if fallback_hard_reject_reason:
        fallback_judge_raw = ""
        fallback_judge_result = _make_rejected_judge_result(fallback_hard_reject_reason)
    else:
        try:
            fallback_judge_raw = _call_gemini(api_key, base_url, planner_model_name, judge_prompt, [fallback_viz_path])
            fallback_judge_result = _parse_json_payload(fallback_judge_raw)
        except Exception as error:
            if _is_network_error(error):
                return []
            return []
        if not isinstance(fallback_judge_result, dict):
            fallback_judge_result = None

    fallback_strict_accept = _get_strict_judge_accept(fallback_judge_result, fallback_hard_reject_reason)
    debug_meta["gemini_fallback_final_judge_raw"] = fallback_judge_raw
    debug_meta["gemini_fallback_final_judge_parsed"] = fallback_judge_result
    debug_meta["gemini_fallback_final_judge_verification_trace"] = (
        fallback_judge_result.get("verification_trace", {}) if isinstance(fallback_judge_result, dict) else {}
    )
    debug_meta["gemini_fallback_accept"] = fallback_strict_accept

    if fallback_strict_accept:
        final_points = _points_to_float_payload(fallback_points)
        debug_meta["final_decision"] = "fallback_accept_use_gemini"
        debug_meta["final_response"] = final_points
        _save_debug_meta(debug_meta_path, debug_meta)
        return final_points

    final_points = _points_to_float_payload(candidate_points)
    debug_meta["final_decision"] = "fallback_reject_keep_molmo2"
    debug_meta["final_response"] = final_points
    _save_debug_meta(debug_meta_path, debug_meta)
    return final_points


"""
示例运行命令

当前对外只保留一条 Molmo2 + Gemini 融合 pipeline：

- `molmo2_guidance_dualquery_refpoint_hybrid_gemini_judge`
  - 流程：`raw user_input -> transform_gemini 改写 -> Gemini box/center grounding -> refpoint-hybrid local Molmo2 -> 仅参考点样本走 Gemini judge/fallback`

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python model_evaluator.py \
  --type molmo2_guidance_dualquery_refpoint_hybrid_gemini_judge \
  --model 1-2 \
  --model_root /path/to/models \
  --query_field enhanced_query \
  --enhance_model gemini-3.1-pro-preview \
  --max_tokens 256 \
  --suffix molmo2_guidance_dualquery_refpoint_hybrid_gemini_judge_exp \
  --start 0 \
  --end -1
```

参数说明与可选值

- `--type`
  - 作用：选择 pipeline 类型。
  - 当前支持值：`molmo2_guidance_dualquery_refpoint_hybrid_gemini_judge`
  - 含义：改写后先用 Gemini 生成 box/center helper，再走 refpoint-hybrid 本地 Molmo2 路由，只对 reference-point 样本继续 judge/fallback。

- `--model`
  - 作用：主模型名。
  - 支持值：
    - `1-1`：映射到 `allenai/Molmo2-8B`
    - `1-2`：映射到 `allenai/Molmo2-4B`
    - 也可直接写完整 HuggingFace 名字，例如 `allenai/Molmo2-4B`

- `--model_root`
  - 作用：Molmo2 本地权重根目录。
  - 可选值：
    - 空字符串：直接按 `--model` 作为 HuggingFace repo id 加载，首次运行时会由 `transformers` 自动下载并缓存
    - 本地目录路径：例如 `/path/to/models`
  - 当传目录时，代码会拼接成 `<model_root>/<model短名>`，例如 `Molmo2-4B`

- `--query_field`
  - 作用：把改写后的 query 回写到数据 JSON 的哪个字段。
  - 常用值：
    - `enhanced_query`：推荐，专门存增强后的 query
    - `user_input`：不推荐，会覆盖原始 query
  - 当前 pipeline 会固定先生成 `enhanced_query`，然后也镜像写入这个字段。

- `--enhance_model`
  - 作用：Gemini 模型名。
  - 当前 pipeline 中同时用于：
    - transform 改写
    - box/center helper grounding
    - judge
    - fallback grounding
  - 常用值：
    - `gemini-3.1-pro-preview`
    - 其他你 API 端实际支持的 Gemini 模型名

- `--max_tokens`
  - 作用：Molmo2 生成时的最大新 token 数。
  - 常用值：
    - `256`：默认推荐
    - 也可以改成更小或更大整数，例如 `128`、`512`

- 并发说明
  - 当前 pipeline 不需要单独传 `--workers`
  - 它会按 `min(可见 GPU 数, 待处理样本数)` 自动决定并发进程数

- `--suffix`
  - 作用：结果目录和 `res_*.json` 的后缀，避免覆盖实验结果。
  - 可选值：任意字符串
  - 例如：
    - `molmo2_guidance_dualquery_refpoint_hybrid_gemini_judge_exp`
    - `molmo2_refpoint_hybrid_pb`

- `--start`
  - 作用：从数据集哪个下标开始跑。
  - 支持值：任意非负整数
  - 例如：`0`、`100`

- `--end`
  - 作用：跑到哪个下标结束。
  - 支持值：
    - `-1`：一直跑到数据集末尾
    - 任意正整数：例如 `200`

- `--resume` / `--no-resume`
  - 作用：是否从已有结果继续跑。
  - 可选值：
    - `--resume`：继续已有结果，默认行为
    - `--no-resume`：忽略已有结果，从头开始

环境变量说明

- `CUDA_VISIBLE_DEVICES`
  - 作用：指定 Molmo2 使用哪张卡。
  - 例如：
    - `CUDA_VISIBLE_DEVICES=0`
    - `CUDA_VISIBLE_DEVICES=1`

- `API_KEY`
  - 作用：Gemini API key。

- `API_BASE_URL`
  - 作用：Gemini API base url。
  - 官方接口可为空，第三方兼容接口填对应地址。

- `API_MODEL_NAME`
  - 可选。
  - 若未显式传 `--enhance_model`，可从这里补默认 Gemini 模型名。

说明

- `transform_gemini` 仍然保留为本文件里的内部函数 `call_transform_gemini(...)`。
- 它内部固定使用原工程的 `legacy` rewrite prompt。
- refpoint-hybrid 版本里，Gemini helper box grounding 现在由 PointBench 在样本内即时执行，不再依赖外部预跑的 `hosted_api_box_center` 目录。
- 如果当前样本的 Gemini helper 没产出 box，local prompt 会自动退化成 dualquery prompt 分支，不会因为缺 box 直接报错。
"""
