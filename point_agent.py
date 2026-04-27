import ast
import colorsys
import io
import json
import os
import re
import threading
import unicodedata
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from box import Box
from google import genai
from google.genai import types
from loguru import logger

POINT_AGENT_MODEL_TYPES = {"point_agent"}

STRICT_JUDGE_FIELDS = [
    "rewrite_faithful",
    "point_count_correct",
    "all_points_clearly_inside_target",
    "spatial_relation_correct",
    "high_confidence",
]

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

_GLOBAL_LOCK = threading.Lock()
_GLOBAL_MODEL = None
_GLOBAL_TOKENIZER = None
_GLOBAL_PROCESSOR = None
_GEMINI_CLIENTS: Dict[Tuple[str, str], Any] = {}
_GEMINI_CLIENTS_LOCK = Lock()


def _parse_json_payload(text):
    """从自由文本中抽取 JSON/Python 字面量，统一给点位解析和 judge 结果解析使用。"""
    if not text:
        return None

    cleaned = str(text).strip().replace("```json", "```")
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()

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


def _normalize_final_text_no_markdown(value):
    """把中间输出统一转成不带 Markdown fence 的文本，方便后续再解析。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value).strip()


def _recover_gemini_point(norm_y, norm_x, orig_w, orig_h):
    """Gemini grounding prompt 约定输出 [y, x]，这里恢复为 PointBench 的像素 [x, y]。"""
    pixel_x = int((float(norm_x) / 1000.0) * orig_w)
    pixel_y = int((float(norm_y) / 1000.0) * orig_h)
    return [pixel_x, pixel_y]


def _normalize_gemini_generated_points(value, orig_w, orig_h, category_name):
    parsed = value
    if isinstance(value, str):
        parsed = _parse_json_payload(value)
        if parsed is None:
            coords = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", value)
            converted = []
            for y_str, x_str in coords:
                try:
                    converted.append(_recover_gemini_point(float(y_str), float(x_str), orig_w, orig_h))
                except Exception:
                    continue
            if category_name != "counting" and len(converted) > 1:
                converted = [converted[0]]
            return json.dumps(converted, ensure_ascii=False) if converted else "[]"

    if not isinstance(parsed, list):
        return "[]"

    converted_points = []
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
    return json.dumps(converted_points, ensure_ascii=False) if converted_points else "[]"


def _extract_gemini_point_response(text, orig_w, orig_h, category_name):
    parsed = _parse_json_payload(text)
    if parsed:
        normalized_text = _normalize_gemini_generated_points(parsed, orig_w, orig_h, category_name)
        normalized_value = _parse_json_payload(normalized_text)
        return normalized_value if isinstance(normalized_value, list) else []

    coords = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", text or "")
    raw_points = [[float(first), float(second)] for first, second in coords]
    normalized_text = _normalize_gemini_generated_points(raw_points, orig_w, orig_h, category_name)
    normalized_value = _parse_json_payload(normalized_text)
    return normalized_value if isinstance(normalized_value, list) else []


def _get_local_device(config):
    if not torch.cuda.is_available():
        return "cpu"

    worker_gpu_id = getattr(config.args, "worker_gpu_id", None)
    if worker_gpu_id is not None and str(worker_gpu_id) != "":
        device_index = int(worker_gpu_id)
    else:
        device_index = 0

    # point_agent 子进程通常只暴露一张卡；显式设置当前设备，避免 remote code 默认落到别的 cuda。
    torch.cuda.set_device(device_index)
    return f"cuda:{device_index}"


def _load_sa2va_model_bundle(path, device):
    """在 PointBench 进程内缓存 Sa2VA 权重，避免每张图重复加载大模型。"""
    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _GLOBAL_PROCESSOR

    with _GLOBAL_LOCK:
        if _GLOBAL_MODEL is None:
            from transformers import AutoModel, AutoProcessor, AutoTokenizer

            model = AutoModel.from_pretrained(
                path,
                torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
                low_cpu_mem_usage=True,
                use_flash_attn=device.startswith("cuda"),
                trust_remote_code=True,
            ).to(device).eval()

            tokenizer = None
            processor = None
            if "Intern" in path:
                tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
            else:
                processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, use_fast=False)

            _GLOBAL_MODEL = model
            _GLOBAL_TOKENIZER = tokenizer
            _GLOBAL_PROCESSOR = processor

    return _GLOBAL_MODEL, _GLOBAL_TOKENIZER, _GLOBAL_PROCESSOR


def _normalize_masks(masks: Union[torch.Tensor, np.ndarray, List]) -> List[np.ndarray]:
    mask_list = []
    if isinstance(masks, list):
        for mask in masks:
            if isinstance(mask, torch.Tensor):
                mask = mask.to("cpu").numpy()
            if mask.ndim > 2:
                mask = np.squeeze(mask)
            mask_list.append(mask > 0)
        return mask_list

    if isinstance(masks, torch.Tensor):
        masks = masks.to("cpu").numpy()

    if isinstance(masks, np.ndarray):
        if masks.ndim == 3:
            for index in range(masks.shape[0]):
                mask_list.append(masks[index] > 0)
        else:
            mask_list.append(masks > 0)
    return mask_list


def _draw_marker(draw, point, marker_color, use_cross=True):
    cx, cy = int(point[0]), int(point[1])
    outline_color = (255, 255, 255)
    r_outer = 6
    r_inner = 5
    cross_len = 4

    draw.ellipse([cx - r_outer - 1, cy - r_outer - 1, cx + r_outer + 1, cy + r_outer + 1], fill=outline_color)
    draw.ellipse([cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer], fill=marker_color)
    draw.ellipse([cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner], fill=outline_color)
    if use_cross:
        draw.line([cx - cross_len, cy, cx + cross_len, cy], fill=marker_color, width=2)
        draw.line([cx, cy - cross_len, cx, cy + cross_len], fill=marker_color, width=2)


def _generate_palette(num_regions):
    palette = []
    for index in range(num_regions):
        hue = index / num_regions
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        palette.append([int(channel * 255) for channel in rgb])
    return palette


def _append_region_center(region_mask_uint8, centers, width, height):
    """mask 点击点先取区域质心；质心落在外部时退回到距离变换的最内点。"""
    moments = cv2.moments(region_mask_uint8)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        if 0 <= cx < width and 0 <= cy < height and region_mask_uint8[cy, cx] > 0:
            centers.append([cx, cy])
            return

    dist_transform = cv2.distanceTransform(region_mask_uint8, cv2.DIST_L2, 5)
    _, _, _, max_loc = cv2.minMaxLoc(dist_transform)
    centers.append([max_loc[0], max_loc[1]])


def _get_overlay_with_centers(image, masks, alpha=0.5, area_ratio=0.25, original_points_in_image=None):
    """把 Sa2VA mask 转成 overlay 图和后续评测使用的中心点。"""
    if original_points_in_image is None:
        original_points_in_image = []

    image_np = np.array(image.convert("RGB"))
    mask_list = _normalize_masks(masks)
    if not mask_list:
        return Image.fromarray(image_np), []

    valid_region_counts = []
    total_regions = 0
    for layer_mask in mask_list:
        mask_uint8 = (layer_mask > 0).astype(np.uint8)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)
        if num_labels <= 1:
            valid_region_counts.append(0)
            continue

        component_areas = stats[1:, cv2.CC_STAT_AREA]
        max_component_area = np.max(component_areas)
        valid_count = int(np.sum(component_areas >= (area_ratio * max_component_area)))
        valid_region_counts.append(valid_count)
        total_regions += valid_count

    if total_regions == 0:
        return Image.fromarray(image_np), []

    palette = _generate_palette(total_regions)
    centers = []
    overlay = np.zeros_like(image_np, dtype=np.uint8)
    region_idx = 0
    height, width = image_np.shape[:2]

    for layer_mask in mask_list:
        mask_uint8 = (layer_mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)
        if num_labels <= 1:
            continue

        component_areas = stats[1:, cv2.CC_STAT_AREA]
        max_component_area = np.max(component_areas)
        for label_idx in range(1, num_labels):
            component_area = stats[label_idx, cv2.CC_STAT_AREA]
            if component_area < (area_ratio * max_component_area):
                continue

            region_mask = labels == label_idx
            if not np.any(region_mask):
                continue

            overlay[region_mask] = palette[region_idx % len(palette)]
            region_idx += 1
            _append_region_center(region_mask.astype(np.uint8), centers, width, height)

    combined_np = (image_np * (1 - alpha) + overlay * alpha).astype(np.uint8)
    combined_img = Image.fromarray(combined_np)
    draw = ImageDraw.Draw(combined_img)

    for point in original_points_in_image:
        _draw_marker(draw, point, marker_color=(0, 0, 255), use_cross=False)
    for center in centers:
        _draw_marker(draw, center, marker_color=(255, 0, 0), use_cross=True)

    return combined_img, centers


def _get_task_question(item, question_field="user_input"):
    field_name = (question_field or "user_input").strip() or "user_input"
    preferred_value = item.get(field_name)
    if isinstance(preferred_value, str) and preferred_value.strip():
        return preferred_value.strip()

    raw_value = item.get("user_input", "")
    if isinstance(raw_value, str):
        return raw_value.strip()
    return str(raw_value or "").strip()


def _append_context(question, extra_context):
    question_text = (question or "").strip()
    context_text = (extra_context or "").strip()
    if question_text and context_text:
        return f"{question_text}\n{context_text}"
    return question_text or context_text


def _build_task_prompt_for_field(config, item, question_field="user_input", image_size=None):
    category = item.get("category", "")
    if image_size is not None:
        width, height = image_size
    else:
        with Image.open(item.get("image_path")) as image:
            width, height = image.size

    original_points_info = item.get("original_points_info", "")
    question = _get_task_question(item, question_field)
    question_without_prefix = " ".join(question.split(" ")[2:]).strip()

    base_constraints = (
        f"- Image Resolution: {width}px x {height}px.\n"
        "- Coordinate System: ABSOLUTE PIXEL COORDINATES.\n"
        "- STRICT RULE: ONLY return points as [x, y].\n"
        "- Format: Return ONLY valid JSON list."
    )
    if category == "counting":
        default_system_prompt = (
            f"You are a precise image coordinate extractor. {original_points_info}\n"
            f"{base_constraints}\n"
            "Format: [{'point': [x, y]}, ...]"
        )
    else:
        default_system_prompt = (
            f"You are a precise image coordinate extractor. {original_points_info}\n"
            f"{base_constraints}\n"
            "IMPORTANT: Return EXACTLY ONE point.\n"
            "Format: [{'point': [x, y]}]"
        )

    prompt_registry = {
        "default": {"system_prompt": default_system_prompt, "user_prompt": question},
        "sa2va": {
            "system_prompt": "",
            "user_prompt": _append_context(question, original_points_info),
        },
        "gemini": {
            "system_prompt": "",
            "user_prompt": _append_context(question, original_points_info),
        },
    }

    selected_prompt = prompt_registry.get(config.model_info.prompt_strategy, prompt_registry["default"])
    return selected_prompt["system_prompt"], selected_prompt["user_prompt"]



def _get_gemini_client(api_key, base_url):
    cache_key = (api_key, base_url)
    with _GEMINI_CLIENTS_LOCK:
        client = _GEMINI_CLIENTS.get(cache_key)
        if client is not None:
            return client

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["http_options"] = types.HttpOptions(base_url=base_url)
        client = genai.Client(**client_kwargs)
        _GEMINI_CLIENTS[cache_key] = client
        return client


def _call_gemini_judge(local_resource, prompt_text, image_path_for_judge, logger):
    try:
        api_key, base_url, model_name = local_resource.get("planner_api_key"), local_resource.get("planner_base_url"), local_resource.get("planner_model_name")
        if not api_key or not model_name:
            return ""

        with open(image_path_for_judge, "rb") as image_file:
            image_bytes = image_file.read()

        client = _get_gemini_client(api_key, base_url)
        response = client.models.generate_content(
            model=model_name,
            contents=[
                prompt_text,
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            ],
        )
        return (response.text or "").strip()
    except Exception as error:
        logger.warning(f"Gemini justify call failed: {error}")
        return ""


def _call_gemini_grounding(local_resource, prompt_text, image_path_for_grounding, logger):
    try:
        api_key, base_url, model_name = local_resource.get("planner_api_key"), local_resource.get("planner_base_url"), local_resource.get("planner_model_name")
        if not api_key or not model_name:
            return ""

        with Image.open(image_path_for_grounding) as raw_image:
            image = raw_image.convert("RGB")
            image_buffer = io.BytesIO()
            file_extension = os.path.splitext(image_path_for_grounding)[1].lower()
            if file_extension == ".png":
                mime_type = "image/png"
                image_format = "PNG"
            elif file_extension in {".jpg", ".jpeg"}:
                mime_type = "image/jpeg"
                image_format = "JPEG"
            elif file_extension == ".webp":
                mime_type = "image/webp"
                image_format = "WEBP"
            elif file_extension == ".gif":
                mime_type = "image/gif"
                image_format = "GIF"
            else:
                mime_type = "image/jpeg"
                image_format = "JPEG"
            image.save(image_buffer, format=image_format)
            image_bytes = image_buffer.getvalue()

        client = _get_gemini_client(api_key, base_url)
        response = client.models.generate_content(
            model=model_name,
            contents=[
                prompt_text,
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
        )
        return (response.text or "").strip()
    except Exception as error:
        logger.warning(f"Gemini grounding fallback call failed. image_path: {image_path_for_grounding}, error: {error}")
        return ""


def _build_point_only_visualization(base_image, points, original_points):
    canvas = base_image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    for point in original_points:
        _draw_marker(draw, point, marker_color=(0, 0, 255), use_cross=False)
    for point in points:
        _draw_marker(draw, point, marker_color=(255, 0, 0), use_cross=True)
    return canvas


def _save_points_visualization(base_image, points, original_points, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    save_path = unicodedata.normalize("NFC", os.path.join(save_dir, filename))
    _build_point_only_visualization(base_image, points, original_points).save(save_path)
    return save_path


def _build_judge_prompt(category, system_prompt, raw_user_input, enhanced_user_input, question_field, user_question, sa2va_response, image_w, image_h):
        """构建与 norefine v1 一致的 judge prompt。"""
        return f'''You are a strict and conservative verifier for point grounding results.

        Your goal is to maximize precision:
        - Accept only when the red point result is clearly correct.
        - If there is any uncertainty, ambiguity, missing target, extra target, boundary issue, or rewrite mismatch, reject.
        - It is better to reject a correct result than to accept an incorrect result.

        You will receive one visualization image:
        - this is the original image without segmentation mask overlay;
        - it may be a contact sheet containing a full-image view and zoomed crops around red points;
        - red points are the post-processed point results from Sa2VA;
        - blue points, if present, are reference points.
        - the blue reference points are context points provided by the task, especially important for steerable / relational grounding;
        - blue reference points are not prediction targets unless the textual instruction explicitly requires the same location.

        Your task is only to judge whether the red points correctly satisfy the query.
        You must treat the original user input as the primary ground-truth instruction.
        First check whether the enhanced text input is faithful to the original user input.
        If the enhanced text input adds an unsupported object, attribute, count, location, or spatial relation,
        ignore the enhanced text and judge only by the original user input.
        If the candidate points only satisfy the enhanced text but not the original user input, reject.
        After that, check whether the image content supports the original user input and, if valid, the enhanced text input as well.
        Finally judge whether the predicted red points match the trustworthy textual instruction and the actual image content.
        If blue reference points are present, you must also use them to understand relative spatial constraints such as left/right/near/behind/closest/to the reference point.

        Strict accept criteria:
        1. The candidate must satisfy the original user input, not merely a plausible rewrite.
        2. Infer from the original user input whether the task asks for one target or multiple targets.
        3. If the instruction asks for one specific target, there must be exactly one red point.
        4. If the instruction asks for all/every/both/multiple targets, all clearly visible target instances must be pointed to, with no obvious missing, duplicate, or extra red points.
        5. If you cannot confidently determine whether the number of red points matches the instruction and image, set "point_count_correct" to false.
        6. The center of every red marker must be clearly inside the visible target object/body.
        7. Reject if a red point is only near the target, on the boundary, on background, on shadow, on an adjacent object, or on an ambiguous part.
        8. For small targets such as logos, letters, lights, keys, dials, handles, or tips, the marker center must be on that exact part, not just on the larger object.
        9. For spatial or steerable tasks, verify the exact relation to the blue reference point or referenced object; reject if the relation is only approximately satisfied.
        10. For direction, future-position, or empty-space tasks, the marker center must clearly match the requested location/region, not just a plausible nearby object.
        11. Reject if you cannot confidently verify the exact point location from the image.

        Rules:
        - Return JSON only.
        - Use "accept": true only for high-confidence correct results.
        - Use "accept": false for uncertain, partially correct, near-miss, or ambiguous results.
        - For fields that are not applicable, set them to true only if they do not block correctness.
        - The final "accept" must be true only when all verification fields are true.
        - Do not generate new points.

        Original user input: {raw_user_input}
        Enhanced text input: {enhanced_user_input}
        Final text used by Sa2VA: {user_question}
        Sa2VA candidate points: {sa2va_response}
        Image resolution: {image_w} x {image_h}

        Return exactly this JSON schema:
        {{
            "rewrite_faithful": true or false,
            "point_count_correct": true or false,
            "all_points_clearly_inside_target": true or false,
            "spatial_relation_correct": true or false,
            "high_confidence": true or false,
            "accept": true or false,
            "reason": "brief reason"
        }}'''


def _normalize_query_text(text):
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _extract_direct_target_phrase(query_text):
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


def _has_explicit_multi_target(query_text):
    if re.search(r"\bpoint to\s+(?:counting\s+)?(?:all|every|each|both)\b", query_text):
        return True
    if re.search(rf"\bpoint to\s+(?:the\s+)?(?:first\s+)?(?:{MULTI_COUNT_WORD_PATTERN})\b", query_text):
        return True
    if re.search(r"\bpoint to\s+(?:the\s+)?(?:first\s+)?(?:[2-9]|10)\b(?!-)", query_text):
        return True
    return bool(re.search(r"\bpoint to\s+(?:the\s+)?[a-z-]+s\s+of\s+all\b", query_text))


def _direct_target_looks_plural(target_phrase):
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


def _query_requests_multiple(raw_user_input, enhanced_user_input):
    """
    线上没有 category/count 真值，只从文本估计是否多目标。
    这个判断只影响是否保留多个 mask 连通域，不把它当最终数量答案。
    """
    raw_query_text = _normalize_query_text(raw_user_input)
    enhanced_query_text = _normalize_query_text(enhanced_user_input)

    if _has_explicit_multi_target(raw_query_text):
        return True
    if _direct_target_looks_plural(_extract_direct_target_phrase(raw_query_text)):
        return True
    return _has_explicit_multi_target(enhanced_query_text)


def _draw_hollow_marker(draw: ImageDraw.ImageDraw, cx, cy, radius, marker_color, use_cross):
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


def _draw_point_index(draw: ImageDraw.ImageDraw, cx, cy, label, marker_color):
    text = str(label)
    text_x = cx + 8
    text_y = cy + 8
    bbox = draw.textbbox((text_x, text_y), text)
    pad = 2
    draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad], fill=(255, 255, 255))
    draw.text((text_x, text_y), text, fill=marker_color)


def _build_norefine_judge_visualization(base_image, points, original_points):
    """
    norefine v1 的 judge 图：左侧全图，右侧最多 8 个红点的局部放大。
    """
    base = base_image.convert("RGB")
    image_w, image_h = base.size
    label_h = 28
    gap = 12
    resample = Image.Resampling.LANCZOS

    def draw_marker_on_panel(draw, point, scale_x, scale_y, marker_color, use_cross, radius):
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


def _get_hard_reject_reason(candidate_points, image_w, image_h):
    if not candidate_points:
        return "hard_rule_reject:no_candidate_points"

    for idx, point in enumerate(candidate_points):
        x, y = int(point[0]), int(point[1])
        if x < 0 or y < 0 or x >= image_w or y >= image_h:
            return f"hard_rule_reject:point_{idx}_outside_image_bounds"
    return ""


def _get_strict_judge_accept(judge_result, hard_reject_reason):
    if hard_reject_reason or not isinstance(judge_result, dict):
        return False
    if judge_result.get("accept") is not True:
        return False
    return all(judge_result.get(field) is True for field in STRICT_JUDGE_FIELDS)


def _build_gemini_fallback_prompt(
    raw_user_input,
    enhanced_user_input,
    user_question,
    sa2va_response,
    judge_reason,
    image_w,
    image_h,
    original_points,
):
    return f'''You are a visual point grounding model.
You will receive the original image and the task instruction.

Your task is to return the target point grounding result directly from the image.

If reference points are provided in the task metadata, treat them as contextual anchors for the instruction, especially for steerable or relational queries.
Those reference points help define where the target should be relative to them, and are not automatically the answer points.

Rules:
- Return JSON only.
- Infer from the original user input whether the task asks for one target or multiple targets.
- If the instruction asks for one specific target, return exactly one point.
- If the instruction asks for all/every/both/multiple targets, return one point for each clearly visible target instance.
- If the number of targets is uncertain, be conservative and do not invent extra points.
- Use Gemini coordinate format: [y, x] normalized to 0-1000.
- Prefer this schema: [{{"point": [y, x]}}].
- Do not provide explanation.

Original user input: {raw_user_input}
Enhanced text input: {enhanced_user_input}
Final text used by Sa2VA: {user_question}
Image resolution: {image_w} x {image_h}
Original reference points text backup (if any): {json.dumps(original_points, ensure_ascii=False)}
'''


def _run_sa2va_agent_justify_and_process_gemini_norefine(config, item, logger):
    """PointBench 本地版 Sa2VA norefine + Gemini judge/fallback 流程。"""
    local_resource = config.model_info.runtime_resource
    path = local_resource.get("sa2va_model_path") or local_resource.get("model_path")
    device = _get_local_device(config)
    print(f"{path=}")
    try:
        if not path:
            logger.error("sa2va_norefine requires sa2va_model_path/model_path in runtime_resource")
            return None, False

        model, tokenizer, processor = _load_sa2va_model_bundle(path, device)

        image_path = item.get("image_path")
        with Image.open(image_path) as raw_image:
            image = raw_image.convert("RGB")
        image_w, image_h = image.size

        question_field = getattr(config.args, "question_field")
        raw_user_input = (item.get("user_input") or "").strip()
        enhanced_user_input = (item.get(question_field) or "").strip()
        user_question = (enhanced_user_input or raw_user_input or item.get("user_prompt") or "").strip()
        system_prompt = (item.get("system_prompt") or "").strip()
        category = item.get("category", "")
        multi_target_query = _query_requests_multiple(raw_user_input, enhanced_user_input)
        original_points = item.get("original_points_in_image", []) or []
        image_name = unicodedata.normalize("NFC", item.get("image_filename") or os.path.basename(image_path or "image"))
        viz_dir = os.path.join(config.data.visualizations_dir, "predmask", category)
        os.makedirs(viz_dir, exist_ok=True)

        prompts = "<image>".join([system_prompt, item.get("user_prompt", user_question)])
        inputs = {
            "image": image,
            "text": prompts,
            "past_text": "",
            "mask_prompts": None,
        }
        if "Intern" in path:
            inputs["tokenizer"] = tokenizer
        else:
            inputs["processor"] = processor

        # 第一阶段只跑一次 Sa2VA，从 mask 中提取候选点；后续是否采用交给 judge。
        res_dict = model.predict_forward(**inputs)
        sa2va_response = res_dict.get("prediction", "")
        masks = res_dict.get("prediction_masks", None)
        overlay_img = image.copy()
        centers = []

        if masks is not None and len(masks) > 0:
            area_ratio = 0.20 if multi_target_query else 0.99
            overlay_img, centers = _get_overlay_with_centers(
                image,
                masks[0],
                area_ratio=area_ratio,
                original_points_in_image=original_points,
            )
            sa2va_response = str(centers)

        sa2va_response = _normalize_final_text_no_markdown(sa2va_response)
        overlay_save_path = unicodedata.normalize("NFC", os.path.join(viz_dir, image_name))
        overlay_img.save(overlay_save_path)
        point_only_image = _build_norefine_judge_visualization(image, centers, original_points)
        point_only_save_path = unicodedata.normalize("NFC", os.path.join(viz_dir, f"{os.path.splitext(image_name)[0]}_points_only.jpg"))
        point_only_image.save(point_only_save_path)

        judge_prompt = _build_judge_prompt(
            category,
            system_prompt,
            raw_user_input,
            enhanced_user_input,
            question_field,
            user_question,
            sa2va_response,
            image_w,
            image_h,
        )
        hard_reject_reason = _get_hard_reject_reason(centers, image_w, image_h)

        if hard_reject_reason:
            judge_raw = ""
            judge_result = {
                "rewrite_faithful": False,
                "point_count_correct": False,
                "all_points_clearly_inside_target": False,
                "spatial_relation_correct": False,
                "high_confidence": False,
                "accept": False,
                "reason": hard_reject_reason,
            }
            logger.info(f"Sa2VA norefine hard-rejected before Gemini judge: {hard_reject_reason}")
        else:
            judge_raw = _call_gemini_judge(local_resource, judge_prompt, point_only_save_path, logger)
            judge_result = _parse_json_payload(judge_raw)
            logger.info(f"Sa2VA norefine judge raw output: {judge_raw}")

        if not hard_reject_reason and not isinstance(judge_result, dict):
            logger.warning("Sa2VA norefine judge output is not valid JSON, fallback to Gemini direct grounding.")
        judge_model_accept = bool(judge_result.get("accept", False)) if isinstance(judge_result, dict) else None
        accept_sa2va = _get_strict_judge_accept(judge_result, hard_reject_reason)

        image_stem, _ = os.path.splitext(image_name)
        debug_meta_path = unicodedata.normalize("NFC", os.path.join(viz_dir, f"{image_stem}_justify_meta.json"))
        debug_meta = {
            "image_path": image_path,
            "overlay_image_path": overlay_save_path,
            "point_only_image_path": point_only_save_path,
            "debug_meta_path": debug_meta_path,
            "category": category,
            "system_prompt": system_prompt,
            "raw_user_input": raw_user_input,
            "enhanced_user_input": enhanced_user_input,
            "user_query": user_question,
            "original_points_in_image": original_points,
            "sa2va_raw_prediction": res_dict.get("prediction", ""),
            "sa2va_points": centers,
            "sa2va_response_text": sa2va_response,
            "judge_raw": judge_raw,
            "judge_parsed": judge_result,
            "judge_model_accept": judge_model_accept,
            "judge_accept": accept_sa2va,
            "judge_strict_accept": accept_sa2va,
            "judge_strict_required_fields": STRICT_JUDGE_FIELDS,
            "judge_hard_reject_reason": hard_reject_reason,
            "judge_reason": str(judge_result.get("reason", "")) if isinstance(judge_result, dict) else "",
            "has_mask": bool(masks is not None and len(masks) > 0),
            "multi_target_query_inferred": multi_target_query,
            "candidate_point_count": len(centers),
            "question_field": question_field,
            "judge_prompt": judge_prompt,
            "judge_visualization_note": "norefine v1 uses full-image plus up-to-8 zoom-crop contact sheet with enlarged hollow markers",
            "reference_points_semantics": "blue points are provided reference/context points; in steerable tasks they define relative spatial constraints rather than prediction targets",
            "gemini_fallback_prompt": "",
            "gemini_fallback_raw": "",
            "gemini_fallback_points": [],
            "gemini_fallback_response_text": "",
            "gemini_first_point_only_image_path": "",
        }

        if accept_sa2va:
            logger.info("Sa2VA norefine judge accepted Sa2VA result.")
            debug_meta["final_decision"] = "accept_sa2va"
            debug_meta["final_response"] = sa2va_response
            with open(debug_meta_path, "w", encoding="utf-8") as file:
                json.dump(debug_meta, file, ensure_ascii=False, indent=2)
            return sa2va_response, True

        logger.info("Sa2VA norefine judge rejected Sa2VA result, fallback to Gemini direct grounding.")
        judge_reason = str(judge_result.get("reason", "")) if isinstance(judge_result, dict) else ""
        grounding_prompt = _build_gemini_fallback_prompt(
            raw_user_input=raw_user_input,
            enhanced_user_input=enhanced_user_input,
            user_question=user_question,
            sa2va_response=sa2va_response,
            judge_reason=judge_reason,
            image_w=image_w,
            image_h=image_h,
            original_points=original_points,
        )
        fallback_raw = _call_gemini_grounding(local_resource, grounding_prompt, image_path, logger)
        fallback_parse_category = "counting" if multi_target_query else ""
        fallback_points = _extract_gemini_point_response(fallback_raw, image_w, image_h, fallback_parse_category)
        fallback_response_text = _normalize_final_text_no_markdown(fallback_points)
        fallback_viz_path = ""

        if fallback_points:
            fallback_viz_path = _save_points_visualization(image, fallback_points, original_points, viz_dir, f"{image_stem}_gemini_first_points_only.jpg")

        debug_meta["gemini_fallback_prompt"] = grounding_prompt
        debug_meta["gemini_fallback_raw"] = fallback_raw
        debug_meta["gemini_fallback_parse_category"] = fallback_parse_category
        debug_meta["gemini_fallback_points"] = fallback_points
        debug_meta["gemini_fallback_response_text"] = fallback_response_text
        debug_meta["gemini_first_point_only_image_path"] = fallback_viz_path
        debug_meta["final_decision"] = "fallback_to_gemini_grounding"
        debug_meta["final_response"] = fallback_points
        with open(debug_meta_path, "w", encoding="utf-8") as file:
            json.dump(debug_meta, file, ensure_ascii=False, indent=2)
        return fallback_response_text or "[]", True
    except Exception as error:
        logger.error(f"sa2va_norefine error for {item.get('image_filename')}: {error}")
        import traceback

        logger.error(traceback.format_exc())
        return None, False


def _get_original_points_context(image_path, image_points_map):
    """Return both original point prompt text and pixel coordinates for this image."""
    image_filename = os.path.basename(image_path)
    if image_filename not in image_points_map:
        return "", []

    with Image.open(image_path) as img:
        img_width, img_height = img.size

    original_points = image_points_map[image_filename]
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


def _resolve_sa2va_model_path(model_name, model_root=""):
    """Resolve point_agent weights from model_root, or use the HuggingFace repo id directly."""
    if model_root:
        return str(Path(model_root) / model_name)
    return model_name


def build_sa2va_norefine_config(
    model_name,
    query_field,
    model_root="",
    visualizations_dir="visualizations/model_evaluator_sa2va_norefine",
):
    """Build minimal config object consumed by the local Sa2VA norefine pipeline."""
    model_path = _resolve_sa2va_model_path("ByteDance/Sa2VA-Qwen3-VL-4B", model_root)
    logger.info(f"Resolved Sa2VA model path: {model_path}")
    # import sys
    # sys.exit(0)
    runtime_resource = {
        "sa2va_model_path": model_path,
        "sa2va_model_name": model_name,
        "planner_model_name": os.getenv("SA2VA_PLANNER_MODEL", "gemini-3.1-pro-preview"),
        "planner_api_key": os.getenv("API_KEY", ""),
        "planner_base_url": os.getenv("API_BASE_URL", ""),
    }
    return Box(
        {
            "model_info": {
                "runtime_resource": runtime_resource,
                "prompt_strategy": "sa2va",
            },
            "data": {
                "visualizations_dir": visualizations_dir,
            },
            "args": {
                "question_field": query_field,
                "worker_gpu_id": None,
            },
        }
    )


def _build_query_enhancement_prompts(raw_input, category=""):
    """Build the active target-preserving rewrite prompts from the tuned best prompt."""
    system_prompt = """
Rewrite raw pointing instructions for visual grounding.

Given an image and a raw instruction, first resolve the final target. Then rewrite the instruction as a short instruction that directly points to that target.

Rules:
- Output one sentence only.
- Use plain English.
- Always start with "Point to ..."
- Keep the same target and any needed count, subset, or spatial relation.
- If the image already contains a point, treat it as the reference point mentioned in the instruction.
- Use the final target's direct visible name whenever possible.
- Prefer the smallest noun phrase that still preserves the same target.
- When the raw instruction uses a generic placeholder such as object, thing, area, region, space, or place, and visible relations already identify the target, preserve that relation-defined target unless a more concrete noun is clearly equivalent.
- When the raw instruction intentionally targets an unspecified area such as somewhere, area, region, or space, preserve that area target instead of inventing a specific object.
- When the raw instruction explicitly asks for a word, text, letter, number, date, time, price, or other visible text token, preserve that text-type target and any needed anchor relation.
- For function, affordance, identity-evidence, or support-location instructions, prefer the smallest direct object, object part, or visible evidence that fulfills the description.
- Only output the exact visible string when it is clearly legible and necessary to preserve the same target; otherwise keep a generic text-target noun.
- Do not expand the target to a surrounding parent object, support surface, container, or larger group when a smaller direct target preserves the same meaning.
- Do not add colors, materials, support relations, parent objects, or other descriptive details unless they are needed to keep the same target.
- Never rewrite a plural or set target to a single instance, and never rewrite a single target to a set.
- Do not replace the target with the anchor point, red circle, or a nearby container, support object, or parent region.
- For movement instructions, rewrite to the final destination.
- If a movement instruction already names the destination after phrases like reach, touch, on, onto, into, or until, keep that literal destination object or area rather than switching to an intermediate object, nearby support object, or the anchor mark.
- Do not explain or guess invisible details.

Examples:
- "Point to the tool people use to write." -> "Point to the pen."
- "Point to the jewelry boxes." -> "Point to the jewelry boxes."
- "Point to the object on which plants are placed." -> "Point to the blue inflatable tray."
- "Point to the place where a water bottle could be placed." -> "Point to the bike's water bottle cage."
- "Point to what can record or take pictures." -> "Point to the drone's camera."
- "Point to moving left until you are on the ground." -> "Point to the ground."
- "Point to moving the point down until it touches the bowl of water." -> "Point to the bowl of water."
- "Point to the object to the left of the red car and to the right of the black car." -> "Point to the tree to the left of the red car and to the right of the black car."
- "Point to the number to the right of the current point on the image." -> "Point to the number to the right of the current point."
- "Point to the space at the crossroads." -> "Point to the road intersection at the crossroads."
""".strip()
    user_prompt = f"""
Rewrite the following raw instruction into a direct, visually grounded instruction that already names the final target and preserves any count, subset, or anchor constraints still needed to identify it.

Raw instruction:
{raw_input}
""".strip()
    return system_prompt, user_prompt


def _generate_enhanced_query_once(item_ctx, image_path, model_name):
    """Generate one enhanced query for the current item before norefine inference."""
    raw_input = str(item_ctx.get("user_input") or "").strip()
    category = str(item_ctx.get("category") or "").strip()
    system_prompt, user_prompt = _build_query_enhancement_prompts(raw_input, category)

    api_key = os.getenv("API_KEY", "").strip()
    if not api_key:
        raise ValueError("API_KEY is required for auto query enhancement.")
    base_url = os.getenv("API_BASE_URL", "").strip()
    if base_url:
        client = genai.Client(http_options=types.HttpOptions(base_url=base_url), api_key=api_key)
    else:
        client = genai.Client(api_key=api_key)

    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")
        image_bytes_io = io.BytesIO()
        image.save(image_bytes_io, format="JPEG")
        image_bytes = image_bytes_io.getvalue()

    response = client.models.generate_content(
        model=model_name,
        contents=[
            user_prompt,
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        ],
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    content = (response.text or "").strip().replace("```json", "").replace("```", "").strip()
    if not content:
        raise ValueError("Enhance query model returned empty content.")
    if len(content) >= 2 and content[0] == content[-1] and content[0] in {'"', "'"}:
        content = content[1:-1].strip()
    return content


def _parse_sa2va_response_points(response_text):
    """Normalize Sa2VA response into PointBench point dict format."""
    parsed = _parse_json_payload(response_text)
    points_xy = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict) and isinstance(item.get("point"), (list, tuple)) and len(item["point"]) == 2:
                x_value, y_value = item["point"][0], item["point"][1]
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                x_value, y_value = item[0], item[1]
            else:
                continue
            points_xy.append([float(x_value), float(y_value)])

    if not points_xy:
        coords = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", str(response_text or ""))
        for x_value, y_value in coords:
            points_xy.append([float(x_value), float(y_value)])

    return [{"point": [x_value, y_value]} for x_value, y_value in points_xy]


def call_sa2va_agent_justify_and_process_gemini_norefine(
    image_path,
    object_name,
    model_name="ByteDance/Sa2VA-Qwen3-VL-4B",
    category=None,
    item_ctx=None,
    runtime_options=None,
    *,
    image_points_map,
    logger,
):
    """Run local Sa2VA norefine pipeline and return PointBench point dicts."""
    options = runtime_options or {}
    query_field = options.get("query_field", "user_input")
    model_root = options.get("model_root", "")
    enhance_model = options.get("enhance_model", "gemini-3.1-pro-preview")

    config = options.get("sa2va_config") or build_sa2va_norefine_config(model_name, query_field, model_root)
    config.args.question_field = query_field

    pipeline_item = dict(item_ctx or {})
    pipeline_item["image_path"] = image_path
    pipeline_item["image_filename"] = pipeline_item.get("image_filename") or os.path.basename(image_path)
    pipeline_item["category"] = category or pipeline_item.get("category", "")
    pipeline_item["user_input"] = str(pipeline_item.get("user_input") or object_name or "").strip()

    original_points_info, original_points_in_image = _get_original_points_context(image_path, image_points_map)
    pipeline_item["original_points_info"] = original_points_info
    pipeline_item["original_points_in_image"] = original_points_in_image

    # point_agent 当前固定先在线生成增强 query，不读取数据集里可能存在的预生成字段。
    # enhanced_query = _generate_enhanced_query_once(
    #     item_ctx=pipeline_item,
    #     image_path=image_path,
    #     model_name=enhance_model,
    # )
    enhanced_query = pipeline_item["user_input"]
    pipeline_item["enhanced_query"] = enhanced_query
    pipeline_item[query_field] = enhanced_query
    if item_ctx is not None:
        item_ctx["enhanced_query"] = enhanced_query
        if query_field != "user_input":
            item_ctx[query_field] = enhanced_query
    logger.info(f"Generated query for {pipeline_item['image_filename']}: {enhanced_query}")

    with Image.open(image_path) as image:
        image_size = image.size

    system_prompt, user_prompt = _build_task_prompt_for_field(
        config,
        pipeline_item,
        question_field=query_field,
        image_size=image_size,
    )
    pipeline_item["system_prompt"] = system_prompt
    pipeline_item["user_prompt"] = user_prompt

    response_text, success = _run_sa2va_agent_justify_and_process_gemini_norefine(config, pipeline_item, logger)
    if not success:
        return []

    return _parse_sa2va_response_points(response_text)
