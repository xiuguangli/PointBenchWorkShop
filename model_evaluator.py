import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from queue import Empty
import random
import csv
import re  # Add explicit import for re
from pathlib import Path
import shutil
from typing import List, Dict, Tuple, Any, Optional, Union
import numpy as np
from PIL import Image, ImageDraw
from dotenv import load_dotenv
import io
import unicodedata
from tqdm import tqdm
from point_agent import (
    POINT_AGENT_MODEL_TYPES,
    build_sa2va_norefine_config,
    call_sa2va_agent_justify_and_process_gemini_norefine as run_sa2va_agent_justify_and_process_gemini_norefine,
)

# Import the same model interfaces and helpers as the main app
from openai import OpenAI
# import google.generativeai as genai
from google import genai
from google.genai import types
import torch
# from transformers import (
#     AutoModelForCausalLM, 
#     AutoProcessor, 
#     AutoTokenizer, 
#     GenerationConfig,
#     Qwen2_5_VLForConditionalGeneration, 
#     AutoModelForVision2Seq,
#     LlavaOnevisionForConditionalGeneration
# )
import base64
import anthropic
# from wandb import api

# Load environment variables
load_dotenv()

def get_logger(logs_dir="logs", log_name="log"):
    from loguru import logger
    from datetime import datetime
    import sys
    Path(logs_dir).mkdir(exist_ok=True, parents=True)
    
    def pretty_dict(obj, indent=0, ignore_keys=None):
        if ignore_keys is None:
            ignore_keys = []
        if not isinstance(obj, dict):
            return str(obj)
        lines = []
        for key in obj:
            if any(ik in str(key) for ik in ignore_keys):
                continue
            value = obj[key]
            prefix = ' ' * indent + str(key) + ': '
            if isinstance(value, dict):
                lines.append(prefix)
                lines.append(pretty_dict(value, indent + 4, ignore_keys))
            else:
                lines.append(prefix + str(value))
        return '\n'.join(lines)

    logger.remove()
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name:<4}</cyan>:<cyan>{function:<15}</cyan>:<cyan>{line:<4}</cyan> | "
        "{message}"
    )
    current_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_name = f"{logs_dir}/{log_name}_{current_date}.log"
    logger.add(log_name, format=log_format, level="INFO")
    logger.add(sys.stderr, format=log_format, level="INFO")
    ignore_keys = ['key','api']

    # monkey patch logger.info: 支持 logger.info(dict) 自动美化
    _orig_info = getattr(logger, "_pointbench_original_info", logger.info)
    logger._pointbench_original_info = _orig_info
    return logger

import os
logger = get_logger(logs_dir=f"logs/{os.path.basename(__file__)}", log_name="logs")

# Configure API keys and clients
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# client = OpenAI(api_key=os.getenv("UNIAPI_KEY"), base_url=os.getenv("UNIAPI_BASE_URL", "https://hk.uniapi.io"))
client = None
# anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
anthropic_client = None
# xai_client = OpenAI(
#     api_key=os.getenv("XAI_API_KEY"),
#     base_url="https://api.x.ai/v1",
# )
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Constants
IMAGES_DIR = Path("data/images")
MASKS_DIR = Path("data/masks")
POINT_ON_MASK_DIR = Path("point_on_mask")  # New directory for visualization images
SA2VA_PROCESS_VISIBLE_GPU_ID = None
SA2VA_RESULT_MESSAGE = "result"
SA2VA_DONE_MESSAGE = "done"

# Create the point_on_mask directory if it doesn't exist
POINT_ON_MASK_DIR.mkdir(exist_ok=True, parents=True)

def _safe_path_part(value):
    normalized = unicodedata.normalize("NFC", str(value).strip())
    normalized = normalized.replace("/", "_").replace("\\", "_")
    return re.sub(r"[^0-9A-Za-z._-]+", "_", normalized).strip("_")

def _build_run_output_name(model_type, model_name, result_suffix):
    # 每次评测产物都用“模型类型-模型名字_suffix”分桶，避免不同实验堆到同一层目录。
    run_name = f"{_safe_path_part(model_type)}-{_safe_path_part(model_name)}"
    if result_suffix:
        run_name = f"{run_name}_{_safe_path_part(result_suffix)}"
    return run_name

def _prepare_run_output_paths(run_output_name):
    output_paths = {
        "results_dir": Path("static_results") / run_output_name,
        "point_on_mask_dir": POINT_ON_MASK_DIR / run_output_name,
        "sa2va_visualizations_dir": Path("visualizations/point_agent") / run_output_name,
        "logs_dir": Path("logs") / os.path.basename(__file__) / run_output_name,
    }
    for path in output_paths.values():
        path.mkdir(exist_ok=True, parents=True)
    return output_paths

def _parse_cuda_visible_devices():
    value = os.getenv("CUDA_VISIBLE_DEVICES", "")
    return [device.strip() for device in value.split(",") if device.strip()]

def _set_cuda_visible_devices(value):
    if value is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = value


def _run_sa2va_process_worker(gpu_id, worker_items, result_queue, logs_dir):
    global logger, SA2VA_PROCESS_VISIBLE_GPU_ID

    SA2VA_PROCESS_VISIBLE_GPU_ID = "0"
    logger = get_logger(logs_dir=logs_dir, log_name=f"gpu_{_safe_path_part(gpu_id)}")
    logger.info(f"point_agent process worker started on physical GPU {gpu_id} as cuda:0")

    for task_args in worker_items:
        result_queue.put((SA2VA_RESULT_MESSAGE, _evaluate_single_item(task_args)))
    result_queue.put((SA2VA_DONE_MESSAGE, gpu_id))

def _apply_sa2va_process_runtime(runtime_options):
    if SA2VA_PROCESS_VISIBLE_GPU_ID is None:
        return runtime_options

    worker_runtime_options = dict(runtime_options)
    config = worker_runtime_options["sa2va_config"]
    # 每个 point_agent 子进程只暴露一张物理卡，因此进程内部统一使用逻辑 cuda:0。
    config.args.worker_gpu_id = SA2VA_PROCESS_VISIBLE_GPU_ID
    worker_runtime_options["sa2va_config"] = config
    return worker_runtime_options

# Load the image_filename to points mapping from CSV file
IMAGE_POINTS_MAP = {}
try:
    with open('data/pixmo_metadata.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['points'] and row['points'] != '[]':
                IMAGE_POINTS_MAP[row['image_filename']] = json.loads(row['points'])
    logger.info(f"Loaded points data for {len(IMAGE_POINTS_MAP)} images from pixmo_metadata.csv")
except Exception as e:
    logger.info(f"Error loading pixmo_metadata.csv: {e}")
    IMAGE_POINTS_MAP = {}

# Available models
OPENAI_MODELS = ["gpt-5.4","gpt-5.4-pro","gpt-4o", "o3", "gpt-4.1"]
GEMINI_MODELS = ["gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-preview-05-06","gemini-2.0-flash"]
GEMINI_MODELS = ["gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-3.1-pro-preview","gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-preview-05-06","gemini-2.0-flash"]
MOLMO_MODELS = ["Molmo-7B-D-0924", "Molmo-7B-O-0924", "Molmo-72B-0924"]
QWEN_MODELS = ["Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct"]
LLAVA_MODELS = ["llava-onevision-qwen2-7b-ov-hf"]
CLAUDE_MODELS = ["claude-3-7-sonnet-20250219"]
GROK_MODELS = ["grok-2-vision-latest"]
POINT_AGENT_MODELS = ["point_agent"]

# Use local models
USE_LOCAL_MODELS = True
if USE_LOCAL_MODELS:
    SAVED_MODELS_DIR = Path(os.getenv("SAVED_MODELS_DIR", "models"))
    SAVED_MODELS_DIR.mkdir(exist_ok=True, parents=True)
else:
    SAVED_MODELS_DIR = None

# Initialize Molmo model and processor (lazy loading)
molmo_model = None
molmo_processor = None

# Initialize Qwen model and processor (lazy loading)
qwen_model = None
qwen_processor = None

# Initialize LLaVA model and processor (lazy loading)
llava_model = None
llava_processor = None

# Add a utility function to print complete prompts near the beginning of the file, after imports
def print_complete_prompt(system_content, user_content, model_name, image_path):
    """Print the complete prompt including system content and user content."""
    logger.info("\n" + "="*80)
    logger.info(f"COMPLETE PROMPT FOR {model_name} ON {image_path}:")
    logger.info("-"*80)
    if system_content:
        logger.info(f"SYSTEM CONTENT:\n{system_content}")
        logger.info("-"*80)
    logger.info(f"USER CONTENT:\n{user_content}")
    logger.info("="*80 + "\n")

def initialize_molmo(model_name="allenai/Molmo-7B-D-0924"):
    """Initialize Molmo model and processor if not already initialized."""
    from transformers import (
        AutoModelForCausalLM, 
        AutoProcessor, 
        AutoTokenizer, 
        GenerationConfig,
        Qwen2_5_VLForConditionalGeneration, 
        AutoModelForVision2Seq,
        LlavaOnevisionForConditionalGeneration
    )
    global molmo_model, molmo_processor
    
    if molmo_model is None or molmo_processor is None:
        # Get model short name
        model_short_name = model_name.split('/')[-1]
        
        if USE_LOCAL_MODELS:
            # Use local model
            local_model_dir = SAVED_MODELS_DIR / model_short_name
            
            if not local_model_dir.exists():
                raise ValueError(f"Model directory does not exist: {local_model_dir}. Please ensure the model has been downloaded to this directory.")
            
            logger.info(f"Loading Molmo model from local directory: {local_model_dir}")
            
            # Load from local directory
            molmo_processor = AutoProcessor.from_pretrained(
                local_model_dir,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto'
            )
            
            molmo_model = AutoModelForCausalLM.from_pretrained(
                local_model_dir,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto'
            )
        else:
            # Use remote model
            logger.info(f"Loading Molmo model from Hugging Face: {model_name}")
            
            # Load processor from remote
            molmo_processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto'
            )
            
            # Load model from remote
            molmo_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype='auto',
                device_map='auto'
            )
        
    return molmo_model, molmo_processor

def initialize_qwen(model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    """Initialize Qwen model and processor if not already initialized."""
    
    from transformers import (
        AutoModelForCausalLM, 
        AutoProcessor, 
        AutoTokenizer, 
        GenerationConfig,
        Qwen2_5_VLForConditionalGeneration, 
        AutoModelForVision2Seq,
        LlavaOnevisionForConditionalGeneration
    )
    global qwen_model, qwen_processor
    
    if qwen_model is None or qwen_processor is None:
        # Get model short name
        model_short_name = model_name.split('/')[-1]
        
        if USE_LOCAL_MODELS:
            # Use local model
            local_model_dir = SAVED_MODELS_DIR / model_short_name
            
            if not local_model_dir.exists():
                raise ValueError(f"Model directory does not exist: {local_model_dir}. Please ensure the model has been downloaded to this directory.")
            
            logger.info(f"Loading Qwen model from local directory: {local_model_dir}")
            
            # Load from local directory
            qwen_processor = AutoProcessor.from_pretrained(
                local_model_dir,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
            
            qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                local_model_dir,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                # attn_implementation="flash_attention_2",
                device_map='auto'
            )

            logger.info(qwen_model.hf_device_map)
        else:
            # Use remote model
            logger.info(f"Loading Qwen model from Hugging Face: {model_name}")
            
            # Load processor from remote
            qwen_processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='auto',
            )
            
            # Load model from remote
            qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                # attn_implementation="flash_attention_2",
                device_map='auto'
            )
        
    return qwen_model, qwen_processor

def initialize_llava(model_name="llava-hf/llava-onevision-qwen2-7b-ov-hf"):
    """Initialize LLaVA-OneVision model and processor if not already initialized."""
    from transformers import (
        AutoModelForCausalLM, 
        AutoProcessor, 
        AutoTokenizer, 
        GenerationConfig,
        Qwen2_5_VLForConditionalGeneration, 
        AutoModelForVision2Seq,
        LlavaOnevisionForConditionalGeneration
    )
    global llava_model, llava_processor
    
    if llava_model is None or llava_processor is None:
        # Get model short name
        model_short_name = model_name.split('/')[-1]
        
        if USE_LOCAL_MODELS:
            # Use local model
            local_model_dir = SAVED_MODELS_DIR / model_short_name
            
            if not local_model_dir.exists():
                raise ValueError(f"Model directory does not exist: {local_model_dir}. Please ensure the model has been downloaded to this directory.")
            
            logger.info(f"Loading LLaVA-OneVision model from local directory: {local_model_dir}")
            
            # Load the model and processor using standard approach - LLaVA-OneVision HF should work with this
            try:
                logger.info("[DEBUG] Loading processor from local directory")
                llava_processor = AutoProcessor.from_pretrained(
                    local_model_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                
                logger.info("[DEBUG] Loading model from local directory")
                # Use the specialized model class for LLaVA-OneVision HF version
                llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                    local_model_dir,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map='auto'
                )
                logger.info(f"[DEBUG] Model type: {type(llava_model).__name__}")
            except Exception as e:
                logger.info(f"[DEBUG] Error loading model: {e}")
                raise
        else:
            # Use remote model
            logger.info(f"Loading LLaVA-OneVision model from Hugging Face: {model_name}")
            
            # Load processor and model using standard approach
            try:
                logger.info("[DEBUG] Loading processor from Hugging Face")
                llava_processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                
                logger.info("[DEBUG] Loading model from Hugging Face")
                # Use the specialized model class for LLaVA-OneVision HF version
                llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                    model_name,
                    trust_remote_code=True, 
                    torch_dtype=torch.float16,
                    device_map='auto'
                )
                logger.info(f"[DEBUG] Model type: {type(llava_model).__name__}")
            except Exception as e:
                logger.info(f"[DEBUG] Error loading model: {e}")
                raise
        
    return llava_model, llava_processor

def get_original_points_info(image_path, category):
    """
    Get information about original points for steerable images.
    
    Args:
        image_path (str): Path to the image file
        category (str): Image category
        
    Returns:
        str: Information string about original points or empty string if not applicable
    """
    if category != "steerable":
        return ""
    original_points_info, _ = get_original_points_context(image_path)
    return original_points_info


def get_original_points_context(image_path):
    """Return both original point prompt text and pixel coordinates for this image."""
    image_filename = os.path.basename(image_path)
    if image_filename not in IMAGE_POINTS_MAP:
        return "", []

    with Image.open(image_path) as img:
        img_width, img_height = img.size

    original_points = IMAGE_POINTS_MAP[image_filename]
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


def call_sa2va_agent_justify_and_process_gemini_norefine(
    image_path,
    object_name,
    model_name="ByteDance/Sa2VA-Qwen3-VL-4B",
    category=None,
    item_ctx=None,
    runtime_options=None,
):
    """Delegate Sa2VA norefine inference to dedicated module."""
    return run_sa2va_agent_justify_and_process_gemini_norefine(
        image_path=image_path,
        object_name=object_name,
        model_name=model_name,
        category=category,
        item_ctx=item_ctx,
        runtime_options=runtime_options,
        image_points_map=IMAGE_POINTS_MAP,
        logger=logger,
    )

def call_openai0(image_path, object_name, model_name="gpt-4o", category=None):
    """Call OpenAI model to get points for the specified object."""

    # Read the image file
    with open(image_path, "rb") as image_file:
        # Encode the image as base64
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Get image dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Determine MIME type based on file extension
    file_extension = os.path.splitext(image_path)[1].lower()
    if file_extension == '.png':
        mime_type = "image/png"
    elif file_extension in ['.jpg', '.jpeg']:
        mime_type = "image/jpeg"
    elif file_extension == '.webp':
        mime_type = "image/webp"
    elif file_extension == '.gif':
        mime_type = "image/gif"
    else:
        # Default to jpeg for other formats
        mime_type = "image/jpeg"
    
    # Get information about original points for steerable images
    original_points_info = get_original_points_info(image_path, category)
    
    # Check if category is counting - limit points accordingly
    if category == "counting":
            prompt = f"""
            {object_name}.
            The image dimensions are width={img_width}px, height={img_height}px.{original_points_info}
            The answer should follow the json format: [{{"point": <point>}}, ...]. 
            IMPORTANT: The points MUST be in [x, y] format where x is the horizontal position (left-to-right) and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
            Example: For a point in the center of the image, return [width/2, height/2].
            """
    else:
        prompt = f"""
        {object_name}.
        The image dimensions are width={img_width}px, height={img_height}px.{original_points_info}
        The answer should follow the json format: [{{"point": <point>}}]. 
        IMPORTANT: Return EXACTLY ONE POINT. The point MUST be in [x, y] format where x is the horizontal position (left-to-right) and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
        Example: For a point in the center of the image, return [width/2, height/2].
            """
    
    # Define system content
    system_content = "You are a helpful assistant that can identify objects in images and provide their coordinates."
    
    # Print complete prompt
    print_complete_prompt(system_content, prompt, model_name, image_path)
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
                ]}
            ],
        )
        
        content = response.choices[0].message.content
        # Extract JSON from the response
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end]
            points = json.loads(json_str)
            # If not counting category and more than one point was returned, limit to first point
            if category != "counting" and len(points) > 1:
                points = [points[0]]
            return points
        else:
            return []
    except Exception as e:
        logger.error(f"Error with {model_name} on {image_path}: {e}")
        return []

def call_openai(image_path, object_name, model_name="gpt-4o", category=None):
    """Call OpenAI model to get points for the specified object."""

    # Read the image file
    with open(image_path, "rb") as image_file:
        # Encode the image as base64
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Get image dimensions
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Determine MIME type based on file extension
    file_extension = os.path.splitext(image_path)[1].lower()
    if file_extension == '.png':
        mime_type = "image/png"
    elif file_extension in ['.jpg', '.jpeg']:
        mime_type = "image/jpeg"
    elif file_extension == '.webp':
        mime_type = "image/webp"
    elif file_extension == '.gif':
        mime_type = "image/gif"
    else:
        # Default to jpeg for other formats
        mime_type = "image/jpeg"
    
    # Get information about original points for steerable images
    original_points_info = get_original_points_info(image_path, category)
    
    # Check if category is counting - limit points accordingly
    if category == "counting":
            prompt = f"""
            {object_name}.
            The image dimensions are width={img_width}px, height={img_height}px.{original_points_info}
            The answer should follow the json format: [{{"point": <point>}}, ...]. 
            IMPORTANT: The points MUST be in [x, y] format where x is the horizontal position (left-to-right) and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
            Example: For a point in the center of the image, return [width/2, height/2].
            """
    else:
        prompt = f"""
        {object_name}.
        The image dimensions are width={img_width}px, height={img_height}px.{original_points_info}
        The answer should follow the json format: [{{"point": <point>}}]. 
        IMPORTANT: Return EXACTLY ONE POINT. The point MUST be in [x, y] format where x is the horizontal position (left-to-right) and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
        Example: For a point in the center of the image, return [width/2, height/2].
            """
    
    # Define system content
    system_content = "You are a helpful assistant that can identify objects in images and provide their coordinates."
    
    # Print complete prompt
    print_complete_prompt(system_content, prompt, model_name, image_path)
    
    try:
        # response = client.chat.completions.create(
        response = client.responses.create(
            model=model_name,
            # messages=[
            input=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:{mime_type};base64,{base64_image}"}
                ]}
            ],
        )
        
        # content = response.choices[0].message.content
        content = response.output_text
        # print(f"Raw response content: {content}")

        # Extract JSON from the response
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end]
            points = json.loads(json_str)
            # If not counting category and more than one point was returned, limit to first point
            if category != "counting" and len(points) > 1:
                points = [points[0]]
            return points
        else:
            return []
    except Exception as e:
        logger.error(f"Error with {model_name} on {image_path}: {e}")
        return []


def call_gemini(image_path, object_name, model_name="gemini-2.0-flash", category=None):
    """Call Gemini to get points for the specified object."""
    try:
        # Configure the model
        # model = genai.GenerativeModel(model_name)
        # api_key = os.getenv("GOOGLE_API_KEY")
        # client = genai.Client(api_key=api_key)
        api_key = os.getenv("UNIAPI_KEY")
        client = genai.Client(http_options=types.HttpOptions(base_url='https://api.uniapi.io/gemini'),
        api_key=api_key)
        # logger.info(f"{api_key=}")
        
        # Get image dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Ensure image is in a supported format
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Determine MIME type based on file extension
        file_extension = os.path.splitext(image_path)[1].lower()
        if file_extension == '.png':
            mime_type = "image/png"
            img_format = 'PNG'
        elif file_extension in ['.jpg', '.jpeg']:
            mime_type = "image/jpeg"
            img_format = 'JPEG'
        elif file_extension == '.webp':
            mime_type = "image/webp"
            img_format = 'WEBP'
        elif file_extension == '.gif':
            mime_type = "image/gif"
            img_format = 'GIF'
        else:
            # Default to jpeg for other formats
            mime_type = "image/jpeg"
            img_format = 'JPEG'
        
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img_format)
        image_data = img_byte_arr.getvalue()
        
        # Get information about original points for steerable images
        original_points_info = get_original_points_info(image_path, category)
        
    
        # NOTE: Gemini uses a different coordinate system: [y, x] format and 0-1000 normalization
        # Check if category is counting - limit points accordingly
        if category == "counting":
            prompt = f"""
            {object_name}
            {original_points_info}
            """
        else:
            prompt = f"""
            {object_name}
            {original_points_info}
            """
        
        # Prepare the content parts in the order of text first, then image
        # prompt_parts = [
        #     prompt,
        #     {
        #         "mime_type": mime_type,
        #         "data": image_data
        #     }
        # ]
        prompt_parts=[
            prompt,
            types.Part.from_bytes(
                data=image_data,
                mime_type=mime_type,
            ), 
        ]
        
        logger.info(f"\nSending prompt to Gemini ({model_name}) with image {image_path}...")
        
        # Make the API call
        # response = model.generate_content(prompt_parts)
        # response = model.generate_content(prompt_parts)
        response = client.models.generate_content(
            model=model_name, 
            contents=prompt_parts,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="high")
            ),
        )
        
        # response = client.models.generate_content(
        #     model='gemini-3-flash-preview',
        #     contents=[
        #     types.Part.from_bytes(
        #         data=image_bytes,
        #         mime_type='image/jpeg',
        #     ),
        #     'Caption this image.'
        #     ]
        # )
        
        # Check if response parts exist and have content
        # if response.parts:
        if response.text:
            content = response.text
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason'):
            raise ValueError(f"Content blocked: {getattr(response.prompt_feedback, 'block_reason_message', '') or response.prompt_feedback.block_reason}")
        else:
            raise ValueError("No text content received from Gemini, or response was empty/unexpected.")
        
        logger.info(f"\n[DEBUG] Raw Gemini output for {object_name} in {image_path}:")
        logger.info(content)
        
        # Extract JSON from the response
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end]
            logger.info(f"[DEBUG] Extracted JSON string: {json_str}")
            
            # Parse the JSON
            raw_points = json.loads(json_str)
            
            # Convert from Gemini's format ([y, x] in 0-1000 range) to standard format ([x, y] in pixels)
            points = []
            for item in raw_points:
                if isinstance(item, dict) and "point" in item:
                    if isinstance(item["point"], list) and len(item["point"]) == 2:
                        # Gemini format: [y, x] normalized to 0-1000
                        # We need to: 1) swap coordinates and 2) convert to pixels
                        y, x = item["point"]
                        # Convert normalized coordinates (0-1000) to pixel coordinates
                        pixel_x = (x / 1000.0) * img_width
                        pixel_y = (y / 1000.0) * img_height
                        # Add to points list in standard format
                        points.append({"point": [pixel_x, pixel_y]})
            
            logger.info(f"[DEBUG] Converted points: {points}")
            
            # If no valid points were found or conversion failed, try regex to extract coordinates
            if not points:
                import re
                # Look for patterns like [y, x] or [number, number]
                coords = re.findall(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]', json_str)
                if coords:
                    logger.info(f"[DEBUG] Coordinates extracted via regex: {coords}")
                    # First coordinate is y, second is x in Gemini's format
                    for y_str, x_str in coords:
                        try:
                            y, x = float(y_str), float(x_str)
                            # Convert normalized coordinates (0-1000) to pixel coordinates
                            pixel_x = (x / 1000.0) * img_width
                            pixel_y = (y / 1000.0) * img_height
                            points.append({"point": [pixel_x, pixel_y]})
                        except ValueError:
                            continue
                    logger.info(f"[DEBUG] Points after regex extraction: {points}")
            
            # If not counting category and more than one point was returned, limit to first point
            if category != "counting" and len(points) > 1:
                points = [points[0]]
            
            return points
        else:
            return []
    except Exception as e:
        logger.error(f"Error with {model_name} on {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return []

def call_claude(image_path, object_name, model_name="claude-3-7-sonnet-20250219", category=None):
    """Call Claude to get points for the specified object."""
    try:
        # Read the image file as base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Get image dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Determine MIME type based on file extension
        file_extension = os.path.splitext(image_path)[1].lower()
        if file_extension == '.png':
            mime_type = "image/png"
        elif file_extension in ['.jpg', '.jpeg']:
            mime_type = "image/jpeg"
        elif file_extension == '.webp':
            mime_type = "image/webp"
        elif file_extension == '.gif':
            mime_type = "image/gif"
        else:
            # Default to jpeg for other formats
            mime_type = "image/jpeg"
        
        # Get information about original points for steerable images
        original_points_info = get_original_points_info(image_path, category)
        
        # Define system content 
        system_content = "You are a helpful assistant that can identify objects in images and provide their coordinates."
        
        # Check if category is counting - limit points accordingly
        if category == "counting":
            prompt = f"""
            {object_name}.
            The image dimensions are width={img_width}px, height={img_height}px.{original_points_info}
            The answer should follow the json format: [{{"point": <point>}}, ...]. 
            IMPORTANT: The points MUST be in [x, y] format where x is the horizontal position (left-to-right) and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
            Example: For a point in the center of the image, return [width/2, height/2].
            """
        else:
            prompt = f"""
            {object_name}.
            The image dimensions are width={img_width}px, height={img_height}px.{original_points_info}
            The answer should follow the json format: [{{"point": <point>}}]. 
            IMPORTANT: Return EXACTLY ONE POINT. The point MUST be in [x, y] format where x is the horizontal position (left-to-right) and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
            Example: For a point in the center of the image, return [width/2, height/2].
            """
        
        # Print complete prompt
        print_complete_prompt(system_content, prompt, model_name, image_path)
        
        # Call the Claude API
        response = anthropic_client.messages.create(
            model=model_name,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"{system_content}\n\n{prompt}"
                        }
                    ],
                }
            ],
        )
        
        content = response.content[0].text
        # Extract JSON from the response
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end]
            points = json.loads(json_str)
            # If not counting category and more than one point was returned, limit to first point
            if category != "counting" and len(points) > 1:
                points = [points[0]]
            return points
        else:
            return []
    except Exception as e:
        logger.error(f"Error with {model_name} on {image_path}: {e}")
        return []

def call_grok(image_path, object_name, model_name="grok-2-vision-latest", category=None):
    """Call Grok to get points for the specified object."""
    try:
        # Determine MIME type based on file extension
        file_extension = os.path.splitext(image_path)[1].lower()
        if file_extension == '.png':
            mime_type = "image/png"
        elif file_extension in ['.jpg', '.jpeg']:
            mime_type = "image/jpeg"
        elif file_extension == '.webp':
            mime_type = "image/webp"
        elif file_extension == '.gif':
            mime_type = "image/gif"
        else:
            # Default to jpeg for other formats
            mime_type = "image/jpeg"
        
        # Read the image file as base64
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Get image dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # Get information about original points for steerable images
        original_points_info = get_original_points_info(image_path, category)
        
        # Define system content
        system_content = "You are a helpful assistant that can identify objects in images and provide their coordinates."
        
        # Check if category is counting - limit points accordingly
        if category == "counting":
            prompt = f"""
            {object_name}.
            The image dimensions are width={img_width}px, height={img_height}px.{original_points_info}
            The answer should follow the json format: [{{"point": <point>}}, ...]. 
            IMPORTANT: The points MUST be in [x, y] format where x is the horizontal position (left-to-right) and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
            Example: For a point in the center of the image, return [width/2, height/2].
            """
        else:
            prompt = f"""
            {object_name}.
            The image dimensions are width={img_width}px, height={img_height}px.{original_points_info}
            The answer should follow the json format: [{{"point": <point>}}]. 
            IMPORTANT: Return EXACTLY ONE POINT. The point MUST be in [x, y] format where x is the horizontal position (left-to-right) and y is the vertical position (top-to-bottom) in PIXEL COORDINATES (not normalized).
            Example: For a point in the center of the image, return [width/2, height/2].
            """
        
        # Print complete prompt
        print_complete_prompt(system_content, prompt, model_name, image_path)
        
        # Set up messages for the XAI API call
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_data}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": f"{system_content}\n\n{prompt}",
                    },
                ],
            },
        ]
        
        # Call the XAI API
        response = xai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.01,
        )
        
        content = response.choices[0].message.content
        # Extract JSON from the response
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end]
            points = json.loads(json_str)
            # If not counting category and more than one point was returned, limit to first point
            if category != "counting" and len(points) > 1:
                points = [points[0]]
            return points
        else:
            return []
    except Exception as e:
        logger.error(f"Error with {model_name} on {image_path}: {e}")
        return []

def extract_points(text, image_w, image_h):
    """Extract points from text using multiple regex patterns.
    
    Extracts normalized coordinates (0-100 range) and converts them to pixel coordinates.
    Handles multiple formats like Click(x,y), (x,y), x="x" y="y", and p=xxx,yyy.
    
    Args:
        text: Text containing coordinate information
        image_w: Image width in pixels
        image_h: Image height in pixels
        
    Returns:
        List of points as numpy arrays in pixel coordinates
    """
    all_points = []
    for match in re.finditer(r"Click\(([0-9]+\.[0-9]), ?([0-9]+\.[0-9])\)", text):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)

    for match in re.finditer(r"\(([0-9]+\.[0-9]),? ?([0-9]+\.[0-9])\)", text):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', text):
        try:
            point = [float(match.group(i)) for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    for match in re.finditer(r'(?:\d+|p)\s*=\s*([0-9]{3})\s*,\s*([0-9]{3})', text):
        try:
            point = [int(match.group(i)) / 10.0 for i in range(1, 3)]
        except ValueError:
            pass
        else:
            point = np.array(point)
            if np.max(point) > 100:
                # Treat as an invalid output
                continue
            point /= 100.0
            point = point * np.array([image_w, image_h])
            all_points.append(point)
    return all_points


def call_qwen(image_path, object_name, model_name="Qwen/Qwen2.5-VL-7B-Instruct", category=None):
    """Call Qwen model to get points for the specified object."""
    try:
        # Initialize model and processor if not already done
        model, processor = initialize_qwen(model_name)
        
        # Load the image
        image = Image.open(image_path)
        img_width, img_height = image.size
        logger.info(f"[DEBUG] Image dimensions: {img_width}x{img_height}")
        
        # Get information about original points for steerable images
        original_points_info = get_original_points_info(image_path, category)
        
        # Define system content
        system_content = "You are a helpful assistant."
        
        # Prepare the prompt based on category
        if category == "counting":
            prompt = f"""
            {object_name}
            Output its coordinates in XML format <points x y>object</points>.
            {original_points_info}
            """
        else:
            prompt = f"""
            {object_name}
            Output its coordinates in XML format <points x y>object</points>.
            {original_points_info}
           """
        
        # Print complete prompt
        print_complete_prompt(system_content, prompt, model_name, image_path)
        
        # Qwen2.5-VL uses a specific format for multimodal inputs
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        # Apply chat template
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Process the input
        inputs = processor(
            text=text,
            images=image,
            return_tensors="pt"
        ).to(model.device)

     

        # Generate output with torch.autocast for better performance
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.bfloat16):
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )
        
        # Decode the generated tokens
        content = processor.tokenizer.decode(output_ids[0][inputs.input_ids.size(1):], skip_special_tokens=True)
        
        logger.info(f"\n[DEBUG] Raw Qwen output for {object_name} in {image_path}:")
        logger.info(content)
        
        # First try to parse XML format: <points x y>object</points>
        import re
        # Print the raw response for debugging
        logger.info(f"[DEBUG] Looking for XML patterns in content: '{content[:200]}...'")
        
        # Try several patterns to match different possible XML formats
        xml_patterns = [
            # Format: <points x1="790" y1="46" alt="...">text</points> - with double or single quotes
            r'<points\s+x1=["\'"]?(\d+\.?\d*)["\'"]?\s+y1=["\'"]?(\d+\.?\d*)["\'"]?.*?>.*?</points>',
            # Format: <points x="123" y="456">text</points> - with double or single quotes
            r'<points\s+x=["\'"]?(\d+\.?\d*)["\'"]?\s+y=["\'"]?(\d+\.?\d*)["\'"]?.*?>.*?</points>',
            # Format: <points 123 456>text</points>
            r'<points\s+(\d+\.?\d*)\s+(\d+\.?\d*)>.*?</points>'
        ]
        
        points = []
        xml_matches_found = False
        
        # Try each pattern
        for pattern in xml_patterns:
            xml_matches = re.findall(pattern, content)
            if xml_matches:
                xml_matches_found = True
                logger.info(f"[DEBUG] XML points format detected with pattern '{pattern}': {xml_matches}")
                
                # Convert to standard point format
                for match in xml_matches:
                    points.append({"point": [float(match[0]), float(match[1])]})
        
        if xml_matches_found:
            logger.info(f"[DEBUG] Extracted points from XML: {points}")
            
            # If not counting category and more than one point was returned, limit to first point
            if category != "counting" and len(points) > 1:
                logger.info(f"[DEBUG] Multiple points detected but not counting category. Limiting to first point.")
                points = [points[0]]
            
            return points
        
        # If no XML matches found, try simple pattern matching for coordinate pairs
        if not xml_matches_found:
            logger.info("[DEBUG] No XML matches found, trying to extract any coordinate pairs")
            
            # Simple number pair extraction
            number_pairs = re.findall(r'(?:x|x1)[=:" ]+(\d+\.?\d*)[ ",]*(?:y|y1)[=:" ]+(\d+\.?\d*)', content)
            if number_pairs:
                logger.info(f"[DEBUG] Found coordinate pairs from attribute-like text: {number_pairs}")
                # Convert to points
                for x, y in number_pairs:
                    points.append({"point": [float(x), float(y)]})
                
                logger.info(f"[DEBUG] Extracted points: {points}")
                
                # If not counting category and more than one point was returned, limit to first point
                if category != "counting" and len(points) > 1:
                    points = [points[0]]
                
                return points
        
        # If we have points from any method, return them
        if points:
            return points
            
        # If no XML format found, try to extract JSON as a fallback
        # Extract JSON from the response
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end]
            logger.info(f"[DEBUG] Extracted JSON string: {json_str}")
            
            # Try to extract coordinates using regex first
            import re
            
            # First try to find point_2d format which returns pixel coordinates
            pixel_coords = re.findall(r'"point_2d":\s*\[(\d+\.?\d*),\s*(\d+\.?\d*)\]', json_str)
            if pixel_coords:
                logger.info(f"[DEBUG] Pixel coordinates extracted via 'point_2d': {pixel_coords}")
                # These are already in pixel coordinates
                points = [{"point": [float(x), float(y)]} for x, y in pixel_coords]
                logger.info(f"[DEBUG] Extracted points: {points}")
                
                # If not counting category and more than one point was returned, limit to first point
                if category != "counting" and len(points) > 1:
                    logger.info(f"[DEBUG] Multiple points detected but not counting category. Limiting to first point.")
                    points = [points[0]]
                
                return points
            
            # If no point_2d, try regular [x,y] format
            coords = re.findall(r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]', json_str)
            if coords:
                logger.info(f"[DEBUG] Coordinates extracted via regex: {coords}")
                # Convert to standard pixel format
                points = [{"point": [float(x), float(y)]} for x, y in coords]
                logger.info(f"[DEBUG] Extracted points: {points}")
                
                # If not counting category and more than one point was returned, limit to first point
                if category != "counting" and len(points) > 1:
                    logger.info(f"[DEBUG] Multiple points detected but not counting category. Limiting to first point.")
                    points = [points[0]]
                
                return points
            
            # If regex fails, try to parse as JSON
            try:
                # Try to fix common JSON format errors
                raw_points = json.loads(json_str)
                logger.info(f"[DEBUG] Raw points parsed from JSON: {raw_points}")
                
                # Handle different possible formats
                points = []
                if isinstance(raw_points, list):
                    for item in raw_points:
                        # Check for point_2d format (direct pixel coordinates)
                        if isinstance(item, dict) and "point_2d" in item:
                            if isinstance(item["point_2d"], list) and len(item["point_2d"]) == 2:
                                x, y = item["point_2d"]
                                points.append({"point": [float(x), float(y)]})
                        # Check for direct [x, y] format
                        elif isinstance(item, list) and len(item) == 2:
                            x, y = item
                            points.append({"point": [float(x), float(y)]})
                        # Check for {"point": [x, y]} format
                        elif isinstance(item, dict) and "point" in item:
                            if isinstance(item["point"], list) and len(item["point"]) == 2:
                                x, y = item["point"]
                                points.append({"point": [float(x), float(y)]})
                
                if points:
                    logger.info(f"[DEBUG] Points after parsing: {points}")
                    # If not counting category and more than one point was returned, limit to first point
                    if category != "counting" and len(points) > 1:
                        logger.info(f"[DEBUG] Multiple points detected but not counting category. Limiting to first point.")
                        points = [points[0]]
                    return points
                
                logger.info("[DEBUG] No valid points extracted from JSON")
                
                # As a last resort, check for any pair of numbers in the content
                number_pairs = re.findall(r'(\d+\.?\d*)\s*[,\s]\s*(\d+\.?\d*)', content)
                if number_pairs:
                    logger.info(f"[DEBUG] Found potential coordinate pairs: {number_pairs}")
                    # Use the first pair as a point
                    x, y = number_pairs[0]
                    points = [{"point": [float(x), float(y)]}]
                    return points
                
                return []
            except Exception as e:
                logger.info(f"[DEBUG] Error parsing coordinates from JSON: {e}")
                logger.info(f"Error parsing coordinates from {model_name} on {image_path}: {e}")
                
                # As a last resort, check for any pair of numbers in the content
                number_pairs = re.findall(r'(\d+\.?\d*)\s*[,\s]\s*(\d+\.?\d*)', content)
                if number_pairs:
                    logger.info(f"[DEBUG] Found potential coordinate pairs: {number_pairs}")
                    # Use the first pair as a point
                    x, y = number_pairs[0]
                    points = [{"point": [float(x), float(y)]}]
                    return points
                
                return []
        else:
            # If no JSON format detected, try to find any pair of numbers as coordinates
            logger.info(f"[DEBUG] No JSON brackets found in response. Looking for coordinate pairs.")
            number_pairs = re.findall(r'(\d+\.?\d*)\s*[,\s]\s*(\d+\.?\d*)', content)
            if number_pairs:
                logger.info(f"[DEBUG] Found potential coordinate pairs: {number_pairs}")
                # Convert to points
                points = [{"point": [float(x), float(y)]} for x, y in number_pairs]
                
                # If not counting category and more than one point was returned, limit to first point
                if category != "counting" and len(points) > 1:
                    points = [points[0]]
                
                return points
            
            logger.info(f"[DEBUG] Unable to extract coordinates from {model_name} on {image_path}")
            return []
    except Exception as e:
        logger.info(f"Error calling {model_name} on {image_path}: {e}")
        logger.info(f"Exception details: {str(e)}")
        # import traceback
        # traceback.print_exc()
        return []

def call_llava(image_path, object_name, model_name="llava-hf/llava-onevision-qwen2-7b-ov-hf", category=None):
    """Call LLaVA-OneVision model to get points for the specified object."""
    try:
        # Initialize model and processor if not already done
        model, processor = initialize_llava(model_name)
        
        # Load the image
        image = Image.open(image_path)
        img_width, img_height = image.size
        logger.info(f"[DEBUG] Image dimensions: {img_width}x{img_height}")
        
        # Get information about original points for steerable images
        original_points_info = get_original_points_info(image_path, category)
        
        # Define system content
        system_content = "You are a helpful assistant that can identify objects in images and provide their coordinates."
        
        # Prepare the prompt based on category
        if category == "counting":
            prompt = f"""
            {object_name}. 
            The image dimensions are width={img_width}px, height={img_height}px.{original_points_info}
            For each point, give EXACT PIXEL COORDINATES in [x, y] format, where x is horizontal (left-to-right) and y is vertical (top-to-bottom).
            Output format should be: [x, y], [x, y], etc. for multiple points.
            ONLY return the coordinates with no additional text or explanations.
            """
        else:
            prompt = f"""
            {object_name}.
            The image dimensions are width={img_width}px, height={img_height}px.{original_points_info}
            Give EXACT PIXEL COORDINATES in [x, y] format, where x is horizontal (left-to-right) and y is vertical (top-to-bottom).
            ONLY return the coordinates with no additional text or explanations.
            """
        
        # Print complete prompt
        print_complete_prompt(system_content, prompt, model_name, image_path)
        
        # Format the prompt correctly for LLaVA-OneVision HF version
        # Use the chat template approach from the HF model card
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        # Use the processor's apply_chat_template method
        logger.info("[DEBUG] Applying chat template")
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Process inputs with the processor
        logger.info("[DEBUG] Processing inputs")
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        
        # Generate output
        logger.info("[DEBUG] Generating output")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )
        
        # Decode the output
        logger.info("[DEBUG] Decoding output")
        content = processor.decode(output_ids[0][2:], skip_special_tokens=True)
        
        logger.info(f"\n[DEBUG] Raw LLaVA output for {object_name} in {image_path}:")
        logger.info(content)
        
        # Use a robust approach to extract coordinates from the response
        # First, try to extract using regex for [x, y] pattern
        import re
        
        # Look for coordinate pairs like [x, y]
        coord_pattern = r'\[(\d+\.?\d*),\s*(\d+\.?\d*)\]'
        coords = re.findall(coord_pattern, content)
        
        if coords:
            logger.info(f"[DEBUG] Coordinates extracted via regex: {coords}")
            
            # Convert to standard point format
            points = [{"point": [float(x), float(y)]} for x, y in coords]
            logger.info(f"[DEBUG] Points after extraction: {points}")
            
            # If not counting category and more than one point was returned, limit to first point
            if category != "counting" and len(points) > 1:
                logger.info(f"[DEBUG] Multiple points detected but not counting category. Limiting to first point.")
                points = [points[0]]
                logger.info(f"[DEBUG] Final point: {points}")
            
            return points
        else:
            # No coordinate pattern found, try other patterns
            
            # Look for numbers that might be coordinates (fallback)
            number_pairs = re.findall(r'(\d+\.?\d*)\s*,\s*(\d+\.?\d*)', content)
            if number_pairs:
                logger.info(f"[DEBUG] Found potential coordinate pairs: {number_pairs}")
                # Convert each pair to points
                points = [{"point": [float(x), float(y)]} for x, y in number_pairs]
                
                # Limit to first point if not counting
                if category != "counting" and len(points) > 1:
                    points = [points[0]]
                
                return points
            
            # Look for individual numbers as last resort
            numbers = re.findall(r'\b(\d+\.?\d*)\b', content)
            if len(numbers) >= 2:
                logger.info(f"[DEBUG] Found individual numbers: {numbers}")
                # Try to pair them up as x,y coordinates
                points = []
                for i in range(0, len(numbers)-1, 2):
                    x, y = float(numbers[i]), float(numbers[i+1])
                    points.append({"point": [x, y]})
                
                # Limit to first point if not counting
                if category != "counting" and len(points) > 1:
                    points = [points[0]]
                
                return points
            
            logger.info("[DEBUG] No valid coordinates found in response")
            return []
            
    except Exception as e:
        logger.info(f"Error calling {model_name} on {image_path}: {e}")
        logger.info(f"Exception details: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def load_mask(mask_path):
    """Load a binary mask from a PNG file."""
    try:
        # Load the mask image
        mask_img = Image.open(mask_path)
        
        # Convert to numpy array (values will be 0 for black and 255 for white)
        mask_array = np.array(mask_img)
        
        # Normalize to binary (True/False) mask
        # For grayscale, consider any value > 127 as True
        if len(mask_array.shape) == 2:
            binary_mask = mask_array > 127
        # For RGB, consider any channel > 127 as True (if any channel is white)
        elif len(mask_array.shape) == 3:
            binary_mask = np.any(mask_array > 127, axis=2)
        else:
            raise ValueError(f"Unexpected mask shape: {mask_array.shape}")
        
        return binary_mask
    except Exception as e:
        logger.info(f"Error loading mask {mask_path}: {e}")
        return None

def is_point_in_mask(point, mask, img_width, img_height):
    """Check if a point is inside the mask."""
    if mask is None or point is None:
        logger.info(f"[DEBUG MASK] Invalid mask or point: mask={mask is not None}, point={point}")
        return False
    
    # Unpack point (x, y format in pixel coordinates)
    x, y = point["point"]
    logger.info(f"[DEBUG MASK] Checking point x={x}, y={y} (pixel coordinates)")
    
    # Convert to integers for indexing
    pixel_x = int(x)
    pixel_y = int(y)
    logger.info(f"[DEBUG MASK] Pixel coordinates: x={pixel_x}, y={pixel_y}, image size: {img_width}x{img_height}")
    
    # Ensure coordinates are within image bounds
    if pixel_y < 0 or pixel_y >= img_height or pixel_x < 0 or pixel_x >= img_width:
        logger.info(f"[DEBUG MASK] Point outside image bounds: x={pixel_x}, y={pixel_y}")
        return False
    
    # Check if point falls within the mask
    is_in_mask = mask[pixel_y, pixel_x]
    logger.info(f"[DEBUG MASK] Point in mask: {is_in_mask}")
    return is_in_mask

def visualize_points_on_mask(image_path, mask, points, output_path, img_width, img_height):
    """Create a visualization of points overlaid on the mask and save it."""
    try:
        logger.info(f"\n[DEBUG VISUALIZATION] Creating visualization for {output_path}")
        logger.info(f"[DEBUG VISUALIZATION] Image dimensions: {img_width}x{img_height}")
        logger.info(f"[DEBUG VISUALIZATION] Points to visualize: {points}")
        
        # Create a visualization of the mask (white foreground, black background)
        mask_vis = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        mask_vis[mask] = 255  # White mask
        
        # Convert to PIL image
        mask_image = Image.fromarray(mask_vis, mode="RGB")
        
        # Draw points on the mask image
        draw = ImageDraw.Draw(mask_image)
        for point in points:
            # Unpack point (x, y format in pixel coordinates)
            x, y = point["point"]
            logger.info(f"[DEBUG VISUALIZATION] Processing point: x={x}, y={y} (pixel coordinates)")
            
            # Convert to integers for drawing
            pixel_x = int(x)
            pixel_y = int(y)
            logger.info(f"[DEBUG VISUALIZATION] Drawing at pixel coordinates: x={pixel_x}, y={pixel_y}")
            
            # Draw a cross at the point location (red for better visibility on white mask)
            point_size = max(5, min(img_width, img_height) // 100)  # Adaptive point size
            logger.info(f"[DEBUG VISUALIZATION] Drawing point with size {point_size} at ({pixel_x}, {pixel_y})")
            draw.line((pixel_x - point_size, pixel_y, pixel_x + point_size, pixel_y), fill=(255, 0, 0), width=3)
            draw.line((pixel_x, pixel_y - point_size, pixel_x, pixel_y + point_size), fill=(255, 0, 0), width=3)
            
            # Add a circle around the point
            draw.ellipse((pixel_x - point_size, pixel_y - point_size, 
                         pixel_x + point_size, pixel_y + point_size), 
                         outline=(255, 0, 0), width=2)
        
        # Save the image
        mask_image.save(output_path)
        logger.info(f"[DEBUG VISUALIZATION] Visualization saved to {output_path}")
        return True
    except Exception as e:
        logger.info(f"[DEBUG VISUALIZATION] Error creating visualization: {e}")
        logger.info(f"Error creating visualization: {e}")
        return False

def _evaluate_single_item(task_args):
    display_index, source_index, item, data_size, model_name, model_type, model_func, query_field, runtime_options, point_on_mask_dir = task_args
    image_filename = item["image_filename"]
    mask_filename = item["mask_filename"]
    
    messages = [f"Processing image {display_index+1}/{data_size}: {image_filename}"]

    def report(message):
        logger.info(message)
        messages.append(message)

    def build_item_result(success, detail):
        result = {
            "success": success,
            "messages": messages,
            "detail": detail,
        }
        if model_type.lower() in POINT_AGENT_MODEL_TYPES and item.get("enhanced_query"):
            result["data_index"] = source_index
            result["query_field"] = query_field
            result["enhanced_query"] = item["enhanced_query"]
        return result

    category = item.get("category", "")
    image_path = None
    if category:
        image_path = unicodedata.normalize('NFC', os.path.join(IMAGES_DIR, category, image_filename))
        mask_path = unicodedata.normalize('NFC', os.path.join(MASKS_DIR, category, mask_filename))
    else:
        image_path = unicodedata.normalize('NFC', os.path.join(IMAGES_DIR, image_filename))
        mask_path = unicodedata.normalize('NFC', os.path.join(MASKS_DIR, mask_filename))

    if image_path is None:
        report(f"Image not found: {image_filename} in category: {category}")
        return build_item_result(
            False,
            {
                "image": image_filename,
                "success": False,
                "reason": f"Image not found in category: {category}"
            },
        )

    if not os.path.exists(mask_path):
        report(f"Mask not found: {mask_filename}")
        return build_item_result(
            False,
            {
                "image": image_filename,
                "success": False,
                "reason": "Mask not found"
            },
        )

    query = item.get(query_field)
    raw_query = item.get("user_input")
    expected_count = item.get("count", 1)

    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        report(f"Error loading image {image_path}: {e}")
        return build_item_result(
            False,
            {
                "image": image_filename,
                "success": False,
                "reason": f"Error loading image: {e}"
            },
        )

    try:
        mask = load_mask(mask_path)
        if mask is None:
            raise ValueError("Failed to load mask")
    except Exception as e:
        report(f"Error loading mask {mask_path}: {e}")
        return build_item_result(
            False,
            {
                "image": image_filename,
                "success": False,
                "reason": f"Error loading mask: {e}"
            },
        )

    try:
        if model_type.lower() in POINT_AGENT_MODEL_TYPES:
            report(f"Testing {model_name} on image {image_filename}; point_agent will generate query from user_input: '{raw_query}'")
            runtime_options = _apply_sa2va_process_runtime(runtime_options)
            points = model_func(
                image_path,
                raw_query,
                model_name,
                category,
                item_ctx=item,
                runtime_options=runtime_options,
            )
        else:
            report(f"Testing {model_name} on image {image_filename} with query: '{query}'")
            points = model_func(image_path, query, model_name, category)

        if not points:
            report(f"No points returned for {image_filename}")
            return build_item_result(
                False,
                {
                    "image": image_filename,
                    "success": False,
                    "reason": "No points returned"
                },
            )

        category_vis_dir = point_on_mask_dir / category if category else point_on_mask_dir
        category_vis_dir.mkdir(exist_ok=True, parents=True)
        vis_filename = f"{Path(image_filename).stem}.jpg"
        vis_path = category_vis_dir / vis_filename
        vis_path = unicodedata.normalize('NFC', str(vis_path))
        visualize_points_on_mask(image_path, mask, points, vis_path, img_width, img_height)

        if category == "counting" and len(points) != expected_count:
            report(f"Count mismatch for {image_filename}: expected {expected_count}, got {len(points)}")
            return build_item_result(
                False,
                {
                    "image": image_filename,
                    "success": False,
                    "reason": f"Count mismatch: expected {expected_count}, got {len(points)}"
                },
            )

        points_in_mask = True
        for point in points:
            if not is_point_in_mask(point, mask, img_width, img_height):
                points_in_mask = False
                break

        if points_in_mask:
            report(f"Success for {image_filename}")
            return build_item_result(
                True,
                {
                    "image": image_filename,
                    "success": True,
                    "points_count": len(points),
                    "visualization": str(vis_path)
                },
            )

        report(f"Failure for {image_filename}: points outside mask")
        return build_item_result(
            False,
            {
                "image": image_filename,
                "success": False,
                "reason": "Points outside mask",
                "visualization": str(vis_path)
            },
        )
    except Exception as e:
        report(f"Error processing {image_filename} with {model_name}: {e}")
        return build_item_result(
            False,
            {
                "image": image_filename,
                "success": False,
                "reason": f"Processing error: {e}"
            },
        )

def _terminate_sa2va_processes(processes):
    for process in processes:
        if process.is_alive():
            logger.info(f"Terminating point_agent worker process pid={process.pid}")
            process.terminate()
    for process in processes:
        process.join(timeout=10)
    for process in processes:
        if process.is_alive():
            logger.info(f"Killing point_agent worker process pid={process.pid}")
            process.kill()
            process.join()


def _iter_sa2va_process_results(pending_items, cuda_devices, logs_dir):
    ctx = get_context("spawn")
    result_queue = ctx.Queue()
    buckets = {gpu_id: [] for gpu_id in cuda_devices}
    for index, task_args in enumerate(pending_items):
        buckets[cuda_devices[index % len(cuda_devices)]].append(task_args)

    processes = []
    pid_to_gpu = {}
    finished_gpus = set()
    original_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    interrupted = False
    try:
        for gpu_id, worker_items in buckets.items():
            if not worker_items:
                continue
            # spawn 子进程会重新 import 本文件；这里先收缩可见卡，保证 torch import 前只看到单卡。
            _set_cuda_visible_devices(str(gpu_id))
            process = ctx.Process(
                target=_run_sa2va_process_worker,
                args=(gpu_id, worker_items, result_queue, str(logs_dir)),
            )
            process.start()
            processes.append(process)
            pid_to_gpu[process.pid] = gpu_id
            logger.info(f"Started point_agent worker pid={process.pid} on physical GPU {gpu_id}")

        _set_cuda_visible_devices(original_cuda_visible_devices)
        while len(finished_gpus) < len(processes):
            try:
                message_type, payload = result_queue.get(timeout=1)
            except Empty:
                for process in processes:
                    gpu_id = pid_to_gpu.get(process.pid)
                    if gpu_id in finished_gpus:
                        continue
                    if not process.is_alive() and process.exitcode not in (None, 0):
                        finished_gpus.add(gpu_id)
                        logger.error(
                            f"point_agent worker pid={process.pid} on GPU {gpu_id} "
                            f"exited with code {process.exitcode}"
                        )
                continue

            if message_type == SA2VA_RESULT_MESSAGE:
                yield payload
            elif message_type == SA2VA_DONE_MESSAGE:
                finished_gpus.add(payload)

        for process in processes:
            process.join()
    except KeyboardInterrupt:
        interrupted = True
        logger.info("Evaluation interrupted; terminating point_agent worker processes.")
        raise
    finally:
        _set_cuda_visible_devices(original_cuda_visible_devices)
        if interrupted or any(process.is_alive() for process in processes):
            _terminate_sa2va_processes(processes)
        result_queue.close()
        result_queue.join_thread()


def _iter_evaluation_results(pending_items, model_name, max_workers, cuda_devices, logs_dir):
    if cuda_devices:
        yield from _iter_sa2va_process_results(pending_items, cuda_devices, logs_dir)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        yield from executor.map(_evaluate_single_item, pending_items)

def _record_item_result(item_result, results, item_count, results_file, progress_callback, res_data=None, res_path=None):
    results["total"] += 1
    item_count += 1

    if item_result["success"]:
        results["success"] += 1
    else:
        results["failure"] += 1
    results["details"].append(item_result["detail"])

    if progress_callback:
        for message in item_result["messages"]:
            progress_callback(message)

    if res_data is not None and res_path and item_result.get("enhanced_query"):
        data_index = item_result["data_index"]
        query_field = item_result["query_field"]
        # res 文件保留原始 user_input；增强后的文本单独落到 enhanced_query。
        res_data[data_index]["enhanced_query"] = item_result["enhanced_query"]
        if query_field != "user_input":
            res_data[data_index][query_field] = item_result["enhanced_query"]
        with open(res_path, "w") as f:
            json.dump(res_data, f, indent=2)
        logger.info(f"Enhanced query saved to {res_path} for {item_result['detail']['image']}")

    # 本评测经常跑很久，仍然保持每完成一张图就落盘。
    if results["total"] > 0:
        success_rate = results["success"] / results["total"] * 100
        logger.info(f"\nIntermediate results after {item_count} processed images:")
        logger.info(f"Total images: {results['total']}")
        logger.info(f"Successful predictions: {results['success']}")
        logger.info(f"Failed predictions: {results['failure']}")
        logger.info(f"Current success rate: {success_rate:.2f}%")

        if progress_callback:
            progress_callback(f"\nIntermediate results after {item_count} processed images:")
            progress_callback(f"Total images: {results['total']}")
            progress_callback(f"Successful predictions: {results['success']}")
            progress_callback(f"Failed predictions: {results['failure']}")
            progress_callback(f"Current success rate: {success_rate:.2f}%")

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Intermediate results saved to {results_file}")
    if progress_callback:
        progress_callback(f"Intermediate results saved to {results_file}")
    return item_count

def evaluate_model(
    model_name,
    model_type,
    progress_callback=None,
    resume=True,
    query_field="user_input",
    max_workers=4,
    result_suffix="",
    enhance_model="gemini-3.1-pro-preview",
    model_root="",
    start=0,
    end=-1,
):
    """Evaluate model performance on the dataset."""
    global logger

    # Select the appropriate model call function based on model type
    model_type_lower = model_type.lower()
    if model_type_lower == "qwen" and not model_name.startswith("Qwen/"):
        model_name = f"Qwen/{model_name}"
    elif model_type_lower == "llava" and not model_name.startswith("llava-hf/"):
        model_name = f"llava-hf/{model_name}"

    run_output_name = _build_run_output_name(model_type, model_name, result_suffix)
    output_paths = _prepare_run_output_paths(run_output_name)
    logger = get_logger(logs_dir=str(output_paths["logs_dir"]), log_name="logs")
    logger.info(f"Run output bucket: {run_output_name}")
    logger.info(f"Results dir: {output_paths['results_dir']}")
    logger.info(f"Point visualizations dir: {output_paths['point_on_mask_dir']}")
    logger.info(f"point_agent visualizations dir: {output_paths['sa2va_visualizations_dir']}")

    if model_type_lower == "openai":
        model_func = call_openai
    elif model_type_lower == "gemini":
        model_func = call_gemini
    elif model_type_lower == "qwen":
        model_func = call_qwen
    elif model_type_lower == "llava":
        model_func = call_llava
    elif model_type_lower == "claude":
        model_func = call_claude
    elif model_type_lower == "grok":
        model_func = call_grok
    elif model_type_lower in POINT_AGENT_MODEL_TYPES:
        model_func = call_sa2va_agent_justify_and_process_gemini_norefine
        cuda_devices = _parse_cuda_visible_devices()
        runtime_options = {
            "query_field": query_field,
            "model_root": model_root,
            "enhance_model": enhance_model,
            "sa2va_config": build_sa2va_norefine_config(
                model_name,
                query_field,
                model_root,
                visualizations_dir=str(output_paths["sa2va_visualizations_dir"]),
            ),
        }
        if cuda_devices:
            max_workers = len(cuda_devices)
            logger.info(f"point_agent uses one process per GPU from CUDA_VISIBLE_DEVICES: {cuda_devices}")
        else:
            logger.info("CUDA_VISIBLE_DEVICES is empty; point_agent uses single-process execution.")
            max_workers = 1
    else:
        logger.info(f"Unknown model type: {model_type}")
        if progress_callback:
            progress_callback(f"Unknown model type: {model_type}")
        return

    if model_type_lower not in POINT_AGENT_MODEL_TYPES:
        runtime_options = {}
        cuda_devices = []

    # Load data.json file
    try:
        data_path = "data/data.json"
        res_path = f"data/res_{model_name}_{result_suffix}.json"
        # with open("data/data.json", "r") as f:
        if not os.path.exists(res_path):
            shutil.copy(data_path, res_path)
        with open(res_path, "r") as f:
            all_data = json.load(f)
    except Exception as e:
        logger.info(f"Error loading data.json: {e}")
        if progress_callback:
            progress_callback(f"Error loading data.json: {e}")
        return

    # 小样本测试入口：end 为 -1 时从 start 一直处理到数据末尾。
    data = all_data[start:] if end == -1 else all_data[start:end]
    logger.info(f"Using data range start={start}, end={end}, selected {len(data)} items")
    
    # 结果文件固定放在本次 run bucket 内，目录名已经携带模型类型、模型名和 suffix。
    results_file = output_paths["results_dir"] / "results.json"
    
    # Initialize or load existing results
    if resume and os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
            logger.info(f"Resuming from existing results file with {results['success']} successes and {results['failure']} failures")
            if progress_callback:
                progress_callback(f"Resuming from existing results file with {results['success']} successes and {results['failure']} failures")
                
            # Get the list of already processed images
            processed_images = set(detail["image"] for detail in results["details"])
        except Exception as e:
            logger.info(f"Error loading existing results file: {e}")
            if progress_callback:
                progress_callback(f"Error loading existing results file: {e}")
            results = {
                "total": 0,
                "success": 0,
                "failure": 0,
                "details": []
            }
            processed_images = set()
    else:
        results = {
            "total": 0,
            "success": 0,
            "failure": 0,
            "details": []
        }
        processed_images = set()
    
    # Build work first so the thread pool only receives samples that need evaluation.
    item_count = 0
    pending_items = []
    for i, item in enumerate(data):
        if "mask_filename" not in item:
            continue

        image_filename = item["image_filename"]
        if image_filename in processed_images:
            logger.info(f"Skipping already processed image: {image_filename}")
            if progress_callback:
                progress_callback(f"Skipping already processed image: {image_filename}")
            continue
        source_index = start + i
        pending_items.append((
            i,
            source_index,
            item,
            len(data),
            model_name,
            model_type,
            model_func,
            query_field,
            runtime_options,
            output_paths["point_on_mask_dir"],
        ))

    worker_kind = "processes" if cuda_devices else "worker threads"
    logger.info(f"Evaluating {len(pending_items)} images with {max_workers} {worker_kind}")
    if progress_callback:
        progress_callback(f"Evaluating {len(pending_items)} images with {max_workers} {worker_kind}")

    # workers 只返回单条样本结果，父进程统一聚合和写 JSON，避免多进程并发写同一个结果文件。
    worker_results = _iter_evaluation_results(
        pending_items,
        model_name,
        max_workers,
        cuda_devices,
        output_paths["logs_dir"],
    )
    for item_result in tqdm(worker_results, total=len(pending_items), desc=f"Evaluating {model_name}"):
        item_count = _record_item_result(
            item_result,
            results,
            item_count,
            results_file,
            progress_callback,
            res_data=all_data,
            res_path=res_path,
        )
        
    # Calculate final success rate
    if results["total"] > 0:
        success_rate = results["success"] / results["total"] * 100
        logger.info(f"\nEvaluation results for {model_name}:")
        logger.info(f"Total images: {results['total']}")
        logger.info(f"Successful predictions: {results['success']}")
        logger.info(f"Failed predictions: {results['failure']}")
        logger.info(f"Success rate: {success_rate:.2f}%")
        logger.info(f"Visualizations saved to {output_paths['point_on_mask_dir']}/")
        
        if progress_callback:
            progress_callback(f"\nEvaluation results for {model_name}:")
            progress_callback(f"Total images: {results['total']}")
            progress_callback(f"Successful predictions: {results['success']}")
            progress_callback(f"Failed predictions: {results['failure']}")
            progress_callback(f"Success rate: {success_rate:.2f}%")
            progress_callback(f"Visualizations saved to {output_paths['point_on_mask_dir']}/")
        
        # Save final results
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Final results saved to {results_file}")
        if progress_callback:
            progress_callback(f"Final results saved to {results_file}")
    else:
        logger.info("No images were processed. Check that data.json contains valid entries and masks exist.")
        if progress_callback:
            progress_callback("No images were processed. Check that data.json contains valid entries and masks exist.")
    
    return results

def _add_common_cli_args(parser):
    parser.add_argument("--model", required=True, help="Model name to evaluate")
    parser.add_argument("--type", required=True, choices=["openai", "gemini", "molmo", "qwen", "llava", "claude", "grok", "point_agent"], help="Model type")
    parser.add_argument("--resume", action="store_true", help="Resume from previous evaluation state if available")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start evaluation from beginning")
    parser.add_argument("--query_field", default="user_input", help="Field name for the query in the JSON data")
    parser.add_argument("--workers", type=int, default=16, help="Number of worker threads for non-point_agent evaluation")
    parser.add_argument("--suffix", default="", help="Suffix appended to the per-model output bucket")
    parser.add_argument("--start", type=int, default=0, help="Start index for data slicing")
    parser.add_argument("--end", type=int, default=-1, help="End index for data slicing, -1 means all remaining items")
    parser.set_defaults(resume=True)


def _add_point_agent_cli_args(parser):
    point_agent_group = parser.add_argument_group("point_agent options")
    point_agent_group.add_argument("--model_root", default="", help="Local weight root. Empty means load --model as a HuggingFace repo id")
    point_agent_group.add_argument("--enhance_model", default="gemini-3.1-pro-preview", help="Model name used for per-item query enhancement, gemini series is recommended for best performance. ")


def _get_cli_type(argv):
    for index, arg in enumerate(argv):
        if arg == "--type" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--type="):
            return arg.split("=", 1)[1]
    return ""


def _build_cli_parser(argv):
    parser = argparse.ArgumentParser(description="Evaluate model performance on point prediction tasks")
    _add_common_cli_args(parser)
    # point_agent 专用参数只在 point_agent 模型类型下挂载，普通模型 CLI 保持干净。
    if _get_cli_type(argv).lower() in POINT_AGENT_MODEL_TYPES:
        _add_point_agent_cli_args(parser)
    return parser


def main():
    import sys

    parser = _build_cli_parser(sys.argv[1:])
    args = parser.parse_args()
    
    # Validate model name based on type
    valid_models = {
        "openai": OPENAI_MODELS,
        "gemini": GEMINI_MODELS,
        "molmo": MOLMO_MODELS,
        "qwen": QWEN_MODELS,
        "llava": LLAVA_MODELS,
        "claude": CLAUDE_MODELS,
        "grok": GROK_MODELS,
        "point_agent": POINT_AGENT_MODELS,
    }
    
    if args.type in valid_models and args.model not in valid_models[args.type]:
        logger.info(f"Warning: {args.model} is not in the list of known {args.type} models.")
        logger.info(f"Available {args.type} models: {', '.join(valid_models[args.type])}")
        return
                
    point_agent_kwargs = {}
    if args.type.lower() in POINT_AGENT_MODEL_TYPES:
        point_agent_kwargs = {
            "model_root": args.model_root,
            "enhance_model": args.enhance_model,
        }

    # Evaluate the specified model
    evaluate_model(
        args.model,
        args.type,
        resume=args.resume,
        query_field=args.query_field,
        max_workers=args.workers,
        result_suffix=args.suffix,
        start=args.start,
        end=args.end,
        **point_agent_kwargs,
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user.")
