#!/usr/bin/env bash
set -euo pipefail

# Support the --gpu parameter. Accepts --gpu N, --gpu=N,
# or a comma-separated list of GPUs, e.g. --gpu 0,1,2
GPU=0
MODEL_ROOT=""
EXTRA_ARGS=()
# Ensure `uv` is installed before proceeding. The script uses `uv run ...` below.
if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' is not installed or not in PATH. Please install 'uv' (e.g. 'pip install uv' or via your virtualenv) and retry." >&2
  exit 1
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --gpu" >&2
        exit 1
      fi
      GPU="$2"
      shift 2
      ;;
    --gpu=*)
      GPU="${1#--gpu=}"
      shift
      ;;
    --model_root)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --model_root" >&2
        exit 1
      fi
      MODEL_ROOT="$2"
      shift 2
      ;;
    --model_root=*)
      MODEL_ROOT="${1#--model_root=}"
      shift
      ;;
    --workers)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --workers" >&2
        exit 1
      fi
      echo "Ignoring --workers=$2 because this pipeline now auto-scales from visible GPUs." >&2
      shift 2
      ;;
    --workers=*)
      echo "Ignoring $1 because this pipeline now auto-scales from visible GPUs." >&2
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

# Strip spaces to support inputs like "1, 2,3"
GPU="${GPU// /}"

# Validate format: only digits and commas are allowed, e.g. 0 or 0,1,2
if ! [[ "$GPU" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Invalid --gpu value: $GPU" >&2
  echo "Expected single id (e.g. 1) or comma-separated list (e.g. 0,1,2)" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"

CMD=(
  uv run python model_evaluator.py
  --model 1-2
  --type molmo2_guidance_dualquery_refpoint_hybrid_gemini_judge
  --query_field enhanced_query
  --enhance_model gemini-3.1-pro-preview
  --rewrite_model gemini-3.5-flash
  --max_tokens 256
  --start 0
  --end -1
  --suffix exp_flash_rewrite
)

if [[ -n "$MODEL_ROOT" ]]; then
  CMD+=(--model_root "$MODEL_ROOT")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

printf 'Launching PointBench with CUDA_VISIBLE_DEVICES=%s, model_root=%s\n' \
  "$CUDA_VISIBLE_DEVICES" "${MODEL_ROOT:-<huggingface-auto-download>}"

"${CMD[@]}"
