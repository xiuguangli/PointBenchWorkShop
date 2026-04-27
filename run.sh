#!/usr/bin/env bash
set -euo pipefail

# Support the --gpu parameter. Accepts --gpu N, --gpu=N,
# or a comma-separated list of GPUs, e.g. --gpu 0,1,2
GPU=0
if [[ $# -ge 1 ]]; then
  if [[ "$1" == "--gpu" ]]; then
    if [[ $# -lt 2 ]]; then
      echo "Missing value for --gpu" >&2
      exit 1
    fi
    GPU="$2"
  elif [[ "$1" == --gpu=* ]]; then
    GPU="${1#--gpu=}"
  fi
fi

# Strip spaces to support inputs like "1, 2,3"
GPU="${GPU// /}"

# Validate format: only digits and commas are allowed, e.g. 0 or 0,1,2
if ! [[ "$GPU" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Invalid --gpu value: $GPU" >&2
  echo "Expected single id (e.g. 1) or comma-separated list (e.g. 0,1,2)" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"

uv run python model_evaluator.py \
  --model point_agent \
  --type point_agent \
  --start 0 \
  --end 1 \
  --suffix exp
