#!/bin/bash
set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/InternVL3-78B/}"
META_JSON="${META_JSON:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/data_GRPO2_Q2/grpo2_val.json}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-2-eval/}"
RUN_TAG="${RUN_TAG:-internvl_q2_eval_$(date +%Y%m%d_%H%M%S)}"
INFER_BACKEND="${INFER_BACKEND:-transformers}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0}"

export MODEL_ID META_JSON OUTPUT_BASE_DIR RUN_TAG INFER_BACKEND MAX_BATCH_SIZE MAX_NEW_TOKENS TEMPERATURE

exec bash "${SCRIPT_DIR}/06_run_baseline_infer_q2_eval.sh" "$@"
