#!/bin/bash
set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_ID="${MODEL_ID:-lmms-lab/LLaVA-OneVision-1.5-8B-Instruct}"
META_JSON="${META_JSON:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/data_GRPO2_Q2/grpo2_val.json}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-2-eval/}"
RUN_TAG="${RUN_TAG:-llavaov_q2_eval_$(date +%Y%m%d_%H%M%S)}"
INFER_BACKEND="${INFER_BACKEND:-vllm}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0}"
CACHE_ROOT="${CACHE_ROOT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/SFT_Q1}"
VLLM_TP="${VLLM_TP:-4}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-1500}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
VLLM_LIMIT_MM_PER_PROMPT="${VLLM_LIMIT_MM_PER_PROMPT:-}"

export MODEL_ID META_JSON OUTPUT_BASE_DIR RUN_TAG INFER_BACKEND MAX_BATCH_SIZE MAX_NEW_TOKENS TEMPERATURE CACHE_ROOT
export VLLM_TP VLLM_GPU_MEMORY_UTILIZATION VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS
export VLLM_ENFORCE_EAGER VLLM_DISABLE_CUSTOM_ALL_REDUCE VLLM_LIMIT_MM_PER_PROMPT

exec bash "${SCRIPT_DIR}/06_run_baseline_infer_q2_eval.sh" "$@"
