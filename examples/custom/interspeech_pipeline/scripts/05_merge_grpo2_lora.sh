#!/usr/bin/env bash
set -euo pipefail

# Sequential merge workflow:
# 1) BASE_MODEL_ID should be the GRPO1-merged (or multiturn-SFT-merged) model.
# 2) ADAPTER_PATH should be the GRPO2 LoRA checkpoint trained on top of BASE_MODEL_ID.
# 3) This script merges only the GRPO2 adapter into that fixed base.
# Result: final model = GRPO1 effects + GRPO2 effects.

BASE_MODEL_ID="${BASE_MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-1-ms-swift/lr1e-5_r8_a32_lora_merged}"
ADAPTER_PATH="${ADAPTER_PATH:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-2-ms-swift/grpo2_curr_fullReward_lora/checkpoint-360}"
OUTPUT_DIR="${OUTPUT_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-2-ms-swift/grpo2_curr_fullReward_lora_merged}"
CACHE_ROOT="${CACHE_ROOT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/SFT_Q1}"

mkdir -p "${CACHE_ROOT}/triton" "${CACHE_ROOT}/torch_extensions" "${CACHE_ROOT}/hf" \
  "${CACHE_ROOT}/xdg_cache" "${CACHE_ROOT}/modelscope" "${CACHE_ROOT}/datasets" \
  "${CACHE_ROOT}/vllm" "${CACHE_ROOT}/flashinfer"

export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export HF_HOME="${CACHE_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export DATASETS_CACHE="${CACHE_ROOT}/datasets"
export MODELSCOPE_CACHE="${CACHE_ROOT}/modelscope"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg_cache"
export VLLM_CONFIG_ROOT="${CACHE_ROOT}/vllm"
export VLLM_NO_USAGE_STATS=1
export FLASHINFER_WORKSPACE_BASE="${CACHE_ROOT}"
export FLASHINFER_WORKSPACE_DIR="${CACHE_ROOT}/flashinfer"
export FLASHINFER_JIT_CACHE_DIR="${FLASHINFER_WORKSPACE_DIR}"
unset TRANSFORMERS_CACHE

echo "[merge] BASE_MODEL_ID=${BASE_MODEL_ID}"
echo "[merge] ADAPTER_PATH=${ADAPTER_PATH}"
echo "[merge] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[merge] CACHE_ROOT=${CACHE_ROOT}"

swift export \
  --model "${BASE_MODEL_ID}" \
  --adapters "${ADAPTER_PATH}" \
  --load_args false \
  --merge_lora true \
  --output_dir "${OUTPUT_DIR}"
