#!/bin/bash
set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi

QWEN_VL_DIR="${QWEN_VL_DIR:-/scratch3/che489/Ha/interspeech/VLM/Qwen3-VL}"
MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT_ms_swift/stage1_query1_lora_Qwen2.5-VL-7B-Instruct/job20706766/v0-20260224-145751/checkpoint-774-merged}"
META_JSON="${META_JSON:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/data/stage1_query1_train_swift.json}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final__baseline__lightweight__VLM/}"
SHARD_COUNT="${SHARD_COUNT:-4}"
OVERWRITE="${OVERWRITE:-1}"
RUN_TAG="${RUN_TAG:-train_$(date +%Y%m%d_%H%M%S)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_TAG="$(basename "${MODEL_ID%/}" | tr -cs 'A-Za-z0-9._-' '_')"
RUN_DIR="${OUTPUT_BASE_DIR%/}/${MODEL_TAG}/${RUN_TAG}"
MERGED_DIR="${RUN_DIR}/merged_for_eval"

mkdir -p "${RUN_DIR}" "${MERGED_DIR}"

echo "[run] QWEN_VL_DIR=${QWEN_VL_DIR}"
echo "[run] MODEL_ID=${MODEL_ID}"
echo "[run] META_JSON=${META_JSON}"
echo "[run] RUN_DIR=${RUN_DIR}"

conda activate vllm
cd "${QWEN_VL_DIR}"
MODEL_ID="${MODEL_ID}" OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR}" META_JSON="${META_JSON}" RUN_TAG="${RUN_TAG}" SHARD_COUNT="${SHARD_COUNT}" OVERWRITE="${OVERWRITE}" \
  bash run_qwen_baseline_stage1_interactive.sh

for s in "${RUN_DIR}"/shard_*_of_${SHARD_COUNT}; do
  if [ -d "${s}" ]; then
    cp -rn "${s}"/* "${MERGED_DIR}/" || true
  fi
done

echo "[done] merged_dir=${MERGED_DIR}"
echo "[done] note=use merged_dir as input to build_swift_grpo_prompt2_dataset_from_p1_pred.py"
