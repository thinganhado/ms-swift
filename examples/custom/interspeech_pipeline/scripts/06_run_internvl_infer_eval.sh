#!/bin/bash
set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi

INTERNVL_DIR="${INTERNVL_DIR:-/scratch3/che489/Ha/interspeech/VLM/InternVL}"
MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/InternVL3-78B/}"
META_JSON="${META_JSON:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/data/stage1_query1_val_swift.json}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final___baseline__strong__InternVL/}"
SHARD_COUNT="${SHARD_COUNT:-4}"
OVERWRITE="${OVERWRITE:-1}"
RUN_TAG="${RUN_TAG:-eval_$(date +%Y%m%d_%H%M%S)}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-1500}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(cd "${SCRIPT_DIR}/../tools" && pwd)"

MODEL_TAG="$(basename "${MODEL_ID%/}" | tr -cs 'A-Za-z0-9._-' '_')"
RUN_DIR="${OUTPUT_BASE_DIR%/}/${MODEL_TAG}/${RUN_TAG}"
MERGED_DIR="${RUN_DIR}/merged_for_eval"
EVAL_JSON="${RUN_DIR}/eval_metrics.json"

mkdir -p "${RUN_DIR}" "${MERGED_DIR}"

echo "[run] INTERNVL_DIR=${INTERNVL_DIR}"
echo "[run] MODEL_ID=${MODEL_ID}"
echo "[run] META_JSON=${META_JSON}"
echo "[run] RUN_DIR=${RUN_DIR}"
echo "[run] VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}"

conda activate vllm
cd "${INTERNVL_DIR}"
MODEL_ID="${MODEL_ID}" \
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR}" \
META_JSON="${META_JSON}" \
RUN_TAG="${RUN_TAG}" \
SHARD_COUNT="${SHARD_COUNT}" \
OVERWRITE="${OVERWRITE}" \
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}" \
bash run_internvl_baseline_stage1_interactive.sh

for s in "${RUN_DIR}"/shard_*_of_${SHARD_COUNT}; do
  if [ -d "${s}" ]; then
    cp -rn "${s}"/* "${MERGED_DIR}/" || true
  fi
done

conda activate deepfake
python "${TOOLS_DIR}/eval_baseline_strongvlm_localization.py" \
  --model-dir "${MERGED_DIR}" \
  --save-json "${EVAL_JSON}"

echo "[done] merged_dir=${MERGED_DIR}"
echo "[done] eval_json=${EVAL_JSON}"
