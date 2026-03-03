#!/bin/bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${PIPELINE_DIR}/scripts"

RUN_DIR="${RUN_DIR:-}"
META_JSON="${META_JSON:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/data_GRPO2_Q2/grpo2_val_sft_q2_swift.json}"
RAW_RESULT_JSONL="${RAW_RESULT_JSONL:-}"
PARSED_JSONL="${PARSED_JSONL:-}"
VERIFIER_OUTPUT_DIR="${VERIFIER_OUTPUT_DIR:-}"
VERIFIER_SYSTEM_FILE="${VERIFIER_SYSTEM_FILE:-}"
VERIFIER_USER_FILE="${VERIFIER_USER_FILE:-}"
QWEN3_DIR="${QWEN3_DIR:-/scratch3/che489/Ha/interspeech/LLM/Qwen3}"
VERIFIER_MODEL_ID="${VERIFIER_MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/LLM/Qwen3-30B-A3B-Instruct-2507/}"
VERIFIER_GT_CSV="${VERIFIER_GT_CSV:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final__En/union_all3_only.csv}"
VERIFIER_BATCH_SIZE="${VERIFIER_BATCH_SIZE:-8}"
VERIFIER_TENSOR_PARALLEL_SIZE="${VERIFIER_TENSOR_PARALLEL_SIZE:-4}"
VERIFIER_GPU_MEMORY_UTILIZATION="${VERIFIER_GPU_MEMORY_UTILIZATION:-0.85}"
VERIFIER_OVERWRITE="${VERIFIER_OVERWRITE:-0}"

if [ -z "${RUN_DIR}" ]; then
  echo "[error] RUN_DIR is required." >&2
  exit 1
fi

RAW_RESULT_JSONL="${RAW_RESULT_JSONL:-${RUN_DIR}/infer_result.jsonl}"
PARSED_JSONL="${PARSED_JSONL:-${RUN_DIR}/en_only_for_verifier_parsed.jsonl}"
VERIFIER_OUTPUT_DIR="${VERIFIER_OUTPUT_DIR:-${RUN_DIR}/verifier_from_parsed}"
VERIFIER_SYSTEM_FILE="${VERIFIER_SYSTEM_FILE:-${RUN_DIR}/q2_verifier_system.txt}"
VERIFIER_USER_FILE="${VERIFIER_USER_FILE:-${RUN_DIR}/q2_verifier_user.txt}"

mkdir -p "${RUN_DIR}" "${VERIFIER_OUTPUT_DIR}"

if [ ! -f "${RAW_RESULT_JSONL}" ]; then
  echo "[error] raw generation output not found: ${RAW_RESULT_JSONL}" >&2
  exit 1
fi

echo "[run] RUN_DIR=${RUN_DIR}"
echo "[run] META_JSON=${META_JSON}"
echo "[run] RAW_RESULT_JSONL=${RAW_RESULT_JSONL}"
echo "[run] PARSED_JSONL=${PARSED_JSONL}"
echo "[run] VERIFIER_OUTPUT_DIR=${VERIFIER_OUTPUT_DIR}"

python "${SCRIPTS_DIR}/07_build_q2_verifier_input.py" \
  --meta-json "${META_JSON}" \
  --raw-result-jsonl "${RAW_RESULT_JSONL}" \
  --output-jsonl "${PARSED_JSONL}"

if [ ! -s "${PARSED_JSONL}" ]; then
  echo "[error] parsed verifier input is empty: ${PARSED_JSONL}" >&2
  exit 1
fi

cat > "${VERIFIER_SYSTEM_FILE}" <<'EOF'
Infer the following features from <Explanation> and output:
  - Time: speech or non-speech
  - Frequency: low, mid, or high
  - Phonetic: consonant, vowel, or unvoiced
If unsure about a field, output "ambiguous" for that field.

OUTPUT FORMAT (must follow exactly):

{
  "time": "speech/non-speech/ambiguous",
  "frequency": "low/mid/high/ambiguous",
  "phonetic": "consonant/vowel/unvoiced/ambiguous"
}
EOF

cat > "${VERIFIER_USER_FILE}" <<'EOF'
This is an artifact description for a spectrogram region: {description}. Please strictly follow the instructions to extract the information.
EOF

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi

conda activate vllm

cd "${QWEN3_DIR}"
SCRIPT_DIR_OVERRIDE="${QWEN3_DIR}" \
MODEL_ID="${VERIFIER_MODEL_ID}" \
INPUT_MODEL_FOLDER="${RUN_DIR}" \
INPUT_JSONL="${PARSED_JSONL}" \
GT_CSV="${VERIFIER_GT_CSV}" \
SYSTEM_FILE="${VERIFIER_SYSTEM_FILE}" \
USER_TEMPLATE_FILE="${VERIFIER_USER_FILE}" \
SHARD_COUNT=1 \
SHARD_ID=0 \
TENSOR_PARALLEL_SIZE="${VERIFIER_TENSOR_PARALLEL_SIZE}" \
OVERWRITE="${VERIFIER_OVERWRITE}" \
OUTPUT_DIR="${VERIFIER_OUTPUT_DIR}" \
BATCH_SIZE="${VERIFIER_BATCH_SIZE}" \
VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
VLLM_ENFORCE_EAGER=1 \
VLLM_GPU_MEMORY_UTILIZATION="${VERIFIER_GPU_MEMORY_UTILIZATION}" \
bash run_qwen_region_full.sbatch

echo "[done] parsed_jsonl=${PARSED_JSONL}"
echo "[done] verifier_output_dir=${VERIFIER_OUTPUT_DIR}"
