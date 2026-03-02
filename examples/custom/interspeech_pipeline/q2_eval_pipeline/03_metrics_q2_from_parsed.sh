#!/bin/bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${PIPELINE_DIR}/scripts"

RUN_DIR="${RUN_DIR:-}"
META_JSON="${META_JSON:-}"
RAW_RESULT_JSONL="${RAW_RESULT_JSONL:-}"
VERIFIER_OUTPUT_DIR="${VERIFIER_OUTPUT_DIR:-}"
EVAL_JSON="${EVAL_JSON:-}"

if [ -z "${RUN_DIR}" ]; then
  echo "[error] RUN_DIR is required." >&2
  exit 1
fi

if [ -z "${META_JSON}" ]; then
  echo "[error] META_JSON is required." >&2
  exit 1
fi

RAW_RESULT_JSONL="${RAW_RESULT_JSONL:-${RUN_DIR}/infer_result.jsonl}"
VERIFIER_OUTPUT_DIR="${VERIFIER_OUTPUT_DIR:-${RUN_DIR}/verifier_from_parsed}"
EVAL_JSON="${EVAL_JSON:-${RUN_DIR}/q2_eval_metrics_from_parsed.json}"

if [ ! -f "${RAW_RESULT_JSONL}" ]; then
  echo "[error] raw generation output not found: ${RAW_RESULT_JSONL}" >&2
  exit 1
fi

if [ ! -d "${VERIFIER_OUTPUT_DIR}" ]; then
  echo "[error] verifier output directory not found: ${VERIFIER_OUTPUT_DIR}" >&2
  exit 1
fi

echo "[run] RUN_DIR=${RUN_DIR}"
echo "[run] META_JSON=${META_JSON}"
echo "[run] RAW_RESULT_JSONL=${RAW_RESULT_JSONL}"
echo "[run] VERIFIER_OUTPUT_DIR=${VERIFIER_OUTPUT_DIR}"
echo "[run] EVAL_JSON=${EVAL_JSON}"

python "${SCRIPTS_DIR}/08_run_q2_eval_from_parsed.py" \
  --meta-json "${META_JSON}" \
  --raw-result-jsonl "${RAW_RESULT_JSONL}" \
  --verifier-output-dir "${VERIFIER_OUTPUT_DIR}" \
  --output-json "${EVAL_JSON}"

echo "[done] eval_json=${EVAL_JSON}"
