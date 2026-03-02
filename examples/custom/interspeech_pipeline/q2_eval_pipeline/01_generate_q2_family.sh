#!/bin/bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${PIPELINE_DIR}/scripts"

FAMILY="${FAMILY:-}"
FINALIZE_SHARDS=0
export FINALIZE_SHARDS

if [ -z "${FAMILY}" ]; then
  cat >&2 <<'EOF'
[error] FAMILY is required.
Set FAMILY to one of:
  - qwen (or baseline)
  - internvl
  - llava (or llavaonevision)
  - deepseek (or deepseekvl2)
EOF
  exit 1
fi

case "${FAMILY}" in
  qwen|baseline)
    TARGET_SCRIPT="${SCRIPTS_DIR}/06_run_baseline_infer_q2_eval.sh"
    ;;
  internvl)
    TARGET_SCRIPT="${SCRIPTS_DIR}/06_run_internvl_infer_q2_eval.sh"
    ;;
  llava|llavaonevision)
    TARGET_SCRIPT="${SCRIPTS_DIR}/06_run_llavaonevision_infer_q2_eval.sh"
    ;;
  deepseek|deepseekvl2)
    TARGET_SCRIPT="${SCRIPTS_DIR}/06_run_deepseekvl2_infer_q2_eval.sh"
    ;;
  *)
    echo "[error] unsupported FAMILY='${FAMILY}'" >&2
    exit 1
    ;;
esac

echo "[run] FAMILY=${FAMILY}"
echo "[run] TARGET_SCRIPT=${TARGET_SCRIPT}"
echo "[run] FINALIZE_SHARDS=${FINALIZE_SHARDS}"
echo "[run] This generation stage only writes infer_result.jsonl outputs."

bash "${TARGET_SCRIPT}" "$@"
