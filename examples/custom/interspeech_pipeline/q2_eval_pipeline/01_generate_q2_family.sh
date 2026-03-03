#!/bin/bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${PIPELINE_DIR}/scripts"

FAMILY="${FAMILY:-}"
PRINT_OUTPUTS="${PRINT_OUTPUTS:-0}"
PRINT_OUTPUT_LIMIT="${PRINT_OUTPUT_LIMIT:-10}"
PRINT_OUTPUT_CHARS="${PRINT_OUTPUT_CHARS:-1200}"
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
echo "[run] PRINT_OUTPUTS=${PRINT_OUTPUTS}"

bash "${TARGET_SCRIPT}" "$@"

if [ "${PRINT_OUTPUTS}" = "1" ]; then
  MODEL_ID="${MODEL_ID:-}"
  OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-}"
  RUN_TAG="${RUN_TAG:-}"
  SHARD_COUNT="${SHARD_COUNT:-1}"
  SHARD_ID="${SHARD_ID:-0}"
  MODEL_TAG="$(basename "${MODEL_ID%/}" | tr -cs 'A-Za-z0-9._-' '_')"
  BASE_RUN_DIR="${OUTPUT_BASE_DIR%/}/${MODEL_TAG}/${RUN_TAG}"
  RUN_DIR="${BASE_RUN_DIR}"
  if [ "${SHARD_COUNT}" -gt 1 ] && [ "${FINALIZE_SHARDS}" != "1" ]; then
    RUN_DIR="${BASE_RUN_DIR}/shard_${SHARD_ID}_of_${SHARD_COUNT}"
  fi
  RAW_RESULT_JSONL="${RUN_DIR}/infer_result.jsonl"
  if [ ! -f "${RAW_RESULT_JSONL}" ]; then
    echo "[warn] PRINT_OUTPUTS requested but result file not found: ${RAW_RESULT_JSONL}" >&2
    exit 0
  fi
  python - <<'PY' "${RAW_RESULT_JSONL}" "${PRINT_OUTPUT_LIMIT}" "${PRINT_OUTPUT_CHARS}"
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
limit = max(1, int(sys.argv[2]))
max_chars = max(1, int(sys.argv[3]))

def get_response(obj):
    if isinstance(obj.get("response"), str):
        return obj["response"]
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        choice0 = choices[0]
        if isinstance(choice0, dict):
            msg = choice0.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
    return ""

print(f"[print] source={src}")
shown = 0
with src.open("r", encoding="utf-8") as f:
    for line in f:
        if shown >= limit:
            break
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        resp = get_response(obj)
        sample_id = obj.get("sample_id")
        text = resp if isinstance(resp, str) else str(resp)
        text = text[:max_chars]
        print(f"\n--- output {shown} ---")
        print(f"sample_id: {sample_id}")
        print(text)
        shown += 1
print(f"[print] shown={shown} limit={limit}")
PY
fi
