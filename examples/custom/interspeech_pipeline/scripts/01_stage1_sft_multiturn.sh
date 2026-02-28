#!/usr/bin/env bash
set -euo pipefail

# ===== User-settable =====
MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/Qwen3-VL-8B-Instruct/}"
SFT_JSON_SWIFT="${SFT_JSON_SWIFT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/SFT_2turn/stage1_multiturn_train_swift.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT_ms_swift/stage1_mt_lora_Qwen3-VL-8B-Instruct}"
SFT_DEBUG_DATA="${SFT_DEBUG_DATA:-1}"
SFT_DEBUG_SAMPLES="${SFT_DEBUG_SAMPLES:-3}"
CACHE_ROOT="${CACHE_ROOT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/SFT_Q1}"
TMPDIR_BASE="${TMPDIR_BASE:-/tmp/${USER}_mswift_mt}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SAVE_STEPS="${SAVE_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

mkdir -p "${CACHE_ROOT}/triton" "${CACHE_ROOT}/torch_extensions" "${CACHE_ROOT}/hf" \
  "${CACHE_ROOT}/xdg_cache" "${CACHE_ROOT}/modelscope" "${CACHE_ROOT}/datasets" "${TMPDIR_BASE}"

export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export HF_HOME="${CACHE_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export DATASETS_CACHE="${CACHE_ROOT}/datasets"
export MODELSCOPE_CACHE="${CACHE_ROOT}/modelscope"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg_cache"
export TMPDIR="${TMPDIR_BASE}"
unset TRANSFORMERS_CACHE

if [[ ! -f "${SFT_JSON_SWIFT}" ]]; then
  echo "ERROR: dataset not found: ${SFT_JSON_SWIFT}"
  echo "Build it first with examples/custom/interspeech_pipeline/tools/build_swift_sft_multiturn_dataset.py"
  exit 1
fi

if [[ "${SFT_DEBUG_DATA}" == "1" ]]; then
  python - "${SFT_JSON_SWIFT}" "${SFT_DEBUG_SAMPLES}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1]).expanduser().resolve()
n_show = max(1, int(sys.argv[2]))
data = json.loads(src.read_text(encoding="utf-8"))
print(f"[mt_dbg] rows={len(data)}")

def text_only(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")).strip())
        return " ".join([p for p in parts if p]).strip()
    return str(content or "").strip()

def has_image(content):
    if not isinstance(content, list):
        return False
    for item in content:
        if isinstance(item, dict) and item.get("type") == "image":
            return True
    return False

def extract_user_texts(messages):
    out = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        out.append(text_only(msg.get("content", "")))
    return out

for i, row in enumerate(data[:n_show]):
    sid = row.get("sample_id", f"row_{i}")
    print(f"[mt_dbg] sample={i} sample_id={sid}")
    user_texts = extract_user_texts(row.get("messages", []))
    if len(user_texts) >= 1:
        q1 = user_texts[0].replace("\n", " ")
        if len(q1) > 420:
            q1 = q1[:420] + "..."
        print(f"[mt_dbg]   embedded_q1_system_plus_user={q1!r}")
    if len(user_texts) >= 2:
        q2 = user_texts[1].replace("\n", " ")
        if len(q2) > 420:
            q2 = q2[:420] + "..."
        print(f"[mt_dbg]   embedded_q2_system_plus_user={q2!r}")
    for j, msg in enumerate(row.get("messages", [])):
        role = msg.get("role", "")
        content = msg.get("content", "")
        txt = text_only(content).replace("\n", " ")
        if len(txt) > 220:
            txt = txt[:220] + "..."
        print(f"[mt_dbg]   turn={j} role={role} has_image={has_image(content)} text={txt!r}")
PY
fi

# ===== Train =====
NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift sft \
  --model "${MODEL_ID}" \
  --dataset "${SFT_JSON_SWIFT}" \
  --tuner_type lora \
  --torch_dtype "${TORCH_DTYPE}" \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --freeze_llm true \
  --freeze_vit false \
  --freeze_aligner false \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type cosine \
  --max_pixels 524288 \
  --deepspeed zero2 \
  --logging_steps "${LOGGING_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --output_dir "${OUTPUT_DIR}" \
  --report_to tensorboard
