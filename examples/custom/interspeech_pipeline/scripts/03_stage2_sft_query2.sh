#!/usr/bin/env bash
set -euo pipefail

# ===== User-settable =====
# Base model should be the GRPO1-merged checkpoint so Q1 behavior is baked into the base.
MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/GRPO-1/base_q1_merged/}"
Q2_JSON_IN="${Q2_JSON_IN:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/data_GRPO2_Q2/grpo2_train.json}"
Q2_JSON_SWIFT="${Q2_JSON_SWIFT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/data_GRPO2_Q2/grpo2_train_sft_q2_swift.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/SFT_Q2/stage2_q2_lora}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are an expert in deepfake speech spectrogram forensics.

You are given a spectrogram and transcript. You have already selected exactly 3 region IDs, in order: ID1, ID2, ID3.
For each ID, infer timing information (T), frequency band (F), phonetic category (P), and a visual description of the artifact and the likely audio impact implied by the artificial signs (En).

OUTPUT FORMAT (must follow exactly):
(T1=..., F1=..., P1=..., En1=\"...\"); (T2=..., F2=..., P2=..., En2=\"...\"); (T3=..., F3=..., P3=..., En3=\"...\")

Field definitions:
- Fields ending in 1, 2, and 3 correspond to ID1, ID2, and ID3 respectively.
- T: one of {speech, non-speech}
- F: one of {low, mid, high}
- P: one of {consonant, vowel, unvoiced}
- En: textual description, must be enclosed in double quotes.

Do not output any other text outside the three tuples.}"
SFT_DEBUG_DATA="${SFT_DEBUG_DATA:-1}"
SFT_DEBUG_SAMPLES="${SFT_DEBUG_SAMPLES:-3}"

CACHE_ROOT="${CACHE_ROOT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/SFT_Q1}"
TMPDIR_BASE="${TMPDIR_BASE:-/tmp/${USER}_mswift_q2}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
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

mkdir -p "$(dirname "${Q2_JSON_SWIFT}")" \
  "${CACHE_ROOT}/triton" "${CACHE_ROOT}/torch_extensions" "${CACHE_ROOT}/hf" \
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

if [[ ! -f "${Q2_JSON_IN}" ]]; then
  echo "ERROR: Q2 dataset not found: ${Q2_JSON_IN}"
  exit 1
fi

# ===== Convert Q2-only GT data into a 2-turn SFT dataset =====
python - "${Q2_JSON_IN}" "${Q2_JSON_SWIFT}" "${SYSTEM_PROMPT}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1]).expanduser().resolve()
dst = Path(sys.argv[2]).expanduser().resolve()
system_prompt = str(sys.argv[3] or "").strip()
data = json.loads(src.read_text(encoding="utf-8"))

def text_blocks_from_content(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content.strip()}] if content.strip() else []
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    txt = str(item.get("text", "")).strip()
                    if txt:
                        out.append({"type": "text", "text": txt})
                elif item.get("type") == "image":
                    out.append({"type": "image", "image": item.get("image")})
        return out
    return []

def prepend_system_to_user(msg):
    content = msg.get("content", [])
    blocks = text_blocks_from_content(content)
    user_text = " ".join(
        str(b.get("text", "")).strip()
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "text"
    ).strip()
    new_text = system_prompt if not user_text else f"{system_prompt}\n\n{user_text}"
    new_blocks = []
    inserted = False
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "image":
            new_blocks.append(b)
        elif not inserted:
            new_blocks.append({"type": "text", "text": new_text})
            inserted = True
    if not inserted:
        new_blocks.append({"type": "text", "text": new_text})
    return {"role": "user", "content": new_blocks}

out = []
for row in data:
    msgs = row.get("messages") or row.get("conversations") or []
    if not isinstance(msgs, list) or not msgs:
        continue
    gt = str(row.get("gt_prompt2", "")).strip()
    if not gt:
        continue
    user = prepend_system_to_user(msgs[0] if isinstance(msgs[0], dict) else {"role": "user", "content": []})
    assistant = {"role": "assistant", "content": [{"type": "text", "text": gt}]}
    rec = {
        "messages": [user, assistant],
        "sample_id": str(row.get("sample_id", "")).strip(),
        "prompt1_output": str(row.get("prompt1_output", "")).strip(),
        "gt_prompt2": gt,
    }
    if "transcript" in row:
        rec["transcript"] = row["transcript"]
    out.append(rec)

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"saved: {dst} (n={len(out)})")
PY

if [[ "${SFT_DEBUG_DATA}" == "1" ]]; then
  python - "${Q2_JSON_SWIFT}" "${SFT_DEBUG_SAMPLES}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1]).expanduser().resolve()
n_show = max(1, int(sys.argv[2]))
data = json.loads(src.read_text(encoding="utf-8"))
print(f"[q2_dbg] rows={len(data)}")

def text_only(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            str(item.get("text", "")).strip()
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
    return str(content or "").strip()

def has_image(content):
    return isinstance(content, list) and any(
        isinstance(item, dict) and item.get("type") == "image" for item in content
    )

for i, row in enumerate(data[:n_show]):
    print(f"[q2_dbg] sample={i} sample_id={row.get('sample_id', f'row_{i}')}")
    for j, msg in enumerate(row.get("messages", [])):
        txt = text_only(msg.get("content", "")).replace("\n", " ")
        if len(txt) > 220:
            txt = txt[:220] + "..."
        print(f"[q2_dbg]   turn={j} role={msg.get('role')} has_image={has_image(msg.get('content'))} text={txt!r}")
PY
fi

# ===== Train Q2-only SFT adapter =====
NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift sft \
  --model "${MODEL_ID}" \
  --dataset "${Q2_JSON_SWIFT}" \
  --tuner_type lora \
  --torch_dtype "${TORCH_DTYPE}" \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --freeze_llm true \
  --freeze_vit true \
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
