#!/bin/bash
#SBATCH --account=OD-202968
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --time=31:00:00
#SBATCH --output=/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/query1/logs/%x_%j.out
#SBATCH --error=/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/query1/logs/%x_%j.err

set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
conda activate deepfake

cd /scratch3/che489/Ha/interspeech/training/ms-swift

MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/Qwen3-VL-8B-Instruct/}"
SFT_JSON_SWIFT="${SFT_JSON_SWIFT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/data/stage1_query1_train_swift.json}"
SFT_VAL_JSON_SWIFT="${SFT_VAL_JSON_SWIFT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/data/stage1_query1_val_swift.json}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-10}"
OUTPUT_DIR="${OUTPUT_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT_ms_swift/stage1_query1_lora_Qwen3-VL-8B-Instruct}"
SYSTEM_PROMPT="As an expert in deepfake speech spectrogram forensics, you can detect regions containing deepfake artifacts by analysing spectrogram segments. Return only the JSON array of your three chosen region IDs."
SAVE_STEPS="${SAVE_STEPS:-200}"
MAX_LENGTH="${MAX_LENGTH:-768}"
DISABLE_EVAL="${DISABLE_EVAL:-0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-10}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
CACHE_ROOT="${CACHE_ROOT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/SFT_Q1}"
TMPDIR_BASE="${TMPDIR_BASE:-/tmp/${USER}_mswift_q1}"
SFT_DEBUG_GT="${SFT_DEBUG_GT:-1}"
SFT_DEBUG_SAMPLES="${SFT_DEBUG_SAMPLES:-8}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
TRAINING_CONFIG_JSON="${TRAINING_CONFIG_JSON:-}"
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-${OUTPUT_DIR}}"
RESUME_SEARCH_DIR="${RESUME_SEARCH_DIR:-${OUTPUT_DIR_BASE}}"
RUN_TAG="${RUN_TAG:-job${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}"
RUN_OUTPUT_DIR="${RUN_OUTPUT_DIR:-${OUTPUT_DIR_BASE%/}/${RUN_TAG}}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

if [ -n "${TRAINING_CONFIG_JSON}" ]; then
  if [ ! -f "${TRAINING_CONFIG_JSON}" ]; then
    echo "ERROR: TRAINING_CONFIG_JSON not found: ${TRAINING_CONFIG_JSON}"
    exit 1
  fi
  eval "$(
    python - "${TRAINING_CONFIG_JSON}" <<'PY'
import json
import pathlib
import shlex
import sys

p = pathlib.Path(sys.argv[1]).expanduser().resolve()
cfg = json.loads(p.read_text(encoding="utf-8"))
if not isinstance(cfg, dict):
    raise SystemExit("TRAINING_CONFIG_JSON must be a JSON object.")
allowed = [
    "MODEL_ID",
    "SFT_JSON_SWIFT",
    "SFT_VAL_JSON_SWIFT",
    "EVAL_MAX_SAMPLES",
    "OUTPUT_DIR",
    "OUTPUT_DIR_BASE",
    "RESUME_SEARCH_DIR",
    "RUN_TAG",
    "RUN_OUTPUT_DIR",
    "SAVE_STEPS",
    "MAX_LENGTH",
    "NUM_TRAIN_EPOCHS",
    "DISABLE_EVAL",
    "PER_DEVICE_TRAIN_BATCH_SIZE",
    "GRADIENT_ACCUMULATION_STEPS",
    "TORCH_DTYPE",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "WARMUP_RATIO",
    "DATALOADER_NUM_WORKERS",
    "DATASET_NUM_PROC",
    "AUTO_RESUME",
]
for k in allowed:
    if k in cfg:
        print(f'export {k}={shlex.quote(str(cfg[k]))}')
PY
  )"
  echo "Loaded training config: ${TRAINING_CONFIG_JSON}"
fi

# Keep OUTPUT_DIR as the per-run directory used by swift and post-run parsing.
OUTPUT_DIR="${RUN_OUTPUT_DIR}"
export OUTPUT_DIR RUN_TAG
mkdir -p "${OUTPUT_DIR}"
echo "Run tag: ${RUN_TAG}"
echo "Run output dir: ${OUTPUT_DIR}"
echo "Resume search dir: ${RESUME_SEARCH_DIR}"

mkdir -p /datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/query1/logs
mkdir -p "${CACHE_ROOT}/triton" "${CACHE_ROOT}/torch_extensions" "${CACHE_ROOT}/hf" "${CACHE_ROOT}/xdg_cache" "${CACHE_ROOT}/modelscope" "${CACHE_ROOT}/datasets" "${TMPDIR_BASE}"

# Avoid home-dir quota issues (e.g., /home/<user>/.triton).
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export HF_HOME="${CACHE_ROOT}/hf"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export DATASETS_CACHE="${CACHE_ROOT}/datasets"
export MODELSCOPE_CACHE="${CACHE_ROOT}/modelscope"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg_cache"
export TMPDIR="${TMPDIR_BASE}"

if [ ! -f "${SFT_JSON_SWIFT}" ]; then
  echo "ERROR: dataset not found: ${SFT_JSON_SWIFT}"
  echo "Run: bash examples/custom/interspeech_pipeline/scripts/00_build_sft_query1_dataset.sh"
  exit 1
fi
if [ ! -f "${SFT_VAL_JSON_SWIFT}" ]; then
  echo "ERROR: validation dataset not found: ${SFT_VAL_JSON_SWIFT}"
  exit 1
fi

if [ "${SFT_DEBUG_GT}" = "1" ]; then
  python - "${SFT_JSON_SWIFT}" "${SFT_VAL_JSON_SWIFT}" "${SFT_DEBUG_SAMPLES}" <<'PY'
import json
import re
import sys
from pathlib import Path

train_p = Path(sys.argv[1])
val_p = Path(sys.argv[2])
n_show = max(1, int(sys.argv[3]))


def parse_ids(text):
    return [int(x) for x in re.findall(r"\d+", str(text or ""))]


def extract_assistant_text(row):
    msgs = row.get("messages", [])
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = []
            for item in c:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            if parts:
                return "\n".join(parts)
    return ""


def inspect(path: Path, tag: str):
    data = json.loads(path.read_text(encoding="utf-8"))
    total = len(data)
    null_gt = 0
    parsed_ge3 = 0
    print(f"[sft_gt] {tag}_rows={total}")
    for i, row in enumerate(data):
        gt_text = extract_assistant_text(row)
        ids = parse_ids(gt_text)
        if not gt_text.strip():
            null_gt += 1
        if len(ids) >= 3:
            parsed_ge3 += 1
        if i < n_show:
            sid = row.get("sample_id", f"{tag}_{i}")
            print(
                f"[sft_gt] {tag} i={i} sample_id={sid} gt_text={gt_text!r} parsed_ids={ids[:3]}",
                flush=True,
            )
    print(
        f"[sft_gt] {tag}_null_or_empty_gt={null_gt} "
        f"{tag}_rows_with_gte3_ids={parsed_ge3}",
        flush=True,
    )


inspect(train_p, "train")
inspect(val_p, "val")
PY
fi

RESUME_ARGS=()
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
  echo "Resuming from explicit checkpoint: ${RESUME_FROM_CHECKPOINT}"
elif [ "${AUTO_RESUME}" = "1" ]; then
  LATEST_CKPT="$(
    python - "${RESUME_SEARCH_DIR}" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1]).expanduser().resolve()
if not root.exists():
    print("")
    raise SystemExit(0)

candidates = [p for p in root.rglob("checkpoint-*") if p.is_dir()]
if not candidates:
    print("")
    raise SystemExit(0)

latest = max(candidates, key=lambda p: p.stat().st_mtime)
print(str(latest))
PY
  )"
  if [ -n "${LATEST_CKPT}" ]; then
    RESUME_ARGS=(--resume_from_checkpoint "${LATEST_CKPT}")
    echo "Auto-resume from latest checkpoint: ${LATEST_CKPT}"
  else
    echo "No existing checkpoint found under ${RESUME_SEARCH_DIR}; starting fresh."
  fi
else
  echo "AUTO_RESUME=0; starting fresh."
fi

VAL_DATASET_ARGS=(--val_dataset "${SFT_VAL_JSON_SWIFT}#${EVAL_MAX_SAMPLES}")
EVAL_ARGS=(
  --eval_strategy steps
  --eval_steps "${SAVE_STEPS}"
  --metric_for_best_model loss
  --greater_is_better false
)
if [ "${DISABLE_EVAL}" = "1" ]; then
  VAL_DATASET_ARGS=()
  EVAL_ARGS=(--eval_strategy no)
  echo "Evaluation disabled (DISABLE_EVAL=1)."
fi

NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift sft \
  --model "${MODEL_ID}" \
  --dataset "${SFT_JSON_SWIFT}" \
  "${VAL_DATASET_ARGS[@]}" \
  --system "${SYSTEM_PROMPT}" \
  --tuner_type lora \
  --torch_dtype "${TORCH_DTYPE}" \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --freeze_llm true \
  --freeze_vit false \
  --freeze_aligner false \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --max_length "${MAX_LENGTH}" \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type cosine \
  "${EVAL_ARGS[@]}" \
  --max_pixels 524288 \
  --deepspeed zero2 \
  --logging_steps 100 \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 5 \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --output_dir "${OUTPUT_DIR}" \
  --report_to tensorboard \
  "${RESUME_ARGS[@]}"

python - <<'PY'
import json
import os
from pathlib import Path

out = Path("/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/query1/logs")
root = Path(os.environ["OUTPUT_DIR"])
states = sorted(root.rglob("trainer_state.json"), key=lambda p: p.stat().st_mtime)
if not states:
    print("WARN: trainer_state.json not found under output dir")
    raise SystemExit(0)

state_path = states[-1]
state = json.loads(state_path.read_text(encoding="utf-8"))
best_ckpt = state.get("best_model_checkpoint", "")
if not best_ckpt:
    best = None
    for row in state.get("log_history", []):
        if "eval_loss" in row and "step" in row:
            if best is None or row["eval_loss"] < best[0]:
                best = (row["eval_loss"], int(row["step"]))
    if best is not None:
        best_ckpt = str(state_path.parent / f"checkpoint-{best[1]}")

if best_ckpt:
    print(f"BEST_CHECKPOINT={best_ckpt}")
    out.mkdir(parents=True, exist_ok=True)
    run_tag = os.environ.get("RUN_TAG", "")
    if run_tag:
        (out / f"best_query1_checkpoint_{run_tag}.txt").write_text(best_ckpt + "\n", encoding="utf-8")
    (out / "best_query1_checkpoint.txt").write_text(best_ckpt + "\n", encoding="utf-8")
else:
    print("WARN: could not determine best checkpoint")
PY
