#!/bin/bash
#SBATCH --account=OD-237777
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
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
SFT_JSON_SWIFT="${SFT_JSON_SWIFT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/query1/stage1_query1_train_swift.json}"
SFT_VAL_JSON_SWIFT="${SFT_VAL_JSON_SWIFT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/query1/stage1_query1_val_swift.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT_ms_swift/stage1_query1_lora_Qwen3-VL-8B-Instruct}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-/scratch3/che489/Ha/interspeech/VLM/Qwen3-VL/baseline_prompts/baseline_system.txt}"
SAVE_STEPS="${SAVE_STEPS:-200}"
CACHE_ROOT="${CACHE_ROOT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/SFT_Q1}"
TMPDIR_BASE="${TMPDIR_BASE:-/tmp/${USER}_mswift_q1}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

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

RESUME_ARGS=()
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
  echo "Resuming from explicit checkpoint: ${RESUME_FROM_CHECKPOINT}"
elif [ "${AUTO_RESUME}" = "1" ]; then
  LATEST_CKPT="$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
  if [ -n "${LATEST_CKPT}" ]; then
    RESUME_ARGS=(--resume_from_checkpoint "${LATEST_CKPT}")
    echo "Auto-resume from latest checkpoint: ${LATEST_CKPT}"
  else
    echo "No existing checkpoint found under ${OUTPUT_DIR}; starting fresh."
  fi
else
  echo "AUTO_RESUME=0; starting fresh."
fi

NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift sft \
  --model "${MODEL_ID}" \
  --dataset "${SFT_JSON_SWIFT}" \
  --val_dataset "${SFT_VAL_JSON_SWIFT}" \
  --system "${SYSTEM_PROMPT_FILE}" \
  --tuner_type lora \
  --torch_dtype float16 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --freeze_llm true \
  --freeze_vit false \
  --freeze_aligner false \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --eval_strategy steps \
  --eval_steps "${SAVE_STEPS}" \
  --max_pixels 524288 \
  --deepspeed zero2 \
  --logging_steps 10 \
  --save_steps "${SAVE_STEPS}" \
  --metric_for_best_model loss \
  --greater_is_better false \
  --save_total_limit 5 \
  --dataloader_num_workers 1 \
  --dataset_num_proc 1 \
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
    (out / "best_query1_checkpoint.txt").write_text(best_ckpt + "\n", encoding="utf-8")
else:
    print("WARN: could not determine best checkpoint")
PY
