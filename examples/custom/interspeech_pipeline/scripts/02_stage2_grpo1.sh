#!/usr/bin/env bash
set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
conda activate vllm

# ===== User-settable =====
MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_mt_lora_Qwen3-VL-8B-Instruct_merged/}"
TRAIN_JSON_SWIFT="/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/data/stage1_query1_train_swift.json"
VAL_JSON_SWIFT="/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/data/stage1_query1_val_swift.json"
OUTPUT_DIR="${OUTPUT_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/GRPO-1/}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-As an expert in deepfake speech spectrogram forensics, you can detect regions containing deepfake artifacts by analysing spectrogram segments. Return only the JSON array of your three chosen region IDs.}"
CACHE_ROOT="${CACHE_ROOT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/SFT_Q1}"
TMPDIR_BASE="${TMPDIR_BASE:-/tmp/${USER}_mswift_q1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
USE_VLLM="${USE_VLLM:-true}"
VLLM_MODE="${VLLM_MODE:-colocate}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
SLEEP_LEVEL="${SLEEP_LEVEL:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
LOGGING_STEPS="${LOGGING_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-200}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
AUTO_MERGE_AFTER_TRAIN="${AUTO_MERGE_AFTER_TRAIN:-1}"
MERGE_SOURCE="${MERGE_SOURCE:-best}" # best | last
TEMPERATURE="${TEMPERATURE:-1.3}"
TOP_P="${TOP_P:-0.95}"
INTERSPEECH_DEBUG_REWARD="${INTERSPEECH_DEBUG_REWARD:-1}"
INTERSPEECH_DEBUG_SAMPLES="${INTERSPEECH_DEBUG_SAMPLES:-8}"

# Optional positional override: first arg as MODEL_ID
if [[ $# -ge 1 && -n "${1:-}" ]]; then
  MODEL_ID="$1"
fi

if [[ ! -f "${TRAIN_JSON_SWIFT}" ]]; then
  echo "ERROR: training dataset not found: ${TRAIN_JSON_SWIFT}"
  exit 1
fi
if [[ ! -f "${VAL_JSON_SWIFT}" ]]; then
  echo "ERROR: validation dataset not found: ${VAL_JSON_SWIFT}"
  exit 1
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
  echo "[run] Resuming from checkpoint: ${RESUME_FROM_CHECKPOINT}"
fi

mkdir -p "${CACHE_ROOT}/triton" "${CACHE_ROOT}/torch_extensions" "${CACHE_ROOT}/hf" \
  "${CACHE_ROOT}/xdg_cache" "${CACHE_ROOT}/modelscope" "${CACHE_ROOT}/datasets" "${CACHE_ROOT}/vllm" \
  "${CACHE_ROOT}/flashinfer" "${TMPDIR_BASE}"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export HF_HOME="${CACHE_ROOT}/hf"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export DATASETS_CACHE="${CACHE_ROOT}/datasets"
export MODELSCOPE_CACHE="${CACHE_ROOT}/modelscope"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg_cache"
export VLLM_CONFIG_ROOT="${CACHE_ROOT}/vllm"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export SWIFT_FIX_MISTRAL_REGEX="${SWIFT_FIX_MISTRAL_REGEX:-true}"
export FLASHINFER_WORKSPACE_BASE="${CACHE_ROOT}"
export FLASHINFER_WORKSPACE_DIR="${CACHE_ROOT}/flashinfer"
export FLASHINFER_JIT_CACHE_DIR="${FLASHINFER_WORKSPACE_DIR}"
export TMPDIR="${TMPDIR_BASE}"
export INTERSPEECH_DEBUG_REWARD="${INTERSPEECH_DEBUG_REWARD}"
export INTERSPEECH_DEBUG_SAMPLES="${INTERSPEECH_DEBUG_SAMPLES}"

# ===== Train GRPO-1 =====
NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL_ID}" \
  --model_kwargs '{"fix_mistral_regex": true}' \
  --dataset "${TRAIN_JSON_SWIFT}" \
  --system "${SYSTEM_PROMPT}" \
  --external_plugins examples/custom/interspeech_pipeline/plugins/interspeech_rewards.py \
  --reward_funcs external_interspeech_p1 \
  --use_vllm "${USE_VLLM}" \
  --vllm_mode "${VLLM_MODE}" \
  --vllm_tensor_parallel_size "${VLLM_TP}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --sleep_level "${SLEEP_LEVEL}" \
  --beta 0.01 \
  --num_generations 8 \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_llm true \
  --freeze_vit true \
  --freeze_aligner false \
  --torch_dtype "${TORCH_DTYPE}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_steps -1 \
  --max_completion_length 128 \
  --max_length 768 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --weight_decay 0.1 \
  --warmup_ratio 0.05 \
  --lr_scheduler_type cosine \
  --logging_steps "${LOGGING_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 10 \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --dataset_num_proc "${DATASET_NUM_PROC}" \
  --deepspeed zero2 \
  --output_dir "${OUTPUT_DIR}" \
  --report_to tensorboard \
  --log_completions true \
  "${RESUME_ARGS[@]}"

if [[ "${AUTO_MERGE_AFTER_TRAIN}" == "1" ]]; then
  if ! command -v swift >/dev/null 2>&1; then
    echo "WARN: swift command not found in PATH; skipping merge export."
    exit 0
  fi

  MERGE_CKPT="$(
    python - "${OUTPUT_DIR}" "${MERGE_SOURCE}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
merge_source = sys.argv[2].strip().lower()

states = sorted(root.rglob("trainer_state.json"), key=lambda p: p.stat().st_mtime)
if not states:
    print("")
    raise SystemExit(0)

state = json.loads(states[-1].read_text(encoding="utf-8"))
best_ckpt = str(state.get("best_model_checkpoint", "")).strip()
last_ckpt = str(state.get("last_model_checkpoint", "")).strip()

if merge_source == "last":
    print(last_ckpt or best_ckpt)
else:
    print(best_ckpt or last_ckpt)
PY
  )"

  if [[ -z "${MERGE_CKPT}" ]]; then
    echo "WARN: Could not find best/last checkpoint under ${OUTPUT_DIR}; skipping merge export."
    exit 0
  fi
  if [[ ! -d "${MERGE_CKPT}" ]]; then
    echo "WARN: Selected merge checkpoint does not exist as directory: ${MERGE_CKPT}"
    exit 0
  fi

  MERGED_OUT="${MERGE_CKPT%/}-merged"
  echo "[merge] source=${MERGE_CKPT}"
  echo "[merge] output=${MERGED_OUT}"
  swift export \
    --adapters "${MERGE_CKPT}" \
    --merge_lora true \
    --output_dir "${MERGED_OUT}"
fi
