#!/usr/bin/env bash
set -euo pipefail

# ===== User-settable =====
MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_mt_lora_Qwen3-VL-8B-Instruct_merged/}"
GRPO1_JSON_SWIFT="${GRPO1_JSON_SWIFT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_train_swift_grpo1.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/GRPO-1/}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-/scratch3/che489/Ha/interspeech/VLM/Qwen3-VL/prompts/region_forensics_system.txt}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
USE_VLLM="${USE_VLLM:-true}"
VLLM_MODE="${VLLM_MODE:-colocate}"
VLLM_TP="${VLLM_TP:-1}"
SLEEP_LEVEL="${SLEEP_LEVEL:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
LOGGING_STEPS="${LOGGING_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-200}"

# Optional positional override: first arg as MODEL_ID
if [[ $# -ge 1 && -n "${1:-}" ]]; then
  MODEL_ID="$1"
fi

if [[ ! -f "${GRPO1_JSON_SWIFT}" ]]; then
  echo "ERROR: GRPO1 dataset not found: ${GRPO1_JSON_SWIFT}"
  echo "Please prepare it first (build_swift_grpo_prompt1_dataset.py)."
  exit 1
fi

# ===== Train GRPO-1 =====
NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL_ID}" \
  --dataset "${GRPO1_JSON_SWIFT}" \
  --system "${SYSTEM_PROMPT_FILE}" \
  --external_plugins examples/custom/interspeech_pipeline/plugins/interspeech_rewards.py \
  --reward_funcs external_interspeech_p1 \
  --use_vllm "${USE_VLLM}" \
  --vllm_mode "${VLLM_MODE}" \
  --vllm_tensor_parallel_size "${VLLM_TP}" \
  --sleep_level "${SLEEP_LEVEL}" \
  --beta 0.01 \
  --num_generations 8 \
  --temperature 1.1 \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --freeze_llm true \
  --freeze_vit true \
  --freeze_aligner false \
  --torch_dtype float16 \
  --num_train_epochs 2 \
  --max_steps -1 \
  --max_completion_length 128 \
  --max_length 128 \
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
  --log_completions true
