#!/usr/bin/env bash
set -euo pipefail

# ===== User-settable =====
MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_mt_lora_Qwen3-VL-8B-Instruct_merged/}"
GRPO1_JSON_IN="${GRPO1_JSON_IN:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_train.json}"
GRPO1_JSON_SWIFT="${GRPO1_JSON_SWIFT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/stage1_train_swift_grpo1.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-1-ms-swift/lr1e-5_r8_a32_lora}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-/scratch3/che489/Ha/interspeech/VLM/Qwen3-VL/prompts/region_forensics_system.txt}"
SFT_TOP3="${SFT_TOP3:-13, 1, 2}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# ===== Build GRPO1 dataset =====
python examples/custom/interspeech_pipeline/tools/build_swift_grpo_prompt1_dataset.py \
  --input-json "${GRPO1_JSON_IN}" \
  --output-json "${GRPO1_JSON_SWIFT}" \
  --default-sft-top3 "${SFT_TOP3}"

# ===== Train GRPO-1 =====
INTERSPEECH_SFT_TOP3="${SFT_TOP3}" \
NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift rlhf \
  --rlhf_type grpo \
  --model "${MODEL_ID}" \
  --dataset "${GRPO1_JSON_SWIFT}" \
  --system "${SYSTEM_PROMPT_FILE}" \
  --external_plugins examples/custom/interspeech_pipeline/plugins/interspeech_rewards.py \
  --reward_funcs external_interspeech_p1 \
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
  --logging_steps 20 \
  --save_steps 100 \
  --save_total_limit 10 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --deepspeed zero2 \
  --output_dir "${OUTPUT_DIR}" \
  --report_to tensorboard \
  --log_completions true
