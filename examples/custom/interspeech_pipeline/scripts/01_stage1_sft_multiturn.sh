#!/usr/bin/env bash
set -euo pipefail

# ===== User-settable =====
MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/Qwen3-VL-8B-Instruct/}"
SFT_JSON_IN="${SFT_JSON_IN:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/SFT_2turn/stage1_multiturn_train.json}"
SFT_JSON_SWIFT="${SFT_JSON_SWIFT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/SFT_2turn/stage1_multiturn_train_swift.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT_ms_swift/stage1_mt_lora_Qwen3-VL-8B-Instruct}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# ===== Convert to Swift messages format =====
python examples/custom/interspeech_pipeline/tools/build_swift_sft_multiturn_dataset.py \
  --input-json "${SFT_JSON_IN}" \
  --output-json "${SFT_JSON_SWIFT}"

# ===== Train =====
NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift sft \
  --model "${MODEL_ID}" \
  --dataset "${SFT_JSON_SWIFT}" \
  --tuner_type lora \
  --torch_dtype float16 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --freeze_llm true \
  --freeze_vit false \
  --freeze_aligner false \
  --num_train_epochs 5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --max_pixels 524288 \
  --deepspeed zero2 \
  --logging_steps 10 \
  --save_steps 200 \
  --save_total_limit 5 \
  --dataloader_num_workers 1 \
  --dataset_num_proc 4 \
  --output_dir "${OUTPUT_DIR}" \
  --report_to tensorboard
