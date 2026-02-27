#!/usr/bin/env bash
set -euo pipefail

# ===== User-settable =====
MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-1-ms-swift/lr1e-5_r8_a32_lora_merged}"
GRPO2_GT_JSON_IN="${GRPO2_GT_JSON_IN:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-2/grpo2_gt.json}"
GRPO2_PRED_JSON_IN="${GRPO2_PRED_JSON_IN:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-2/grpo2_pred.json}"
GRPO2_GT_JSON_SWIFT="${GRPO2_GT_JSON_SWIFT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-2/grpo2_gt_swift_grpo2.json}"
GRPO2_PRED_JSON_SWIFT="${GRPO2_PRED_JSON_SWIFT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-2/grpo2_pred_swift_grpo2.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-2-ms-swift/grpo2_curr_fullReward_lora}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are an expert in deepfake speech spectrogram forensics.

You are given a spectrogram and transcript. You have already selected exactly 3 region IDs, in order: ID1, ID2, ID3.
For each ID, infer timing information (T), frequency band (F), phonetic category (P), and a visual description of the artifact and the likely audio impact implied by the artificial signs (En).

OUTPUT FORMAT (must follow exactly):
(Cn=ID1, T=..., F=..., P=..., En=\"...\"); (Cn=ID2, T=..., F=..., P=..., En=\"...\"); (Cn=ID3, T=..., F=..., P=..., En=\"...\")

Field definitions:
- Cn: region_id
- T: one of {speech, non-speech}
- F: one of {low, mid, high}
- P: one of {consonant, vowel, unvoiced}
- En: textual description, must be enclosed in double quotes.

Do not output any other text outside the three tuples.}"

WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
WARMUP_MAX_STEPS="${WARMUP_MAX_STEPS:-120}"
TOTAL_MAX_STEPS="${TOTAL_MAX_STEPS:-360}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
USE_VLLM="${USE_VLLM:-true}"
VLLM_MODE="${VLLM_MODE:-colocate}"
VLLM_TP="${VLLM_TP:-1}"
SLEEP_LEVEL="${SLEEP_LEVEL:-1}"

COMMON_ARGS=(
  --rlhf_type grpo
  --model "${MODEL_ID}"
  --system "${SYSTEM_PROMPT}"
  --external_plugins examples/custom/interspeech_pipeline/plugins/interspeech_rewards.py
  --reward_funcs external_interspeech_p2
  --use_vllm "${USE_VLLM}"
  --vllm_mode "${VLLM_MODE}"
  --vllm_tensor_parallel_size "${VLLM_TP}"
  --sleep_level "${SLEEP_LEVEL}"
  --beta 0.1
  --num_generations 4
  --temperature 0.9
  --tuner_type lora
  --lora_rank 8
  --lora_alpha 32
  --target_modules all-linear
  --freeze_llm true
  --freeze_vit true
  --freeze_aligner false
  --torch_dtype float16
  --max_completion_length 768
  --max_length 512
  --per_device_train_batch_size 8
  --gradient_accumulation_steps 1
  --learning_rate 1e-5
  --weight_decay 0.1
  --warmup_ratio 0.05
  --lr_scheduler_type cosine
  --logging_steps 20
  --save_steps 100
  --save_total_limit 10
  --dataloader_num_workers 4
  --dataset_num_proc 4
  --deepspeed zero2
  --output_dir "${OUTPUT_DIR}"
  --report_to tensorboard
  --log_completions true
)

# ===== Build GRPO2 datasets =====
python examples/custom/interspeech_pipeline/tools/build_swift_grpo_prompt2_dataset.py \
  --input-json "${GRPO2_PRED_JSON_IN}" \
  --output-json "${GRPO2_PRED_JSON_SWIFT}"

python examples/custom/interspeech_pipeline/tools/build_swift_grpo_prompt2_dataset.py \
  --input-json "${GRPO2_GT_JSON_IN}" \
  --output-json "${GRPO2_GT_JSON_SWIFT}"

# ===== Phase 1: warmup on pred =====
NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift rlhf \
  "${COMMON_ARGS[@]}" \
  --dataset "${GRPO2_PRED_JSON_SWIFT}" \
  --num_train_epochs "${WARMUP_EPOCHS}" \
  --max_steps "${WARMUP_MAX_STEPS}"

# ===== Phase 2: continue on GT =====
NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift rlhf \
  "${COMMON_ARGS[@]}" \
  --dataset "${GRPO2_GT_JSON_SWIFT}" \
  --resume_from_checkpoint true \
  --num_train_epochs "${TOTAL_EPOCHS}" \
  --max_steps "${TOTAL_MAX_STEPS}"
