#!/usr/bin/env bash
set -euo pipefail

# ===== User-settable =====
# Base model should be the GRPO1-merged (or multiturn-SFT-merged) checkpoint.
# GRPO2 trains a NEW LoRA adapter on top of this fixed base model.
BASE_MODEL_ID="${BASE_MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-1-ms-swift/lr1e-5_r8_a32_lora_merged}"
GRPO2_GT_JSON_IN="${GRPO2_GT_JSON_IN:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-2/grpo2_gt.json}"
GRPO2_PRED_JSON_IN="${GRPO2_PRED_JSON_IN:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-2/grpo2_pred.json}"
GRPO2_GT_JSON_SWIFT="${GRPO2_GT_JSON_SWIFT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-2/grpo2_gt_swift_grpo2.json}"
GRPO2_PRED_JSON_SWIFT="${GRPO2_PRED_JSON_SWIFT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-2/grpo2_pred_swift_grpo2.json}"
GRPO2_PRED_PREBUILT="${GRPO2_PRED_PREBUILT:-1}"
OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-2-ms-swift/grpo2_curr_fullReward_lora}"
RUN_TAG="${RUN_TAG:-v0-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_DIR_BASE%/}/${RUN_TAG}}"
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

WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
WARMUP_MAX_STEPS="${WARMUP_MAX_STEPS:-120}"
TOTAL_MAX_STEPS="${TOTAL_MAX_STEPS:-360}"
USE_PRED_PHASE="${USE_PRED_PHASE:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
USE_VLLM="${USE_VLLM:-true}"
VLLM_MODE="${VLLM_MODE:-colocate}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
SLEEP_LEVEL="${SLEEP_LEVEL:-1}"
INTERSPEECH_DEBUG_REWARD="${INTERSPEECH_DEBUG_REWARD:-0}"
INTERSPEECH_DEBUG_SAMPLES="${INTERSPEECH_DEBUG_SAMPLES:-8}"
INTERSPEECH_LOG_EVERY_STEPS="${INTERSPEECH_LOG_EVERY_STEPS:-1}"
INTERSPEECH_GROUP_SIZE="${INTERSPEECH_GROUP_SIZE:-8}"
GRPO_P2_RETRY_MISSING_FIELDS="${GRPO_P2_RETRY_MISSING_FIELDS:-1}"

mkdir -p "${OUTPUT_DIR}"
echo "[run] BASE_MODEL_ID=${BASE_MODEL_ID}"
echo "[run] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[run] RUN_TAG=${RUN_TAG}"
echo "[run] USE_PRED_PHASE=${USE_PRED_PHASE}"
echo "[run] VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}"
echo "[run] GRPO2_PRED_PREBUILT=${GRPO2_PRED_PREBUILT}"

COMMON_ARGS=(
  --rlhf_type grpo
  --model "${BASE_MODEL_ID}"
  --system "${SYSTEM_PROMPT}"
  --external_plugins examples/custom/interspeech_pipeline/plugins/interspeech_rewards.py
  --reward_funcs external_interspeech_p2
  --use_vllm "${USE_VLLM}"
  --vllm_mode "${VLLM_MODE}"
  --vllm_tensor_parallel_size "${VLLM_TP}"
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
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
if [[ "${USE_PRED_PHASE}" == "1" ]]; then
  if [[ "${GRPO2_PRED_PREBUILT}" == "1" ]]; then
    mkdir -p "$(dirname "${GRPO2_PRED_JSON_SWIFT}")"
    cp -f "${GRPO2_PRED_JSON_IN}" "${GRPO2_PRED_JSON_SWIFT}"
  else
    python examples/custom/interspeech_pipeline/tools/build_swift_grpo_prompt2_dataset.py \
      --input-json "${GRPO2_PRED_JSON_IN}" \
      --output-json "${GRPO2_PRED_JSON_SWIFT}"
  fi
fi

python examples/custom/interspeech_pipeline/tools/build_swift_grpo_prompt2_dataset.py \
  --input-json "${GRPO2_GT_JSON_IN}" \
  --output-json "${GRPO2_GT_JSON_SWIFT}"

if [[ "${USE_PRED_PHASE}" == "1" ]]; then
  # ===== Phase 1: warmup on pred =====
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  INTERSPEECH_DEBUG_REWARD="${INTERSPEECH_DEBUG_REWARD}" \
  INTERSPEECH_DEBUG_SAMPLES="${INTERSPEECH_DEBUG_SAMPLES}" \
  INTERSPEECH_LOG_EVERY_STEPS="${INTERSPEECH_LOG_EVERY_STEPS}" \
  INTERSPEECH_GROUP_SIZE="${INTERSPEECH_GROUP_SIZE}" \
  GRPO_P2_RETRY_MISSING_FIELDS="${GRPO_P2_RETRY_MISSING_FIELDS}" \
  swift rlhf \
    "${COMMON_ARGS[@]}" \
    --dataset "${GRPO2_PRED_JSON_SWIFT}" \
    --num_train_epochs "${WARMUP_EPOCHS}" \
    --max_steps "${WARMUP_MAX_STEPS}"
fi

if [[ "${USE_PRED_PHASE}" == "1" ]]; then
  # ===== Phase 2: continue on GT =====
  GT_PHASE_EXTRA_ARGS=(
    --resume_from_checkpoint true
    --num_train_epochs "${TOTAL_EPOCHS}"
    --max_steps "${TOTAL_MAX_STEPS}"
  )
else
  # ===== GT-only mode =====
  GT_PHASE_EXTRA_ARGS=(
    --num_train_epochs "${TOTAL_EPOCHS}"
    --max_steps "${TOTAL_MAX_STEPS}"
  )
fi

NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
INTERSPEECH_DEBUG_REWARD="${INTERSPEECH_DEBUG_REWARD}" \
INTERSPEECH_DEBUG_SAMPLES="${INTERSPEECH_DEBUG_SAMPLES}" \
INTERSPEECH_LOG_EVERY_STEPS="${INTERSPEECH_LOG_EVERY_STEPS}" \
INTERSPEECH_GROUP_SIZE="${INTERSPEECH_GROUP_SIZE}" \
GRPO_P2_RETRY_MISSING_FIELDS="${GRPO_P2_RETRY_MISSING_FIELDS}" \
swift rlhf \
  "${COMMON_ARGS[@]}" \
  --dataset "${GRPO2_GT_JSON_SWIFT}" \
  "${GT_PHASE_EXTRA_ARGS[@]}"
