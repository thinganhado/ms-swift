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
OUTPUT_DIR="${OUTPUT_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT_ms_swift/stage1_query1_lora_Qwen3-VL-8B-Instruct}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-/scratch3/che489/Ha/interspeech/VLM/Qwen3-VL/prompts/region_forensics_system.txt}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

mkdir -p /datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/query1/logs

if [ ! -f "${SFT_JSON_SWIFT}" ]; then
  echo "ERROR: dataset not found: ${SFT_JSON_SWIFT}"
  echo "Run: bash examples/custom/interspeech_pipeline/scripts/00_build_sft_query1_dataset.sh"
  exit 1
fi

NPROC_PER_NODE="${NPROC_PER_NODE}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
swift sft \
  --model "${MODEL_ID}" \
  --dataset "${SFT_JSON_SWIFT}" \
  --system "${SYSTEM_PROMPT_FILE}" \
  --tuner_type lora \
  --torch_dtype float16 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --target_modules all-linear \
  --freeze_llm true \
  --freeze_vit false \
  --freeze_aligner false \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
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
