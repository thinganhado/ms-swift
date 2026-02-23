#!/usr/bin/env bash
set -euo pipefail

ADAPTER_PATH="${ADAPTER_PATH:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-2-ms-swift/grpo2_curr_fullReward_lora/checkpoint-360}"
OUTPUT_DIR="${OUTPUT_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-2-ms-swift/grpo2_curr_fullReward_lora_merged}"

swift export \
  --adapters "${ADAPTER_PATH}" \
  --merge_lora true \
  --output_dir "${OUTPUT_DIR}"

