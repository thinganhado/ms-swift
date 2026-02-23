#!/usr/bin/env bash
set -euo pipefail

# ADAPTER_PATH can be an output checkpoint dir or output root with args.json/checkpoints
ADAPTER_PATH="${ADAPTER_PATH:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-1-ms-swift/lr1e-5_r8_a32_lora/checkpoint-400}"
OUTPUT_DIR="${OUTPUT_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-1-ms-swift/lr1e-5_r8_a32_lora_merged}"

swift export \
  --adapters "${ADAPTER_PATH}" \
  --merge_lora true \
  --output_dir "${OUTPUT_DIR}"

