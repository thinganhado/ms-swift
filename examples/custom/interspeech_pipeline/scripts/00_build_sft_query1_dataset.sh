#!/usr/bin/env bash
set -euo pipefail

CSV_IN="${CSV_IN:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/final_mask_topk/region_phone_table_topk3.csv}"
SFT_JSON_SWIFT="${SFT_JSON_SWIFT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/baseline_SFT/query1/stage1_query1_train_swift.json}"
USER_PROMPT="${USER_PROMPT:-Select the top 3 regions that most likely contain spoof artifacts.}"
REGIONS_COL="${REGIONS_COL:-}"
IMAGE_COL="${IMAGE_COL:-img_path}"

EXTRA_ARGS=()
if [ -n "${REGIONS_COL}" ]; then
  EXTRA_ARGS+=(--regions-col "${REGIONS_COL}")
fi

python examples/custom/interspeech_pipeline/tools/build_swift_sft_query1_dataset.py \
  --input-csv "${CSV_IN}" \
  --image-col "${IMAGE_COL}" \
  --user-prompt "${USER_PROMPT}" \
  --output-json "${SFT_JSON_SWIFT}" \
  "${EXTRA_ARGS[@]}"

echo "Built dataset: ${SFT_JSON_SWIFT}"
