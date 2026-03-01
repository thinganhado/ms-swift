#!/bin/bash
set -euo pipefail

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="$(cd "${SCRIPT_DIR}/../tools" && pwd)"

DEFAULT_LLAVA_DIR="$(cd "${SCRIPT_DIR}/../../../../../LLaVA-OneVision-1.5" 2>/dev/null && pwd || true)"

LLAVA_OV_DIR="${LLAVA_OV_DIR:-${DEFAULT_LLAVA_DIR}}"
MODEL_ID="${MODEL_ID:-lmms-lab/LLaVA-OneVision-1.5-8B-Instruct}"
META_JSON="${META_JSON:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/data/stage1_query1_val_swift.json}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final__baseline__lightweight__VLM/}"
SHARD_COUNT="${SHARD_COUNT:-4}"
OVERWRITE="${OVERWRITE:-1}"
RUN_TAG="${RUN_TAG:-eval_$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

MODEL_TAG="$(basename "${MODEL_ID%/}" | tr -cs 'A-Za-z0-9._-' '_')"
RUN_DIR="${OUTPUT_BASE_DIR%/}/${MODEL_TAG}/${RUN_TAG}"
MERGED_DIR="${RUN_DIR}/merged_for_eval"
EVAL_JSON="${RUN_DIR}/eval_metrics.json"
TMP_PY="${RUN_DIR}/run_llava_q1_infer.py"

mkdir -p "${RUN_DIR}" "${MERGED_DIR}"

echo "[run] LLAVA_OV_DIR=${LLAVA_OV_DIR}"
echo "[run] MODEL_ID=${MODEL_ID}"
echo "[run] META_JSON=${META_JSON}"
echo "[run] RUN_DIR=${RUN_DIR}"
echo "[run] SHARD_COUNT=${SHARD_COUNT}"
echo "[run] GPU_IDS=${GPU_IDS}"

cat > "${TMP_PY}" <<'PY'
#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path

from PIL import Image


def _norm(x):
    return str(x or "").strip()


def _extract_text(content):
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(_norm(block.get("text")))
        return "\n".join(p for p in parts if p).strip()
    return _norm(content)


def _extract_image_paths(content):
    if not isinstance(content, list):
        return []
    out = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image":
            path = _norm(block.get("image"))
            if path:
                out.append(path)
    return out


def _find_first_user_message(row):
    msgs = row.get("messages") or row.get("conversations") or []
    if not isinstance(msgs, list):
        return None
    for msg in msgs:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg
    return None


def _canonical_gt(row):
    candidates = []
    for key in ("gt_regions", "assistant", "solution", "sft_top3"):
        if _norm(row.get(key)):
            candidates.append(_norm(row.get(key)))

    msgs = row.get("messages") or row.get("conversations") or []
    if isinstance(msgs, list):
        for msg in msgs:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                txt = _extract_text(msg.get("content"))
                if txt:
                    candidates.append(txt)
                    break

    for text in candidates:
        ints = [int(x) for x in re.findall(r"-?\d+", text)]
        if len(ints) >= 3:
            ints = ints[:3]
            return f"[{ints[0]}, {ints[1]}, {ints[2]}]"
    return ""


def _prepare_messages(row):
    user_msg = _find_first_user_message(row)
    if not user_msg:
        return None, [], ""
    content = user_msg.get("content", [])
    image_paths = _extract_image_paths(content)
    prompt_text = _extract_text(content)
    msg = {"role": "user", "content": []}
    for img in image_paths:
        msg["content"].append({"type": "image", "image": img})
    if prompt_text:
        msg["content"].append({"type": "text", "text": prompt_text})
    return [msg], image_paths, prompt_text


def _load_model(model_id, attn_impl):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    processor_kwargs = {
        "trust_remote_code": True,
        "use_fast": False,
        "fix_mistral_regex": True,
    }
    model_kwargs = {
        "dtype": "auto",
        "device_map": "auto",
        "trust_remote_code": True,
    }
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if attn_impl:
        # Force attention implementation through config as well, since some
        # custom model paths ignore the from_pretrained kwarg and read only
        # config internals during init.
        for attr in ("_attn_implementation", "_attn_implementation_internal"):
            try:
                setattr(config, attr, attn_impl)
            except Exception:
                pass
        for sub_name in ("text_config", "vision_config"):
            sub = getattr(config, sub_name, None)
            if sub is None:
                continue
            for attr in ("_attn_implementation", "_attn_implementation_internal"):
                try:
                    setattr(sub, attr, attn_impl)
                except Exception:
                    pass
        model_kwargs["attn_implementation"] = attn_impl

    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, **model_kwargs)
    return model, processor


def _run_one(model, processor, messages, image_paths, max_new_tokens):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(
        text=[text],
        images=images if images else None,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0] if output_text else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--meta-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--overwrite", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--attn-implementation", default="")
    args = parser.parse_args()

    llava_dir = os.environ.get("LLAVA_OV_DIR", "")
    if llava_dir and Path(llava_dir).exists():
        sys.path.insert(0, llava_dir)

    records = json.loads(Path(args.meta_json).read_text(encoding="utf-8"))
    shard_rows = [row for i, row in enumerate(records) if i % args.shard_count == args.shard_id]

    model, processor = _load_model(args.model_id, args.attn_implementation)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0

    for row in shard_rows:
        sample_id = _norm(row.get("sample_id"))
        if not sample_id:
            skipped += 1
            continue

        sample_dir = out_root / sample_id
        out_json = sample_dir / "json"
        if out_json.exists() and args.overwrite == 0:
            skipped += 1
            continue

        messages, image_paths, prompt_text = _prepare_messages(row)
        if not messages or not image_paths:
            skipped += 1
            continue

        response = _run_one(model, processor, messages, image_paths, args.max_new_tokens)
        payload = {
            "sample_id": sample_id,
            "prompt": prompt_text,
            "response": response,
            "gt_regions": _canonical_gt(row),
        }

        sample_dir.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        done += 1

    print(json.dumps({
        "shard_id": args.shard_id,
        "total_in_shard": len(shard_rows),
        "written": done,
        "skipped": skipped,
        "output_dir": str(out_root),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
PY

chmod +x "${TMP_PY}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"

conda activate vllm

pids=()
for (( shard_id=0; shard_id<SHARD_COUNT; shard_id++ )); do
  gpu="${GPU_ARRAY[$(( shard_id % ${#GPU_ARRAY[@]} ))]}"
  shard_dir="${RUN_DIR}/shard_${shard_id}_of_${SHARD_COUNT}"
  mkdir -p "${shard_dir}"

  echo "[launch] shard=${shard_id}/${SHARD_COUNT} gpu=${gpu} out=${shard_dir}"
  (
    export LLAVA_OV_DIR="${LLAVA_OV_DIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      python "${TMP_PY}" \
        --model-id "${MODEL_ID}" \
        --meta-json "${META_JSON}" \
        --output-dir "${shard_dir}" \
        --shard-count "${SHARD_COUNT}" \
        --shard-id "${shard_id}" \
        --overwrite "${OVERWRITE}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --attn-implementation "${ATTN_IMPLEMENTATION}"
  ) &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

for s in "${RUN_DIR}"/shard_*_of_${SHARD_COUNT}; do
  if [ -d "${s}" ]; then
    cp -rn "${s}"/* "${MERGED_DIR}/" || true
  fi
done

conda activate deepfake
python "${TOOLS_DIR}/eval_baseline_strongvlm_localization.py" \
  --model-dir "${MERGED_DIR}" \
  --save-json "${EVAL_JSON}"

echo "[done] merged_dir=${MERGED_DIR}"
echo "[done] eval_json=${EVAL_JSON}"
