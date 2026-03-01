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

DEFAULT_DEEPSEEK_DIR="$(cd "${SCRIPT_DIR}/../../../../../DeepSeek-VL2" 2>/dev/null && pwd || true)"

DEEPSEEK_VL2_DIR="${DEEPSEEK_VL2_DIR:-${DEFAULT_DEEPSEEK_DIR}}"
MODEL_ID="${MODEL_ID:-deepseek-ai/deepseek-vl2}"
META_JSON="${META_JSON:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/data/stage1_query1_val_swift.json}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final__baseline__lightweight__VLM/}"
SHARD_COUNT="${SHARD_COUNT:-4}"
OVERWRITE="${OVERWRITE:-1}"
RUN_TAG="${RUN_TAG:-eval_$(date +%Y%m%d_%H%M%S)}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MAX_CONCURRENT_SHARDS="${MAX_CONCURRENT_SHARDS:-${SHARD_COUNT}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
CHUNK_SIZE="${CHUNK_SIZE:--1}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
CACHE_ROOT="${CACHE_ROOT:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/SFT_Q1}"

mkdir -p "${CACHE_ROOT}/triton" "${CACHE_ROOT}/torch_extensions" "${CACHE_ROOT}/hf" \
  "${CACHE_ROOT}/xdg_cache" "${CACHE_ROOT}/modelscope" "${CACHE_ROOT}/datasets" \
  "${CACHE_ROOT}/vllm" "${CACHE_ROOT}/flashinfer"

export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
export HF_HOME="${CACHE_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export DATASETS_CACHE="${CACHE_ROOT}/datasets"
export MODELSCOPE_CACHE="${CACHE_ROOT}/modelscope"
export MODELSCOPE_HOME="${CACHE_ROOT}/modelscope"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg_cache"
export VLLM_CONFIG_ROOT="${CACHE_ROOT}/vllm"
export VLLM_NO_USAGE_STATS=1
export FLASHINFER_WORKSPACE_BASE="${CACHE_ROOT}"
export FLASHINFER_WORKSPACE_DIR="${CACHE_ROOT}/flashinfer"
export FLASHINFER_JIT_CACHE_DIR="${FLASHINFER_WORKSPACE_DIR}"
unset TRANSFORMERS_CACHE

MODEL_TAG="$(basename "${MODEL_ID%/}" | tr -cs 'A-Za-z0-9._-' '_')"
RUN_DIR="${OUTPUT_BASE_DIR%/}/${MODEL_TAG}/${RUN_TAG}"
MERGED_DIR="${RUN_DIR}/merged_for_eval"
EVAL_JSON="${RUN_DIR}/eval_metrics.json"
TMP_PY="${RUN_DIR}/run_deepseek_q1_infer.py"

mkdir -p "${RUN_DIR}" "${MERGED_DIR}"

echo "[run] DEEPSEEK_VL2_DIR=${DEEPSEEK_VL2_DIR}"
echo "[run] MODEL_ID=${MODEL_ID}"
echo "[run] META_JSON=${META_JSON}"
echo "[run] RUN_DIR=${RUN_DIR}"
echo "[run] SHARD_COUNT=${SHARD_COUNT}"
echo "[run] GPU_IDS=${GPU_IDS}"
echo "[run] MAX_CONCURRENT_SHARDS=${MAX_CONCURRENT_SHARDS}"
echo "[run] CACHE_ROOT=${CACHE_ROOT}"

cat > "${TMP_PY}" <<'PY'
#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
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
                txt = _norm(block.get("text"))
                if txt:
                    parts.append(txt)
        return "\n".join(parts).strip()
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
    for key in ("gt_regions", "assistant", "solution", "sft_top3", "prompt1_output"):
        val = _norm(row.get(key))
        if val:
            candidates.append(val)

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


def _prepare_prompt(row):
    user_msg = _find_first_user_message(row)
    if not user_msg:
        return None, [], ""
    content = user_msg.get("content", [])
    image_paths = _extract_image_paths(content)
    prompt_text = _extract_text(content)
    if not image_paths or not prompt_text:
        return None, image_paths, prompt_text
    prompt_content = "<image>\n" * len(image_paths) + prompt_text
    conversation = [
        {
            "role": "<|User|>",
            "content": prompt_content.strip(),
            "images": image_paths,
        },
        {
            "role": "<|Assistant|>",
            "content": "",
        },
    ]
    return conversation, image_paths, prompt_text


def _dtype_from_name(name):
    name = _norm(name).lower()
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    if name in {"float32", "fp32"}:
        return torch.float32
    return torch.float16


def _load_model(model_id, deepseek_dir, dtype):
    if deepseek_dir:
        pkg_path = Path(deepseek_dir)
        if pkg_path.exists():
            # Accept either the repo root (containing deepseek_vl2/) or the
            # package directory itself (.../deepseek_vl2).
            if pkg_path.name == "deepseek_vl2":
                sys.path.insert(0, str(pkg_path.parent))
            else:
                sys.path.insert(0, str(pkg_path))

    # DeepSeek-VL2 targets an older transformers API that still exposed
    # LlamaFlashAttention2. On newer transformers builds, fall back to the
    # regular attention class so the import path remains available.
    import transformers.models.llama.modeling_llama as llama_mod
    if not hasattr(llama_mod, "LlamaFlashAttention2") and hasattr(llama_mod, "LlamaAttention"):
        llama_mod.LlamaFlashAttention2 = llama_mod.LlamaAttention

    from transformers import AutoModelForCausalLM, GenerationConfig
    from transformers.cache_utils import DynamicCache
    from transformers.generation import GenerationMixin
    from deepseek_vl2.models import DeepseekVLV2Processor

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

    processor = DeepseekVLV2Processor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    # transformers>=4.50 no longer provides GenerationMixin via PreTrainedModel.
    # DeepSeek-VL2 remote-code models define prepare_inputs_for_generation but may
    # not inherit GenerationMixin, so patch in the full mixin API explicitly.
    if not isinstance(model, GenerationMixin):
        for attr_name, attr_value in GenerationMixin.__dict__.items():
            if attr_name.startswith("__"):
                continue
            if hasattr(model.__class__, attr_name):
                continue
            setattr(model.__class__, attr_name, attr_value)
    decoder_config = (getattr(model.config, "language_config", None)
                      or getattr(model.config, "llm_config", None)
                      or getattr(model.config, "text_config", None))
    if decoder_config is not None:
        for attr_name in ("num_hidden_layers", "num_attention_heads", "hidden_size", "num_key_value_heads"):
            if not hasattr(model.config, attr_name) and hasattr(decoder_config, attr_name):
                setattr(model.config, attr_name, getattr(decoder_config, attr_name))
    if getattr(model, "generation_config", None) is None:
        model.generation_config = GenerationConfig.from_model_config(model.config)
    model = model.cuda().eval()
    return model, processor


def _load_images(conversation):
    pil_images = []
    for message in conversation:
        for image_path in message.get("images", []):
            pil_images.append(Image.open(image_path).convert("RGB"))
    return pil_images


def _run_one(model, processor, conversation, max_new_tokens, temperature, top_p, repetition_penalty, chunk_size, dtype):
    tokenizer = processor.tokenizer
    pil_images = _load_images(conversation)
    prepare_inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt="",
    ).to(model.device, dtype=dtype)

    with torch.no_grad():
        if chunk_size != -1:
            inputs_embeds, past_key_values = model.incremental_prefilling(
                input_ids=prepare_inputs.input_ids,
                images=prepare_inputs.images,
                images_seq_mask=prepare_inputs.images_seq_mask,
                images_spatial_crop=prepare_inputs.images_spatial_crop,
                attention_mask=prepare_inputs.attention_mask,
                chunk_size=chunk_size,
            )
        else:
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
            past_key_values = None

        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "input_ids": prepare_inputs.input_ids,
            "images": prepare_inputs.images,
            "images_seq_mask": prepare_inputs.images_seq_mask,
            "images_spatial_crop": prepare_inputs.images_spatial_crop,
            "attention_mask": prepare_inputs.attention_mask,
            "past_key_values": past_key_values,
            "pad_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
        }
        if temperature > 0:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
            })
        else:
            gen_kwargs["do_sample"] = False

        outputs = model.generate(**gen_kwargs)

    start = len(prepare_inputs.input_ids[0])
    answer = tokenizer.decode(outputs[0][start:].cpu().tolist(), skip_special_tokens=False)
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--meta-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--overwrite", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--chunk-size", type=int, default=-1)
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--deepseek-dir", default="")
    args = parser.parse_args()

    dtype = _dtype_from_name(args.torch_dtype)
    records = json.loads(Path(args.meta_json).read_text(encoding="utf-8"))
    shard_rows = [row for i, row in enumerate(records) if i % args.shard_count == args.shard_id]

    model, processor = _load_model(args.model_id, args.deepseek_dir, dtype)
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

        conversation, image_paths, prompt_text = _prepare_prompt(row)
        if not conversation or not image_paths:
            skipped += 1
            continue

        response = _run_one(
            model,
            processor,
            conversation,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.repetition_penalty,
            args.chunk_size,
            dtype,
        )

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
active_jobs=0
for (( shard_id=0; shard_id<SHARD_COUNT; shard_id++ )); do
  gpu="${GPU_ARRAY[$(( shard_id % ${#GPU_ARRAY[@]} ))]}"
  shard_dir="${RUN_DIR}/shard_${shard_id}_of_${SHARD_COUNT}"
  mkdir -p "${shard_dir}"

  echo "[launch] shard=${shard_id}/${SHARD_COUNT} gpu=${gpu} out=${shard_dir}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" \
      python "${TMP_PY}" \
        --model-id "${MODEL_ID}" \
        --meta-json "${META_JSON}" \
        --output-dir "${shard_dir}" \
        --shard-count "${SHARD_COUNT}" \
        --shard-id "${shard_id}" \
        --overwrite "${OVERWRITE}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --temperature "${TEMPERATURE}" \
        --top-p "${TOP_P}" \
        --repetition-penalty "${REPETITION_PENALTY}" \
        --chunk-size "${CHUNK_SIZE}" \
        --torch-dtype "${TORCH_DTYPE}" \
        --deepseek-dir "${DEEPSEEK_VL2_DIR}"
  ) &
  pids+=("$!")
  active_jobs=$((active_jobs + 1))
  if [ "${active_jobs}" -ge "${MAX_CONCURRENT_SHARDS}" ]; then
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
    pids=()
    active_jobs=0
  fi
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
