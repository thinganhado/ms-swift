#!/bin/bash
set -euo pipefail

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

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DEEPSEEK_DIR="$(cd "${SCRIPT_DIR}/../../../../../DeepSeek-VL2" 2>/dev/null && pwd || true)"

DEEPSEEK_VL2_DIR="${DEEPSEEK_VL2_DIR:-${DEFAULT_DEEPSEEK_DIR}}"
MODEL_ID="${MODEL_ID:-deepseek-ai/deepseek-vl2}"
META_JSON="${META_JSON:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/data_GRPO2_Q2/grpo2_val.json}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-2-eval/}"
RUN_TAG="${RUN_TAG:-deepseek_q2_eval_$(date +%Y%m%d_%H%M%S)}"
SHARD_COUNT="${SHARD_COUNT:-4}"
OVERWRITE="${OVERWRITE:-1}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
MAX_CONCURRENT_SHARDS="${MAX_CONCURRENT_SHARDS:-1}"
FINALIZE_SHARDS="${FINALIZE_SHARDS:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.1}"
CHUNK_SIZE="${CHUNK_SIZE:--1}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
QWEN3_DIR="${QWEN3_DIR:-/scratch3/che489/Ha/interspeech/LLM/Qwen3}"
VERIFIER_MODEL_ID="${VERIFIER_MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/LLM/Qwen3-30B-A3B-Instruct-2507/}"
VERIFIER_GT_CSV="${VERIFIER_GT_CSV:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final__En/union_all3_only.csv}"
VERIFIER_BATCH_SIZE="${VERIFIER_BATCH_SIZE:-8}"
VERIFIER_TENSOR_PARALLEL_SIZE="${VERIFIER_TENSOR_PARALLEL_SIZE:-4}"
VERIFIER_GPU_MEMORY_UTILIZATION="${VERIFIER_GPU_MEMORY_UTILIZATION:-0.85}"

MODEL_TAG="$(basename "${MODEL_ID%/}" | tr -cs 'A-Za-z0-9._-' '_')"
RUN_DIR="${OUTPUT_BASE_DIR%/}/${MODEL_TAG}/${RUN_TAG}"
RAW_RESULT_JSONL="${RUN_DIR}/infer_result.jsonl"
VERIFIER_INPUT_JSONL="${RUN_DIR}/en_only_for_verifier.jsonl"
VERIFIER_OUTPUT_DIR="${RUN_DIR}/verifier"
VERIFIER_SYSTEM_FILE="${RUN_DIR}/q2_verifier_system.txt"
VERIFIER_USER_FILE="${RUN_DIR}/q2_verifier_user.txt"
EVAL_JSON="${RUN_DIR}/q2_eval_metrics.json"
TMP_PY="${RUN_DIR}/run_deepseek_q2_infer.py"

mkdir -p "${RUN_DIR}"

echo "[run] DEEPSEEK_VL2_DIR=${DEEPSEEK_VL2_DIR}"
echo "[run] MODEL_ID=${MODEL_ID}"
echo "[run] META_JSON=${META_JSON}"
echo "[run] RUN_DIR=${RUN_DIR}"
echo "[run] RAW_RESULT_JSONL=${RAW_RESULT_JSONL}"
echo "[run] VERIFIER_INPUT_JSONL=${VERIFIER_INPUT_JSONL}"
echo "[run] VERIFIER_OUTPUT_DIR=${VERIFIER_OUTPUT_DIR}"
echo "[run] EVAL_JSON=${EVAL_JSON}"
echo "[run] SHARD_COUNT=${SHARD_COUNT}"
echo "[run] GPU_IDS=${GPU_IDS}"
echo "[run] MAX_CONCURRENT_SHARDS=${MAX_CONCURRENT_SHARDS}"
echo "[run] FINALIZE_SHARDS=${FINALIZE_SHARDS}"
echo "[run] CACHE_ROOT=${CACHE_ROOT}"

cat > "${TMP_PY}" <<'PY'
#!/usr/bin/env python3
import argparse
import json
import re
import sys
import types
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
            if pkg_path.name == "deepseek_vl2":
                sys.path.insert(0, str(pkg_path.parent))
            else:
                sys.path.insert(0, str(pkg_path))

    import transformers.models.llama.modeling_llama as llama_mod
    if not hasattr(llama_mod, "LlamaFlashAttention2") and hasattr(llama_mod, "LlamaAttention"):
        llama_mod.LlamaFlashAttention2 = llama_mod.LlamaAttention

    try:
        import xformers.ops  # noqa: F401
    except Exception:
        xformers_mod = types.ModuleType("xformers")
        xformers_ops_mod = types.ModuleType("xformers.ops")

        def memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, scale=None, op=None):
            dropout_p = p if p and torch.is_grad_enabled() else 0.0
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias, dropout_p=dropout_p, scale=scale)

        xformers_ops_mod.memory_efficient_attention = memory_efficient_attention
        xformers_mod.ops = xformers_ops_mod
        sys.modules["xformers"] = xformers_mod
        sys.modules["xformers.ops"] = xformers_ops_mod

    from transformers import AutoModelForCausalLM, GenerationConfig
    from transformers.cache_utils import DynamicCache
    from transformers.generation import GenerationMixin
    from deepseek_vl2.models import DeepseekVLV2Processor

    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: None

    processor = DeepseekVLV2Processor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
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
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--overwrite", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
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

    out_path = Path(args.output_jsonl)
    if out_path.exists() and args.overwrite == 0:
        print(json.dumps({
            "shard_id": args.shard_id,
            "total_in_shard": len(shard_rows),
            "written": 0,
            "skipped": len(shard_rows),
            "output_jsonl": str(out_path),
        }, ensure_ascii=False))
        return

    model, processor = _load_model(args.model_id, args.deepseek_dir, dtype)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in shard_rows:
            sample_id = _norm(row.get("sample_id"))
            prompt1_output = _norm(row.get("prompt1_output"))
            if not sample_id:
                skipped += 1
                continue

            conversation, image_paths, _ = _prepare_prompt(row)
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
                "prompt1_output": prompt1_output,
                "response": response,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            done += 1

    print(json.dumps({
        "shard_id": args.shard_id,
        "total_in_shard": len(shard_rows),
        "written": done,
        "skipped": skipped,
        "output_jsonl": str(out_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
PY

chmod +x "${TMP_PY}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"

conda activate vllm

if [ "${FINALIZE_SHARDS}" = "1" ]; then
  if [ "${SHARD_COUNT}" -le 1 ]; then
    echo "[error] FINALIZE_SHARDS=1 requires SHARD_COUNT > 1" >&2
    exit 1
  fi
  rm -f "${RAW_RESULT_JSONL}"
  for (( shard_id=0; shard_id<SHARD_COUNT; shard_id++ )); do
    shard_jsonl="${RUN_DIR}/shard_${shard_id}_of_${SHARD_COUNT}/infer_result.jsonl"
    if [ ! -f "${shard_jsonl}" ]; then
      echo "[error] missing shard output: ${shard_jsonl}" >&2
      exit 1
    fi
    cat "${shard_jsonl}" >> "${RAW_RESULT_JSONL}"
  done
elif [ "${SHARD_COUNT}" -gt 1 ]; then
  pids=()
  active_jobs=0
  for (( shard_id=0; shard_id<SHARD_COUNT; shard_id++ )); do
    gpu="${GPU_ARRAY[$(( shard_id % ${#GPU_ARRAY[@]} ))]}"
    shard_dir="${RUN_DIR}/shard_${shard_id}_of_${SHARD_COUNT}"
    shard_jsonl="${shard_dir}/infer_result.jsonl"
    mkdir -p "${shard_dir}"

    echo "[launch] shard=${shard_id}/${SHARD_COUNT} gpu=${gpu} out=${shard_jsonl}"
    (
      CUDA_VISIBLE_DEVICES="${gpu}" \
        python "${TMP_PY}" \
          --model-id "${MODEL_ID}" \
          --meta-json "${META_JSON}" \
          --output-jsonl "${shard_jsonl}" \
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

  echo "[done] shard generation complete under ${RUN_DIR}/shard_*_of_${SHARD_COUNT}"
  echo "[done] run FINALIZE_SHARDS=1 with the same RUN_TAG to merge shards and run verifier."
  exit 0
else
  shard_jsonl="${RAW_RESULT_JSONL}"
  echo "[launch] shard=0/1 gpu=${GPU_ARRAY[0]} out=${shard_jsonl}"
  CUDA_VISIBLE_DEVICES="${GPU_ARRAY[0]}" \
    python "${TMP_PY}" \
      --model-id "${MODEL_ID}" \
      --meta-json "${META_JSON}" \
      --output-jsonl "${shard_jsonl}" \
      --shard-count 1 \
      --shard-id 0 \
      --overwrite "${OVERWRITE}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --temperature "${TEMPERATURE}" \
      --top-p "${TOP_P}" \
      --repetition-penalty "${REPETITION_PENALTY}" \
      --chunk-size "${CHUNK_SIZE}" \
      --torch-dtype "${TORCH_DTYPE}" \
      --deepseek-dir "${DEEPSEEK_VL2_DIR}"
fi

cat > "${VERIFIER_SYSTEM_FILE}" <<'EOF'
Infer the following features from <Explanation> and output: 
  - Time: speech or non-speech
  - Frequency: low, mid, or high
  - Phonetic: consonant, vowel, or unvoiced
If unsure about a field, output "ambiguous" for that field.

OUTPUT FORMAT (must follow exactly):

{
  "time": "speech/non-speech/ambiguous",
  "frequency": "low/mid/high/ambiguous",
  "phonetic": "consonant/vowel/unvoiced/ambiguous"
}
EOF

cat > "${VERIFIER_USER_FILE}" <<'EOF'
This is an artifact description for a spectrogram region: {description}. Please strictly follow the instructions to extract the information.
EOF

python - <<'PY' "${META_JSON}" "${RAW_RESULT_JSONL}" "${VERIFIER_INPUT_JSONL}"
import json
import re
import sys
from pathlib import Path

meta = Path(sys.argv[1])
src = Path(sys.argv[2])
dst = Path(sys.argv[3])

en_pat = re.compile(r'\bEn([123])\s*=\s*"((?:[^"\\]|\\.)*)"', re.I | re.S)

def norm(x):
    return str(x or "").strip()

meta_rows = json.loads(meta.read_text(encoding="utf-8"))
rows = []
with src.open("r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue

        response = norm(obj.get("response"))
        matches = {int(i): txt for i, txt in en_pat.findall(response)}
        meta_row = meta_rows[idx] if idx < len(meta_rows) and isinstance(meta_rows[idx], dict) else {}
        sample_id = norm(obj.get("sample_id")) or norm(meta_row.get("sample_id"))
        prompt1_output = norm(obj.get("prompt1_output")) or norm(meta_row.get("prompt1_output"))
        ids = [int(x) for x in re.findall(r"\d+", prompt1_output)][:3]

        for slot in (1, 2, 3):
            if slot not in matches:
                continue
            region_id = ids[slot - 1] if len(ids) >= slot else None
            if region_id is None:
                continue
            explanation = f"<Explanation>{matches[slot]}</Explanation>"
            rows.append({
                "sample_id": sample_id,
                "region_id": region_id,
                "response": explanation,
                "raw_response": response,
                "slot": slot,
            })

with dst.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"saved_verifier_rows: {len(rows)}")
print(f"saved: {dst}")
PY

mkdir -p "${VERIFIER_OUTPUT_DIR}"

cd "${QWEN3_DIR}"
SCRIPT_DIR_OVERRIDE="${QWEN3_DIR}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
MODEL_ID="${VERIFIER_MODEL_ID}" \
INPUT_MODEL_FOLDER="${RUN_DIR}" \
INPUT_JSONL="${VERIFIER_INPUT_JSONL}" \
GT_CSV="${VERIFIER_GT_CSV}" \
SYSTEM_FILE="${VERIFIER_SYSTEM_FILE}" \
USER_TEMPLATE_FILE="${VERIFIER_USER_FILE}" \
SHARD_COUNT=1 \
SHARD_ID=0 \
TENSOR_PARALLEL_SIZE="${VERIFIER_TENSOR_PARALLEL_SIZE}" \
OVERWRITE=0 \
OUTPUT_DIR="${VERIFIER_OUTPUT_DIR}" \
BATCH_SIZE="${VERIFIER_BATCH_SIZE}" \
VLLM_DISABLE_CUSTOM_ALL_REDUCE=1 \
VLLM_ENFORCE_EAGER=1 \
VLLM_GPU_MEMORY_UTILIZATION="${VERIFIER_GPU_MEMORY_UTILIZATION}" \
bash run_qwen_region_full.sbatch

python - <<'PY' "${META_JSON}" "${RAW_RESULT_JSONL}" "${VERIFIER_OUTPUT_DIR}" "${EVAL_JSON}"
import json
import re
import sys
from pathlib import Path
from statistics import mean

meta_json = Path(sys.argv[1])
raw_jsonl = Path(sys.argv[2])
ver_dir = Path(sys.argv[3])
out_json = Path(sys.argv[4])

TIME_LABELS = {"speech", "non-speech"}
FREQ_LABELS = {"low", "mid", "high"}
PHON_LABELS = {"consonant", "vowel", "unvoiced"}

TIME_LEXICON = {
    "speech": [r"\bspeech\b", r"\bvoiced\b", r"\bspoken\b"],
    "non-speech": [r"\bsilence\b", r"\bpause\b", r"\bnon[- ]?speech\b", r"\bbackground\b", r"\bnoise[- ]?only\b", r"\bunvoiced\b"],
}
FREQ_LEXICON = {
    "low": [r"\blow\b"],
    "mid": [r"\bmid\b", r"\bmiddle\b"],
    "high": [r"\bhigh\b"],
}
PHON_LEXICON = {
    "vowel": [r"\bvowel\b", r"\bformant\b"],
    "consonant": [r"\bconsonant\b", r"\bstop\b", r"\bfricative\b"],
    "unvoiced": [r"\bunvoiced\b", r"\bvoiceless\b", r"\baspiration\b", r"\bburst\b"],
}

slot_pat = {
    "T": re.compile(r'\bT([123])\s*=\s*([^,;)\n]+)', re.I),
    "F": re.compile(r'\bF([123])\s*=\s*([^,;)\n]+)', re.I),
    "P": re.compile(r'\bP([123])\s*=\s*([^,;)\n]+)', re.I),
    "En": re.compile(r'\bEn([123])\s*=\s*"((?:[^"\\]|\\.)*)"', re.I | re.S),
}

def norm(x):
    return str(x or "").strip()

def low(x):
    return norm(x).lower()

def parse_prompt_ids(s):
    return [int(x) for x in re.findall(r"\d+", norm(s))][:3]

def parse_indexed_tuple_text(text):
    text = norm(text)
    out = {}
    for name, pat in slot_pat.items():
        for idx, val in pat.findall(text):
            idx = int(idx)
            out.setdefault(idx, {})
            out[idx][name] = norm(val)
    return out

def _normalize_time(value):
    v = low(value)
    if v in TIME_LABELS:
        return v
    if v in {"nonspeech", "non speech", "non_speech"}:
        return "non-speech"
    return None

def _normalize_freq(value):
    v = low(value)
    if v in FREQ_LABELS:
        return v
    if v == "middle":
        return "mid"
    return None

def _normalize_phon(value):
    v = low(value)
    if v in PHON_LABELS:
        return v
    if v == "voiceless":
        return "unvoiced"
    return None

def _extract_label_by_lexicon(en_text, lexicon):
    text = low(en_text)
    best_label = None
    best_count = 0
    best_pos = 10**9
    for label, patterns in lexicon.items():
        count = 0
        first_pos = 10**9
        for pat in patterns:
            matches = list(re.finditer(pat, text, flags=re.I))
            if not matches:
                continue
            count += len(matches)
            first_pos = min(first_pos, matches[0].start())
        if count > best_count or (count == best_count and count > 0 and first_pos < best_pos):
            best_label = label
            best_count = count
            best_pos = first_pos
    return best_label

def _independent_extract_from_en(en_text):
    return {
        "T": _extract_label_by_lexicon(en_text, TIME_LEXICON),
        "F": _extract_label_by_lexicon(en_text, FREQ_LEXICON),
        "P": _extract_label_by_lexicon(en_text, PHON_LEXICON),
    }

def _field_metrics(y_true, y_pred, labels):
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if p == t)
    acc = (correct / n) if n else 0.0
    f1_per_class = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        denom = 2 * tp + fp + fn
        f1_per_class.append((2 * tp / denom) if denom > 0 else 0.0)
    return {"accuracy": acc, "macro_f1": (mean(f1_per_class) if f1_per_class else 0.0)}

def _caption_scores(gt_caps, pred_caps):
    scores = {"ROUGE_L": None, "METEOR": None, "BERTScore_F1": None}
    if not gt_caps:
        return scores
    try:
        from caption_metrics import meteor_score, rouge_l_score
        gts = {i: [gt] for i, gt in enumerate(gt_caps)}
        res = {i: [pd] for i, pd in enumerate(pred_caps)}
        rouge_l, _ = rouge_l_score(gts, res)
        meteor, _ = meteor_score(gts, res)
        scores["ROUGE_L"] = float(rouge_l)
        scores["METEOR"] = float(meteor)
    except Exception:
        pass
    try:
        from bert_score import score as bert_score
        _, _, f1 = bert_score(pred_caps, gt_caps, lang="en", verbose=False)
        scores["BERTScore_F1"] = float(f1.mean().item())
    except Exception:
        pass
    return scores

meta_rows = json.loads(meta_json.read_text(encoding="utf-8"))

gt_by_key = {}
gt_by_sample_slot = {}
for row in meta_rows:
    if not isinstance(row, dict):
        continue
    sample_id = norm(row.get("sample_id"))
    prompt_ids = parse_prompt_ids(row.get("prompt1_output"))
    gt_slots = parse_indexed_tuple_text(row.get("gt_prompt2"))
    for slot in (1, 2, 3):
        if slot > len(prompt_ids):
            continue
        region_id = prompt_ids[slot - 1]
        slot_data = gt_slots.get(slot, {})
        gt_entry = {
            "time": _normalize_time(slot_data.get("T")),
            "frequency": _normalize_freq(slot_data.get("F")),
            "phonetic": _normalize_phon(slot_data.get("P")),
            "en": norm(slot_data.get("En")),
            "slot": slot,
        }
        gt_by_key[(sample_id, region_id)] = gt_entry
        gt_by_sample_slot[(sample_id, slot)] = gt_entry

pred_en_by_sample_slot = {}
with raw_jsonl.open("r", encoding="utf-8") as f:
    for line in f:
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        sample_id = norm(obj.get("sample_id"))
        parsed = parse_indexed_tuple_text(norm(obj.get("response")))
        for slot in (1, 2, 3):
            pred_en_by_sample_slot[(sample_id, slot)] = norm(parsed.get(slot, {}).get("En"))

verifier_by_key = {}
for p in ver_dir.rglob("*.json"):
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        continue
    sample_id = norm(d.get("sample_id")) or p.parent.name
    try:
        region_id = int(d.get("region_id", p.stem))
    except Exception:
        continue
    s = d.get("output_structured") or {}
    verifier_by_key[(sample_id, region_id)] = {
        "time": _normalize_time(s.get("time")),
        "frequency": _normalize_freq(s.get("frequency")),
        "phonetic": _normalize_phon(s.get("phonetic")),
    }

y_true_t, y_pred_t = [], []
y_true_f, y_pred_f = [], []
y_true_p, y_pred_p = [], []
extractable = {"T": 0, "F": 0, "P": 0}
agree = {"T": 0, "F": 0, "P": 0}
gt_caps = []
pred_caps = []
sample_slot_correct = {}
field_rows_with_verifier = 0

for (sample_id, region_id), gt in gt_by_key.items():
    pred = verifier_by_key.get((sample_id, region_id), {})
    pt = pred.get("time")
    pf = pred.get("frequency")
    pp = pred.get("phonetic")

    y_true_t.append(gt["time"])
    y_pred_t.append(pt)
    y_true_f.append(gt["frequency"])
    y_pred_f.append(pf)
    y_true_p.append(gt["phonetic"])
    y_pred_p.append(pp)

    pred_slot = gt["slot"]
    pred_en = pred_en_by_sample_slot.get((sample_id, pred_slot), "")
    gt_caps.append(gt["en"])
    pred_caps.append(pred_en)

    if pt is not None and pf is not None and pp is not None:
        field_rows_with_verifier += 1
        extracted = _independent_extract_from_en(pred_en)
        if extracted["T"] is not None:
            extractable["T"] += 1
            if extracted["T"] == pt:
                agree["T"] += 1
        if extracted["F"] is not None:
            extractable["F"] += 1
            if extracted["F"] == pf:
                agree["F"] += 1
        if extracted["P"] is not None:
            extractable["P"] += 1
            if extracted["P"] == pp:
                agree["P"] += 1

    sample_slot_correct[(sample_id, pred_slot)] = (
        pt == gt["time"] and pf == gt["frequency"] and pp == gt["phonetic"]
    )

n_regions = len(gt_by_key)
t_metrics = _field_metrics(y_true_t, y_pred_t, sorted(TIME_LABELS))
f_metrics = _field_metrics(y_true_f, y_pred_f, sorted(FREQ_LABELS))
p_metrics = _field_metrics(y_true_p, y_pred_p, sorted(PHON_LABELS))
mean_fieldacc_macro_3 = mean([t_metrics["accuracy"], f_metrics["accuracy"], p_metrics["accuracy"]]) if n_regions else 0.0
mean_fieldacc_macro = mean_fieldacc_macro_3

coverage_t = (extractable["T"] / n_regions) if n_regions else 0.0
coverage_f = (extractable["F"] / n_regions) if n_regions else 0.0
coverage_p = (extractable["P"] / n_regions) if n_regions else 0.0
agreement_t = (agree["T"] / extractable["T"]) if extractable["T"] else 0.0
agreement_f = (agree["F"] / extractable["F"]) if extractable["F"] else 0.0
agreement_p = (agree["P"] / extractable["P"]) if extractable["P"] else 0.0
coverage_avg = mean([coverage_t, coverage_f, coverage_p]) if n_regions else 0.0
agreement_given_extractable_avg = mean([agreement_t, agreement_f, agreement_p]) if n_regions else 0.0
consscore = (agree["T"] + agree["F"] + agree["P"]) / (3.0 * n_regions) if n_regions else 0.0

sample_ids = {sid for sid, _ in gt_by_sample_slot.keys()}
sample_all9_correct = 0
for sid in sample_ids:
    if all(sample_slot_correct.get((sid, slot), False) for slot in (1, 2, 3)):
        sample_all9_correct += 1

caption = _caption_scores(gt_caps, pred_caps)

metrics = {
    "num_regions_scored": n_regions,
    "field_rows_with_verifier": field_rows_with_verifier,
    "field_coverage_rate": (field_rows_with_verifier / n_regions) if n_regions else 0.0,
    "accuracy": {
        "Accuracy_T": t_metrics["accuracy"],
        "Accuracy_F": f_metrics["accuracy"],
        "Accuracy_P": p_metrics["accuracy"],
        "MacroF1_T": t_metrics["macro_f1"],
        "MacroF1_F": f_metrics["macro_f1"],
        "MacroF1_P": p_metrics["macro_f1"],
        "MeanFieldAcc_macro_3fields": mean_fieldacc_macro_3,
        "MeanFieldAcc_macro": mean_fieldacc_macro,
        "sample_all9_accuracy": (sample_all9_correct / len(sample_ids)) if sample_ids else 0.0,
    },
    "consistency": {
        "ConsScore": consscore,
        "Coverage_T": coverage_t,
        "Coverage_F": coverage_f,
        "Coverage_P": coverage_p,
        "CoverageAvg": coverage_avg,
        "AgreementGivenExtractable_T": agreement_t,
        "AgreementGivenExtractable_F": agreement_f,
        "AgreementGivenExtractable_P": agreement_p,
        "AgreementGivenExtractableAvg": agreement_given_extractable_avg,
    },
    "caption_quality_en": {
        "ROUGE_L": caption["ROUGE_L"],
        "METEOR": caption["METEOR"],
        "BERTScore_F1": caption["BERTScore_F1"],
    },
}

out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print(f"saved_metrics: {out_json}")
PY

echo "[done] raw_result_jsonl=${RAW_RESULT_JSONL}"
echo "[done] verifier_input_jsonl=${VERIFIER_INPUT_JSONL}"
echo "[done] verifier_output_dir=${VERIFIER_OUTPUT_DIR}"
echo "[done] eval_json=${EVAL_JSON}"
