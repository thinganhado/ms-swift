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

MODEL_ID="${MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/VLM/Qwen2.5-VL-7B-Instruct/}"
META_JSON="${META_JSON:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final_run/data_GRPO2_Q2/grpo2_val.json}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/GRPO-2-eval/}"
RUN_TAG="${RUN_TAG:-q2_eval_$(date +%Y%m%d_%H%M%S)}"
INFER_BACKEND="${INFER_BACKEND:-vllm}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0}"
QWEN3_DIR="${QWEN3_DIR:-/scratch3/che489/Ha/interspeech/LLM/Qwen3}"
VERIFIER_MODEL_ID="${VERIFIER_MODEL_ID:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/LLM/Qwen3-30B-A3B-Instruct-2507/}"
VERIFIER_GT_CSV="${VERIFIER_GT_CSV:-/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final__En/union_all3_only.csv}"
VERIFIER_BATCH_SIZE="${VERIFIER_BATCH_SIZE:-8}"
VERIFIER_TENSOR_PARALLEL_SIZE="${VERIFIER_TENSOR_PARALLEL_SIZE:-4}"
VERIFIER_GPU_MEMORY_UTILIZATION="${VERIFIER_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_TP="${VLLM_TP:-4}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
VLLM_LIMIT_MM_PER_PROMPT="${VLLM_LIMIT_MM_PER_PROMPT:-}"
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_ID="${SHARD_ID:-0}"
FINALIZE_SHARDS="${FINALIZE_SHARDS:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_TAG="$(basename "${MODEL_ID%/}" | tr -cs 'A-Za-z0-9._-' '_')"
BASE_RUN_DIR="${OUTPUT_BASE_DIR%/}/${MODEL_TAG}/${RUN_TAG}"
RUN_DIR="${BASE_RUN_DIR}"
if [ "${SHARD_COUNT}" -gt 1 ] && [ "${FINALIZE_SHARDS}" != "1" ]; then
  if [ "${SHARD_ID}" -lt 0 ] || [ "${SHARD_ID}" -ge "${SHARD_COUNT}" ]; then
    echo "[error] invalid shard config: SHARD_ID=${SHARD_ID}, SHARD_COUNT=${SHARD_COUNT}" >&2
    exit 1
  fi
  RUN_DIR="${BASE_RUN_DIR}/shard_${SHARD_ID}_of_${SHARD_COUNT}"
fi
RAW_RESULT_JSONL="${RUN_DIR}/infer_result.jsonl"
VERIFIER_INPUT_JSONL="${RUN_DIR}/en_only_for_verifier.jsonl"
VERIFIER_OUTPUT_DIR="${RUN_DIR}/verifier"
VERIFIER_SYSTEM_FILE="${RUN_DIR}/q2_verifier_system.txt"
VERIFIER_USER_FILE="${RUN_DIR}/q2_verifier_user.txt"
EVAL_JSON="${RUN_DIR}/q2_eval_metrics.json"
INFER_META_JSON="${META_JSON}"
SHARD_META_JSON="${RUN_DIR}/val_dataset_shard.json"

mkdir -p "${RUN_DIR}"

if [ "${SHARD_COUNT}" -gt 1 ] && [ "${FINALIZE_SHARDS}" != "1" ]; then
  python - <<'PY' "${META_JSON}" "${SHARD_META_JSON}" "${SHARD_ID}" "${SHARD_COUNT}"
import json
import math
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
shard_id = int(sys.argv[3])
shard_count = int(sys.argv[4])

rows = json.loads(src.read_text(encoding='utf-8'))
if not isinstance(rows, list):
    raise ValueError(f'Expected a JSON array in {src}')
total = len(rows)
shard_size = math.ceil(total / shard_count) if shard_count > 0 else total
start = shard_id * shard_size
end = min(start + shard_size, total)
subset = rows[start:end]
dst.write_text(json.dumps(subset, ensure_ascii=False), encoding='utf-8')
print(f'shard_range: [{start}, {end})')
print(f'shard_rows: {len(subset)} / {total}')
print(f'saved_shard_meta: {dst}')
PY
  INFER_META_JSON="${SHARD_META_JSON}"
fi

echo "[run] MODEL_ID=${MODEL_ID}"
echo "[run] META_JSON=${META_JSON}"
echo "[run] INFER_META_JSON=${INFER_META_JSON}"
echo "[run] RUN_DIR=${RUN_DIR}"
echo "[run] RAW_RESULT_JSONL=${RAW_RESULT_JSONL}"
echo "[run] VERIFIER_INPUT_JSONL=${VERIFIER_INPUT_JSONL}"
echo "[run] VERIFIER_OUTPUT_DIR=${VERIFIER_OUTPUT_DIR}"
echo "[run] EVAL_JSON=${EVAL_JSON}"
echo "[run] QWEN3_DIR=${QWEN3_DIR}"
echo "[run] VERIFIER_MODEL_ID=${VERIFIER_MODEL_ID}"
echo "[run] CACHE_ROOT=${CACHE_ROOT}"
echo "[run] SHARD_ID=${SHARD_ID}"
echo "[run] SHARD_COUNT=${SHARD_COUNT}"
echo "[run] FINALIZE_SHARDS=${FINALIZE_SHARDS}"
if [ "${INFER_BACKEND}" = "vllm" ]; then
  echo "[run] VLLM_TP=${VLLM_TP}"
  echo "[run] VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}"
  echo "[run] VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}"
  echo "[run] VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}"
fi

conda activate vllm

cd "$(cd "${SCRIPT_DIR}/../../.." && pwd)"

SWIFT_INFER_ARGS=(
  --model "${MODEL_ID}"
  --infer_backend "${INFER_BACKEND}"
  --val_dataset "${INFER_META_JSON}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --result_path "${RAW_RESULT_JSONL}"
)

if [ "${INFER_BACKEND}" = "transformers" ]; then
  SWIFT_INFER_ARGS+=(--max_batch_size "${MAX_BATCH_SIZE}")
elif [ "${INFER_BACKEND}" = "vllm" ]; then
  SWIFT_INFER_ARGS+=(
    --vllm_tensor_parallel_size "${VLLM_TP}"
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --vllm_max_num_seqs "${VLLM_MAX_NUM_SEQS}"
    --vllm_disable_custom_all_reduce "${VLLM_DISABLE_CUSTOM_ALL_REDUCE}"
    --vllm_enforce_eager "${VLLM_ENFORCE_EAGER}"
  )
  if [ -n "${VLLM_MAX_MODEL_LEN}" ]; then
    SWIFT_INFER_ARGS+=(--vllm_max_model_len "${VLLM_MAX_MODEL_LEN}")
  fi
  if [ -n "${VLLM_LIMIT_MM_PER_PROMPT}" ]; then
    SWIFT_INFER_ARGS+=(--vllm_limit_mm_per_prompt "${VLLM_LIMIT_MM_PER_PROMPT}")
  fi
fi

if [ "${FINALIZE_SHARDS}" = "1" ]; then
  if [ "${SHARD_COUNT}" -le 1 ]; then
    echo "[error] FINALIZE_SHARDS=1 requires SHARD_COUNT > 1" >&2
    exit 1
  fi
  rm -f "${RAW_RESULT_JSONL}"
  for (( merge_shard_id=0; merge_shard_id<SHARD_COUNT; merge_shard_id++ )); do
    shard_jsonl="${BASE_RUN_DIR}/shard_${merge_shard_id}_of_${SHARD_COUNT}/infer_result.jsonl"
    if [ ! -f "${shard_jsonl}" ]; then
      echo "[error] missing shard output: ${shard_jsonl}" >&2
      exit 1
    fi
    cat "${shard_jsonl}" >> "${RAW_RESULT_JSONL}"
  done
elif [ "${SHARD_COUNT}" -gt 1 ]; then
  swift infer "${SWIFT_INFER_ARGS[@]}"
  echo "[done] shard generation complete: ${RAW_RESULT_JSONL}"
  echo "[done] run FINALIZE_SHARDS=1 with the same RUN_TAG to merge shards and run verifier."
  exit 0
else
  swift infer "${SWIFT_INFER_ARGS[@]}"
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

python - <<'PY' "${INFER_META_JSON}" "${RAW_RESULT_JSONL}" "${VERIFIER_INPUT_JSONL}"
import json
import re
import sys
from pathlib import Path

meta = Path(sys.argv[1])
src = Path(sys.argv[2])
dst = Path(sys.argv[3])

en_pat = re.compile(r'\bEn([123])\s*=\s*"((?:[^"\\]|\\.)*)"', re.I | re.S)
region_heading_pat = re.compile(r'(?im)^[ \t>#*\-]*Region ID\s+(\d+)\s*:?[ \t*]*$')

def norm(x):
    return str(x or "").strip()

def get_response(obj):
    if isinstance(obj.get("response"), str):
        return obj["response"]
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            return norm(msg.get("content"))
    return ""

def parse_response_regions(response, prompt_ids):
    slot_matches = {int(i): norm(txt) for i, txt in en_pat.findall(response)}
    rows = []
    if slot_matches:
        for slot in (1, 2, 3):
            if slot not in slot_matches:
                continue
            region_id = prompt_ids[slot - 1] if len(prompt_ids) >= slot else None
            if region_id is None:
                continue
            rows.append((slot, region_id, slot_matches[slot]))
        if rows:
            return rows

    matches = list(region_heading_pat.finditer(response))
    if not matches:
        return rows

    prompt_id_to_slot = {region_id: idx + 1 for idx, region_id in enumerate(prompt_ids[:3])}
    for idx, match in enumerate(matches):
        region_id = int(match.group(1))
        slot = prompt_id_to_slot.get(region_id)
        if slot is None:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(response)
        explanation = norm(response[start:end])
        if not explanation:
            explanation = norm(response[match.start():end])
        if not explanation:
            continue
        rows.append((slot, region_id, explanation))
    return rows

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

        response = get_response(obj)
        meta_row = meta_rows[idx] if idx < len(meta_rows) and isinstance(meta_rows[idx], dict) else {}
        sample_id = norm(obj.get("sample_id")) or norm(meta_row.get("sample_id"))
        prompt1_output = norm(obj.get("prompt1_output")) or norm(meta_row.get("prompt1_output"))
        ids = [int(x) for x in re.findall(r"\d+", prompt1_output)][:3]

        for slot, region_id, explanation_text in parse_response_regions(response, ids):
            rows.append({
                "sample_id": sample_id,
                "region_id": region_id,
                "response": f"<Explanation>{explanation_text}</Explanation>",
                "raw_response": response,
                "slot": slot,
            })

with dst.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"saved_verifier_rows: {len(rows)}")
print(f"saved: {dst}")
PY

if [ ! -s "${VERIFIER_INPUT_JSONL}" ]; then
  echo "[error] verifier input is empty: ${VERIFIER_INPUT_JSONL}" >&2
  echo "[error] No region explanations were extracted from ${RAW_RESULT_JSONL}." >&2
  echo "[error] Check the model outputs in ${RAW_RESULT_JSONL}; this pipeline expects either En1=\"...\" fields or Region ID sections." >&2
  exit 1
fi

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

python - <<'PY' "${INFER_META_JSON}" "${RAW_RESULT_JSONL}" "${VERIFIER_OUTPUT_DIR}" "${EVAL_JSON}"
import json
import re
import sys
from pathlib import Path
from statistics import mean


def _meteor_score(gts, res):
    try:
        from pycocoevalcap.meteor.meteor import Meteor

        class RobustMeteor(Meteor):
            def _score(self, hypothesis_str, reference_list):
                references = " ||| ".join(reference_list)
                stat = self._stat(hypothesis_str, reference_list)
                eval_line = "EVAL" + " ||| " + stat
                self.meteor_p.stdin.write(f"{eval_line}\n")
                self.meteor_p.stdin.flush()
                score = float(self.meteor_p.stdout.readline().strip())
                scores = list(map(float, self.meteor_p.stdout.readline().strip().split()))
                return score, scores

        scorer = RobustMeteor()
        return scorer.compute_score(gts, res)
    except Exception:
        from nltk.translate.meteor_score import meteor_score as _nltk_meteor

        keys = list(gts.keys())
        sample_scores = []
        for k in keys:
            refs = gts[k]
            hyp = res[k][0] if isinstance(res.get(k), list) and res[k] else ""
            try:
                sample_scores.append(_nltk_meteor(refs, hyp))
            except Exception:
                sample_scores.append(0.0)
        score = sum(sample_scores) / max(len(sample_scores), 1)
        return score, sample_scores


def _rouge_l_score(gts, res):
    from pycocoevalcap.rouge.rouge import Rouge

    scorer = Rouge()
    return scorer.compute_score(gts, res)

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
region_heading_pat = re.compile(r'(?im)^[ \t>#*\-]*Region ID\s+(\d+)\s*:?[ \t*]*$')

def norm(x):
    return str(x or "").strip()

def low(x):
    return norm(x).lower()

def get_response(obj):
    if isinstance(obj.get("response"), str):
        return obj["response"]
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            return norm(msg.get("content"))
    return ""

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

def parse_predicted_sections(text, prompt_ids):
    parsed = parse_indexed_tuple_text(text)
    if any(norm(section.get("En")) for section in parsed.values()):
        return {slot: norm(section.get("En")) for slot, section in parsed.items()}

    matches = list(region_heading_pat.finditer(text))
    if not matches:
        return {}

    prompt_id_to_slot = {region_id: idx + 1 for idx, region_id in enumerate(prompt_ids[:3])}
    out = {}
    for idx, match in enumerate(matches):
        region_id = int(match.group(1))
        slot = prompt_id_to_slot.get(region_id)
        if slot is None:
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        explanation = norm(text[start:end])
        if not explanation:
            explanation = norm(text[match.start():end])
        if explanation:
            out[slot] = explanation
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
        gts = {i: [gt] for i, gt in enumerate(gt_caps)}
        res = {i: [pd] for i, pd in enumerate(pred_caps)}
        rouge_l, _ = _rouge_l_score(gts, res)
        meteor, _ = _meteor_score(gts, res)
        scores["ROUGE_L"] = float(rouge_l)
        scores["METEOR"] = float(meteor)
    except Exception as e:
        print(f"[warn] caption ROUGE/METEOR unavailable: {e}", file=sys.stderr)
    try:
        from bert_score import score as bert_score
        _, _, f1 = bert_score(pred_caps, gt_caps, lang="en", verbose=False)
        scores["BERTScore_F1"] = float(f1.mean().item())
    except Exception as e:
        print(f"[warn] caption BERTScore unavailable: {e}", file=sys.stderr)
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
    for idx, line in enumerate(f):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        meta_row = meta_rows[idx] if idx < len(meta_rows) and isinstance(meta_rows[idx], dict) else {}
        sample_id = norm(obj.get("sample_id")) or norm(meta_row.get("sample_id"))
        prompt_ids = parse_prompt_ids(obj.get("prompt1_output") or meta_row.get("prompt1_output"))
        parsed = parse_predicted_sections(get_response(obj), prompt_ids)
        for slot in (1, 2, 3):
            pred_en_by_sample_slot[(sample_id, slot)] = norm(parsed.get(slot))

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
