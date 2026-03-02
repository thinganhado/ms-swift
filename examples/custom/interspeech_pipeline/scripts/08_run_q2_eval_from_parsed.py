#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path
from statistics import mean


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

SLOT_PATTERNS = {
    "T": re.compile(r"\bT([123])\s*=\s*([^,;)\n]+)", re.I),
    "F": re.compile(r"\bF([123])\s*=\s*([^,;)\n]+)", re.I),
    "P": re.compile(r"\bP([123])\s*=\s*([^,;)\n]+)", re.I),
    "En": re.compile(r'\bEn([123])\s*=\s*"((?:[^"\\]|\\.)*)"', re.I | re.S),
}
REGION_PAT = re.compile(r"(?i)\bRegion ID\s+(\d+)\b")


def norm(value):
    return str(value or "").strip()


def low(value):
    return norm(value).lower()


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Q2 metrics from saved generation outputs and verifier outputs.")
    parser.add_argument("--meta-json", required=True)
    parser.add_argument("--raw-result-jsonl", required=True)
    parser.add_argument("--verifier-output-dir", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def get_response(obj):
    if isinstance(obj.get("response"), str):
        return obj["response"]
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        choice0 = choices[0]
        if isinstance(choice0, dict):
            msg = choice0.get("message")
            if isinstance(msg, dict):
                return norm(msg.get("content"))
    return ""


def parse_prompt_ids(text):
    return [int(x) for x in re.findall(r"\d+", norm(text))][:3]


def parse_indexed_tuple_text(text):
    text = norm(text)
    out = {}
    for name, pattern in SLOT_PATTERNS.items():
        for idx, val in pattern.findall(text):
            idx = int(idx)
            out.setdefault(idx, {})
            out[idx][name] = norm(val)
    return out


def parse_predicted_sections(text, prompt_ids):
    parsed = parse_indexed_tuple_text(text)
    if any(norm(section.get("En")) for section in parsed.values()):
        return {slot: norm(section.get("En")) for slot, section in parsed.items()}

    matches = list(REGION_PAT.finditer(text))
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


def normalize_time(value):
    v = low(value)
    if v in TIME_LABELS:
        return v
    if v in {"nonspeech", "non speech", "non_speech"}:
        return "non-speech"
    return None


def normalize_freq(value):
    v = low(value)
    if v in FREQ_LABELS:
        return v
    if v == "middle":
        return "mid"
    return None


def normalize_phon(value):
    v = low(value)
    if v in PHON_LABELS:
        return v
    if v == "voiceless":
        return "unvoiced"
    return None


def extract_label_by_lexicon(en_text, lexicon):
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


def independent_extract_from_en(en_text):
    return {
        "T": extract_label_by_lexicon(en_text, TIME_LEXICON),
        "F": extract_label_by_lexicon(en_text, FREQ_LEXICON),
        "P": extract_label_by_lexicon(en_text, PHON_LEXICON),
    }


def field_metrics(y_true, y_pred, labels):
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


def caption_scores(gt_caps, pred_caps):
    scores = {"ROUGE_L": None, "METEOR": None, "BERTScore_F1": None}
    if not gt_caps:
        return scores
    try:
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge

        gts = {i: [gt] for i, gt in enumerate(gt_caps)}
        res = {i: [pd] for i, pd in enumerate(pred_caps)}
        scores["ROUGE_L"], _ = Rouge().compute_score(gts, res)
        scores["METEOR"], _ = Meteor().compute_score(gts, res)
        scores["ROUGE_L"] = float(scores["ROUGE_L"])
        scores["METEOR"] = float(scores["METEOR"])
    except Exception as e:
        print(f"[warn] caption ROUGE/METEOR unavailable: {e}")
    try:
        from bert_score import score as bert_score

        _, _, f1 = bert_score(pred_caps, gt_caps, lang="en", verbose=False)
        scores["BERTScore_F1"] = float(f1.mean().item())
    except Exception as e:
        print(f"[warn] caption BERTScore unavailable: {e}")
    return scores


def main():
    args = parse_args()
    meta_json = Path(args.meta_json)
    raw_jsonl = Path(args.raw_result_jsonl)
    ver_dir = Path(args.verifier_output_dir)
    out_json = Path(args.output_json)

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
                "time": normalize_time(slot_data.get("T")),
                "frequency": normalize_freq(slot_data.get("F")),
                "phonetic": normalize_phon(slot_data.get("P")),
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
    verifier_by_sample_slot = {}
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
        structured = d.get("output_structured") or {}
        slot_payloads = {}
        if isinstance(structured, dict):
            for key in ("1", "2", "3"):
                value = structured.get(key)
                if isinstance(value, dict):
                    slot_payloads[int(key)] = value
            for key in ("slot_1", "slot_2", "slot_3"):
                value = structured.get(key)
                if isinstance(value, dict):
                    try:
                        slot_payloads[int(key.split("_")[-1])] = value
                    except Exception:
                        pass
            slots_obj = structured.get("slots")
            if isinstance(slots_obj, dict):
                for key, value in slots_obj.items():
                    if isinstance(value, dict):
                        try:
                            slot_payloads[int(str(key).split("_")[-1])] = value
                        except Exception:
                            pass

        if slot_payloads:
            for slot, slot_structured in slot_payloads.items():
                verifier_by_sample_slot[(sample_id, slot)] = {
                    "time": normalize_time(slot_structured.get("time")),
                    "frequency": normalize_freq(slot_structured.get("frequency")),
                    "phonetic": normalize_phon(slot_structured.get("phonetic")),
                }

        verifier_by_key[(sample_id, region_id)] = {
            "time": normalize_time(structured.get("time")),
            "frequency": normalize_freq(structured.get("frequency")),
            "phonetic": normalize_phon(structured.get("phonetic")),
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
        if not any(v is not None for v in pred.values()):
            pred = verifier_by_sample_slot.get((sample_id, gt["slot"]), {})
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
            extracted = independent_extract_from_en(pred_en)
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
    t_metrics = field_metrics(y_true_t, y_pred_t, sorted(TIME_LABELS))
    f_metrics = field_metrics(y_true_f, y_pred_f, sorted(FREQ_LABELS))
    p_metrics = field_metrics(y_true_p, y_pred_p, sorted(PHON_LABELS))
    mean_fieldacc_macro_3 = mean([t_metrics["accuracy"], f_metrics["accuracy"], p_metrics["accuracy"]]) if n_regions else 0.0

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

    caption = caption_scores(gt_caps, pred_caps)

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
            "MeanFieldAcc_macro": mean_fieldacc_macro_3,
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
        "caption_quality_en": caption,
    }

    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"saved_metrics: {out_json}")


if __name__ == "__main__":
    main()
