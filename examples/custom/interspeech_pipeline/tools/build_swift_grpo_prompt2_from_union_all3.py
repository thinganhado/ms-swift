#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_GT_CSV = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/final_mask_topk/region_phone_table_topk3.csv"
DEFAULT_DIFF_CSV = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/final_mask_topk/region_diff_stats.csv"
DEFAULT_UNION_CSV = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final__En/union_all3_only.csv"
DEFAULT_OUT_JSON = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/final__En/grpo2_tset_full3only_train.json"
DEFAULT_SPEC_ROOT = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/specs/grid"
DEFAULT_MFA_ROOT = "/scratch3/che489/Ha/interspeech/datasets/vocv4_mfa_aligned"
DEFAULT_USER_TEMPLATE = (
    "Explain the spoof artifact for each of the three selected region IDs in {prompt1_output} . "
    "This is the transcript for context: {transcript}"
)


def norm(x) -> str:
    return str(x or "").strip()


def to_one_line(text: str) -> str:
    return " ".join(str(text or "").split())


def normalize_explanation(explanation: str) -> str:
    s = str(explanation or "").strip()
    s = re.sub(r"</?Explanation>", "", s, flags=re.IGNORECASE).strip()
    s = s.replace("\\\\", "\\").replace('\\"', '"').replace("\\'", "'")
    s = re.sub(r'"([^"]+)"', r"[w:\1]", s)
    s = s.replace('"', "")
    return to_one_line(s)


def extract_transcript_word_tier_like_qwen_region(mfa_root: Path, sample_id: str) -> str:
    p = mfa_root / f"{sample_id}.json"
    if not p.exists():
        return ""
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ""

    if isinstance(obj, dict):
        entries = obj.get("tiers", {}).get("words", {}).get("entries", [])
        parts: List[str] = []
        for e in entries:
            if not isinstance(e, (list, tuple)) or len(e) < 3:
                continue
            try:
                start = float(e[0])
                end = float(e[1])
                word = str(e[2]).strip()
            except Exception:
                continue
            if word:
                parts.append(f"[{start:.2f}-{end:.2f}] {word}")
        if parts:
            return to_one_line(" ".join(parts))

    if isinstance(obj, str):
        return to_one_line(obj)
    return ""


def pair_key(row: Dict) -> Optional[Tuple[str, int]]:
    sid = norm(row.get("sample_id"))
    try:
        rid = int(norm(row.get("region_id")))
    except Exception:
        return None
    return sid, rid


def parse_args():
    ap = argparse.ArgumentParser(description="Build GRPO2 dataset from union_all3 CSV with overlap-based region ordering.")
    ap.add_argument("--gt-csv", default=DEFAULT_GT_CSV)
    ap.add_argument("--diff-csv", default=DEFAULT_DIFF_CSV)
    ap.add_argument("--union-csv", default=DEFAULT_UNION_CSV)
    ap.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    ap.add_argument("--spec-root", default=DEFAULT_SPEC_ROOT)
    ap.add_argument("--mfa-root", default=DEFAULT_MFA_ROOT)
    ap.add_argument("--user-template", default=DEFAULT_USER_TEMPLATE)
    ap.add_argument(
        "--sample-id-contains",
        default="_LA_T_",
        help="Filter samples by substring in sample_id. Use empty string to disable.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    gt_csv = Path(args.gt_csv).expanduser().resolve()
    diff_csv = Path(args.diff_csv).expanduser().resolve()
    union_csv = Path(args.union_csv).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()
    spec_root = Path(args.spec_root).expanduser().resolve()
    mfa_root = Path(args.mfa_root).expanduser().resolve()
    user_template = str(args.user_template)
    sample_filter = str(args.sample_id_contains or "")

    overlap_map: Dict[Tuple[str, int], float] = {}
    with diff_csv.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            sid = norm(r.get("sample_id"))
            try:
                rid = int(norm(r.get("region_id")))
            except Exception:
                continue
            try:
                ov = float(norm(r.get("overlap_pixels")))
            except Exception:
                ov = -1.0
            overlap_map[(sid, rid)] = ov

    gt_ids_by_sid: "OrderedDict[str, List[int]]" = OrderedDict()
    with gt_csv.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            sid = norm(r.get("sample_id"))
            if not sid:
                continue
            if sample_filter and sample_filter not in sid:
                continue
            try:
                rid = int(norm(r.get("region_id")))
            except Exception:
                continue
            gt_ids_by_sid.setdefault(sid, [])
            if rid not in gt_ids_by_sid[sid]:
                gt_ids_by_sid[sid].append(rid)

    ordered_gt3_by_sid: "OrderedDict[str, List[int]]" = OrderedDict()
    for sid, ids in gt_ids_by_sid.items():
        if len(ids) < 3:
            continue
        ids_sorted = sorted(ids, key=lambda rid: (-overlap_map.get((sid, rid), -1.0), rid))
        ordered_gt3_by_sid[sid] = ids_sorted[:3]

    union_index: Dict[Tuple[str, int], Dict] = {}
    transcript_by_sid: Dict[str, str] = {}
    with union_csv.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            k = pair_key(r)
            if not k:
                continue
            sid, rid = k
            union_index[(sid, rid)] = r
            t = to_one_line(norm(r.get("transcript")))
            if t and sid not in transcript_by_sid:
                transcript_by_sid[sid] = t

    out = []
    missing_full3 = 0
    missing_fields = 0

    for sid, ids in ordered_gt3_by_sid.items():
        if not all((sid, rid) in union_index for rid in ids):
            missing_full3 += 1
            continue

        rows = [union_index[(sid, rid)] for rid in ids]
        tuples = []
        ok = True
        for r in rows:
            rid = norm(r.get("region_id"))
            t = norm(r.get("T"))
            fband = norm(r.get("F"))
            p = norm(r.get("P_type"))
            en_raw = norm(r.get("refined_explanation")) or norm(r.get("output_explanation")) or norm(r.get("En"))
            en = normalize_explanation(en_raw)
            if not (rid and t and fband and p and en):
                ok = False
                break
            tuples.append(f'(Cn={rid}, T={t}, F={fband}, P={p}, En="{en}")')
        if not ok:
            missing_fields += 1
            continue

        prompt1_output = f"[{', '.join(map(str, ids))}]"
        transcript = transcript_by_sid.get(sid) or extract_transcript_word_tier_like_qwen_region(mfa_root, sid)
        transcript = to_one_line(transcript)
        user_text = user_template.format(prompt1_output=prompt1_output, transcript=transcript)

        out.append(
            {
                "sample_id": sid,
                "prompt1_output": prompt1_output,
                "transcript": transcript,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str((spec_root / f"{sid}_grid_img_edge_number_axes.png").as_posix())},
                            {"type": "text", "text": user_text},
                        ],
                    }
                ],
                "gt_prompt2": "; ".join(tuples),
            }
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("saved:", out_json)
    print("total_samples_with_gt>=3:", len(ordered_gt3_by_sid))
    print("kept_full3_samples:", len(out))
    print("dropped_missing_full3:", missing_full3)
    print("dropped_missing_fields:", missing_fields)


if __name__ == "__main__":
    main()
