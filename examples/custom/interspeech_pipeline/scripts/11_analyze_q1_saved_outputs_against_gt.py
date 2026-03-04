#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


REGION_PATTERNS = [
    re.compile(r"\bRegion(?:\s+ID)?\s*\[?(\d+)\]?\b", re.I),
    re.compile(r"\bregions?\s*[:\-]?\s*\[([0-9,\s]+)\]", re.I),
    re.compile(r"^\s*\[([0-9,\s]+)\]\s*$"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Q1 predictions from merged_for_eval or analyze extracted predictions against GT."
    )
    parser.add_argument(
        "--mode",
        choices=("extract", "analyze"),
        required=True,
        help="extract: read merged_for_eval and write parsed predictions; analyze: compare parsed predictions against GT.",
    )
    parser.add_argument(
        "--q1-output-root",
        help="Root directory containing saved Q1 response JSONs (e.g. merged_for_eval). Required in extract mode.",
    )
    parser.add_argument(
        "--pred-json",
        help="Parsed Q1 prediction JSON. Written in extract mode, read in analyze mode.",
    )
    parser.add_argument(
        "--gt-meta-json",
        help="GT Q1 metadata JSON (e.g. stage1_query1_val_swift.json). Required in analyze mode.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="How many top confusion pairs to report.")
    parser.add_argument("--output-json", help="Optional output JSON path for extracted predictions or analysis summary.")
    parser.add_argument(
        "--debug-limit",
        type=int,
        default=20,
        help="How many skipped examples to include in extract-mode debug output.",
    )
    return parser.parse_args()


def norm(value):
    return str(value or "").strip()


def extract_ints(text):
    return [int(x) for x in re.findall(r"-?\d+", text or "")]


def parse_predicted_ids(response: str):
    ids = []
    for pattern in REGION_PATTERNS:
        for match in pattern.finditer(response):
            if match.lastindex == 1 and pattern is REGION_PATTERNS[0]:
                try:
                    ids.append(int(match.group(1)))
                except Exception:
                    pass
            else:
                for token in re.findall(r"\d+", match.group(1)):
                    try:
                        ids.append(int(token))
                    except Exception:
                        pass
    deduped = []
    seen = set()
    for rid in ids:
        if rid in seen:
            continue
        seen.add(rid)
        deduped.append(rid)
    if len(deduped) < 3:
        for rid in re.findall(r"\d+", response or ""):
            try:
                rid = int(rid)
            except Exception:
                continue
            if rid in seen:
                continue
            seen.add(rid)
            deduped.append(rid)
            if len(deduped) == 3:
                break
    return deduped[:3]


def normalize_predicted_ids(pred_ids):
    cleaned = []
    seen = set()
    for rid in pred_ids:
        try:
            rid = int(rid)
        except Exception:
            continue
        if rid in seen:
            continue
        if not (1 <= rid <= 16):
            continue
        seen.add(rid)
        cleaned.append(rid)
        if len(cleaned) == 3:
            break
    return cleaned


def region_to_rc(region_id):
    idx = region_id - 1
    return divmod(idx, 4)


def manhattan_distance(a, b):
    ar, ac = region_to_rc(a)
    br, bc = region_to_rc(b)
    return abs(ar - br) + abs(ac - bc)


def ratio(numer, denom):
    return (numer / denom) if denom else 0.0


def iter_payloads(root: Path):
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name not in {"json", "json.json"} and path.suffix.lower() != ".json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        yield path, payload


def extract_rows(q1_root: Path, debug_limit: int = 20):
    rows = []
    skipped = 0
    skip_reasons = Counter()
    skipped_examples = []
    for path, payload in iter_payloads(q1_root):
        sample_id = norm(payload.get("sample_id"))
        response = norm(payload.get("response"))
        if not response:
            skipped += 1
            skip_reasons["empty_response"] += 1
            if len(skipped_examples) < debug_limit:
                skipped_examples.append(
                    {
                        "reason": "empty_response",
                        "sample_id": sample_id,
                        "source_file": path.as_posix(),
                        "raw_response": response,
                    }
                )
            continue
        pred_ids = normalize_predicted_ids(parse_predicted_ids(response))
        if len(pred_ids) < 3:
            skipped += 1
            skip_reasons["parsed_fewer_than_3_valid_ids"] += 1
            if len(skipped_examples) < debug_limit:
                skipped_examples.append(
                    {
                        "reason": "parsed_fewer_than_3_valid_ids",
                        "sample_id": sample_id,
                        "source_file": path.as_posix(),
                        "raw_response": response[:500],
                        "parsed_ids_before_filter": parse_predicted_ids(response),
                        "parsed_ids_after_filter": pred_ids,
                    }
                )
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "img_path": norm(payload.get("img_path")),
                "source_file": path.as_posix(),
                "raw_response": response,
                "prompt1_output": f"[{', '.join(map(str, pred_ids))}]",
            }
        )
    return rows, skipped, skip_reasons, skipped_examples


def extract_gt_ids(row):
    if not isinstance(row, dict):
        return []
    if row.get("prompt1_output"):
        return extract_ints(norm(row.get("prompt1_output")))[:3]
    for key in ("labels", "response", "target"):
        value = row.get(key)
        if isinstance(value, list):
            ints = []
            for item in value:
                try:
                    ints.append(int(item))
                except Exception:
                    pass
            if ints:
                return ints[:3]
        if isinstance(value, str):
            ints = extract_ints(value)
            if ints:
                return ints[:3]

    messages = row.get("messages") or row.get("conversations") or []
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = norm(msg.get("role") or msg.get("from")).lower()
            if role not in {"assistant", "gpt"}:
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        ints = extract_ints(norm(part.get("text")))
                        if ints:
                            return ints[:3]
            elif isinstance(content, str):
                ints = extract_ints(content)
                if ints:
                    return ints[:3]
            value = msg.get("value")
            if isinstance(value, str):
                ints = extract_ints(value)
                if ints:
                    return ints[:3]
    return []


def analyze_rows(pred_rows, gt_rows, top_k: int):
    gt_by_sample = {}
    for row in gt_rows:
        if not isinstance(row, dict):
            continue
        sample_id = norm(row.get("sample_id"))
        if not sample_id:
            continue
        gt_ids = extract_gt_ids(row)
        if len(gt_ids) >= 3:
            gt_by_sample[sample_id] = gt_ids[:3]

    pred_by_sample = {}
    for row in pred_rows:
        if not isinstance(row, dict):
            continue
        sample_id = norm(row.get("sample_id"))
        pred_ids = extract_ints(norm(row.get("prompt1_output")))[:3]
        if sample_id and len(pred_ids) >= 3:
            pred_by_sample[sample_id] = pred_ids

    n_samples = 0
    missing_gt_rows = 0
    missing_pred_rows = 0
    exact_match = 0
    same_set_wrong_order = 0
    overlap_bucket = Counter()
    slot_exact = Counter()
    gt_region_count = Counter()
    pred_region_count = Counter()
    gt_to_pred_confusion = Counter()
    rank_confusion = defaultdict(Counter)
    distance_hist = Counter()
    near_miss_same_row = 0
    near_miss_same_col = 0
    near_miss_count = 0

    for sample_id, gt_ids in gt_by_sample.items():
        pred_ids = pred_by_sample.get(sample_id)
        if pred_ids is None:
            missing_pred_rows += 1
            continue

        n_samples += 1

        for rid in gt_ids:
            gt_region_count[rid] += 1
        for rid in pred_ids:
            pred_region_count[rid] += 1

        overlap = len(set(pred_ids) & set(gt_ids))
        overlap_bucket[overlap] += 1

        if pred_ids[:3] == gt_ids[:3]:
            exact_match += 1
        elif set(pred_ids[:3]) == set(gt_ids[:3]):
            same_set_wrong_order += 1

        for idx in range(3):
            if pred_ids[idx] == gt_ids[idx]:
                slot_exact[idx + 1] += 1
            rank_confusion[idx + 1][(gt_ids[idx], pred_ids[idx])] += 1

        unmatched_pred = [rid for rid in pred_ids if rid not in gt_ids]
        unmatched_gt = [rid for rid in gt_ids if rid not in pred_ids]
        for gt_rid, pred_rid in zip(unmatched_gt, unmatched_pred):
            gt_to_pred_confusion[(gt_rid, pred_rid)] += 1
            if 1 <= gt_rid <= 16 and 1 <= pred_rid <= 16:
                dist = manhattan_distance(gt_rid, pred_rid)
                distance_hist[dist] += 1
                near_miss_count += 1
                gr, gc = region_to_rc(gt_rid)
                pr, pc = region_to_rc(pred_rid)
                if gr == pr:
                    near_miss_same_row += 1
                if gc == pc:
                    near_miss_same_col += 1

    return {
        "num_gt_rows": len(gt_by_sample),
        "num_pred_rows": len(pred_by_sample),
        "num_samples_scored": n_samples,
        "missing_gt_rows": missing_gt_rows,
        "missing_pred_rows": missing_pred_rows,
        "exact_match_count": exact_match,
        "exact_match_rate": ratio(exact_match, n_samples),
        "same_set_wrong_order_count": same_set_wrong_order,
        "same_set_wrong_order_rate": ratio(same_set_wrong_order, n_samples),
        "overlap_buckets": {
            "overlap_3": overlap_bucket[3],
            "overlap_2": overlap_bucket[2],
            "overlap_1": overlap_bucket[1],
            "overlap_0": overlap_bucket[0],
            "overlap_3_rate": ratio(overlap_bucket[3], n_samples),
            "overlap_2_rate": ratio(overlap_bucket[2], n_samples),
            "overlap_1_rate": ratio(overlap_bucket[1], n_samples),
            "overlap_0_rate": ratio(overlap_bucket[0], n_samples),
        },
        "slot_accuracy": {
            "slot1_exact": ratio(slot_exact[1], n_samples),
            "slot2_exact": ratio(slot_exact[2], n_samples),
            "slot3_exact": ratio(slot_exact[3], n_samples),
        },
        "near_miss": {
            "count": near_miss_count,
            "same_row_rate": ratio(near_miss_same_row, near_miss_count),
            "same_col_rate": ratio(near_miss_same_col, near_miss_count),
            "distance_histogram": {str(k): v for k, v in sorted(distance_hist.items())},
        },
        "top_confusions_overall": [
            {"gt_region": gt_rid, "pred_region": pred_rid, "count": count}
            for (gt_rid, pred_rid), count in gt_to_pred_confusion.most_common(top_k)
        ],
        "top_confusions_by_slot": {
            f"slot{slot}": [
                {"gt_region": gt_rid, "pred_region": pred_rid, "count": count}
                for (gt_rid, pred_rid), count in counter.most_common(top_k)
            ]
            for slot, counter in rank_confusion.items()
        },
        "most_common_gt_regions": [
            {"region": rid, "count": count} for rid, count in gt_region_count.most_common(top_k)
        ],
        "most_common_pred_regions": [
            {"region": rid, "count": count} for rid, count in pred_region_count.most_common(top_k)
        ],
    }


def main():
    args = parse_args()

    if args.mode == "extract":
        if not args.q1_output_root:
            raise ValueError("--q1-output-root is required in extract mode.")
        q1_root = Path(args.q1_output_root).expanduser().resolve()
        rows, skipped, skip_reasons, skipped_examples = extract_rows(q1_root, debug_limit=args.debug_limit)
        result = {
            "num_rows": len(rows),
            "skipped_rows": skipped,
            "skip_reasons": dict(skip_reasons),
            "skipped_examples": skipped_examples,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        out_path = Path(args.output_json or args.pred_json).expanduser().resolve() if (args.output_json or args.pred_json) else None
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"saved_predictions: {out_path}")
        return

    if not args.pred_json or not args.gt_meta_json:
        raise ValueError("--pred-json and --gt-meta-json are required in analyze mode.")
    pred_rows = json.loads(Path(args.pred_json).expanduser().resolve().read_text(encoding="utf-8"))
    gt_rows = json.loads(Path(args.gt_meta_json).expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(pred_rows, list) or not isinstance(gt_rows, list):
        raise ValueError("Both pred-json and gt-meta-json must be JSON arrays.")

    summary = analyze_rows(pred_rows, gt_rows, top_k=args.top_k)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_summary: {out_path}")


if __name__ == "__main__":
    main()
