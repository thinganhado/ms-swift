#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Q1 localization errors by comparing predicted top-3 region IDs against GT."
    )
    parser.add_argument("--pred-meta-json", required=True, help="Prediction-aware JSON containing predicted prompt1_output.")
    parser.add_argument("--gt-meta-json", required=True, help="GT JSON containing gold prompt1_output.")
    parser.add_argument("--top-k", type=int, default=10, help="How many top confusion pairs to print.")
    parser.add_argument("--output-json", help="Optional path to save the summary JSON.")
    return parser.parse_args()


def norm(value):
    return str(value or "").strip()


def parse_ids(text):
    return [int(x) for x in re.findall(r"\d+", norm(text))][:3]


def region_to_rc(region_id):
    # 4x4 grid, ids 1..16 in row-major order.
    idx = region_id - 1
    return divmod(idx, 4)


def manhattan_distance(a, b):
    ar, ac = region_to_rc(a)
    br, bc = region_to_rc(b)
    return abs(ar - br) + abs(ac - bc)


def ratio(numer, denom):
    return (numer / denom) if denom else 0.0


def main():
    args = parse_args()
    pred_rows = json.loads(Path(args.pred_meta_json).read_text(encoding="utf-8"))
    gt_rows = json.loads(Path(args.gt_meta_json).read_text(encoding="utf-8"))
    if not isinstance(pred_rows, list) or not isinstance(gt_rows, list):
        raise ValueError("Both pred-meta-json and gt-meta-json must be JSON arrays.")

    gt_by_sample = {}
    for row in gt_rows:
        if not isinstance(row, dict):
            continue
        sample_id = norm(row.get("sample_id"))
        gt_by_sample[sample_id] = parse_ids(row.get("prompt1_output"))

    n_samples = 0
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
    missing_gt_rows = 0

    for row in pred_rows:
        if not isinstance(row, dict):
            continue
        sample_id = norm(row.get("sample_id"))
        pred_ids = parse_ids(row.get("prompt1_output"))
        gt_ids = gt_by_sample.get(sample_id)
        if gt_ids is None or len(gt_ids) < 3:
            missing_gt_rows += 1
            continue
        if len(pred_ids) < 3:
            pred_ids = pred_ids + [-1] * (3 - len(pred_ids))

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

    summary = {
        "num_samples_scored": n_samples,
        "missing_gt_rows": missing_gt_rows,
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
            for (gt_rid, pred_rid), count in gt_to_pred_confusion.most_common(args.top_k)
        ],
        "top_confusions_by_slot": {
            f"slot{slot}": [
                {"gt_region": gt_rid, "pred_region": pred_rid, "count": count}
                for (gt_rid, pred_rid), count in counter.most_common(args.top_k)
            ]
            for slot, counter in rank_confusion.items()
        },
        "most_common_gt_regions": [
            {"region": rid, "count": count} for rid, count in gt_region_count.most_common(args.top_k)
        ],
        "most_common_pred_regions": [
            {"region": rid, "count": count} for rid, count in pred_region_count.most_common(args.top_k)
        ],
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_summary: {out_path}")


if __name__ == "__main__":
    main()
