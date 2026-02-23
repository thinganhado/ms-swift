#!/usr/bin/env python3
import argparse
import csv
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_PROMPT = "Select the top 3 regions that most likely contain spoof artifacts."
DEFAULT_INPUT_CSV = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/final_mask_topk/region_phone_table_topk3.csv"
DEFAULT_ORDER_CSV = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/final_mask_topk/region_diff_stats.csv"
DEFAULT_IMAGE_DIR = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/specs/grid"
DEFAULT_IMAGE_SUFFIX = "_grid_img_edge_number_axes.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MS-Swift SFT dataset for query1 directly from a CSV."
    )
    parser.add_argument(
        "--input-csv",
        default=DEFAULT_INPUT_CSV,
        help=f"Input CSV path. Default: {DEFAULT_INPUT_CSV}",
    )
    parser.add_argument("--output-json", default=None, help="Output JSON path (single-file mode).")
    parser.add_argument("--train-json", default=None, help="Train JSON output path (split mode).")
    parser.add_argument("--val-json", default=None, help="Validation JSON output path (split mode).")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio for random split mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for random split mode.")
    parser.add_argument("--split-by-path", action="store_true", help="Split train/val by path keys.")
    parser.add_argument("--train-key", default="_LA_T_", help="Path substring routed to train.")
    parser.add_argument("--val-key", default="_LA_D_", help="Path substring routed to val.")
    parser.add_argument(
        "--strict-path-split",
        action="store_true",
        help="Drop unmatched samples when --split-by-path is set.",
    )
    parser.add_argument(
        "--image-col",
        default="img_path",
        help="CSV column containing full image path. If missing/empty, sample_id+region_id mode is used.",
    )
    parser.add_argument("--sample-id-col", default="sample_id", help="CSV column containing sample id.")
    parser.add_argument("--region-id-col", default="region_id", help="CSV column containing region id.")
    parser.add_argument(
        "--order-csv",
        default=DEFAULT_ORDER_CSV,
        help=f"Optional order CSV used for ranking. Default: {DEFAULT_ORDER_CSV}",
    )
    parser.add_argument(
        "--order-sample-id-col",
        default="image",
        help="Sample-id column in --order-csv.",
    )
    parser.add_argument(
        "--order-region-id-col",
        default="region_id",
        help="Region-id column in --order-csv.",
    )
    parser.add_argument(
        "--order-score-col",
        default="region_pixels",
        help="Score column in --order-csv. Higher means earlier rank.",
    )
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR, help="Image directory for sample_id mode.")
    parser.add_argument(
        "--image-suffix",
        default=DEFAULT_IMAGE_SUFFIX,
        help="Image filename suffix for sample_id mode.",
    )
    parser.add_argument(
        "--regions-col",
        default="",
        help="CSV column containing region target text, e.g. '13,1,2'. Auto-detect if omitted.",
    )
    parser.add_argument("--user-prompt", default=DEFAULT_PROMPT, help="Query1 user prompt.")
    parser.add_argument(
        "--json-array-target",
        action="store_true",
        help="Write assistant target as JSON array string, e.g. [13, 1, 2].",
    )
    return parser.parse_args()


def _to_json_array_string(raw: str) -> str:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    vals = []
    for p in parts:
        vals.append(int(p) if p.isdigit() else p)
    return json.dumps(vals)


def _extract_target(row: Dict[str, str], regions_col: str) -> Optional[str]:
    if regions_col:
        val = str(row.get(regions_col, "")).strip()
        return val or None

    for col in ("regions", "topk3", "top3", "pred_top3", "region_topk3"):
        val = str(row.get(col, "")).strip()
        if val:
            return val

    multi_col_candidates = [
        ("topk1", "topk2", "topk3"),
        ("top1", "top2", "top3"),
        ("region1", "region2", "region3"),
        ("r1", "r2", "r3"),
    ]
    for cols in multi_col_candidates:
        vals: List[str] = []
        for c in cols:
            v = str(row.get(c, "")).strip()
            if not v:
                vals = []
                break
            vals.append(v)
        if vals:
            return ",".join(vals)

    return None


def _region_sort_key(x: str) -> Tuple[int, object]:
    if x.isdigit():
        return (0, int(x))
    return (1, x)


def _build_records_from_img_rows(
    reader: csv.DictReader,
    image_col: str,
    regions_col: str,
    user_prompt: str,
    json_array_target: bool,
) -> Tuple[List[Dict], List[Tuple[str, Dict]], int]:
    records: List[Dict] = []
    path_aware_rows: List[Tuple[str, Dict]] = []
    skipped = 0

    for idx, row in enumerate(reader):
        image_path = str(row.get(image_col, "")).strip()
        target_raw = _extract_target(row, regions_col)
        if not image_path or not target_raw:
            skipped += 1
            continue
        target = _to_json_array_string(target_raw) if json_array_target else target_raw
        rec = _build_record(idx=idx, image_path=image_path, target=target, user_prompt=user_prompt)
        records.append(rec)
        path_aware_rows.append((image_path, rec))
    return records, path_aware_rows, skipped


def _build_records_from_sample_rows(
    reader: csv.DictReader,
    sample_id_col: str,
    region_id_col: str,
    order_scores: Dict[Tuple[str, str], float],
    image_dir: str,
    image_suffix: str,
    user_prompt: str,
    json_array_target: bool,
) -> Tuple[List[Dict], List[Tuple[str, Dict]], int]:
    sample_to_regions: "OrderedDict[str, List[str]]" = OrderedDict()
    skipped = 0

    for row in reader:
        sample_id = str(row.get(sample_id_col, "")).strip()
        region_id = str(row.get(region_id_col, "")).strip()
        if not sample_id or not region_id:
            skipped += 1
            continue
        if sample_id not in sample_to_regions:
            sample_to_regions[sample_id] = []
        if region_id not in sample_to_regions[sample_id]:
            sample_to_regions[sample_id].append(region_id)

    records: List[Dict] = []
    path_aware_rows: List[Tuple[str, Dict]] = []
    idx = 0
    for sample_id, region_list in sample_to_regions.items():
        if len(region_list) < 3:
            skipped += 1
            continue
        ranked_regions = sorted(
            region_list,
            key=lambda rid: (-order_scores.get((sample_id, rid), float("-inf")), _region_sort_key(rid)),
        )
        top3 = ranked_regions[:3]
        target_raw = ",".join(top3)
        target = _to_json_array_string(target_raw) if json_array_target else target_raw
        image_path = str(Path(image_dir) / f"{sample_id}{image_suffix}")
        rec = _build_record(idx=idx, image_path=image_path, target=target, user_prompt=user_prompt)
        records.append(rec)
        path_aware_rows.append((image_path, rec))
        idx += 1

    return records, path_aware_rows, skipped


def _load_order_scores(
    order_csv: str,
    sample_id_col: str,
    region_id_col: str,
    score_col: str,
) -> Dict[Tuple[str, str], float]:
    if not order_csv:
        return {}

    order_path = Path(order_csv).expanduser()
    if not order_path.exists():
        print(f"warning: order-csv not found, fallback to id tie-break only: {order_path}")
        return {}

    out: Dict[Tuple[str, str], float] = {}
    with order_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row.get(sample_id_col, "")).strip()
            region_id = str(row.get(region_id_col, "")).strip()
            score_raw = str(row.get(score_col, "")).strip()
            if not sample_id or not region_id or not score_raw:
                continue
            try:
                score = float(score_raw)
            except ValueError:
                continue
            key = (sample_id, region_id)
            old = out.get(key, float("-inf"))
            if score > old:
                out[key] = score

    print(f"loaded order scores: {len(out)} pairs from {order_path}")
    return out


def _build_record(idx: int, image_path: str, target: str, user_prompt: str) -> Dict:
    return {
        "sample_id": f"stage1_q1_{idx:06d}",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_prompt},
                ],
            },
            {"role": "assistant", "content": target},
        ],
    }


def _write_json(path: Path, data: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input_csv).expanduser()
    order_scores = _load_order_scores(
        order_csv=args.order_csv,
        sample_id_col=args.order_sample_id_col,
        region_id_col=args.order_region_id_col,
        score_col=args.order_score_col,
    )

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        use_img_mode = bool(args.image_col) and args.image_col in fieldnames
        if use_img_mode:
            records, path_aware_rows, skipped = _build_records_from_img_rows(
                reader=reader,
                image_col=args.image_col,
                regions_col=args.regions_col,
                user_prompt=args.user_prompt,
                json_array_target=args.json_array_target,
            )
            print(f"mode: img_path (image_col='{args.image_col}')")
        else:
            records, path_aware_rows, skipped = _build_records_from_sample_rows(
                reader=reader,
                sample_id_col=args.sample_id_col,
                region_id_col=args.region_id_col,
                order_scores=order_scores,
                image_dir=args.image_dir,
                image_suffix=args.image_suffix,
                user_prompt=args.user_prompt,
                json_array_target=args.json_array_target,
            )
            print(
                "mode: sample_id+region_id "
                f"(sample_id_col='{args.sample_id_col}', region_id_col='{args.region_id_col}')"
            )

    split_mode = args.train_json is not None and args.val_json is not None
    if split_mode:
        if args.split_by_path:
            train_data: List[Dict] = []
            val_data: List[Dict] = []
            unmatched: List[Dict] = []
            for image_path, rec in path_aware_rows:
                if args.train_key in image_path:
                    train_data.append(rec)
                elif args.val_key in image_path:
                    val_data.append(rec)
                else:
                    unmatched.append(rec)
            if unmatched and not args.strict_path_split:
                train_data.extend(unmatched)

            print(f"path split: train_key='{args.train_key}', val_key='{args.val_key}'")
            print(f"unmatched: {len(unmatched)}")
            if args.strict_path_split and unmatched:
                print("dropped unmatched due to --strict-path-split")
        else:
            random.seed(args.seed)
            random.shuffle(records)
            n_val = int(len(records) * args.val_ratio)
            val_data = records[:n_val]
            train_data = records[n_val:]

        train_path = Path(args.train_json).expanduser()
        val_path = Path(args.val_json).expanduser()
        _write_json(train_path, train_data)
        _write_json(val_path, val_data)
        print(f"saved train: {len(train_data)} -> {train_path}")
        print(f"saved val:   {len(val_data)} -> {val_path}")
        print(f"skipped rows: {skipped}")
        return

    if args.output_json is None:
        raise ValueError("Provide --output-json, or provide both --train-json and --val-json.")

    output_path = Path(args.output_json).expanduser()
    _write_json(output_path, records)
    print(f"saved: {output_path} (n={len(records)})")
    print(f"skipped rows: {skipped}")


if __name__ == "__main__":
    main()
