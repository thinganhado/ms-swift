#!/usr/bin/env python3
import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_PROMPT = "Select the top 3 regions that most likely contain spoof artifacts."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MS-Swift SFT dataset for query1 directly from a CSV."
    )
    parser.add_argument("--input-csv", required=True, help="Input CSV path.")
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
    parser.add_argument("--image-col", default="img_path", help="CSV column containing image path.")
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

    records: List[Dict] = []
    path_aware_rows: List[tuple[str, Dict]] = []
    skipped = 0

    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            image_path = str(row.get(args.image_col, "")).strip()
            target_raw = _extract_target(row, args.regions_col)
            if not image_path or not target_raw:
                skipped += 1
                continue
            target = _to_json_array_string(target_raw) if args.json_array_target else target_raw
            rec = _build_record(idx=idx, image_path=image_path, target=target, user_prompt=args.user_prompt)
            records.append(rec)
            path_aware_rows.append((image_path, rec))

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
