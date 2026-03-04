#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


REGION_PATTERNS = [
    re.compile(r"\bRegion(?:\s+ID)?\s*\[?(\d+)\]?\b", re.I),
    re.compile(r"\bregions?\s*[:\-]?\s*\[([0-9,\s]+)\]", re.I),
]

USER_IDS_PATTERN = re.compile(
    r"(selected region IDs in )\[[^\]]*\]",
    re.I,
)


def norm(value):
    return str(value or "").strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a Q2 prediction metadata JSON by replacing prompt1_output using saved Q1 eval outputs."
    )
    parser.add_argument(
        "--q1-output-root",
        required=True,
        help="Root directory containing saved Q1 response JSONs (e.g. merged_for_eval).",
    )
    parser.add_argument(
        "--gt-meta-json",
        required=True,
        help="Ground-truth Q2 metadata JSON used as the template and GT source.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write the prediction-aware Q2 metadata JSON.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="If set, keep rows even when a Q1 prediction file is missing or unparsable.",
    )
    return parser.parse_args()


def find_q1_payload(root: Path, sample_id: str):
    candidates = [
        root / sample_id / "json",
        root / sample_id / "json.json",
        root / sample_id / f"{sample_id}.json",
        root / f"{sample_id}.json",
    ]
    for path in candidates:
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None

    sample_dir = root / sample_id
    if sample_dir.is_dir():
        for path in sorted(sample_dir.glob("*.json")):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


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
    return deduped[:3]


def clone_user_message_with_new_ids(msg, prompt1_output: str):
    if not isinstance(msg, dict):
        return {"role": "user", "content": []}
    content = msg.get("content")
    if isinstance(content, list):
        new_content = []
        for part in content:
            if not isinstance(part, dict):
                continue
            new_part = dict(part)
            if new_part.get("type") == "text":
                text = norm(new_part.get("text"))
                text = USER_IDS_PATTERN.sub(rf"\1{prompt1_output}", text, count=1)
                new_part["text"] = text
            new_content.append(new_part)
        return {"role": "user", "content": new_content}
    if isinstance(content, str):
        text = USER_IDS_PATTERN.sub(rf"\1{prompt1_output}", content, count=1)
        return {"role": "user", "content": [{"type": "text", "text": text}]}
    return {"role": "user", "content": []}


def main():
    args = parse_args()
    q1_root = Path(args.q1_output_root).expanduser().resolve()
    gt_meta_path = Path(args.gt_meta_json).expanduser().resolve()
    out_path = Path(args.output_json).expanduser().resolve()

    gt_rows = json.loads(gt_meta_path.read_text(encoding="utf-8"))
    if not isinstance(gt_rows, list):
        raise ValueError(f"Expected a JSON array in {gt_meta_path}")

    out_rows = []
    skipped = 0
    for row in gt_rows:
        if not isinstance(row, dict):
            continue
        sample_id = norm(row.get("sample_id"))
        payload = find_q1_payload(q1_root, sample_id)
        pred_ids = parse_predicted_ids(norm(payload.get("response"))) if isinstance(payload, dict) else []
        if len(pred_ids) < 3:
            if not args.allow_missing:
                skipped += 1
                continue
            pred_ids = [int(x) for x in re.findall(r"\d+", norm(row.get("prompt1_output")))[:3]]
        if len(pred_ids) < 3:
            skipped += 1
            continue

        prompt1_output = f"[{', '.join(map(str, pred_ids))}]"
        msgs = row.get("messages") or []
        user_msg = msgs[0] if msgs and isinstance(msgs[0], dict) else {"role": "user", "content": []}
        out_row = {
            "messages": [clone_user_message_with_new_ids(user_msg, prompt1_output)],
            "sample_id": sample_id,
            "prompt1_output": prompt1_output,
            "gt_prompt2": norm(row.get("gt_prompt2")),
        }
        if "transcript" in row:
            out_row["transcript"] = row["transcript"]
        out_rows.append(out_row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_path} (n={len(out_rows)})")
    print(f"skipped_rows: {skipped}")


if __name__ == "__main__":
    main()
