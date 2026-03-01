#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


EN_PAT = re.compile(r'\bEn([123])\s*=\s*"((?:[^"\\]|\\.)*)"', re.I | re.S)
REGION_PAT = re.compile(r'(?i)\bRegion ID\s+(\d+)\b')


def norm(value):
    return str(value or "").strip()


def get_response(obj):
    if isinstance(obj.get("response"), str):
        return obj["response"]
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            return norm(msg.get("content"))
    return ""


def parse_prompt_ids(text):
    return [int(x) for x in re.findall(r"\d+", norm(text))][:3]


def parse_response_regions(response, prompt_ids):
    slot_matches = {int(i): norm(txt) for i, txt in EN_PAT.findall(response)}
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

    matches = list(REGION_PAT.finditer(response))
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
        if explanation:
            rows.append((slot, region_id, explanation))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Build Q2 verifier input JSONL from model generation outputs.")
    parser.add_argument("--meta-json", required=True, help="Path to source dataset JSON.")
    parser.add_argument("--raw-result-jsonl", required=True, help="Path to infer_result.jsonl.")
    parser.add_argument("--output-jsonl", required=True, help="Path to write verifier input JSONL.")
    args = parser.parse_args()

    meta_path = Path(args.meta_json)
    raw_path = Path(args.raw_result_jsonl)
    out_path = Path(args.output_jsonl)

    meta_rows = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta_rows, list):
        raise ValueError(f"Expected a JSON array in {meta_path}")

    out_rows = []
    with raw_path.open("r", encoding="utf-8") as f:
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
            prompt_ids = parse_prompt_ids(prompt1_output)

            for slot, region_id, explanation in parse_response_regions(response, prompt_ids):
                out_rows.append({
                    "sample_id": sample_id,
                    "region_id": region_id,
                    "response": f"<Explanation>{explanation}</Explanation>",
                    "raw_response": response,
                    "slot": slot,
                })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"saved_verifier_rows: {len(out_rows)}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
