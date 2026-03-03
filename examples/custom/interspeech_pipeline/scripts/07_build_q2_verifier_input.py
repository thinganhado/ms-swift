#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


SLOT_PATTERNS = {
    "T": re.compile(r"\bT([123])\s*=\s*([^,;)\n]+)", re.I),
    "F": re.compile(r"\bF([123])\s*=\s*([^,;)\n]+)", re.I),
    "P": re.compile(r"\bP([123])\s*=\s*([^,;)\n]+)", re.I),
    "EnIndexed": re.compile(r'\bEn([123])\s*=\s*"((?:[^"\\]|\\.)*)"', re.I | re.S),
}


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


def parse_indexed_tuple_text(text):
    text = norm(text)
    out = {}
    for name, pattern in SLOT_PATTERNS.items():
        for idx, val in pattern.findall(text):
            idx = int(idx)
            out.setdefault(idx, {})
            if name == "EnIndexed":
                out[idx]["En"] = norm(val)
            else:
                out[idx][name] = norm(val)
    return out


def parse_response_regions(response, prompt_ids):
    if len(prompt_ids) < 3:
        return []
    parsed = parse_indexed_tuple_text(response)
    tuple_parts = [norm(part) for part in re.split(r"\)\s*;\s*\(", response)]
    rows = []
    for slot in (1, 2, 3):
        fields = parsed.get(slot, {})
        if not norm(fields.get("En")) and slot <= len(tuple_parts):
            m = re.search(r'\bEn\s*=\s*"((?:[^"\\]|\\.)*)"', tuple_parts[slot - 1], re.I | re.S)
            if m:
                fields["En"] = norm(m.group(1))
        # Enforce the exact expected structure: all four indexed fields must be present.
        if not all(norm(fields.get(name)) for name in ("T", "F", "P", "En")):
            return []
        rows.append((slot, prompt_ids[slot - 1], fields["En"]))
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

            parsed_rows = parse_response_regions(response, prompt_ids)
            if not parsed_rows:
                continue

            for slot, region_id, explanation in parsed_rows:
                out_rows.append({
                    "sample_id": sample_id,
                    "region_id": region_id,
                    "slot": slot,
                    "response": f"<Explanation>{explanation}</Explanation>",
                    "raw_response": response,
                    "prompt1_output": prompt1_output,
                })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"saved_verifier_rows: {len(out_rows)}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
