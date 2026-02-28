#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HARDCODED_QUERY2_USER_TEMPLATE = (
    "Explain the spoof artifact for each of the three selected region IDs in "
    "{prompt1_output} . This is the transcript for context: {transcript}"
)
DEFAULT_SPEC_ROOT = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/specs/grid"
DEFAULT_MFA_ROOT = "/scratch3/che489/Ha/interspeech/datasets/vocv4_mfa_aligned"


def _map_role(role: str) -> str:
    role = str(role or "").lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant"}:
        return "assistant"
    return role or "user"


def _extract_conv(item: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    conv = item.get("messages") or item.get("conversations")
    if isinstance(conv, list) and len(conv) >= 1:
        return conv
    return None


def _extract_text(msg: Dict[str, Any]) -> str:
    c = msg.get("content", msg.get("value", ""))
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        texts = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                texts.append(str(p.get("text", "")))
        return "\n".join([t for t in texts if t]).strip()
    return str(c or "")


def _split_top_level_tuples(text: str) -> List[str]:
    s = str(text or "")
    out: List[str] = []
    start: Optional[int] = None
    depth = 0
    in_quote = False
    escape = False
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    out.append(s[start:i + 1].strip())
                    start = None
    return out


def _strip_outer_parens(s: str) -> str:
    x = str(s or "").strip()
    if len(x) >= 2 and x[0] == "(" and x[-1] == ")":
        return x[1:-1].strip()
    return x


def _normalize_gt_prompt2_no_cn(text: str) -> str:
    tuples = _split_top_level_tuples(text)
    if not tuples:
        return str(text or "").strip()

    out: List[str] = []
    for idx, tup in enumerate(tuples, start=1):
        body = _strip_outer_parens(tup)

        m_t = re.search(rf"\bT{idx}\s*=\s*([^,]+?)\s*(?:,|$)", body, flags=re.IGNORECASE) or re.search(
            r"\bT\s*=\s*([^,]+?)\s*(?:,|$)", body, flags=re.IGNORECASE
        )
        m_f = re.search(rf"\bF{idx}\s*=\s*([^,]+?)\s*(?:,|$)", body, flags=re.IGNORECASE) or re.search(
            r"\bF\s*=\s*([^,]+?)\s*(?:,|$)", body, flags=re.IGNORECASE
        )
        m_p = re.search(rf"\bP{idx}\s*=\s*([^,]+?)\s*(?:,|$)", body, flags=re.IGNORECASE) or re.search(
            r"\bP\s*=\s*([^,]+?)\s*(?:,|$)", body, flags=re.IGNORECASE
        )
        m_en = re.search(rf"\bEn{idx}\s*=\s*(.+)\s*$", body, flags=re.IGNORECASE | re.DOTALL) or re.search(
            r"\bEn\s*=\s*(.+)\s*$", body, flags=re.IGNORECASE | re.DOTALL
        )
        if m_t and m_f and m_p and m_en:
            t = m_t.group(1).strip()
            fband = m_f.group(1).strip()
            p = m_p.group(1).strip()
            en = m_en.group(1).strip()
            out.append(f'(T{idx}={t}, F{idx}={fband}, P{idx}={p}, En{idx}={en})')
            continue

        parts = [p.strip() for p in body.split(",", 4)]
        if len(parts) == 5:
            _, t, fband, p, en = parts
            out.append(f'(T{idx}={t}, F{idx}={fband}, P{idx}={p}, En{idx}={en})')
            continue
        if len(parts) == 4:
            t, fband, p, en = parts
            out.append(f'(T{idx}={t}, F{idx}={fband}, P{idx}={p}, En{idx}={en})')
            continue

        out.append(tup.strip())
    return "; ".join(out)


def _extract_image(item: Dict[str, Any], user_msg: Dict[str, Any]) -> Optional[str]:
    for k in ("image", "img_path", "p1"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    c = user_msg.get("content", user_msg.get("value", ""))
    if isinstance(c, list):
        for p in c:
            if isinstance(p, dict) and p.get("type") == "image":
                img = p.get("image")
                if isinstance(img, str) and img.strip():
                    return img.strip()
    return None


def _build_user_message(item: Dict[str, Any], user_msg: Dict[str, Any], image: Optional[str]) -> Dict[str, Any]:
    role = _map_role(user_msg.get("role", user_msg.get("from", "")))
    prompt1_output = str(item.get("prompt1_output", "")).strip()
    transcript = str(item.get("transcript", "")).strip()
    text = HARDCODED_QUERY2_USER_TEMPLATE.format(
        prompt1_output=prompt1_output,
        transcript=transcript,
    )
    if image:
        return {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    return {"role": role, "content": text}


def _extract_gt_prompt2(item: Dict[str, Any], conv: List[Dict[str, Any]]) -> str:
    for k in ("gt_prompt2", "prompt2_target", "assistant"):
        if k in item and str(item[k]).strip():
            return _normalize_gt_prompt2_no_cn(str(item[k]).strip())
    if len(conv) >= 2:
        return _normalize_gt_prompt2_no_cn(_extract_text(conv[1]))
    return ""


def _norm(x) -> str:
    return str(x or "").strip()


def _to_one_line(text: str) -> str:
    return " ".join(str(text or "").split())


def _normalize_explanation(explanation: str) -> str:
    s = str(explanation or "").strip()
    s = re.sub(r"</?Explanation>", "", s, flags=re.IGNORECASE).strip()
    s = s.replace("\\\\", "\\").replace('\\"', '"').replace("\\'", "'")
    s = re.sub(r'"([^"]+)"', r"[w:\1]", s)
    s = s.replace('"', "")
    return _to_one_line(s)


def _extract_transcript_word_tier_like_qwen_region(mfa_root: Path, sample_id: str) -> str:
    p = mfa_root / f"{sample_id}.json"
    if not p.exists():
        return ""
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return ""

    if isinstance(obj, dict):
        entries = obj.get("tiers", {}).get("words", {}).get("entries", [])
        parts = []
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
            return _to_one_line(" ".join(parts))
    if isinstance(obj, str):
        return _to_one_line(obj)
    return ""


def _pair_key(row: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    sid = _norm(row.get("sample_id"))
    try:
        rid = int(_norm(row.get("region_id")))
    except Exception:
        return None
    return sid, rid


def _build_from_csv(args: argparse.Namespace) -> List[Dict[str, Any]]:
    src = Path(args.input_json).expanduser().resolve()
    spec_root = Path(args.spec_root).expanduser().resolve()
    mfa_root = Path(args.mfa_root).expanduser().resolve()
    sample_filter = str(args.sample_id_contains or "")
    gt_csv = Path(args.gt_csv).expanduser().resolve() if args.gt_csv else None

    # Optional overlap ordering map.
    overlap_map: Dict[Tuple[str, int], float] = {}
    if args.diff_csv:
        diff_csv = Path(args.diff_csv).expanduser().resolve()
        if diff_csv.exists():
            with diff_csv.open("r", encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    sid = _norm(r.get("sample_id"))
                    try:
                        rid = int(_norm(r.get("region_id")))
                    except Exception:
                        continue
                    try:
                        ov = float(_norm(r.get("overlap_pixels")))
                    except Exception:
                        ov = -1.0
                    overlap_map[(sid, rid)] = ov

    rows_by_pair: Dict[Tuple[str, int], Dict[str, Any]] = {}
    ids_by_sid: "OrderedDict[str, List[int]]" = OrderedDict()
    transcript_by_sid: Dict[str, str] = {}

    with src.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            k = _pair_key(r)
            if not k:
                continue
            sid, rid = k
            if sample_filter and sample_filter not in sid:
                continue

            rows_by_pair[(sid, rid)] = r

            t = _to_one_line(_norm(r.get("transcript")))
            if t and sid not in transcript_by_sid:
                transcript_by_sid[sid] = t

    # ID source:
    # - default: input csv rows
    # - if --gt-csv provided: use GT csv IDs/order source, and still pull fields from input csv rows.
    if gt_csv and gt_csv.exists():
        with gt_csv.open("r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                sid = _norm(r.get("sample_id"))
                if sample_filter and sample_filter not in sid:
                    continue
                try:
                    rid = int(_norm(r.get("region_id")))
                except Exception:
                    continue
                ids_by_sid.setdefault(sid, [])
                if rid not in ids_by_sid[sid]:
                    ids_by_sid[sid].append(rid)
    else:
        for sid, rid in rows_by_pair.keys():
            ids_by_sid.setdefault(sid, [])
            if rid not in ids_by_sid[sid]:
                ids_by_sid[sid].append(rid)

    out: List[Dict[str, Any]] = []
    for sid, ids in ids_by_sid.items():
        if len(ids) < 3:
            continue

        if overlap_map:
            ids = sorted(ids, key=lambda rid: (-overlap_map.get((sid, rid), -1.0), rid))[:3]
        else:
            ids = ids[:3]

        if not all((sid, rid) in rows_by_pair for rid in ids):
            continue
        ordered_rows = [rows_by_pair[(sid, rid)] for rid in ids]

        tuples = []
        ok = True
        for idx, r in enumerate(ordered_rows, start=1):
            rid = _norm(r.get("region_id"))
            t = _norm(r.get("T"))
            fband = _norm(r.get("F"))
            p = _norm(r.get("P_type"))
            en_raw = _norm(r.get("refined_explanation")) or _norm(r.get("output_explanation")) or _norm(r.get("En"))
            en = _normalize_explanation(en_raw)
            if not (rid and t and fband and p and en):
                ok = False
                break
            tuples.append(f'(T{idx}={t}, F{idx}={fband}, P{idx}={p}, En{idx}="{en}")')
        if not ok:
            continue

        prompt1_output = f"[{', '.join(map(str, ids))}]"
        transcript = transcript_by_sid.get(sid) or _extract_transcript_word_tier_like_qwen_region(mfa_root, sid)
        transcript = _to_one_line(transcript)
        user_text = HARDCODED_QUERY2_USER_TEMPLATE.format(
            prompt1_output=prompt1_output,
            transcript=transcript,
        )

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
    return out


def _build_from_json(args: argparse.Namespace) -> List[Dict[str, Any]]:
    src = Path(args.input_json).expanduser().resolve()
    data = json.loads(src.read_text(encoding="utf-8-sig"))

    out: List[Dict[str, Any]] = []
    for item in data:
        conv = _extract_conv(item)
        if conv is None:
            continue
        user_msg = conv[0]
        image = _extract_image(item, user_msg)
        gt = _extract_gt_prompt2(item, conv)
        if not gt:
            continue
        row = {
            "messages": [_build_user_message(item, user_msg, image)],
            "gt_prompt2": gt,
        }
        for k in ("sample_id", "sample_id_raw", "id", "prompt1_output", "transcript"):
            if k in item:
                row[k] = item[k]
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument(
        "--input-format",
        choices=["auto", "json", "csv"],
        default="auto",
        help="auto: infer from file suffix; csv: build grouped prompt2 rows from region-level CSV.",
    )
    ap.add_argument("--diff-csv", default="", help="Optional CSV with overlap_pixels for ordering top3 region IDs.")
    ap.add_argument(
        "--gt-csv",
        default="",
        help="Optional GT CSV used as source of sample_id/region_id triplets. "
        "When set, top3 IDs come from this file (then optionally overlap-ordered by --diff-csv).",
    )
    ap.add_argument("--spec-root", default=DEFAULT_SPEC_ROOT, help="Spectrogram grid image root.")
    ap.add_argument("--mfa-root", default=DEFAULT_MFA_ROOT, help="MFA JSON root.")
    ap.add_argument(
        "--sample-id-contains",
        default="_LA_T_",
        help="When input is csv, keep only sample_ids containing this substring; empty disables filter.",
    )
    args = ap.parse_args()

    dst = Path(args.output_json).expanduser().resolve()
    src = Path(args.input_json).expanduser().resolve()
    in_fmt = args.input_format
    if in_fmt == "auto":
        in_fmt = "csv" if src.suffix.lower() == ".csv" else "json"

    if in_fmt == "csv":
        out = _build_from_csv(args)
    else:
        out = _build_from_json(args)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {dst} (n={len(out)}, input_format={in_fmt})")


if __name__ == "__main__":
    main()
