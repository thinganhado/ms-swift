#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

HARDCODED_QUERY2_USER_TEMPLATE = (
    "Explain the spoof artifact for each of the three selected region IDs in "
    "{prompt1_output} . This is the transcript for context: {transcript}"
)
DEFAULT_SPEC_ROOT = "/datasets/work/dss-deepfake-audio/work/data/datasets/interspeech/img/specs/grid"
DEFAULT_MFA_ROOT = "/scratch3/che489/Ha/interspeech/datasets/vocv4_mfa_aligned"
DEFAULT_IMAGE_SUFFIX = "_grid_img_edge_number_axes.png"


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _to_one_line(text: str) -> str:
    return " ".join(str(text or "").split())


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


def _extract_image_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        c = msg.get("content", msg.get("value", ""))
        if not isinstance(c, list):
            continue
        for part in c:
            if isinstance(part, dict) and part.get("type") == "image":
                img = _norm(part.get("image"))
                if img:
                    return img
    return ""


def _extract_text_from_messages(messages: Any, prefer_role: str = "assistant") -> str:
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = _norm(msg.get("role", msg.get("from", ""))).lower()
        if prefer_role and role and role != prefer_role:
            continue
        c = msg.get("content", msg.get("value", ""))
        if isinstance(c, str):
            t = c.strip()
            if t:
                return t
        if isinstance(c, list):
            texts = []
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(str(part.get("text", "")))
            t = "".join(texts).strip()
            if t:
                return t
    return ""


def _extract_ids_3(text: str) -> Optional[List[int]]:
    # Strict parse: first three integers in [1, 16], unique.
    ids = [int(x) for x in re.findall(r"\d+", str(text or ""))]
    if len(ids) < 3:
        return None
    ids = ids[:3]
    if len(set(ids)) != 3:
        return None
    if any(i < 1 or i > 16 for i in ids):
        return None
    return ids


def _infer_sample_id(item: Dict[str, Any], image_suffix: str) -> str:
    sid = _norm(item.get("sample_id")) or _norm(item.get("sample_id_raw")) or _norm(item.get("id"))
    if sid:
        return sid
    img = _norm(item.get("image")) or _norm(item.get("img_path")) or _norm(item.get("p1"))
    if not img:
        img = _extract_image_from_messages(item.get("messages") or item.get("conversations"))
    if not img:
        return ""
    name = Path(img).name
    if image_suffix and name.endswith(image_suffix):
        return name[: -len(image_suffix)]
    return Path(name).stem


def _extract_prediction_text(item: Dict[str, Any]) -> str:
    # Preferred known keys from inference outputs.
    for k in ("response", "pred_top3", "prediction", "output", "assistant"):
        t = _norm(item.get(k))
        if t:
            return t
    msgs = item.get("messages") or item.get("conversations")
    t = _extract_text_from_messages(msgs, prefer_role="assistant")
    if t:
        return t
    # Last fallback: any text-like field
    for k in ("text", "content"):
        t = _norm(item.get(k))
        if t:
            return t
    return ""


def _extract_transcript(item: Dict[str, Any], mfa_root: Path, sample_id: str) -> str:
    t = _to_one_line(_norm(item.get("transcript")))
    if t:
        return t
    return _extract_transcript_word_tier_like_qwen_region(mfa_root, sample_id)


def _build_row(
    sample_id: str,
    ids: List[int],
    transcript: str,
    spec_root: Path,
    user_template: str,
    raw_pred: str,
) -> Dict[str, Any]:
    prompt1_output = f"[{', '.join(map(str, ids))}]" if ids else ""
    user_text = user_template.format(prompt1_output=prompt1_output, transcript=transcript)
    image_path = str((spec_root / f"{sample_id}{DEFAULT_IMAGE_SUFFIX}").as_posix())
    return {
        "sample_id": sample_id,
        "prompt1_output": prompt1_output,
        "transcript": transcript,
        "p1_raw_pred": raw_pred,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_text},
                ],
            }
        ],
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build GRPO2 prompt2 dataset from GRPO1 prompt1 prediction JSON. "
        "Output is user-only (no gt_prompt2)."
    )
    ap.add_argument("--input-json", default="", help="Prediction JSON path (list[dict]).")
    ap.add_argument(
        "--input-dir",
        default="",
        help="Optional prediction directory (e.g., merged_for_eval with per-sample 'json' files).",
    )
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--spec-root", default=DEFAULT_SPEC_ROOT)
    ap.add_argument("--mfa-root", default=DEFAULT_MFA_ROOT)
    ap.add_argument("--image-suffix", default=DEFAULT_IMAGE_SUFFIX)
    ap.add_argument("--sample-id-contains", default="_LA_T_")
    ap.add_argument("--strict-ids", action="store_true", help="Drop records if top3 parsing fails strict checks.")
    ap.add_argument("--keep-invalid", action="store_true", help="Keep rows with unparsable top3 as empty prompt1_output.")
    ap.add_argument(
        "--user-template",
        default=HARDCODED_QUERY2_USER_TEMPLATE,
        help="Prompt2 user template with {prompt1_output} and {transcript}.",
    )
    return ap.parse_args()


def _load_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.input_dir:
        root = Path(args.input_dir).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"--input-dir not found: {root}")
        records: List[Dict[str, Any]] = []
        for fp in sorted(root.glob("*/json")):
            try:
                obj = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(obj, dict):
                records.append(obj)
        print(f"loaded from dir: {root} (n={len(records)})")
        return records

    if not args.input_json:
        raise ValueError("Provide either --input-json or --input-dir.")
    src = Path(args.input_json).expanduser().resolve()
    data = json.loads(src.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {src}")
    print(f"loaded from json: {src} (n={len(data)})")
    return [x for x in data if isinstance(x, dict)]


def main() -> None:
    args = parse_args()
    dst = Path(args.output_json).expanduser().resolve()
    spec_root = Path(args.spec_root).expanduser().resolve()
    mfa_root = Path(args.mfa_root).expanduser().resolve()

    data = _load_records(args)

    out: List[Dict[str, Any]] = []
    kept = 0
    dropped_no_sid = 0
    dropped_filter = 0
    dropped_no_ids = 0

    for item in data:
        if not isinstance(item, dict):
            continue
        sample_id = _infer_sample_id(item, args.image_suffix)
        if not sample_id:
            dropped_no_sid += 1
            continue
        if args.sample_id_contains and args.sample_id_contains not in sample_id:
            dropped_filter += 1
            continue

        pred_text = _extract_prediction_text(item)
        ids = _extract_ids_3(pred_text)
        if ids is None and not args.keep_invalid:
            dropped_no_ids += 1
            continue
        if ids is None:
            ids = []
        if args.strict_ids and len(ids) != 3:
            dropped_no_ids += 1
            continue

        transcript = _extract_transcript(item, mfa_root, sample_id)
        rec = _build_row(
            sample_id=sample_id,
            ids=ids,
            transcript=_to_one_line(transcript),
            spec_root=spec_root,
            user_template=args.user_template,
            raw_pred=pred_text,
        )
        out.append(rec)
        kept += 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {dst}")
    print(f"total_in: {len(data)}")
    print(f"kept: {kept}")
    print(f"dropped_no_sample_id: {dropped_no_sid}")
    print(f"dropped_sample_filter: {dropped_filter}")
    print(f"dropped_no_valid_top3: {dropped_no_ids}")


if __name__ == "__main__":
    main()
