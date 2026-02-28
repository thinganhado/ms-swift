#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_Q1_SYSTEM_PROMPT = (
    "As an expert in deepfake speech spectrogram forensics, you can detect regions containing "
    "deepfake artifacts by analysing spectrogram segments. Return only the JSON array of your "
    "three chosen region IDs."
)
DEFAULT_Q2_SYSTEM_PROMPT = """You are an expert in deepfake speech spectrogram forensics.

You are given a spectrogram and transcript. You have already selected exactly 3 region IDs, in order: ID1, ID2, ID3.
For each ID, infer timing information (T), frequency band (F), phonetic category (P), and a visual description of the artifact and the likely audio impact implied by the artificial signs (En).

OUTPUT FORMAT (must follow exactly):
(T1=..., F1=..., P1=..., En1=\"...\"); (T2=..., F2=..., P2=..., En2=\"...\"); (T3=..., F3=..., P3=..., En3=\"...\")

Field definitions:
- Fields ending in 1, 2, and 3 correspond to ID1, ID2, and ID3 respectively.
- T: one of {speech, non-speech}
- F: one of {low, mid, high}
- P: one of {consonant, vowel, unvoiced}
- En: textual description, must be enclosed in double quotes.

Do not output any other text outside the three tuples."""


def _map_role(role: str) -> str:
    role = str(role or "").lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant"}:
        return "assistant"
    return role or "user"


def _extract_image_path(item: Dict[str, Any], user_content: Any) -> Optional[str]:
    for key in ("image", "img_path", "p1"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    if isinstance(user_content, list):
        for part in user_content:
            if isinstance(part, dict) and part.get("type") == "image":
                v = part.get("image")
                if isinstance(v, str) and v.strip():
                    return v.strip()
    return None


def _extract_text_only(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
        return "\n".join([t for t in texts if t]).strip()
    return str(content or "")


def _extract_message_image(msg: Dict[str, Any]) -> Optional[str]:
    c = msg.get("content", msg.get("value", ""))
    if isinstance(c, list):
        for part in c:
            if isinstance(part, dict) and part.get("type") == "image":
                img = part.get("image")
                if isinstance(img, str) and img.strip():
                    return img.strip()
    return None


def _norm_message_content(msg: Dict[str, Any], image_path: Optional[str]) -> Dict[str, Any]:
    role = _map_role(msg.get("role", msg.get("from", "")))
    content = msg.get("content", msg.get("value", ""))

    if isinstance(content, list):
        return {"role": role, "content": content}

    text = str(content or "")
    if role == "user" and image_path:
        return {
            "role": role,
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text},
            ],
        }
    return {
        "role": role,
        "content": [{"type": "text", "text": text}],
    }


def _assistant_text_message(text: str) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": str(text or "").strip()}],
    }


def _system_text_message(text: str) -> Dict[str, Any]:
    return {
        "role": "system",
        "content": [{"type": "text", "text": str(text or "").strip()}],
    }


def _prepend_text_to_message(msg: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    prefix = str(prefix or "").strip()
    if not prefix:
        return msg

    role = _map_role(msg.get("role", msg.get("from", "")))
    content = msg.get("content", msg.get("value", ""))

    if isinstance(content, list):
        new_content = []
        inserted = False
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and not inserted:
                text = str(part.get("text", "")).strip()
                joined = f"{prefix}\n\n{text}".strip() if text else prefix
                new_part = dict(part)
                new_part["text"] = joined
                new_content.append(new_part)
                inserted = True
            else:
                new_content.append(part)
        if not inserted:
            new_content.insert(0, {"type": "text", "text": prefix})
        return {"role": role, "content": new_content}

    text = str(content or "").strip()
    joined = f"{prefix}\n\n{text}".strip() if text else prefix
    return {"role": role, "content": [{"type": "text", "text": joined}]}


def _load_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON in {path}")
    return [x for x in data if isinstance(x, dict)]


def _make_join_keys(sample_id: str, image_path: Optional[str]) -> List[Tuple[str, str]]:
    keys: List[Tuple[str, str]] = []
    sid = str(sample_id or "").strip()
    img = str(image_path or "").strip()
    if img:
        keys.append(("image", img))
    if sid:
        keys.append(("sample_id", sid))
    return keys


def _extract_q1_target(item: Dict[str, Any], conv: List[Dict[str, Any]]) -> str:
    # Prompt1 GRPO builder stores the target in metadata, not as an assistant turn.
    for key in ("gt_regions", "assistant", "prompt1_target"):
        value = str(item.get(key, "")).strip()
        if value:
            return value
    if len(conv) >= 2:
        return _extract_text_only(conv[1].get("content", conv[1].get("value", ""))).strip()
    return ""


def _extract_ints(text: str) -> List[int]:
    return [int(x) for x in re.findall(r"\d+", str(text or ""))]


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
        if ch == "\\" and in_quote:
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


def _strip_outer_parens(text: str) -> str:
    s = str(text or "").strip()
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        return s[1:-1].strip()
    return s


def _parse_tuple_fields_any(raw: str, idx: int) -> Optional[Tuple[str, str, str, str]]:
    body = _strip_outer_parens(raw)
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
        return (
            m_t.group(1).strip(),
            m_f.group(1).strip(),
            m_p.group(1).strip(),
            m_en.group(1).strip(),
        )

    parts = [p.strip() for p in body.split(",", 4)]
    if len(parts) == 5:
        _, t, fband, p, en = parts
        return t, fband, p, en
    if len(parts) == 4:
        t, fband, p, en = parts
        return t, fband, p, en
    return None


def _normalize_gt_prompt2_indexed(text: str) -> str:
    tuple_matches = _split_top_level_tuples(text)
    if not tuple_matches:
        return str(text or "").strip()
    rebuilt: List[str] = []
    for idx, raw in enumerate(tuple_matches[:3], start=1):
        parsed = _parse_tuple_fields_any(raw, idx)
        if parsed is None:
            return str(text or "").strip()
        t, fband, p, en = parsed
        rebuilt.append(f'(T{idx}={t}, F{idx}={fband}, P{idx}={p}, En{idx}={en})')
    return "; ".join(rebuilt)


def _reorder_q2_by_q1(assistant1_text: str, q2_item: Dict[str, Any], user2: Dict[str, Any], gt_prompt2: str) -> Tuple[str, str]:
    q1_ids = _extract_ints(assistant1_text)
    if len(q1_ids) < 3:
        return _extract_text_only(user2.get("content", user2.get("value", ""))).strip(), gt_prompt2
    q1_ids = q1_ids[:3]

    tuple_matches = _split_top_level_tuples(gt_prompt2)
    if len(tuple_matches) < 3:
        return _extract_text_only(user2.get("content", user2.get("value", ""))).strip(), gt_prompt2

    q2_ids = _extract_ints(str(q2_item.get("prompt1_output", "")))
    if len(q2_ids) < 3:
        return _extract_text_only(user2.get("content", user2.get("value", ""))).strip(), gt_prompt2
    q2_ids = q2_ids[:3]

    tuple_by_source_id: Dict[int, Tuple[str, str, str, str]] = {}
    for src_idx, (src_id, raw) in enumerate(zip(q2_ids, tuple_matches[:3]), start=1):
        parsed = _parse_tuple_fields_any(raw, src_idx)
        if parsed is None:
            continue
        tuple_by_source_id[int(src_id)] = parsed

    if not all(cn in tuple_by_source_id for cn in q1_ids):
        return _extract_text_only(user2.get("content", user2.get("value", ""))).strip(), gt_prompt2

    prompt1_output = f"[{', '.join(map(str, q1_ids))}]"
    transcript = str(q2_item.get("transcript", "")).strip()
    user2_text = (
        "Explain the spoof artifact for each of the three selected region IDs in "
        f"{prompt1_output} . This is the transcript for context: {transcript}"
    ).strip()
    rebuilt = []
    for dst_idx, cn in enumerate(q1_ids, start=1):
        t, fband, p, en = tuple_by_source_id[cn]
        rebuilt.append(f'(T{dst_idx}={t}, F{dst_idx}={fband}, P{dst_idx}={p}, En{dst_idx}={en})')
    gt_prompt2_reordered = "; ".join(rebuilt)
    return user2_text, gt_prompt2_reordered


def _build_from_two_sources(
    q1_data: List[Dict[str, Any]],
    q2_data: List[Dict[str, Any]],
    q1_system_prompt: str,
    q2_system_prompt: str,
) -> List[Dict[str, Any]]:
    q2_index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in q2_data:
        conv = item.get("messages") or item.get("conversations")
        if not isinstance(conv, list) or len(conv) < 1:
            continue
        gt_prompt2 = str(item.get("gt_prompt2", "")).strip()
        if not gt_prompt2:
            continue
        user2 = conv[0]
        img2 = _extract_message_image(user2) or str(item.get("image", item.get("img_path", item.get("p1", "")))).strip()
        sid2 = str(item.get("sample_id", "")).strip()
        for k in _make_join_keys(sid2, img2):
            if k not in q2_index:
                q2_index[k] = item

    out: List[Dict[str, Any]] = []
    for item in q1_data:
        conv = item.get("messages") or item.get("conversations")
        if not isinstance(conv, list) or len(conv) < 1:
            continue

        user1 = conv[0]
        assistant1_text = _extract_q1_target(item, conv)
        if not assistant1_text:
            continue

        sid1 = str(item.get("sample_id", "")).strip()
        img1 = _extract_message_image(user1) or _extract_image_path(item, user1.get("content", user1.get("value", "")))

        q2_item = None
        for k in _make_join_keys(sid1, img1):
            q2_item = q2_index.get(k)
            if q2_item is not None:
                break
        if q2_item is None:
            continue

        q2_conv = q2_item.get("messages") or q2_item.get("conversations")
        if not isinstance(q2_conv, list) or len(q2_conv) < 1:
            continue
        user2 = q2_conv[0]
        gt_prompt2 = _normalize_gt_prompt2_indexed(str(q2_item.get("gt_prompt2", "")).strip())
        if not gt_prompt2:
            continue

        q1_ids = _extract_ints(assistant1_text)[:3]
        user2_text, gt_prompt2 = _reorder_q2_by_q1(assistant1_text, q2_item, user2, gt_prompt2)

        # Keep both turns multimodal/text exactly as builders provide.
        msg_system1 = _system_text_message(q1_system_prompt)
        msg_user1 = _norm_message_content(user1, img1)
        msg_assistant1 = _assistant_text_message(assistant1_text)
        msg_system2 = _system_text_message(q2_system_prompt)
        msg_user2 = {
            "role": "user",
            "content": [
                {"type": "image", "image": _extract_message_image(user2)},
                {"type": "text", "text": user2_text},
            ],
        } if _extract_message_image(user2) else {"role": "user", "content": user2_text}
        msg_assistant2 = _assistant_text_message(gt_prompt2)

        rec: Dict[str, Any] = {
            "messages": [msg_system1, msg_user1, msg_assistant1, msg_system2, msg_user2, msg_assistant2],
            "sample_id": str(q2_item.get("sample_id", sid1 or "")),
        }
        for key in ("sample_id_raw", "id", "prompt1_output", "transcript", "gt_prompt2"):
            if key in q2_item:
                rec[key] = q2_item[key]
            elif key in item and key not in rec:
                rec[key] = item[key]
        if len(q1_ids) == 3:
            rec["prompt1_output"] = f"[{', '.join(map(str, q1_ids))}]"
        rec["gt_prompt2"] = gt_prompt2
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", default="", help="Legacy single-source mode.")
    ap.add_argument("--q1-json", default="", help="Query1 builder output JSON.")
    ap.add_argument("--q2-json", default="", help="Prompt2 builder output JSON (GT-based).")
    ap.add_argument("--q1-system-prompt", default=DEFAULT_Q1_SYSTEM_PROMPT)
    ap.add_argument("--q2-system-prompt", default=DEFAULT_Q2_SYSTEM_PROMPT)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    dst = Path(args.output_json).expanduser().resolve()
    out: List[Dict[str, Any]]

    if args.q1_json and args.q2_json:
        q1_path = Path(args.q1_json).expanduser().resolve()
        q2_path = Path(args.q2_json).expanduser().resolve()
        q1_data = _load_json(q1_path)
        q2_data = _load_json(q2_path)
        out = _build_from_two_sources(q1_data, q2_data, args.q1_system_prompt, args.q2_system_prompt)
        print(f"mode: two-source (q1={q1_path}, q2={q2_path})")
    else:
        if not args.input_json:
            raise ValueError("Provide either --input-json (legacy) or both --q1-json and --q2-json.")
        src = Path(args.input_json).expanduser().resolve()
        data = _load_json(src)
        out = []
        for item in data:
            conv = item.get("messages") or item.get("conversations")
            if not isinstance(conv, list) or len(conv) < 2:
                continue

            image_path = None
            for m in conv:
                if _map_role(m.get("role", m.get("from", ""))) == "user":
                    image_path = _extract_image_path(item, m.get("content", m.get("value")))
                    if image_path:
                        break

            messages = []
            pending_system_texts: List[str] = []
            user_idx = 0
            for m in conv:
                role = _map_role(m.get("role", m.get("from", "")))
                if role == "system":
                    sys_text = _extract_text_only(m.get("content", m.get("value", ""))).strip()
                    if sys_text:
                        pending_system_texts.append(sys_text)
                    continue

                img_for_msg = image_path if role == "user" and user_idx == 0 else None
                norm_msg = _norm_message_content(m, img_for_msg)
                if role == "user":
                    user_idx += 1
                    if pending_system_texts:
                        norm_msg = _prepend_text_to_message(norm_msg, "\n\n".join(pending_system_texts))
                        pending_system_texts = []
                elif pending_system_texts:
                    # Fallback: attach stray system text to the next non-system message.
                    norm_msg = _prepend_text_to_message(norm_msg, "\n\n".join(pending_system_texts))
                    pending_system_texts = []
                messages.append(norm_msg)

            if len(messages) < 2:
                continue

            rec: Dict[str, Any] = {"messages": messages}
            for key in ("sample_id", "sample_id_raw", "id"):
                if key in item:
                    rec[key] = item[key]
            out.append(rec)
        print(f"mode: legacy single-source (input={src})")

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {dst} (n={len(out)})")


if __name__ == "__main__":
    main()
