#!/usr/bin/env python3
import argparse
import json
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
(Cn=ID1, T=..., F=..., P=..., En=\"...\"); (Cn=ID2, T=..., F=..., P=..., En=\"...\"); (Cn=ID3, T=..., F=..., P=..., En=\"...\")

Field definitions:
- Cn: region_id
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
    return {"role": role, "content": text}


def _assistant_text_message(text: str) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": str(text or "").strip()}],
    }


def _system_text_message(text: str) -> Dict[str, Any]:
    return {"role": "system", "content": str(text or "").strip()}


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
        gt_prompt2 = str(q2_item.get("gt_prompt2", "")).strip()
        if not gt_prompt2:
            continue

        # Keep both turns multimodal/text exactly as builders provide.
        msg_system1 = _system_text_message(q1_system_prompt)
        msg_user1 = _norm_message_content(user1, img1)
        msg_assistant1 = _assistant_text_message(assistant1_text)
        msg_system2 = _system_text_message(q2_system_prompt)
        msg_user2 = _norm_message_content(user2, _extract_message_image(user2))
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

            first = conv[0]
            image_path = _extract_image_path(item, first.get("content", first.get("value")))
            messages = [_norm_message_content(m, image_path if i == 0 else None) for i, m in enumerate(conv)]

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
