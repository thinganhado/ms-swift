#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    src = Path(args.input_json).expanduser().resolve()
    dst = Path(args.output_json).expanduser().resolve()

    data = json.loads(src.read_text(encoding="utf-8-sig"))
    out: List[Dict[str, Any]] = []

    for idx, item in enumerate(data):
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

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {dst} (n={len(out)})")


if __name__ == "__main__":
    main()

