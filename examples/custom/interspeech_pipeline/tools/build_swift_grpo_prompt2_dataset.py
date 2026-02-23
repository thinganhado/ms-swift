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


def _build_user_message(user_msg: Dict[str, Any], image: Optional[str]) -> Dict[str, Any]:
    role = _map_role(user_msg.get("role", user_msg.get("from", "")))
    text = _extract_text(user_msg)
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
            return str(item[k]).strip()
    if len(conv) >= 2:
        return _extract_text(conv[1])
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    src = Path(args.input_json).expanduser().resolve()
    dst = Path(args.output_json).expanduser().resolve()
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
            "messages": [_build_user_message(user_msg, image)],
            "gt_prompt2": gt,
        }
        for k in ("sample_id", "sample_id_raw", "id", "prompt1_output", "transcript"):
            if k in item:
                row[k] = item[k]
        out.append(row)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {dst} (n={len(out)})")


if __name__ == "__main__":
    main()

