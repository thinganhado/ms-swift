import math
import os
import re
from typing import Dict, List, Optional, Sequence

from swift.rewards import ORM, orms


def _parse_ids(text: str) -> List[int]:
    return [int(x) for x in re.findall(r"\d+", str(text or ""))]


def _parse_pred_top3(text: str) -> Optional[List[int]]:
    ids = _parse_ids(text)
    if len(ids) != 3:
        return None
    if len(set(ids)) != 3:
        return None
    if any(i < 1 or i > 16 for i in ids):
        return None
    return ids


def _extract_assistant_text_from_messages(messages) -> str:
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text = content.strip()
            if text:
                return text
        if isinstance(content, list):
            chunks = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    chunks.append(str(part.get("text", "")))
            text = "".join(chunks).strip()
            if text:
                return text
    return ""


def _ndcg_at_3(pred: Sequence[int], gt: Sequence[int]) -> float:
    rel_by_id: Dict[int, int] = {}
    for rank, rid in enumerate(gt[:3]):
        if rid not in rel_by_id:
            rel_by_id[rid] = 3 - rank

    dcg = 0.0
    for j in range(1, 4):
        rel = rel_by_id.get(pred[j - 1], 0)
        dcg += (2**rel - 1) / math.log2(j + 1)

    ideal_rels = sorted(rel_by_id.values(), reverse=True)
    m = min(3, len(ideal_rels))
    if m == 0:
        return 0.0

    idcg = 0.0
    for j in range(1, m + 1):
        rel = ideal_rels[j - 1]
        idcg += (2**rel - 1) / math.log2(j + 1)
    return dcg / idcg


class InterspeechPrompt1Reward(ORM):
    """
    Prompt-1 reward:
      R = 1.0 * nDCG@3 + 0.1 * format
    """

    W_NDCG = 1.0
    W_FORMAT = 0.1

    def __init__(self):
        super().__init__()
        self._last_logged_step = -1
        self._log_every_steps = max(1, int(os.getenv("INTERSPEECH_LOG_EVERY_STEPS", "1")))
        self._group_size_for_metrics = max(1, int(os.getenv("INTERSPEECH_GROUP_SIZE", "8")))
        self._rank = int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0")))
        self._debug_reward = os.getenv("INTERSPEECH_DEBUG_REWARD", "0").lower() in {"1", "true", "yes", "on"}
        self._debug_print_samples = max(1, int(os.getenv("INTERSPEECH_DEBUG_SAMPLES", "8")))
        self._debug_source_printed = False

    def __call__(self, completions, gt_regions=None, assistant=None, sft_top3=None, messages=None, **kwargs) -> List[float]:
        # Robust per-sample fallback chain:
        # gt_regions -> assistant -> solution -> sft_top3
        solution = kwargs.get("solution")

        def _pick_gt(i: int):
            candidates = [
                ("gt_regions", gt_regions),
                ("assistant", assistant),
                ("solution", solution),
                ("sft_top3", sft_top3),
            ]
            for name, src in candidates:
                if src is None:
                    continue
                if i >= len(src):
                    continue
                text = str(src[i] if src[i] is not None else "").strip()
                if text:
                    return text, name
            # SFT-style fallback: extract assistant GT directly from messages.
            if messages is not None and i < len(messages):
                msg_text = _extract_assistant_text_from_messages(messages[i])
                if msg_text:
                    return msg_text, "messages.assistant"
            return "", "none"

        rewards: List[float] = []
        valid_count = 0
        ndcg_sum_valid = 0.0
        overlap_sum_valid = 0.0
        valid_triplets: List[tuple] = []
        used_source_counts = {
            "gt_regions": 0,
            "assistant": 0,
            "solution": 0,
            "sft_top3": 0,
            "messages.assistant": 0,
            "none": 0,
        }
        for i, completion in enumerate(completions):
            pred = _parse_pred_top3(completion)
            fmt = 1.0 if pred is not None else 0.0

            gt_text, used_source = _pick_gt(i)
            used_source_counts[used_source] += 1
            gt_ids = _parse_ids(gt_text)
            ndcg = _ndcg_at_3(pred, gt_ids) if pred is not None and gt_ids else 0.0
            if pred is not None:
                valid_count += 1
                ndcg_sum_valid += ndcg
                gt_set_for_overlap = set(gt_ids[:3]) if gt_ids else set()
                overlap_sum_valid += len(set(pred) & gt_set_for_overlap) / 3.0
                valid_triplets.append(tuple(pred))

            rewards.append(self.W_NDCG * ndcg + self.W_FORMAT * fmt)

        # Step-level diagnostics (rank-0 only).
        trainer_state = kwargs.get("trainer_state", None)
        global_step = int(getattr(trainer_state, "global_step", -1))
        should_log_step = (
            global_step >= 0
            and global_step != self._last_logged_step
            and (global_step % self._log_every_steps == 0)
        )
        if self._rank == 0 and should_log_step:
            self._last_logged_step = global_step
            total_n = max(1, len(completions))
            valid_rate = 100.0 * valid_count / total_n
            mean_ndcg_valid = (ndcg_sum_valid / valid_count) if valid_count > 0 else 0.0
            mean_overlap_valid = (overlap_sum_valid / valid_count) if valid_count > 0 else 0.0
            unique_list_rate = (
                100.0 * len(set(valid_triplets)) / len(valid_triplets)
                if valid_triplets
                else 0.0
            )

            group_size = self._group_size_for_metrics
            group_vars = []
            if group_size > 1:
                for s in range(0, len(rewards), group_size):
                    rg = rewards[s:s + group_size]
                    if len(rg) > 1:
                        m = sum(rg) / len(rg)
                        group_vars.append(sum((x - m) ** 2 for x in rg) / len(rg))
            reward_var_within_group = sum(group_vars) / len(group_vars) if group_vars else 0.0

            print(
                f"[p1_metrics] step={global_step} "
                f"valid_format_rate={valid_rate:.2f}% "
                f"mean_ndcg3_valid={mean_ndcg_valid:.4f} "
                f"mean_set_overlap_valid={mean_overlap_valid:.4f} "
                f"reward_var_within_group={reward_var_within_group:.6f} "
                f"unique_list_rate={unique_list_rate:.2f}%",
                flush=True,
            )
            if self._debug_reward:
                if not self._debug_source_printed:
                    print(
                        f"[p1_debug] gt_source_counts={used_source_counts} "
                        f"len_completions={len(completions)} "
                        f"len_gt_regions={len(gt_regions) if gt_regions is not None else 0} "
                        f"len_assistant={len(assistant) if assistant is not None else 0} "
                        f"len_solution={len(solution) if solution is not None else 0} "
                        f"len_sft_top3={len(sft_top3) if sft_top3 is not None else 0}",
                        flush=True,
                    )
                    self._debug_source_printed = True
                n = min(self._debug_print_samples, len(completions))
                for i in range(n):
                    pred_i = _parse_pred_top3(completions[i])
                    gt_text_i, used_source_i = _pick_gt(i)
                    gt_ids_i = _parse_ids(gt_text_i)[:3]
                    print(
                        f"[p1_debug] i={i} src={used_source_i} pred={pred_i} gt={gt_ids_i} "
                        f"raw_pred={str(completions[i])[:120]!r} raw_gt={str(gt_text_i)[:120]!r}",
                        flush=True,
                    )

        return rewards


TIME_LABELS = {"speech", "non-speech"}
FREQ_LABELS = {"low", "mid", "high"}
PHON_LABELS = {"consonant", "vowel", "unvoiced"}


def _norm_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _norm_time(v: str) -> Optional[str]:
    x = _norm_space(v)
    if x in TIME_LABELS:
        return x
    if x in {"nonspeech", "non speech"}:
        return "non-speech"
    return None


def _norm_freq(v: str) -> Optional[str]:
    x = _norm_space(v)
    if x in FREQ_LABELS:
        return x
    if x == "middle":
        return "mid"
    return None


def _norm_phon(v: str) -> Optional[str]:
    x = _norm_space(v)
    if x in PHON_LABELS:
        return x
    if x == "voiceless":
        return "unvoiced"
    return None


_REGION_PATTERN = re.compile(
    r"\(\s*(\d+)\s*,\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*(.*?)\s*\)",
    flags=re.DOTALL,
)


def _parse_region_tuples(text: str) -> Optional[List[Dict[str, str]]]:
    matches = _REGION_PATTERN.findall(str(text or ""))
    if len(matches) != 3:
        return None
    rows = []
    seen = set()
    for cn_raw, t_raw, f_raw, p_raw, en_raw in matches:
        cn = str(int(cn_raw))
        if cn in seen:
            return None
        seen.add(cn)
        t = _norm_time(t_raw)
        f = _norm_freq(f_raw)
        p = _norm_phon(p_raw)
        en = str(en_raw).strip()
        if t is None or f is None or p is None or not en:
            return None
        rows.append({"Cn": cn, "T": t, "F": f, "P": p, "En": en})
    return rows


TIME_LEXICON = {
    "speech": [r"\bspeech\b", r"\bvoiced\b", r"\bspoken\b"],
    "non-speech": [r"\bsilence\b", r"\bpause\b", r"\bnon[- ]?speech\b", r"\bbackground\b", r"\bnoise[- ]?only\b", r"\bunvoiced\b"],
}
FREQ_LEXICON = {"low": [r"\blow\b"], "mid": [r"\bmid\b", r"\bmiddle\b"], "high": [r"\bhigh\b"]}
PHON_LEXICON = {
    "vowel": [r"\bvowel\b", r"\bformant\b"],
    "consonant": [r"\bconsonant\b", r"\bstop\b", r"\bfricative\b"],
    "unvoiced": [r"\bunvoiced\b", r"\bvoiceless\b", r"\baspiration\b", r"\bburst\b"],
}


def _extract_label(en_text: str, lexicon: Dict[str, List[str]]) -> Optional[str]:
    text = str(en_text or "").lower()
    best = None
    best_count = 0
    best_pos = 10**9
    for label, pats in lexicon.items():
        cnt = 0
        pos = 10**9
        for pat in pats:
            ms = list(re.finditer(pat, text, flags=re.IGNORECASE))
            if not ms:
                continue
            cnt += len(ms)
            pos = min(pos, ms[0].start())
        if cnt > best_count or (cnt == best_count and cnt > 0 and pos < best_pos):
            best = label
            best_count = cnt
            best_pos = pos
    return best


def _independent_extract_from_en(en_text: str) -> Dict[str, Optional[str]]:
    return {
        "T": _extract_label(en_text, TIME_LEXICON),
        "F": _extract_label(en_text, FREQ_LEXICON),
        "P": _extract_label(en_text, PHON_LEXICON),
    }


class InterspeechPrompt2Reward(ORM):
    """
    Prompt-2 reward:
      avg_over_3_regions(0.75*field_acc + 0.25*consistency + 0.1*format)
    """

    W_ACC = 0.75
    W_CONS = 0.25
    W_FMT = 0.1

    def __call__(self, completions, gt_prompt2=None, assistant=None, prompt2_target=None, **kwargs) -> List[float]:
        gt_source = gt_prompt2
        if gt_source is None:
            gt_source = prompt2_target
        if gt_source is None:
            gt_source = assistant

        rewards: List[float] = []
        for i, completion in enumerate(completions):
            pred_rows = _parse_region_tuples(completion)
            gt_text = gt_source[i] if gt_source is not None and i < len(gt_source) else ""
            gt_rows = _parse_region_tuples(gt_text)

            if pred_rows is None or gt_rows is None:
                rewards.append(0.0)
                continue

            gt_by_cn = {r["Cn"]: r for r in gt_rows}
            fmt = 1.0
            total = 0.0
            for pred in pred_rows:
                gt = gt_by_cn.get(pred["Cn"])
                if gt is None:
                    acc = 0.0
                else:
                    acc = (
                        (1.0 if pred["T"] == gt["T"] else 0.0)
                        + (1.0 if pred["F"] == gt["F"] else 0.0)
                        + (1.0 if pred["P"] == gt["P"] else 0.0)
                    ) / 3.0

                ext = _independent_extract_from_en(pred["En"])
                cons = (
                    (1.0 if ext["T"] is not None and ext["T"] == pred["T"] else 0.0)
                    + (1.0 if ext["F"] is not None and ext["F"] == pred["F"] else 0.0)
                    + (1.0 if ext["P"] is not None and ext["P"] == pred["P"] else 0.0)
                ) / 3.0

                total += self.W_ACC * acc + self.W_CONS * cons + self.W_FMT * fmt

            rewards.append(total / 3.0)

        return rewards


orms["external_interspeech_p1"] = InterspeechPrompt1Reward
orms["external_interspeech_p2"] = InterspeechPrompt2Reward
