import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

from swift.rewards import ORM, orms


def _parse_ids(text: str) -> List[int]:
    return [int(x) for x in re.findall(r"\d+", str(text or ""))]


def _parse_pred_top3(text: str) -> Optional[List[int]]:
    s = str(text or "").strip()
    # Strict output format only: allow optional [] wrapper, otherwise digits+commas+spaces only.
    if not re.fullmatch(r"\[?\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]?", s):
        return None
    ids = [int(x) for x in re.findall(r"\d+", s)]
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


def _grid_dist_similarity(pred: Sequence[int], gt: Sequence[int]) -> float:
    if not pred or not gt:
        return 0.0

    def _to_rc(idx: int):
        x = int(idx) - 1
        return (x // 4, x % 4)

    gt_rc = [_to_rc(i) for i in gt if 1 <= int(i) <= 16]
    if not gt_rc:
        return 0.0

    sims = []
    for p in pred:
        if not (1 <= int(p) <= 16):
            continue
        pr, pc = _to_rc(int(p))
        min_dist = min(abs(pr - gr) + abs(pc - gc) for gr, gc in gt_rc)
        sims.append(1.0 / (1.0 + float(min_dist)))
    if not sims:
        return 0.0
    return sum(sims) / len(sims)


class InterspeechPrompt1Reward(ORM):
    """
    Prompt-1 reward:
      R = 1.0 * nDCG@3 + 0.5 * hit + 0.1 * format
    """

    W_NDCG = 0.3
    W_HIT = 1.0
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

        def _canonicalize_gt_text(text: str) -> str:
            ids = _parse_ids(text)[:3]
            if ids:
                return f"[{', '.join(map(str, ids))}]"
            return str(text or "").strip()

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
                    return _canonicalize_gt_text(text), name
            # SFT-style fallback: extract assistant GT directly from messages.
            if messages is not None and i < len(messages):
                msg_text = _extract_assistant_text_from_messages(messages[i])
                if msg_text:
                    return _canonicalize_gt_text(msg_text), "messages.assistant"
            return "", "none"

        rewards: List[float] = []
        valid_count = 0
        ndcg_sum_valid = 0.0
        hit_sum_valid = 0.0
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
            hit = (len(set(pred) & set(gt_ids[:3])) / 3.0) if (pred is not None and gt_ids) else 0.0
            if pred is not None:
                valid_count += 1
                ndcg_sum_valid += ndcg
                hit_sum_valid += hit
                gt_set_for_overlap = set(gt_ids[:3]) if gt_ids else set()
                overlap_sum_valid += len(set(pred) & gt_set_for_overlap) / 3.0
                valid_triplets.append(tuple(pred))

            rewards.append(self.W_NDCG * ndcg + self.W_HIT * hit + self.W_FORMAT * fmt)

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
            mean_hit_valid = (hit_sum_valid / valid_count) if valid_count > 0 else 0.0
            mean_overlap_valid = (overlap_sum_valid / valid_count) if valid_count > 0 else 0.0
            unique_list_rate = (
                100.0 * len(set(valid_triplets)) / len(valid_triplets)
                if valid_triplets
                else 0.0
            )

            group_size = self._group_size_for_metrics
            group_vars = []
            unique_ids_per_group = []
            max_id_freq_per_group = []
            if group_size > 1:
                for s in range(0, len(rewards), group_size):
                    rg = rewards[s:s + group_size]
                    if len(rg) > 1:
                        m = sum(rg) / len(rg)
                        group_vars.append(sum((x - m) ** 2 for x in rg) / len(rg))
                    # Group diversity diagnostics on valid parsed triplets only.
                    ids = []
                    end = min(s + group_size, len(completions))
                    for j in range(s, end):
                        trip = _parse_pred_top3(completions[j])
                        if trip is not None:
                            ids.extend(trip)
                    if ids:
                        c = Counter(ids)
                        unique_ids_per_group.append(float(len(c)))
                        max_id_freq_per_group.append(float(max(c.values())))
            reward_var_within_group = sum(group_vars) / len(group_vars) if group_vars else 0.0
            mean_unique_ids_in_group = (
                sum(unique_ids_per_group) / len(unique_ids_per_group) if unique_ids_per_group else 0.0)
            mean_max_id_freq_in_group = (
                sum(max_id_freq_per_group) / len(max_id_freq_per_group) if max_id_freq_per_group else 0.0)

            print(
                f"[p1_metrics] step={global_step} "
                f"valid_format_rate={valid_rate:.2f}% "
                f"mean_hit_valid={mean_hit_valid:.4f} "
                f"mean_ndcg3_valid={mean_ndcg_valid:.4f} "
                f"mean_set_overlap_valid={mean_overlap_valid:.4f} "
                f"reward_var_within_group={reward_var_within_group:.6f} "
                f"unique_ids_in_group={mean_unique_ids_in_group:.2f} "
                f"max_id_freq_in_group={mean_max_id_freq_in_group:.2f} "
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


_T_PATTERN = re.compile(r"\bt\s*=\s*([^,]+?)\s*(?:,|$)", flags=re.IGNORECASE)
_F_PATTERN = re.compile(r"\bf\s*=\s*([^,]+?)\s*(?:,|$)", flags=re.IGNORECASE)
_P_PATTERN = re.compile(r"\bp\s*=\s*([^,]+?)\s*(?:,|$)", flags=re.IGNORECASE)
_EN_PATTERN = re.compile(r"\ben\s*=\s*(.+)\s*$", flags=re.IGNORECASE | re.DOTALL)


def _strip_wrapped_quotes(s: str) -> str:
    x = str(s or "").strip()
    if len(x) >= 2 and x[0] == '"' and x[-1] == '"':
        return x[1:-1]
    return x


def _split_top_level_tuples(text: str) -> List[str]:
    s = str(text or "")
    out: List[str] = []
    start = None
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


def _parse_region_tuples(text: str, expected_ids: Optional[Sequence[int]] = None) -> Optional[List[Dict[str, object]]]:
    raw_tuples = _split_top_level_tuples(text)
    if len(raw_tuples) != 3:
        return None

    rows: List[Dict[str, object]] = []
    for idx, raw in enumerate(raw_tuples):
        slot = idx + 1
        body = _strip_wrapped_quotes(str(raw or "").strip())
        if body.startswith("(") and body.endswith(")"):
            body = body[1:-1].strip()
        t_raw = ""
        f_raw = ""
        p_raw = ""
        en_raw = ""

        # Position-aligned format:
        # (T1=speech, F1=mid, P1=vowel, En1="...")
        # fallback tolerated: (T=speech, F=mid, P=vowel, En="...")
        m_t = re.search(rf"\bT{slot}\s*=\s*([^,]+?)\s*(?:,|$)", body, flags=re.IGNORECASE) or _T_PATTERN.search(body)
        m_f = re.search(rf"\bF{slot}\s*=\s*([^,]+?)\s*(?:,|$)", body, flags=re.IGNORECASE) or _F_PATTERN.search(body)
        m_p = re.search(rf"\bP{slot}\s*=\s*([^,]+?)\s*(?:,|$)", body, flags=re.IGNORECASE) or _P_PATTERN.search(body)
        m_en = re.search(rf"\bEn{slot}\s*=\s*(.+)\s*$", body, flags=re.IGNORECASE | re.DOTALL) or _EN_PATTERN.search(body)
        if m_t and m_f and m_p and m_en:
            t_raw = m_t.group(1).strip()
            f_raw = m_f.group(1).strip()
            p_raw = m_p.group(1).strip()
            en_raw = m_en.group(1).strip()
        else:
            # Positional fallback without Cn:
            # (speech, mid, vowel, ...)
            parts = [p.strip() for p in body.split(",", 3)]
            if len(parts) == 4:
                t_raw, f_raw, p_raw, en_raw = parts
            else:
                return None

        if expected_ids is not None and idx < len(expected_ids):
            try:
                cn_i = int(expected_ids[idx])
            except Exception:
                cn_i = -1
        else:
            cn_i = -1

        t = _norm_time(t_raw)
        f = _norm_freq(f_raw)
        p = _norm_phon(p_raw)
        en_full = str(en_raw or "").strip()
        en_text = _strip_wrapped_quotes(en_full)
        en_quoted = len(en_full) >= 2 and en_full[0] == '"' and en_full[-1] == '"'

        rows.append(
            {
                "Cn": str(cn_i) if 1 <= cn_i <= 16 else "",
                "Cn_i": cn_i,
                "T": t,
                "F": f,
                "P": p,
                "En": en_text,
                "en_quoted": en_quoted,
            }
        )
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
    Prompt-2 reward (overlap-masked):
      R = (1/3) * sum_i (W_FMT*fmt_i + m_i*R_sup_i + (1-m_i)*R_rob_i)

    where:
      R_sup_i = W_ACC*acc_i + W_CONS*cons_i
      R_rob_i = W_BOUND*bound_i + W_UNC*unc_i + W_CONS_ROB*cons_i
    """

    W_ACC = 0.6
    W_CONS = 0.3
    W_FMT = 0.1
    W_BOUND = 0.1
    W_UNC = 0.1
    W_CONS_ROB = 0.0

    def __init__(self):
        super().__init__()
        self._last_logged_step = -1
        self._log_every_steps = max(1, int(os.getenv("INTERSPEECH_LOG_EVERY_STEPS", "1")))
        self._group_size_for_metrics = max(1, int(os.getenv("INTERSPEECH_GROUP_SIZE", "8")))
        self._rank = int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0")))
        self._debug_reward = os.getenv("INTERSPEECH_DEBUG_REWARD", "0").lower() in {"1", "true", "yes", "on"}
        self._debug_print_samples = max(1, int(os.getenv("INTERSPEECH_DEBUG_SAMPLES", "8")))

    @staticmethod
    def _format_tuple_score(pred: Dict[str, object]) -> float:
        t = pred.get("T")
        f = pred.get("F")
        p = pred.get("P")
        en = str(pred.get("En", "")).strip()
        en_quoted = bool(pred.get("en_quoted", False))
        ok = (t is not None) and (f is not None) and (p is not None) and bool(en) and en_quoted
        return 1.0 if ok else 0.0

    @staticmethod
    def _bound_score(pred: Dict[str, object]) -> float:
        en = str(pred.get("En", "")).strip()
        if not en:
            return 0.0
        ids = [int(x) for x in re.findall(r"\d+", en) if 1 <= int(x) <= 16]
        # Position-aligned mode: En should not mention explicit region IDs.
        return 1.0 if not ids else 0.0

    @staticmethod
    def _unc_score(pred: Dict[str, object]) -> float:
        en = _norm_space(str(pred.get("En", "")))
        return 1.0 if en.startswith("uncertain") else 0.0

    @staticmethod
    def _consistency_score(pred: Dict[str, object]) -> float:
        t = pred.get("T")
        f = pred.get("F")
        p = pred.get("P")
        en = str(pred.get("En", "")).strip()
        if t is None or f is None or p is None or not en:
            return 0.0
        ext = _independent_extract_from_en(en)
        return (
            (1.0 if ext["T"] is not None and ext["T"] == t else 0.0)
            + (1.0 if ext["F"] is not None and ext["F"] == f else 0.0)
            + (1.0 if ext["P"] is not None and ext["P"] == p else 0.0)
        ) / 3.0

    @staticmethod
    def _extract_gt_ids_and_rows(
        i: int,
        gt_prompt2,
        prompt2_target,
        gt_regions,
        assistant,
        prompt1_output,
    ) -> (set, Dict[str, Dict[str, object]]):
        expected_ids: List[int] = []
        for src in (prompt1_output, gt_regions, assistant):
            if src is not None and i < len(src):
                t = str(src[i] if src[i] is not None else "").strip()
                if not t:
                    continue
                ids = [x for x in _parse_ids(t)[:3] if 1 <= int(x) <= 16]
                if ids:
                    expected_ids = ids
                    break

        # Prefer full GT tuple source.
        gt_tuple_text = ""
        for src in (gt_prompt2, prompt2_target):
            if src is not None and i < len(src):
                t = str(src[i] if src[i] is not None else "").strip()
                if t:
                    gt_tuple_text = t
                    break

        gt_rows_by_cn: Dict[str, Dict[str, object]] = {}
        gt_ids: set = set()
        if gt_tuple_text:
            gt_rows = _parse_region_tuples(gt_tuple_text, expected_ids=expected_ids)
            if gt_rows is not None:
                for r in gt_rows:
                    cn = str(r.get("Cn", "")).strip()
                    if not cn:
                        continue
                    gt_rows_by_cn[cn] = r
                    gt_ids.add(int(cn))
                return gt_ids, gt_rows_by_cn

        # Fallback for pred-phase: only GT IDs are needed to build overlap mask m_i.
        if expected_ids:
            return set(expected_ids), {}
        return set(), {}

    def __call__(
        self,
        completions,
        gt_prompt2=None,
        assistant=None,
        prompt2_target=None,
        gt_regions=None,
        prompt1_output=None,
        **kwargs,
    ) -> List[float]:

        rewards: List[float] = []
        valid_count = 0
        tuple_total = 0
        tuple_fmt_sum = 0.0
        tuple_m_sum = 0.0
        sup_tuple_count = 0
        rob_tuple_count = 0
        acc_sup_sum = 0.0
        cons_sup_sum = 0.0
        cons_rob_sum = 0.0
        bound_rob_sum = 0.0
        unc_rob_sum = 0.0
        gt_tuple_available = 0

        debug_rows: List[Dict[str, object]] = []
        for i, completion in enumerate(completions):
            expected_ids: List[int] = []
            for src in (prompt1_output, gt_regions, assistant):
                if src is not None and i < len(src):
                    t = str(src[i] if src[i] is not None else "").strip()
                    if not t:
                        continue
                    ids = [x for x in _parse_ids(t)[:3] if 1 <= int(x) <= 16]
                    if ids:
                        expected_ids = ids
                        break

            pred_rows = _parse_region_tuples(completion, expected_ids=expected_ids)
            if pred_rows is None:
                rewards.append(0.0)
                if len(debug_rows) < self._debug_print_samples:
                    debug_rows.append(
                        {
                            "i": i,
                            "parsed": False,
                            "expected_ids": expected_ids,
                            "tuple_fields_ok": [],
                            "reward": 0.0,
                            "raw_pred": str(completion)[:200],
                        }
                    )
                continue

            valid_count += 1
            gt_ids, gt_by_cn = self._extract_gt_ids_and_rows(
                i=i,
                gt_prompt2=gt_prompt2,
                prompt2_target=prompt2_target,
                gt_regions=gt_regions,
                assistant=assistant,
                prompt1_output=prompt1_output,
            )
            if gt_by_cn:
                gt_tuple_available += 1

            total = 0.0
            tuple_fields_ok: List[int] = []
            for pred in pred_rows:
                fmt_i = self._format_tuple_score(pred)
                cn = str(pred.get("Cn", "")).strip()
                cn_i = int(pred.get("Cn_i", -1))
                m_i = 1.0 if (cn_i in gt_ids) else 0.0
                tuple_fields_ok.append(int(fmt_i))
                tuple_total += 1
                tuple_fmt_sum += fmt_i
                tuple_m_sum += m_i

                cons = self._consistency_score(pred)

                if m_i > 0.0:
                    sup_tuple_count += 1
                    gt = gt_by_cn.get(cn)
                    if gt is None:
                        acc = 0.0
                    else:
                        acc = (
                            (1.0 if pred.get("T") == gt.get("T") else 0.0)
                            + (1.0 if pred.get("F") == gt.get("F") else 0.0)
                            + (1.0 if pred.get("P") == gt.get("P") else 0.0)
                        ) / 3.0
                    acc_sup_sum += acc
                    cons_sup_sum += cons
                    r_sup = self.W_ACC * acc + self.W_CONS * cons
                    total += self.W_FMT * fmt_i + r_sup
                else:
                    rob_tuple_count += 1
                    bound = self._bound_score(pred)
                    unc = self._unc_score(pred)
                    cons_rob_sum += cons
                    bound_rob_sum += bound
                    unc_rob_sum += unc
                    r_rob = self.W_BOUND * bound + self.W_UNC * unc + self.W_CONS_ROB * cons
                    total += self.W_FMT * fmt_i + r_rob

            sample_reward = total / 3.0
            rewards.append(sample_reward)
            if len(debug_rows) < self._debug_print_samples:
                debug_rows.append(
                    {
                        "i": i,
                        "parsed": True,
                        "expected_ids": expected_ids,
                        "tuple_fields_ok": tuple_fields_ok,
                        "position_match_mask": [int(int(pred.get("Cn_i", -1)) in gt_ids) for pred in pred_rows],
                        "reward": sample_reward,
                        "raw_pred": str(completion)[:200],
                    }
                )

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
            mean_tuple_fmt = (tuple_fmt_sum / tuple_total) if tuple_total > 0 else 0.0
            supervised_tuple_rate = (tuple_m_sum / tuple_total) if tuple_total > 0 else 0.0
            mean_acc_sup = (acc_sup_sum / sup_tuple_count) if sup_tuple_count > 0 else 0.0
            mean_cons_sup = (cons_sup_sum / sup_tuple_count) if sup_tuple_count > 0 else 0.0
            mean_cons_rob = (cons_rob_sum / rob_tuple_count) if rob_tuple_count > 0 else 0.0
            mean_bound_rob = (bound_rob_sum / rob_tuple_count) if rob_tuple_count > 0 else 0.0
            mean_unc_rob = (unc_rob_sum / rob_tuple_count) if rob_tuple_count > 0 else 0.0
            gt_tuple_avail_rate = 100.0 * gt_tuple_available / total_n

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
                f"[p2_metrics] step={global_step} "
                f"valid_format_rate={valid_rate:.2f}% "
                f"mean_tuple_fmt={mean_tuple_fmt:.4f} "
                f"supervised_tuple_rate={supervised_tuple_rate:.4f} "
                f"mean_acc_sup={mean_acc_sup:.4f} "
                f"mean_cons_sup={mean_cons_sup:.4f} "
                f"mean_bound_rob={mean_bound_rob:.4f} "
                f"mean_unc_rob={mean_unc_rob:.4f} "
                f"mean_cons_rob={mean_cons_rob:.4f} "
                f"gt_tuple_avail_rate={gt_tuple_avail_rate:.2f}% "
                f"reward_var_within_group={reward_var_within_group:.6f}",
                flush=True,
            )
            if self._debug_reward:
                print(
                    f"[p2_debug] lens: completions={len(completions)} "
                    f"gt_prompt2={len(gt_prompt2) if gt_prompt2 is not None else 0} "
                    f"prompt2_target={len(prompt2_target) if prompt2_target is not None else 0} "
                    f"gt_regions={len(gt_regions) if gt_regions is not None else 0} "
                    f"assistant={len(assistant) if assistant is not None else 0}",
                    flush=True,
                )
                for row in debug_rows:
                    print(
                        f"[p2_debug] i={row['i']} parsed={row['parsed']} expected_ids={row['expected_ids']} "
                        f"tuple_fields_ok={row['tuple_fields_ok']} pos_match={row.get('position_match_mask', [])} "
                        f"reward={float(row['reward']):.4f} "
                        f"raw_pred={row['raw_pred']!r}",
                        flush=True,
                    )

        return rewards


orms["external_interspeech_p1"] = InterspeechPrompt1Reward
orms["external_interspeech_p2"] = InterspeechPrompt2Reward
