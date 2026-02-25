import math
import os
import re
from collections import Counter, defaultdict
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
      R = 1.0 * nDCG@3 + 0.1 * format + 0.5 * novelty
      novelty = (|pred ∩ GT| - |pred ∩ SFT_top3|) / |pred|
    """

    W_NDCG = 1.0
    W_FORMAT = 0.1
    W_NOVELTY = 0.0
    W_DIVERSITY = 0.0
    W_DUP = float(os.getenv("INTERSPEECH_W_DUP", "0.05"))
    W_ID_FREQ = float(os.getenv("INTERSPEECH_W_ID_FREQ", "0.03"))
    W_TRIPLET = float(os.getenv("INTERSPEECH_W_TRIPLET", "0.02"))
    EMA_DECAY = float(os.getenv("INTERSPEECH_EMA_DECAY", "0.95"))
    MAX_DIVERSITY_PENALTY = float(os.getenv("INTERSPEECH_MAX_DIVERSITY_PENALTY", "0.2"))

    def __init__(self):
        super().__init__()
        self._id_ema = defaultdict(float)
        self._triplet_ema = defaultdict(float)

    @staticmethod
    def _clip_top3_any(text: str) -> Optional[List[int]]:
        ids = _parse_ids(text)
        if len(ids) < 3:
            return None
        pred = ids[:3]
        if any(i < 1 or i > 16 for i in pred):
            return None
        return pred

    def __call__(self, completions, gt_regions=None, assistant=None, sft_top3=None, **kwargs) -> List[float]:
        gt_source = gt_regions if gt_regions is not None else assistant
        default_sft = os.getenv("INTERSPEECH_SFT_TOP3", "13,1,2")
        sft_source = sft_top3

        rewards: List[float] = []
        batch_preds_for_ema: List[List[int]] = []
        for i, completion in enumerate(completions):
            pred = _parse_pred_top3(completion)
            fmt = 1.0 if pred is not None else 0.0
            pred_any = self._clip_top3_any(completion)

            gt_text = gt_source[i] if gt_source is not None and i < len(gt_source) else ""
            gt_ids = _parse_ids(gt_text)
            ndcg = _ndcg_at_3(pred, gt_ids) if pred is not None and gt_ids else 0.0

            sft_text = (
                sft_source[i] if sft_source is not None and i < len(sft_source) else default_sft
            )
            pred_set = set(pred) if pred is not None else set()
            gt_set = set(gt_ids)
            sft_set = set(_parse_ids(sft_text))
            novelty = 0.0
            if pred_set:
                novelty = (len(pred_set & gt_set) - len(pred_set & sft_set)) / float(len(pred_set))

            # Anti-collapse regularization:
            # - duplicate IDs in top-3
            # - globally overused IDs (EMA frequency)
            # - globally repeated triplets (EMA frequency)
            dup_penalty = 0.0
            id_freq_penalty = 0.0
            triplet_penalty = 0.0
            if pred_any is not None:
                pred_unique = list(dict.fromkeys(pred_any))
                dup_cnt = len(pred_any) - len(set(pred_any))
                dup_penalty = -dup_cnt / 2.0

                if pred_unique:
                    id_freq_penalty = -sum(self._id_ema[rid] for rid in pred_unique) / len(pred_unique)
                triplet_penalty = -self._triplet_ema[tuple(pred_any)]
                batch_preds_for_ema.append(pred_any)

            anti_collapse = (
                self.W_DUP * dup_penalty
                + self.W_ID_FREQ * id_freq_penalty
                + self.W_TRIPLET * triplet_penalty
            )
            anti_collapse = max(-self.MAX_DIVERSITY_PENALTY, min(0.0, anti_collapse))

            base_reward = self.W_NDCG * ndcg + self.W_FORMAT * fmt + self.W_NOVELTY * novelty
            rewards.append(base_reward + self.W_DIVERSITY * 0.0 + anti_collapse)

        # Update EMA stats after scoring the batch.
        if batch_preds_for_ema:
            id_counts = Counter()
            triplet_counts = Counter()
            for pred_any in batch_preds_for_ema:
                for rid in set(pred_any):
                    id_counts[rid] += 1
                triplet_counts[tuple(pred_any)] += 1

            bsz = float(len(batch_preds_for_ema))
            one_minus_decay = 1.0 - self.EMA_DECAY

            for rid, cnt in id_counts.items():
                freq = cnt / bsz
                self._id_ema[rid] = self.EMA_DECAY * self._id_ema[rid] + one_minus_decay * freq

            for trip, cnt in triplet_counts.items():
                freq = cnt / bsz
                self._triplet_ema[trip] = self.EMA_DECAY * self._triplet_ema[trip] + one_minus_decay * freq

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
