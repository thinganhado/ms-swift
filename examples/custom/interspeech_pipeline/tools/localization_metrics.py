import math
from typing import List, Sequence, Set


def recall_at_k(pred: Sequence[int], gt: Sequence[int], k: int) -> float:
    topk = pred[:k]
    gset: Set[int] = set(gt)
    return len([r for r in topk if r in gset]) / k


def dcg_at_k(pred: Sequence[int], gt: Sequence[int], k: int) -> float:
    gset: Set[int] = set(gt)
    dcg = 0.0
    for j in range(1, k + 1):
        rel = 1 if pred[j - 1] in gset else 0
        dcg += (2 ** rel - 1) / math.log2(j + 1)
    return dcg


def ndcg_at_k(pred: Sequence[int], gt: Sequence[int], k: int) -> float:
    dcg = dcg_at_k(pred, gt, k)
    m = min(k, len(set(gt)))
    if m == 0:
        return 0.0
    idcg = 0.0
    for j in range(1, m + 1):
        idcg += 1.0 / math.log2(j + 1)
    return dcg / idcg


def average_precision(pred: Sequence[int], gt: Sequence[int]) -> float:
    gset: Set[int] = set(gt)
    k = len(gset)
    if k == 0:
        return 0.0

    hits = 0
    score = 0.0
    for j, rj in enumerate(pred, start=1):
        if rj in gset:
            hits += 1
            score += hits / j
    return score / k


def mean_average_precision(preds: List[Sequence[int]], gts: List[Sequence[int]]) -> float:
    return sum(average_precision(p, g) for p, g in zip(preds, gts)) / len(preds)
