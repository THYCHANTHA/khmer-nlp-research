"""Ranking metrics (binary relevance) and bootstrap helpers. ``ranking`` is a list of doc ids, best first."""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence

import numpy as np


def precision_at_k(ranking: Sequence, relevant: Collection, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(1 for d in ranking[:k] if d in relevant) / k


def recall_at_k(ranking: Sequence, relevant: Collection, k: int) -> float:
    if not relevant:
        return float("nan")
    return sum(1 for d in ranking[:k] if d in relevant) / len(relevant)


def reciprocal_rank(ranking: Sequence, relevant: Collection) -> float:
    for i, d in enumerate(ranking, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranking: Sequence, relevant: Collection, k: int) -> float:
    if not relevant:
        return float("nan")
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranking[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def mean_relevant_rank(ranking: Sequence, relevant: Collection) -> float:
    """Mean 1-based position of the relevant documents in the full ranking (lower is better)."""
    pos = [i for i, d in enumerate(ranking, start=1) if d in relevant]
    return float(np.mean(pos)) if pos else float("nan")


def evaluate_ranking(ranking: Sequence, relevant: Collection) -> dict[str, float]:
    return {
        "P@5": precision_at_k(ranking, relevant, 5),
        "P@10": precision_at_k(ranking, relevant, 10),
        "R@5": recall_at_k(ranking, relevant, 5),
        "R@10": recall_at_k(ranking, relevant, 10),
        "MRR": reciprocal_rank(ranking, relevant),
        "nDCG@10": ndcg_at_k(ranking, relevant, 10),
        "MeanRelRank": mean_relevant_rank(ranking, relevant),
    }


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 42, alpha: float = 0.05):
    """Percentile bootstrap CI of the mean over queries (NaNs dropped)."""
    v = np.asarray([x for x in values if not math.isnan(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 42):
    """Mean of (b - a) over the same queries with a percentile CI; positive means b is higher."""
    x = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    keep = ~np.isnan(x)
    return bootstrap_ci(x[keep], n_boot=n_boot, seed=seed)
