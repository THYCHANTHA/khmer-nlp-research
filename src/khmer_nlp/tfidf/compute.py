"""TF-IDF written out explicitly (so every quantity can be inspected), verified against scikit-learn in the tests.

tf(t, d)  = raw count of t in d (optionally 1 + log tf)
idf(t)    = ln((1 + N) / (1 + df(t))) + 1      (smoothed, same as scikit-learn's default)
w(t, d)   = tf * idf, then each document vector is L2-normalised.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class TfidfModel:
    vocab: list[str]
    index: dict[str, int]
    idf: np.ndarray
    n_docs: int
    df: np.ndarray
    sublinear: bool = False

    def transform(self, docs: Sequence[Sequence[str]]) -> sparse.csr_matrix:
        rows, cols, vals = [], [], []
        for r, toks in enumerate(docs):
            for term, c in Counter(t for t in toks if t in self.index).items():
                j = self.index[term]
                tf = 1.0 + math.log(c) if self.sublinear else float(c)
                rows.append(r)
                cols.append(j)
                vals.append(tf * self.idf[j])
        m = sparse.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocab)), dtype=np.float64)
        return l2_normalize(m)


def l2_normalize(m: sparse.csr_matrix) -> sparse.csr_matrix:
    norms = np.sqrt(m.multiply(m).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    return sparse.diags(1.0 / norms) @ m


def fit_tfidf(
    docs: Sequence[Sequence[str]], *, min_df: int = 1, max_features: int | None = None, sublinear: bool = False
) -> TfidfModel:
    df: Counter[str] = Counter()
    tf_total: Counter[str] = Counter()
    for toks in docs:
        df.update(set(toks))
        tf_total.update(toks)
    terms = [t for t, c in df.items() if c >= min_df]
    if max_features is not None and len(terms) > max_features:
        terms = sorted(terms, key=lambda t: (-tf_total[t], t))[:max_features]
    terms = sorted(terms)
    n = len(docs)
    dfv = np.array([df[t] for t in terms], dtype=np.float64)
    idf = np.log((1 + n) / (1 + dfv)) + 1.0
    return TfidfModel(terms, {t: i for i, t in enumerate(terms)}, idf, n, dfv, sublinear)


def top_keywords(model: TfidfModel, matrix: sparse.csr_matrix, k: int = 20) -> list[tuple[str, float]]:
    """Corpus-level keywords ranked by mean TF-IDF weight over all documents."""
    if matrix.shape[0] == 0:
        return []
    mean_w = np.asarray(matrix.mean(axis=0)).ravel()
    order = np.lexsort((np.array(model.vocab), -mean_w))[:k]
    return [(model.vocab[i], float(mean_w[i])) for i in order]


def distribution_stats(docs: Sequence[Sequence[str]], top: int = 10) -> dict[str, float]:
    """Corpus-level descriptive statistics of the term distribution (vocabulary, concentration, entropy)."""
    tf: Counter[str] = Counter()
    for toks in docs:
        tf.update(toks)
    total = sum(tf.values())
    v = len(tf)
    if total == 0:
        return {"tokens": 0, "vocab": 0, "top_share": 0.0, "norm_entropy": 0.0, "hapax_share": 0.0}
    p = np.array(list(tf.values()), dtype=np.float64) / total
    entropy = float(-(p * np.log(p)).sum())
    top_share = float(sum(c for _, c in tf.most_common(top)) / total)
    return {
        "tokens": total,
        "vocab": v,
        f"top{top}_share": top_share,
        "norm_entropy": entropy / math.log(v) if v > 1 else 0.0,
        "hapax_share": sum(1 for c in tf.values() if c == 1) / v,
    }


def token_reduction(before: int, after: int) -> float:
    """Percentage of tokens removed (0 when there was nothing to remove)."""
    return 100.0 * (before - after) / before if before else 0.0
