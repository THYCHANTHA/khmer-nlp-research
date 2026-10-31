"""Vector-space retrieval: TF-IDF vectors, cosine similarity, full rankings."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np

from khmer_nlp.tfidf.compute import fit_tfidf


class RetrievalIndex:
    """TF-IDF index over a fixed collection. IDF comes from the indexed documents only (queries never leak in)."""

    def __init__(
        self, ids: Sequence[Hashable], docs: Sequence[Sequence[str]], *, min_df: int = 1, sublinear: bool = True
    ):
        if len(ids) != len(docs):
            raise ValueError("ids and docs must have the same length")
        if len(set(ids)) != len(ids):
            raise ValueError("document ids must be unique")
        self.ids = list(ids)
        self.model = fit_tfidf(docs, min_df=min_df, sublinear=sublinear)
        self.matrix = self.model.transform(docs)

    def scores(self, query: Sequence[str]) -> np.ndarray:
        """Cosine similarity of the query to every indexed document (0 for an empty / out-of-vocabulary query)."""
        q = self.model.transform([list(query)])
        return (self.matrix @ q.T).toarray().ravel()

    def rank(self, query: Sequence[str]) -> list[Hashable]:
        """Full ranking, best first; ties keep collection order (stable sort) so results are deterministic."""
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")
        return [self.ids[i] for i in order]

    def search(self, query: Sequence[str], k: int = 10) -> list[tuple[Hashable, float]]:
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")[:k]
        return [(self.ids[i], float(s[i])) for i in order]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    return float(x @ y / (nx * ny)) if nx and ny else 0.0
