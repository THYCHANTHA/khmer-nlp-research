"""Evaluation protocol helpers: held-out queries, leakage guards, proxy relevance, known-item demonstration.

Three kinds of evaluation are kept apart:
1. known-item (the query document is IN the index): sanity check only, cannot show retrieval quality;
2. proxy evaluation with category-derived labels (weak, noisy, NOT human judgements);
3. human-judged relevance: not available (future work).
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence

import numpy as np

from khmer_nlp.retrieval.index import RetrievalIndex


class LeakageError(AssertionError):
    pass


def split_queries(ids: Sequence[Hashable], n_queries: int, seed: int = 42) -> tuple[list, list]:
    """Disjoint (query_ids, index_ids). Deterministic for a given seed."""
    if n_queries >= len(ids):
        raise ValueError("n_queries must be smaller than the collection")
    rng = random.Random(seed)
    order = sorted(ids, key=str)
    queries = sorted(rng.sample(order, n_queries), key=str)
    qs = set(queries)
    return queries, [i for i in order if i not in qs]


def assert_no_leakage(query_ids: Sequence[Hashable], index_ids: Sequence[Hashable]) -> None:
    both = set(query_ids) & set(index_ids)
    if both:
        raise LeakageError(f"{len(both)} query documents are also indexed, e.g. {sorted(map(str, both))[:3]}")


def near_duplicate_queries(
    query_docs: Mapping[Hashable, Sequence[str]], index_docs: Mapping[Hashable, Sequence[str]], threshold: float = 0.9
) -> dict[Hashable, float]:
    """Queries whose cosine similarity to some indexed document reaches ``threshold`` (text duplicates)."""
    idx = RetrievalIndex(list(index_docs), list(index_docs.values()), sublinear=True)
    flagged = {}
    for qid, toks in query_docs.items():
        s = idx.scores(toks)
        if s.size and s.max() >= threshold:
            flagged[qid] = float(s.max())
    return flagged


def usable_categories(members: Mapping[str, set], min_size: int = 5, max_size: int = 60) -> set[str]:
    """Categories specific enough to define a topic: neither tiny nor huge (huge ones relate unrelated pages)."""
    return {c for c, m in members.items() if min_size <= len(m) <= max_size}


def category_members(doc_categories: Mapping[Hashable, set[str]]) -> dict[str, set]:
    out: dict[str, set] = defaultdict(set)
    for d, cats in doc_categories.items():
        for c in cats:
            out[c].add(d)
    return dict(out)


def proxy_relevant(
    query_id: Hashable, doc_categories: Mapping[Hashable, set[str]], usable: set[str], index_ids: set
) -> set:
    """Indexed documents sharing at least one usable category with the query document (weak proxy label)."""
    qcats = doc_categories.get(query_id, set()) & usable
    if not qcats:
        return set()
    return {d for d in index_ids if doc_categories.get(d, set()) & qcats}


def known_item_ranks(
    docs: Mapping[Hashable, Sequence[str]], query_ids: Sequence[Hashable], k_tokens: int | None = None
) -> list[int]:
    """Demonstration only: the query is a document that is itself indexed; returns its rank for each query."""
    idx = RetrievalIndex(list(docs), list(docs.values()), sublinear=True)
    ranks = []
    for qid in query_ids:
        toks = list(docs[qid])[:k_tokens] if k_tokens else docs[qid]
        ranks.append(idx.rank(toks).index(qid) + 1)
    return ranks


def summarize_ranks(ranks: Sequence[int]) -> dict[str, float]:
    r = np.asarray(ranks, dtype=float)
    return {"mean_rank": float(r.mean()), "top1": float((r == 1).mean()), "top10": float((r <= 10).mean())}
