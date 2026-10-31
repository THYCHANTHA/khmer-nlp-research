"""Descriptive analysis of what a strategy removes and keeps. Works for any Strategy (dictionary- or frequency-based).

Linguistic categories come only from an imported dictionary; nothing here assigns linguistic labels itself.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif

from khmer_nlp.stopwords.strategies import Strategy


def removal_profile(strategy: Strategy, docs: Sequence[Sequence[str]], top: int = 30) -> dict:
    """Total / removed token counts and the most frequently removed terms with their decision details."""
    rm = strategy.remove_set
    total = removed = 0
    freq: Counter[str] = Counter()
    for toks in docs:
        total += len(toks)
        for t in toks:
            if t in rm:
                removed += 1
                freq[t] += 1
    rows = []
    for term, c in freq.most_common(top):
        d = strategy.decisions[term]
        rows.append(
            {
                "word": term,
                "count": c,
                "share_of_tokens_pct": 100 * c / total if total else 0.0,
                "category": d.category,
                "tier": d.tier,
                "reason": d.reason,
            }
        )
    return {
        "tokens": total,
        "removed": removed,
        "removed_pct": 100 * removed / total if total else 0.0,
        "types_removed_observed": len(freq),
        "types_in_strategy": len(rm),
        "top_removed": rows,
    }


def category_contribution(strategy: Strategy, docs: Sequence[Sequence[str]]) -> pd.DataFrame:
    """Tokens and types per dictionary category among the terms the strategy removes."""
    rm = strategy.remove_set
    tok: Counter[str] = Counter()
    types: dict[str, set[str]] = defaultdict(set)
    total = 0
    for toks in docs:
        total += len(toks)
        for t in toks:
            if t in rm:
                cat = strategy.decisions[t].category or "(no category)"
                tok[cat] += 1
                types[cat].add(t)
    df = pd.DataFrame(
        [
            {"category": c, "tokens": n, "token_share_pct": 100 * n / total if total else 0.0, "types": len(types[c])}
            for c, n in tok.most_common()
        ]
    )
    return df if len(df) else pd.DataFrame(columns=["category", "tokens", "token_share_pct", "types"])


def retained_protected(strategy: Strategy, docs: Sequence[Sequence[str]], top: int = 30) -> pd.DataFrame:
    """Occurrences of terms that the dictionary lists but the strategy deliberately keeps (e.g. negations)."""
    keep = {w: d for w, d in strategy.decisions.items() if d.action == "keep" and d.tier == "protected"}
    cnt: Counter[str] = Counter(t for toks in docs for t in toks if t in keep)
    return pd.DataFrame(
        [
            {"word": w, "count": c, "category": keep[w].category, "reason": keep[w].reason}
            for w, c in cnt.most_common(top)
        ],
        columns=["word", "count", "category", "reason"],
    )


def potential_false_removals(
    strategy: Strategy, docs: Sequence[Sequence[str]], labels: Sequence[str], top: int = 20
) -> pd.DataFrame:
    """Removed terms that are nevertheless informative about the class label (mutual information, presence/absence).

    This flags *candidates* for manual review. A high value does not prove the removal is wrong.
    """
    rm = sorted(strategy.remove_set)
    if not rm or len(set(labels)) < 2:
        return pd.DataFrame(columns=["word", "mutual_info", "doc_freq"])
    cv = CountVectorizer(analyzer=lambda x: x, binary=True, vocabulary=rm)
    x = cv.transform(list(docs))
    mi = mutual_info_classif(x, np.asarray(labels), discrete_features=True, random_state=42)
    dfreq = np.asarray(x.sum(axis=0)).ravel()
    out = pd.DataFrame({"word": rm, "mutual_info": mi, "doc_freq": dfreq})
    return out[out.doc_freq > 0].sort_values("mutual_info", ascending=False).head(top).reset_index(drop=True)
