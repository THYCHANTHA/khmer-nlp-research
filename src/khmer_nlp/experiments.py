"""Reproducible experiments on a tokenised article collection. Every number written to ``reports/`` comes from here.

Public build: strategies are A (none), B-freq, C-freq (document-frequency lists) and a random-term control.
Dictionary-based tier strategies B/C run only when the user supplies a dictionary (outputs then go to a local folder).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from khmer_nlp.data.corpus import Article
from khmer_nlp.evaluation.metrics import bootstrap_ci, evaluate_ranking, paired_bootstrap_diff
from khmer_nlp.evaluation.protocol import (
    assert_no_leakage,
    category_members,
    near_duplicate_queries,
    proxy_relevant,
    split_queries,
    summarize_ranks,
    usable_categories,
)
from khmer_nlp.linguistic_analysis.analysis import potential_false_removals, removal_profile
from khmer_nlp.retrieval.index import RetrievalIndex
from khmer_nlp.stopwords.strategies import (
    Decision,
    Strategy,
    build_frequency_strategy,
    document_frequency,
    no_removal,
)
from khmer_nlp.tfidf.compute import distribution_stats, fit_tfidf, token_reduction, top_keywords

METRICS = ["P@5", "P@10", "R@5", "R@10", "MRR", "nDCG@10", "MeanRelRank"]


def build_random_strategy(docs, n: int, seed: int, name: str) -> Strategy:
    """Control: n randomly chosen vocabulary terms (seeded)."""
    df, _ = document_frequency(docs)
    rng = random.Random(seed)
    chosen = rng.sample(sorted(df), min(n, len(df)))
    strat = Strategy(name, f"{len(chosen)} randomly chosen terms (control).")
    for t in chosen:
        strat.decisions[t] = Decision(t, "", "random", "remove", "random control term", f"seed {seed}", True)
    return strat


def public_strategies(docs, cfg: dict) -> dict[str, Strategy]:
    s = cfg["strategies"]
    return {
        "A: none": no_removal(),
        f"B-freq: top {s['freq_b_top']} DF": build_frequency_strategy(
            docs, s["freq_b_top"], f"B-freq: top {s['freq_b_top']} DF"
        ),
        f"C-freq: top {s['freq_c_top']} DF": build_frequency_strategy(
            docs, s["freq_c_top"], f"C-freq: top {s['freq_c_top']} DF"
        ),
        f"control: random {s['random_control_top']}": build_random_strategy(
            docs, s["random_control_top"], cfg["seed"], f"control: random {s['random_control_top']}"
        ),
    }


def apply_all(strategy: Strategy, docs):
    rm = strategy.remove_set
    return [[t for t in d if t not in rm] for d in docs]


# ------------------------------------------------------------------ TF-IDF


def tfidf_experiment(articles: list[Article], cfg: dict, strategies: dict[str, Strategy]) -> dict:
    docs = [a.tokens for a in articles]
    base_tokens = sum(len(d) for d in docs)
    stats_rows, kw_rows, freq_rows, rf_rows = [], [], [], []
    top_sets: dict[str, set[str]] = {}
    for name, strat in strategies.items():
        kept = apply_all(strat, docs)
        st = distribution_stats(kept)
        model = fit_tfidf(kept, min_df=cfg["tfidf"]["min_df"], sublinear=cfg["tfidf"]["sublinear"])
        mat = model.transform(kept)
        kws = top_keywords(model, mat, max(cfg["tfidf"]["top_keywords"]))
        top_sets[name] = {w for w, _ in kws}
        nonempty = [d for d in kept if d]
        stats_rows.append(
            {
                "strategy": name,
                "documents": len(docs),
                "tokens": st["tokens"],
                "vocab": st["vocab"],
                "token_reduction_pct": token_reduction(base_tokens, st["tokens"]),
                "vocab_tfidf_min_df": len(model.vocab),
                "top10_token_share": st["top10_share"],
                "norm_entropy": st["norm_entropy"],
                "hapax_share": st["hapax_share"],
                "mean_types_per_doc": float(np.mean([len(set(d)) for d in nonempty])) if nonempty else 0.0,
                "empty_documents": len(kept) - len(nonempty),
            }
        )
        for r, (w, v) in enumerate(kws, start=1):
            kw_rows.append({"strategy": name, "rank": r, "term": w, "mean_tfidf": v})
        tf = Counter(t for d in kept for t in d)
        df, n = document_frequency(kept)
        counts = sorted(tf.values(), reverse=True)
        ranks = sorted({int(round(x)) for x in np.geomspace(1, max(len(counts), 1), 120)})
        rf_rows.extend({"strategy": name, "rank": r, "tf": counts[r - 1]} for r in ranks if r <= len(counts))
        for r, (w, c) in enumerate(tf.most_common(30), start=1):
            freq_rows.append({"strategy": name, "rank": r, "term": w, "tf": c, "df": df[w], "df_share": df[w] / n})
    names = list(strategies)
    base = top_sets[names[0]]
    overlap = [
        {"strategy": n, "top_keyword_jaccard_vs_A": len(base & top_sets[n]) / len(base | top_sets[n])} for n in names
    ]
    return {
        "stats": pd.DataFrame(stats_rows),
        "keywords": pd.DataFrame(kw_rows),
        "top_terms": pd.DataFrame(freq_rows),
        "rank_frequency": pd.DataFrame(rf_rows),
        "overlap": pd.DataFrame(overlap),
    }


# ------------------------------------------------------------------ retrieval


def retrieval_experiment(articles: list[Article], cfg: dict, seed: int | None = None) -> dict:
    r = cfg["retrieval"]
    seed = cfg["seed"] if seed is None else seed
    by_id = {a.pageid: a for a in articles}
    cats = {a.pageid: a.categories for a in articles}
    members = category_members(cats)
    usable = usable_categories(members, r["category_min_size"], r["category_max_size"])
    eligible = [a.pageid for a in articles if a.categories & usable]
    n_q = min(r["n_queries"], len(eligible) // 3)
    q_pool, _ = split_queries(eligible, n_q, seed)
    q_ids = set(q_pool)
    index_ids = [a.pageid for a in articles if a.pageid not in q_ids]
    assert_no_leakage(q_pool, index_ids)

    qlen = r["query_tokens"]
    q_orig = {q: by_id[q].tokens[:qlen] for q in q_pool}
    idx_orig = {i: by_id[i].tokens for i in index_ids}
    dup = near_duplicate_queries(q_orig, idx_orig, r["near_duplicate_threshold"])
    rel_sets = {q: proxy_relevant(q, cats, usable, set(index_ids)) for q in q_pool if q not in dup}
    queries = [q for q, rel in rel_sets.items() if rel]
    meta = {
        "seed": seed,
        "usable_categories": len(usable),
        "eligible_articles": len(eligible),
        "queries_sampled": len(q_pool),
        "dropped_near_duplicate": len(dup),
        "dropped_no_relevant": len(rel_sets) - len(queries),
        "queries_evaluated": len(queries),
        "indexed_documents": len(index_ids),
        "mean_relevant_per_query": float(np.mean([len(rel_sets[q]) for q in queries])) if queries else 0.0,
        "median_relevant_per_query": float(np.median([len(rel_sets[q]) for q in queries])) if queries else 0.0,
    }

    index_docs = [by_id[i].tokens for i in index_ids]
    strategies = public_strategies(index_docs, cfg)  # lists are learned from the INDEXED documents only
    per_query: dict[str, dict[str, list[float]]] = {}
    for name, strat in strategies.items():
        idx = RetrievalIndex(index_ids, apply_all(strat, index_docs), sublinear=True)
        rows: dict[str, list[float]] = {m: [] for m in METRICS}
        for q in queries:
            qt, _ = strat.apply(q_orig[q])
            ranking = idx.rank(qt)
            for m, v in evaluate_ranking(ranking, rel_sets[q]).items():
                rows[m].append(v)
        per_query[name] = rows
    return {"meta": meta, "per_query": per_query, "strategies": list(strategies)}


def summarize_retrieval(res: dict, n_boot: int, seed: int) -> dict[str, pd.DataFrame]:
    names = res["strategies"]
    base = names[0]
    summ, paired = [], []
    for n in names:
        row = {"strategy": n}
        for m in METRICS:
            mean, lo, hi = bootstrap_ci(res["per_query"][n][m], n_boot, seed)
            row.update({m: mean, f"{m}_lo": lo, f"{m}_hi": hi})
        summ.append(row)
        if n != base:
            for m in METRICS:
                d, lo, hi = paired_bootstrap_diff(res["per_query"][base][m], res["per_query"][n][m], n_boot, seed)
                paired.append(
                    {
                        "strategy": n,
                        "metric": m,
                        "diff_vs_A": d,
                        "lo": lo,
                        "hi": hi,
                        "ci_excludes_zero": bool(lo > 0 or hi < 0),
                    }
                )
    return {"summary": pd.DataFrame(summ), "paired": pd.DataFrame(paired)}


def known_item_experiment(articles: list[Article], cfg: dict) -> pd.DataFrame:
    """Sanity check: the query document is itself in the index. NOT a measure of retrieval quality."""
    r = cfg["retrieval"]
    raw = {a.pageid: a.tokens for a in articles}
    q_ids = random.Random(cfg["seed"]).sample(sorted(raw), min(r["n_queries"], len(raw)))
    rows = []
    for name, strat in public_strategies(list(raw.values()), cfg).items():
        idx = RetrievalIndex(list(raw), apply_all(strat, raw.values()), sublinear=True)
        for variant, k in (("first 40 tokens", r["query_tokens"]), ("full document", None)):
            ranks = []
            for q in q_ids:
                qt, _ = strat.apply(raw[q][:k] if k else raw[q])
                ranks.append(idx.rank(qt).index(q) + 1)
            rows.append({"query": variant, "strategy": name, **summarize_ranks(ranks)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ classification


def classification_labels(articles: list[Article], cfg: dict) -> tuple[list[Article], list[str], dict]:
    """Single-label subset: articles in exactly one of the K largest categories (K fixed in the config)."""
    c = cfg["classification"]
    members = category_members({a.pageid: a.categories for a in articles})
    big = sorted(((len(m), name) for name, m in members.items() if len(m) >= c["min_class_size"]), reverse=True)
    classes = [name for _, name in big[: c["max_classes"]]]
    chosen = set(classes)
    arts, labels = [], []
    for a in articles:
        hit = a.categories & chosen
        if len(hit) == 1:
            arts.append(a)
            labels.append(next(iter(hit)))
    info = {
        "classes": {k: v for k, v in Counter(labels).most_common()},
        "articles_labelled": len(arts),
        "articles_total": len(articles),
    }
    return arts, labels, info


def classification_experiment(articles: list[Article], cfg: dict) -> dict:
    arts, labels, info = classification_labels(articles, cfg)
    c = cfg["classification"]
    y = np.array(labels)
    counts = Counter(labels)
    keep = [i for i, lab in enumerate(labels) if counts[lab] >= c["folds"]]
    docs = [arts[i].tokens for i in keep]
    y = y[keep]
    skf = StratifiedKFold(n_splits=c["folds"], shuffle=True, random_state=cfg["seed"])
    rows = []
    maj = Counter(y).most_common(1)[0][1] / len(y)
    for fold, (tr, te) in enumerate(skf.split(docs, y), start=1):
        train_docs = [docs[i] for i in tr]
        for name, strat in public_strategies(train_docs, cfg).items():
            tr_x = apply_all(strat, train_docs)
            te_x = apply_all(strat, [docs[i] for i in te])
            model = fit_tfidf(tr_x, min_df=cfg["tfidf"]["min_df"], sublinear=True)
            clf = LinearSVC(C=c["C"], random_state=cfg["seed"], dual="auto").fit(model.transform(tr_x), y[tr])
            pred = clf.predict(model.transform(te_x))
            rows.append(
                {
                    "fold": fold,
                    "strategy": name,
                    "accuracy": accuracy_score(y[te], pred),
                    "macro_f1": f1_score(y[te], pred, average="macro"),
                }
            )
    df = pd.DataFrame(rows)
    names = list(df.strategy.unique())
    summary = (
        df.groupby("strategy", sort=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd=("macro_f1", "std"),
        )
        .reset_index()
    )
    base = df[df.strategy == names[0]].set_index("fold")
    diffs = []
    for n in names[1:]:
        cur = df[df.strategy == n].set_index("fold")
        d = cur.macro_f1 - base.macro_f1
        diffs.append(
            {
                "strategy": n,
                "macro_f1_diff_vs_A_mean": d.mean(),
                "macro_f1_diff_sd": d.std(),
                "folds_better": int((d > 0).sum()),
                "folds_worse": int((d < 0).sum()),
            }
        )
    info.update({"documents_used": len(docs), "majority_class_accuracy": maj, "folds": c["folds"]})
    return {"info": info, "folds": df, "summary": summary, "diffs": pd.DataFrame(diffs), "labels": (docs, list(y))}


# ------------------------------------------------------------------ linguistic descriptives


def linguistic_experiment(articles: list[Article], cfg: dict, cls: dict) -> dict:
    docs = [a.tokens for a in articles]
    out = {}
    strategies = public_strategies(docs, cfg)
    ldocs, ylab = cls["labels"]
    fr = {}
    for name, strat in strategies.items():
        if name.startswith("A"):
            continue
        prof = removal_profile(strat, docs, top=30)
        out[name] = {k: v for k, v in prof.items() if k != "top_removed"}
        out[name]["top_removed"] = prof["top_removed"]
        fr[name] = potential_false_removals(strat, ldocs, ylab, top=20)
    return {"profiles": out, "possible_false_removals": fr}


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
