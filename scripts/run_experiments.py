"""Run every public experiment and write results to reports/experiments/ (CSV / JSON).

    python scripts/run_experiments.py                # uses the cached token file if present
    python scripts/run_experiments.py --retokenize   # rebuild the cache with khmer-nltk
    python scripts/run_experiments.py --dictionary my.csv --term-col term --category-col category
        # optional: ALSO run dictionary tier strategies B/C with YOUR dictionary; outputs go to a git-ignored folder

No third-party stop-word dictionary is bundled or used by the public run.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib import metadata
from pathlib import Path

import pandas as pd
import yaml

from khmer_nlp import experiments as ex
from khmer_nlp.data.corpus import load_records, load_tokens, prepare, save_tokens
from khmer_nlp.tokenization.tokenizers import get_tokenizer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "km_wikipedia_leads.jsonl"
CACHE = ROOT / "data" / "processed" / "km_wikipedia_tokens.json.gz"
OUT = ROOT / "reports" / "experiments"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    ap.add_argument("--retokenize", action="store_true")
    ap.add_argument("--dictionary", type=Path, default=None)
    ap.add_argument("--term-col", default="term")
    ap.add_argument("--category-col", default="category")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    t0 = time.time()

    if args.retokenize or not CACHE.exists():
        tok = get_tokenizer(cfg["corpus"]["tokenizer"])
        arts, rep = prepare(
            load_records(RAW),
            tok,
            min_chars=cfg["corpus"]["min_chars"],
            min_khmer_ratio=cfg["corpus"]["min_khmer_ratio"],
            min_tokens=cfg["corpus"]["min_tokens"],
            maintenance_words=cfg["maintenance_words"],
        )
        meta = {
            "filter_report": rep,
            "tokenizer": tok.name,
            "tokenizer_version": getattr(tok, "version", None),
            "python": platform.python_version(),
        }
        save_tokens(arts, CACHE, meta)
    arts, meta = load_tokens(CACHE)
    print(f"{len(arts)} articles; tokenizer {meta['tokenizer']} {meta['tokenizer_version']}")

    # --- corpus summary (descriptive facts of the analysed sample)
    cats_per = [len(a.categories) for a in arts]
    corpus = {
        **meta,
        "articles": len(arts),
        "tokens": sum(len(a.tokens) for a in arts),
        "articles_with_topical_category": sum(1 for c in cats_per if c),
        "distinct_topical_categories": len({c for a in arts for c in a.categories}),
        "median_tokens_per_article": float(pd.Series([len(a.tokens) for a in arts]).median()),
        "revision_ids_recorded": sum(1 for a in arts if a.revid),
        "versions": {p: metadata.version(p) for p in ("khmer-nltk", "scikit-learn", "numpy", "pandas", "scipy")},
    }
    raw_summary = RAW.parent / "download_summary.json"
    if raw_summary.exists():
        corpus["download"] = json.loads(raw_summary.read_text(encoding="utf-8"))
    ex.save_json(corpus, OUT / "corpus_summary.json")

    # --- TF-IDF
    strategies = ex.public_strategies([a.tokens for a in arts], cfg)
    tf = ex.tfidf_experiment(arts, cfg, strategies)
    for k, df in tf.items():
        ex.save_table(df, OUT / f"tfidf_{k}.csv")
    ex.save_table(
        pd.concat([s.decision_table().assign(strategy=n) for n, s in strategies.items() if s.decisions]),
        OUT / "decision_table_public_strategies.csv",
    )

    # --- retrieval: proxy labels, primary seed + stability seeds
    res = ex.retrieval_experiment(arts, cfg)
    tabs = ex.summarize_retrieval(res, cfg["retrieval"]["bootstrap"], cfg["seed"])
    ex.save_table(tabs["summary"], OUT / "ir_proxy_summary.csv")
    ex.save_table(tabs["paired"], OUT / "ir_proxy_paired_vs_A.csv")
    ex.save_json(res["meta"], OUT / "ir_proxy_meta.json")
    seed_rows = []
    for sd in [cfg["seed"], *cfg["retrieval"]["extra_seeds"]]:
        r2 = res if sd == cfg["seed"] else ex.retrieval_experiment(arts, cfg, seed=sd)
        for n in r2["strategies"]:
            row = {"seed": sd, "strategy": n, "queries": r2["meta"]["queries_evaluated"]}
            row.update({m: sum(r2["per_query"][n][m]) / len(r2["per_query"][n][m]) for m in ex.METRICS})
            seed_rows.append(row)
    ex.save_table(pd.DataFrame(seed_rows), OUT / "ir_proxy_by_seed.csv")
    ex.save_table(ex.known_item_experiment(arts, cfg), OUT / "known_item_sanity_check.csv")

    # --- classification
    cls = ex.classification_experiment(arts, cfg)
    ex.save_json(cls["info"], OUT / "classification_info.json")
    for k in ("folds", "summary", "diffs"):
        ex.save_table(cls[k], OUT / f"classification_{k}.csv")

    # --- linguistic descriptives
    ling = ex.linguistic_experiment(arts, cfg, cls)
    ex.save_json(ling["profiles"], OUT / "removal_profiles.json")
    for name, df in ling["possible_false_removals"].items():
        ex.save_table(df, OUT / f"possible_false_removals_{name.split(':')[0]}.csv")

    # --- optional: user dictionary (never the public default)
    if args.dictionary:
        from khmer_nlp.stopwords.dictionary import load_dictionary
        from khmer_nlp.stopwords.strategies import build_tier_strategy, load_config

        d = load_dictionary(args.dictionary, args.term_col, args.category_col)
        local = ROOT / "reports" / "local_dictionary_run"
        ex.save_json(d.report.as_dict(), local / "dictionary_report.json")
        scfg = load_config()
        for tiers, nm in (([1], "B"), ([1, 2], "C")):
            s = build_tier_strategy(d, scfg, tiers, name=nm)
            ex.save_table(s.decision_table(), local / f"decision_table_{nm}.csv")
            ex.save_json(ex.removal_profile(s, [a.tokens for a in arts]), local / f"removal_profile_{nm}.json")
        print("local dictionary outputs written to", local)

    print(f"done in {time.time() - t0:.0f}s; results in {OUT}")


if __name__ == "__main__":
    main()
