import math

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from khmer_nlp.evaluation.metrics import (
    bootstrap_ci,
    evaluate_ranking,
    mean_relevant_rank,
    ndcg_at_k,
    paired_bootstrap_diff,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from khmer_nlp.evaluation.protocol import (
    LeakageError,
    assert_no_leakage,
    category_members,
    known_item_ranks,
    near_duplicate_queries,
    proxy_relevant,
    split_queries,
    summarize_ranks,
    usable_categories,
)
from khmer_nlp.linguistic_analysis.analysis import (
    category_contribution,
    potential_false_removals,
    removal_profile,
    retained_protected,
)
from khmer_nlp.retrieval.index import RetrievalIndex, cosine
from khmer_nlp.stopwords.strategies import build_tier_strategy
from khmer_nlp.tfidf.compute import (
    distribution_stats,
    fit_tfidf,
    token_reduction,
    top_keywords,
)

# ---------------------------------------------------------------- TF-IDF


def test_tfidf_matches_scikit_learn(docs):
    nonempty = [d for d in docs if d]
    m = fit_tfidf(nonempty)
    ours = m.transform(nonempty).toarray()
    sk = TfidfVectorizer(analyzer=lambda x: x, norm="l2", smooth_idf=True, sublinear_tf=False)
    ref = sk.fit_transform(nonempty).toarray()
    order = [sk.vocabulary_[t] for t in m.vocab]
    assert np.allclose(ours, ref[:, order], atol=1e-12)


def test_tfidf_hand_computed_idf():
    m = fit_tfidf([["ក", "ខ"], ["ក"]])
    # N=2: df(ក)=2 -> ln(3/3)+1 = 1 ; df(ខ)=1 -> ln(3/2)+1
    assert m.idf[m.index["ក"]] == pytest.approx(1.0)
    assert m.idf[m.index["ខ"]] == pytest.approx(math.log(1.5) + 1)


def test_tfidf_rows_are_unit_length_and_empty_doc_is_zero(docs):
    m = fit_tfidf(docs)
    x = m.transform(docs)
    norms = np.sqrt(x.multiply(x).sum(axis=1)).A1
    assert np.allclose(norms[:4], 1.0) and norms[4] == 0.0


def test_unseen_terms_are_ignored_and_min_df_applies():
    m = fit_tfidf([["ក", "ខ"], ["ក"]], min_df=2)
    assert m.vocab == ["ក"]
    assert m.transform([["ឃ"]]).nnz == 0


def test_top_keywords_and_stats(docs):
    m = fit_tfidf(docs)
    kw = top_keywords(m, m.transform(docs), 3)
    assert len(kw) == 3 and kw[0][1] >= kw[1][1] >= kw[2][1]
    s = distribution_stats(docs)
    assert s["tokens"] == 20 and s["vocab"] == len({t for d in docs for t in d})
    assert 0 < s["norm_entropy"] <= 1
    assert distribution_stats([[], []])["tokens"] == 0
    assert top_keywords(fit_tfidf([]), fit_tfidf([]).transform([])) == []


def test_token_reduction():
    assert token_reduction(10, 7) == pytest.approx(30.0)
    assert token_reduction(0, 0) == 0.0


# ---------------------------------------------------------------- retrieval


def test_cosine_values():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([1, 1], [1, 0]) == pytest.approx(1 / math.sqrt(2))
    assert cosine([0, 0], [1, 0]) == 0.0


def test_index_ranks_the_matching_document_first(docs):
    idx = RetrievalIndex(list("abcde"), docs)
    assert idx.rank(["ស្រូវ", "ដី"])[0] in {"c", "d"}
    assert idx.rank(["សិស្ស"])[0] == "a"
    s = idx.scores([])  # empty query: all zero, no crash
    assert not s.any()
    top = idx.search(["សាលា"], k=2)
    assert top[0][1] >= top[1][1] and top[0][0] in {"a", "b"}


def test_ranking_is_deterministic_on_ties():
    idx = RetrievalIndex([1, 2, 3], [["ក"], ["ខ"], ["គ"]])
    assert idx.rank(["ឃ"]) == [1, 2, 3]


def test_index_rejects_duplicate_ids_and_length_mismatch():
    with pytest.raises(ValueError):
        RetrievalIndex([1, 1], [["ក"], ["ខ"]])
    with pytest.raises(ValueError):
        RetrievalIndex([1], [["ក"], ["ខ"]])


# ---------------------------------------------------------------- metrics


def test_metrics_hand_computed():
    ranking = ["a", "b", "c", "d", "e"]
    rel = {"b", "e", "z"}
    assert precision_at_k(ranking, rel, 5) == pytest.approx(2 / 5)
    assert recall_at_k(ranking, rel, 5) == pytest.approx(2 / 3)
    assert reciprocal_rank(ranking, rel) == pytest.approx(1 / 2)
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(ranking, rel, 5) == pytest.approx(dcg / ideal)
    assert mean_relevant_rank(ranking, rel) == pytest.approx((2 + 5) / 2)
    assert reciprocal_rank(["x"], {"y"}) == 0.0
    assert math.isnan(recall_at_k(ranking, set(), 5)) and math.isnan(ndcg_at_k(ranking, set(), 5))
    with pytest.raises(ValueError):
        precision_at_k(ranking, rel, 0)
    assert set(evaluate_ranking(ranking, rel)) == {"P@5", "P@10", "R@5", "R@10", "MRR", "nDCG@10", "MeanRelRank"}


def test_perfect_ranking_has_ndcg_one():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 10) == pytest.approx(1.0)


def test_bootstrap_is_seeded_and_ordered():
    v = [0.0, 1.0, 0.5, 0.25, 0.75]
    m1, lo1, hi1 = bootstrap_ci(v, seed=1)
    assert (m1, lo1, hi1) == bootstrap_ci(v, seed=1) and lo1 <= m1 <= hi1
    assert math.isnan(bootstrap_ci([float("nan")])[0])
    d, lo, hi = paired_bootstrap_diff([0, 0, 0, 0], [1, 1, 1, 1])
    assert d == lo == hi == 1.0


# ---------------------------------------------------------------- leakage protocol


def test_split_is_disjoint_and_deterministic():
    ids = list(range(100))
    q1, i1 = split_queries(ids, 10, seed=7)
    q2, i2 = split_queries(ids, 10, seed=7)
    assert (q1, i1) == (q2, i2) and not set(q1) & set(i1) and len(q1) + len(i1) == 100
    assert split_queries(ids, 10, seed=8)[0] != q1
    with pytest.raises(ValueError):
        split_queries(ids, 100)


def test_leakage_assertion_raises_when_query_is_indexed():
    assert_no_leakage([1, 2], [3, 4])
    with pytest.raises(LeakageError):
        assert_no_leakage([1, 2], [2, 3])


def test_near_duplicate_queries_are_flagged():
    index_docs = {1: ["ក", "ខ", "គ", "ឃ"], 2: ["ង", "ច", "ឆ", "ជ"]}
    queries = {"dup": ["ក", "ខ", "គ", "ឃ"], "fresh": ["ញ", "ដ", "ឋ", "ឌ"]}
    flagged = near_duplicate_queries(queries, index_docs, threshold=0.9)
    assert set(flagged) == {"dup"}


def test_known_item_is_trivially_rank_one():
    docs = {i: [f"t{i}", "common", f"u{i}"] for i in range(20)}
    ranks = known_item_ranks(docs, [0, 5, 9])
    assert ranks == [1, 1, 1]  # identity retrieval: says nothing about general quality
    assert summarize_ranks(ranks) == {"mean_rank": 1.0, "top1": 1.0, "top10": 1.0}


def test_proxy_relevance_uses_specific_categories_and_excludes_query():
    cats = {1: {"A", "big"}, 2: {"A"}, 3: {"big"}, 4: {"B"}, 5: {"big"}}
    members = category_members(cats)
    usable = usable_categories(members, min_size=2, max_size=3)
    assert usable == {"A", "big"}
    rel = proxy_relevant(1, cats, usable={"A"}, index_ids={2, 3, 4, 5})
    assert rel == {2}
    assert proxy_relevant(4, cats, usable={"A"}, index_ids={2, 3}) == set()  # no usable category -> no labels


# ---------------------------------------------------------------- linguistic analysis


def test_removal_profile_and_category_contribution(fixture_dict, config, docs):
    b = build_tier_strategy(fixture_dict, config, [1])
    prof = removal_profile(b, docs)
    assert prof["tokens"] == 20 and prof["removed"] == 4  # នៃ x2, និង x2
    assert prof["removed_pct"] == pytest.approx(20.0)
    cc = category_contribution(b, docs)
    assert cc.tokens.sum() == 4
    empty = category_contribution(build_tier_strategy(fixture_dict, config, [9]), docs)
    assert len(empty) == 0


def test_retained_protected_lists_negations(fixture_dict, config, docs):
    kept = retained_protected(build_tier_strategy(fixture_dict, config, [1, 2]), docs)
    assert set(kept.word) == {"មិន", "ទេ"} and kept["count"].sum() == 2


def test_potential_false_removals_flags_label_informative_terms(fixture_dict, config, docs):
    b = build_tier_strategy(fixture_dict, config, [1, 2])
    labels = ["school", "school", "farm", "farm", "school"]
    out = potential_false_removals(b, docs, labels, top=5)
    assert {"word", "mutual_info", "doc_freq"} <= set(out.columns)
    assert out.mutual_info.is_monotonic_decreasing
    assert potential_false_removals(b, docs, ["x"] * 5).empty
