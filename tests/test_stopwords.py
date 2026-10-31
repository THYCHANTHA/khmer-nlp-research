import pandas as pd
import pytest

from khmer_nlp.stopwords.dictionary import load_dictionary, load_flat_list, parse_rows
from khmer_nlp.stopwords.strategies import (
    build_frequency_strategy,
    build_tier_strategy,
    map_label,
    no_removal,
)


def test_dictionary_report_counts(fixture_dict):
    r = fixture_dict.report
    assert r.raw_rows == 17
    assert r.empty_entries == 1
    assert r.padded_entries == 1
    assert r.entries_with_invisible_chars == 1
    assert r.multiword_entries == 1  # only 'ពី ទៅ' contains a space
    assert r.non_khmer_entries == 1
    assert r.duplicate_entries == 2  # 'នៃ' twice (padded) + 'ដែល' twice
    assert r.conflicting_terms == 1
    assert r.unique_normalized == len(fixture_dict) == 14
    assert r.unique_raw >= r.unique_normalized


def test_multiword_count_is_consistent():
    d = parse_rows([("ក ខ", "x"), ("គ", "x")])
    assert d.report.multiword_entries == 1


def test_load_dictionary_handles_bom_and_missing_column(tmp_path):
    p = tmp_path / "d.csv"
    p.write_bytes("\N{ZERO WIDTH NO-BREAK SPACE}term,category\r\nនៃ,Prepositions\r\nនិង,Conjunctions\r\n".encode())
    d = load_dictionary(p)
    assert len(d) == 2 and d.report.has_categories
    with pytest.raises(ValueError):
        load_dictionary(p, term_col="word")
    flat = load_dictionary(p, category_col=None)
    assert not flat.report.has_categories


def test_load_flat_list(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("នៃ\nនិង\n\nនៃ\n", encoding="utf-8")
    d = load_flat_list(p)
    assert len(d) == 2 and d.report.empty_entries == 1 and d.report.duplicate_entries == 1


def test_label_mapping_is_generic(config):
    assert map_label("Prepositions / Relational Words", config) == {"preposition"}
    assert map_label("Question & Negation Words", config) == {"negation", "question"}
    assert map_label("Mystery", config) == set()


def test_strategy_a_removes_nothing(docs):
    kept, removed = no_removal().apply(docs[0])
    assert kept == docs[0] and removed == []


def test_strategy_b_tier1_and_c_extends(fixture_dict, config, docs):
    b = build_tier_strategy(fixture_dict, config, [1])
    c = build_tier_strategy(fixture_dict, config, [1, 2], name="C")
    assert {"នៃ", "និង", "ដ៏", "ដែល", "កា"} <= b.remove_set
    assert "ខ្ញុំ" not in b.remove_set and "ខ្ញុំ" in c.remove_set and "នេះ" in c.remove_set
    assert b.remove_set < c.remove_set  # nested strategies
    kept, removed = b.apply(docs[0])
    assert kept == ["សិស្ស", "សាលា", "គ្រូ"] and removed == ["នៃ", "និង"]


def test_negations_are_never_removed(fixture_dict, config):
    for tiers in ([1], [1, 2], [1, 2, 3]):
        s = build_tier_strategy(fixture_dict, config, tiers)
        assert "មិន" not in s.remove_set and "ទេ" not in s.remove_set
        assert s.decisions["មិន"].tier == "protected" and s.decisions["មិន"].manual_review


def test_excluded_labels_phrases_and_unlabelled_are_kept(fixture_dict, config):
    s = build_tier_strategy(fixture_dict, config, [1, 2])
    for w in ("ទៅដល់ទីនោះ", "ធ្វើតាម", "ពី ទៅ", "ថា"):
        assert s.decisions[w].action == "keep" and s.decisions[w].manual_review


def test_conflicting_labels_flagged_for_review(fixture_dict, config):
    s = build_tier_strategy(fixture_dict, config, [1])
    d = s.decisions["ដែល"]
    assert d.action == "remove" and d.manual_review


def test_decision_table_columns_and_explain(fixture_dict, config):
    s = build_tier_strategy(fixture_dict, config, [1])
    t = s.decision_table()
    assert list(t.columns) == ["word", "category", "tier", "action", "reason", "source", "manual_review"]
    assert isinstance(t, pd.DataFrame) and len(t) == len(s.decisions)
    assert s.explain("នៃ").tier == "1" and s.explain("not-there") is None


def test_flat_dictionary_cannot_build_tier_strategy(config):
    flat = parse_rows([("នៃ", None)], has_categories=False)
    with pytest.raises(ValueError):
        build_tier_strategy(flat, config, [1])


def test_frequency_strategy(docs):
    s = build_frequency_strategy(docs, top_n=2, name="freq2")
    # df: សាលា 2, គ្រូ 2, នៃ 2, និង 2, ស្រូវ 2, ដី 2, កសិកម្ម 2 ... ties broken alphabetically (deterministic)
    assert len(s.remove_set) == 2
    assert s.remove_set == build_frequency_strategy(docs, 2, "freq2").remove_set  # deterministic
    assert all(d.tier == "data-driven" and d.manual_review for d in s.decisions.values())


def test_apply_does_not_mutate_input(fixture_dict, config, docs):
    before = [list(d) for d in docs]
    build_tier_strategy(fixture_dict, config, [1, 2]).apply(docs[0])
    assert docs == before
