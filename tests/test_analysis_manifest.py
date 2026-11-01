import csv
import hashlib
import json
from pathlib import Path

from khmer_nlp.data.corpus import (
    ANALYSIS_MANIFEST_COLUMNS,
    analysis_manifest_rows,
    prepare,
    sample_checksum,
    write_analysis_manifest,
)
from khmer_nlp.tokenization.tokenizers import WhitespaceTokenizer

ROOT = Path(__file__).resolve().parents[1]
KH = "កខគឃង"
KW = {"min_chars": 200, "min_khmer_ratio": 0.5, "min_tokens": 20}


def rec(pid, text):
    return {"pageid": pid, "title": f"t{pid}", "revid": pid * 10, "timestamp": "2026-01-01T00:00:00Z", "text": text}


RECORDS = [
    rec(5, (KH + " ") * 40),  # included
    rec(1, "short"),  # too short
    rec(2, "abc " * 80),  # long but not Khmer
    rec(3, (KH + " ") * 40),  # included
    rec(4, KH * 50),  # Khmer, long, but a single whitespace token
]


def build():
    arts, report = prepare(RECORDS, WhitespaceTokenizer(), **KW)
    rows = analysis_manifest_rows(RECORDS, {a.pageid: a for a in arts}, **KW)
    return arts, report, rows


def test_manifest_has_one_row_per_record_sorted_by_page_id():
    _, _, rows = build()
    assert [r["pageid"] for r in rows] == [1, 2, 3, 4, 5]


def test_status_and_reason_agree_with_prepare():
    arts, report, rows = build()
    status = {r["pageid"]: (r["status"], r["exclusion_reason"]) for r in rows}
    assert status == {
        1: ("excluded", "too_short"),
        2: ("excluded", "not_khmer"),
        3: ("included", ""),
        4: ("excluded", "few_tokens"),
        5: ("included", ""),
    }
    assert {r["pageid"] for r in rows if r["status"] == "included"} == {a.pageid for a in arts}
    for reason in ("too_short", "not_khmer", "few_tokens"):
        assert sum(r["exclusion_reason"] == reason for r in rows) == report[reason]
    assert sum(r["status"] == "included" for r in rows) == report["kept"]


def test_no_article_text_is_stored_only_a_hash():
    _, _, rows = build()
    assert set(rows[0]) == set(ANALYSIS_MANIFEST_COLUMNS)
    assert "text" not in ANALYSIS_MANIFEST_COLUMNS
    expect = hashlib.sha256(RECORDS[0]["text"].encode("utf-8")).hexdigest()
    assert {r["pageid"]: r["lead_sha256"] for r in rows}[5] == expect


def test_sample_checksum_is_order_independent_and_sensitive_to_revisions():
    _, _, rows = build()
    c1 = sample_checksum(rows)
    assert c1 == sample_checksum(list(reversed(rows)))
    changed = [dict(r) for r in rows]
    changed[2]["revid"] += 1  # a different revision of an included article
    assert sample_checksum(changed) != c1


def test_manifest_round_trips_through_csv(tmp_path):
    _, _, rows = build()
    p = tmp_path / "m.csv"
    write_analysis_manifest(rows, p)
    back = list(csv.DictReader(p.open(encoding="utf-8")))
    assert [r["pageid"] for r in back] == ["1", "2", "3", "4", "5"]
    assert list(back[0]) == ANALYSIS_MANIFEST_COLUMNS


def test_tracked_manifest_matches_tracked_summary_and_download_manifest():
    man = list(csv.DictReader((ROOT / "data/manifest/km_wikipedia_analysis_sample.csv").open(encoding="utf-8")))
    dl = list(csv.DictReader((ROOT / "data/manifest/km_wikipedia_manifest.csv").open(encoding="utf-8")))
    summary = json.loads((ROOT / "reports/experiments/corpus_summary.json").read_text(encoding="utf-8"))
    assert {r["pageid"] for r in man} == {r["pageid"] for r in dl}
    assert len({r["pageid"] for r in man}) == len(man)
    dl_by_id = {r["pageid"]: r for r in dl}
    assert all(r["revid"] == dl_by_id[r["pageid"]]["revid"] for r in man)
    included = [r for r in man if r["status"] == "included"]
    assert len(included) == summary["articles"] == summary["analysis_sample"]["included"]
    assert len(man) == summary["analysis_sample"]["articles_downloaded"] == summary["filter_report"]["input"]
    for reason in ("too_short", "not_khmer", "few_tokens"):
        n = sum(r["exclusion_reason"] == reason for r in man)
        assert n == summary["filter_report"][reason] == summary["analysis_sample"]["excluded_by_reason"][reason]
    assert all(r["exclusion_reason"] == "" for r in included)
    assert sum(int(r["tokens"]) for r in included) == summary["tokens"]
    assert sample_checksum(man) == summary["analysis_sample"]["sample_sha256"]
