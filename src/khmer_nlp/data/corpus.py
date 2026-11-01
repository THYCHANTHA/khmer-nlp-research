"""Corpus preparation from the downloaded Wikipedia leads: quality filter, category cleaning, tokenisation cache."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from khmer_nlp.normalization.text import has_khmer, khmer_ratio
from khmer_nlp.tokenization.tokenizers import Tokenizer, tokenize

CATEGORY_PREFIX = "ចំណាត់ថ្នាក់ក្រុម:"


@dataclass
class Article:
    pageid: int
    title: str
    url: str
    revid: int | None
    tokens: list[str]
    categories: set[str] = field(default_factory=set)
    n_chars: int = 0


def clean_category(raw: str) -> str:
    return raw[len(CATEGORY_PREFIX) :] if raw.startswith(CATEGORY_PREFIX) else raw


def is_topical_category(name: str, maintenance_words: list[str]) -> bool:
    """Heuristic: wiki-maintenance categories are Latin-script or contain a Khmer maintenance word."""
    if any(c.isascii() and c.isalpha() for c in name):
        return False
    return not any(w in name for w in maintenance_words)


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def prepare(
    records: list[dict],
    tokenizer: Tokenizer,
    *,
    min_chars: int = 200,
    min_khmer_ratio: float = 0.5,
    min_tokens: int = 20,
    maintenance_words: list[str] | None = None,
) -> tuple[list[Article], dict]:
    """Filter and tokenise. Returns articles plus a report of how many records each filter removed."""
    mw = maintenance_words or []
    rep = {"input": len(records), "too_short": 0, "not_khmer": 0, "few_tokens": 0, "kept": 0}
    out: list[Article] = []
    for r in records:
        text = r.get("text") or ""
        if len(text) < min_chars:
            rep["too_short"] += 1
            continue
        if not has_khmer(text) or khmer_ratio(text) < min_khmer_ratio:
            rep["not_khmer"] += 1
            continue
        toks = tokenize(text, tokenizer).tokens
        if len(toks) < min_tokens:
            rep["few_tokens"] += 1
            continue
        cats = {c for c in map(clean_category, r.get("categories", [])) if is_topical_category(c, mw)}
        out.append(Article(r["pageid"], r["title"], r.get("url", ""), r.get("revid"), toks, cats, len(text)))
        rep["kept"] += 1
    return out, rep


def save_tokens(articles: list[Article], path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        for a in articles:
            row = {
                "pageid": a.pageid,
                "title": a.title,
                "url": a.url,
                "revid": a.revid,
                "tokens": a.tokens,
                "categories": sorted(a.categories),
                "n_chars": a.n_chars,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_tokens(path: Path) -> tuple[list[Article], dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        meta = json.loads(fh.readline())["meta"]
        arts = []
        for line in fh:
            d = json.loads(line)
            arts.append(
                Article(d["pageid"], d["title"], d["url"], d["revid"], d["tokens"], set(d["categories"]), d["n_chars"])
            )
    return arts, meta


ANALYSIS_MANIFEST_COLUMNS = [
    "pageid",
    "title",
    "revid",
    "timestamp",
    "lead_chars",
    "lead_sha256",
    "status",
    "exclusion_reason",
    "tokens",
    "topical_categories",
]
EXCLUSION_REASONS = {
    "too_short": "lead shorter than the minimum number of characters",
    "not_khmer": "Khmer-script share of non-space characters below the minimum",
    "few_tokens": "fewer than the minimum number of tokens after tokenisation",
}


def analysis_manifest_rows(
    records: list[dict],
    included: dict[int, Article],
    *,
    min_chars: int,
    min_khmer_ratio: float,
    min_tokens: int = 0,
) -> list[dict]:
    """One row per downloaded article: included or excluded, and why.

    Applies the same checks, in the same order, as ``prepare``. No article text is stored.
    ``lead_sha256`` is the SHA-256 of the lead text as downloaded (UTF-8): a re-download can be
    compared with the analysed version without redistributing the text. ``min_tokens`` is
    accepted for symmetry with ``prepare``; articles that pass the first two checks but are not
    in ``included`` are reported as ``few_tokens``.
    """
    rows = []
    for r in sorted(records, key=lambda x: x["pageid"]):
        text = r.get("text") or ""
        art = included.get(r["pageid"])
        reason = ""
        if art is None:
            if len(text) < min_chars:
                reason = "too_short"
            elif not has_khmer(text) or khmer_ratio(text) < min_khmer_ratio:
                reason = "not_khmer"
            else:
                reason = "few_tokens"
        rows.append(
            {
                "pageid": r["pageid"],
                "title": r["title"],
                "revid": r.get("revid"),
                "timestamp": r.get("timestamp"),
                "lead_chars": len(text),
                "lead_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "status": "included" if art is not None else "excluded",
                "exclusion_reason": reason,
                "tokens": len(art.tokens) if art is not None else "",
                "topical_categories": len(art.categories) if art is not None else "",
            }
        )
    return rows


def write_analysis_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ANALYSIS_MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def sample_checksum(rows: list[dict]) -> str:
    """SHA-256 over the sorted 'pageid:revid' lines of the included articles: identifies the analysed sample."""
    lines = sorted(f"{r['pageid']}:{r['revid']}" for r in rows if r["status"] == "included")
    return hashlib.sha256(chr(10).join(lines).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
