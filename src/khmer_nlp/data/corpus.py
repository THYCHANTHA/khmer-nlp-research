"""Corpus preparation from the downloaded Wikipedia leads: quality filter, category cleaning, tokenisation cache."""

from __future__ import annotations

import gzip
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
