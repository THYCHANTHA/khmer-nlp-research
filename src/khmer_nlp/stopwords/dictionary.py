"""Import and validation of an external stop-word dictionary (CSV).

This repository ships NO linguistic dictionary. A user supplies one (CSV with a term column and, optionally, a
category column). The importer reports data-quality facts and never alters the source file.
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from khmer_nlp.normalization.text import (
    DELETE_CHARS,
    ZWSP,
    is_khmer_token,
    normalize_dictionary_term,
)


@dataclass
class DictionaryReport:
    source: str
    raw_rows: int = 0
    unique_raw: int = 0
    unique_normalized: int = 0
    duplicate_entries: int = 0
    empty_entries: int = 0
    padded_entries: int = 0
    entries_with_invisible_chars: int = 0
    non_khmer_entries: int = 0
    multiword_entries: int = 0
    conflicting_terms: int = 0
    has_categories: bool = False
    category_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class StopwordDictionary:
    """Normalised term -> set of category labels (empty set for a flat list)."""

    entries: dict[str, set[str]]
    report: DictionaryReport

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def terms(self) -> set[str]:
        return set(self.entries)

    def categories_of(self, term: str) -> set[str]:
        return self.entries.get(normalize_dictionary_term(term), set())


def _clean_label(label: str) -> str:
    return re.sub(r"\s+", " ", label).strip()


def parse_rows(
    rows: list[tuple[str, str | None]], source: str = "<memory>", has_categories: bool = True
) -> StopwordDictionary:
    """Core of the importer, separated from file reading so that it can be tested with fixtures."""
    rep = DictionaryReport(source=source, raw_rows=len(rows), has_categories=has_categories)
    raw_terms = [t for t, _ in rows]
    rep.unique_raw = len(set(raw_terms))
    entries: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    invisible = set(DELETE_CHARS) | {ZWSP}
    for term, label in rows:
        norm = normalize_dictionary_term(term or "")
        if not norm:
            rep.empty_entries += 1
            continue
        if term != term.strip():
            rep.padded_entries += 1
        if any(c in term for c in invisible):
            rep.entries_with_invisible_chars += 1
        if " " in norm:
            rep.multiword_entries += 1
        elif not is_khmer_token(norm):
            rep.non_khmer_entries += 1
        counts[norm] += 1
        entries[norm]  # ensure key exists
        if has_categories and label and _clean_label(label):
            entries[norm].add(_clean_label(label))
    rep.unique_normalized = len(entries)
    rep.duplicate_entries = sum(c - 1 for c in counts.values())
    rep.conflicting_terms = sum(1 for cats in entries.values() if len(cats) > 1)
    if has_categories:
        label_counter: Counter[str] = Counter()
        for cats in entries.values():
            label_counter.update(cats)
        rep.category_counts = dict(label_counter.most_common())
    return StopwordDictionary(dict(entries), rep)


def load_dictionary(
    path: str | Path, term_col: str = "term", category_col: str | None = "category"
) -> StopwordDictionary:
    """Read a CSV (UTF-8, BOM tolerated). With ``category_col=None`` or a missing column it is a flat list."""
    p = Path(path)
    with p.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or term_col not in reader.fieldnames:
            raise ValueError(f"column {term_col!r} not found; columns are {reader.fieldnames}")
        use_cat = bool(category_col) and category_col in reader.fieldnames
        rows = [(r.get(term_col) or "", (r.get(category_col) if use_cat else None)) for r in reader]
    return parse_rows(rows, source=p.name, has_categories=use_cat)


def load_flat_list(path: str | Path) -> StopwordDictionary:
    """One term per line (UTF-8)."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    return parse_rows([(ln, None) for ln in lines], source=Path(path).name, has_categories=False)
