"""Stop-word removal strategies.

Strategy A  : no removal (baseline).
Strategy B  : tier-1 categories of an imported, categorised dictionary.
Strategy C  : tier-1 + tier-2 categories.
Frequency analogues (B-freq / C-freq): top-N document-frequency terms, a purely statistical, non-linguistic list that
needs no dictionary. They are NOT claimed to equal B / C.

Protection: terms whose category (after mapping) is protected, e.g. negation, are never removed, whatever the tier.
Every term gets a ``Decision`` row so that the decision table can be exported and reviewed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from khmer_nlp.stopwords.dictionary import StopwordDictionary

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "strategies.yaml"


@dataclass(frozen=True)
class Decision:
    word: str
    category: str
    tier: str  # "1", "2", "data-driven", "protected", "excluded", "unmapped"
    action: str  # "remove" or "keep"
    reason: str
    source: str
    manual_review: bool


@dataclass
class Strategy:
    name: str
    description: str
    decisions: dict[str, Decision] = field(default_factory=dict)

    @property
    def remove_set(self) -> frozenset[str]:
        return frozenset(w for w, d in self.decisions.items() if d.action == "remove")

    def apply(self, tokens: Iterable[str]) -> tuple[list[str], list[str]]:
        """Return (kept, removed) preserving order; the input is never mutated."""
        rm = self.remove_set
        kept, removed = [], []
        for t in tokens:
            (removed if t in rm else kept).append(t)
        return kept, removed

    def explain(self, word: str) -> Decision | None:
        return self.decisions.get(word)

    def decision_table(self) -> pd.DataFrame:
        cols = ["word", "category", "tier", "action", "reason", "source", "manual_review"]
        rows = [vars(d) for d in self.decisions.values()]
        return pd.DataFrame(rows, columns=cols)


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def map_label(label: str, config: dict) -> set[str]:
    """Map a dictionary label to generic category keys by alias substring (case-insensitive)."""
    low = label.lower()
    return {key for key, spec in config["categories"].items() if any(a in low for a in spec["aliases"])}


def no_removal() -> Strategy:
    return Strategy("A: none", "No stop-word removal (baseline).")


def build_tier_strategy(
    dictionary: StopwordDictionary, config: dict, tiers: Iterable[int], name: str | None = None
) -> Strategy:
    """Tier strategy from a categorised dictionary. Needs category labels and a label-to-key mapping."""
    if not dictionary.report.has_categories:
        raise ValueError("tier strategies need a dictionary with a category column")
    tiers = sorted(set(tiers))
    tier_of = {key: int(t) for t, keys in config["tiers"].items() for key in keys}
    protected = set(config.get("protected_categories", []))
    protected_terms = set(config.get("protected_terms", []))
    excluded = [x.lower() for x in config.get("excluded_label_substrings", [])]
    strat = Strategy(
        name or ("B: tier 1" if tiers == [1] else f"tiers {tiers}"),
        f"Remove dictionary terms whose mapped category is in tier(s) {tiers}; protected categories are kept.",
    )
    src = f"imported dictionary {dictionary.report.source}"
    for term, labels in dictionary.entries.items():
        if not labels:
            strat.decisions[term] = Decision(term, "", "unmapped", "keep", "no category label", src, True)
            continue
        label_txt = " | ".join(sorted(labels))
        keys: set[str] = set()
        for lab in labels:
            keys |= map_label(lab, config)
        conflict = len(labels) > 1
        if term in protected_terms or keys & protected:
            strat.decisions[term] = Decision(
                term, label_txt, "protected", "keep", "protected category/term (e.g. negation): kept", src, True
            )
        elif any(x in lab.lower() for lab in labels for x in excluded) or " " in term:
            why = (
                "multi-word entry cannot match a single token" if " " in term else "label marks a non-function category"
            )
            strat.decisions[term] = Decision(term, label_txt, "excluded", "keep", why, src, True)
        elif not keys:
            strat.decisions[term] = Decision(
                term, label_txt, "unmapped", "keep", "label not mapped to a tier", src, True
            )
        else:
            tier_vals = {tier_of[k] for k in keys if k in tier_of}
            if not tier_vals:
                strat.decisions[term] = Decision(
                    term, label_txt, "unmapped", "keep", "category not assigned to a tier", src, True
                )
            elif tier_vals <= set(tiers) and not conflict:
                strat.decisions[term] = Decision(
                    term, label_txt, str(min(tier_vals)), "remove", f"category in tier {min(tier_vals)}", src, False
                )
            elif tier_vals <= set(tiers):
                strat.decisions[term] = Decision(
                    term,
                    label_txt,
                    str(min(tier_vals)),
                    "remove",
                    "conflicting labels, all inside the selected tiers",
                    src,
                    True,
                )
            else:
                strat.decisions[term] = Decision(
                    term, label_txt, str(min(tier_vals)), "keep", "tier not selected for this strategy", src, conflict
                )
    return strat


def document_frequency(docs: Iterable[list[str]]) -> tuple[Counter[str], int]:
    df: Counter[str] = Counter()
    n = 0
    for toks in docs:
        n += 1
        df.update(set(toks))
    return df, n


def build_frequency_strategy(docs: Iterable[list[str]], top_n: int, name: str) -> Strategy:
    """Remove the ``top_n`` terms with the highest document frequency (ties broken alphabetically)."""
    df, n = document_frequency(docs)
    ranked = sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    strat = Strategy(name, f"Remove the {top_n} most document-frequent terms (statistical list, not linguistic).")
    for term, c in ranked:
        strat.decisions[term] = Decision(
            term,
            "",
            "data-driven",
            "remove",
            f"appears in {c}/{n} documents ({c / n:.1%})",
            "document frequency on the indexed collection",
            True,
        )
    return strat
