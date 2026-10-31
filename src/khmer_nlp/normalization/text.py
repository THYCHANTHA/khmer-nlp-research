"""Khmer text normalisation. Every step is reversible in the sense that the raw text is never overwritten:
functions return new strings and a report of what changed.

Documented configuration (see docs/methodology.md):
1. Unicode NFC.
2. Invisible characters: BOM, word joiner, soft hyphen, ZWNJ, ZWJ are deleted; ZERO WIDTH SPACE (U+200B) is the
   Khmer word separator in some texts, so it is turned into an ordinary space (it can be deleted instead).
   NO-BREAK and other exotic spaces become ordinary spaces.
3. Whitespace runs collapse to one space; text is stripped.
Khmer coeng/vowel re-ordering is NOT attempted (listed in docs/NEXT_STEPS.md).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

DELETE_CHARS = {
    "\N{ZERO WIDTH NO-BREAK SPACE}": "BOM",
    "\N{WORD JOINER}": "WORD JOINER",
    "\N{SOFT HYPHEN}": "SOFT HYPHEN",
    "\N{ZERO WIDTH NON-JOINER}": "ZWNJ",
    "\N{ZERO WIDTH JOINER}": "ZWJ",
}
ZWSP = "\N{ZERO WIDTH SPACE}"
SPACE_LIKE = {
    "\N{NO-BREAK SPACE}",
    "\N{EN SPACE}",
    "\N{EM SPACE}",
    "\N{THIN SPACE}",
    "\N{NARROW NO-BREAK SPACE}",
    "\N{IDEOGRAPHIC SPACE}",
    "\N{LINE SEPARATOR}",
    "\N{PARAGRAPH SEPARATOR}",
    "\t",
    "\r",
    "\n",
}

KHMER_PUNCTUATION = "។៕៖ៗ៘៙៚៛"
LATIN_PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~«»“”‘’„…–—‹›"
PUNCTUATION = set(KHMER_PUNCTUATION) - {"ៗ", "៛"} | set(LATIN_PUNCTUATION)
# U+17D7 (repetition sign) is part of a word, U+17DB (riel) is a currency symbol: neither is treated as punctuation.

_KHMER_RANGE = re.compile(r"[ក-៿᧠-᧿]")
_WS = re.compile(r"\s+")


@dataclass
class NormalizationReport:
    deleted: dict[str, int] = field(default_factory=dict)
    zwsp_replaced: int = 0
    space_like_replaced: int = 0
    whitespace_runs_collapsed: int = 0
    nfc_changed: bool = False

    @property
    def total_changes(self) -> int:
        return (
            sum(self.deleted.values())
            + self.zwsp_replaced
            + self.space_like_replaced
            + self.whitespace_runs_collapsed
            + int(self.nfc_changed)
        )


def normalize(text: str, zwsp: str = "space") -> tuple[str, NormalizationReport]:
    """Return (normalised text, report). ``zwsp`` is ``"space"`` (default) or ``"delete"``."""
    if zwsp not in {"space", "delete"}:
        raise ValueError("zwsp must be 'space' or 'delete'")
    report = NormalizationReport()
    if not text:
        return "", report
    nfc = unicodedata.normalize("NFC", text)
    report.nfc_changed = nfc != text
    out: list[str] = []
    for ch in nfc:
        if ch in DELETE_CHARS:
            report.deleted[DELETE_CHARS[ch]] = report.deleted.get(DELETE_CHARS[ch], 0) + 1
        elif ch == ZWSP:
            report.zwsp_replaced += 1
            if zwsp == "space":
                out.append(" ")
        elif ch in SPACE_LIKE:
            report.space_like_replaced += 1
            out.append(" ")
        else:
            out.append(ch)
    joined = "".join(out)
    report.whitespace_runs_collapsed = len(re.findall(r"\s{2,}", joined))
    return _WS.sub(" ", joined).strip(), report


def normalize_text(text: str, zwsp: str = "space") -> str:
    return normalize(text, zwsp)[0]


def has_khmer(text: str) -> bool:
    return bool(_KHMER_RANGE.search(text))


def khmer_ratio(text: str) -> float:
    """Share of non-space characters that are Khmer script."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if _KHMER_RANGE.match(c)) / len(chars)


def is_punctuation_token(token: str) -> bool:
    return bool(token) and all(c in PUNCTUATION or unicodedata.category(c).startswith("P") for c in token)


def is_khmer_token(token: str) -> bool:
    return bool(token) and all(_KHMER_RANGE.match(c) for c in token)


def normalize_dictionary_term(term: str) -> str:
    """Normalisation used for stop-word matching: same as text, but ZWSP is deleted and the result is trimmed."""
    return normalize(term, zwsp="delete")[0]
