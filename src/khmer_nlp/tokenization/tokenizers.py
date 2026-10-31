"""Tokenizers behind one small interface, so that results can be compared across segmentation tools.

* ``KhmerNltkTokenizer``: CRF word segmenter from the ``khmer-nltk`` package (version is recorded).
* ``SyllableTokenizer``: independent rule-based baseline that splits into Khmer character clusters (a base
  consonant/vowel plus its coeng-subscripts, vowel signs and diacritics). It does not know words.
* ``WhitespaceTokenizer``: splits on spaces only (useful for pre-segmented text and for tests).
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from importlib import metadata
from typing import Protocol

from khmer_nlp.normalization.text import is_punctuation_token, normalize

_LATIN_WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+(?:[.,]\d+)*")
# Khmer character cluster: base (consonant, independent vowel) + (coeng + consonant)* + dependent signs.
_CLUSTER = re.compile(r"[ក-ឳ](?:្[ក-អ]|[ា-៑៓៝])*")
_KHMER_DIGITS = "០១២៣៤៥៦៧៨៩"


class Tokenizer(Protocol):
    name: str

    def segment(self, text: str) -> list[str]: ...


class WhitespaceTokenizer:
    name = "whitespace"

    def segment(self, text: str) -> list[str]:
        return text.split()


class SyllableTokenizer:
    name = "khmer-cluster"

    def segment(self, text: str) -> list[str]:
        out: list[str] = []
        for chunk in text.split():
            pos = 0
            while pos < len(chunk):
                ch = chunk[pos]
                if "ក" <= ch <= "៿":
                    m = _CLUSTER.match(chunk, pos)
                    if m and m.end() > pos:
                        out.append(m.group())
                        pos = m.end()
                        continue
                    out.append(ch)
                    pos += 1
                else:
                    m = _LATIN_WORD.match(chunk, pos)
                    if m:
                        out.append(m.group())
                        pos = m.end()
                    else:
                        out.append(ch)
                        pos += 1
        return out


class KhmerNltkTokenizer:
    name = "khmer-nltk"

    def __init__(self) -> None:
        logging.getLogger("khmer-nltk").setLevel(logging.WARNING)
        from khmernltk import word_tokenize  # imported lazily: loads a model file

        self._tok = word_tokenize
        self.version = metadata.version("khmer-nltk")

    def segment(self, text: str) -> list[str]:
        return [t for t in self._tok(text) if t.strip()]


def get_tokenizer(name: str) -> Tokenizer:
    if name == "khmer-nltk":
        return KhmerNltkTokenizer()
    if name == "khmer-cluster":
        return SyllableTokenizer()
    if name == "whitespace":
        return WhitespaceTokenizer()
    raise ValueError(f"unknown tokenizer {name!r}")


@dataclass
class ProcessedText:
    """Original text is always kept next to the processed forms."""

    original: str
    normalized: str
    tokens: list[str]
    dropped_punctuation: int
    dropped_empty: int


def tokenize(
    text: str | None,
    tokenizer: Tokenizer,
    *,
    drop_punctuation: bool = True,
    lowercase_latin: bool = True,
    zwsp: str = "space",
) -> ProcessedText:
    """normalise -> segment -> clean tokens. Empty / None input gives an empty token list."""
    original = text or ""
    norm, _ = normalize(original, zwsp=zwsp)
    if not norm:
        return ProcessedText(original, "", [], 0, 0)
    raw_tokens = tokenizer.segment(norm)
    tokens: list[str] = []
    dropped_p = dropped_e = 0
    for t in raw_tokens:
        t = t.strip()
        if not t:
            dropped_e += 1
        elif drop_punctuation and is_punctuation_token(t):
            dropped_p += 1
        else:
            tokens.append(t.lower() if lowercase_latin and t.isascii() else t)
    return ProcessedText(original, norm, tokens, dropped_p, dropped_e)


def duplicate_tokens(tokens: list[str]) -> dict[str, object]:
    """Repeated-token diagnostics: counts above one and immediate repetitions (``a a``)."""
    counts = Counter(tokens)
    consecutive = [a for a, b in zip(tokens, tokens[1:], strict=False) if a == b]
    return {
        "repeated_types": {t: c for t, c in counts.items() if c > 1},
        "consecutive_repeats": consecutive,
        "type_token_ratio": (len(counts) / len(tokens)) if tokens else 0.0,
    }


def segmentation_agreement(a: list[str], b: list[str]) -> float:
    """Boundary F1 between two segmentations of the same character sequence (1.0 = identical)."""

    def bounds(tokens: list[str]) -> set[int]:
        pos, out = 0, set()
        for t in tokens:
            pos += len(t)
            out.add(pos)
        return out

    ba, bb = bounds(a), bounds(b)
    if not ba and not bb:
        return 1.0
    inter = len(ba & bb)
    p = inter / len(bb) if bb else 0.0
    r = inter / len(ba) if ba else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0
