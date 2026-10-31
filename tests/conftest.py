"""Synthetic fixtures. They exercise the SOFTWARE; they are not linguistic evidence and are not a stop-word resource.

The 'dictionary' below is invented for tests: terms and category labels are arbitrary test inputs.
"""

import sys
from pathlib import Path

import pytest

from khmer_nlp.stopwords.dictionary import parse_rows
from khmer_nlp.stopwords.strategies import load_config

# `streamlit run` puts the app folder on sys.path; AppTest does not.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

# term, label  (labels deliberately mimic the shapes a real import may have: padded, compound, conflicting)
FIXTURE_ROWS = [
    ("នៃ", "Prepositions / Relational Words"),
    ("និង", "Conjunctions"),
    ("នៃ ", "Prepositions / Relational Words"),  # padded duplicate
    ("ដ៏", "Particles"),
    ("ខ្ញុំ", "Pronouns"),
    ("នេះ", "Determiners & Quantifiers"),
    ("មិន", "Question & Negation Words"),  # protected (negation)
    ("ដែល", "Conjunctions"),
    ("ដែល", "Prepositions / Relational Words"),  # conflicting labels, same tier
    ("ទេ", "Question & Negation Words"),
    ("ទៅដល់ទីនោះ", "Phrase"),  # label excluded
    ("ធ្វើតាម", "Content word (verb)"),  # label excluded
    ("ក\N{ZERO WIDTH SPACE}ា", "Particles"),  # contains ZWSP
    ("", "Particles"),  # empty
    ("the", "Particles"),  # non-Khmer
    ("ពី ទៅ", "Prepositions / Relational Words"),  # multi-word
    ("ថា", ""),  # no label
]


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture()
def fixture_dict():
    return parse_rows(FIXTURE_ROWS, source="fixture.csv", has_categories=True)


@pytest.fixture()
def docs():
    return [
        ["សិស្ស", "នៃ", "សាលា", "និង", "គ្រូ"],
        ["គ្រូ", "មិន", "ខ្ញុំ", "សាលា", "សាលា"],
        ["កសិកម្ម", "នៃ", "ស្រូវ", "និង", "ដី"],
        ["ស្រូវ", "ដី", "នេះ", "កសិកម្ម", "ទេ"],
        [],
    ]
