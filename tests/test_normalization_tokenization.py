import pytest

from khmer_nlp.normalization.text import (
    has_khmer,
    is_punctuation_token,
    khmer_ratio,
    normalize,
    normalize_dictionary_term,
)
from khmer_nlp.tokenization.tokenizers import (
    KhmerNltkTokenizer,
    SyllableTokenizer,
    WhitespaceTokenizer,
    duplicate_tokens,
    segmentation_agreement,
    tokenize,
)

SENT = "ខ្ញុំស្រលាញ់ប្រទេសកម្ពុជា"


def test_zwsp_becomes_space_or_is_deleted():
    assert normalize("ខ្ញុំ\N{ZERO WIDTH SPACE}ស្រលាញ់")[0] == "ខ្ញុំ ស្រលាញ់"
    assert normalize("ខ្ញុំ\N{ZERO WIDTH SPACE}ស្រលាញ់", zwsp="delete")[0] == "ខ្ញុំស្រលាញ់"
    with pytest.raises(ValueError):
        normalize("x", zwsp="bad")


def test_invisible_characters_are_removed_and_counted():
    raw = (
        "\N{ZERO WIDTH NO-BREAK SPACE}ក\N{ZERO WIDTH NON-JOINER}ខ\N{ZERO WIDTH JOINER}គ\N{WORD JOINER}ឃ\N{SOFT HYPHEN}"
    )
    out, rep = normalize(raw)
    assert out == "កខគឃ"
    assert rep.deleted == {"BOM": 1, "ZWNJ": 1, "ZWJ": 1, "WORD JOINER": 1, "SOFT HYPHEN": 1}


def test_whitespace_normalised_including_nbsp_and_repeats():
    out, rep = normalize("  ក\N{NO-BREAK SPACE}\N{NO-BREAK SPACE}ខ \t\n គ  ")
    assert out == "ក ខ គ"
    assert rep.space_like_replaced >= 3


def test_empty_none_and_whitespace_only():
    assert normalize("")[0] == ""
    assert normalize("   \n ")[0] == ""
    assert tokenize(None, WhitespaceTokenizer()).tokens == []
    assert tokenize("   ", WhitespaceTokenizer()).tokens == []


def test_nfc_applied():
    out, rep = normalize("é")  # decomposed e-acute
    assert out == "é" and rep.nfc_changed


def test_original_text_is_preserved():
    raw = "  ក\N{ZERO WIDTH SPACE}ខ  "
    p = tokenize(raw, WhitespaceTokenizer())
    assert p.original == raw and p.normalized == "ក ខ" and p.tokens == ["ក", "ខ"]


def test_punctuation_is_dropped_and_counted():
    p = tokenize("ក ។ ខ ៕ ! គ", WhitespaceTokenizer())
    assert p.tokens == ["ក", "ខ", "គ"] and p.dropped_punctuation == 3
    kept = tokenize("ក ។", WhitespaceTokenizer(), drop_punctuation=False).tokens
    assert kept == ["ក", "។"]
    assert is_punctuation_token("។។") and not is_punctuation_token("ក។")


def test_repetition_sign_and_riel_are_not_punctuation():
    assert not is_punctuation_token("ៗ") and not is_punctuation_token("៛")


def test_mixed_khmer_english():
    p = tokenize("ខ្ញុំរៀន Python នៅ ITC 2024", SyllableTokenizer())
    assert "python" in p.tokens and "itc" in p.tokens and "2024" in p.tokens  # Latin lower-cased, kept whole
    assert has_khmer("ខ្ញុំ abc") and not has_khmer("abc")
    assert 0.0 < khmer_ratio("ខ្ញុំ abc") < 1.0 and khmer_ratio("") == 0.0


def test_cluster_tokenizer_is_lossless():
    toks = SyllableTokenizer().segment(SENT)
    assert "".join(toks) == SENT and len(toks) > 4


def test_khmer_nltk_segments_into_words():
    tok = KhmerNltkTokenizer()
    toks = tok.segment(SENT)
    assert "".join(toks) == SENT and 2 <= len(toks) < len(SyllableTokenizer().segment(SENT))
    assert tok.version  # recorded for reproducibility


def test_tokenizers_disagree_and_agreement_is_measured():
    a = KhmerNltkTokenizer().segment(SENT)
    b = SyllableTokenizer().segment(SENT)
    assert a != b  # segmentation tools are not interchangeable
    assert segmentation_agreement(a, a) == 1.0
    assert 0.0 < segmentation_agreement(a, b) < 1.0


def test_duplicate_token_detection():
    d = duplicate_tokens(["ក", "ក", "ខ", "គ", "ខ"])
    assert d["repeated_types"] == {"ក": 2, "ខ": 2}
    assert d["consecutive_repeats"] == ["ក"]
    assert d["type_token_ratio"] == pytest.approx(3 / 5)
    assert duplicate_tokens([])["type_token_ratio"] == 0.0


def test_dictionary_term_normalisation_for_matching():
    assert normalize_dictionary_term(" នៃ\N{ZERO WIDTH SPACE} ") == "នៃ"
