"""Shared helpers for the Streamlit app: theme, cached resources, result loading, dictionary upload."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from khmer_nlp.data.corpus import Article, load_tokens  # noqa: E402
from khmer_nlp.retrieval.index import RetrievalIndex  # noqa: E402
from khmer_nlp.stopwords.dictionary import StopwordDictionary, load_dictionary  # noqa: E402
from khmer_nlp.stopwords.strategies import Strategy, build_frequency_strategy, no_removal  # noqa: E402
from khmer_nlp.tokenization.tokenizers import get_tokenizer  # noqa: E402

EXP = ROOT / "reports" / "experiments"
TOKENS = ROOT / "data" / "processed" / "km_wikipedia_tokens.json.gz"
DEMO_DICT = ROOT / "data" / "samples" / "synthetic_demo_dictionary.csv"

CSS = """
<style>
html, body, [class*="css"], .stMarkdown, .stTextArea textarea, .stDataFrame {
  font-family: "Khmer UI", "Noto Sans Khmer", "Leelawadee UI", "Segoe UI", system-ui, sans-serif;
}
h1, h2, h3 { color: #12294d; letter-spacing: -0.01em; }
.hero { background: linear-gradient(120deg, #eef5fc 0%, #ffffff 70%); border: 1px solid #d6e6f6;
        border-radius: 14px; padding: 1.2rem 1.5rem; animation: fadein .6s ease-out; }
.hero h1 { margin: 0 0 .3rem 0; font-size: 1.9rem; }
.note { border-left: 4px solid #2f80c9; background: #f4f9fe; padding: .6rem .9rem; border-radius: 6px; margin: .6rem 0; }
.warn { border-left: 4px solid #d9a441; background: #fffaf0; padding: .6rem .9rem; border-radius: 6px; margin: .6rem 0; }
.kh { font-size: 1.15rem; line-height: 1.9; }
.pill { display: inline-block; padding: 1px 9px; border-radius: 999px; background: #dcecfa; margin: 2px; }
.pill.rm { background: #fde8e8; text-decoration: line-through; }
@keyframes fadein { from { opacity: 0; transform: translateY(6px);} to { opacity: 1; transform: none;} }
</style>
"""


def setup(title: str, icon: str = "📘") -> None:
    st.set_page_config(page_title=f"{title} · Khmer NLP Lab", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("**Khmer NLP Lab**  \nStop-word research and text intelligence")
        st.caption(
            "Independent research implementation informed by an earlier collaborative ITC project. "
            "Not an authorised continuation of that project."
        )


def note(text: str, kind: str = "note") -> None:
    st.markdown(f'<div class="{kind}">{text}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def table(name: str) -> pd.DataFrame | None:
    p = EXP / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data(show_spinner=False)
def jsonfile(name: str) -> dict | None:
    p = EXP / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def need_results() -> bool:
    if table("tfidf_stats.csv") is None:
        note(
            "No experiment results found. Run <code>python scripts/download_wikipedia.py</code> and "
            "<code>python scripts/run_experiments.py</code> first (see the README).",
            "warn",
        )
        return True
    return False


@st.cache_resource(show_spinner="Loading tokenizer…")
def tokenizer(name: str):
    return get_tokenizer(name)


@st.cache_resource(show_spinner="Loading corpus…")
def corpus() -> list[Article] | None:
    return load_tokens(TOKENS)[0] if TOKENS.exists() else None


@st.cache_resource(show_spinner="Building statistical stop lists…")
def corpus_strategies() -> dict[str, Strategy]:
    arts = corpus()
    cfgs = {"B-freq (top 50)": 50, "C-freq (top 200)": 200}
    out = {"A: none": no_removal()}
    if arts:
        docs = [a.tokens for a in arts]
        for name, n in cfgs.items():
            out[name] = build_frequency_strategy(docs, n, name)
    return out


@st.cache_resource(show_spinner="Indexing the Wikipedia sample…")
def search_index(strategy_name: str) -> RetrievalIndex | None:
    arts = corpus()
    if not arts:
        return None
    strat = corpus_strategies()[strategy_name]
    rm = strat.remove_set
    return RetrievalIndex(
        [a.pageid for a in arts], [[t for t in a.tokens if t not in rm] for a in arts], sublinear=True
    )


def dictionary_sidebar() -> StopwordDictionary | None:
    """Optional dictionary: upload a CSV (kept in memory only) or load the synthetic demo."""
    with st.sidebar:
        st.markdown("### Dictionary (optional)")
        up = st.file_uploader("CSV with columns term, category", type="csv")
        use_demo = st.checkbox("Use the synthetic demo dictionary", value=st.query_params.get("demo") == "1")
        tcol = st.text_input("term column", "term")
        ccol = st.text_input("category column", "category")
    try:
        if up is not None:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "upload.csv"
                path.write_bytes(up.getvalue())
                d = load_dictionary(path, tcol, ccol or None)
            d.report.source = up.name
            return d
        if use_demo:
            return load_dictionary(DEMO_DICT)
    except (ValueError, UnicodeDecodeError) as exc:
        st.sidebar.error(f"Could not import the dictionary: {exc}")
    return None


def demo_banner(d: StopwordDictionary | None) -> None:
    if d is not None and d.report.source == DEMO_DICT.name:
        note(
            "<b>Synthetic software demonstration.</b> This 14-term dictionary is invented for the demo; it is not "
            "linguistic evidence and no reported result uses it.",
            "warn",
        )
