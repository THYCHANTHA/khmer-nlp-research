"""Page 5: search the Wikipedia sample and compare strategies."""

import pandas as pd
import streamlit as st

from common import corpus, corpus_strategies, note, search_index, setup, tokenizer
from khmer_nlp.tokenization.tokenizers import tokenize

setup("Information Retrieval", "🔎")
st.title("Information retrieval demo")
arts = corpus()
if not arts:
    note(
        "The Wikipedia token cache is missing. Run <code>scripts/download_wikipedia.py</code> and "
        "<code>scripts/run_experiments.py</code>.",
        "warn",
    )
    st.stop()
by_id = {a.pageid: a for a in arts}
note(
    f"Searching a sample of {len(arts):,} Khmer Wikipedia lead sections (CC BY-SA 4.0). Each result links to the "
    "source article and its history for attribution. This is a demonstration; retrieval quality is evaluated separately "
    "on the evaluation page."
)

query = st.text_input("Khmer query", "ព្រះវិហារអង្គរវត្ត")
strategies = list(corpus_strategies())
chosen = st.selectbox("Stop-word strategy", strategies)
k = st.slider("Results", 5, 20, 10)

qt_all = tokenize(query, tokenizer("khmer-nltk")).tokens
if not qt_all:
    st.info("Enter a query.")
    st.stop()


def run(name: str) -> list[tuple[int, float]]:
    s = corpus_strategies()[name]
    qt, _ = s.apply(qt_all)
    return search_index(name).search(qt, k)


res = run(chosen)
q_kept, q_removed = corpus_strategies()[chosen].apply(qt_all)
st.caption(f"Query tokens: {' | '.join(qt_all)} — kept {len(q_kept)}, removed {len(q_removed)}")
rows = [
    {
        "rank": i,
        "article": by_id[pid].title,
        "cosine": round(sc, 4),
        "link": by_id[pid].url,
        "categories": "; ".join(sorted(by_id[pid].categories)[:3]),
    }
    for i, (pid, sc) in enumerate(res, start=1)
]
st.dataframe(
    pd.DataFrame(rows),
    hide_index=True,
    width="stretch",
    column_config={"link": st.column_config.LinkColumn("Wikipedia", display_text="open")},
)
if res and res[0][1] == 0:
    st.warning("All similarities are zero: no query term occurs in the index after stop-word removal.")

st.subheader("Compare strategies")
rank_tab: dict[int, dict] = {}
for name in strategies:
    for i, (pid, _score) in enumerate(run(name), start=1):
        rank_tab.setdefault(pid, {"article": by_id[pid].title})[name] = i
cmp = pd.DataFrame(rank_tab.values()).fillna("–")
st.dataframe(cmp, hide_index=True, width="stretch")
st.caption("Cells show the rank in each strategy's top results; “–” means not in the top results.")
