"""Page 4: TF-IDF analysis."""

import streamlit as st

from common import need_results, note, setup, table
from khmer_nlp.viz import figures as fg

setup("TF-IDF Analysis", "📊")
st.title("TF-IDF analysis")
if need_results():
    st.stop()

stats, kw, top, rf, ov = (
    table(f"tfidf_{k}.csv") for k in ("stats", "keywords", "top_terms", "rank_frequency", "overlap")
)
strategies = list(stats.strategy)

st.subheader("Vocabulary and document statistics")
st.dataframe(
    stats.rename(columns={"top10_token_share": "top-10 share", "norm_entropy": "norm. entropy"}),
    hide_index=True,
    width="stretch",
)
c1, c2 = st.columns(2)
c1.plotly_chart(fg.token_reduction(stats), width="stretch")
c2.plotly_chart(fg.concentration(stats), width="stretch")
st.plotly_chart(fg.vocabulary(stats), width="stretch")
note(
    "<b>Reading.</b> Removing the most frequent terms removes many tokens but almost no vocabulary: the stop lists are short "
    "relative to the vocabulary. Concentration measures (share of the 10 commonest terms, entropy) change because the "
    "dominant terms disappear. That is a property of the distribution; it is <b>not</b> evidence of better classification "
    "or retrieval. The random-term control shows what a same-size removal of arbitrary terms does."
)

st.subheader("Top keywords")
k = st.radio("Show top", [20, 50], horizontal=True)
shown = st.multiselect("Strategies", strategies, default=strategies[:3])
if shown:
    st.plotly_chart(fg.top_keywords(kw, shown, k), width="stretch")
    wide = kw[kw.strategy.isin(shown) & (kw["rank"] <= k)].pivot(index="rank", columns="strategy", values="term")[shown]
    st.dataframe(wide, width="stretch")
st.markdown("**Overlap of the top-50 keyword set with Strategy A (Jaccard)**")
st.dataframe(ov, hide_index=True, width="stretch")

st.subheader("Term frequency")
st.plotly_chart(fg.rank_frequency(rf), width="stretch")
a, b = st.columns(2)
sa = a.selectbox("Before", strategies, index=0)
sb = b.selectbox("After", strategies, index=min(1, len(strategies) - 1))
st.plotly_chart(fg.before_after_terms(top, sa, sb, 15), width="stretch")
t = top[top.strategy == sb].head(30)
st.dataframe(t[["rank", "term", "tf", "df", "df_share"]], hide_index=True, width="stretch")
