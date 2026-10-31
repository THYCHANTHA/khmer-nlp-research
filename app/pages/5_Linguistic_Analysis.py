"""Page 6: linguistic analysis of removals."""

import pandas as pd
import streamlit as st

from common import corpus, demo_banner, dictionary_sidebar, jsonfile, need_results, note, setup, table
from khmer_nlp.linguistic_analysis.analysis import category_contribution, removal_profile, retained_protected
from khmer_nlp.stopwords.strategies import build_tier_strategy, load_config
from khmer_nlp.viz import figures as fg

setup("Linguistic Analysis", "🧭")
st.title("Linguistic analysis")
dictionary = dictionary_sidebar()
demo_banner(dictionary)

st.subheader("What the statistical stop lists remove")
prof = jsonfile("removal_profiles.json")
if prof is None:
    need_results()
else:
    name = st.selectbox("Strategy", list(prof))
    p = prof[name]
    m = st.columns(3)
    m[0].metric("Tokens removed", f"{p['removed']:,}", f"{p['removed_pct']:.1f}%", delta_color="off")
    m[1].metric("Types in the list", p["types_in_strategy"])
    m[2].metric("Types observed", p["types_removed_observed"])
    st.plotly_chart(fg.removed_words(p, f"Most frequently removed words — {name}"), width="stretch")
    key = name.split(":")[0]
    fr = table(f"possible_false_removals_{key}.csv")
    if fr is not None and len(fr):
        st.markdown(
            "**Potential false removals** — removed terms that are most informative about the Wikipedia category "
            "(mutual information). Candidates for manual review, *not* proof that removal is wrong."
        )
        st.dataframe(fr, hide_index=True, width="stretch")
    note(
        "These lists are <b>statistical</b>. They contain no linguistic categories, so they say nothing about which "
        "<i>linguistic</i> groups should be removed (RQ4)."
    )

st.subheader("Linguistic categories (needs an imported dictionary)")
if dictionary is None or not dictionary.report.has_categories:
    note(
        "No categorised dictionary is loaded. Upload a CSV or tick the synthetic demo to see category distribution, "
        "removal tiers, retained negations and dictionary quality."
    )
    st.stop()

cfg = load_config()
arts = corpus()
st.plotly_chart(fg.dictionary_quality(dictionary.report.as_dict()), width="stretch")
st.plotly_chart(fg.category_distribution(dictionary.report.category_counts), width="stretch")
if not arts:
    st.warning("Token cache missing: corpus-based frequency analysis unavailable.")
    st.stop()
docs = [a.tokens for a in arts]
b = build_tier_strategy(dictionary, cfg, [1], "B: tier 1")
c = build_tier_strategy(dictionary, cfg, [1, 2], "C: tiers 1+2")
rows = [
    {
        "strategy": s.name,
        "tokens removed %": removal_profile(s, docs)["removed_pct"],
        "types removed": len(s.remove_set),
    }
    for s in (b, c)
]
st.markdown("**Removal tiers on the Wikipedia sample**")
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.markdown("**Contribution by dictionary category (strategy C)**")
st.dataframe(category_contribution(c, docs), hide_index=True, width="stretch")
kept = retained_protected(c, docs)
st.markdown("**Protected words that stay in the text**")
if len(kept):
    st.plotly_chart(fg.retained(kept), width="stretch")
else:
    st.write("No protected dictionary words occur in the sample.")
st.markdown("**Decision table (every dictionary entry)**")
st.dataframe(c.decision_table(), hide_index=True, width="stretch")
