"""Page 1: Research overview."""

import streamlit as st

from common import need_results, note, setup, table

setup("Research Overview", "🔬")

st.markdown(
    '<div class="hero"><h1>Khmer NLP Lab — Stop-Word Research and Text Intelligence</h1>'
    "Linguistically informed stop-word removal, TF-IDF analysis and exploratory retrieval for Khmer text.</div>",
    unsafe_allow_html=True,
)
st.write("")
note(
    "<b>Positioning.</b> This is an <b>independent research implementation</b> informed by an earlier collaborative "
    "university project at the Institute of Technology of Cambodia (ITC): "
    '<a href="https://github.com/Nagamiji/Stop_word_Project">Nagamiji/Stop_word_Project</a>. '
    "It does not reuse that project's code, dictionary or corpus (the upstream repository carries no licence and "
    "permission has not been obtained). It is not an authorised continuation of that work."
)

c1, c2 = st.columns([3, 2])
with c1:
    st.subheader("Background")
    st.write(
        "Khmer is written without spaces between words, and its function words (particles, prepositions, "
        "conjunctions, negations …) are frequent. Whether removing them helps downstream tasks, and which linguistic "
        "categories are safe to remove, is an empirical question. Fixed stop-word lists also fail on context-dependent "
        "words. This project builds the tooling to test those questions reproducibly, and runs it on openly licensed text."
    )
    st.subheader("Research questions")
    st.markdown(
        "1. **RQ1** How does stop-word removal affect Khmer text preprocessing?\n"
        "2. **RQ2** How does removing different word groups affect the TF-IDF representation?\n"
        "3. **RQ3** *(exploratory)* Does removal change retrieval performance, measured with category-derived proxy labels?\n"
        "4. **RQ4** Which linguistic categories should be removed, kept or treated as context-dependent? "
        "*(needs a validated categorised dictionary — see below)*\n"
        "5. **RQ5** What are the limits of fixed stop-word lists?"
    )
with c2:
    st.subheader("Pipeline")
    st.markdown(
        "1. Normalisation and tokenisation\n2. Stop-word strategies A / B / C\n3. TF-IDF comparison\n"
        "4. Retrieval and classification experiments\n5. Linguistic analysis of removals"
    )
    st.subheader("Status of the evidence")
    st.markdown(
        "- ✅ Public experiments on Khmer Wikipedia lead sections (CC BY-SA 4.0)\n"
        "- ⚠️ Retrieval relevance = **category-derived proxy labels**, not human judgements\n"
        "- ❌ No validated, categorised Khmer stop-word dictionary is bundled; tier strategies B/C run on a dictionary you import\n"
        "- ❌ Human-judged relevance: future work"
    )

st.subheader("Main findings from the executed experiments")
if not need_results():
    stats = table("tfidf_stats.csv")
    summ = table("ir_proxy_summary.csv")
    paired = table("ir_proxy_paired_vs_A.csv")
    cls = table("classification_summary.csv")
    a = stats.iloc[0]
    st.markdown(
        f"**Sample.** {int(a.documents):,} Wikipedia lead sections, {int(a.tokens):,} tokens, {int(a.vocab):,} distinct terms."
    )
    for _, r in stats.iloc[1:].iterrows():
        st.markdown(
            f"- **{r.strategy}** removes {r.token_reduction_pct:.1f}% of tokens but the vocabulary changes by "
            f"{(r.vocab - a.vocab) / a.vocab:+.2%}; share of tokens in the 10 commonest terms: "
            f"{a.top10_token_share:.3f} → {r.top10_token_share:.3f}."
        )
    st.markdown(
        "**Exploratory retrieval (proxy labels, nDCG@10, difference vs. no removal, paired 95% bootstrap CI):**"
    )
    for _, r in paired[paired.metric == "nDCG@10"].iterrows():
        verdict = "CI excludes 0" if r.ci_excludes_zero else "CI includes 0"
        st.markdown(f"- {r.strategy}: {r.diff_vs_A:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}] ({verdict})")
    best = cls.sort_values("macro_f1_mean", ascending=False).iloc[0]
    st.markdown(
        f"**Classification.** Best macro-F1 {best.macro_f1_mean:.3f} ({best.strategy}); see the evaluation page for "
        "per-strategy values and why this task is close to a ceiling."
    )
    note(
        "These are descriptive results on one sample with weak labels. They do not show that stop-word removal "
        "improves real-world retrieval, and they say nothing about linguistically categorised dictionaries.",
        "warn",
    )
