"""Page 7: actual experimental results and their limits."""

import streamlit as st

from common import jsonfile, need_results, note, setup, table
from khmer_nlp.viz import figures as fg

setup("Research Evaluation", "🧪")
st.title("Research evaluation")
if need_results():
    st.stop()

summ, paired, seeds, known = (
    table(n)
    for n in ("ir_proxy_summary.csv", "ir_proxy_paired_vs_A.csv", "ir_proxy_by_seed.csv", "known_item_sanity_check.csv")
)
meta = jsonfile("ir_proxy_meta.json")
cls_sum, cls_diff, cls_fold = (table(f"classification_{n}.csv") for n in ("summary", "diffs", "folds"))
cls_info = jsonfile("classification_info.json")
stats = table("tfidf_stats.csv")

SECTIONS = [
    "1 · Known-item sanity check",
    "2 · Proxy retrieval (exploratory)",
    "3 · Classification",
    "4 · Token reduction and TF-IDF",
    "5 · Limitations",
]
tab1, tab2, tab3, tab4, tab5 = (st.container() for _ in SECTIONS)
for box, title in zip((tab1, tab2, tab3, tab4, tab5), SECTIONS, strict=True):
    box.header(title)

with tab1:
    st.subheader("Known-item retrieval — a demonstration only")
    note(
        "Here the query is a document that <b>is in the index</b>. The unmodified text always matches itself best, so the "
        "target is ranked first almost by construction. The result cannot show retrieval quality, and it cannot show that "
        "stop-word removal helps or hurts: all strategies score about the same, as expected.",
        "warn",
    )
    st.dataframe(known, hide_index=True, width="stretch")

with tab2:
    st.subheader("Held-out queries with category-derived proxy labels")
    note(
        "<b>Not human judgements, not ground truth.</b> A document is called “relevant” to a query if it shares at least "
        "one specific Wikipedia category with the query article. Categories are incomplete and noisy; many truly relevant "
        "articles are unlabelled, and some pairs sharing a category are not relevant to a searcher. Treat the numbers as "
        "exploratory."
    )
    c = st.columns(5)
    c[0].metric("Queries evaluated", meta["queries_evaluated"])
    c[1].metric("Indexed documents", f"{meta['indexed_documents']:,}")
    c[2].metric("Usable categories", meta["usable_categories"])
    c[3].metric("Median relevant docs / query", f"{meta['median_relevant_per_query']:.0f}")
    c[4].metric("Dropped near-duplicates", meta["dropped_near_duplicate"])
    st.plotly_chart(fg.ir_metrics(summ), width="stretch")
    st.dataframe(
        summ[["strategy", "P@5", "P@10", "R@5", "R@10", "MRR", "nDCG@10", "MeanRelRank"]].round(4),
        hide_index=True,
        width="stretch",
    )
    st.plotly_chart(fg.ir_paired(paired), width="stretch")
    st.dataframe(paired.round(4), hide_index=True, width="stretch")
    st.markdown("**Stability across query samples (mean metric per seed)**")
    st.dataframe(seeds.round(4), hide_index=True, width="stretch")
    note(
        "Queries are held out of the index; IDF and the frequency-based stop lists are learned from the indexed documents "
        "only; the query set and the query text are identical for every strategy; near-duplicates of the query in the index "
        "are removed. The random-term control shows the size of change expected from removing arbitrary terms."
    )

with tab3:
    st.subheader("TF-IDF + Linear SVC: predicting a Wikipedia category")
    st.write(
        f"{cls_info['documents_used']} articles in {len(cls_info['classes'])} classes (articles with exactly one of the largest "
        f"categories); {cls_info['folds']}-fold stratified CV; majority-class accuracy {cls_info['majority_class_accuracy']:.3f}. "
        "Stop lists and IDF are learned inside each training fold."
    )
    st.plotly_chart(fg.classification(cls_sum), width="stretch")
    st.dataframe(cls_sum.round(4), hide_index=True, width="stretch")
    st.dataframe(cls_diff.round(4), hide_index=True, width="stretch")
    st.json(cls_info["classes"], expanded=False)
    note(
        "The classes are large, structurally distinct Wikipedia categories (for example administrative units), so the task "
        "is easy and scores sit near a ceiling; small differences are within fold-to-fold variation. This experiment does "
        "not rank the strategies."
    )

with tab4:
    st.dataframe(stats.round(4), hide_index=True, width="stretch")
    st.plotly_chart(fg.token_reduction(stats), width="stretch")

with tab5:
    st.subheader("Dataset limitations")
    st.markdown(
        "- Wikipedia **lead sections** only (short, encyclopaedic, many stubs); not news, social media or conversational text.\n"
        "- Only articles with at least 200 characters of mostly Khmer text are used; many Khmer articles are short stubs or translations.\n"
        "- Category names are filtered with a heuristic (Latin-script and a few Khmer maintenance words); some maintenance categories remain.\n"
        "- Only part of the articles carry a topical category, which biases the query set towards well-categorised topics."
    )
    st.subheader("Evaluation limitations")
    st.markdown(
        "- Relevance is a **proxy**; human-judged relevance is future work.\n"
        "- One tokenizer (khmer-nltk 1.6) for all runs; results may differ with other segmenters.\n"
        "- The public stop lists are **frequency-based**, not linguistic; nothing here tests linguistically categorised tiers.\n"
        "- Single TF-IDF/cosine retrieval model; no tuning of weighting, no neural models.\n"
        "- Bootstrap intervals describe query-sampling variation only, not label noise or corpus choice."
    )
