"""Page 3: Strategy A / B / C side by side."""

import streamlit as st

from common import corpus_strategies, demo_banner, dictionary_sidebar, note, setup, tokenizer
from khmer_nlp.stopwords.strategies import build_tier_strategy, load_config
from khmer_nlp.tfidf.compute import token_reduction
from khmer_nlp.tokenization.tokenizers import tokenize

setup("Stop-word Comparison", "⚖️")
st.title("Stop-word strategies side by side")
dictionary = dictionary_sidebar()
demo_banner(dictionary)

SAMPLE = "ខ្ញុំមិនទៅសាលារៀននៅថ្ងៃនេះទេ ព្រោះភ្លៀងធ្លាក់ខ្លាំង និងផ្លូវជន់លិច។"
text = st.text_area("Khmer text", SAMPLE, height=90)
tokens = tokenize(text, tokenizer("khmer-nltk")).tokens

strategies = dict(corpus_strategies())
if dictionary is not None and dictionary.report.has_categories:
    cfg = load_config()
    strategies["B: tier 1 (your dictionary)"] = build_tier_strategy(dictionary, cfg, [1], "B: tier 1 (your dictionary)")
    strategies["C: tiers 1+2 (your dictionary)"] = build_tier_strategy(
        dictionary, cfg, [1, 2], "C: tiers 1+2 (your dictionary)"
    )
elif dictionary is None:
    note(
        "Tier strategies B and C need a categorised dictionary. None is bundled (no validated, redistributable "
        "Khmer dictionary was available). Upload your own CSV or tick the synthetic demo in the sidebar. "
        "The statistical analogues B-freq / C-freq are shown instead."
    )
if len(strategies) == 1:
    note(
        "The Wikipedia token cache is missing, so the statistical stop lists are unavailable. "
        "Run the download and experiment scripts.",
        "warn",
    )

cols = st.columns(len(strategies))
removed_all: dict[str, dict] = {}
for col, (name, strat) in zip(cols, strategies.items(), strict=True):
    kept, removed = strat.apply(tokens)
    removed_all[name] = {t: strat.decisions[t] for t in removed}
    with col:
        st.markdown(f"##### {name}")
        st.metric(
            "Tokens kept",
            f"{len(kept)} / {len(tokens)}",
            f"-{token_reduction(len(tokens), len(kept)):.1f}%" if removed else "no change",
            delta_color="off",
        )
        st.markdown("**Remaining**")
        st.markdown("".join(f'<span class="pill kh">{t}</span>' for t in kept) or "—", unsafe_allow_html=True)
        st.markdown("**Removed**")
        st.markdown("".join(f'<span class="pill rm kh">{t}</span>' for t in removed) or "—", unsafe_allow_html=True)

st.subheader("Why was a word removed (or kept)?")
c1, c2 = st.columns(2)
name = c1.selectbox("Strategy", list(strategies))
choices = sorted(set(tokens))
word = c2.selectbox("Word in the text", choices) if choices else None
if word:
    d = strategies[name].explain(word)
    if d is None:
        st.info(f"“{word}” is not in the {name} decision table: it is kept (this strategy has no rule for it).")
    else:
        st.table(
            {
                "word": d.word,
                "category": d.category or "—",
                "tier": d.tier,
                "action": d.action,
                "reason": d.reason,
                "source": d.source,
                "manual review needed": d.manual_review,
            }.items()
        )
note(
    "Negations are protected: a dictionary entry whose category maps to negation is never removed by tier strategies, "
    "because dropping it flips the meaning of a sentence (compare ទេ / មិន in the default sentence)."
)
