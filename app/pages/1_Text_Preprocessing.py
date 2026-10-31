"""Page 2: Khmer text preprocessing."""

import pandas as pd
import streamlit as st

from common import note, setup, tokenizer
from khmer_nlp.normalization.text import khmer_ratio, normalize
from khmer_nlp.tokenization.tokenizers import duplicate_tokens, segmentation_agreement, tokenize

setup("Text Preprocessing", "🔤")
st.title("Khmer text preprocessing")
st.caption("Normalise → segment → clean. The original text is never overwritten.")

SAMPLE = "ប្រទេសកម្ពុជាមានខេត្តចំនួន២៥ ហើយរាជធានីភ្នំពេញជាទីក្រុងធំជាងគេ។  Phnom Penh   is the capital."
text = st.text_area("Enter Khmer (or mixed Khmer / English) text", SAMPLE, height=110)
c1, c2, c3 = st.columns(3)
tok_name = c1.selectbox("Tokenizer", ["khmer-nltk", "khmer-cluster", "whitespace"])
zw = c2.selectbox("Zero-width space", ["space", "delete"], help="U+200B is a word separator in some Khmer text")
drop_p = c3.checkbox("Drop punctuation tokens", True)

norm, rep = normalize(text, zwsp=zw)
proc = tokenize(text, tokenizer(tok_name), drop_punctuation=drop_p, zwsp=zw)

st.subheader("Original and normalised text")
a, b = st.columns(2)
a.markdown("**Original**")
a.markdown(f'<div class="kh">{text or "(empty)"}</div>', unsafe_allow_html=True)
b.markdown("**Normalised**")
b.markdown(f'<div class="kh">{norm or "(empty)"}</div>', unsafe_allow_html=True)

m = st.columns(5)
m[0].metric("Characters (orig → norm)", f"{len(text)} → {len(norm)}")
m[1].metric("Khmer script share", f"{khmer_ratio(norm):.0%}")
m[2].metric("Tokens", len(proc.tokens))
m[3].metric("Punctuation dropped", proc.dropped_punctuation)
m[4].metric("Normalisation changes", rep.total_changes)
with st.expander("Normalisation report"):
    st.json(
        {
            "deleted_invisible": rep.deleted,
            "zwsp_replaced": rep.zwsp_replaced,
            "space_like_replaced": rep.space_like_replaced,
            "whitespace_runs_collapsed": rep.whitespace_runs_collapsed,
            "nfc_changed": rep.nfc_changed,
        }
    )

st.subheader("Tokens")
st.markdown("".join(f'<span class="pill kh">{t}</span>' for t in proc.tokens) or "(no tokens)", unsafe_allow_html=True)
dup = duplicate_tokens(proc.tokens)
st.caption(
    f"Type/token ratio {dup['type_token_ratio']:.2f}; repeated types: {len(dup['repeated_types'])}; "
    f"immediate repeats: {len(dup['consecutive_repeats'])}"
)

st.subheader("Tokenizer comparison")
rows, segs = [], {}
for n in ("khmer-nltk", "khmer-cluster", "whitespace"):
    p = tokenize(text, tokenizer(n), drop_punctuation=drop_p, zwsp=zw)
    segs[n] = p.tokens
    rows.append({"tokenizer": n, "tokens": len(p.tokens), "distinct": len(set(p.tokens))})
df = pd.DataFrame(rows)
df["boundary F1 vs khmer-nltk"] = [segmentation_agreement(segs["khmer-nltk"], segs[n]) for n in df.tokenizer]
st.dataframe(df, hide_index=True, width="stretch")
note(
    "Segmentation tools do not agree: the choice of tokenizer changes every downstream count. "
    "<code>khmer-cluster</code> is a rule-based baseline that splits into character clusters, not words."
)
