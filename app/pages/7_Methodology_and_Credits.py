"""Page 8: methodology, data sources, dictionary information, team credits, references, limitations."""

import streamlit as st

from common import jsonfile, note, setup

setup("Methodology and Credits", "📚")
st.title("Methodology and team credits")

corpus = jsonfile("corpus_summary.json")

st.header("Methodology")
st.markdown(
    "1. **Normalisation**: Unicode NFC; invisible characters (BOM, word joiner, soft hyphen, ZWNJ, ZWJ) deleted; zero-width space "
    "turned into a space; whitespace collapsed. Raw text is kept.\n"
    "2. **Tokenisation**: `khmer-nltk` word segmentation (CRF); a character-cluster baseline and whitespace splitting are used for comparison.\n"
    "3. **Strategies**: A none; B / C tier strategies from an imported categorised dictionary (negations protected); B-freq / C-freq "
    "document-frequency lists; a same-size random control.\n"
    "4. **TF-IDF**: smoothed IDF, sublinear TF, L2 normalisation, verified against scikit-learn.\n"
    "5. **Retrieval**: held-out queries, cosine similarity, proxy relevance from Wikipedia categories, near-duplicate filter, bootstrap CIs.\n"
    "6. **Classification**: TF-IDF + Linear SVC, stratified 5-fold CV, preprocessing learned in-fold.\n"
    "Full details: `docs/methodology.md`."
)

st.header("Data sources")
if corpus:
    d = corpus.get("download", {})
    st.markdown(
        f"- **Khmer Wikipedia** lead sections via the MediaWiki API, downloaded on {d.get('retrieved_utc', '?')[:10]}. Licence reported by the "
        f"site: {d.get('rights', {}).get('text', '?')} ({d.get('rights', {}).get('url', '')}). "
        f"After filtering: {corpus['articles']:,} articles, {corpus['tokens']:,} tokens; all revision ids recorded; the per-article inclusion status is in `data/manifest/km_wikipedia_analysis_sample.csv`.\n"
        "- Text is **not redistributed** in this repository; scripts re-download it. Attribution is through each article's page "
        "and history link. Wikipedia text is licensed separately (CC BY-SA 4.0) from this project's software licence (MIT)."
    )
    st.markdown(
        f"- **Tokenizer**: khmer-nltk {corpus.get('tokenizer_version')} · Python {corpus.get('python')} · "
        f"{', '.join(f'{k} {v}' for k, v in corpus['versions'].items())}"
    )

st.header("Stop-word dictionary information")
note(
    "<b>No stop-word dictionary is bundled or used in the public experiments.</b> The original ITC project's dictionaries are "
    "not redistributed here because reuse permission has not been established. The importer accepts any CSV."
)
st.markdown(
    "Any imported dictionary needs linguistic validation by a Khmer linguist before it is treated as a stop-word resource. "
    "Until a validated, categorised dictionary is available, RQ4 (which linguistic categories to remove) remains unanswered."
)

st.header("Team and contribution information")
st.markdown(
    "The original research was carried out collaboratively at the Institute of Technology of Cambodia. Accounts visible in the "
    "public commit history of the upstream repository: **Nagamiji**, **THYCHANTHA**, **sokuntheary21ifl-droid**, **sreyneathros**. "
    "Roles and individual contributions are **pending confirmation** by the team; none is assigned here."
)
st.markdown(
    "**This repository's author** (Thy Chantha): student team member and technical contributor to the original project (eight upstream "
    "commits touching preprocessing notebooks, notebooks 03 and 04, the README and `report/IR.md`; this does not establish sole "
    "authorship or a leadership role). Everything in *this* repository is newly written."
)

st.header("References (DOIs checked against Crossref)")
st.markdown(
    "- Salton, G., Wong, A., Yang, C. S. (1975). A vector space model for automatic indexing. *Communications of the ACM*. doi:10.1145/361219.361220\n"
    "- Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation*. doi:10.1108/eb026526\n"
    "- Salton, G., Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*. doi:10.1016/0306-4573(88)90021-0\n"
    "- Järvelin, K., Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM TOIS*. doi:10.1145/582415.582418\n"
    "- Joachims, T. (1998). Text categorization with support vector machines. *ECML*. doi:10.1007/bfb0026683\n"
    "- Efron, B. (1979). Bootstrap methods: another look at the jackknife. *Annals of Statistics*. doi:10.1214/aos/1176344552\n"
    "- khmer-nltk (Python package, Apache-2.0 per PyPI): https://github.com/VietHoang1710/khmer-nltk\n"
    "- Wikipedia text: Creative Commons Attribution-ShareAlike 4.0, https://creativecommons.org/licenses/by-sa/4.0/"
)

st.header("Limitations")
st.markdown(
    "See `docs/limitations.md` and the evaluation page. In short: proxy labels, one tokenizer, no validated linguistic dictionary, "
    "Wikipedia lead sections only."
)
