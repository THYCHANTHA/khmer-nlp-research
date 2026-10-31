# Methodology

All configuration values are in `configs/experiment.yaml` and were fixed before any result was inspected. The random seed
is 42 everywhere (query sampling, cross-validation folds, bootstrap, random control, SVM).

## 1. Data

**Public experiments** use the lead section (the text before the first heading) of articles of the Khmer Wikipedia
(`km.wikipedia.org`), fetched with the MediaWiki API by `scripts/download_wikipedia.py`.

* Licence: the API reports "Creative Commons Attribution-Share Alike 4.0" for the site. Text is therefore CC BY-SA 4.0 and
  is **not** covered by this repository's MIT software licence. The raw text is not committed; the script re-downloads it.
* Recorded per article: page id, title, URL, **revision id and timestamp**, history URL, visible categories.
* Only non-redirect articles in the main namespace are listed. Only the lead section is fetched, because the full-text
  extract API serves one page per request.
* Filters (see `corpus_summary.json` for how many records each removed): lead shorter than 200 characters; Khmer-script
  share of non-space characters below 0.5; fewer than 20 tokens.

**Not used**: the original ITC team's dictionary, corpus, code and results (no licence, no permission). See
`docs/upstream_audit.md`.

## 2. Normalisation and tokenisation (RQ1)

Order: NFC → delete BOM/word-joiner/soft-hyphen/ZWNJ/ZWJ → zero-width space becomes a space (or is deleted) → exotic spaces
become spaces → whitespace collapsed → tokenizer → remove empty and punctuation-only tokens → lower-case Latin tokens.
Khmer punctuation (។ ៕ ៖ ៘ ៙ ៚) is dropped as tokens; the repetition sign ៗ and the riel sign ៛ are not treated as
punctuation. The raw string is kept in `ProcessedText.original`.

Tokenizers (`khmer_nlp.tokenization`): `khmer-nltk` (CRF word segmenter; version recorded), a rule-based **character-cluster**
baseline, and whitespace splitting. They disagree, measured as boundary-F1; every downstream number depends on this choice.
Version compatibility of `khmer-nltk` was checked with `scripts/check_tokenizer_versions.py` (see `reports/tokenizer_versions.md`).

## 3. Stop-word strategies (RQ2, RQ4)

| Strategy | Definition | Needs |
|---|---|---|
| A | no removal | — |
| B | remove dictionary terms in tier 1 categories (particle, preposition, conjunction) | categorised dictionary |
| C | tiers 1 + 2 (adds pronoun, determiner) | categorised dictionary |
| B-freq / C-freq | remove the 50 / 200 terms with the highest document frequency | none |
| control | 50 randomly chosen vocabulary terms | none |

The tier assignment follows the hypothesis stated in the original project's README (particles, prepositions and
conjunctions first; pronouns and determiners optional); here it is a **configurable assumption to be tested**
(`configs/strategies.yaml`), not a finding.

**Protection.** Dictionary labels that map to negation, question words or discourse markers are never removed. Entries whose
label marks a phrase or a content word, multi-word entries, unlabelled entries and entries whose category is not assigned to a
tier are kept. Terms with conflicting labels are removed only if all their labels are inside the selected tiers, and are
flagged for manual review. Every term gets a row in a decision table (word, category, tier, action, reason, source,
manual-review flag).

Frequency lists are learned from the documents that are *indexed* (retrieval) or from the *training fold* (classification).

## 4. TF-IDF (RQ2)

`tf` = raw count (sublinear `1 + ln tf` in experiments), `idf = ln((1+N)/(1+df)) + 1`, L2-normalised rows. The
implementation is checked against scikit-learn's `TfidfVectorizer` on a fixture. Reported per strategy: tokens, vocabulary,
token reduction, share of tokens in the 10 commonest terms, normalised entropy, hapax share, top-20/50 keywords by mean TF-IDF,
Jaccard overlap of keyword sets with strategy A, and rank–frequency curves. Changes in the top keywords are descriptive and
are **not** used as evidence of better retrieval or classification.

## 5. Retrieval evaluation (RQ3, exploratory)

Three evaluations are kept strictly apart:

1. **Known-item** (query document is in the index): sanity check only — the unmodified text matches itself.
2. **Category-derived proxy labels**: see below. Exploratory.
3. **Human-judged relevance**: not available; future work.

*Proxy labels.* Visible category names are cleaned: names containing Latin letters or a Khmer maintenance word (list in the
config) are dropped as wiki-maintenance categories. A category is "usable" if it has between 5 and 60 member articles in the
sample (smaller: no signal; larger: relates unrelated pages). A held-out query article's relevant set is the set of indexed
articles sharing at least one usable category. This is weak and noisy: categories are incomplete, hierarchical, and sometimes
administrative rather than topical.

*Protocol.* Query articles (300 requested, drawn from articles with at least one usable category) are removed from the index.
The query is the first 40 tokens of the held-out article. Queries with a near-duplicate in the index (cosine ≥ 0.9 under strategy
A) are dropped for all strategies, as are queries with no relevant document. Query set and query text are identical across
strategies; the strategy is applied to the query after truncation. Metrics: P@5, P@10, R@5, R@10, MRR, nDCG@10 (binary gain), mean
rank of relevant documents. Uncertainty: percentile bootstrap over queries (2000 resamples) and paired differences against A.
Stability: four additional query samples (seeds 1–4).

## 6. Classification (exploratory)

Task: predict which of the largest topical categories an article belongs to (articles with exactly one of the K largest
categories, K ≤ 12, size ≥ 40). TF-IDF + `LinearSVC(C=1)`; stratified 5-fold CV; stop list, vocabulary and IDF are learned on
the training fold. Reported: accuracy, macro-F1, fold-wise paired differences against A. Because the classes are structurally
distinct, the task is easy and near a ceiling.

## 7. Linguistic analysis (RQ4, RQ5)

For each strategy: share of tokens removed, most frequent removed words, contribution by dictionary category (when a
categorised dictionary is imported), protected words that remain, and *potential false removals* — removed terms with the
highest mutual information with the class label. Linguistic categories come only from an imported dictionary; this code never
assigns linguistic labels. High mutual information marks a candidate for manual review, not an error.

## 8. Reproduction

```
pip install -r requirements.txt
python scripts/download_wikipedia.py            # resumable; ~13k articles
python scripts/run_experiments.py --retokenize  # writes reports/experiments/*.csv|json
python scripts/render_figures.py                # static PNGs (needs Microsoft Edge)
streamlit run app/Home.py
pytest && ruff check .
```
The Wikipedia content changes over time: use the recorded revision ids in `data/manifest/` to retrieve the exact versions
analysed.
