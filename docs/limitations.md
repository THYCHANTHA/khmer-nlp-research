# Limitations

**Scope of the evidence.** The public experiments use Khmer Wikipedia lead sections. They do not use the original ITC
team's corpus or dictionary, so they neither confirm nor refute that team's results.

**Dictionary.** No categorised Khmer stop-word dictionary is bundled. The tier strategies B and C are implemented and tested
on synthetic fixtures only; they have **not** been run on research data. The public experiments compare frequency-based
stop lists, which are statistical, not linguistic, so RQ4 (which linguistic categories to remove) is not answered here.

**Retrieval labels.** Relevance is derived from Wikipedia categories. These are incomplete, hierarchical and sometimes
administrative, so "relevant" and "not labelled relevant" are both noisy. They are not human judgements and not ground truth.
Metrics are therefore exploratory (RQ3).

**Known-item retrieval** (query document in the index) is reported only as a sanity check; it cannot measure quality.

**Classification** uses a few large, structurally distinct Wikipedia categories, so the task is easy and cannot rank
the strategies.

**Corpus.** Lead sections only; short, encyclopaedic; many stubs are dropped by the 200-character filter; only part of the
articles have a topical category, so the query set favours well-categorised topics. The maintenance-category filter is a
heuristic and some maintenance categories remain.

**Tokenisation.** One word segmenter (khmer-nltk 1.6) is used for all results. Different releases of the package segment
differently (see `reports/tokenizer_versions.md`), and a rule-based cluster baseline disagrees with it; conclusions may not
transfer to other segmenters.

**Statistics.** One corpus snapshot, one set of hyper-parameters fixed in advance; bootstrap intervals reflect query sampling
only. Differences between strategies are small relative to the label noise.

**Normalisation.** No canonical reordering of Khmer sequences is attempted.

**Not verified.** GitHub Actions has never run for this repository (account billing lock, not yet pushed); the app was
checked on desktop width with headless Edge only; Python 3.11 was not run (3.12.5 only).
