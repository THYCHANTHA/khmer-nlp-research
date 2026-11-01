# Dataset card: Khmer Wikipedia lead sections (derived sample)

| Field | Value |
|---|---|
| Source | Khmer Wikipedia, `https://km.wikipedia.org`, MediaWiki API |
| What | lead section (plain text) of main-namespace, non-redirect articles; visible categories; page id, revision id and timestamp |
| Licence of the text | Creative Commons Attribution-ShareAlike 4.0 (as reported by the site's `rightsinfo`) |
| Licence of this repository's code | MIT — **does not** apply to the Wikipedia text |
| Redistribution | The text is **not** committed. `scripts/download_wikipedia.py` re-downloads it. Two metadata manifests (page id, title, URL, history URL, revision id, timestamp, lead length; and the analysis-sample manifest with inclusion status, exclusion reason and a SHA-256 of each lead text) are provided in `data/manifest/` |
| Attribution | Each article's contributors are listed in its page history; the manifest gives the history URL of every article |
| Modification notice | Text is normalised (Unicode NFC, invisible characters removed, whitespace collapsed), segmented and filtered; tokens are cached locally. Any adapted text redistributed from this sample must carry CC BY-SA 4.0, a link to the licence, a note that changes were made, and the same licence |
| Snapshot | article counts, the sample checksum and the hash of the raw file analysed are in `reports/experiments/corpus_summary.json`; its `download.retrieved_utc` is the end time of the last download invocation, not the start (no download duration was recorded) |
| Intended use | reproducible methodology experiments on Khmer text processing |
| Not suitable for | measuring real-world search quality; judging the linguistic correctness of a stop-word list |
| Known issues | stubs, translation artefacts, maintenance categories, topical imbalance (many administrative-unit articles) |
| Personal data | Wikipedia articles can mention living people; only the public article text and title are used, no user data is collected |

The Wikipedia text changes continuously. The analysis-sample manifest identifies exactly which of the downloaded articles were analysed (8,806 of 14,701) and why the others were excluded (5,660 lead shorter than 200 characters; 235 Khmer-script share below 50 %; 0 with fewer than 20 tokens). The selection is not a random sample: all listed non-redirect main-namespace articles are filtered in that order. Revision ids and per-article lead-text hashes allow a re-download to be compared with the analysed version; the raw file hash and the sample checksum are in `reports/experiments/corpus_summary.json`.
