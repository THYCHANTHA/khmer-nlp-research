# Dataset card: Khmer Wikipedia lead sections (derived sample)

| Field | Value |
|---|---|
| Source | Khmer Wikipedia, `https://km.wikipedia.org`, MediaWiki API |
| What | lead section (plain text) of main-namespace, non-redirect articles; visible categories; page id, revision id and timestamp |
| Licence of the text | Creative Commons Attribution-ShareAlike 4.0 (as reported by the site's `rightsinfo`) |
| Licence of this repository's code | MIT — **does not** apply to the Wikipedia text |
| Redistribution | The text is **not** committed. `scripts/download_wikipedia.py` re-downloads it. A metadata manifest (page id, title, URL, history URL, revision id, timestamp) is provided in `data/manifest/` |
| Attribution | Each article's contributors are listed in its page history; the manifest gives the history URL of every article |
| Modification notice | Text is normalised (Unicode NFC, invisible characters removed, whitespace collapsed), segmented and filtered; tokens are cached locally. Any adapted text redistributed from this sample must carry CC BY-SA 4.0, a link to the licence, a note that changes were made, and the same licence |
| Snapshot | retrieval time and article counts are written to `reports/experiments/corpus_summary.json` |
| Intended use | reproducible methodology experiments on Khmer text processing |
| Not suitable for | measuring real-world search quality; judging the linguistic correctness of a stop-word list |
| Known issues | stubs, translation artefacts, maintenance categories, topical imbalance (many administrative-unit articles) |
| Personal data | Wikipedia articles can mention living people; only the public article text and title are used, no user data is collected |

The Wikipedia text changes continuously: use the manifest's revision ids to retrieve the exact versions analysed.
