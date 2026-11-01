# Data

Nothing in this folder is third-party text, except where stated.

| Path | Content | Licence | Tracked in git |
|---|---|---|---|
| `raw/km_wikipedia_leads.jsonl` | Khmer Wikipedia lead sections, downloaded by `scripts/download_wikipedia.py` | CC BY-SA 4.0 (Wikipedia contributors) | **no** (git-ignored; reproducible) |
| `processed/km_wikipedia_tokens.json.gz` | tokenised, filtered, normalised adaptation of the above | CC BY-SA 4.0 (adapted) | **no** |
| `manifest/km_wikipedia_manifest.csv` | all 14,701 downloaded articles: page id, title, URL, history URL, revision id, timestamp, lead length | metadata (facts) | yes |
| `manifest/km_wikipedia_analysis_sample.csv` | the same 14,701 articles with `status` (included / excluded), `exclusion_reason`, token count, and a SHA-256 of each lead text (no text) | metadata (facts) | yes |
| `samples/synthetic_demo_dictionary.csv` | 14 invented demo entries for the app; not research data | MIT | yes |

## Licences are separate

* The **software** is MIT (see `/LICENSE`).
* **Wikipedia text** is CC BY-SA 4.0: <https://creativecommons.org/licenses/by-sa/4.0/>. If you redistribute the downloaded
  text or the tokenised adaptation, you must give attribution (the manifest lists each article and its history page, where
  contributors are credited), link the licence, state that changes were made (normalisation, segmentation, filtering) and
  share under the same licence.
* **Stop-word dictionaries** are not included. If you import one, check its licence and provenance first. The original ITC
  team's dictionary and corpus are *not* part of this project and must not be added without written permission from the team.

## Reproduce

First follow the install steps in the top-level README (`pip install -r requirements.txt` and `pip install -e . --no-deps`).

```
python scripts/download_wikipedia.py      # resumable; rate-limited (0.2 s pause per batch of 20 articles); about 44 MB of text
python scripts/run_experiments.py --retokenize
```
