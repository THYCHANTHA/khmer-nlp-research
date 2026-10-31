# Validation record (2026-10-31)

Run from a fresh `git clone` of commit `c30b83c` into a new virtual environment created without system site-packages, with
`pip install -r requirements.txt` only (Python 3.12.5, Windows 11). Exit codes were read directly, not through a pipe.

| Check | Result |
|---|---|
| `pip install -r requirements.txt` | exit 0 |
| `pip check` | "No broken requirements found", exit 0 |
| `ruff check .` | "All checks passed!", exit 0 |
| `pytest` (no network, no downloaded data in the clone) | **69 passed in 8.13 s**, exit 0 |
| Reproduction: copy of the downloaded Wikipedia file, then `python scripts/run_experiments.py --retokenize` | exit 0, 268 s |
| Reproduced result tables vs committed tables (`git diff`) | **identical** (only `corpus_summary.json` differs, in the recorded library versions and line endings) |

Library versions: development environment scikit-learn 1.7.2, numpy 2.4.3, pandas 2.3.3, scipy 1.17.0; fresh environment
scikit-learn 1.9.1, numpy 2.5.4, pandas 3.1.0, scipy 1.18.1. Results did not change between them.

## Not verified

* GitHub Actions has never run for this repository (not pushed; account billing lock on the owner's other repositories).
* Python 3.11 was not run (the code targets 3.11+).
* The live download was not repeated in the fresh clone (Wikipedia content changes over time; revision ids are in `data/manifest`).
* The Streamlit app was checked with `streamlit.testing` (all 8 pages) and with headless-Edge screenshots at desktop width only; mobile width and other browsers were not checked.
* No Khmer linguist has reviewed any text or label in this repository.
