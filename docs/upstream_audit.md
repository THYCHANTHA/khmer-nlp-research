# Audit of the original team repository

Repository audited: <https://github.com/Nagamiji/Stop_word_Project> (public, owned by the Nagamiji account, ITC team).
Audit date: 2026-10-31. Method: read-only clone to a scratch folder outside this project; files were read and counted, none
were executed, nothing was modified, nothing was pushed. Nothing from that clone is copied into this repository.

## 1. Licence and ownership

| Item | Finding |
|---|---|
| Licence | **None.** The GitHub API reports `license: null` and the file list has no LICENSE. Public visibility does not grant reuse rights. |
| Consequence | This repository contains no upstream code, dictionary, corpus, notebook output or figure. It links to the upstream project. |
| Contributors in history | 5 author identities: Nagamiji (8 commits), THYCHANTHA / Thy Chantha (8 commits under two display names and two e-mail identities), sokuntheary21ifl-droid (4), sreyneathros (1). |
| Personal data in upstream history | Commit metadata of the public upstream contains personal e-mail addresses of several contributors, including a personal Gmail address of the portfolio owner. They are not reproduced here. Rewriting upstream history is not possible or appropriate from this project. |

## 2. What the repository contains (116 tracked files, about 550 MB of `data/`)

`README.md`, `check_lib.py`, 5 notebooks, `report/` (2 markdown files), `stopwords/` (2 CSV files), and a large `data/`
tree: one raw corpus (`news_text_file_150k.txt`, 73 MB, 150,000 lines, one commit, **source and licence not stated**), many
generated TF-IDF, IR, diagnostic, comparative-analysis and visualisation outputs. There is no `requirements.txt`, no tests,
no CITATION file.

## 3. Verified dictionary facts (counted from the files, not from the README)

| Measure | `FIle_Stopwords.csv` (master) | `khmer_stopwords_broad_classes.csv` |
|---|---|---|
| Lines in file (header included) | 771 | **511** |
| Data rows | 770 | **510** |
| Unique raw terms | 745 | 510 |
| Unique after NFC + invisible-character removal + trimming | 726 | 510 |
| Duplicate entries (rows beyond first, normalised) | 44 | 0 |
| Entries with leading/trailing whitespace | 189 | 0 |
| Empty entries | 0 | 0 |
| Entries containing zero-width characters | 0 | 0 |
| Entries not entirely Khmer script | 0 | 0 |
| Distinct category labels | 64 | 10 |

* **The "511-word dictionary" is the line count of the broad-class file including its header. It holds 510 entries.**
  The figure 511 must not be reported as a word count.
* The 510 broad-class entries are a strict subset of the master file (0 missing). The other 216 master terms
  (normalised) have no broad class.
* In the master file, 22 normalised terms carry two or three different category labels (for example one term is labelled both
  Conjunction and Preposition).
* The master file labels 132 rows "Phrase" and about 60 rows "Content word (...)", i.e. multi-word phrases and content words
  are inside the stop-word list. Many labels are compound or inconsistent (for example trailing spaces, "+ Verb").
* Notebook 03 builds its stop-word set from the master file by keeping every row whose label does not contain "content
  word". Phrase rows and the 22 conflicting terms therefore all count as stop words there.

## 4. Claims in the documentation compared with the recorded evidence

| Claim | Evidence in the repository | Assessment |
|---|---|---|
| README: removing stop words "improves the accuracy of TF-IDF weighting and search ranking" | Recorded known-item results: Original mean rank 1.00, Recall@10 1.00; "No_All_Stopwords" mean rank 1.20, Recall@10 0.98 | **Not supported.** Removal did not improve and slightly lowered the recorded known-item scores. |
| Known-item retrieval validates stop-word removal | The query is a document that is itself in the index (`evaluate_retrieval` in notebook 03 uses `corpus[i]` as the query and looks up the same index `i`) | **Optimistic by construction:** the unmodified text always matches itself, so rank 1 is expected. It cannot show retrieval quality. |
| `report/IR.md`: noise "mathematically proven in the IR Evaluation phase" | No proof exists; the recorded results show no improvement | **Overclaim.** (This file was committed under the portfolio owner's name; see `team_contributions.md`.) |
| `report/IR.md`: Particles have "Extremely High" frequency | Team export `category_impact.csv`: Particles are 2.5 % of dictionary-word tokens; Prepositions 6.8 %, Conjunctions 6.2 % | **Contradicted by the team's own numbers.** |
| README Tier 1 = Particles, Prepositions, Conjunctions; Tier 2 = Pronouns, Determiners | Team export `tiered_recommendations.csv`: Tier 1 = Prepositions, Conjunctions only; Particles, Determiners, Auxiliaries, Function Nouns = Tier 2; Pronouns, Numbers = Tier 3; Question & Negation, Politeness = Tier 4. Recorded IR improvement per category is between about -0.17 % and +0.08 %, with one value of -153.9 % for Pronouns | The tier schemes differ; the exported "IR improvement" numbers are tiny and sometimes negative, so tiers rest on frequency, not demonstrated retrieval benefit. The -153.9 % needs explanation before it is quoted. |
| README steps refer to `01_preprocessing.ipynb` | That notebook was replaced by `Text_preprocessing.ipynb` and `Text_Apply_StopWord.ipynb` (commit 31c2343) | README is stale. |
| Diagnostic report: 58,409 documents, ~30,000 total tokens | A 150,000-line corpus cannot have 58,409 documents with 30,000 tokens in total | Internally inconsistent; cannot be interpreted without the notebook that produced it. |
| Evaluation size | Notebook 03, basic cell: 50 random queries (`random.seed(42)`) drawn from the first 3,000 documents. Enhanced cell / exported table: about 58,000 documents, 10,000 TF-IDF features, 12 corpus variants; the query count is not printed in the table (Recall values of 0.98 are consistent with 50 queries, which is an inference) | Few queries (one miss changes a score by 0.02); the basic cell uses first-N documents, not a random sample. |

## 5. Reproducibility of the original

* Notebooks contain hard-coded `D:\` and `/content/` (Google Colab, `drive.mount`) paths.
* No pinned dependencies; `khmer-nltk` is imported as `khmernltk` and installed unpinned. The PyPI package has seven
  releases (1.0 to 1.6; latest 1.6, Apache-2.0 per PyPI metadata). **Whether results depend on the version is untested.**
* Only notebook 03 sets a seed; notebooks 02 and 04 set none.
* Notebook 04 has no stored outputs.

## 6. What this audit permits

1. An **independent extension** can be built and published: new code, new tests, new documentation, with the upstream
   work cited as the origin of the research questions.
2. The team's dictionary and corpus **cannot be redistributed** here without written permission. Code written in this
   project can read a dictionary from a local, git-ignored path that the owner already has access to, so the experiments
   can be run locally; any published results derived from the dictionary must wait for permission (decision recorded in
   `docs/NEXT_STEPS.md` / `BUILD_STATUS.md`).
3. For public, redistributable experiments a separate openly licensed corpus is needed (candidate: Khmer Wikipedia,
   reachable on 2026-10-31 with 12,934 articles per the site statistics; licence still to be read from the publisher's
   page).

## 7. Decisions taken after the audit (owner, 2026-10-31)

* Permission from the team is **not granted** (not requested yet). None of the team's dictionary, corpus, code, notebooks,
  tables or exported results is used, copied, run or stored in this project, not even in git-ignored folders.
* The project proceeds as an independent implementation; the importer accepts an external CSV dictionary so that the
  tier strategies can be run later if permission or another licensed dictionary becomes available.
* Public experiments use Khmer Wikipedia (CC BY-SA 4.0) and statistical stop lists.
* The figures in section 3 and 4 are kept here as audit facts (counts only; no dictionary contents are reproduced).
