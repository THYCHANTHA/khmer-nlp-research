# Next steps (the 20 % that is documented, not built)

Ordered by how much they would strengthen the research.

1. **Permission and a validated dictionary.** Obtain written permission (and the corpus source) from the original ITC team, or
   build/obtain an openly licensed Khmer stop-word resource. Then run the tier strategies B and C
   (`scripts/run_experiments.py --dictionary …`) and resolve data-quality issues first: duplicates, conflicting labels,
   phrase and content-word entries. Record the permission scope in `docs/team_contributions.md`.
2. **Human-judged relevance.** Build a small query set with independent relevance judgements (two annotators, agreement
   reported), so that RQ3 can move from "exploratory, proxy labels" to an evaluation.
3. **Linguistic validation.** Review category labels with a Khmer linguist; annotate context-dependent words at token level.
4. **More tokenizers.** Compare further Khmer segmenters (for example `khmercut`) with the same protocol, and check how much
   the stop-word effect depends on segmentation.
5. **Unicode normalisation.** Add canonical reordering of Khmer coeng/vowel/diacritic sequences and report its effect.
6. **Context-aware removal.** Part-of-speech or window-based decisions instead of a fixed list.
7. **Models.** Khmer transformer baselines (for example XLM-R), sentence embeddings, neural and hybrid retrieval, semantic search.
8. **Larger and more diverse benchmarks.** Full-length Wikipedia articles, news and conversational text; multilingual benchmarks.
9. **CI.** Run `pytest` and `ruff` in GitHub Actions once the account billing issue is resolved (a workflow is provided but has
   never run).
10. **Packaging and deployment.** Docker image; a hosted demo (Hugging Face Space) after the licence position is settled.
