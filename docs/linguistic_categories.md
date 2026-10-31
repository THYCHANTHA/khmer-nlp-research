# Linguistic categories: schema, not content

This repository ships **no** Khmer linguistic annotation. It defines the *schema* that an imported dictionary is mapped to,
so that categorised stop-word research can be run on a dictionary you are entitled to use.

## Generic categories (`configs/strategies.yaml`)

| Key | Matched label substrings (case-insensitive) | Tier in the default config |
|---|---|---|
| particle | particle | 1 |
| preposition | preposition, relational | 1 |
| conjunction | conjunction | 1 |
| pronoun | pronoun | 2 |
| determiner | determiner, quantifier | 2 |
| auxiliary, function_noun, number_time, politeness | auxiliary / aspect, function noun, number / time, politeness / honorific | not assigned (kept) |
| negation | negation | **protected** |
| question | question | **protected** |
| discourse_marker | discourse | **protected** |

Labels that contain "phrase" or "content word" are treated as non-function categories and are never removed automatically.

## Status of the tier hypothesis

Tier 1 = particles, prepositions, conjunctions; tier 2 = pronouns, determiners; protected = negations and discourse markers,
is the working hypothesis taken from the stated aims of the earlier ITC project. In this repository it is a **configuration
to be tested**, not a result. The audit of that project found that its own exported evidence did not demonstrate a retrieval
benefit for any category (see `docs/upstream_audit.md`).

## Decision table

Strategy objects export a table with: word, category, tier, action (remove/keep), reason, source (which dictionary), and a
`manual_review` flag. Entries need manual review when: labels conflict; a protected category matched; the entry is a phrase or
multi-word string; the label is not mapped; or the list is statistical (frequency-based).

## What would make the linguistic findings valid

1. A dictionary with a documented source and licence.
2. Per-entry category labels reviewed by a Khmer linguist (inter-annotator agreement if more than one).
3. Resolution of conflicting and compound labels.
4. Context-sensitive cases (a word that is a stop word in one sentence and a content word in another) annotated at token level.
These are future work; none has been done here.
