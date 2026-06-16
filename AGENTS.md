# AGENTS.md

This file is the operating guide for coding agents working on MedColBERT.
Follow it before making any code, data, config, or documentation change.

## Project Purpose

MedColBERT is a biomedical and clinical late-interaction retrieval project.
The target model is a ColBERT-style retriever initialized from
BioClinical-ModernBERT and trained with real-corpus, ontology-controlled
synthetic supervision.

The project aims to show that ontology-controlled synthetic data improves
retrieval under medical vocabulary shift, such as:

- layperson wording to clinical terminology
- abbreviation to expanded term
- brand drug to generic or ingredient
- symptom phrase to diagnosis or treatment evidence
- biomedical literature phrasing to clinical phrasing
- patient summary to trial or case-report evidence

## Current Repo State

The repository is currently scaffolded. Most scripts are placeholders. Treat
planning docs and configs as design intent, but implement reusable logic under
`src/medcolbert/` and keep scripts thin.

Important files:

- [GOALS.md](GOALS.md): phase index and go/no-go gates.
- [docs/goals/](docs/goals): detailed phase goals.
- [.claude/custom/INITIAL_PLAN.md](.claude/custom/INITIAL_PLAN.md): earlier master research plan.
- [.claude/custom/EXECUTION_ROADMAP.md](.claude/custom/EXECUTION_ROADMAP.md): earlier execution roadmap.
- [configs/data.yaml](configs/data.yaml): data and quality-gate defaults.
- [configs/generation.yaml](configs/generation.yaml): synthetic generation defaults.
- [configs/train_base.yaml](configs/train_base.yaml): base training defaults.
- [configs/eval.yaml](configs/eval.yaml): evaluation defaults.
- [docs/goals/09-llm-co-researchers.md](docs/goals/09-llm-co-researchers.md): how to use open LLMs as co-researchers.

## Core Architecture

Planned package structure:

```text
src/medcolbert/
  cli.py
  data/
    umls.py
    ontology.py
    corpora.py
    passages.py
    triplets.py
    negatives.py
    decontaminate.py
    audits.py
  generation/
    dspy_programs.py
    prompts.py
    validators.py
    teacher_filter.py
  training/
    pylate_train.py
    datasets.py
    losses.py
  eval/
    beir.py
    r2med.py
    trec_ct.py
    vocab_shift.py
    baselines.py
  utils/
    io.py
    text.py
    hashing.py
    logging.py
```

Use script files in `scripts/` only as CLI wrappers around library functions.

## Data Privacy Rules

UMLS, SNOMED CT, RxNorm, and some source vocabularies have license constraints.
This project may use them privately, but must not publish restricted source
vocabulary content.

Private-only artifacts include:

- raw UMLS files such as `MRCONSO.RRF`, `MRREL.RRF`, `MRSTY.RRF`, `MRDEF.RRF`
- CUI to string tables
- synonym tables
- matched spans containing UMLS source strings
- SNOMED hierarchy dumps
- UMLS-derived triplets where query or passage text is copied from restricted
  vocabulary strings at scale
- audit files containing restricted terms

Public-safe artifacts include:

- code
- configs
- toy examples that do not contain restricted vocabulary content
- manifests with opaque IDs, hashes, source names, and counts
- reproduction scripts requiring the user to provide licensed UMLS access
- model cards and provenance summaries

Default private path:

```text
data/processed/private/
```

Default public path:

```text
data/processed/public/
```

If unsure whether an artifact is public-safe, keep it private.

## Research Controls

Every training example should carry metadata:

- `example_id`
- `source`
- `source_doc_id`
- `positive_passage_id`
- `generation_mode`
- `target_cuis`
- `semantic_groups`
- `vocab_shift_type`
- `negative_type`
- `generator_model`
- `prompt_hash`
- `validator_version`
- `teacher_model`
- `teacher_margin`
- `data_hash`

Every experiment should record:

- git commit
- config files and hashes
- model backbone and revision
- training data manifest and hashes
- dependency lockfile
- random seed
- hardware summary
- wall-clock runtime

## Required Baselines

Do not claim SOTA or mechanism-level improvement without these controls:

- BM25
- BM25 plus query expansion or RM3 when practical
- strong current general embedding model
- MedCPT or another biomedical dense baseline
- BMRetriever when practical
- generic ColBERT/PyLate baseline
- BioClinical-ColBERT without ontology-controlled supervision
- MedColBERT
- optional hybrid BM25 plus MedColBERT via RRF

The BioClinical-ColBERT without ontology supervision baseline is mandatory. It
separates the value of the backbone from the value of the ontology-controlled
training recipe.

MedCPT is a useful historical biomedical retrieval baseline and may be useful
for cheap candidate mining. Do not treat it as the main teacher or arbiter of
medical relevance by default. Prefer a current strong LLM judge, such as
Gemma 4 31B or another validated open model, for small-batch adjudication,
synthetic-data filtering, and preference labeling. If scale requires a cheaper
teacher, distill the LLM-labeled judgments into a project-specific cross-encoder
or reranker and validate it against manual audits.

## LLM Co-Researcher Rules

This project should use open LLMs in two different roles.

Strong open LLMs are co-researchers. They may:

- critique hypotheses
- compare data sources
- propose ablations
- judge generated examples
- find false negatives
- review evaluation tables
- red-team paper claims
- create preference labels for a smaller distilled judge

Small LLMs are executors. They may:

- implement bounded tasks from [GOALS.md](GOALS.md)
- write tests for specified behavior
- refactor code without changing research intent
- run data processing commands
- summarize logs and audit counts

Small LLMs must not:

- change the research claim
- add or remove required baselines
- decide that a noisy data source is acceptable
- weaken privacy rules
- skip decontamination
- treat synthetic evals as primary evidence

When a decision affects the research claim, data mixture, filtering rubric,
baseline set, or release policy, create an evidence packet and use the
[LLM Co-Researcher Protocol](docs/goals/09-llm-co-researchers.md).

## Implementation Style

- Use Python 3.11 or 3.12.
- Prefer typed dataclasses or Pydantic-style schemas for records that cross
  module boundaries.
- Use pandas/pyarrow for parquet data processing.
- Use deterministic IDs and stable hashes for generated artifacts.
- Write small pure functions for parsing, normalization, filtering, and hashing.
- Keep comments short and useful.
- Avoid custom retrieval math until PyLate/ColBERT tools are proven inadequate.
- Avoid hardcoded local paths except defaults loaded from config files.
- Use `pathlib.Path` for filesystem paths.
- Do not silently drop examples; return counts and rejection reasons.

## Common Commands

Install base dependencies:

```bash
uv sync
```

Install optional groups:

```bash
uv sync --extra data --extra eval --extra generation --extra training --extra dev
```

Run tests:

```bash
uv run pytest
```

Run lint:

```bash
uv run ruff check .
```

Planned CLI commands:

```bash
medcolbert umls extract-pairs --config configs/data.yaml
medcolbert umls build-relations --config configs/data.yaml
medcolbert corpora ingest --config configs/data.yaml --source pubmed
medcolbert synth generate --config configs/generation.yaml --limit 1000
medcolbert data build-triplets --config configs/data.yaml --limit 100000
medcolbert train pylate --config configs/train_base.yaml
medcolbert eval run --config configs/eval.yaml
medcolbert eval vocab-shift --config configs/eval.yaml
```

## Definition of Done

A code change is done only when:

- the relevant phase goal is advanced
- tests or smoke checks were run when practical
- private/public data boundaries are preserved
- generated files are documented or ignored
- configs are updated if defaults changed
- audit counts are exposed for data transformations
- no unrelated user changes are reverted

For docs-only changes, tests are not required.
