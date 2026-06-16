# Phase 4: Synthetic Data Engine

## Goal

Generate high-quality synthetic medical queries over real public passages using
the private ontology control layer.

## Core Principle

Generate synthetic demand over real medical supply. Avoid synthetic documents as
the main training corpus.

## Inputs

- Real passage store from Phase 3.
- Private ontology pairs and concept edges from Phase 2.
- Generation config from `configs/generation.yaml`.
- OpenAI-compatible local or remote LLM endpoint.
- Optional modern LLM judge, distilled reranker, or candidate-mining model.

## Outputs

Private generated examples:

```text
data/processed/private/synthetic/candidates.parquet
data/processed/private/synthetic/accepted.parquet
data/processed/private/synthetic/rejected.parquet
data/processed/private/synthetic/audit_samples.parquet
```

Public-safe generation manifest:

```text
data/processed/public/manifests/synthetic_generation.json
```

## Task Families

Generate examples by task family. Every task family must be separately
identifiable in metadata.

| Task Family | Query Style | Positive Document |
|---|---|---|
| lay_to_clinical | patient wording | clinical or biomedical passage |
| abbreviation_to_expanded | short clinical abbreviation | passage with expanded term |
| brand_generic | brand or generic drug mention | drug evidence passage |
| symptom_to_diagnosis | symptom phrase | diagnosis or treatment passage |
| trial_matching | patient vignette | ClinicalTrials.gov trial |
| biomedical_to_clinical | research wording | clinical passage |
| patient_similarity | patient summary | case report or patient passage |
| no_direct_lexical_overlap | paraphrased concept query | passage with target concept |

## Generation Record Schema

```text
example_id: str
generation_mode: str
task_family: str
query: str
positive_passage_id: str
target_cuis: list[str]
semantic_groups: list[str]
vocab_shift_type: str
role: str
generator_model: str
generator_version: str
prompt_hash: str
program_hash: str
source_passage_hash: str
forbidden_surface_terms_hash: str
validation_status: str
rejection_reasons: list[str]
teacher_model: str | null
teacher_score_positive: float | null
teacher_score_best_negative: float | null
teacher_margin: float | null
```

## Required Generation Modes

Build these modes so the main ablation is easy:

1. `generic_synthetic`: LLM query from passage without ontology controls.
2. `passage_grounded`: query must be answerable from the passage.
3. `ontology_grounded`: query must target selected CUIs and vocabulary shift.
4. `ontology_grounded_teacher_filtered`: ontology-grounded plus LLM/reranker
   margin.

## Validators

Every candidate must pass deterministic validators before teacher filtering:

- query length within configured bounds
- target CUI coverage when possible
- no excessive forbidden surface-term copying
- not a near-duplicate of existing query
- not a generic medical question
- answerable from the positive passage
- no obvious PHI pattern
- language is English unless configured otherwise
- role and task family are consistent

## Teacher Filtering

Use a current strong LLM judge, such as Gemma 4 31B or another validated open
model, as the preferred high-quality teacher for pilot filtering and preference
labels. Use older biomedical dense models as candidate miners or baselines, not
as the default source of truth.

Teacher filtering should score the positive against a candidate negative pool.
Keep the example only if:

```text
teacher_score_positive - teacher_score_best_negative >= configured_margin
```

Teacher scores are quality gates, not the final truth. Always keep hard-label
metadata and negative source metadata.

For scale, use a two-stage teacher:

1. Use the LLM judge on pilot data and hard/ambiguous examples.
2. Distill those judgments into a cheaper cross-encoder or reranker for bulk
   filtering.

Validate the distilled teacher against manual audits before trusting it.

## Manual Audit

Create audit samples by:

- role
- task family
- semantic group
- generation mode
- corpus source
- accepted/rejected status

Audit labels:

```text
good
minor_issue
bad_query
bad_positive
copies_passage
wrong_concept
too_generic
unsafe_or_phi
```

## Acceptance Criteria

- 1k generated candidates exist.
- Accepted examples show real vocabulary shift.
- Manual acceptance is high enough to justify 50k to 100k pilot generation.
- Ontology-grounded examples are better than generic synthetic examples in audit.
- All generated records have prompt/model/config provenance.

## Common Mistakes

- Do not ask the LLM to invent both query and document.
- Do not let the LLM choose target concepts without verification.
- Do not use chain-of-thought rationales in released data.
- Do not filter only by lexical overlap; check answerability and concept intent.
- Do not scale generation if audit samples look mediocre.
