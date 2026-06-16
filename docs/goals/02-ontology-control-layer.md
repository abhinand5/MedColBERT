# Phase 2: Ontology Control Layer

## Goal

Build a private UMLS-first control layer for concept sampling, vocabulary-shift
generation, semantic grouping, and hard-negative construction.

## Core Decision

Use UMLS as the v1 source of truth. Treat SNOMED CT, RxNorm, MeSH, ICD10CM,
LOINC, and other vocabularies as source vocabularies inside UMLS when available.
Only add separate SNOMED RF2 later if UMLS relationship quality is insufficient.

## Inputs

Private UMLS files:

```text
MRCONSO.RRF
MRREL.RRF
MRSTY.RRF
MRDEF.RRF optional
```

Config:

```text
configs/data.yaml
```

## Outputs

Private parquet files:

```text
data/processed/private/ontology/concepts.parquet
data/processed/private/ontology/strings.parquet
data/processed/private/ontology/concept_pairs.parquet
data/processed/private/ontology/concept_edges.parquet
data/processed/private/ontology/semantic_types.parquet
```

Public-safe manifests:

```text
data/processed/public/manifests/ontology_counts.json
data/processed/public/manifests/ontology_hashes.json
```

## Tasks

1. Parse `MRCONSO.RRF`.
2. Keep English, non-suppressed terms.
3. Normalize strings for filtering only:
   - lowercase
   - Unicode normalization
   - whitespace collapse
   - punctuation trimming
4. Keep original strings only in private files.
5. Build ambiguity reports:
   - normalized string to number of CUIs
   - abbreviation to candidate CUIs
   - source vocabulary coverage
   - semantic group coverage
6. Parse `MRSTY.RRF` and attach semantic types.
7. Parse `MRREL.RRF` and build candidate edges:
   - parent
   - child
   - broader
   - narrower
   - related
8. Build vocabulary-shift pair pools:
   - consumer to clinical
   - abbreviation to expanded term
   - brand to generic or ingredient
   - MeSH or biomedical to clinical
   - clinical variant to clinical variant
9. Write private audit samples.
10. Write public-safe count manifests.

## Data Schema

Concept string record:

```text
cui: str
sab: str
tty: str
code: str
scui: str | null
string: str
normalized_string: str
is_abbreviation: bool
semantic_types: list[str]
semantic_group: str
source_group: str
```

Pair record:

```text
pair_id: str
cui: str
left_string_id: str
right_string_id: str
left_source_group: str
right_source_group: str
vocab_shift_type: str
semantic_group: str
quality_flags: list[str]
```

Edge record:

```text
edge_id: str
source_cui: str
target_cui: str
relation: str
rela: str | null
sab: str
negative_use: str
confidence: str
```

## Quality Gates

- Concepts, strings, pairs, and edges all have stable IDs.
- Ambiguous abbreviations are marked or filtered.
- High-frequency normalized strings mapping to many CUIs are excluded from
  generation pools.
- Concept-near negatives do not share the same CUI.
- Manual audit samples exist for every major pair and edge type.
- Public manifests contain no restricted vocabulary strings.

## Acceptance Criteria

- Several hundred thousand candidate concept pairs exist after filtering, or the
  audit explains why the local UMLS distribution is too limited.
- Relationship-derived negative candidates exist for major semantic groups.
- At least these groups are separately countable:
  - disorders
  - procedures
  - drugs and chemicals
  - anatomy
  - findings, signs, and symptoms
  - genes/proteins when biomedical literature tasks are enabled

## Common Mistakes

- Do not publish CUI to string mappings.
- Do not assume CUI IDs alone prove public release safety for derived datasets.
- Do not use abbreviations without ambiguity controls.
- Do not treat all MRREL edges as equally useful hard negatives.
- Do not assume every UMLS SAB has the same redistribution terms.

