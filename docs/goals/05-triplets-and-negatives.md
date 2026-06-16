# Phase 5: Triplets and Negatives

## Goal

Convert generated examples, qrel-backed examples, citation-context examples, and
ontology controls into high-quality training triplets with strong negative
metadata and false-negative audits.

## Inputs

- Accepted synthetic examples from Phase 4.
- Passage store from Phase 3.
- Ontology edges and pairs from Phase 2.
- BM25 index.
- Optional dense retriever checkpoint.
- Eval corpus blocklists and fingerprints.

## Outputs

Private training files:

```text
data/processed/private/triplets/pilot_triplets.parquet
data/processed/private/triplets/dev_triplets.parquet
data/processed/private/triplets/v1_triplets.parquet
data/processed/private/triplets/negative_audit_samples.parquet
```

Public-safe manifests:

```text
data/processed/public/manifests/triplet_counts.json
data/processed/public/manifests/decontamination_report.json
```

## Triplet Schema

```text
triplet_id: str
query: str
positive_passage_id: str
negative_passage_id: str
source: str
generation_mode: str | null
task_family: str | null
target_cuis: list[str]
positive_cuis: list[str]
negative_cuis: list[str]
negative_type: str
negative_source: str
teacher_margin: float | null
split: str
data_hash: str
```

## Negative Types

Use multiple negative types:

- `bm25_hard`: lexically similar, not target concept.
- `ontology_parent_child`: parent or child concept, wrong target.
- `ontology_sibling`: sibling under same parent, wrong target.
- `ontology_related`: related but distinct concept.
- `same_semantic_group`: same semantic group, different concept.
- `dense_mined`: retrieved by an interim dense or ColBERT model.
- `random`: calibration only, low fraction.

## False-Negative Controls

Reject a negative if:

- it shares any primary target CUI with the query
- it is a near-duplicate of the positive
- it has the same PMID, PMCID, NCT ID, or source document when that would make
  it an accidental positive
- teacher or heuristic checks suggest it is relevant
- manual audit repeatedly marks the negative type as ambiguous

## Data Mixture

For pilot training, prefer diversity over size:

- ontology-grounded synthetic examples
- passage-grounded generic synthetic examples for ablation
- citation-context examples where available
- qrel-backed examples where available
- clinical-trial examples if the corpus is ready

For v1 training, use source-aware sampling. Do not let any one source dominate
without an explicit config weight.

## Decontamination

Before writing train files:

1. Load eval document IDs and query IDs.
2. Block exact IDs:
   - PMID
   - PMCID
   - NCT ID
   - dataset-specific IDs
3. Block exact text hashes.
4. Block normalized text hashes.
5. Block near-duplicates by MinHash or SimHash threshold.
6. Block synthetic queries too similar to eval queries.
7. Write a public-safe count report.

## Acceptance Criteria

- 50k to 100k pilot triplets exist after decontamination.
- Each triplet has source, task family, generation mode, and negative type.
- Manual negative audit has acceptable false-negative rate.
- No known eval leakage remains.
- Ablation subsets can be selected by `generation_mode` and `negative_type`.

## Common Mistakes

- Do not treat ontology-near concepts as automatically wrong in every context.
- Do not include dense-mined negatives before a first model exists.
- Do not build one huge triplet file without source metadata.
- Do not decontaminate only by IDs; use text similarity too.
- Do not ignore ambiguous negatives. They poison contrastive training.

