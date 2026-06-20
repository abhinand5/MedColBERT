# Phase 3: Public Corpus Registry

## Goal

Build real public passage stores for training and evaluation while keeping
corpus provenance, licenses, and decontamination metadata explicit.

## Core Principle

Use real documents and synthetic supervision. The document side should come
from public biomedical or clinical corpora wherever possible.

## Priority Sources

Use these first:

- PubMed abstracts or MedRAG/pubmed-style corpora.
- ClinicalTrials.gov trial records.
- BioASQ and TREC qrel-backed corpora for supervised examples where allowed.

Use OpenMed sources selectively:

- MedDialog for patient-style query seeds.
- Medical reasoning datasets for query style and task templates.
- Drug/protein datasets for entity-heavy slices and hard-negative ideas.
- PII/de-identification models or datasets for privacy audit support.

Do not treat OpenMed reasoning or dialogue data as ground-truth retrieval labels
without independent passage grounding and filtering.

## Inputs

- Public corpus paths or Hugging Face dataset IDs.
- License metadata.
- Split metadata.
- Optional qrels.
- Phase 2 ontology string matcher.

## Outputs

Private passage store:

```text
data/processed/private/passages/passages.parquet
data/processed/private/passages/passage_cui_matches.parquet
data/processed/private/passages/fingerprints.parquet
```

Public-safe corpus manifest:

```text
data/processed/public/manifests/corpora.json
```

## Passage Schema

```text
passage_id: str
source: str
source_doc_id: str
split: str | null
title: str | null
text: str
url: str | null
license: str | null
metadata: dict
text_hash: str
normalized_text_hash: str
minhash: bytes | str
```

Private annotation schema:

```text
passage_id: str
cui: str
matched_span: str
start_char: int
end_char: int
sab: str
source_group: str
semantic_group: str
match_method: str
confidence: str
```

## Tasks

1. Implement a corpus registry with source name, loader, license, split, and
   contamination policy.
2. Ingest one corpus at a time into the standard passage schema.
3. Chunk long documents into retrieval passages:
   - default max tokens: 256 to 384
   - overlap: 32 to 64 tokens
   - keep source document ID
4. Add deterministic fingerprints:
   - exact hash
   - normalized hash
   - MinHash or SimHash
   - PMID, PMCID, NCT ID where available
5. Annotate passages with UMLS matches privately.
6. Report concept coverage by corpus and semantic group.
7. Save public-safe corpus manifests.

## Acceptance Criteria

- At least one corpus is ingested end to end.
- Every passage has stable IDs and source metadata.
- Fingerprints exist before triplet construction.
- CUI annotations are private-only.
- Corpus manifests include license and split notes.
- Eval corpora are clearly marked and blocked from training where needed.

## Common Mistakes

- Do not train on eval splits or near-duplicates.
- Do not discard source document IDs.
- Do not mix public manifests with private matched spans.
- Do not rely only on raw lexical matching for final quality claims.
- Do not let a single corpus dominate all generated examples.

