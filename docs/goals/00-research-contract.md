# Phase 0: Research Contract

## Goal

Lock the exact research claim, required controls, and unacceptable shortcuts
before building more pipeline code.

## Claim

MedColBERT should show that real-corpus, ontology-controlled synthetic
supervision trains a late-interaction medical retriever that transfers to real
biomedical and clinical retrieval benchmarks, with the largest gains under
vocabulary shift.

## Non-Goals

- Do not claim universal medical retrieval SOTA.
- Do not claim dense retrievers cannot model synonymy.
- Do not make synthetic evaluation the primary evidence.
- Do not make MedEmbed or any single synthetic dataset the main quality anchor.
- Do not require comprehensive manual CUI labeling before the first model.

## Required Comparisons

At minimum, the paper-quality comparison must include:

- BM25
- BM25 plus expansion or RM3 when available
- biomedical dense retriever such as MedCPT as a historical baseline
- strong general embedding model
- generic ColBERT or PyLate ColBERT
- BioClinical-ColBERT without ontology-controlled supervision
- MedColBERT

Add BMRetriever, SPLADE, cross-encoder reranking, and hybrid RRF baselines when
practical.

## Main Ablation

The primary ablation is the training data recipe:

| Variant | Purpose |
|---|---|
| no synthetic | measures value of generated supervision |
| generic synthetic | tests plain LLM query generation |
| passage-grounded synthetic | tests answerability from real passages |
| ontology-grounded synthetic | tests UMLS/RxNorm/SNOMED control |
| ontology-grounded plus teacher-filtered | expected strongest recipe |

The preferred teacher for filtering is a current strong LLM judge or a reranker
distilled from its judgments. Older biomedical dense retrievers can help mine
candidates, but should not be assumed to be the best relevance judge.

## Success Criteria

The core hypothesis is alive if MedColBERT:

- beats BioClinical-ColBERT without ontology supervision on at least one
  important benchmark or slice
- improves on vocabulary-shift buckets without destroying same-vocabulary
  performance
- remains competitive with strong biomedical dense baselines
- shows a clear advantage over BM25 plus ontology/query expansion on at least
  one cross-vocabulary slice

## Failure Criteria

Reframe or stop scaling if:

- gains appear only on synthetic validation data
- MedColBERT does not beat the no-ontology ColBERT control
- ontology-grounded examples are mostly lexical copies
- negatives are frequently false negatives
- training wins only because eval passages leaked into training

## Implementation Notes

Create a small machine-readable research contract file before paper-grade runs,
for example:

```text
runs/contracts/medcolbert_v1.yaml
```

It should record:

- accepted claims
- blocked claims
- required baselines
- benchmark list
- data sources
- release constraints
- model variants
- random seeds

## Exit Gate

Phase 0 is complete when [GOALS.md](../../GOALS.md), [AGENTS.md](../../AGENTS.md),
and this contract agree on the claim, baseline requirements, and data privacy
rules.
