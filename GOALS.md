# MedColBERT Goals

This file is the phase index for MedColBERT. Read [AGENTS.md](AGENTS.md)
first, then execute phases in order. The project is organized around
go/no-go gates, not fixed dates.

## Research Thesis

MedColBERT should establish that a late-interaction retriever trained with
ontology-controlled synthetic supervision can transfer to real biomedical and
clinical retrieval benchmarks, especially under medical vocabulary shift.

The strongest claim is not:

> We trained ColBERT on medical text.

The strongest claim is:

> Real-corpus, ontology-controlled synthetic supervision improves medical
> retrieval when queries and relevant documents express the same concept with
> different vocabulary.

## Strategy

Use real public corpora as documents, synthetic supervision as the main training
signal, UMLS/SNOMED/RxNorm/MeSH as a private control layer, and real external
benchmarks for evaluation.

Do not build a giant manually CUI-labeled dataset before training. Instead,
generate demand over real medical supply:

1. Select real passages from public corpora.
2. Detect or sample target medical concepts.
3. Generate role-specific queries that intentionally use different surface forms.
4. Attach positives and hard negatives.
5. Filter with deterministic checks and modern LLM/reranker margins.
6. Train and evaluate against strong baselines.

## Phase Index

| Phase | Goal | Detailed Plan |
|---|---|---|
| 0 | Lock the research contract and claim boundaries. | [Phase 0: Research Contract](docs/goals/00-research-contract.md) |
| 1 | Make the repo runnable, private-data-safe, and agent-friendly. | [Phase 1: Environment and Privacy](docs/goals/01-environment-and-privacy.md) |
| 2 | Build the private ontology control layer from UMLS-first sources. | [Phase 2: Ontology Control Layer](docs/goals/02-ontology-control-layer.md) |
| 3 | Register public corpora and build real passage stores. | [Phase 3: Public Corpus Registry](docs/goals/03-public-corpus-registry.md) |
| 4 | Generate ontology-controlled synthetic queries over real passages. | [Phase 4: Synthetic Data Engine](docs/goals/04-synthetic-data-engine.md) |
| 5 | Build triplets with strong positives, hard negatives, and audits. | [Phase 5: Triplets and Negatives](docs/goals/05-triplets-and-negatives.md) |
| 6 | Train baseline and MedColBERT models with clear ablations. | [Phase 6: ColBERT Training](docs/goals/06-colbert-training.md) |
| 7 | Evaluate against real benchmarks and vocabulary-shift slices. | [Phase 7: Evaluation and Claims](docs/goals/07-evaluation-and-claims.md) |
| 8 | Prepare release artifacts, model cards, and paper evidence. | [Phase 8: Release and Paper](docs/goals/08-release-and-paper.md) |
| Cross-cutting | Use open LLMs as co-researchers and small LLMs as executors. | [LLM Co-Researcher Protocol](docs/goals/09-llm-co-researchers.md) |

## Master Go/No-Go Gates

Proceed from one phase to the next only when the gate passes.

| Gate | Required Evidence |
|---|---|
| Ontology gate | UMLS terms, concept IDs, semantic types, and relation candidates load from private paths; no restricted strings appear in public outputs. |
| Corpus gate | At least one public passage corpus is ingested with stable IDs, source metadata, and decontamination fingerprints. |
| Synthetic gate | Pilot generated examples show vocabulary shift, answerability, target concept coverage, and acceptable manual audit quality. |
| Triplet gate | Negatives are medically meaningful and false-negative rate is acceptable in manual samples. |
| Training gate | BioClinical-ColBERT baseline and MedColBERT both train without numerical failures and retrieve positives in sanity checks. |
| Evaluation gate | MedColBERT beats the no-ontology ColBERT control and at least one strong biomedical baseline on an important vocabulary-shift slice. |
| Release gate | Public release contains no UMLS/SNOMED-derived restricted string tables and includes complete provenance. |

## Non-Negotiable Controls

- Always keep UMLS-derived strings, synonym tables, concept annotations, and
  relation files in private paths.
- Never train only on raw synonym pairs. The final model must learn query to
  passage relevance.
- Never evaluate the core claim only on synthetic evaluation data.
- Always include a BioClinical-ColBERT without ontology supervision baseline.
- Always include a UMLS/query-expansion or lexical expansion baseline when
  claiming learned vocabulary-shift gains.
- Do not treat older biomedical dense models as authoritative teachers. Use
  them as baselines, cheap miners, or sanity checks unless they win current
  validation.
- Strong open LLMs may critique, judge, design ablations, and stress-test
  claims. Small LLMs should execute bounded task cards from the markdown goals,
  not invent project strategy.
- Always log source, prompt, generator, verifier, negative type, and data hashes.

## Expected Public Artifacts

- Source code and configs.
- Reproduction scripts that require licensed users to provide their own UMLS.
- Model weights, only after license and memorization review.
- Evaluation harness and run files.
- Public corpus manifests that do not expose restricted vocabulary strings.
- A model card with data provenance, known limitations, and safety notes.
