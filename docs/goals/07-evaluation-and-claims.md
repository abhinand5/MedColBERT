# Phase 7: Evaluation and Claims

## Goal

Evaluate MedColBERT on real external benchmarks and targeted vocabulary-shift
slices, then state only the claims supported by evidence.

## Inputs

- Trained checkpoints from Phase 6.
- Eval config from `configs/eval.yaml`.
- Benchmark corpora and qrels.
- Decontamination reports.
- UMLS/private annotation tools for analysis only.

## Outputs

```text
runs/eval/main_table.json
runs/eval/vocab_shift_table.json
runs/eval/efficiency_table.json
runs/eval/baseline_table.json
runs/eval/decontamination_summary.json
```

## Primary Benchmarks

Use a focused set first:

- R2MED for reasoning-driven medical retrieval.
- TREC Clinical Trials for long clinical query to trial matching.
- PMC-Patients or Open-Patients style patient/case retrieval.
- One or two BEIR/MTEB biomedical tasks such as NFCorpus, SciFact, BioASQ, or
  TREC-COVID.

Add MIRAGE or MedRGB only after retrieval quality is established, because RAG
benchmarks entangle retriever quality with generator and prompt choices.

## Required Metrics

- nDCG@10
- Recall@100
- Recall@1000 when candidate recall matters
- MAP@10 when benchmark-standard
- index size
- query latency
- encoding throughput
- MRL dim 32/64/128 tradeoff when MRL is enabled

## Vocabulary-Shift Evaluation

Build a slice analysis with buckets:

- same_vocab
- lay_to_clinical
- abbreviation_to_expanded
- brand_generic
- symptom_to_diagnosis_or_treatment
- biomedical_to_clinical
- no_direct_lexical_overlap

The key result is not average score alone. The key result is whether gains are
concentrated in vocabulary-shift buckets while same-vocabulary performance
stays competitive.

## Mini MedVocabShift Benchmark

If existing benchmarks do not isolate vocabulary shift well enough, build a
small benchmark over public passages.

Target size:

```text
500 to 2000 queries
```

Rules:

- documents must be public
- labels must be auditable
- examples must be bucketed by shift type
- do not train on this benchmark
- keep UMLS-derived strings private if used during construction
- release only public-safe components

## Required Baseline Table

Report at least:

- BM25
- BM25 plus expansion or RM3
- strong general embedding model
- MedCPT or another biomedical dense baseline, reported as a baseline rather
  than treated as an oracle
- BMRetriever when practical
- generic ColBERT
- BioClinical-ColBERT without ontology
- MedColBERT
- optional BM25 plus MedColBERT hybrid

## Claim Wording

Acceptable:

> MedColBERT achieves competitive or state-of-the-art results on selected
> biomedical and clinical retrieval benchmarks, with the largest gains on
> vocabulary-shift slices.

Avoid:

> MedColBERT is SOTA on medical retrieval.

Avoid:

> Dense retrieval cannot solve medical synonymy.

## Acceptance Criteria

- All reported eval corpora are decontaminated from training.
- Baselines are run with comparable corpora and documented settings.
- The no-ontology ColBERT control is included.
- Vocabulary-shift buckets are reported.
- Efficiency is reported.
- Claims are benchmark-scoped.

## Common Mistakes

- Do not tune prompts or data generation against the test set.
- Do not compare against stale or weak baselines only.
- Do not hide same-vocabulary regressions.
- Do not overclaim based on one benchmark.
- Do not report RAG gains as pure retrieval gains.
