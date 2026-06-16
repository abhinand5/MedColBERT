# Phase 6: ColBERT Training

## Goal

Train ColBERT-style models that isolate the value of ontology-controlled
synthetic supervision and can be compared fairly against dense, sparse, and
late-interaction baselines.

## Inputs

- Training triplets from Phase 5.
- Configs:
  - `configs/base.yaml`
  - `configs/train_base.yaml`
- BioClinical-ModernBERT model.
- PyLate or fallback ColBERT training stack.
- GPU compute.

## Outputs

```text
checkpoints/bioclinical_colbert_no_ontology/
checkpoints/medcolbert_pilot/
checkpoints/medcolbert_v1/
runs/training/*/metrics.jsonl
runs/training/*/config_snapshot.yaml
runs/training/*/data_manifest.json
```

## Required Model Variants

Train these in order:

1. `generic_colbert`: off-the-shelf ColBERT or PyLate baseline when practical.
2. `bioclinical_colbert_no_ontology`: same backbone, no ontology-controlled
   generation.
3. `medcolbert_ontology_grounded`: ontology-controlled synthetic supervision.
4. `medcolbert_teacher_filtered`: ontology-controlled plus modern LLM/reranker
   filtering.
5. `medcolbert_dense_negative_refresh`: continue or retrain with dense-mined
   negatives.

Large model training is optional and should happen only after base model
ablations support the hypothesis.

## Training Stages

### Stage 0: Smoke Test

- 1k examples.
- Verify forward pass.
- Verify loss decreases.
- Verify no NaN or Inf.
- Build tiny index.
- Retrieve positives above simple negatives.

### Stage 1: Short Concept Warmup

Use concept-aligned examples only briefly. Prefer query to real passage examples
over raw synonym pair memorization.

### Stage 2: Main Triplet Training

Use mixed triplets with source-aware sampling. Keep the first serious model
simple: 128-dim ColBERT projection, stable query/document lengths, no large
architectural experiments.

### Stage 3: Dense-Negative Refresh

Mine hard negatives with the Stage 2 checkpoint, rebuild triplets, and continue
training or retrain from the warmup checkpoint.

### Stage 4: MRL and Efficiency

Add MRL only after the baseline model is stable. Report dim 32, 64, and 128
tradeoffs for index size, latency, and quality.

## Run Metadata

Every run must write:

- git commit
- training config snapshot
- data manifest and hashes
- model name and revision
- tokenizer revision
- dependency lockfile hash
- seed
- hardware summary
- precision
- gradient accumulation
- effective batch size

## Acceptance Criteria

- BioClinical-ColBERT-no-ontology trains successfully.
- MedColBERT trains successfully with the same backbone and comparable compute.
- Training curves are stable.
- Tiny retrieval sanity passes.
- Candidate indexes can be built.
- Model variants are reproducible from config and manifest files.

## Common Mistakes

- Do not compare MedColBERT to a weaker backbone baseline only.
- Do not change backbone, data, negatives, and loss all at once.
- Do not overfit raw UMLS synonym strings.
- Do not skip index-building smoke tests.
- Do not train a large model before base ablations are interpretable.
