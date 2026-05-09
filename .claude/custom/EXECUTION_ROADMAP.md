# MedColBERT Execution Roadmap

This roadmap converts the master research plan into a phase-by-phase execution plan.
It separates what Abhinand should decide or provide from what an agent should build.

Current state:

- Research plan: credible.
- Repo implementation: scaffold only.
- Main blockers: private UMLS integration, data quality, decontamination, training smoke test.
- Immediate milestone: 1k audited concept-grounded triplets and a tiny ColBERT training run
  that ranks positives above negatives.

---

## Phase 0: Environment and Data Access

### Goal

Make the project runnable and confirm private data/model access.

### Abhinand Requirements

- Provide the private UMLS Hugging Face repo ID or local path.
- Confirm expected UMLS file layout:
  - `MRCONSO.RRF`
  - `MRREL.RRF`
  - `MRSTY.RRF`
  - optional `MRDEF.RRF`
- Confirm available compute for smoke tests.
- Confirm whether PubMed/PMC downloads happen locally or on remote compute.
- Keep UMLS and UMLS-derived string artifacts private.

### Agent Tasks

- Create package structure under `src/medcolbert/`.
- Add a CLI skeleton.
- Verify dependency install with project extras.
- Verify BioClinical-ModernBERT loads.
- Verify PyLate can instantiate a ColBERT model.
- Add a tiny model forward-pass smoke test.

### Requirements

- Python 3.11 or 3.12.
- HF token with private UMLS access.
- Enough disk for private data, generated artifacts, and indexes.

### Exit Criteria

- One command can locate/load UMLS files.
- One command can run a tiny BioClinical-ModernBERT/ColBERT forward pass.
- No public path contains UMLS-derived strings.

---

## Phase 1: UMLS Extraction and Audit

### Goal

Create high-confidence concept pairs and concept-near negative candidates.

### Abhinand Requirements

- Confirm UMLS source path/repo.
- Review audit samples.
- Decide whether pair quality is acceptable before scaling.

### Agent Tasks

- Parse `MRCONSO.RRF`.
- Filter English, non-suppressed terms.
- Normalize strings for filtering only.
- Build ambiguity reports:
  - normalized string -> number of CUIs.
  - abbreviation -> candidate CUIs.
  - source vocabulary coverage.
- Extract private cross-vocabulary pairs:
  - consumer/CHV -> clinical.
  - abbreviation -> expanded clinical term.
  - RxNorm brand/generic/ingredient where available.
  - clinical vocabulary variants.
- Parse `MRSTY.RRF` for semantic type filtering.
- Parse `MRREL.RRF` for related/parent/child concept candidates.
- Generate private parquet outputs.
- Generate audit samples as CSV/Markdown.

### Requirements

- UMLS files.
- `pandas`, `pyarrow`.
- Private output path, e.g. `data/processed/private/`.

### Exit Criteria

- Several hundred thousand high-confidence pairs after filtering.
- Ambiguity report exists.
- Concept-near negative candidates exist.
- Manual sample acceptance target: 80%+.

---

## Phase 2: Passage Corpus and Annotation

### Goal

Build retrievable passages grounded to UMLS CUIs.

### Abhinand Requirements

- Decide initial passage sources:
  - recommended first: MedEmbed data and PubMed abstracts.
  - later: PMC OA.
  - defer: MIMIC unless credentialing and privacy constraints are already solved.
- Provide MedEmbed data location if it should be used.
- Approve chunking defaults.

### Agent Tasks

- Define a standard passage schema:
  - `passage_id`
  - `source`
  - `source_doc_id`
  - `title`
  - `text`
  - `metadata`
  - `cuis`
  - `matched_spans`
- Ingest MedEmbed/PubMed/PMC into this schema.
- Chunk long documents into 256-384 token passages with overlap.
- Build a UMLS dictionary matcher for passage annotation.
- Store `passage_id -> CUIs` privately.
- Build exact and near-duplicate fingerprints for decontamination.

### Requirements

- Passage corpus.
- UMLS term index from Phase 1.
- Disk for passage parquet files and fingerprints.

### Exit Criteria

- At least 100k annotated passages for pilot work.
- Passage-CUI coverage report exists.
- Decontamination fingerprints exist.

---

## Phase 3: DSPy Synthetic Query Pilot

### Goal

Generate a small, high-quality synthetic query set before scaling.

### Abhinand Requirements

- Choose or provide the open-source LLM endpoint:
  - vLLM or SGLang recommended.
  - OpenAI-compatible API preferred.
- Review 200-500 generated examples.
- Mark examples as good/bad for prompt and metric tuning.

### Agent Tasks

- Implement DSPy signatures/modules for medical query generation.
- Add deterministic validators:
  - query length.
  - target CUI coverage.
  - forbidden surface-term overlap.
  - near-duplicate detection.
  - generic-query rejection.
  - passage answerability.
- Generate the first 1k examples.
- Produce audit files with:
  - passage.
  - target CUIs.
  - alternative terms.
  - generated query.
  - validation outcomes.
  - rejection reason if any.
- Iterate DSPy prompt/program based on audit.
- Scale only to 50k-100k pilot examples after passing audit.

### Requirements

- Running open-source LLM endpoint.
- DSPy configured through `dspy.LM`.
- Annotated passages.
- UMLS alternatives from Phase 1.

### Exit Criteria

- 1k generated examples.
- 80%+ manual acceptance.
- Query target CUIs match passage CUIs.
- Queries show real vocabulary shift rather than passage copying.

---

## Phase 4: Triplet Construction

### Goal

Produce train/dev triplets with multiple negative types.

### Abhinand Requirements

- Review negative samples by type.
- Approve whether negatives look medically meaningful.
- Confirm eval datasets that must be blocked from training.

### Agent Tasks

- Build positives from query target CUIs and passage-CUI annotations.
- Mine BM25 negatives.
- Build UMLS concept-near negatives:
  - parent/child.
  - related concepts.
  - same semantic type, different CUI.
- Add random same-semantic-group negatives.
- Add dense-mined negatives later, after a first trained model exists.
- Build train/dev splits.
- Run decontamination against all eval corpora available so far.
- Produce audit samples per negative type.

### Requirements

- Annotated passages.
- Generated queries.
- UMLS relation graph.
- BM25 index.
- Decontamination fingerprints.

### Exit Criteria

- 50k-100k pilot triplets.
- Each triplet has source and negative-type metadata.
- False-negative rate is acceptable in manual audit.
- No known eval leakage.

---

## Phase 5: Training Smoke Test

### Goal

Prove the model can train end-to-end before scaling.

### Abhinand Requirements

- Provide GPU compute.
- Review first learning curves and retrieval sanity metrics.

### Agent Tasks

- Run Stage 0 tiny training on 1k triplets.
- Run short concept warm-up on UMLS pairs.
- Run pilot ColBERT training with PyLate.
- Build a tiny PLAID index.
- Verify positives rank above negatives.
- Save run metadata:
  - git hash.
  - config hash.
  - model name.
  - data file hashes.
  - dependency lockfile.

### Requirements

- GPU.
- PyLate working with BioClinical-ModernBERT.
- Pilot triplets.

### Exit Criteria

- Loss decreases.
- No NaN/Inf.
- Tiny retrieval sanity passes.
- Training and indexing commands are reproducible.

---

## Phase 6: Pilot Evaluation and Go/No-Go

### Goal

Decide whether the core paper hypothesis is alive.

### Abhinand Requirements

- Review early results.
- Decide whether to continue, reframe, or adjust data.
- Focus on cross-vocabulary lift, not only average nDCG.

### Agent Tasks

- Run pilot baselines:
  - BM25.
  - BioClinical dense bi-encoder.
  - generic ColBERT/PyLate.
  - MedColBERT pilot.
- Run initial eval on:
  - NFCorpus.
  - SciFact.
  - BioASQ if split is ready.
  - R2MED.
  - TREC Clinical Trials if available.
- Build the first cross-vocabulary split.
- Report metrics by bucket:
  - same vocabulary.
  - lay -> clinical.
  - abbreviation -> expanded.
  - brand/generic.
  - no direct lexical overlap.

### Requirements

- Eval harness.
- Baseline implementations.
- UMLS annotation over eval queries/docs.

### Exit Criteria

- MedColBERT beats at least one strong baseline in a meaningful setting.
- Cross-vocabulary bucket shows measurable lift.
- If not, data generation/training is adjusted before scaling.

---

## Phase 7: Scale V1 Base Model

### Goal

Train a serious base model with 1M-2M high-quality triplets.

### Abhinand Requirements

- Secure multi-GPU compute.
- Approve data scale after audit.
- Avoid changing multiple variables mid-run.

### Agent Tasks

- Generate 1M-2M validated triplets.
- Train base model.
- Mine dense negatives using the trained checkpoint.
- Rebuild triplets with dense negatives.
- Continue training or retrain from warm-up checkpoint.
- Run full primary eval suite.
- Generate model card and data provenance draft.

### Requirements

- 4x A100/H100 preferred.
- Stable dataset files.
- Stable configs.
- Run tracking.

### Exit Criteria

- Full benchmark table exists.
- Cross-vocabulary table exists.
- Efficiency table exists.
- Main ablations are queued.

---

## Phase 8: Paper-Grade Experiments

### Goal

Produce the final evidence package for a good conference submission.

### Abhinand Requirements

- Decide target venue and paper framing.
- Review claims conservatively.
- Decide whether large model training is worth the compute.

### Agent Tasks

- Run required ablations:
  - no UMLS warm-up.
  - no synthetic data.
  - unconstrained DSPy synthetic data.
  - no concept-near negatives.
  - no MRL.
  - dense vs late interaction.
  - base vs large if large is trained.
- Run full benchmark suite:
  - NFCorpus.
  - TREC-COVID.
  - BioASQ.
  - SciFact.
  - TREC Clinical Trials.
  - R2MED.
  - PMC-Patients.
  - optional CliniQ.
- Run downstream RAG only after retrieval is strong:
  - MIRAGE.
  - MedRGB.
- Prepare paper tables and plots.
- Prepare release audit.

### Requirements

- Stable eval scripts.
- Baseline reproducibility.
- Decontamination logs.
- Compute budget for ablations.

### Exit Criteria

- Results support a precise, benchmark-scoped SOTA claim.
- Cross-vocabulary analysis supports the core mechanism.
- Code and model release are license-safe.
- UMLS-derived artifacts remain private/gated.

---

## Immediate Next Step

Do not start large-scale generation or training yet.

Next agent task:

1. Implement the UMLS loader/parser.
2. Generate the first UMLS extraction audit report.
3. Confirm whether the private UMLS files contain enough relationship data for concept-near negatives.

Next Abhinand task:

1. Provide the private UMLS HF repo ID or local path.
2. Confirm the compute target for smoke tests.
3. Review the first UMLS audit samples before approving synthetic generation.
