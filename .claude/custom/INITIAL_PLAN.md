# MedColBERT Master Plan

## 0. Executive Summary

MedColBERT should be built as a license-aware, ontology-guided late-interaction
retriever for biomedical and clinical search. The strongest publishable claim is
not simply "we trained ColBERT on medical data." The claim is:

> Late interaction trained with concept-grounded supervision improves retrieval
> under medical terminology mismatch, especially when queries and documents use
> different vocabularies for the same concept.

This plan assumes:

- UMLS is available in a private Hugging Face repository.
- UMLS-derived artifacts containing source vocabulary strings remain private or gated.
- Public releases include code, model weights where allowed, configs, evaluation harnesses,
  and reproducible manifests that require the user to provide their own UMLS access.
- Synthetic data is generated with open-source LLMs through DSPy and validated with
  deterministic concept/lexical checks plus sampled human review.
- UMLS is enough for v1. Separate SNOMED RF2 access is optional, not required, unless
  the UMLS distribution available locally does not include the SNOMEDCT_US content needed
  for hierarchy-derived negatives.

The previous plan was directionally useful but too optimistic on licensing, too loose
on UMLS/SNOMED identifiers, and too vague on what gets implemented first. This version
is the practical master plan.

---

## 1. Non-Negotiable Constraints

### 1.1 UMLS and SNOMED Licensing

UMLS is not an open dataset. A private HF repo is appropriate for personal or controlled
research workflows, but raw UMLS files and UMLS-derived synonym tables should not be
published openly.

Safe public artifacts:

- Source code that reads UMLS from a local path or private HF dataset.
- Configuration files specifying allowed SABs, semantic types, and filters.
- CUI-level manifests where no source vocabulary strings or definitions are exposed.
- Dataset generation recipes and hashes/fingerprints.
- Small illustrative examples in the paper, kept minimal and citation-aware.
- Model weights, subject to legal review and dataset provenance notes.

Unsafe or high-risk public artifacts:

- Full `MRCONSO` subsets.
- CUI-to-string synonym tables.
- SNOMED hierarchy dumps.
- UMLS-derived triplets where the positive/query text is copied directly from restricted
  source vocabularies at scale.
- Public HF datasets containing raw UMLS strings, definitions, or source-code mappings.

Practical policy:

1. Keep private data under `data/raw/umls` or a private HF dataset.
2. Mark all UMLS-derived intermediate files as non-public.
3. Release only regeneration scripts plus a clear "requires UMLS license" note.
4. Before public release, run a license audit that checks whether each output field
   contains UMLS/SNOMED source strings.

### 1.2 Can We Link CUIs?

Yes, with care. CUI identifiers are much safer than redistributing source vocabulary
strings, and the UMLS license expects applications to retain identifiers so source
provenance can be determined. However, a full public CUI-to-string mapping is still a
UMLS subset. The public release should use CUIs as opaque concept IDs or manifests, not
as a way to reconstruct restricted vocabulary content.

Recommended public format:

```json
{
  "example_id": "train_000001",
  "target_cuis": ["C0027051"],
  "source_tags": ["umls_concept_grounded", "synthetic_query"],
  "requires_umls_to_reconstruct": true
}
```

Avoid public records like:

```json
{
  "cui": "C0027051",
  "synonyms": ["... many UMLS strings ..."]
}
```

### 1.3 Is UMLS Enough, or Do We Need Separate SNOMED?

For v1, UMLS is enough.

Use UMLS files first:

- `MRCONSO.RRF`: strings, source vocabularies, CUIs, source concept IDs.
- `MRREL.RRF`: concept relationships, including hierarchical and related relations.
- `MRSTY.RRF`: semantic types for filtering and negative construction.
- `MRDEF.RRF`: definitions, but use cautiously because definitions can be source-restricted.

Why UMLS is enough:

- UMLS already contains SNOMEDCT_US entries in many distributions, subject to UMLS/SNOMED
  terms.
- `MRCONSO` maps UMLS CUIs to SNOMED source concept IDs via `SAB`, `SCUI`, and `CODE`.
- `MRREL` provides usable concept relationships for hierarchy/relatedness negatives.
- The first paper does not need a perfect SNOMED RF2 hierarchy. It needs credible,
  reproducible concept-grounded hard negatives and ablations.

When to add separate SNOMED RF2 later:

- If local UMLS lacks enough SNOMEDCT_US relationships.
- If MRREL hierarchy quality is insufficient after audit.
- If reviewers or experiments show that high-fidelity SNOMED siblings are central to
  the contribution.
- If you want a paper section specifically about SNOMED graph structure.

Fallback hierarchy sources if SNOMED is not usable:

- MeSH tree numbers for biomedical topical hierarchy.
- ICD10CM parent/child structure for diagnosis-like concepts.
- UMLS `PAR`, `CHD`, `RB`, `RN`, and source-asserted relationships.
- Same semantic type + high lexical/embedding similarity but different CUI.

---

## 2. Research Claim and Paper Positioning

### 2.1 Primary Hypothesis

MedColBERT improves medical retrieval by learning token-level concept alignments from
UMLS-grounded supervision. These alignments are most valuable when the query and relevant
document use different surface forms for the same medical concept.

Examples:

- Lay query -> clinical document.
- Abbreviation query -> expanded clinical document.
- Brand/generic drug mismatch.
- Symptom phrasing -> diagnosis or treatment evidence.
- Long clinical note -> trial eligibility text.

### 2.2 What We Must Prove

The paper should prove four things:

1. A domain clinical backbone helps.
2. Late interaction helps beyond dense pooling.
3. UMLS-guided cross-vocabulary supervision helps beyond generic medical triplets.
4. The largest gains occur in cross-vocabulary or concept-shift cases.

### 2.3 Claims to Avoid

Do not claim dense retrievers "cannot" solve synonymy. Strong dense models can learn
many synonym mappings. The defensible claim is that late interaction gives a better
inductive bias for fine-grained medical term matching and can be empirically better
when trained with the right concept-level signal.

Do not claim UMLS/SNOMED are "free" data. They are available under license and must be
handled carefully.

Do not claim the synthetic dataset is openly releasable until the license audit is done.

---

## 3. Model Architecture

### 3.1 Backbone

Default backbones:

- `thomas-sounack/BioClinical-ModernBERT-base`
- `thomas-sounack/BioClinical-ModernBERT-large`

Rationale:

- Biomedical + clinical continued pretraining.
- 8192-token context support in the encoder.
- MIT-licensed model weights.
- Strong practical starting point without spending the project budget on backbone
  pretraining.

### 3.2 Late Interaction Head

Model structure:

```text
tokenizer -> BioClinical-ModernBERT -> linear projection -> L2-normalized token vectors
```

Default projection dimension:

- `colbert_dim = 128`

MRL dimensions:

- `[32, 64, 128]`

Scoring:

```text
score(q, d) = sum_i max_j dot(q_i, d_j)
```

Implementation rule:

- Use paired/batched MaxSim during training.
- Use a ColBERT/PLAID-style index for retrieval.
- Do not brute-force a full query x all docs x all tokens tensor except in tiny tests.

### 3.3 Query and Document Lengths

Use different length regimes by task.

Default retrieval:

- `query_maxlen: 64`
- `doc_maxlen: 256`

Biomedical expert questions:

- `query_maxlen: 96`
- `doc_maxlen: 256-384`

Clinical-trials / long patient notes:

- `query_maxlen: 512`
- `long_query_maxlen: 1024` for specific TREC-CT style runs.
- `doc_maxlen: 384-512`

Reasoning:

- ColBERT storage cost scales with document tokens.
- Long queries are cheaper than long documents because queries are encoded at runtime,
  but 1024-token queries still slow MaxSim and can dilute signal.
- Use long-query mode only for benchmarks that genuinely need it, such as clinical
  trial matching from full patient descriptions.

### 3.4 Implementation Library Decision

Use PyLate as the primary training and retrieval library for v1.

Why:

- It provides ColBERT model wrappers, training losses, evaluation helpers, and PLAID
  indexing.
- It integrates with Sentence Transformers and Hugging Face tooling.
- It reduces the amount of custom MaxSim/index code needed before experiments.

Keep custom code for:

- UMLS extraction and validation.
- DSPy synthetic generation.
- Dataset construction and decontamination.
- Evaluation wrappers and cross-vocabulary analysis.
- Any model patches needed for BioClinical-ModernBERT compatibility.

Fallback:

- If PyLate cannot load BioClinical-ModernBERT correctly or cannot support required
  MRL/multi-negative behavior, use a small custom training loop for experiments and
  Stanford ColBERT/PLAID for indexing.

---

## 4. Data Strategy

### 4.1 Data Tiers

Tier A: Real supervised retrieval data

- MedEmbed training triplets, if license-compatible.
- BioASQ training splits, excluding eval years/tasks.
- TREC Clinical Trials training material, excluding eval topics/corpus leakage.
- Other non-eval biomedical retrieval datasets with clear licenses.

Tier B: UMLS concept supervision

- Cross-vocabulary term pairs.
- Abbreviation/full-form pairs.
- Brand/generic and ingredient drug pairs from RxNorm when license-safe.
- Same-CUI variants filtered for ambiguity.

Tier C: UMLS relationship hard negatives

- Parent/child near misses.
- Siblings under the same parent when reliable.
- Related-but-not-same concepts.
- Same semantic type, similar lexical/embedding neighborhood, different CUI.

Tier D: Synthetic query-passage pairs

- Passage-grounded queries generated by open-source LLMs through DSPy.
- Each generated query must target specific CUIs found in the passage.
- Each query should intentionally use alternative vocabulary where possible.

### 4.2 UMLS Extraction Filters

Input files:

- `MRCONSO.RRF`
- `MRREL.RRF`
- `MRSTY.RRF`
- Optional: `MRDEF.RRF`

Core filters:

- `LAT == ENG`
- `SUPPRESS == N`
- Remove empty, extremely short, or purely punctuation strings.
- Normalize for filtering only: lowercase, Unicode normalization, whitespace collapse,
  punctuation trimming.
- Keep original text only in private intermediate artifacts.

Source vocabulary groups:

```yaml
consumer:
  - CHV
clinical:
  - SNOMEDCT_US
  - ICD10CM
  - ICD9CM
  - MSH
  - NCI
drug:
  - RXNORM
  - VANDF
```

Do not assume every SAB is license-safe for public redistribution. SABs are for private
training pipeline control, not public data release permission.

Ambiguity filters:

- Drop normalized strings that map to too many CUIs.
- Drop abbreviations unless they are unique within the selected semantic group or
  validated by source/context.
- Drop pairs where strings are near-identical after normalization unless the pair is
  used for within-vocabulary augmentation.
- Drop concepts with semantic types outside the target biomedical/clinical groups.
- Record semantic type and source vocabulary for analysis.

Recommended semantic groups:

- Disorders.
- Procedures.
- Chemicals and drugs.
- Anatomy.
- Findings/signs/symptoms.
- Genes/proteins only if the evaluation includes biomedical literature tasks.

### 4.3 Building Positives

Do not train only on raw synonym strings. Synonym pairs are useful for warm-up, but the
final retriever must learn query-to-passage relevance.

Positive passage sources:

- PubMed abstracts.
- PMC OA paragraphs.
- MedEmbed positives.
- Clinical trial eligibility text.
- MIMIC notes only if credentialing and release constraints are handled.

Passage annotation:

- Start with dictionary/UMLS string matching for scale.
- Add abbreviation handling.
- Later, compare QuickUMLS/scispaCy only if installation and throughput are practical.
- Store `passage_id -> {cuis, matched_spans, source_doc_id}` privately.

Positive construction:

- Query targets one or more CUIs.
- Positive passage contains or is annotated with the target CUI(s).
- Prefer passages with enough context, not just a string mention.
- Avoid positives from eval corpora.

### 4.4 Negative Construction

Use four negative types:

1. BM25 negatives: lexically similar but not target CUI.
2. Concept-near negatives: related/sibling/parent concepts, different CUI.
3. Dense negatives: mined with a trained interim model.
4. Random semantic-type negatives: same semantic group, unrelated CUI, for calibration.

False-negative control:

- A negative must not share the target CUI.
- A negative should not share a close synonym-only variant unless the task is deliberately
  fine-grained.
- For multi-CUI queries, reject negatives containing any primary target CUI.
- Audit 100 examples per negative type before scaling.

### 4.5 Target Dataset Sizes

Pilot:

- 50k-100k triplets.
- Enough to test model loading, loss behavior, indexing, and first eval.

V1 base model:

- 1M-2M high-quality triplets.
- Better than 10M noisy triplets for the first serious paper run.

Full run:

- 5M-8M triplets if pilot ablations justify scale.

Recommended composition for v1:

| Source | Target |
|---|---:|
| Real curated retrieval | 300k-800k |
| UMLS warm-up pairs | 300k-1M |
| UMLS concept-grounded synthetic | 1M-3M |
| Concept-near negatives | 500k-1.5M |
| Dense-mined negatives | added after first model |

---

## 5. DSPy Synthetic Generation Plan

### 5.1 Why DSPy

DSPy gives us a structured way to turn generation into an auditable program instead of
a pile of prompts. Use it for:

- Typed generation signatures.
- Validators and suggestions/refinement.
- Prompt optimization on a small gold set.
- Usage tracking and caching.
- Swapping local/open-source LLM backends.

### 5.2 LLM Strategy

Use strong open-source LLMs strategically:

- 70B/72B-class model for seed data, prompt optimization, and quality judging.
- 8B/14B-class model for bulk generation after optimization.
- Optional medical-tuned model only if it improves factuality without copying passage
  surface forms.

Serve models through an OpenAI-compatible local endpoint such as vLLM or SGLang, then
connect DSPy through `dspy.LM`.

Generation roles:

- Patient / consumer.
- Physician.
- Nurse.
- Researcher.
- Medical student.

### 5.3 DSPy Program Shape

Signature:

```python
class GenerateMedicalQuery(dspy.Signature):
    passage: str = dspy.InputField()
    target_concepts: str = dspy.InputField()
    alternative_terms: str = dspy.InputField()
    role: str = dspy.InputField()
    forbidden_surface_terms: str = dspy.InputField()

    query: str = dspy.OutputField()
    rationale_brief: str = dspy.OutputField()
    target_cuis: str = dspy.OutputField()
```

Only `query` and `target_cuis` should flow into the dataset. The rationale is for
debugging and should not be released.

Validators:

- Query length within role-specific bounds.
- Query does not copy forbidden passage surface terms above a threshold.
- Query maps back to at least one target CUI or acceptable synonym.
- Query is not a generic medical question.
- Query does not ask for information absent from the passage.
- Query is not near-duplicate of existing queries.
- Query does not include unsafe clinical advice wording like "what should I take now"
  without context; prefer retrieval intent.

DSPy optimization:

- Build a small dev set of 200-500 manually reviewed examples.
- Metric combines concept match, lexical divergence, answerability from passage,
  role realism, and brevity.
- Optimize the DSPy program before bulk generation.
- Keep model/version/prompt/program hashes with every output.

### 5.4 Quality Gates

Automatic gates:

- CUI target coverage.
- Lexical overlap.
- MinHash near-duplicate removal.
- Embedding similarity range: not too close to passage, not unrelated.
- Toxicity/PHI pattern checks.
- Language detection.

Manual audit:

- 100 examples per role.
- 100 examples per major semantic group.
- 100 examples per negative source.
- Target acceptance rate: at least 80% for bulk generation.

If acceptance is below 80%, improve DSPy program and regenerate rather than filtering
millions of poor examples downstream.

---

## 6. Training Plan

### 6.1 Stages

Stage 0: Smoke test

- 1k examples.
- Verify BioClinical-ModernBERT loads.
- Verify PyLate ColBERT forward pass.
- Verify loss decreases for 100-1k steps.
- Verify PLAID index can retrieve a tiny corpus.

Stage 1: Concept warm-up

- Data: filtered UMLS pairs.
- Objective: contrastive same-concept alignment.
- Use in-batch negatives.
- Keep this short; do not overfit to term strings.

Stage 2: Main ColBERT training

- Data: real + synthetic + UMLS-grounded triplets.
- Objective: contrastive/pairwise ColBERT loss with hard negatives.
- Use PyLate first.
- Train base model first.

Stage 3: Dense-negative refresh

- Mine negatives using Stage 2 checkpoint.
- Rebuild high-quality triplets.
- Continue training or retrain from Stage 1 checkpoint.

Stage 4: Distillation, optional

- Teacher: strong medical cross-encoder such as MedCPT cross-encoder or a fine-tuned
  BioClinical cross-encoder.
- Use only after Stage 2 shows promise.
- Distillation is not required for the core ontology-guided claim.

Stage 5: Large model

- Train only after base ablations support the hypothesis.

### 6.2 MRL

Use MRL in Stage 2 after the basic ColBERT training loop is stable.

Dims:

- 32, 64, 128.

Weights:

- Start with `[0.1, 0.2, 0.7]`.
- Ablate equal weights and no-MRL.

### 6.3 Hyperparameter Starting Points

Base model:

```yaml
lr: 1e-5
warmup_ratio: 0.03
batch_size_per_device: 16-32
gradient_accumulation: tune_to_effective_256_or_512
bf16: true
max_steps_pilot: 5_000
max_steps_v1: 50_000-150_000
```

Large model:

```yaml
lr: 5e-6
bf16: true
gradient_checkpointing: true
train_only_after_base_success: true
```

Do not commit to 400k steps or 10M examples until pilot curves and eval justify it.

---

## 7. Evaluation Plan

### 7.1 Primary Retrieval Benchmarks

The evaluation suite must support two different claims:

1. SOTA or competitive retrieval quality on established biomedical/clinical retrieval.
2. A mechanistic claim that gains are largest under medical vocabulary mismatch.

Use a tiered benchmark suite rather than one flat list.

#### Tier 1: Main SOTA Retrieval Table

These are the benchmarks that should appear in the headline retrieval table:

- NFCorpus.
- TREC-COVID, eval only.
- BioASQ, with strict split control.
- SciFact.
- TREC Clinical Trials 2021/2022, preferably with long-query mode.
- R2MED, now publicly available.
- PMC-Patients.

Rationale:

- NFCorpus, TREC-COVID, BioASQ, and SciFact connect the paper to BEIR/MTEB-style
  biomedical retrieval reporting.
- TREC Clinical Trials tests long clinical query matching against trial documents.
- R2MED tests reasoning-driven medical retrieval and is a strong fit for the
  "not just lexical matching" argument.
- PMC-Patients adds patient/case retrieval, which is clinically closer than
  literature-only benchmarks.

#### Tier 2: Clinical Access / Optional Table

Run these if data access and protocol are practical:

- CliniQ: EHR retrieval benchmark. Highly relevant, but may involve EHR/MIMIC-style
  access constraints.
- Additional TREC Clinical Decision Support / Precision Medicine tasks if easy to
  standardize through `ir-datasets`.

These are valuable but should not block the paper.

#### Tier 3: Downstream RAG Impact Table

Run after retrieval quality is established:

- MIRAGE.
- MedRGB.

Use these to show retrieval improvements translate into medical QA/RAG gains. Do not
make them the primary SOTA claim because they entangle retriever quality with generator,
prompt, context length, and answer-scoring choices.

Metrics:

- nDCG@10.
- Recall@100.
- Recall@1000 where candidate recall matters.
- MAP@10 for R2MED if standard in their scripts.
- Latency and index size for practical comparison.
- MRL dimension tradeoff: dim=32, dim=64, dim=128.

Report separate tables:

1. Main retrieval quality: nDCG@10, Recall@100, Recall@1000.
2. Clinical/reasoning retrieval: TREC-CT, R2MED, PMC-Patients.
3. Efficiency: index size, encoding throughput, query latency, MRL dimension.
4. RAG impact: MIRAGE/MedRGB, only after retrieval tables are strong.

### 7.2 Baselines

Required:

- BM25.
- BM25 + RM3/query expansion if easy.
- BioClinical-ModernBERT dense bi-encoder.
- Generic ColBERT/PyLate baseline.
- MedCPT dense and/or cross-encoder reranker if feasible.
- MedEmbed v1 if available.
- BMRetriever.
- Strong general embedding model baselines: BGE/E5/GTE/NV-Embed class.

Baseline grouping for the paper:

- Sparse: BM25, BM25+RM3.
- General dense: strong current general embedding models.
- Biomedical dense: MedCPT, BMRetriever, MedEmbed.
- Late interaction: generic ColBERT/PyLate, BioClinical-ColBERT without UMLS.
- Reranking upper bounds: MedCPT cross-encoder or BioClinical cross-encoder.

Ablations:

- No UMLS warm-up.
- No synthetic data.
- Synthetic without UMLS constraints.
- BM25 negatives only.
- No concept-near negatives.
- No MRL.
- Base vs large.
- UMLS-constrained DSPy generation vs unconstrained DSPy generation.
- UMLS-only v1 without separate SNOMED RF2 vs optional SNOMED RF2 if added later.

### 7.3 Cross-Vocabulary Evaluation

This is the key paper contribution.

Build a query-document vocabulary divergence classifier:

- Annotate query and relevant docs with CUIs.
- Identify shared target CUIs.
- Determine whether matched surface forms come from same source group or different
  source group: consumer, clinical, drug, abbreviation, biomedical literature.
- Bucket examples:
  - same-vocab.
  - lay-to-clinical.
  - abbreviation-to-expanded.
  - brand/generic.
  - symptom-to-diagnosis/treatment evidence.
  - no direct lexical overlap.

Report metrics by bucket. The hypothesis is supported if gains are concentrated in
cross-vocabulary buckets while same-vocabulary performance remains competitive.

Recommended table:

| Bucket | BM25 | MedCPT | Generic ColBERT | BioClinical dense | MedColBERT |
|---|---:|---:|---:|---:|---:|
| Same vocabulary | | | | | |
| Lay -> clinical | | | | | |
| Abbreviation -> expanded | | | | | |
| Brand/generic | | | | | |
| Symptom -> diagnosis/treatment evidence | | | | | |
| No direct lexical overlap | | | | | |

This table is more important for the paper's novelty than a small average gain on
NFCorpus or TREC-COVID.

### 7.4 SOTA Claim Wording

Use careful wording:

- Strong: "MedColBERT achieves SOTA or competitive performance across biomedical,
  clinical, and reasoning-driven medical retrieval benchmarks, with the largest gains
  on cross-vocabulary evaluation slices."
- Avoid: "MedColBERT is SOTA on all medical retrieval."

The global claim is too broad because "medical retrieval" spans biomedical literature,
EHR retrieval, clinical trial matching, patient similarity, multimodal retrieval, and
RAG QA. The defensible claim is benchmark-scoped and mechanism-backed.

### 7.5 Benchmark Sources to Track

- BEIR/MTEB-style biomedical tasks: NFCorpus, TREC-COVID, BioASQ, SciFact.
- R2MED: `https://hf.co/papers/2505.14558` and
  `https://hf.co/datasets/R2MED/PMC-Treatment`.
- PMC-Patients: `https://hf.co/papers/2202.13876` and
  `https://hf.co/datasets/zhengyun21/PMC-Patients`.
- CliniQ: `https://hf.co/papers/2502.06252`.
- MIRAGE: `https://hf.co/papers/2402.13178`.
- MedRGB: `https://hf.co/papers/2411.09213`.
- BMRetriever baseline: `https://hf.co/papers/2404.18443`.

### 7.6 Decontamination

Use multiple checks:

- Dataset IDs and document IDs where available.
- Exact text hash.
- Normalized text hash.
- MinHash/SimHash near-duplicate detection.
- PubMed ID / PMC ID / ClinicalTrials.gov ID checks.
- Query leakage checks for synthetic data.

Rule:

- If an eval passage, title, abstract, clinical trial record, or near-duplicate appears
  in training, remove the training item.

---

## 8. Release Plan

Public release:

- Code.
- Configs.
- Evaluation harness.
- Model card with full data provenance.
- Scripts to regenerate private UMLS-derived artifacts.
- Small toy data that does not contain restricted UMLS content.
- CUI-level manifests only if they do not expose source strings.

Private/gated release:

- UMLS-derived synonym pairs.
- UMLS-derived concept annotations.
- SNOMED-derived hierarchy files.
- Any triplets with raw UMLS/SNOMED vocabulary strings.

Paper wording:

- "We use UMLS under license to generate concept-grounded supervision."
- "We release code to reproduce UMLS-derived artifacts for licensed users."
- "We do not redistribute restricted UMLS/SNOMED vocabulary content."

---

## 9. Repository Work Plan

### 9.1 Package Structure

Target structure:

```text
src/medcolbert/
  data/
    umls.py
    concepts.py
    passages.py
    negatives.py
    synthetic.py
    decontaminate.py
  generation/
    dspy_programs.py
    validators.py
    metrics.py
  training/
    pylate_train.py
    datasets.py
  eval/
    beir.py
    r2med.py
    trec_ct.py
    vocab_split.py
  utils/
    io.py
    text.py
    logging.py
```

Scripts should be thin CLI wrappers around library code.

### 9.2 First Implementation Milestone

Build the following before any large generation:

1. UMLS private HF/local loader.
2. `MRCONSO` parser with filters and ambiguity report.
3. `MRREL` concept-near negative candidate builder.
4. Passage schema and toy local corpus ingestion.
5. DSPy query generator with validators.
6. 1k synthetic triplet generation.
7. PyLate smoke training.
8. Tiny PLAID index and retrieval test.
9. Decontamination utility.
10. Audit report script.

### 9.3 CLI Commands

Planned CLI shape:

```bash
medcolbert umls extract-pairs --umls data/raw/umls --out data/processed/private/umls_pairs.parquet
medcolbert umls build-relations --umls data/raw/umls --out data/processed/private/concept_edges.parquet
medcolbert passages ingest-pubmed --input data/raw/pubmed --out data/processed/private/passages.parquet
medcolbert synth generate --config configs/generation.yaml --limit 10000
medcolbert data build-triplets --config configs/data.yaml
medcolbert train pylate --config configs/train_base.yaml
medcolbert eval beir --model checkpoints/base --dataset nfcorpus
medcolbert eval vocab-split --run runs/base_nfcorpus.jsonl
```

---

## 10. Go / No-Go Gates

Gate 1: UMLS extraction quality

- At least several hundred thousand high-confidence pairs after filtering.
- Ambiguous abbreviation rate controlled.
- Concept-near negatives pass manual audit.

Gate 2: Synthetic generation quality

- At least 80% manual acceptance on pilot.
- Lexical copying below threshold.
- Queries map to intended CUIs.

Gate 3: Model smoke test

- Loss decreases.
- No NaN/Inf.
- Tiny index retrieves positives above negatives.

Gate 4: Pilot evaluation

- Base pilot beats BM25 or generic baseline on at least one biomedical benchmark.
- Cross-vocabulary bucket shows a measurable lift.

Gate 5: Full-scale training

- Only scale to multi-million examples after Gate 4.

---

## 11. Timeline

Week 1:

- Fix environment and dependencies.
- Implement UMLS loader/filter reports.
- Confirm BioClinical-ModernBERT + PyLate smoke test.

Week 2:

- Build passage ingestion.
- Build concept annotation.
- Build UMLS relationship negatives.
- Generate and audit first 1k DSPy examples.

Week 3:

- Generate 50k-100k pilot triplets.
- Train base pilot.
- Build tiny and medium indexes.
- Run NFCorpus/BioASQ/R2MED smoke eval.

Week 4:

- Improve filters and DSPy program.
- Run ablations: no-UMLS, unconstrained synthetic, BM25-only negatives.
- Decide whether hypothesis is strong enough.

Weeks 5-7:

- Generate 1M-2M v1 triplets.
- Train base v1.
- Full eval and cross-vocabulary analysis.

Weeks 8-10:

- Dense-negative refresh.
- Train improved base.
- Decide whether to train large.

Weeks 11-14:

- Large model if justified.
- MIRAGE/RAG eval.
- Final ablations.
- Paper tables and release audit.

---

## 12. Immediate Next Actions

1. Change project Python target to 3.11/3.12 for dependency compatibility.
2. Add DSPy, PyLate, datasets, pandas, pyarrow, and evaluation extras.
3. Implement the UMLS extraction/reporting module first.
4. Generate a private audit notebook/report before training.
5. Run a 1k-example end-to-end smoke test.

Do not start 5M-query generation until the pilot data and model gates pass.
