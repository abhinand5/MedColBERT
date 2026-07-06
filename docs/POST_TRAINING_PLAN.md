# MedColBERT Post-Training Plan

**Status:** MedColBERT-base trained and gated; MedColBERT-large training
in progress. This document is the operating plan for everything that follows
training completion through paper submission and public release.

**Scope:** Phases 7 (Evaluation and Claims) and 8 (Release and Paper) from
[GOALS.md](GOALS.md), plus all cross-cutting work that bridges them.

**Primary principle:** no claim without evidence, no evidence without a
baseline control, no baseline control without a comparable compute budget.

**Venue reality (see section 14 for the full ladder):** This is a
recipe paper, not a method paper. The upper bound is **SIGIR main track**
if (and only if) MedColBERT tops at least one public MTEB medical retrieval
leaderboard *and* the no-ontology ablation isolates the mechanism. The
realistic floor is **arXiv + a workshop** (BioNLP / ClinicalNLP). The
deciding experiment is **MTEB / R2MED public retrieval benchmarks**, run
first — before the no-ontology data generation — because the public
leaderboard result determines whether the rest of the plan is worth
executing at the "ECIR / CIKM / SIGIR" level or the "workshop" level.

---

## Table of Contents

1. [Where we are now](#1-where-we-are-now)
2. [Public retrieval benchmarks — the venue-determining experiment](#2-public-retrieval-benchmarks--the-venue-determining-experiment)
3. [Critical path: the no-ontology control](#3-critical-path-the-no-ontology-control)
4. [Required baseline table](#4-required-baseline-table)
5. [External benchmark suite](#5-external-benchmark-suite)
6. [Vocabulary-shift slice evaluation](#6-vocabulary-shift-slice-evaluation)
7. [Decontamination report](#7-decontamination-report)
8. [Efficiency table](#8-efficiency-table)
9. [Additional model variants (optional, deferred)](#9-additional-model-variants-optional-deferred)
10. [Release artifacts and audit](#10-release-artifacts-and-audit)
11. [Paper evidence package](#11-paper-evidence-package)
12. [Concrete task breakdown and ordering](#12-concrete-task-breakdown-and-ordering)
13. [Risks and gates](#13-risks-and-gates)
14. [Venue targeting ladder](#14-venue-targeting-ladder)
15. [Artifacts manifest](#15-artifacts-manifest)

---

## 1. Where we are now

### Done

- **Phase 1-5:** UMLS ontology layer, public corpus registry, synthetic query
  generation, hard-negative triplet mining, and the standalone Hugging Face
  dataset `fierysurf/medcolbert-training-v1` (711,301 rows, 4,273 CUIs,
  17,262 unique positives, 66,108 corpus passages, 5,967 dev queries).

- **Phase 6 (base):** `medcolbert-base-run0` trained 5 epochs
  CachedContrastive on one L40S, 14h40m. Final `train_loss=0.9872`. Loss
  plateaued at epoch 3-4 (avg 0.023 → 0.020). No divergence. All 5 kept
  checkpoints evaluated against the full 66,108-passage PubMed corpus /
  5,967 dev queries.

- **Real-corpus retrieval gate (base):** PASSED.
  - Recall@100 = 0.971 vs BM25 baseline 0.626 (+55% relative)
  - MRR@10 = 0.714, nDCG@10 = 0.754
  - `final` and `epoch4` statistically tied; `final` shipped as canonical
  - Full table in `docs/phase6-training-results.md`
  - Per-checkpoint metrics: `runs/base_stage2/eval_epoch{1,2,4,5}.json`,
    `runs/base_stage2/eval_final.json`, `runs/base_stage2/eval_comparison.txt`
  - PLAID indexes: `runs/base_stage2/eval_index_epoch{1,2,4,5,final}/`

- **Phase 6 (large):** `medcolbert-large-run0` in progress on the L40S.
  - Backbone: `thomas-sounack/BioClinical-ModernBERT-large` (396M params)
  - Config: `configs/train_large.yaml` + CLI overrides (`mini_batch_size=96`)
  - Step time ~7.45 s, ETA ~28h total
  - Loss trajectory mirrors base (epoch 0 avg 7.91 → epoch 1 avg 0.105)
  - See `docs/plans/large-model-training.md` for the full plan

- **Software:** `src/medcolbert/training/{datasets,pylate_train}.py`,
  `src/medcolbert/eval/real_corpus.py`, the training and eval CLI scripts,
  the batch-eval harness, and 35/35 unit tests are committed and lint-clean.
  Wandb integration is live (`WANDB_API_KEY` gated).
  Base run logged at `https://wandb.ai/abhinandb/MedColBERT/runs/zhepk234`.

### Not done (everything this plan covers)

- **Public retrieval benchmarks** (MTEB Medical, R2MED, RTEB-Health) —
  the venue-determining experiment, run **first** (section 2)
- No-ontology ColBERT control (mandatory ablation)
- Required baseline suite (BM25+RM3, strong dense, MedCPT, BMRetriever,
  generic ColBERT)
- Other external benchmarks (TREC Clinical Trials, PMC-Patients, BioASQ)
- Vocabulary-shift slice analysis
- Decontamination report
- Efficiency table
- Model card, memorization audit, release audit
- Paper evidence package

---

## 2. Public retrieval benchmarks — the venue-determining experiment

### Why this runs first

The internal real-corpus gate (0.971 vs BM25 0.626) is a sanity check,
not a result. It compares against BM25 only, on a held-out split of our
own synthetic generator, with bucket labels we generated ourselves. To
know whether MedColBERT is a real result or a fine-tuning artifact, we
need **public, third-party benchmarks with established leaderboards**.

The [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
maintains a **Medical** subdomain benchmark (`MTEB(Medical, v1)`) that
aggregates 12 medical tasks across retrieval, clustering, and reranking.
Sibling benchmarks **R2MED** (8 reasoning-driven medical retrieval tasks)
and **RTEB(Health, beta)** (4 healthcare retrieval tasks) are pure
retrieval and directly comparable to ColBERT.

Running these is **the highest-information-per-hour experiment in the
plan**. Within a few hours of GPU time on one L40S, we will know whether
MedColBERT-base is fighting for a SIGIR main-track paper or a workshop
paper. This decision drives every downstream priority.

### Architectural limitation — ColBERT cannot top MTEB(Medical, v1)

ColBERT produces **token-level** embeddings, not a single pooled
sentence vector. This means it can only compete on **retrieval** tasks.
The 12 tasks in MTEB(Medical, v1) break down as:

| Task | Type | ColBERT eligible? |
|---|---|---|
| CUREv1 | Retrieval (eng/fra/spa) | ✅ English split only |
| NFCorpus | Retrieval (eng) | ✅ |
| TRECCOVID | Retrieval (eng) | ✅ |
| TRECCOVID-PL | Retrieval (pol) | ❌ English-only backbone |
| SciFact | Retrieval (eng) | ✅ |
| SciFact-PL | Retrieval (pol) | ❌ |
| MedicalQARetrieval | Retrieval (eng) | ✅ |
| PublicHealthQA | Retrieval (8 langs) | ❌ English-only |
| MedrxivClusteringP2P.v2 | Clustering | ❌ No pooled vector |
| MedrxivClusteringS2S.v2 | Clustering | ❌ No pooled vector |
| CmedqaRetrieval | Retrieval (cmn) | ❌ English-only |
| CMedQAv2-reranking | Reranking | ❌ Not a cross-encoder |

**We cannot top the aggregate MTEB(Medical, v1) leaderboard.** MTEB
treats missing tasks as 0, and 0s on 5-6 tasks sink the category average
regardless of how strong our retrieval scores are.

We can compete on:
1. **The English retrieval subset** of MTEB(Medical, v1): NFCorpus,
   TRECCOVID, SciFact, MedicalQARetrieval, CUREv1 (English split) — 5
   tasks. Report these as a per-task table in the paper.
2. **R2MED** — 8 tasks, all English, all retrieval, all medical
   reasoning. This is the gold target. SOTA on R2MED is a standalone
   result.
3. **RTEB(Health, beta)** — 4 retrieval tasks (ChatDoctor, CUREv1,
   EnglishHealthcare, GermanHealthcare — we take the English ones).

### What we're competing against

The medical retrieval leaderboards are dominated by **7B+ parameter
models** (NV-Embed-v2, voyage-3-large, bge-multilingual). A 150M or 396M
ColBERT will not beat them on raw aggregate score across all tasks.
Realistic per-task targets:

| Task | Realistic target | Why |
|---|---|---|
| NFCorpus | Plausible top — small medical queries, short abstracts; ColBERT-style late interaction is historically strong here | |
| SciFact | Plausible top-3 — claim verification, abstracts, short queries | |
| TRECCOVID | Competitive but tough — established benchmark, many strong entries | |
| MedicalQARetrieval | Tough — 2048 QA pairs, dense models often win on QA-style | |
| CUREv1 | Plausible — newer benchmark, less saturated | |
| R2MED (8 tasks) | Variable — reasoning-driven; this is where the ontology recipe *could* pay off most | |

### Implementation

Use the `mteb` Python package directly:

```python
import mteb
from pylate import models

model = models.ColBERT("runs/base_stage2/final")
tasks = mteb.get_tasks(tasks=[
    "NFCorpus", "TRECCOVID", "SciFact", "MedicalQARetrieval",
    "CUREv1",  # MTEB(Medical, v1) English retrieval subset
    # R2MED tasks:
    "R2MEDBiologyRetrieval", "R2MEDBioinformaticsRetrieval",
    "R2MEDMedicalSciencesRetrieval", "R2MEDMedXpertQAExamRetrieval",
    "R2MEDMedQADiagRetrieval", "R2MEDPMCTreatmentRetrieval",
    "R2MEDPMCClinicalRetrieval", "R2MEDIIYiClinicalRetrieval",
])
results = mteb.evaluate(model, tasks=tasks, output_folder="runs/eval/mteb_base")
```

ColBERT is not a native MTEB model wrapper. We may need a thin adapter
(`src/medcolbert/eval/mteb_adapter.py`) that wraps a pylate ColBERT
model in MTEB's `Encoder` protocol so `mteb.evaluate` can call it. This
is the one piece of new code this phase requires — see task M1 in
section 12.

### Decision gate after public benchmarks

| Outcome | Implication |
|---|---|
| MedColBERT-base tops 1+ medical retrieval task AND is top-3 on 3+ others | Push for SIGIR / ACL main track. The MTEB leaderboard is the headline evidence. Proceed with no-ontology ablation + vocab-shift as the mechanism story. |
| MedColBERT-base is top-5 on 2+ tasks, competitive on the rest | ECIR / CIKM / Findings target. The public numbers are real but not dominant — the ablation and vocab-shift analysis must carry the mechanism story. |
| MedColBERT-base is "competitive but not best" everywhere | Workshop / arXiv. The internal gate was a fine-tuning artifact, not a research result. Still publishable as a systems paper. |
| MedColBERT-base underperforms BGE-large or MedCPT on the public benchmarks | The internal gate does not generalize. Pivot immediately to diagnosing why (corpus overlap? domain mismatch?) before investing further in ablations. |

This gate must be decided **before** the no-ontology data generation
(Phase A in the old plan), because if the public number is dead, the
ablation doesn't matter for venue purposes. The no-ontology control is
still needed for the *mechanism* claim, but its priority relative to
other work depends on the venue target.

---

## 3. Critical path: the no-ontology control

### Why this is the single most important *mechanism* deliverable

The core MedColBERT research claim, from [GOALS.md](GOALS.md), is:

> Real-corpus, ontology-controlled synthetic supervision improves medical
> retrieval when queries and relevant documents express the same concept
> with different vocabulary.

The mandatory ablation that isolates this claim is: train
**BioClinical-ColBERT-no-ontology** with the same backbone, same compute
budget, same eval pipeline, but **without** the ontology-controlled
synthetic supervision. AGENTS.md is explicit:

> The BioClinical-ColBERT without ontology supervision baseline is
> mandatory. It separates the value of the backbone from the value of the
> ontology-controlled training recipe.

Without this control, the project has a fine-tuned model, not a research
result. The 0.971 Recall@100 could be entirely attributable to the
BioClinical-ModernBERT backbone being strong, not to the ontology recipe.

**Note on ordering:** This section was previously marked as the
critical-path-first deliverable. The MTEB insight reorders it: public
benchmarks (section 2) run first because they determine the venue, and
the venue determines how much to invest in the mechanism story. The
no-ontology control is still mandatory for any mechanism claim, but it
is now sequenced *after* the public benchmark gate so we don't spend
~14h of GPU time on a control whose framing depends on a public result
we haven't measured yet.

### What "no-ontology" means concretely

The current v1 triplets (`fierysurf/medcolbert-training-v1`, `long`
config) were built with:
- Target CUIs sampled from the UMLS concept layer
- Synonym / abbreviation / brand-generic surface forms drawn from UMLS
  source strings under CUI control
- Hard negatives mined from BM25 + concept-near + dense
- Each triplet tagged with `vocab_shift_type`, `target_cuis`,
  `semantic_groups`, `generation_mode`

The no-ontology control must:
- Use **real PubMed passages** as the positive corpus (same as v1)
- Generate queries with an LLM **without CUI grounding, synonym tables,
  or semantic-type controls** — i.e. generic "ask a question this passage
  answers" prompts
- Mine hard negatives with the same BM25 + dense strategy (so the only
  variable is the *query generation* signal, not the negative quality)
- Match the v1 dataset size, passage distribution, and epoch budget

This isolates exactly one variable: whether ontology control over the
*query side* of the triplet adds signal, holding the passage side,
negatives, and compute fixed.

### Proposed implementation

Build a `fierysurf/medcolbert-training-v1-no-ontology` dataset with the
same shape as `medcolbert-training-v1` (columns: `query`, `positive`,
`negative`, plus audit metadata), but with queries generated from a
generic prompt template:

```
You are a medical information-seeker. Write a realistic question that a
patient, nurse, or physician might ask whose answer is contained in the
following passage. Use natural language; do not copy phrases from the
passage.

Passage: {passage_text}

Question:
```

No CUI targeting, no synonym substitution, no abbreviation/brand-generic
shift buckets, no semantic-type filtering. The LLM is free to generate
any surface form. This is the "backbone + generic synthetic supervision"
control.

Negatives: reuse the v1 hard-negative passage pool for each positive
(so the negative-distribution variable is held constant). If the pool
cannot be matched by positive_id (different dataset), mine fresh BM25 +
dense negatives with the same script and same parameters as v1.

Target size: 711,301 rows to match v1 exactly, or a matched subset.

### Training command (skeleton)

```bash
uv run python scripts/training/stage2_colbert_finetune.py \
  --train-config configs/train_base.yaml \
  --model-config configs/base.yaml \
  --run-name medcolbert-no-ontology-run0 \
  --batch-size 256 \
  --mini-batch-size 160 \
  --no-gradient-checkpointing \
  --max-steps 14000 \
  --save-strategy epoch \
  --save-total-limit 3 \
  --output-dir runs/no_ontology_stage2 \
  2>&1 | tee logs/medcolbert-no-ontology-run0.log
```

The only difference from the actual base command is `--output-dir` and
`--run-name`, because the training script loads from
`fierysurf/medcolbert-training-v1` by default. To point it at the
no-ontology dataset, either:

- (a) Create a `configs/train_no_ontology.yaml` that sets a different
  `train_path`, and pass `--train-config configs/train_no_ontology.yaml`;
  or
- (b) Add a `--hf-dataset` flag to `stage2_colbert_finetune.py` that
  overrides the hardcoded `fierysurf/medcolbert-training-v1` repo id.
  This is a five-line change and avoids duplicating the config.

Prefer (b) — a CLI flag is cleaner than a config fork. This is a concrete
code task in section 12.

### Decision gate after no-ontology run

After the no-ontology model is evaluated on the same real-corpus gate:

| Outcome | Implication |
|---|---|
| MedColBERT-base beats no-ontology by a clear margin on Recall@100 and MRR@10 | Mechanism claim supported. Proceed to vocab-shift analysis, baselines, paper. |
| No-ontology matches or beats MedColBERT | Backbone is what matters; the ontology recipe is not the source of gain. Pivot framing to "BioClinical-ModernBERT as a strong medical retriever backbone, with a recipe and suite that others can reuse." Still publishable as a systems paper. |
| No-ontology is close but MedColBERT wins specifically on vocab-shift buckets | Strongest possible result. Proceed with the vocab-shift analysis as the headline claim. |

This decision must be made before writing the paper, and (per AGENTS.md)
before claiming any mechanism-level improvement.

---

## 4. Required baseline table

### Source: AGENTS.md and docs/goals/07-evaluation-and-claims.md

The required baselines, in priority order:

| # | Baseline | Type | Status | What it isolates |
|---|---|---|---|---|
| 1 | BM25 | sparse | DONE (R@100=0.626) | Lexical floor |
| 2 | BM25 + RM3 (or query expansion) | sparse | TODO | How much does lexical expansion alone close the gap? |
| 3 | Strong general embedding model (e.g. bge-large-en-v1.5, GTE-large) | dense | TODO | General-domain dense floor |
| 4 | MedCPT (or another biomedical dense baseline) | dense | TODO | Domain-specific dense; reported as a baseline, not an oracle (per AGENTS.md) |
| 5 | BMRetriever (when practical) | dense | TODO | Another biomedical dense baseline |
| 6 | Generic ColBERT / PyLate (off-the-shelf, no fine-tuning) | late interaction | TODO | Does late interaction alone help, without medical fine-tuning? |
| 7 | BioClinical-ColBERT-no-ontology | late interaction | TODO (section 3) | **Mandatory** — isolates the ontology recipe from the backbone |
| 8 | MedColBERT-base | late interaction | DONE | The proposed method |
| 9 | MedColBERT-large | late interaction | IN PROGRESS | Scaling claim |
| 10 | BM25 + MedColBERT hybrid via RRF | hybrid | OPTIONAL | Does fusion help? |

### Implementation notes per baseline

- **BM25 + RM3:** use Pyserini or rank_bm25 + an RM3 expansion. Use the
  same corpus (66,108 passages) and the same 5,967 dev queries.
- **bge-large / GTE-large:** sentence-transformers `models.SentenceTransformer`
  with mean pooling. Build a dense index + dot product. Same eval script
  shape as the real-corpus eval (`src/medcolbert/eval/real_corpus.py`)
  but swap PLAID for a flat FAISS index or numpy dot-product.
- **MedCPT:** `cambridgeltl/Sap-CBERT` or the official MedCPT article
  encoder. Use the same eval harness.
- **BMRetriever:** Hugging Face `Snowflake/snowflake-arctic-embed-l`
  or the BMRetriever checkpoint, depending on availability. Use the
  same eval harness.
- **Generic ColBERT:** `lightonai/colbertv2.0` or the default pylate
  ColBERT checkpoint. Build a PLAID index over the same 66k corpus and
  retrieve. No fine-tuning.
- **All baselines must use the same eval corpus and the same dev queries.**
  This is a non-negotiable control.

### Decision: build a unified eval harness

The current `scripts/eval/run_real_corpus.py` is ColBERT/PLAID-specific
(it calls `pylate.models.ColBERT` and `pylate.indexes.PLAID`). To run
dense baselines through the same harness, extend it (or add a sibling
script `scripts/eval/run_dense_baseline.py`) that accepts a
`sentence-transformers` model name, encodes the corpus, and retrieves
via cosine / dot-product top-k, then calls the same
`aggregate_metrics` function from `src/medcolbert/eval/real_corpus.py`.
The metric functions are already pure and model-agnostic; only the
encode-and-retrieve step differs.

This is a concrete code task in section 12.

---

## 5. External benchmark suite

### Why we need this

The real-corpus gate (PubMed 66k, 5,967 dev queries) is a held-out
internal validation. External benchmarks are needed for two reasons:
1. They give published numbers to compare against (the "competitive or
   SOTA" claim from docs/goals/07-evaluation-and-claims.md)
2. They test generalization beyond the PubMed corpus the training data
   was built on

### Benchmark list (from configs/eval.yaml)

| Benchmark | Tier | Question | Status |
|---|---|---|---|
| NFCorpus | primary_biomedical | Patients ask nutrition-related queries; find relevant abstracts | TODO |
| TREC-COVID | primary_biomedical | Scientific queries against COVID-19 literature | TODO |
| SciFact | primary_biomedical | Verify scientific claims against a corpus | TODO |
| BioASQ | primary_biomedical | Biomedical QA and retrieval | TODO (strict_split_control) |
| TREC Clinical Trials | primary_clinical | Match patient queries to clinical trial descriptions | TODO |
| R2MED | primary_reasoning | Reasoning-driven medical retrieval over PMC-Treatment | TODO |
| PMC-Patients | primary_clinical | Patient case retrieval | TODO |
| MIRAGE | downstream_rag | Skip in first pass | DEFERRED |
| MedRGB | downstream_rag | Skip in first pass | DEFERRED |

### Implementation

`src/medcolbert/eval/beir.py` is a planned stub (the architecture is in
AGENTS.md). Build a thin BEIR-compatible runner that:
1. Loads each benchmark via `beir.datasets` or direct HF download
2. Uses the benchmark's own qrels and corpus
3. Encodes + retrieves with a given model (ColBERT or dense)
4. Reports the benchmark-standard metrics (nDCG@10, Recall@100,
   Recall@1000, MAP@10)
5. Writes a per-benchmark JSON to `runs/eval/<benchmark>_<model>.json`

Reuse the pure metrics from `src/medcolbert/eval/real_corpus.py` for
consistency. The benchmark loader is the only new code.

### Length modes

`configs/eval.yaml` defines four length modes because ModernBERT
supports up to 8192 tokens. ClinicalTrial and PMC-Patients queries are
long and need `query_maxlen=512` or `1024`. The eval harness must respect
these per-benchmark length overrides, otherwise long clinical queries
will be truncated and retrieval quality will look worse than it is.

### Decontamination control before external eval

TREC-COVID and BioASQ both have `train_contamination_block: true` and
`strict_split_control: true` in `configs/eval.yaml`. Before reporting
external-benchmark numbers, verify that the benchmark corpus passages
do not overlap with the MedColBERT training corpus (the 66,108 PubMed
passages the v1 triplets were built on). Run a decontamination check
(section 7) and exclude any contaminated eval queries from the reported
numbers, with a count of how many were excluded.

---

## 6. Vocabulary-shift slice evaluation

### Why this is the spine of the paper

The real-corpus gate showed massive gains over BM25 (0.971 vs 0.626
Recall@100). The vocab-shift analysis is what distinguishes the MedColBERT
research contribution from a generic fine-tuning result. The claim
(again, from GOALS.md):

> The strongest claim is not "We trained ColBERT on medical text."
> The strongest claim is that ontology-controlled synthetic supervision
> improves retrieval when queries and relevant documents express the
> same concept with different vocabulary.

The evidence for that claim is the per-bucket retrieval table, not the
aggregate score.

### Buckets (from configs/eval.yaml and docs/goals/07)

1. `same_vocab` — query and positive use the same surface terms
2. `lay_to_clinical` — layperson wording → clinical terminology
3. `abbreviation_to_expanded` — abbreviation → expanded term
4. `brand_generic` — brand drug → generic / ingredient
5. `symptom_to_diagnosis_or_treatment` — symptom phrase →
   diagnosis or treatment evidence
6. `biomedical_to_clinical` — biomedical literature phrasing →
   clinical phrasing
7. `no_direct_lexical_overlap` — no shared surface terms at all

### Data source

The v1 dataset already has `vocab_shift_type` tagged on every row (this
is one of the required metadata fields listed in AGENTS.md). The 5,967
dev queries therefore carry bucket labels. The slice eval is:

1. Load dev queries (already loaded by `load_dev_queries`)
2. Group by `vocab_shift_type`
3. For each bucket, compute MRR@10, Recall@10, Recall@100, nDCG@10
   against the same 66k corpus PLAID index already built per model
4. Write `runs/eval/vocab_shift_table.json` keyed by
   `{model, bucket, metric}`

### Key result pattern

The desired result (which would justify the strongest claim):

- MedColBERT beats BM25 and the no-ontology control **most** on
  `lay_to_clinical`, `abbreviation_to_expanded`, `brand_generic`,
  `symptom_to_diagnosis_or_treatment`, `no_direct_lexical_overlap`
- MedColBERT is **competitive** (not necessarily best) on `same_vocab`
- The no-ontology control is strong on `same_vocab` (where surface match
  is easy) but weak on the shift buckets (where it had no signal to
  learn the mapping)

If MedColBERT's gains are uniform across buckets, the ontology recipe is
not specifically responsible — a generic fine-tuning signal would also
produce uniform gains. The bucket-concentrated gain pattern is what
supports the mechanism claim.

### Mini MedVocabShift benchmark (conditional)

If the existing dev buckets are too small or too noisy to support a
bucket-level claim, build a small public benchmark:

- 500-2,000 queries
- Documents must be public PubMed passages (already available)
- Labels auditable, examples bucketed by shift type
- Do not train on this benchmark
- UMLS-derived query strings used during construction must stay private;
  only the public-safe query text is released
- Release rules in docs/goals/07 §Mini MedVocabShift Benchmark

This is conditional on the internal slice analysis being underpowered.
Defer until the first slice table is computed.

---

## 7. Decontamination report

### Why

AGENTS.md and docs/goals/07 require that "all reported eval corpora are
decontaminated from training." Reviewer question (Phase 8): "How was
train/test contamination prevented?"

### What to check

For every eval set (the 5,967 dev queries against the 66k corpus, plus
every external benchmark corpus), verify that:

1. **No eval corpus passage appears in the training corpus.** The
   training corpus is the 66,108 PubMed passages in
   `data/processed/private/passages/real_annotated_500.json`. Check for
   exact-match and near-duplicate overlap against each eval corpus using
   the existing `data/processed/private/passages/fingerprints.parquet`
   (MinHash signatures) — `src/medcolbert/data/decontaminate.py` does
   not exist yet; build it.

2. **No dev query appears as a training query.** The dev queries are
   held out from the v1 dataset split and should not appear in the
   `train` config. Verify by hashing the 5,967 dev query strings against
   the `train` config queries.

3. **External benchmark corpora do not overlap with the training
   corpus.** Run the same fingerprint match for NFCorpus, SciFact,
   TREC-COVID, etc. against the 66k training corpus.

### Implementation

A new module `src/medcolbert/data/decontaminate.py` with:
- `load_fingerprints(path: Path) -> dict` — load MinHash signatures
- `match_passages(eval_passages, train_fingerprints, threshold=0.8) -> list[dict]`
  — return matches with jaccard similarity and the matched passage ids
- `decontaminate_eval(eval_rows, train_corpus, threshold=0.8) -> tuple[list, list]`
  — return (kept, removed) and write a JSON report

Call this from `scripts/eval/run_real_corpus.py` and the external
benchmark runner before reporting any numbers. Write
`runs/eval/decontamination_summary.json` with counts of:
- Eval passages checked
- Train fingerprints checked against
- Matches found (with passage ids and jaccard scores)
- Dev queries checked
- Train queries checked against
- Overlap queries found

### Acceptance

Decontamination passes if 0 exact-match and <0.1% near-duplicate
overlap. Any contaminated examples are excluded from the reported
metrics with a documented count.

---

## 8. Efficiency table

### Required (docs/goals/07)

- Index size on disk
- Query latency (p50, p95, p99) over the 5,967 dev queries against the
  66k corpus
- Encoding throughput (queries/sec, docs/sec) for each model
- MRL dim 32 / 64 / 128 tradeoff if MRL is enabled (deferred — see
  section 9)

### What to measure

For each model variant of interest (MedColBERT-base, MedColBERT-large,
no-ontology control, MedCPT, generic ColBERT), measure:

| Metric | How |
|---|---|
| Index size | `du -sh runs/<model>/eval_index_final/` after the PLAID index build |
| Query encoding throughput | `model.encode(queries, is_query=True)`, time it |
| Doc encoding throughput | `model.encode(corpus, is_query=False)`, time it |
| Retrieval latency p50/p95/p99 | per-query `retriever.retrieve(q_embs[i:i+1], k=100)` with `time.perf_counter` |

### Implementation

Add a `scripts/eval/efficiency.py` that:
1. Takes a `--model` path and an `--index-dir`
2. Loads the existing PLAID index (reuse `--no-override-index`)
3. Times encoding and retrieval, writes `runs/eval/efficiency_<model>.json`

This is mechanical and should be done after all retrieval evals so the
indexes already exist.

---

## 9. Additional model variants (optional, deferred)

docs/goals/06 lists five required model variants. The current
state covers #3 (`medcolbert_ontology_grounded` — DONE) and will cover
#2 (`bioclinical_colbert_no_ontology` — section 3). The remaining
variants are optional and should be deferred until after the base
ablations support the hypothesis.

| Variant | Status | When to do |
|---|---|---|
| `generic_colbert` | TODO (baseline table) | Section 3 — no training, just evaluate an off-the-shelf ColBERT |
| `bioclinical_colbert_no_ontology` | TODO (section 3) | Critical path — do first |
| `medcolbert_ontology_grounded` | DONE (base) + IN PROGRESS (large) | — |
| `medcolbert_teacher_filtered` | DEFERRED | Only if the ontology grounding shows a clear win and we want to isolate the LLM-reranker contribution |
| `medcolbert_dense_negative_refresh` | DEFERRED | Only if the first gate underperforms and v2 data is needed |
| MRL (dim 32/64/128) | DEFERRED (config has it but pylate does not support it) | Efficiency paper section only; not needed for the core claim |

Per AGENTS.md: "Large model training is optional and should happen only
after base model ablations support the hypothesis." The large run is
already in progress, so the order is inverted; that is a deliberate
decision by the user, and the large model will be reported as a scaling
ablation rather than the primary result.

---

## 10. Release artifacts and audit

### Source: docs/goals/08-release-and-paper.md

### Public-safe artifacts (may release)

- Source code (already public-safe in the repo)
- Configs (already public-safe)
- Evaluation harness (`src/medcolbert/eval/`, the scripts)
- Public-safe manifests (passage counts, hashes, source names — no
  restricted strings)
- Reproduction scripts that require licensed users to provide their own
  UMLS access
- Toy examples with no restricted vocabulary content
- Model card (provenance, training data summary, limitations, safety)
- Model weights — only after a license and memorization review
- Benchmark run files where licenses allow

### Private-only artifacts (must not release)

Per AGENTS.md §Data Privacy Rules:
- Raw UMLS files (`MRCONSO.RRF`, `MRREL.RRF`, `MRSTY.RRF`, `MRDEF.RRF`)
- CUI-to-string tables
- Synonym tables
- Matched spans containing UMLS source strings
- SNOMED hierarchy dumps
- UMLS-derived triplets where query or passage text is copied from
  restricted vocabulary strings at scale
- Audit files containing restricted terms

### Model card

Required sections (from docs/goals/08 §Model Card Requirements):
1. Intended use
2. Out-of-scope use
3. Training data summary (size, source, synthetic generation mode
   distribution) — without exposing restricted strings
4. Private data policy (what is and is not released, and why)
5. Synthetic data generation summary (prompt template structure,
   generator model, validator version)
6. Benchmark results (link to the eval tables)
7. Known limitations (corpus coverage, vocabulary shift types not
   covered, English-only, not a clinical decision tool)
8. Safety and clinical-use disclaimer
9. License and provenance notes
10. Memorization and restricted-string audit summary

### Memorization audit

Before releasing the model weights, scan the trained model for
memorization of restricted UMLS/SNOMED strings:
1. Sample 10,000 random passages from the training corpus and probe the
   model for high-confidence continuations
2. Check whether the model reproduces UMLS source strings verbatim
3. Run the restricted-string scanner (currently a stub) over the model
   weights and tokenizer vocabulary for suspicious n-grams
4. Document findings in `runs/release/memorization_audit.json`
5. If memorization is detected, do not release weights publicly; gate
   release or apply differential filtering

### Release audit checklist (from docs/goals/08 §Release Audit)

1. Scan public files for private path leaks (grep for
   `/workspace/`, `/data/processed/private/`, `MRCONSO`, `MRREL`, etc.)
2. Scan public files for UMLS-derived string tables
3. Verify `.gitignore` blocks private data and checkpoints
4. Verify model card provenance is complete
5. Verify eval run files do not contain restricted strings
6. Verify generated examples intended for release are public-safe
7. Record review outcome in a release checklist at
   `docs/release_checklist.md`

---

## 11. Paper evidence package

### Source: docs/goals/08 §Paper Evidence Package

The paper is the last step. All evidence must exist before writing
begins.

### Tables to prepare

1. **Main benchmark table** — every model on every external benchmark
   with nDCG@10, Recall@100, MAP@10. Source: `runs/eval/main_table.json`
2. **Vocabulary-shift table** — per-bucket MRR@10, Recall@10/100 for
   MedColBERT, no-ontology control, BM25 on the internal gate.
   Source: `runs/eval/vocab_shift_table.json`
3. **Ablation table** — MedColBERT vs no-ontology vs generic ColBERT vs
   BM25 on the internal real-corpus gate. Highlight the gap.
   Source: `runs/base_stage2/eval_comparison.txt` extended with the
   no-ontology row
4. **Efficiency table** — index size, latency p50/p95/p99, encoding
   throughput per model. Source: `runs/eval/efficiency_table.json`
5. **Data quality audit table** — counts from the triplet audit
   (already rendered in `data/processed/private/synthetic/consolidated/dataset_stats.json`
   and the audit docs under `docs/audit_*`)

### Qualitative artifacts

- **MaxSim alignment examples.** For 10-20 dev queries where MedColBERT
  wins big over BM25 and over the no-ontology control, dump the
  query-token to passage-token MaxSim alignment (e.g. layperson query
  token "heart attack" aligning to passage token "myocardial
  infarction"). This is a pylate / ColBERT visualization task.
- **Failure cases.** 5-10 queries where MedColBERT fails to retrieve
  the positive in top-100, with analysis of why.

### Reviewer questions to preempt

From docs/goals/08 §Reviewer Questions To Preempt. Each must be
answered explicitly in the paper:

1. Why late interaction instead of dense pooling?
2. Why is synthetic supervision not just prompt-artifact learning?
3. Why do UMLS/SNOMED/RxNorm controls improve over generic synthetic data?
4. How was train/test contamination prevented?
5. Are restricted ontology strings redistributed?
6. Does the model memorize restricted vocabulary content?
7. Does performance improve on real benchmarks, not only synthetic slices?
8. How expensive is indexing and querying?

### Claim wording guardrails (from docs/goals/07)

Acceptable: "MedColBERT achieves competitive or state-of-the-art results
on selected biomedical and clinical retrieval benchmarks, with the
largest gains on vocabulary-shift slices."

Avoid: "MedColBERT is SOTA on medical retrieval."
Avoid: "Dense retrieval cannot solve medical synonymy."

---

## 12. Concrete task breakdown and ordering

The tasks below are ordered by dependency, not by phase. Each task is
meant to be one manageable unit of work.

### Phase M — Public retrieval benchmarks (FIRST — venue-determining)

**M1. Build the MTEB ColBERT adapter**

`src/medcolbert/eval/mteb_adapter.py` — wrap a pylate `models.ColBERT`
in MTEB's `Encoder` protocol so `mteb.evaluate` can call it. The adapter
must:
1. Accept `model.encode(texts, prompt_name="query")` and
   `model.encode(texts, prompt_name="passage")` (MTEB's retrieval task
   protocol distinguishes query vs passage encoding)
2. Return a single vector per text (for dense comparison) **or** a list
   of token vectors (for ColBERT late interaction — this is the path
   MTEB's PLAID/Jina-ColBERT wrappers use)
3. Handle the length-mode overrides from `configs/eval.yaml`

Check whether MTEB already has a ColBERT model wrapper we can subclass
(`mteb.models.ColBERTWrapper` or similar). If not, the adapter is
~50 lines.

**M2. Run MedColBERT-base on MTEB Medical retrieval subset + R2MED**

```bash
uv run python scripts/eval/run_mteb.py \
  --model runs/base_stage2/final \
  --tasks medical_retrieval,r2med \
  --output-dir runs/eval/mteb_base
```

Targets: NFCorpus, TRECCOVID, SciFact, MedicalQARetrieval, CUREv1
(MTEB Medical English retrieval subset) + 8 R2MED tasks. Writes one
JSON per task to `runs/eval/mteb_base/`.

**M3. Run MedColBERT-large on the same tasks** (after large training
completes).

**M4. Run at least one dense baseline on the same tasks** — BGE-large-en-v1.5
and MedCPT. This is the comparison row that tells us whether the public
ranking is real.

**M5. Pull current leaderboard top-5 per task** from the live MTEB
leaderboard (or via the `mteb` package results cache). We need to know
the exact number to beat. Document in `runs/eval/leaderboard_targets.json`.

**M6. Decision gate** — see section 2's decision table. Decide the
venue target based on the public benchmark results. This determines
how much to invest in the no-ontology control, vocab-shift slices, and
the rest of the plan.

### Phase A — Ablation harness (after Phase M gate, in parallel with
remaining work since data generation is CPU/LLM-bound)

**A1. Add `--hf-dataset` flag to `stage2_colbert_finetune.py`**

The training script currently hardcodes
`fierysurf/medcolbert-training-v1`. Add an `--hf-dataset` arg (default
`fierysurf/medcolbert-training-v1`) so the no-ontology run can point at
a different dataset without forking the config. Five-line change.

**A2. Build the no-ontology synthetic dataset**

Generate ~711k triplets where:
- Positive passages are sampled from the same 66k PubMed corpus
- Queries are generated with a generic "ask a question this passage
  answers" prompt (no CUI grounding, no synonym substitution, no
  vocab-shift bucket targeting)
- Hard negatives are mined with the same BM25 + dense pipeline as v1
  (so negative quality is matched)

Output: `fierysurf/medcolbert-training-v1-no-ontology` (private HF dataset)
with the same shape as `medcolbert-training-v1`.

This is a Phase-4-style data/LLM task. It will take longer than any
other task here because it requires LLM generation at scale. Scope the
prompt, run a 1k pilot, audit, then scale to 711k.

**A3. Smoke-test the no-ontology dataset**

```bash
uv run python scripts/training/stage0_smoke.py \
  --train-config configs/train_base.yaml \
  --model-config configs/base.yaml \
  --hf-dataset fierysurf/medcolbert-training-v1-no-ontology \
  --max-examples 1000 --max-steps 1000
```

Verify loss descends cleanly.

**A4. Train `medcolbert-no-ontology-run0`**

```bash
uv run python scripts/training/stage2_colbert_finetune.py \
  --train-config configs/train_base.yaml \
  --model-config configs/base.yaml \
  --hf-dataset fierysurf/medcolbert-training-v1-no-ontology \
  --run-name medcolbert-no-ontology-run0 \
  --batch-size 256 \
  --mini-batch-size 160 \
  --no-gradient-checkpointing \
  --max-steps 14000 \
  --save-strategy epoch \
  --save-total-limit 3 \
  --output-dir runs/no_ontology_stage2 \
  2>&1 | tee logs/medcolbert-no-ontology-run0.log
```

Same compute budget as the base MedColBERT run.

**A5. Eval the no-ontology control**

```bash
bash scripts/eval/eval_all_epochs.sh
# with the script pointed at runs/no_ontology_stage2/
```

Or directly, for the final checkpoint:

```bash
uv run python scripts/eval/run_real_corpus.py \
  --model runs/no_ontology_stage2/final \
  --out runs/no_ontology_stage2/eval_final.json
```

**A6. Decision gate**

Compare MedColBERT-base vs no-ontology on the real-corpus gate. Decide
which framing the paper takes (per the table in section 3). Document this
decision in `docs/ablation_decision.md`.

### Phase B — Baselines and unified eval harness

**B1. Build a dense-baseline eval script**

`sentence-transformers` encoder + cosine top-k against the same 66k
corpus. Extend `src/medcolbert/eval/real_corpus.py` with a
`run_dense_eval(model_name, corpus_path, dev_rows, k)` function, or add
`src/medcolbert/eval/dense_baseline.py` alongside it. Reuse
`aggregate_metrics` from `real_corpus.py` for consistency.

**B2. Run BM25 + RM3**

Use Pyserini (Java) or rank_bm25 + RM3. Same corpus, same dev queries.
Save to `runs/eval/bm25_rm3.json`.

**B3. Run general dense baselines**

- `BAAI/bge-large-en-v1.5` or `Snowflake/snowflake-arctic-embed-l`
- `thenlper/gte-large`
- `intfloat/e5-large-v2`

One command each via the dense-baseline script. Token length mode
`default` (96/256).

**B4. Run biomedical dense baselines**

- MedCPT: `cambridgeltl/Sap-CBERT` article encoder, or the official
  MedCPT repository's QueryEncoder + ArticleEncoder
- BMRetriever: the BMRetriever checkpoint if available

**B5. Run generic ColBERT**

Off-the-shelf `lightonai/colbertv2.0` or the default pylate ColBERT.
Build a PLAID index over the 66k corpus and retrieve. No fine-tuning.

**B6. Assemble the baseline table**

Script `scripts/eval/build_baseline_table.py` reads all
`runs/eval/<model>.json` and writes `runs/eval/baseline_table.json`
and a markdown table.

### Phase C — External benchmarks

**C1. Build `src/medcolbert/eval/beir.py`**

A thin runner that:
1. Downloads each benchmark (NFCorpus, SciFact, TREC-COVID, BioASQ,
   TREC Clinical Trials, R2MED, PMC-Patients) from BEIR or HF
2. Respects the length-mode overrides from `configs/eval.yaml`
3. Encodes + retrieves with either ColBERT/PLAID or a dense model
4. Computes nDCG@10, Recall@100, Recall@1000, MAP@10
5. Writes `runs/eval/<benchmark>_<model>.json`

**C2. Run MedColBERT-base and MedColBERT-large on all enabled
benchmarks.**

**C3. Run the no-ontology control and at least one dense baseline on
the same benchmarks.**

**C4. Assemble `runs/eval/main_table.json`.** Benchmark × model × metric.

### Phase D — Vocab-shift slices

**D1. Build `scripts/eval/vocab_shift.py`**

Loads dev queries, groups by `vocab_shift_type`, computes per-bucket
MRR@10 / Recall@10 / Recall@100 against the existing per-model PLAID
index. Writes `runs/eval/vocab_shift_table.json`.

**D2. Run for MedColBERT-base, no-ontology control, BM25.**

**D3. Inspect the bucket-level gap.** If MedColBERT's gain concentrates
in the shift buckets and the same-vocab bucket is competitive, the
headline claim is supported. If gains are uniform, the framing must
acknowledge that the ontology recipe helps overall but does not
specifically help vocabulary shift — still a valid result, but a weaker
claim.

**D4. (Conditional) Build the Mini MedVocabShift benchmark** if the
internal slice analysis is underpowered. 500-2,000 queries, bucketed,
public-safe. See docs/goals/07 §Mini MedVocabShift Benchmark.

### Phase E — Decontamination

**E1. Build `src/medcolbert/data/decontaminate.py`**

MinHash / fingerprint-based overlap detector. Reuses
`data/processed/private/passages/fingerprints.parquet`.

**E2. Run decontamination for:** the internal eval corpus, every
external benchmark corpus, and dev queries against training queries.

**E3. Write `runs/eval/decontamination_summary.json`.**

**E4. If contamination > 0.1%:** exclude the contaminated eval queries
and document the exclusion count in the paper.

### Phase F — Efficiency

**F1. Build `scripts/eval/efficiency.py`** (see section 8).

**F2. Run for every model variant.** Write
`runs/eval/efficiency_<model>.json`.

**F3. Assemble `runs/eval/efficiency_table.json`.**

### Phase G — Release

**G1. Memorization audit** — probe the trained model for verbatim
UMLS-string reproduction. Write `runs/release/memorization_audit.json`.

**G2. Build the model card** — `docs/model_card.md` (or
`runs/large_stage2/final/MODEL_CARD.md`). Cover the 10 required sections
from section 10.

**G3. Scan public artifacts for private leakage** — grep the repo and
the release bundle for restricted paths, UMLS file names, restricted
strings.

**G4. Run the release audit checklist** — write to
`docs/release_checklist.md`.

**G5. Push model weights** to the private HF internal repo via
`scripts/release/push_checkpoints.sh` (already written).

**G6. (Only after audit passes) Publish weights and model card.**

### Phase H — Paper

**H1. Draft the main table**, the ablation table, the vocab-shift
table, the efficiency table, the data-quality table. All values must
come from committed JSON in `runs/eval/`.

**H2. Generate MaxSim alignment qualitative examples.**

**H3. Write the paper**, answering the 8 reviewer questions
in-line (section 11).

**H4. Internal review with an LLM co-researcher** (per AGENTS.md, an
open LLM may red-team claims and review the eval tables).

---

## 13. Risks and gates

### Research risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **MedColBERT-base underperforms on public MTEB/R2MED benchmarks** | **Medium** | **Critical — caps venue at workshop/arXiv; the internal gate does not generalize** | **Run public benchmarks FIRST (Phase M) before investing in the no-ontology data generation. If the public number is dead, diagnose why (corpus overlap? domain mismatch?) before spending GPU time on ablations.** |
| **ColBERT cannot top MTEB(Medical, v1) aggregate due to missing clustering/reranking/multilingual tasks** | **Certain** | **Cannot claim #1 on the headline leaderboard** | **Compete on the English-retrieval subset and on R2MED (pure retrieval, all English). Report per-task numbers, not the aggregate. The paper targets per-task SOTA, not the MTEB category-average.** |
| No-ontology control matches MedColBERT on the real-corpus gate | Medium | Critical — mechanism claim collapses | Run the control (Phase A) after the public benchmark gate. Pivot framing if it happens. |
| No-ontology control is itself very strong on shift buckets | Low-Medium | Weakens the mechanism claim | The vocab-shift slice analysis is the deciding evidence. If the control is strong on average but weak on `no_direct_lexical_overlap`, the claim still holds at the slice level. |
| External benchmarks don't show gains vs published baselines | Medium | Limits the "competitive or SOTA" claim | The claim is benchmark-scoped. Even if MedColBERT is not SOTA on TREC-COVID, it can be the strongest late-interaction model on the internal real-corpus gate and the published biomedical tasks. Scope the claims accordingly. |
| Decontamination finds overlap | Low | Requires re-running evals after excluding contaminated examples | Run decontamination early (Phase E can start once the eval corpus is fixed). |
| Memorization of UMLS strings in the released model | Low-Medium | Blocks public weight release | Run the memorization audit before release. If found, gate the model or filter. |
| MedColBERT-large underperforms MedColBERT-base | Medium | Weakens the scaling claim | Acceptable — report the large result as an ablation; the base result already clears the gate. |

### Project risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **MTEB ColBERT adapter is non-trivial (PLAID retrieval vs dense encode protocol mismatch)** | **Medium** | **Schedule slip for Phase M** | **Check if MTEB ships a ColBERT wrapper first. If not, the adapter is ~50 lines — the ColBERT model already encodes queries/passages; the adapter just routes MTEB's `encode` calls to `pylate.models.ColBERT.encode` with the right `is_query` flag.** |
| No-ontology dataset generation is slow (LLM at scale) | High | Schedule slip | Start Phase A2 after the Phase M gate. Pilot at 1k first to validate the prompt before committing to 711k. |
| External benchmark code is fiddly (BEIR split conventions vary) | Medium | Schedule slip | Defer BioASQ and TREC-CT to a second pass if the first three (NFCorpus, SciFact, TRECCOVID) work cleanly. |
| GPU contention (only one L40S) | High | Sequential training runs | Sequencing is forced: the no-ontology base run (14h) and the no-ontology large run (23h) cannot run while the current large run is on the GPU. Plan serial training windows. Phase M (benchmark eval) also needs GPU but is read-only — can run on the same GPU if training is paused. |

### Gates

| Gate | Required evidence | What unblocks |
|---|---|---|
| **Public benchmark gate (M6)** | **MedColBERT-base + at least one dense baseline evaluated on MTEB Medical retrieval subset + R2MED. Decision table from section 2 filled in.** | **Venue decision (SIGIR / ECIR-CIKM / workshop). Determines how much to invest in the rest of the plan.** |
| No-ontology gate (A6) | No-ontology model evaluated on the real-corpus gate, side-by-side with MedColBERT-base | Mechanism claim, paper framing, vocab-shift analysis |
| Baseline gate (B6) | All required baselines evaluated on the real-corpus gate with the same dev queries | Main benchmark table draft |
| External benchmark gate (C4) | All enabled benchmarks run for MedColBERT and at least one baseline | "Competitive or SOTA" claim |
| Vocab-shift gate (D3) | Per-bucket metrics for MedColBERT, no-ontology control, and BM25 | Mechanism claim, headline paper result |
| Decontamination gate (E3) | Zero or low contamination, documented exclusions | Reviewer question "How was contamination prevented?" |
| Efficiency gate (F3) | Efficiency table for all variants | "How expensive is indexing and querying?" |
| Release audit gate (G4) | Checklist signed off, no private leakage | Public weight release |
| Memorization gate (part of G1) | No verbatim restricted-string reproduction | Public weight release |

---

## 14. Venue targeting ladder

### Honest assessment of what this paper is

This is a **recipe paper**, not a method paper:
- **No new architecture** — ColBERT + ModernBERT is a known combination
- **No new backbone** — BioClinical-ModernBERT is Sounack et al. 2025 (a strong arXiv preprint, not yet a top-venue paper)
- **The contribution is a training recipe** — ontology-controlled synthetic supervision over real passages, with a matched-compute ablation

That is a known recipe type ("fine-tune a backbone on synthetic data") with one twist (UMLS control over generation). Reviewers at top venues see this pattern constantly. The 0.971 vs 0.626 internal gate is a promising sanity check, not a result — it's against BM25 only, on an internal held-out split from our own synthetic generator.

### The venue ladder

| Venue | Probability | What it requires |
|---|---|---|
| **arXiv preprint** | Always | Clean writing. No peer-review signal. |
| **BioNLP / ClinicalNLP workshop** (ACL/NAACL/ECL) | High (75-85%) | Solid execution + clean ablation. Natural home. |
| **SIGIR / ACL / EMNLP workshop** | High (70-80%) | Same. |
| **ACL/EMNLP/NAACL Findings** | Medium (40-50%) | Strong ablation + at least 2 external benchmark wins + careful claim wording. Findings accepts "good empirical work that doesn't clear the novelty bar." |
| **ECIR / CIKM** (mid-tier) | Medium (40-55%) | Similar to Findings. More receptive to thorough empirical/systems papers than top venues. |
| **SIGIR / ACL / EMNLP main track** | Low (<15%) | Requires either a new benchmark that becomes community-standard, a surprise empirical finding, or architectural novelty. Achievable **only if** MedColBERT tops at least one public MTEB medical retrieval leaderboard *and* the no-ontology ablation shows a surprisingly large ontology-controlled gap. |
| **NeurIPS / ICLR** | Near-zero | No methodological innovation, no theory, no new learning algorithm. Don't waste cycles here. |

### What the MTEB opportunity changes

Before considering MTEB, the realistic ceiling was "ACL/EMNLP Findings at best." A **per-task SOTA on a public MTEB medical retrieval leaderboard** — even on a subset — is verifiable by anyone, community-recognized, and lifts the project out of "internal eval" territory. Specifically:

| Outcome (on public benchmarks) | Venue target |
|---|---|
| Win 1-2 medical retrieval tasks on MTEB(Medical) or R2MED, with the no-ontology ablation showing the ontology recipe matters | **ECIR / CIKM** — real shot. Mid-tier but respectable. |
| Top R2MED overall AND beat NV-Embed / BGE-large on 2+ medical retrieval tasks + vocab-shift slice concentration | **SIGIR short paper or ACL Findings** — plausible. |
| Top the R2MED leaderboard overall AND beat NV-Embed on 2+ medical tasks + a manually-labeled vocab-shift benchmark | **SIGIR full paper** — possible. This is the upper bound. |
| Anything short of topping at least one public benchmark | Workshop / arXiv only. |

### The three probability drivers, in priority order

The paper's fate hinges on three unknowns:

1. **Do public MTEB/R2MED benchmarks show MedColBERT is competitive?** — Run FIRST (Phase M). If MedColBERT doesn't beat BGE-large or MedCPT on NFCorpus, the rest is moot. This is the highest-information experiment per hour of compute.

2. **Does the no-ontology control show a meaningful gap?** — If the gap is small or zero → mechanism claim collapses → paper becomes "we fine-tuned a backbone" → workshop or arXiv only. If the gap is large AND concentrated on vocab-shift buckets → real mechanism story → mid-tier or Findings plausible.

3. **Is there a manually-validated vocab-shift benchmark?** — LLM-labeled slices are circular. Reviewers will hammer this. A small (100-500 examples) hand-labeled vocab-shift benchmark is the single highest-leverage thing for paper credibility. Even tiny, manual labels are unimpeachable.

### Concrete playbook to crack a respectable venue

1. **Run public benchmarks first.** Before the no-ontology data generation. Phase M is the venue gate.
2. **Run MedCPT and BGE-large on your internal gate.** If MedCPT gets 0.95 Recall@100 on your 66k corpus, your 0.971 is not impressive. If MedCPT gets 0.80 and you get 0.97, you have a result.
3. **Only then build the no-ontology control.** Frame the paper around the ablation, not the absolute number.
4. **Build a small hand-labeled vocab-shift benchmark.** 100-300 examples, manually bucketed by a human, not an LLM. This is gold for paper credibility.
5. **Add at least one non-PubMed domain.** PMC-Patients or TREC Clinical Trials. PubMed-only is an easy reject — "you trained on PubMed and tested on PubMed."
6. **Show qualitative MaxSim alignment examples.** The heart-attack → myocardial-infarction token alignment visualizations are 5× more persuasive than a metric table. Reviewers remember them.
7. **Claim wording:** "Ontology-controlled synthetic supervision transfers to real biomedical retrieval, with the largest gains on vocabulary-shift slices" — NOT "MedColBERT is SOTA on medical retrieval."
8. **Target ECIR, CIKM, or Findings.** Don't burn cycles on SIGIR/ACL main track unless the public benchmark wins AND the ablation gap are both large.

### What would elevate it to a top-tier paper

The gap between this and a SIGIR/ACL main-track paper:
- A **new manually-validated benchmark** that becomes community-standard (MedVocabShift with hand-labeled 1k+ examples) — a contribution on its own
- **Best results on 3+ established benchmarks** vs MedCPT, BGE-large, NV-Embed
- A **mechanism analysis** — not just "it works" but *why* the ontology control helps (probe the projection layer, show which token alignments change, quantify the synonymy resolution)
- A **second domain** (clinical notes, trials) showing the recipe transfers

Right now we have none of these. They're achievable but each one is real work. Phase M tells us whether to pursue them.

---

## 15. Artifacts manifest

Target layout under `runs/` after all phases complete:

```
runs/
  base_stage2/                       # DONE — base MedColBERT
    checkpoint-{11104,13880}/
    final/
    eval_epoch{1,2,4,5}.json
    eval_final.json
    eval_comparison.txt
    eval_index_epoch{1,2,4,5,final}/
  large_stage2/                      # IN PROGRESS — large MedColBERT
    checkpoint-{2776,5552,...}/
    final/
    eval_epoch{1,2,4,5}.json
    eval_final.json
    eval_comparison.txt
  no_ontology_stage2/                # Phase A
    checkpoint-*/
    final/
    eval_epoch{1,2,4,5}.json
    eval_final.json
  eval/
    mteb_base/                       # Phase M2 — MTEB Medical + R2MED for MedColBERT-base
      NFCorpus.json
      TRECCOVID.json
      SciFact.json
      MedicalQARetrieval.json
      CUREv1.json
      R2MEDBiologyRetrieval.json
      R2MEDBioinformaticsRetrieval.json
      R2MEDMedicalSciencesRetrieval.json
      R2MEDMedXpertQAExamRetrieval.json
      R2MEDMedQADiagRetrieval.json
      R2MEDPMCTreatmentRetrieval.json
      R2MEDPMCClinicalRetrieval.json
      R2MEDIIYiClinicalRetrieval.json
    mteb_large/                      # Phase M3 — same tasks for MedColBERT-large
    mteb_bge_large/                  # Phase M4 — dense baseline
    mteb_medcpt/                      # Phase M4 — biomedical dense baseline
    leaderboard_targets.json         # Phase M5 — current top-5 per task
    main_table.json                  # Phase C4
    vocab_shift_table.json           # Phase D2
    efficiency_table.json            # Phase F3
    baseline_table.json              # Phase B6
    decontamination_summary.json     # Phase E3
    bm25.json                        # DONE
    bm25_rm3.json                    # Phase B2
    bge_large.json                   # Phase B3
    gte_large.json                   # Phase B3
    e5_large.json                    # Phase B3
    medcpt.json                      # Phase B4
    bmr_retriever.json               # Phase B4
    generic_colbert.json             # Phase B5
    nfcorpus_medcolbert_base.json    # Phase C
    nfcorpus_medcolbert_large.json   # Phase C
    nfcorpus_no_ontology.json        # Phase C
    nfcorpus_medcpt.json             # Phase C
    scifact_*.json                   # Phase C
    trec_covid_*.json                # Phase C
    bioasq_*.json                    # Phase C
    trec_ct_*.json                   # Phase C
    r2med_*.json                     # Phase C
    pmc_patients_*.json              # Phase C
    efficiency_*.json               # Phase F
  release/
    memorization_audit.json          # Phase G1
    model_card.md                    # Phase G2
    release_checklist.md             # Phase G4
```

---

## Summary

The path from here to paper has **two critical gates**, not one:

1. **The public benchmark gate (Phase M).** Run MTEB Medical retrieval
   subset + R2MED on MedColBERT-base *first*, before any new training or
   data generation. This is the highest-information-per-hour experiment
   in the plan. It tells us whether the internal 0.971 gate generalizes
   or is a fine-tuning artifact, and it determines the venue target
   (SIGIR main track, ECIR/CIKM, Findings, or workshop). If the public
   number is dead, the rest of the plan is workshop-level and we don't
   need the full ablation battery.

2. **The no-ontology mechanism gate (Phase A).** Train the same backbone
   on generic (non-ontology-grounded) synthetic queries with matched
   compute, and compare on the internal gate. This isolates whether the
   ontology recipe — not just the backbone — is responsible for the
   retrieval quality. Without it, no mechanism claim is possible. It
   runs *after* Phase M because its framing (and how much to invest in
   it) depends on the public benchmark result.

### Revised ordering

1. **Phase M — Public retrieval benchmarks** (FIRST). MTEB Medical
   retrieval subset, R2MED, with BGE-large + MedCPT as comparison rows.
   Venue-deciding.
2. **Phase A — No-ontology control** (after Phase M gate). Data
   generation is CPU/LLM-bound and can overlap with GPU-bound work.
3. **Phase B — Unified eval harness + required baselines** on the
   internal gate.
4. **Phase C — Remaining external benchmarks** (TREC-CT, PMC-Patients,
   BioASQ) for the "second domain" generalization claim.
5. **Phase D — Vocab-shift slices** — the spine of the mechanism story.
6. **Phase E — Decontamination**, **Phase F — Efficiency** — mechanical.
7. **Phase G — Release audit + model card + memorization audit.**
8. **Phase H — Paper writing**, only after all evidence exists.

The plan deliberately front-loads the **venue** risk (Phase M) before
the **mechanism** risk (Phase A), because the venue determines whether
the mechanism story is worth telling at a top venue or just a workshop.
Phase M can run as soon as the large training completes (or even on the
base model now, if GPU time is available). Phase A's data generation is
the slowest non-GPU task and should start immediately after the Phase M
gate so it can overlap with the remaining GPU-bound eval work.

Each task above is small enough to fit in a single working session, and
the ordering respects the dependency chain so that no task is started
before its prerequisites are in place.

---

**Document status:** Living document. Update task statuses as work
proceeds; the gate table in section 13 is the operating checklist.