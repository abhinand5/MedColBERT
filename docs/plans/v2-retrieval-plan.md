# MedColBERT / MedEmbed v2 — Retrieval Plan

**Status:** authored 2026-10-06 from the planning session. Long-term plan for the
retrieval track. It runs **after** MedDecide (separate repo
`abhinand5/MedDecide`) can act as the relevance judge — see that repo's
`docs/plans/PROGRAM.md`. Loops for this track are planned when it starts; this document
fixes the targets, settled decisions, and stage gates.

---

## 1. Goal

Absolute state of the art in medical retrieval. Not a smaller claim: if SoTA needs a
data moat, build the data moat.

### Scoreboard: MedIR-14 (frozen)

Mean nDCG@10 over 14 English medical retrieval tasks, measured with this repo's MTEB
harness (`scripts/eval/run_mteb_local.py`, validated against published numbers):

- MTEB(Medical) English retrieval: NFCorpus, SciFact, TRECCOVID,
  MedicalQARetrieval, CUREv1 (en), PublicHealthQA (en)
- R2MED: Biology, Bioinformatics, MedicalSciences, MedXpertQAExam, MedQADiag,
  PMCTreatment, PMCClinical, IIYiClinical

**Bar:** the best public score per task and the best public model on the mean (R2MED
mean leader as of 2026-10-05: gte-Qwen2-7B-instruct 0.323; TRECCOVID leader
Yuan-embedding-2.0-en 0.983, Qwen3-Embedding-8B 0.950).

**Mechanism scoreboard:** a vocabulary-shift slice built from **real** queries
(layperson ↔ clinical, abbreviation ↔ expansion, brand ↔ generic), not from our
generator. This backs the ontology-controlled-supervision claim.

The SoTA claim must come from the **retrievers alone** — MedDecide is a general
medical decision model, not a reranker in this claim.

---

## 2. Settled decisions

| # | decision | rationale |
|---|---|---|
| R1 | Two model families: **MedColBERT** (late interaction) and **MedEmbed** (dense); a sparse variant only if it earns its place | v1 already trained both |
| R2 | **Efficient tier:** BioClinical-ModernBERT base/large, **with a retrieval-pretraining stage added** before fine-tuning | the medical MLM is kept for its domain knowledge; v1 failed for lack of retrieval pretraining, not because of the backbone |
| R3 | **Flagship tier (MedEmbed only):** candidates Qwen3-Embedding-4B, Qwen3.5-4B (or 2B), gemma-4-12B-it-qat-w4a16-ct | absolute-#1 contender against 7–8B leaders |
| R4 | **Pareto rule:** if the efficient-tier MedEmbed reaches the Pareto frontier on MedIR-14 (no public model with ≤ its parameters scores higher on the mean), the flagship gets only a sample pass with Qwen3.5-4B/2B for comparison; if that pass is SoTA or near-SoTA, pursue it | operator decision |
| R5 | Every training run is gated on a cheap public-benchmark subset, never on an in-distribution eval | v1's failure |
| R6 | Relevance labels come from **MedDecide** once it beats an off-the-shelf judge on gold relevance; until then from a calibrated off-the-shelf judge | one judge serves both tracks |
| R7 | UMLS stays private; nothing derived from restricted vocabularies is committed (this repo is public) | licence |

Risks recorded: Qwen3.5-4B and gemma-4-12B are chat LLMs and need conversion to
embedders (pooling + contrastive warm-up) — a weak sample pass may mean
"under-converted", not "flagship tier fails". The gemma-4-12B checkpoint is 4-bit
quantised (QLoRA-style training path).

---

## 3. Stages

### S0 — Eval gate and baseline rows

- Make the harness robust: stream large corpora (CUREv1 died at ~23 GB RAM), verify
  result files exist for every task, run tasks as separate processes.
- One-time reference rows through the same harness: BM25, MedCPT, BMRetriever,
  GTE-ModernColBERT-v1, Qwen3-Embedding-0.6B, BioClinical-ModernBERT untrained.
- **Gate-subset** (~20 min on one GPU): NFCorpus, SciFact, a TREC-COVID subset, three
  R2MED tasks. Every checkpoint of every run is scored on it.
- **Gate:** reference rows reproduce published numbers within ±1 pt where published.

### S1 — Retrieval pretraining on BioClinical-ModernBERT

Contrastive pretraining on large weakly supervised medical pairs: PubMed
title ↔ abstract (~25M), PMC section ↔ section, citation pairs, MedQuAD / MedMCQA /
PubMedQA pairs, plus a share of general pairs (MS MARCO) for robustness.
- **A/B control:** the same downstream fine-tuning started from GTE-ModernColBERT-v1
  (ColBERT) / gte-modernbert-base (dense).
- **Gate:** the medical-MLM start ≥ the general-retriever start on the gate-subset.
  If it loses, the efficient tier switches to the general start (the backbone
  question is settled by evidence).

### S2 — Data moat v2

- **Corpus:** full PubMed, PMC-Patients, ClinicalTrials.gov, DailyMed / openFDA labels,
  MedlinePlus (the layperson side of vocabulary shift). Not two 1970s shards.
- **Queries:** ontology-controlled synthetic queries (UMLS-driven vocabulary shift) +
  real query sources (PubMedQA/BioASQ-style questions, consumer health questions).
- **Negatives:** mined from **real** corpora (BM25 + current retriever), judged by the
  relevance judge; keep the band "confidently not relevant but close"
  (e.g. 0.05 < p < 0.3); drop high-p false negatives. No LLM-written negatives as the
  majority class.
- **Soft labels:** judge probabilities for KL distillation
  (`scripts/training/stage3_kl_distillation.py` exists from v1).
- **Decontamination** against every MedIR-14 test corpus and query set.
- **Gate:** a fine-tuned model beats its own starting checkpoint on the gate-subset.

### S3 — Train the efficient tier

MedColBERT and MedEmbed, base and large, on S1 backbones + S2 data, with the
gate-subset scored every epoch. Mine → judge → retrain iterations until the
gate-subset plateaus.
- **Gate:** full MedIR-14; Pareto-frontier check (R4).

### S4 — Flagship sample pass / flagship

Per R4. Qwen3-Embedding-4B continued training on the S2 data is the cheapest strong
flagship; Qwen3.5-4B/2B is the comparison pass.
- **Gate:** absolute MedIR-14 standing per task and on the mean.

### S5 — Mechanism and controls

- **No-ontology control** (mandatory): same recipe, synthetic queries generated without
  ontology control. Separates the backbone and data volume from the ontology recipe.
- Vocabulary-shift slice results; query-expansion / RM3 baseline for vocabulary claims.
- Efficiency table (index size, latency) for ColBERT vs dense.

### S6 — Release and paper

Model cards with provenance, decontamination report, HF release, paper.

---

## 4. Compute

One RTX PRO 6000 (96 GB) covers S0–S3 for the efficient tier; S1 at ~25M pairs is
roughly 1–2 days for a base model (estimate, not measured). Judge labelling runs on the
operator's 4×PRO 6000 teacher machine or with MedDecide once it is ready. A short
multi-GPU burst is worth it only for the final large / flagship runs.

---

## 5. What v1 left behind

- `fierysurf/medcolbert-training-v1` (private HF dataset), the four private v1 model
  repos, the private UMLS ontology tables, synthetic + negative data, and v1 eval
  results — on the L40S pod being retired. Back up before termination (operator).
- The v1 public eval record: [../v1-public-eval.md](../v1-public-eval.md).
