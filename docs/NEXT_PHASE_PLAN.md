# MedColBERT — Next-Phase Plan (Phases 4.5 → 8)

**Audience:** an implementing agent of any skill level. This plan is self-contained:
read it top-to-bottom, execute the tasks in order, do not skip the verification gates.
Each phase lists *what to build*, *exact commands*, *the acceptance gate*, and *the
known pitfalls* (these are real bugs already hit once — re-hitting them wastes hours).

**Last updated:** 2026-06-20. **Current commit:** `176e436` on branch `dev`.

---

## 0. Where we are right now

- **Done (Phase 1–4):** the full data + synthetic-generation stack is built and committed.
  - `src/medcolbert/` package: `data/` (ontology, passages, corpora, umls),
    `generation/` (dspy_programs, direct_generator, validators, teacher_filter, prompts),
    `utils/`, `cli.py` (Typer entrypoint: `umls`/`data`/`synth`/`eval`/`train` sub-apps;
    only `umls`/`data`/`synth` have commands — `eval` and `train` are **empty stubs**).
  - Canonical generation script: `scripts/data/multistyle_batch_v2.py` (vLLM + Gemma-4-31B
    at `http://localhost:8000/v1`). Consolidation/HF push:
    `scripts/data/consolidate_multistyle.py`. Audits: `audit_quality.py` (proxy) and
    `ngram_copy_audit.py` (authoritative — see §1.2).
  - Tests: `uv run --extra dev pytest` → 17 passed, 1 skipped.
  - Dataset on HF: `fierysurf/medcolbert-synthetic-pilot`, **100,867 examples**.

- **The problem (see `docs/audit_2026-06-20.md`):** only **3/8 quality thresholds met**.
  The dataset on HF was generated at `temp=0.0` and the audit verdict is explicit:
  *"NOT suitable for ColBERT contrastive training in current state."* The code fixes are
  in place but **the HF dataset has NOT been regenerated with them yet.**

| Threshold | Target | Actual | Status |
|---|---|---|---|
| UMLS artifacts | <0.5% | 0.0% | ✅ |
| Abbreviation genuine | ≥80% | 52% | ❌ |
| Vocabulary shift (n-gram copy) | ≥60/100 | 10–21/100 | ❌ |
| Hallucinated terms | <5% | 0.9–1.2% | ✅ |
| Template monotony | <30% | 90% | ❌ |
| Data integrity | no corruption | 97/100 | ✅ |
| Dataset size | 100K+ | 100,867 | ✅ |
| Pushed to HF | yes | yes | ✅ |

- **What is NOT done (Phases 5–8):** triplets + negatives, ColBERT training, evaluation,
  release/paper. The numbered scripts `scripts/data/05_mine_bm25_negatives.py`,
  `06_mine_dense_negatives.py`, `07_merge_and_deduplicate.py` are **0-line stubs**.
  Configs `configs/train_base.yaml`, `configs/eval.yaml`, `configs/base.yaml`,
  `configs/large.yaml` already exist and define the contracts below.

- **Housekeeping note:** `download_all_umls.py` is gitignored because it contains a
  **hardcoded UMLS API key** — do not commit it; load keys from env (`UMLS_API_KEY`)
  instead, and rotate the exposed key.

---

## 1. Phase 4.5 — Dataset Quality Remediation (BLOCKER; do this first)

**Goal:** regenerate the dataset so all 8 thresholds pass *before* spending any compute
on triplets/training. Training on the current HF dataset will produce a model that
reformulates rather than substitutes vocabulary — the core hypothesis would be untestable.

**Estimated time:** 6–8 hours of vLLM generation (mostly unattended), ~1 hour of hands-on.

### 1.1 Preconditions
- vLLM serving Gemma-4-31B at `http://localhost:8000/v1`. Verify:
  `curl -s http://localhost:8000/v1/models | jq '.data[0].id'` → must contain `gemma`.
- `HF_TOKEN` exported (for the final push).
- `uv sync --extra generation --extra data --extra dev` is current.

### 1.2 The metric that matters (do not be fooled)
`audit_quality.py` reports 88–94/100 via alt_term presence + term_overlap. **That proxy
is misleading.** The real vocabulary-shift metric is **n-gram copy rate** = fraction of
query 4-grams appearing verbatim in the source passage. Use:

```
uv run python scripts/data/ngram_copy_audit.py --per-family
# score = 100 * (1 - mean_4gram_copy). Good data ≈ 99/100. Threshold: ≥60/100 every family.
```

Note: ~70% of rows have `term_overlap < 0.5` and **that is expected** (keyword queries
are short + use substituted vocab), not hallucination. Do not "fix" low term_overlap.

### 1.3 Tasks

1. **Regenerate with the fixed code.** The fixes are already in
   `multistyle_batch_v2.py` (temp 0.2 fact / 0.7 questions / 0.8 keywords, diversified
   prompts, `QUESTION_OPENERS` + deterministic per-row opener via MD5 of
   `(fact,role,task_family,cui_label,alt_term)`, Q4 term-overlap check, post-generation
   rejection filter). Run the orchestrator to reach 100K accepted:
   ```
   uv run python scripts/data/orchestrator_v2.py --target 100000
   ```
   - Keyword-style queries skip LLM judging (accepted if validation passes) — this is
     intentional, cuts judging time ~75%. Do not add judging back.
   - Outputs land in `v6_fresh/` batch dirs (NOT the legacy `ontology_grounded_teacher_filtered/`).

2. **Regenerate `full_question` rows cheaply if monotony creeps back.** If
   `audit_quality.py` shows the "Does [NP] [VP]?" template >30% on `full_question`, do
   **not** rerun whole batches. Use the forced-opener regen (~5× cheaper, reuses
   extracted facts/passages):
   ```
   uv run python scripts/data/regen_full_questions.py
   ```
   Output → `fq_v6_*` dirs. Consolidation drops old `div_v6` `full_question` rows and
   keeps `fq_v6` + `div_v6` keyword rows.

3. **Fix abbreviation quality (52% → ≥80%).** Run
   `scripts/data/prepare_abbreviation_passages.py`. The pitfalls (already solved in code,
   do not regress) are:
   - **Plain-word alt_terms** (Lung, Eels, zinc, Crohn accepted as "abbreviations"):
     `is_real_abbreviation()` requires a digit OR internal capitalization (camelCase like
     cGMP/hCG) for short mixed-case, plus a large `_COMMON_WORDS` stoplist checked
     case-insensitively.
   - **Numeric alt_terms** (cancer-staging codes "2", "223", "T2"): in
     `build_abbreviation_index()`, require an alphabetic run ≥2 in **both** `abbreviated`
     and `abbreviated_norm`, plus common-word check on both.
   - **Homonym mismatches** (dominant failure): the wrong expansion is chosen because it
     shares generic words with the passage (TTP→"...WITH thrombospondin" matched a
     thymidine passage via "with"). Fix: the context-relevance filter uses
     `_content_words()` (≥4 chars, alpha, **excluding** generic biomedical/English
     stopwords like with/type/protein/gene/nuclear/membrane) and requires **≥2
     distinctive overlaps**. ≥1 overlap leaves ~9% homonyms; ≥2 cuts them to ~zero but
     shrinks the pool ~6× (7300→730 passages). Accept the smaller pool — abbreviation is
     ~3% of the dataset by design; the 4 main families are the bulk.
   - The abbreviation pool is small (~830 passages, ~3K rows after strict filtering).
     That is correct, not a bug.

4. **Backfill underrepresented slices** (audit §5): `abbreviation_to_expanded` had 256
   rows vs 17–31K for others; 6 semantic groups had <100 rows. Re-run
   `multistyle_batch_v2.py` with passage filters targeting the thin groups until each
   task family ≥ ~5K and each semantic group ≥ ~500. Role-style is confounded (each role
   locked to one query style) — loosen that mapping in the prompt if a family is stuck.

5. **Consolidate + push.** `find_all_accepted()` scans `v6_fresh/` **only** (excludes
   legacy temp=0.0 dir). Then push and clean stale files:
   ```
   HF_TOKEN=... uv run python scripts/data/consolidate_multistyle.py --push
   uv run python scripts/data/hf_clean_stale.py
   ```

### 1.4 Gate (must pass before Phase 5)
Re-run both audits against the freshly consolidated dataset and confirm:
- `ngram_copy_audit.py --per-family` → **≥60/100 every family** (aim 90+).
- `audit_quality.py` → abbreviation genuine **≥80%**, template monotony **<30%**
  (keyword queries have 0% monotony by design — do not count their starter-word
  concentration against the threshold).
- Hallucination **<5%**, UMLS artifacts **0%**, integrity **97+/100**, size **≥100K**.

Do not proceed to Phase 5 until all 8 are green. If a threshold fails, the fix is almost
always in §1.3 tasks 1–4, not new code.

---

## 2. Phase 5 — Triplets and Negatives

**Goal:** turn accepted synthetic examples into ColBERT training triplets with strong,
audited negatives and zero eval leakage. Spec: `docs/goals/05-triplets-and-negatives.md`.

**Inputs:** accepted synthetic examples (Phase 4.5), passage store
(`data/processed/private/passages/passages.parquet` + `passage_cui_matches.parquet`),
ontology edges/pairs, eval blocklists + fingerprints.

**Outputs (private):**
```
data/processed/private/triplets/pilot_triplets.parquet
data/processed/private/triplets/dev_triplets.parquet
data/processed/private/triplets/v1_triplets.parquet
data/processed/private/triplets/negative_audit_samples.parquet
```
**Outputs (public-safe manifests):**
```
data/processed/public/manifests/triplet_counts.json
data/processed/public/manifests/decontamination_report.json
```

### 2.1 Triplet schema (exact — every column required)
```
triplet_id, query, positive_passage_id, negative_passage_id, source,
generation_mode, task_family, target_cuis, positive_cuis, negative_cuis,
negative_type, negative_source, teacher_margin, split, data_hash
```

### 2.2 Tasks

1. **Fill the stub scripts.** `scripts/data/05_mine_bm25_negatives.py`,
   `06_mine_dense_negatives.py`, `07_merge_and_deduplicate.py` are empty. Implement them
   as thin wrappers over a new `src/medcolbert/triplets/` module (mirror the
   `data/`/`generation/` layout: library logic in `src/`, scripts are CLI shims). Suggested
   modules: `negatives.py` (mining), `decontam.py` (blocking), `build.py` (assembly).

2. **Negative types** (mix several per query; `random` is calibration-only, low fraction):
   - `bm25_hard` — lexically similar, not the target concept (from a BM25 index over the
     passage store; this is what `05_mine_bm25_negatives.py` builds).
   - `ontology_parent_child`, `ontology_sibling`, `ontology_related` — from ontology edges.
   - `same_semantic_group` — same group, different concept.
   - `dense_mined` — **only after a Stage-2 checkpoint exists** (Phase 6). Do NOT add
     dense-mined negatives before the first model is trained (explicit common-mistake).
   - `random` — small fraction, calibration.

3. **False-negative controls — reject a negative if** it shares any primary target CUI
   with the query; it is a near-duplicate of the positive; it has the same
   PMID/PMCID/NCT/source-doc as the positive (accidental positive); a teacher/heuristic
   flags it relevant; manual audit repeatedly marks the type ambiguous. Ambiguous
   negatives poison contrastive training — when unsure, drop it.

4. **Decontamination (before writing train files):** block exact eval IDs (PMID, PMCID,
   NCT, dataset-specific IDs), exact text hashes, normalized text hashes, and
   near-duplicates by MinHash/SimHash threshold. Also block synthetic queries too similar
   to eval queries. **Do not decontaminate by IDs alone — use text similarity too.**
   Write `decontamination_report.json`.

5. **Splits:** build `pilot_triplets.parquet` (50–100K, diversity over size) and a
   `dev_triplets.parquet` holdout. `v1_triplets.parquet` is the scaled-up version with
   source-aware sampling — no single source dominates without an explicit config weight.

### 2.3 Gate
- 50K–100K pilot triplets after decontamination.
- Every triplet carries `source`, `task_family`, `generation_mode`, `negative_type`.
- `negative_audit_samples.parquet` shows acceptable false-negative rate on manual review.
- `decontamination_report.json` shows zero known eval leakage.
- Ablation subsets selectable by `generation_mode` and `negative_type`.

---

## 3. Phase 6 — ColBERT Training

**Goal:** train the model variants that isolate the value of ontology-controlled
supervision. Spec: `docs/goals/06-colbert-training.md`. Backbone:
`thomas-sounack/BioClinical-ModernBERT` (`configs/base.yaml`). Framework: PyLate.

### 3.1 Model variants — train IN THIS ORDER (each is a controlled comparison)
1. `generic_colbert` — off-the-shelf ColBERT/PyLate baseline.
2. `bioclinical_colbert_no_ontology` — same backbone, no ontology-controlled generation.
3. `medcolbert_ontology_grounded` — ontology-controlled synthetic supervision.
4. `medcolbert_teacher_filtered` — + modern LLM/reranker filtering.
5. `medcolbert_dense_negative_refresh` — continue/retrain with dense-mined negatives
   (requires the Stage-2 checkpoint → this variant closes the loop with Phase 5 §2.2).

**Large-model training is optional** and only after base ablations support the hypothesis.

### 3.2 Tasks

1. **Add `train` commands to the CLI** (`src/medcolbert/cli.py`, `train_app` is registered
   but empty). Suggested: `medcolbert train smoke`, `medcolbert train run --stage`,
   `medcolbert train index` (build a tiny retrieval index for sanity). Put training logic
   in a new `src/medcolbert/train/` module; keep `cli.py` thin.

2. **Stage 0 — smoke test** (`configs/train_base.yaml` `stage0_smoke`): 1K examples,
   verify forward pass, loss decreases, no NaN/Inf, build a tiny index, retrieve positives
   above simple negatives. **Do not skip this.** Config: `max_examples: 1000, max_steps: 1000`.

3. **Stage 1 — short concept warmup** (`stage1_concept_warmup`): concept-aligned examples
   only briefly. Prefer query→real-passage examples over raw synonym-pair memorization
   (overfitting raw UMLS synonym strings is a named common-mistake). LR 3e-5, bs 64×4, 20K
   steps, bf16.

4. **Stage 2 — main triplet training** (`stage2_colbert`): mixed triplets, source-aware
   sampling. Keep the first serious model simple: 128-dim ColBERT projection, stable
   query/doc lengths, no architectural experiments. LR 1e-5, bs 16×16, 100K steps, bf16,
   gradient checkpointing, gather across devices.

5. **Stage 3 — dense-negative refresh:** mine hard negatives with the Stage-2 checkpoint
   (this is `06_mine_dense_negatives.py` finally getting implemented), rebuild triplets,
   continue/retrain from the warmup checkpoint.

6. **Stage 4 — MRL + efficiency:** add Matryoshka (dims 32/64/128, weights
   0.1/0.2/0.7) **only after** the baseline is stable.

7. **Run metadata — every run writes** (`runs/training/<run>/`): git commit, config
   snapshot (`config_snapshot.yaml`), data manifest + hashes (`data_manifest.json`),
   model name+revision, tokenizer revision, dep lockfile hash, seed, hardware, precision,
   grad accumulation, effective batch size. `metrics.jsonl` per step.

### 3.3 Critical discipline (common-mistakes, from the spec)
- Do not compare MedColBERT to a weaker-backbone baseline only — the no-ontology
  BioClinical-ColBERT control (#2) is mandatory.
- Do not change backbone, data, negatives, and loss all at once — one variable per variant.
- Do not overfit raw UMLS synonym strings.
- Do not train a large model before base ablations are interpretable.

### 3.4 Gate
- `bioclinical_colbert_no_ontology` and `medcolbert_*` both train successfully on the same
  backbone with comparable compute.
- Training curves stable (no loss spikes/NaN).
- Tiny-retrieval sanity passes (positives rank above negatives).
- Candidate indexes build.
- Every variant is reproducible from its config + manifest files.

---

## 4. Phase 7 — Evaluation and Claims

**Goal:** evaluate on real external benchmarks + vocabulary-shift slices; state only
evidence-supported claims. Spec: `docs/goals/07-evaluation-and-claims.md`. Config:
`configs/eval.yaml`.

### 4.1 Tasks

1. **Add `eval` commands to the CLI** (`eval_app` is registered but empty). Suggested:
   `medcolbert eval run --benchmark`, `medcolbert eval table`, `medcolbert eval slice`.
   Logic in a new `src/medcolbert/eval/` module.

2. **Primary benchmarks** (focused set first — `configs/eval.yaml` already enables these):
   R2MED (reasoning), TREC Clinical Trials (long clinical→trial), PMC-Patients
   (case retrieval), and 1–2 BEIR/MTEB biomedical tasks (NFCorpus, SciFact, BioASQ, or
   TREC-COVID). **Add MIRAGE/MedRGB only after retrieval quality is established** — RAG
   benchmarks entangle retriever with generator/prompt.

3. **Required metrics:** nDCG@10, Recall@100, Recall@1000 (when candidate recall
   matters), MAP@10 (benchmark-standard), index size, query latency, encoding throughput,
   MRL 32/64/128 tradeoff (when MRL enabled).

4. **Vocabulary-shift slices** (this is the core experiment — `eval.yaml` `vocab_split`
   already lists the buckets): same_vocab, lay_to_clinical, abbreviation_to_expanded,
   brand_generic, symptom_to_diagnosis_or_treatment, biomedical_to_clinical,
   no_direct_lexical_overlap. **The key result is NOT the average** — it is whether gains
   concentrate in vocab-shift buckets while same-vocab stays competitive. Do not hide
   same-vocab regressions.

5. **Mini MedVocabShift benchmark** (only if existing benchmarks don't isolate vocab
   shift): 500–2000 queries over **public** passages, auditable labels, bucketed by shift
   type. **Do not train on it.** Keep UMLS-derived strings private during construction;
   release only public-safe components.

6. **Baseline table** (`runs/eval/baseline_table.json`) — report at least: BM25, BM25+RM3,
   a strong general embedder, MedCPT (baseline, not oracle), BMRetriever, generic ColBERT,
   BioClinical-ColBERT-no-ontology, MedColBERT, optional BM25+MedColBERT hybrid.

7. **Outputs:** `main_table.json`, `vocab_shift_table.json`, `efficiency_table.json`,
   `baseline_table.json`, `decontamination_summary.json` under `runs/eval/`.

### 4.2 Claim discipline
- Acceptable: *"MedColBERT achieves competitive or state-of-the-art results on selected
  biomedical and clinical retrieval benchmarks, with the largest gains on
  vocabulary-shift slices."*
- Avoid: *"MedColBERT is SOTA on medical retrieval."* or *"Dense retrieval cannot solve
  medical synonymy."*
- Do not tune prompts/data generation against the test set. Do not report RAG gains as
  pure retrieval gains.

### 4.3 Gate
- All eval corpora decontaminated from training (cross-check `decontamination_report.json`).
- Baselines run with comparable corpora + documented settings.
- No-ontology ColBERT control included.
- Vocab-shift buckets reported. Efficiency reported. Claims benchmark-scoped.

---

## 5. Phase 8 — Release and Paper (brief)

Spec: `docs/goals/08-release-and-paper.md`. Only after Phase 7 gates pass.
- License-safe code + model release; UMLS-derived artifacts stay private/gated.
- Paper tables/plots from `runs/eval/*`.
- Release audit (data provenance, decontamination logs, reproducibility from
  config+manifest).
- Exit criteria: results support a precise, benchmark-scoped claim; cross-vocabulary
  analysis supports the core mechanism; UMLS artifacts remain private/gated.

---

## 6. Cross-cutting rules (apply to every phase)

1. **Always use `uv`.** `uv run python …`, `uv run pytest`, `uv run medcolbert …`,
   `uv sync --extra <group>`. Never bare `pip`/`python`/`pytest`. Python 3.11 or 3.12.
2. **Private data stays in `data/processed/private/`** — UMLS strings, CUI mappings,
   synonym tables, matched spans, SNOMED dumps, synthetic queries embedding UMLS terms.
   Public manifests (counts/hashes only) go in `data/processed/public/manifests/`.
3. **Library logic in `src/medcolbert/`, scripts are thin CLI shims.** Match the existing
   module layout when adding triplets/train/eval code. Keep `cli.py` thin.
4. **One variable per change.** Don't change backbone + data + negatives + loss together.
5. **Verify before declaring done.** Run the audit/gate of the current phase; report real
   numbers. If a gate fails, say so — don't paper over it.
6. **Commit per phase** with a clear message; keep `dev` branch. Don't commit secrets
   (the UMLS key in `download_all_umls.py` is intentionally gitignored).
7. **Re-read the relevant `docs/goals/0N-*.md`** before starting each phase — it is the
   authoritative spec; this plan is the execution sequence + pitfalls on top of it.

## 7. Suggested execution order for a fresh agent
1. Read `docs/audit_2026-06-20.md` + this plan + `docs/goals/05-triplets-and-negatives.md`.
2. Phase 4.5: regenerate, audit, push. **Do not start Phase 5 until §1.4 gate is green.**
3. Phase 5: build `src/medcolbert/triplets/`, fill `05/06/07` scripts, hit §2.3 gate.
4. Phase 6: add `train` CLI commands, Stage 0 → 4, train variant #2 then #3, hit §3.4 gate.
5. Phase 7: add `eval` CLI commands, run benchmarks + slices + baselines, hit §4.3 gate.
6. Phase 8: release + paper.
