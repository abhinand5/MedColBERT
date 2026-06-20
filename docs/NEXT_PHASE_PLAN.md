# MedColBERT — Next-Phase Plan (Phases 5 → 8)

**Audience:** an implementing agent of any skill level. This plan is self-contained:
read it top-to-bottom, execute the phases in order, do not skip the verification gates.
Each phase lists *what to build*, *exact commands*, *the acceptance gate*, and *known
pitfalls* (these are real bugs already hit once — re-hitting them wastes hours).

**Last updated:** 2026-06-20. **Branch:** `dev`. **Dataset on HF:** `fierysurf/medcolbert-synthetic-pilot`.

> Read [AGENTS.md](../AGENTS.md) and [GOALS.md](../GOALS.md) first. They define the
> research contract, privacy rules, required baselines, and go/no-go gates. This plan is
> the *execution sequence + operational pitfalls* layered on top of the authoritative
> `docs/goals/0N-*.md` specs. When this plan and a goal spec disagree, the goal spec wins.

---

## 0. Where we are right now

### Done — Phases 1–4.5 (data + synthetic generation, regenerated)

The full data + synthetic-generation stack is built, committed, and **the dataset has
been regenerated to pass all quality thresholds**. The old `temp=0.0` corpus (FAKE vocab
shift) on HF has been *replaced*, not appended.

- **`src/medcolbert/` package:** `data/` (ontology, passages, corpora, umls),
  `generation/` (dspy_programs, direct_generator, validators, teacher_filter, prompts),
  `utils/` (config, hashing, io, logging, text), `cli.py` (Typer entrypoint with
  `umls` / `data` / `synth` sub-apps implemented; **`eval` and `train` are registered but empty**).
- **Canonical generation script:** `scripts/data/multistyle_batch_v2.py` (vLLM + Gemma-4-31B
  at `http://localhost:8000/v1`). Consolidation/HF push: `scripts/data/consolidate_multistyle.py`.
  Audits: `audit_quality.py` (proxy) and `ngram_copy_audit.py` (authoritative — see §0.2).
- **Tests:** `uv run --extra dev pytest` → 17 passed, 1 skipped.
- **Dataset on HF:** `fierysurf/medcolbert-synthetic-pilot`, **100,084 examples**, clean repo
  (13 files, zero legacy `mode3`/root parquets).

### Current dataset quality (all 8 thresholds PASS — verified 2026-06-20)

| Threshold | Target | Actual | Status |
|---|---|---|---|
| UMLS artifacts | <0.5% | 0.0% | ✅ |
| Abbreviation genuine | ≥80% | 91.25% | ✅ |
| Vocabulary shift (n-gram copy) | ≥60/100 all families | 99.9/100 | ✅ |
| Hallucinated terms | <5% | 0.0% | ✅ |
| Template monotony | <30% | 0.022% | ✅ |
| Data integrity | no corruption | 97+/100 | ✅ |
| Dataset size | 100K+ | 100,084 | ✅ |
| Pushed to HF | yes | clean replace | ✅ |

Distribution: `symptom_to_diagnosis` 31,308 / `biomedical_to_clinical` 26,657 /
`brand_generic` 21,947 / `lay_to_clinical` 17,827 / `abbreviation_to_expanded` 2,345.
Styles balanced ~25K each: `keyword_technical` 26,701 / `keyword_clinical` 26,474 /
`keyword_layperson` 24,864 / `full_question` 22,045.

### NOT done — Phases 5–8 (where this plan picks up)

- **Phase 5 (triplets + negatives):** `scripts/data/05_mine_bm25_negatives.py`,
  `06_mine_dense_negatives.py`, `07_merge_and_deduplicate.py` are **0-line stubs**. No
  `src/medcolbert/triplets/` module exists.
- **Phase 6 (training):** no `src/medcolbert/train/` module; `train_app` CLI is empty.
  Configs `configs/train_base.yaml`, `configs/base.yaml`, `configs/large.yaml` exist and
  define the stage contracts.
- **Phase 7 (evaluation):** no `src/medcolbert/eval/` module; `eval_app` CLI is empty.
  `configs/eval.yaml` exists.
- **Phase 8 (release/paper):** not started.

### 0.1 The two metrics that matter (do not be fooled)

1. **Vocabulary shift = n-gram copy rate, NOT the proxy.** `audit_quality.py` reports
   88–94/100 via alt_term presence + `term_overlap`. That is misleading and was the reason
   the old dataset passed a fake bar while having 10–21/100 real vocab shift. The real
   metric is **n-gram copy rate** = fraction of query 4-grams appearing verbatim in the
   source passage. Authoritative tool:
   ```
   uv run python scripts/data/ngram_copy_audit.py --per-family
   # score = 100 * (1 - mean_4gram_copy). Good data ≈ 99/100. Threshold: ≥60/100 every family.
   ```
2. **Low `term_overlap` is expected, not a bug.** ~70% of rows have `term_overlap < 0.5`.
   Keyword queries are short and use *substituted* vocabulary — low overlap is the whole
   point. Do NOT "fix" low term_overlap by regenerating; it is not hallucination
   (hallucination is 0.0%, verified separately).

### 0.2 Generation guardrails already baked into the code (do not regress)

These are solved problems. If you regenerate data, keep them:

- **Temperatures:** 0.2 (fact extraction), 0.7 (questions), 0.8 (keywords), `top_p=0.95`,
  `top_k=64`. The old `temp=0.0` caused the template monoculture and fake vocab shift.
- **Forced question openers:** `multistyle_batch_v2.py` has `QUESTION_OPENERS` (10 openers)
  and `_pick_question_opener()` which deterministically assigns one per row via MD5 of
  `(fact, role, task_family, cui_label, alt_term)`. This killed the "Does [NP] [VP]?"
  monoculture (37% → 0.022%). If `full_question` monotony creeps above 30%, regenerate
  *just* those rows cheaply with `scripts/data/regen_full_questions.py` (~5× cheaper than
  full batches; output → `fq_v6_*` dirs).
- **Abbreviation quality (`scripts/data/prepare_abbreviation_passages.py`):** three pitfalls
  already fixed — (a) plain-word alt_terms (Lung, zinc, Crohn) rejected by requiring a
  digit OR internal capitalization + a large `_COMMON_WORDS` stoplist; (b) numeric alt_terms
  ("2", "223", "T2") rejected by requiring an alpha run ≥2 in both `abbreviated` and
  `abbreviated_norm`; (c) homonym mismatches killed by the `_content_words()` definitional
  co-occurrence filter (≥2 distinctive overlaps within 80 chars). The abbreviation pool is
  small (~830 passages, ~2.3K rows) by design — it is ~2% of the dataset; the 4 main
  families are the bulk and matter more.
- **Consolidation scans `v6_fresh/` only** (`consolidate_multistyle.py find_all_accepted()`),
  excluding legacy `ontology_grounded_teacher_filtered/` (temp=0.0). It also drops the old
  low-quality `abbreviation_to_expanded` rows from `div_v6_0001..0018` and the old templated
  `full_question` rows from `div_v6_*` (replaced by `fq_v6_*`).

### 0.3 Housekeeping rules (carry forward)

- `download_all_umls.py` is **gitignored** — it contains a hardcoded UMLS API key. Do not
  commit it. Load keys from env (`UMLS_API_KEY`) and rotate the exposed key.
- Scratch `run_batch_*.py` scripts and large CSV dumps (`keyword_queries.csv`,
  `strong_full_questions.csv`) are gitignored by design — keep them out of git.
- **Always use `uv`.** `uv run python …`, `uv run pytest`, `uv run medcolbert …`. Never bare
  `pip`/`python`. Python 3.11 or 3.12.
- **Private data stays in `data/processed/private/`.** Public manifests (counts/hashes
  only) go in `data/processed/public/manifests/`. UMLS strings, CUI mappings, synonym
  tables, matched spans, synthetic queries embedding UMLS terms are all private.
- **Library logic in `src/medcolbert/`, scripts are thin CLI shims.** Match the existing
  `data/`/`generation/` layout when adding triplets/train/eval code.

---

## 1. Phase 5 — Triplets and Negatives

**Goal:** turn the 100K accepted synthetic examples into ColBERT training triplets with
strong, audited negatives and zero eval leakage. Spec: `docs/goals/05-triplets-and-negatives.md`.

**Inputs (private):**
- Accepted synthetic examples: `data/processed/private/synthetic/consolidated/mode4_teacher_filtered_multistyle.parquet`
  (+ per-style `mode4_*.parquet`).
- Passage store: `data/processed/private/passages/` (`passages.parquet`,
  `passage_cui_matches.parquet`, and `real_annotated_500.json` / `abbreviation_passages.json`
  used by generation).
- Ontology edges/pairs: UMLS `concept_pairs.parquet` / `concept_strings.parquet`.

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

### 1.1 Triplet schema (exact — every column required)
```
triplet_id, query, positive_passage_id, negative_passage_id, source,
generation_mode, task_family, target_cuis, positive_cuis, negative_cuis,
negative_type, negative_source, teacher_margin, split, data_hash
```

### 1.2 Tasks

1. **Create the library module.** Add `src/medcolbert/triplets/` with:
   - `negatives.py` — BM25 + ontology negative mining.
   - `decontam.py` — eval blocklists + text-similarity blocking.
   - `build.py` — triplet assembly, schema enforcement, splits.
   Match the existing `data/`/`generation/` style (typed functions, deterministic IDs,
   `pathlib.Path`, short comments). Then fill the three **0-line stub scripts**
   (`05_mine_bm25_negatives.py`, `06_mine_dense_negatives.py`,
   `07_merge_and_deduplicate.py`) as thin CLI wrappers over these modules.

2. **Negative types** (mix several per query; `random` is calibration-only, low fraction):
   - `bm25_hard` — lexically similar but not the target concept. This is what
     `05_mine_bm25_negatives.py` builds (a BM25 index over the passage store).
   - `ontology_parent_child`, `ontology_sibling`, `ontology_related` — from ontology edges.
   - `same_semantic_group` — same group, different concept.
   - `dense_mined` — **only after a Stage-2 checkpoint exists (Phase 6).** Do NOT add
     dense-mined negatives before the first model is trained. This is the explicit purpose
     of `06_mine_dense_negatives.py` and it must stay a stub until Phase 6 Stage 2.
   - `random` — small fraction, calibration.

3. **False-negative controls — reject a negative if** it shares any primary target CUI with
   the query; it is a near-duplicate of the positive; it has the same
   PMID/PMCID/NCT/source-doc as the positive (accidental positive); a teacher/heuristic
   flags it relevant; manual audit repeatedly marks the type ambiguous. **When unsure, drop
   it** — ambiguous negatives poison contrastive training.

4. **Decontamination (before writing train files):** block exact eval IDs (PMID, PMCID, NCT,
   dataset-specific IDs), exact text hashes, normalized text hashes, and near-duplicates by
   MinHash/SimHash threshold. Also block synthetic queries too similar to eval queries.
   **Do not decontaminate by IDs alone — use text similarity too.** Write
   `decontamination_report.json` with counts of blocked items per method.

5. **Splits:** build `pilot_triplets.parquet` (50–100K, diversity over size) and a
   `dev_triplets.parquet` holdout. `v1_triplets.parquet` is the scaled-up version with
   source-aware sampling — no single source dominates without an explicit config weight in
   `configs/data.yaml`.

### 1.3 Gate (must pass before Phase 6)
- 50K–100K pilot triplets after decontamination.
- Every triplet carries `source`, `task_family`, `generation_mode`, `negative_type`.
- `negative_audit_samples.parquet` shows acceptable false-negative rate on manual review
  (sample ≥200, aim <5% ambiguous/wrong).
- `decontamination_report.json` shows zero known eval leakage.
- Ablation subsets selectable by `generation_mode` and `negative_type` (verify with a count query).

### 1.4 Pitfalls
- A single accidental-positive negative can dominate a batch's loss. Sample negatives, log
  `negative_source`, and audit before training.
- BM25 negatives that share the target CUI are false negatives — the CUI reject rule above
  is mandatory, not optional.

---

## 2. Phase 6 — ColBERT Training

**Goal:** train the model variants that isolate the value of ontology-controlled
supervision. Spec: `docs/goals/06-colbert-training.md`. Backbone:
`thomas-sounack/BioClinical-ModernBERT` (`configs/base.yaml`). Framework: **PyLate**
(already in the `training` extra: `pylate>=1.0.0`).

### 2.1 Model variants — train IN THIS ORDER (each is a controlled comparison)
1. `generic_colbert` — off-the-shelf ColBERT/PyLate baseline.
2. `bioclinical_colbert_no_ontology` — same backbone, no ontology-controlled generation.
3. `medcolbert_ontology_grounded` — ontology-controlled synthetic supervision (the 100K dataset).
4. `medcolbert_teacher_filtered` — + modern LLM/reranker filtering (if a filtered subset is built).
5. `medcolbert_dense_negative_refresh` — continue/retrain with dense-mined negatives
   (requires the Stage-2 checkpoint → closes the loop with Phase 5 §1.2).

**Large-model training is optional** and only after base ablations support the hypothesis.

### 2.2 Tasks

1. **Add `train` commands to the CLI** (`src/medcolbert/cli.py`, `train_app` is registered
   but empty). Suggested: `medcolbert train smoke`, `medcolbert train run --stage`,
   `medcolbert train index`. Put training logic in a new `src/medcolbert/train/` module;
   keep `cli.py` thin (mirror how `synth`/`data` are wired).

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

### 2.3 Critical discipline (common-mistakes, from the spec)
- Do not compare MedColBERT to a weaker-backbone baseline only — the no-ontology
  BioClinical-ColBERT control (#2) is mandatory.
- Do not change backbone, data, negatives, and loss all at once — one variable per variant.
- Do not overfit raw UMLS synonym strings.
- Do not train a large model before base ablations are interpretable.

### 2.4 Gate
- `bioclinical_colbert_no_ontology` and `medcolbert_*` both train successfully on the same
  backbone with comparable compute.
- Training curves stable (no loss spikes/NaN).
- Tiny-retrieval sanity passes (positives rank above negatives).
- Candidate indexes build.
- Every variant is reproducible from its config + manifest files.

---

## 3. Phase 7 — Evaluation and Claims

**Goal:** evaluate on real external benchmarks + vocabulary-shift slices; state only
evidence-supported claims. Spec: `docs/goals/07-evaluation-and-claims.md`. Config:
`configs/eval.yaml`.

### 3.1 Tasks

1. **Add `eval` commands to the CLI** (`eval_app` is registered but empty). Suggested:
   `medcolbert eval run --benchmark`, `medcolbert eval table`, `medcolbert eval slice`.
   Logic in a new `src/medcolbert/eval/` module.

2. **Primary benchmarks** (focused set first — `configs/eval.yaml` already enables these):
   R2MED (reasoning), TREC Clinical Trials (long clinical→trial), PMC-Patients
   (case retrieval), and 1–2 BEIR/MTEB biomedical tasks (NFCorpus, SciFact, BioASQ, or
   TREC-COVID). **Add MIRAGE/MedRGB only after retrieval quality is established** — RAG
   benchmarks entangle retriever with generator/prompt.

3. **Required metrics:** nDCG@10, Recall@100, Recall@1000 (when candidate recall matters),
   MAP@10 (benchmark-standard), index size, query latency, encoding throughput, MRL
   32/64/128 tradeoff (when MRL enabled).

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

### 3.2 Claim discipline
- Acceptable: *"MedColBERT achieves competitive or state-of-the-art results on selected
  biomedical and clinical retrieval benchmarks, with the largest gains on
  vocabulary-shift slices."*
- Avoid: *"MedColBERT is SOTA on medical retrieval."* or *"Dense retrieval cannot solve
  medical synonymy."*
- Do not tune prompts/data generation against the test set. Do not report RAG gains as
  pure retrieval gains.

### 3.3 Gate
- All eval corpora decontaminated from training (cross-check
  `data/processed/public/manifests/decontamination_report.json`).
- Baselines run with comparable corpora + documented settings.
- No-ontology ColBERT control included.
- Vocab-shift buckets reported. Efficiency reported. Claims benchmark-scoped.

---

## 4. Phase 8 — Release and Paper (brief)

Spec: `docs/goals/08-release-and-paper.md`. Only after Phase 7 gates pass.
- License-safe code + model release; UMLS-derived artifacts stay private/gated.
- Paper tables/plots from `runs/eval/*`.
- Release audit (data provenance, decontamination logs, reproducibility from
  config+manifest).
- Exit criteria: results support a precise, benchmark-scoped claim; cross-vocabulary
  analysis supports the core mechanism; UMLS artifacts remain private/gated.

---

## 5. Cross-cutting rules (apply to every phase)

1. **Always use `uv`.** `uv run python …`, `uv run pytest`, `uv run medcolbert …`,
   `uv sync --extra <group>`. Never bare `pip`/`python`/`pytest`. Python 3.11 or 3.12.
2. **Private data stays in `data/processed/private/`.** Public manifests (counts/hashes
   only) go in `data/processed/public/manifests/`.
3. **Library logic in `src/medcolbert/`, scripts are thin CLI shims.** Match the existing
   module layout. Keep `cli.py` thin.
4. **One variable per change.** Don't change backbone + data + negatives + loss together.
5. **Verify before declaring done.** Run the audit/gate of the current phase; report real
   numbers. If a gate fails, say so — don't paper over it.
6. **Commit per phase** with a clear message; keep the `dev` branch. Don't commit secrets.
7. **Re-read the relevant `docs/goals/0N-*.md`** before starting each phase — it is the
   authoritative spec; this plan is the execution sequence + pitfalls on top of it.

---

## 6. Suggested execution order for a fresh agent

1. Read `AGENTS.md` + `GOALS.md` + this plan + `docs/goals/05-triplets-and-negatives.md`.
2. **Phase 5:** build `src/medcolbert/triplets/`, fill the `05/06/07` scripts (keep `06`
   a stub until Phase 6 Stage 2), hit §1.3 gate.
3. **Phase 6:** add `train` CLI commands, Stage 0 → 4, train variant #2 then #3, hit §2.4 gate.
4. **Phase 7:** add `eval` CLI commands, run benchmarks + slices + baselines, hit §3.3 gate.
5. **Phase 8:** release + paper.

If you need to regenerate synthetic data at any point (you should not need to — the
dataset passes all gates), the recipe is in `docs/audit_2026-06-20.md` + the generation
guardrails in §0.2 above. Do not regenerate casually; it is 6–8 hours of vLLM compute.
