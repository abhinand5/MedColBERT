# MedColBERT Hard Negatives — Final QA Report (v1, full run)

**Date:** 2026-06-24
**Dataset:** `negatives_v1.parquet` — 724,337 kept negatives across 53,703 train positives
**Method:** 5 independent subagents (general-purpose), one per task family, each reading a 60-row stratified sample of (query, positive, negative) triples and scoring `false_negative`, `mode_correct`, `grounding`, `hardness` — **without trusting the audit gate's own scores**. Calibrated discipline (a hard negative is *supposed* to look retrievable and be wrong — do not over-flag).

## Verdict: SHIP

| family | n sampled | false_negative | verdict |
|---|---|---|---|
| symptom_to_diagnosis | 60 | 0 | PASS |
| biomedical_to_clinical | 60 | 0 | PASS |
| generic_clinical_retrieval | 59 | 0 | PASS |
| lay_to_clinical | 60 | 0 | PASS |
| brand_generic | 60 | 0 | PASS |
| **TOTAL** | **299** | **0** | **SHIP** |

**Subagent-verified false-negative rate: 0/299 = 0.0%, Wilson 95% upper CI = 1.27%** — well under the 5% ship criterion, and matching the 2k pilot's 0/300. The audit gate's false-negative catch is sound at 724k scale. This is the only number that matters for training safety (a false negative teaches the model the right answer is wrong).

## Generation quality (per source, from the subagent reads)

| source | n (sampled) | mean hardness | mean grounding | false-neg |
|---|---|---|---|---|
| synthetic | 253 | 4.4–4.8 | ~5 | 0 |
| bm25_real | 46 | 3.0–3.6 | 5.0 | 0 |

**Synthetic negatives are uniformly excellent** — every family's synthetic tiers (confusable_disease, wrong_stage_severity, treatment_diagnosis_confound, wrong_context, lexical_trap, abbreviation_trap, biomedical_clinical_mismatch, domain_boundary, same_group_wrong_concept_synth) produce coherent, on-topic-but-wrong passages at hardness 4+. Standouts called out by subagents: `abbreviation_trap` (anti-HBs = anti-human blocking serum vs hepatitis B surface Ab; CAI = cognitive assessment vs instructional method), `confusable_disease` (organic acidemias mimicking DKA with an explicit "do not give insulin" warning), `same_group_wrong_concept_synth` (Billroth II vs Billroth I; postsynaptic density vs presynaptic autoreceptor).

**`lay_clinical_mismatch` gibberish concern — RESOLVED.** The pilot flagged a risk this tier produces incoherent text. At scale, all sampled `lay_clinical_mismatch` negatives are coherent, fluent lay-English passages with clinically-wrong content. The risk did not materialize.

## The one finding: bm25_real weak tail

Across **all 5 families**, subagents independently flagged that ~24% of `bm25_real` negatives are **trivially irrelevant** (hardness 1–2, sharing only 1–2 words with the query — e.g. a UTI-colony-count paper for a kuru-epidemiology query). This is a BM25 lexical-match artifact: a single high-IDF term match surfaces a topically-unrelated real passage. The synthetic weak rate is 1.8%; bm25_real weak rate is 23.9%.

**Impact:** trivially-irrelevant negatives are *harmless dilution*, not corruption — they're easy negatives the model would learn from in-batch anyway. They don't introduce false signal. But for a fixed negative budget per positive, they waste slots that harder negatives could fill.

## Clean variant produced

`negatives_v1_clean.parquet` — applies the QA-justified weak filter (drop `audit_relevant_looks <= 2`, with a floor guaranteeing ≥1 negative/positive and keeping floor-4 where a positive has few):

| | negatives_v1 | negatives_v1_clean |
|---|---|---|
| rows | 724,337 | 711,301 |
| neg/pos | 13.49 | 13.25 |
| weak_rate | 4.90% | 3.15% |
| positives | 53,703 | 53,703 (all kept) |
| dropped | — | 13,036 (1.8%, mostly bm25_real trivially-irrelevant) |

**Recommendation:** use `negatives_v1_clean.parquet` for training (signal-dense; the dropped 1.8% were the subagent-confirmed dead weight). Keep `negatives_v1.parquet` as the full audited source of truth.

## Audit gate performance at scale (from the dropped file)

| drop reason | count | meaning |
|---|---|---|
| answers_query_true | 43,050 | **false negatives caught** (5.61% pre-audit rate) — the gate's primary job |
| grounding_le2 | 20,184 | gibberish / incoherent passages caught |
| intra_sample_near_dup | 686 | Gemma repeating itself within a sample |
| audit_unparsed | 0 | (Gemma parses cleanly; this is a DeepSeek-thinking artifact) |

The gate caught 43,050 false negatives that would otherwise have corrupted training — confirming the calibrated `ANSWERS_QUERY` + `GROUNDING` discipline works at scale.

## Files

| file | rows | purpose |
|---|---|---|
| `negatives_v1.parquet` | 724,337 | full audited negatives (long format) — source of truth |
| `negatives_v1_clean.parquet` | 711,301 | weak-filtered, signal-dense — **use for training** |
| `negatives_v1_dropped.parquet` | 63,920 | audit drops (false-neg + gibberish + near-dup), labeled |
| `negatives_v1_weak_dropped.parquet` | 13,036 | QA-justified weak drops |
| `triplets_dev_candidates.parquet` | 5,967 | held-out 10% real eval split (UNTOUCHED by generation) |

## Boundary — what's next (training phase)

This completes the **negatives phase**. The triplets/training phase is next and is deliberately out of scope here:
- pylate (the ColBERT framework) is **not installed**; training scripts are **empty stubs**.
- The triplet collator schema is a training-phase design decision (tied to pylate's expected format) — do not invent it now.
- First training-phase steps: install pylate, design the triplet collator consuming `negatives_v1_clean.parquet`, build `triplets_v1.parquet` + `triplets_dev.parquet` (referenced by `configs/train_base.yaml`), then the actual ColBERT finetune, then the real-corpus retrieval eval on the held-out dev split (the train-on-synthetic gate, plan §12).
