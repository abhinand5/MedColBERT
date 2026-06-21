# MedColBERT — Synthetic Hard-Negative & Triplet Mining Plan

**Status:** ready to execute (pilot first, then full run)
**Owner-of-record:** session owner
**Date:** 2026-06-21
**Companion runner:** `scripts/data/mine_synthetic_negatives.py`

---

## 0. TL;DR / how to use this document

We have a clean set of **59,670 real-PubMed (query, positive_passage) positives**
(`mode4_teacher_filtered_multistyle_cleaned_v3g.parquet`) and a **66,108-passage
real PubMed corpus** UMLS-annotated per passage. This plan turns those positives into
**hard negatives** for ColBERT contrastive training, using a three-part recipe:

1. **Synthetic negatives (primary, ~11–15/sample)** — Gemma-4-31B writes
   taxonomy-routed, style-conditioned hard negatives. Each negative fails the query
   in a *specified clinical-failure mode* (confusable disease, wrong drug in class,
   abbreviation trap, lay/clinical mismatch, …).
2. **Real anchor (~2/sample)** — mined from the 66k real corpus, no LLM: a BM25
   hard negative + a UMLS category-aware real negative (same `semantic_group`,
   different concept). Anchors the model on real text so it cannot learn
   "spot the LLM distractor."
3. **Real eval split (held-out 10%)** — real queries + real positives, with a BM25
   recall@k sanity now and a neural retrieval eval after the model is trained.

Every negative (synthetic *and* real) passes a **calibrated Gemma audit gate** that
drops **false negatives** (a negative that accidentally *answers* the query — the
most toxic object in contrastive training) and incoherent/garbage passages.

**Run order:** (a) smoke `--limit 24`, (b) **pilot `--limit 2000`** (stratified),
hand-inspect a 200-sample stratified slice, (c) **full run `--limit 0`** (≈60k).
After each run the owner returns to run the **subagent QA step** (§13) which
random-samples N negatives per tier/family and decides PASS / FAIL / RETUNE.

---

## 1. Goal & deliverables

**Goal:** produce a high-quality hard-negative table for every positive in v3g, such
that a ColBERT trainer can mine triplets (`query`, `positive`, `negatives[]`) from it
with low label noise.

**Deliverables from the runner:**

| File | Rows | Purpose |
|---|---|---|
| `negatives_v1.parquet` | one row per kept negative | the negatives table (long format) |
| `negatives_v1_dropped.parquet` | one row per dropped negative | audit + near-dup drops, labeled |
| `negatives_v1_qa_sample.parquet` | ~200 stratified kept negatives | for the subagent QA step |
| `triplets_dev_candidates.parquet` | held-out 10% positives + BM25 recall@100 | real-eval split + sanity |
| `negatives_v1_run_summary.json` | — | counts, drop rates, per-tier/per-family tables |

**Non-deliverables (downstream, documented in §12):** the ColBERT triplet collator,
the neural retrieval-eval run (needs a trained model), pylate install. The runner does
not depend on any of those.

---

## 2. The one real risk, and the three mitigations

At inference ColBERT ranks a query against **real** passages from the index — never
LLM-written distractors. If 100% of training negatives are Gemma-written, the model
can silently learn "spot the LLM-distractor texture" instead of "spot the
real-but-wrong passage." You will not see this on a synthetic eval; you see it the day
you retrieve against a real corpus.

**Mitigation 1 — style-condition every synthetic negative on its real positive.**
Every generation prompt is fed the actual `passage_text` of the positive and
instructed to match its register/length/source-voice (PubMed-abstract style, clinical
note style, etc.). No bullets, no meta-commentary, no "as an AI." This single rule
removes most of the texture gap. (See §6, constraint #1.)

**Mitigation 2 — a real anchor mined from the 66k real corpus, every sample.**
1 BM25 hard negative + 1 UMLS category-aware real negative per sample, drawn from real
PubMed text. Synthetic dominates (~88%), real anchors (~12%). The real negatives are
*also* audited (a BM25 passage can be a false negative). (See §7.)

**Mitigation 3 — a real-passage held-out eval is non-negotiable.**
10% of positives are held out as a real eval split (their positives are *not* used to
generate negatives). The runner does a BM25 recall@k sanity on them *now* (validates
corpus↔query compatibility). A full neural retrieval eval (MRR/Recall@k against the
real corpus) is run **after** the model trains — this is the only gate that catches
train-on-synthetic overfit. Documented in §12; not in the runner because it needs the
trained model.

---

## 3. Corpus facts (verified)

- **Real corpus:** `data/processed/private/passages/real_annotated_500.json`
  (identical to `real_annotated.json`), **66,108 entries**, 91.8 MB. Per-passage fields:
  `passage_id` (hex), `text`, `title`, `source_doc_id`, `cui_label`, `alt_term`,
  `semantic_group`, `vocab_shift_type`. UMLS-annotated.
- **Positive passages are REAL PubMed text** — the synthetic-generation LLM only wrote
  the `query`/`fact` columns; `passage_text` is real PubMed abstract text drawn from
  this same MedRAG/pubmed universe. (Verified in the parquet: e.g. "Cyclic nucleotide
  metabolism in compensatory renal hypertrophy…".)
- **v3g uses 17,462 unique `passage_id` / 17,404 unique `passage_text`.** The v3g
  `passage_id` namespace (`pubmed_pubmed23nXXXXX_NNNN_c`) differs from the corpus
  `passage_id` (hex). **Matching positive↔corpus is done on TEXT, not id.**
- **~48,700 real passages are unused as positives** → the real hard-negative pool.
  `semantic_group` values present: `disorders`, `drugs_chemicals`, `procedures`,
  `anatomy`, `physiology`, … (16 groups). Used for the category-aware real negative.
- **Dropped-rows files = free Tier-5 negatives** (already generated + labeled, no LLM):
  `cleaned_v3_dropped_rows.parquet` (15,776), `cleaned_v3f_dropped_rows.parquet`
  (2,362), `cleaned_v3g_dropped_rows.parquet` (767). These contain refusal/meta-query
  rows and Gemma-flagged-hallucinated rows — perfect garbage/refusal negatives (Tier 5B).

---

## 4. The negative taxonomy (designed around *user query failure modes*)

ColBERT is a **late-interaction** ranker: it scores on token-level interactions. Tiers
that differ from the positive at a *fine token level* (same sentence, swapped drug
name; same passage, swapped dose) are disproportionately informative — they force
token-level discrimination. **Lean into Tier 2 and Tier 4.** Tier 1 and Tier 5 are
easy; the model largely learns them from in-batch negatives, so we keep their counts
low.

| Tier | Failure mode | Why it matters / what it teaches |
|---|---|---|
| **T1 lexical_trap** | Shares 2–3 query terms, different condition/topic. Surface overlap, wrong subject. | Don't rank on shallow lexical match. (Easy; low count.) |
| **T2A confusable_disease** | A disease/condition clinically confusable with the query target (overlapping presentation, same symptom cluster, related pathophys) but a **different diagnosis**. Reads like a plausible differential a clinician weighs then rejects. | The clinical-reasoning failure. **High value.** |
| **T2B wrong_drug_same_class** | A **different drug in the same therapeutic class** as the query target (e.g. metformin vs sitagliptin). | Drug-class confusion; where brand↔generic confusion lives. **High value.** |
| **T2C wrong_stage_severity** | A wrong **stage / severity / variant** of the target (e.g. T1DM ketoacidosis vs T2DM management; acute vs chronic). | Token-level discrimination of the *same* concept. **High value for ColBERT.** |
| **T2D wrong_dose_route** | A wrong **dose / route / administration** of the target drug (IV push vs oral; pediatric vs adult dose). | Token-level swap on the *same* drug. **High value for ColBERT.** |
| **T3A lay_clinical_mismatch** | Layperson-phrased passage that is clinically **wrong** for the query. Targets `lay_to_clinical`. | Register + content mismatch. |
| **T3B abbreviation_trap** | Answers a plausible **wrong expansion** of an ambiguous abbreviation in the query ("MS" → multiple sclerosis vs mitral stenosis vs morphine sulfate). | Re-hardens the abbreviation skill **as a negative** (too noisy as a positive). |
| **T3C biomedical_clinical_mismatch** | Literature/biomedical-phrased passage that is the **wrong clinical answer**. Targets `biomedical_to_clinical`. | Register + content mismatch. |
| **T4A wrong_context** | Right topic, **wrong patient population / context** (pediatric vs adult; acute vs chronic; animal model vs human). | Looks-on-topic, contextually wrong. **High value.** |
| **T4B treatment_diagnosis_confound** | Query asks **treatment** → passage **diagnoses**; query asks diagnosis → passage treats. | Intent vs content mismatch. **High value.** |
| **T5A domain_boundary** | Non-medical but lexically on-topic (chemistry/synthesis passage for a drug query; basic-science for a clinical query). | Domain boundary; cheap. |
| **T5B garbage_refusal** | Reuse the dropped refusal / meta-query / hallucinated rows. **No LLM cost.** | Robustness to junk. |

### Tier ↔ ColBERT note
- **Highest ROI (token-level):** T2A, T2B, T2C, T2D, T4A, T4B. ColBERT *needs* these.
- **Register-targeting (the named families):** T3A, T3B, T3C — directly exercise
  `lay_to_clinical`, abbreviation, `biomedical_to_clinical`.
- **Cheap filler:** T1, T5A, T5B. Keep low; in-batch negatives cover them.

---

## 5. Family → tier routing table

Not every tier applies to every family. Counts below are the **target negatives per
positive** for that tier. Real anchors add +2 (where available).

| task_family | T1 | T2A | T2B | T2C | T2D | T3A | T3B | T3C | T4A | T4B | T5A | T5B | synth total | + real | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| symptom_to_diagnosis | 2 | 4 | – | 3 | – | – | – | – | – | 3 | – | if avail | 9 (+T5B) | +2 | ~11 |
| biomedical_to_clinical | 2 | 2 | – | – | – | – | – | 4 | 2 | – | 2 | if avail | 10 | +2 | ~12 |
| generic_clinical_retrieval | 3 | 2 | – | – | – | – | 2 | – | 2 | – | 2 | if avail | 11 | +2 | ~13 |
| lay_to_clinical | 2 | 2 | – | – | – | 4 | – | – | 3 | – | – | if avail | 11 | +2 | ~13 |
| brand_generic | 2 | – | 5 | – | 3 | – | 2 | – | 1 | – | – | if avail | 13 | +2 | ~15 |

**Real anchors (every family):** +1 BM25 hard negative (T1-style but REAL) + 1
category-aware real negative (T2A/T2B-style but REAL: same `semantic_group`, different
concept). Total per sample ≈ **11–15 negatives**, matching the requested "10–15 per sample."

**T5B (garbage_refusal)** is added opportunistically: for each sample, if a dropped-row
of the same `task_family` is available, attach up to 1 (free, no generation). The
runner cycles through the dropped-rows pool by family.

> Routing is a `dict` in the runner (`FAMILY_ROUTING`). To retune counts, edit that
> dict. The audit gate + QA step will tell you if a tier is under/over-producing.

---

## 6. Generation recipe (Gemma-4-31B)

**Fixed sampling params — DO NOT CHANGE:** `temperature=1.0`, `top_k=64`, `top_p=0.95`.
These are the owner-specified creative-generation settings; they produce diverse
negatives. (Confirmed vLLM accepts `top_k`/`top_p` as top-level request fields — the
repo's `multistyle_batch_v2.py` already does this.)

**Thinking control:** `enable_thinking=False` via `chat_template_kwargs` (top-level
request field). Disables the model's reasoning trace for speed and keeps the output to
just the negative passages (no `<thinking>` tokens to parse around).

**Generation call shape (httpx, OpenAI-compatible):**
```python
payload = {
    "model": "gemma-4-31B",
    "messages": [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": user}],
    "temperature": 1.0, "top_p": 0.95, "top_k": 64,
    "max_tokens": 1400,            # generous: 3–5 passage-length negatives
    "chat_template_kwargs": {"enable_thinking": False},
}
# POST to http://localhost:8000/v1/chat/completions
```

**Audit call shape (different sampling — deterministic judging):**
```python
payload = {
    "model": "gemma-4-31B",
    "messages": [...],
    "temperature": 0.0,            # deterministic for judging
    "top_p": 1.0, "top_k": -1,     # vLLM defaults = full distribution
    "max_tokens": 160,
    "chat_template_kwargs": {"enable_thinking": False},
}
```
Audit uses temperature 0.0 and **no** creative sampling — judging must be reproducible.
(If the pilot shows the judge is too lenient at catching false negatives, enable
thinking for the *audit* only in the full run; generation stays thinking-off for speed.
See §16, abort/retune conditions.)

### 6.1 Shared SYSTEM prompt (NEGATIVE WRITER)

```
You are a senior medical-information-retrieval engineer writing HARD NEGATIVE passages for a contrastive retrieval trainer (ColBERT). A hard negative is a passage that LOOKS retrievable for a given query but does NOT answer that query's specific intent — it is wrong in a precise, specified way.

You will be given: a QUERY, the POSITIVE PASSAGE (the correct answer), the FACT the positive establishes, the intended FAILURE_MODE for each negative, and how many to write.

HARD CONSTRAINTS for EVERY negative passage:
1. REGISTER MATCH — Write in the SAME register, length, density, and source-voice as the POSITIVE PASSAGE. If the positive is a PubMed abstract, write a PubMed-style abstract (single paragraph, ~60–160 words, third-person, no headers). If clinical-note-style, match that. NO bullet points, NO headers, NO "as an AI", NO meta-commentary, NO preamble, NO "here is a negative". It must read as genuine medical literature/clinical text.
2. RETRIEVABLE — The passage must plausibly be retrieved for the QUERY: share enough topical/lexical surface (terms, concept, domain) that a retriever would consider it a candidate. It is NOT a random off-topic passage.
3. DOES NOT ANSWER — The passage must NOT satisfy the query's specific intent. It must be wrong in the exact FAILURE_MODE specified. It must not contain the positive's answer or the positive's FACT.
4. NO LEAKAGE — Do not copy or lightly paraphrase the POSITIVE PASSAGE. Do not reuse its specific result, drug name, dose, or finding. A paraphrase of the positive is a FALSE negative (it corrupts training).
5. MEDICALLY COHERENT — Realistic, internally consistent medical text. You may use real medical facts and real drug/disease names — groundedness is good. The passage may be a real-sounding statement about a DIFFERENT concept than the query asks.
6. DIVERSITY — When writing several negatives, make them genuinely distinct (different wrong concept, different failure mechanism), not minor rewordings of each other.

OUTPUT FORMAT — exactly this, one block per negative, no prose before/after:
<<<NEG>>>
PASSAGE: <one paragraph, ~60–160 words, in the positive's voice>
WHY_WRONG: <one sentence: how it fails the query and in what way>
<<<END>>>

Write the requested number of negatives. Quality over quantity. A passage that accidentally ANSWERS the query is worthless and will be dropped.
```

### 6.2 Per-tier FAILURE_MODE specs (appended to the USER prompt)

| Tier | FAILURE_MODE text appended to the user prompt |
|---|---|
| **T1 lexical_trap** | `FAILURE_MODE: lexical_trap. Write a passage that shares 2–3 surface terms with the QUERY but is about a DIFFERENT condition/topic. A retriever fooled by word overlap would fetch it; a reader sees it is off-subject.` |
| **T2A confusable_disease** | `FAILURE_MODE: confusable_disease. Write about a disease/condition clinically CONFUSABLE with the query's target — overlapping presentation, same symptom cluster, or related pathophysiology — but a DIFFERENT diagnosis. It must read like a plausible differential a clinician would weigh, then reject. It must share symptoms/terminology with the query so a retriever is fooled, but the diagnosis discussed is NOT the query's target.` |
| **T2B wrong_drug_same_class** | `FAILURE_MODE: wrong_drug_same_class. Write about a DIFFERENT drug in the SAME therapeutic class as the query's target drug. Same class/indication family so it looks retrievable, but it is the wrong agent. Do not use the positive's drug.` |
| **T2C wrong_stage_severity** | `FAILURE_MODE: wrong_stage_severity. Write about a WRONG stage / severity / variant of the query's target condition (e.g. acute vs chronic; mild vs severe; Type 1 vs Type 2; pediatric vs adult presentation). Same disease family, wrong specificity — forces token-level discrimination.` |
| **T2D wrong_dose_route** | `FAILURE_MODE: wrong_dose_route. Write about the query's target DRUG but with a WRONG dose, route, or administration detail (e.g. IV push vs oral; loading vs maintenance; pediatric vs adult dose). Same drug, wrong administration token.` |
| **T3A lay_clinical_mismatch** | `FAILURE_MODE: lay_clinical_mismatch. Write in LAYPERSON wording but with CLINICALLY WRONG content for the query. The register matches a lay query but the answer is medically incorrect for what is asked. Do not use the positive's answer.` |
| **T3B abbreviation_trap** | `FAILURE_MODE: abbreviation_trap. The query contains (or implies) an ambiguous abbreviation. Write a passage that answers a PLAUSIBLE BUT WRONG expansion of that abbreviation (e.g. "MS" expanded as multiple sclerosis when the query meant morphine sulfate, or vice versa). It must look like a correct answer to the wrong interpretation.` |
| **T3C biomedical_clinical_mismatch** | `FAILURE_MODE: biomedical_clinical_mismatch. Write in biomedical/literature phrasing but give the WRONG clinical answer for the query. The register matches a biomedical query but the content does not answer what is asked.` |
| **T4A wrong_context** | `FAILURE_MODE: wrong_context. Write about the RIGHT topic/concept but the WRONG patient population or context (pediatric vs adult; acute vs chronic; animal model vs human; prophylaxis vs treatment). On-topic, contextually wrong.` |
| **T4B treatment_diagnosis_confound** | `FAILURE_MODE: treatment_diagnosis_confound. If the QUERY asks about TREATMENT, write a passage that DIAGNOSES the condition instead (and vice versa). Right condition, wrong clinical action axis — intent vs content mismatch.` |
| **T5A domain_boundary** | `FAILURE_MODE: domain_boundary. Write a NON-MEDICAL (or basic-science / chemistry) passage that is lexically on-topic for the query. For a drug query, a chemistry/synthesis passage; for a clinical query, a basic-science mechanism passage. It shares vocabulary but is outside the clinical retrieval domain.` |

(T5B is not generated — it is pulled from the dropped-rows pool. See §7.3.)

### 6.3 USER prompt template (per positive × tier)

```
QUERY: {query}
QUERY_FAMILY: {task_family}  ({query_style})
INTENDED CONCEPT: cui_label={cui_label}  alt_term={alt_term}  semantic_group={semantic_group}
FACT (what the POSITIVE establishes — do NOT repeat this): {fact}

POSITIVE PASSAGE (the correct answer — match its voice/length; do NOT copy it):
{passage_text}

{FAILURE_MODE line for this tier}

Write {n} such negatives, each a DIFFERENT wrong concept. Follow the output format exactly.
```

### 6.4 Robust parsing

Output is delimited with `<<<NEG>>> … <<<END>>>` blocks. The runner splits on these
markers, extracts `PASSAGE:` and `WHY_WRONG:` lines (passage may span multiple lines —
take everything after `PASSAGE:` up to the `WHY_WRONG:` line). A block missing a
non-empty PASSAGE is dropped as a parse failure (counted in the summary). This format is
deliberately tolerant of Gemma's formatting quirks (no strict JSON to mis-emit).

### 6.5 Intra-sample near-duplicate drop

After audit, within each sample's kept negatives, drop pairs with **4-gram overlap
≥ 0.8** (a near-identical negative is redundant — and usually a sign Gemma repeated
itself). This also catches the "minor rewording of another negative" failure. The 4-gram
overlap helper is the same style as the repo's `ngram_copy_audit.py` logic.

---

## 7. Real anchor (from the 66k corpus, no LLM)

Two real negatives per positive where available.

### 7.1 Inline BM25 (no new dependency — pure Python)

The repo has no BM25 library installed (`pyserini`/`rank_bm25` are not installed). The
runner implements BM25 inline:

- **Tokenize:** lowercase, regex `\b[a-z0-9][a-z0-9\-]{1,}\b`, drop a short medical
  stoplist (`the, of, and, to, a, in, for, is, with, by, on, as, at, an, be, or, from,
  was, were, this, that, these, those, it, its`).
- **Index (built once over 66,108 corpus texts):** inverted index
  `term -> [(doc_idx, tf), ...]`, `doc_len[]`, `avgdl`.
- **Score:** BM25 (`k1=1.2`, `b=0.75`). For a query, union the postings of its terms,
  accumulate per-doc score, keep top-30.
- **Exclude the positive + near-dups:** drop any top-30 corpus passage whose **4-gram
  overlap with the positive passage ≥ 0.6** (same abstract / chunk). The positive itself
  is in the corpus and would otherwise rank #1.
- **Take the top-1 surviving** as the BM25 hard negative (label it `bm25_real`,
  failure_mode `lexical_trap_real`).

This is a **real-passage Tier-1 (lexical trap) negative** — a real PubMed abstract that
shares query terms but is a different paper/concept.

### 7.2 UMLS category-aware real negative

For each positive, pick a **real corpus passage in the SAME `semantic_group`** as the
positive but with a **different `cui_label`**, excluding any whose 4-gram overlap with
the positive ≥ 0.6. Prefer (tie-break) passages whose `vocab_shift_type` differs from the
positive's (adds register variety). Sample 1 deterministically (seeded by `example_id`
hash) so the run is reproducible. Label `category_aware_real`, failure_mode
`same_group_wrong_concept_real`.

This is a **real-passage Tier-2 (confusable-same-category) negative** — a real PubMed
abstract about a same-category-but-wrong concept. For `disorders` positives → a real
passage about a different disorder; for `drugs_chemicals` → a different drug/chemical;
etc. This is the highest-value real anchor because it is *both real and in a specified
clinical-failure mode*.

> Matching is on `semantic_group` (clean categorical), **not** on CUI, because both the
> v3g `cui_label` and the corpus `cui_label` are in mixed format (sometimes a CUI like
> `C0031412`, sometimes a concept name like `Cyclic guanosine monophosphate`). The v3g
> `resolved_cui` column is used only opportunistically. `semantic_group` is reliable on
> both sides.

### 7.3 T5B garbage/refusal (free, from dropped rows)

The three `cleaned_v*_dropped_rows.parquet` files contain refusal/meta-query rows and
Gemma-flagged hallucinated rows — already generated and labeled. For each positive, if a
dropped row of the **same `task_family`** is available in the pool, attach up to 1 as a
`garbage_refusal` negative (passage = the dropped row's `passage_text`, query-style
mismatch). The runner cycles through each family's dropped pool deterministically. These
skip generation AND skip the audit grounding check (they're intentionally junk) but are
still checked for `ANSWERS_QUERY` (a dropped row could in principle answer the query —
rare, but the audit catches it).

---

## 8. Audit gate (calibrated Gemma judge — the false-negative killer)

Every negative — synthetic AND real (including T5B's `ANSWERS_QUERY` check) — is audited.

### 8.1 The judge

A **calibrated** Gemma judge (the same discipline that fixed the Round-2 over-firing:
only flag for a *specific* failure, never over-flag normal variation). For negatives the
critical failure is the **false negative** — a negative that *answers* the query. That is
the analogue of the positive-audit's `fact_grounding ≤ 2` hallucination signal.

**Audit SYSTEM prompt:**
```
You are auditing a HARD NEGATIVE for a contrastive retrieval trainer. You are given the QUERY, the POSITIVE PASSAGE (the correct answer), a CANDIDATE NEGATIVE PASSAGE, and its intended FAILURE_MODE.

Score strictly but DO NOT over-flag. A hard negative is SUPPOSED to look retrievable and be wrong — that is its job, not a flaw.

Judge:
- RELEVANT_LOOKS (1-5): would a retriever plausibly rank this passage for the query? (5=clearly on-topic; 3=borderline candidate; 1-2=too easy/irrelevant — still usable but a weak negative)
- ANSWERS_QUERY (TRUE or FALSE): does the passage actually satisfy/answer the query's specific intent? TRUE = this is a FALSE NEGATIVE (a correct answer mislabeled as a negative — it would corrupt training). Default FALSE unless the passage clearly and correctly answers what the query asks.
- GROUNDING (1-5): is the passage medically coherent/realistic? (5=realistic real-style medical text; 1-2=gibberish, self-contradictory, or hallucinated nonsense). A passage may contain REAL medical facts about a DIFFERENT concept than the query — that is correct hard-negative behavior, score 4-5.
- IN_FAILURE_MODE (TRUE or FALSE): is the passage wrong in (or close to) the specified FAILURE_MODE? FALSE = mislabeled; it may still be a valid negative of some other kind (do not drop solely for this).

Respond EXACTLY (one line each, no prose before/after):
RELEVANT_LOOKS: <1-5>
ANSWERS_QUERY: <TRUE or FALSE>
GROUNDING: <1-5>
IN_FAILURE_MODE: <TRUE or FALSE>
NOTE: <one short line>
```

**Audit USER prompt:**
```
QUERY: {query}
INTENDED FAILURE_MODE: {failure_mode}

POSITIVE PASSAGE (correct answer):
{positive_passage}

CANDIDATE NEGATIVE PASSAGE:
{negative_passage}

Judge per the rules above.
```

### 8.2 Drop decision

A negative is **DROPPED** iff any of:
- `ANSWERS_QUERY == TRUE`  → **false negative** (most toxic; the gate's primary job)
- `GROUNDING <= 2`         → incoherent / hallucinated gibberish (would teach "gibberish is a negative")

Otherwise **KEPT**. `IN_FAILURE_MODE == FALSE` → kept but `failure_mode` relabeled to
`"unspecified"` (flagged for the QA step to review whether the routing is mislabeling).
`RELEVANT_LOOKS <= 2` → kept but flagged `weak=True` (a too-easy negative; not dropped,
but the QA step tracks the weak rate — if it's high, the generation prompts need tuning).

### 8.3 Why this gate is the heart of the pipeline

The false-negative rate is the single number that determines whether the negatives are
usable. A false negative tells the model "the correct answer is wrong" — at even a few
percent it measurably damages contrastive retrieval. The audit gate measures and removes
it. **The pilot's primary success criterion is that the post-audit false-negative rate
is low enough to ship** (target: <2% of kept negatives, confirmed by the 200-sample
hand inspection, §10).

---

## 9. Output schema

### 9.1 `negatives_v1.parquet` (long format — one row per kept negative)

| column | dtype | note |
|---|---|---|
| `query_id` | str | = v3g `example_id` (the positive's id) |
| `query` | str | |
| `query_style` | str | |
| `task_family` | str | |
| `positive_passage_id` | str | v3g `passage_id` |
| `cui_label` / `alt_term` / `semantic_group` | str | from v3g |
| `negative_id` | str | `f"{query_id}__{idx}"` |
| `negative_passage` | str | the negative text |
| `negative_source` | str | `synthetic` / `bm25_real` / `category_aware_real` / `garbage_refusal` |
| `failure_mode` | str | tier label (e.g. `confusable_disease`, `lexical_trap_real`, …) |
| `why_wrong` | str | generator's one-line rationale (empty for real anchors) |
| `audit_relevant_looks` | int | 1-5 |
| `audit_answers_query` | bool | must be False for kept |
| `audit_grounding` | int | 1-5 |
| `audit_in_failure_mode` | bool | |
| `audit_note` | str | |
| `weak` | bool | True if RELEVANT_LOOKS ≤ 2 |

### 9.2 `negatives_v1_dropped.parquet`
Same columns + `drop_reason` ∈ {`answers_query_true`, `grounding_le2`, `parse_failure`,
`intra_sample_near_dup`}. For `parse_failure`, the audit columns are null.

### 9.3 `negatives_v1_qa_sample.parquet`
~200 kept negatives, **stratified by (task_family × failure_mode)**, for the subagent QA
step (§13). Includes the positive passage so a reviewer can judge query↔positive↔negative
without loading v3g.

### 9.4 `triplets_dev_candidates.parquet` (held-out real eval)
10% of v3g (stratified by family), **excluded from negative generation**. Columns:
`query`, `query_style`, `task_family`, `positive_passage_id`, `positive_passage_text`,
`bm25_recall_at_10/100` (does BM25 over the 66k corpus surface the positive?),
`positive_in_corpus` (bool). For downstream neural retrieval eval after training (§12).

### 9.5 `negatives_v1_run_summary.json`
`n_positives`, `n_negatives_generated`, `n_kept`, `n_dropped` (by reason), per-family and
per-tier tables, `false_negative_rate_pre_audit`, `false_negative_rate_post_audit`,
`weak_rate`, BM25 recall@100 on the dev split, wall-clock, vLLM model id.

---

## 10. Pilot protocol (the 2k run — DO THIS FIRST)

**Purpose:** de-risk the prompts and the audit gate at 1/30 the cost of the full run,
and produce a hand-inspectable sample before committing ~6 days of vLLM time.

1. **Smoke:** `uv run python scripts/data/mine_synthetic_negatives.py --limit 24`
   (≈5 positives/family). Confirms the pipeline runs end-to-end, the vLLM call shape is
   accepted (esp. `chat_template_kwargs`/`enable_thinking`), the `<<<NEG>>>` parser works,
   and outputs write. Inspect a few generated negatives by eye.
2. **Pilot:** `uv run python scripts/data/mine_synthetic_negatives.py --limit 2000`
   — stratified by family proportional to v3g composition
   (symptom ≈ 970, biomedical ≈ 390, generic_clinical ≈ 310, lay ≈ 280, brand ≈ 50).
3. **Hand inspection of `negatives_v1_qa_sample.parquet` (200 rows):** the owner (or a
   subagent, §13) reads query ↔ positive ↔ negative and judges:
   - **False-negative rate** among kept negatives (target **< 2%**).
   - **Hallucinated-gibberish rate** (target **< 1%**).
   - **Weak rate** (RELEVANT_LOOKS ≤ 2; target **< 15%** — some weak negatives are fine).
   - **Failure-mode accuracy** (does each negative actually fail in its labeled mode?
     target **> 85%**; lower means the per-tier prompts need sharpening).
4. **Go/no-go:** if all four targets are met → green-light the full run. If not → retune
   the offending tier's `FAILURE_MODE` spec or the audit prompt, re-smoke, re-pilot.

---

## 11. Full-run protocol (the ~60k run)

`uv run python scripts/data/mine_synthetic_negatives.py --limit 0`  (0 = all 59,670).

- **Concurrency:** `ThreadPoolExecutor(max_workers=64)` for both generation and audit
  (vLLM handles it; the repo's filters ran 32 comfortably; 64 doubles throughput).
- **Resumability:** the runner writes partial results checkpointed every 500 positives
  to `negatives_v1_partial.parquet` and skips already-done `query_id`s on restart (so a
  crash mid-run doesn't lose hours). Resume = re-run the same command.
- **Wall-time estimate:** ≈60k positives × (~5 generation calls + ~13 audit calls) ≈
  1.08M Gemma calls. At ~15 s/call / 64 workers ≈ **~3 days**. The pilot (§10) refines
  this estimate from real throughput.
- **Cost is the owner's vLLM** — no external API. Acceptable per the owner ("multi-day").

---

## 12. Downstream real-eval (the train-on-synthetic gate)

Two layers, only the first is in the runner:

1. **BM25 recall sanity (in the runner, now).** On the held-out 10% dev split, BM25 over
   the 66k corpus: does the positive surface in top-10 / top-100? Validates that the real
   corpus and the real queries are compatible (the positives must be findable for the
   queries to be a meaningful eval). Expected: decent for keyword styles, lower for lay
   styles. This is a *sanity check on the eval setup*, not a model metric.

2. **Neural retrieval eval (after training — NOT in the runner).** Once a ColBERT model
   is trained on the triplets built from `negatives_v1.parquet`, retrieve each dev query
   against the full 66k real corpus and measure **MRR@10, Recall@100, nDCG@10**. This is
   the only gate that catches train-on-synthetic overfit: if the model ranks synthetic-
   texture distractors well but real-corpus retrieval is poor, the synthetic mix is too
   dominant and the real-anchor ratio (§7) must be raised. Requires: pylate install, a
   triplet collator, an index over the 66k corpus. **These do not exist yet** (training
   scripts and eval scripts are empty stubs); building them is the next phase after the
   negatives ship.

---

## 13. QA protocol (the subagent step the owner triggers after a run)

After the pilot (and again after the full run), the owner returns and the assistant runs
a **fan-out subagent QA** modeled on the v3g audit:

- **Stratified random sample N per (task_family × failure_mode)** from
  `negatives_v1.parquet` (N chosen for tight Wilson 95% CIs — ~120 per tier, like the v3g
  n=120). Each sampled negative carries its positive passage.
- **Subagents (one per family, or one per tier — fanned out):** read
  query ↔ positive ↔ negative and score, per the calibrated discipline:
  - `false_negative` (does it answer the query? — the kill signal)
  - `failure_mode_correct` (does it fail in the labeled way?)
  - `grounding` (is it coherent medical text?)
  - `hardness` (is it actually hard, or trivially irrelevant?)
- **Aggregate** into a verdict table with Wilson CIs on the false-negative rate per tier.
  **Ship criterion:** false-negative upper CI < 5% per tier (analogous to the v3g
  hallucination < 10% criterion; tighter here because false negatives are more damaging).
- **Verdict:** PASS (ship) / RETUNE (re-pilot specific tiers) / FAIL (rework).

This is the same loop that caught the Round-2 over-firing and produced the clean v3g —
applied to negatives instead of positives.

---

## 14. Run commands

```bash
# 0. ensure vLLM is up (Gemma-4-31B at http://localhost:8000/v1)
curl -s http://localhost:8000/v1/models | jq '.data[].id'

# 1. smoke (≈5 positives/family, ~couple minutes)
uv run python scripts/data/mine_synthetic_negatives.py --limit 24 --tag smoke

# 2. pilot (2k stratified — inspect negatives_pilot_qa_sample.parquet after)
uv run python scripts/data/mine_synthetic_negatives.py --limit 2000 --tag pilot

# 3. full run (≈60k, ~3 days, resumable)
uv run python scripts/data/mine_synthetic_negatives.py --limit 0 --tag v1

# (resume after a crash = re-run the SAME command + same --tag; it skips done query_ids
#  via negatives_<tag>_partial.parquet)
```

`--workers N` (default 64) tunes concurrency. Outputs are namespaced by `--tag` so a
smoke/pilot/full run never clobber each other. The dev eval split
(`triplets_dev_candidates.parquet`) is written once and is the same 10% holdout
regardless of `--tag`/`--limit`.

No new dependencies required (BM25 is inline; httpx/pandas/numpy/pyarrow already in the
project). The runner checks vLLM health and aborts cleanly if the model is down.

---

## 15. Abort / retune conditions (built into the runner's self-checks)

| Signal | Threshold | Action |
|---|---|---|
| vLLM down / model id missing | — | abort with a clear message |
| Parse failure rate (per tier) | > 15% | the `<<<NEG>>>` format is being ignored → re-smoke after prompt fix |
| Pre-audit false-negative rate (per tier) | > 8% | generation is producing too many correct answers → sharpen the tier's `DOES NOT ANSWER` constraint |
| Post-audit false-negative rate | > 2% (pilot hand-inspect) | audit gate is too lenient → enable thinking for the audit judge, re-pilot |
| Weak rate (RELEVANT_LOOKS ≤ 2) | > 15% | negatives are too easy/irrelevant → strengthen `RETRIEVABLE` constraint |
| Intra-sample near-dup drop rate | > 25% | Gemma is repeating → lower per-call `n` or add a stronger diversity instruction |

The runner prints all of these in the summary; the owner uses them to decide go/no-go.

---

## 16. File layout (what this plan produces)

```
scripts/data/mine_synthetic_negatives.py        # the runner (self-contained)
docs/TRIPLET_MINING_PLAN.md                     # this document
data/processed/private/synthetic/negatives/
    negatives_v1.parquet                        # kept negatives (long)
    negatives_v1_dropped.parquet                # dropped, labeled
    negatives_v1_qa_sample.parquet              # 200 stratified for QA
    negatives_v1_partial.parquet                # checkpoint (resumability)
    triplets_dev_candidates.parquet             # held-out real eval split + BM25 sanity
    negatives_v1_run_summary.json               # run stats
```

---

## 17. What is explicitly NOT in this plan (next phase)

- ColBERT triplet collator / training data loader (training scripts are empty stubs).
- pylate install + the actual ColBERT training.
- Neural retrieval-eval (MRR/Recall/nDCG) — needs the trained model (§12.2).
- The 05/06/07 negative-mining stubs are superseded by this single runner; they can be
  deleted or repurposed as thin wrappers around `mine_synthetic_negatives.py`.

The boundary is deliberate: **this phase produces the negatives and the real-eval split;
the next phase builds the trainer that consumes them.**
