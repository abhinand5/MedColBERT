# MedColBERT Synthetic Dataset — Cleanup Implementation Plan

**Goal:** Clean `mode4_teacher_filtered_multistyle.parquet` (100,084 rows) per the
agreed priorities, **using UMLS as the ground truth for every terminology
decision**, then self-verify with rule-based checks AND a fan-out subagent
random-sample audit (the same method that found the original problems).

**You (the executor) are expected to one-shot this.** Every path, column name,
and UMLS schema below was verified against the repo on 2026-06-21. If a column is
genuinely missing, adapt — but do not invent paths. Run everything with `uv run`.

---

## 0. Background & agreed decisions

A prior per-family audit (5 subagents × ~48 rows each) found:
- `abbreviation_to_expanded` (2,345 rows): **14.6% hallucination** (homonym traps).
- `brand_generic` (21,947 rows): ~60% are lab reagents / non-drug chemicals, not
  brand↔generic pairs; 2 rows have outright mismatched `alt_term` ("Glutaral").
- `lay_to_clinical` / `biomedical_to_clinical`: `keyword_clinical` and
  `keyword_technical` styles do NOT shift register — they list passage terms.
- ~534 placeholder/rejection queries ("N/A …", "Since the provided fact …").
- A few non-biomedical passages (game theory, plant biology) inside clinical families.

**Agreed decisions (from the user):**
1. General retrieval noise (~2.3% weak positives) is acceptable — **do NOT** chase it.
   (Placeholder rows are still dropped in Step 0 because they are literal garbage,
   not "weak positives" — zero-risk, included by default.)
2. **Drop `abbreviation_to_expanded` entirely** (~2,345 rows).
3. **Drugs are second-class.** Clinical core (disorders, signs/symptoms, diseases,
   procedures, anatomy, physiology) is first. Demote + clean `brand_generic`.
4. **Relabel** the misaligned `keyword_clinical`/`keyword_technical` rows in the two
   register-shift families → a new `generic_clinical_retrieval` family. Do not drop them.

**Additional UMLS-grounded step (recommended, included):** a *passage-purity* filter
that drops rows whose concept `semantic_group` is non-medical, plus an
`alt_term`↔`cui_label` CUI-mismatch integrity check (catches the "Glutaral" bug).

---

## 1. Inputs (verified)

### The dataset
`/workspace/MedColBERT/data/processed/private/synthetic/consolidated/mode4_teacher_filtered_multistyle.parquet`
- 100,084 rows. Columns you will use:
  `example_id, passage_id, passage_text, query, query_style, fact, mode, task_family,
   role, cui_label, alt_term, target_cuis, semantic_group, vocab_shift_type,
   term_overlap, ngram_copy_4, alt_term_overlap`.
- `task_family` ∈ {symptom_to_diagnosis, biomedical_to_clinical, brand_generic,
  lay_to_clinical, abbreviation_to_expanded}.
- `query_style` ∈ {full_question, keyword_clinical, keyword_layperson, keyword_technical}.
- **`semantic_group` is already present per row** (from the matched UMLS concept) — use
  it directly for the purity filter; no re-resolution needed for that step.
- **CRITICAL — `cui_label` and `target_cuis` are MIXED format:**
  - Sometimes a CUI: `"C0031412"` / `'["C0031412"]'`
  - Sometimes a concept name: `"Cyclic guanosine monophosphate"` / `'["Cyclic guanosine monophosphate"]'`
  - `target_cuis` is a **JSON string** (dtype str), e.g. `'["C0031412"]'`. Parse with `json.loads`.
  - Your CUI resolver MUST handle both forms (see script §4).
- `brand_generic` rows carry `vocab_shift_type="consumer_to_clinical"` (NOT
  `brand_to_generic`). The family name is a misnomer; treat it as "drug vocab-shift."

### UMLS parquets (the ground truth — "use UMLS as much as possible")
`/workspace/MedColBERT/data/processed/private/ontology/`
- **`semantic_types.parquet`** — cols: `cui, tui, semantic_group`. 3.67M rows. Use for
  CUI → set of TUIs and CUI → semantic_group.
- **`concept_strings.parquet`** — cols: `cui, sab, tty, code, scui, string,
  normalized_string, is_abbreviation, source_group, string_hash, semantic_types,
  semantic_group`. 9.4M rows. Use for concept-name → CUI resolution and CUI → set of
  source vocabularies (`sab`), e.g. `RXNORM`/`VANDF` = real clinical drugs.
- **`concept_pairs.parquet`** — cols: `pair_id, cui, left_string_id, right_string_id,
  left_source_group, right_source_group, vocab_shift_type, semantic_group,
  quality_flags`. 3.9M rows. (Reference only; not required by the cleaning script.)

### Allowed `semantic_group` values (16, from UMLS Semantic Network)
`disorders, procedures, drugs_chemicals, anatomy, findings_signs_symptoms,
genes_proteins, living_beings, physiology, objects, concepts_ideas, organizations,
activities_behaviors, devices, geographic_areas, groups, other`.

### Existing audit scripts (use for rule-based self-check)
- `scripts/data/ngram_copy_audit.py` — **authoritative** vocab-shift. Run with `--per-family`.
- `scripts/data/audit_quality.py` — the *misleading* 88–94/100 proxy. Run for
  completeness only; do NOT treat its vocab-shift number as ground truth.

---

## 2. Cleaning specification (exact boolean rules)

Order matters — assign the FIRST matching `drop_reason` and stop.

| Step | Rule | `drop_reason` |
|---|---|---|
| 0 | `query` matches `^(N/?A\b\|Since the provided fact\|No clinical\|Non-clinical)` (case-insensitive) | `placeholder_query` |
| 1 | `task_family == "abbreviation_to_expanded"` | `dropped_abbreviation_family` |
| 2 | `semantic_group ∈ {concepts_ideas, geographic_areas, organizations, objects, other}` | `non_medical_semantic_group` |
| 3 | `alt_term` and `cui_label` both resolve to CUIs **and the CUIs differ** | `alt_term_cui_mismatch` |
| 4 | `task_family == "brand_generic"` AND resolved CUI is **not** a clinical drug (see §3) | `not_a_clinical_drug` |

Rows with no `drop_reason` are **kept**. Then two transformations on the kept set:

- **Relabel (priority 4):** if `task_family ∈ {lay_to_clinical, biomedical_to_clinical}`
  AND `query_style ∈ {keyword_clinical, keyword_technical}` → set
  `task_family = "generic_clinical_retrieval"`. Leave `query_style` unchanged.
- **Downsample drugs (priority 3):** if `task_family == "brand_generic"` and the kept
  count > `BRAND_GENERIC_CAP` (default **6000**), keep a reproducible random sample of
  `BRAND_GENERIC_CAP` (seed `20260621`); keep all non-`brand_generic` rows.

### 3. "Clinical drug" definition (UMLS-grounded, for Step 4)
Resolve `cui_label` → CUI (§4). Look up that CUI in `semantic_types.parquet` (TUIs) and
`concept_strings.parquet` (SABs). **Keep** the row if EITHER is true:
- the CUI has a TUI in `{T121 (Pharmacologic Substance), T196 (Clinical Drug), T203 (Vitamin)}`, OR
- the CUI has a string with `sab ∈ {RXNORM, VANDF, MEDRT, NDDF, DRUGBANK}`.

Otherwise drop as `not_a_clinical_drug` (these are lab reagents: Carbohydrates,
Porphyrins, Microcapsules, etc.). After Step 4, **print the TUI distribution of the
dropped brand_generic rows** to confirm they are dominated by `T109`/`T110`
(organic/inorganic chemicals) — this is your self-check that the rule is firing correctly.

---

## 4. Reference implementation (write to `scripts/data/clean_consolidated.py`)

This is a complete, runnable script. It writes a **new** cleaned file (never overwrites
the original) plus a `dropped_rows` file with reasons. Run it, then verify per §5–§6.

```python
"""Clean the consolidated MedColBERT synthetic dataset per docs/CLEANUP_PLAN.md.
UMLS-grounded. Writes cleaned + dropped parquets; never overwrites the source."""
import json, re
import pandas as pd
from pathlib import Path

ROOT = Path("/workspace/MedColBERT")
SRC = ROOT / "data/processed/private/synthetic/consolidated/mode4_teacher_filtered_multistyle.parquet"
ONT = ROOT / "data/processed/private/ontology"
OUT_CLEAN = ROOT / "data/processed/private/synthetic/consolidated/mode4_teacher_filtered_multistyle_cleaned.parquet"
OUT_DROPPED = ROOT / "data/processed/private/synthetic/consolidated/cleaned_dropped_rows.parquet"

CUI_RE = re.compile(r"^C\d{7}$")
PLACEHOLDER_RE = re.compile(r"^(N/?A\b|Since the provided fact|No clinical|Non-clinical)", re.I)
NON_MEDICAL_GROUPS = {"concepts_ideas", "geographic_areas", "organizations", "objects", "other"}
DRUG_TUIS = {"T121", "T196", "T203"}            # Pharmacologic Substance, Clinical Drug, Vitamin
DRUG_SABS = {"RXNORM", "VANDF", "MEDRT", "NDDF", "DRUGBANK"}
BRAND_GENERIC_CAP = 6000
RANDOM_STATE = 20260621

# ---- load source + UMLS ----
df = pd.read_parquet(SRC)
print(f"loaded {len(df):,} rows")
cs = pd.read_parquet(ONT / "concept_strings.parquet",
                     columns=["cui", "string", "normalized_string", "sab", "semantic_group"])
st = pd.read_parquet(ONT / "semantic_types.parquet")  # cui, tui, semantic_group

# concept-name (normalized) -> cui (first match)
csu = cs.dropna(subset=["normalized_string"]).drop_duplicates("normalized_string")
norm_to_cui = dict(zip(csu["normalized_string"], csu["cui"]))
# cui -> set(sab)
sabs_by_cui = cs.dropna(subset=["sab"]).groupby("cui")["sab"].apply(lambda s: set(s))
# cui -> set(tui)
tuis_by_cui = st.dropna(subset=["tui"]).groupby("cui")["tui"].apply(lambda s: set(s))

def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower()) if pd.notna(s) and str(s).strip() else ""

def cui_from_term(term):
    """Resolve a CUI from either a CUI string or a concept name. None if unresolvable."""
    if pd.isna(term): return None
    t = str(term).strip()
    if CUI_RE.match(t): return t
    n = norm(t)
    return norm_to_cui.get(n)

def resolve_cui(row):
    """Prefer target_cuis[0]; fall back to cui_label. Handles mixed formats."""
    tc = row.get("target_cuis")
    if isinstance(tc, str) and tc:
        try:
            lst = json.loads(tc)
            if lst:
                c = cui_from_term(lst[0])
                if c: return c
        except Exception:
            pass
    return cui_from_term(row.get("cui_label"))

print("resolving CUIs (row-wise, ~1 min)...")
df["resolved_cui"] = df.apply(resolve_cui, axis=1)
print(f"  resolved: {df['resolved_cui'].notna().sum():,} / {len(df):,}")

def is_clinical_drug(cui):
    if pd.isna(cui): return False
    tuis = tuis_by_cui.get(cui, set()) or set()
    sabs = sabs_by_cui.get(cui, set()) or set()
    return bool(tuis & DRUG_TUIS) or bool(sabs & DRUG_SABS)

# ---- assign drop_reason (first match wins) ----
reason = pd.Series([None] * len(df), index=df.index, dtype=object)
reason[df["query"].fillna("").str.match(PLACEHOLDER_RE)] = "placeholder_query"
m = reason.isna() & (df["task_family"] == "abbreviation_to_expanded"); reason[m] = "dropped_abbreviation_family"
m = reason.isna() & (df["semantic_group"].isin(NON_MEDICAL_GROUPS)); reason[m] = "non_medical_semantic_group"

# alt_term <-> cui_label CUI mismatch (only when both resolve and differ)
alt_cui = df["alt_term"].apply(cui_from_term)
m = reason.isna() & df["resolved_cui"].notna() & alt_cui.notna() & (df["resolved_cui"] != alt_cui)
reason[m] = "alt_term_cui_mismatch"

# brand_generic not-a-clinical-drug
m = reason.isna() & (df["task_family"] == "brand_generic") & (~df["resolved_cui"].apply(is_clinical_drug))
reason[m] = "not_a_clinical_drug"

df["drop_reason"] = reason
kept = df[reason.isna()].copy()
dropped = df[reason.notna()].copy()
print("\n=== drop reasons ===")
print(dropped["drop_reason"].value_counts().to_string())

# ---- relabel misaligned register-shift keyword styles (priority 4) ----
relabel_mask = (
    kept["task_family"].isin(["lay_to_clinical", "biomedical_to_clinical"]) &
    kept["query_style"].isin(["keyword_clinical", "keyword_technical"])
)
kept.loc[relabel_mask, "task_family"] = "generic_clinical_retrieval"
print(f"\nrelabelled {int(relabel_mask.sum()):,} rows -> generic_clinical_retrieval")

# ---- downsample brand_generic (priority 3) ----
bg = kept[kept["task_family"] == "brand_generic"]
if len(bg) > BRAND_GENERIC_CAP:
    keep_bg = bg.sample(n=BRAND_GENERIC_CAP, random_state=RANDOM_STATE)
    kept = pd.concat([kept[kept["task_family"] != "brand_generic"], keep_bg], ignore_index=False)
    print(f"downsampled brand_generic {len(bg):,} -> {BRAND_GENERIC_CAP:,}")

# ---- write outputs ----
kept.to_parquet(OUT_CLEAN, index=False)
dropped.to_parquet(OUT_DROPPED, index=False)
print(f"\nwrote {len(kept):,} cleaned -> {OUT_CLEAN}")
print(f"wrote {len(dropped):,} dropped  -> {OUT_DROPPED}")

# ---- composition + integrity report ----
print("\n=== FINAL task_family x query_style (cleaned) ===")
print(pd.crosstab(kept["task_family"], kept["query_style"]).to_string())
print("\n=== dropped TUI check (brand_generic not_a_clinical_drug) ===")
bg_drop = dropped[dropped["drop_reason"] == "not_a_clinical_drug"]
bg_drop_tuis = bg_drop["resolved_cui"].map(lambda c: (tuis_by_cui.get(c, set()) or set()))
exploded = pd.Series([t for s in bg_drop_tuis for t in s])
print("dropped brand_generic TUI counts (expect T109/T110 to dominate):")
print(exploded.value_counts().head(10).to_string())
assert "abbreviation_to_expanded" not in kept["task_family"].values, "abbreviation family must be gone"
assert kept["query"].fillna("").str.match(PLACEHOLDER_RE).sum() == 0, "placeholders remain"
print("\nALL ASSERTIONS PASSED")
```

Run: `uv run python scripts/data/clean_consolidated.py`

---

## 5. Rule-based self-verification (run after the script)

1. `uv run python scripts/data/ngram_copy_audit.py --per-family`
   → every surviving family must still be **≥ 60/100** (cleaning only removes rows; it
   cannot lower this, but confirm nothing regressed).
2. Re-run the integrity asserts from the script mentally against the printed composition:
   - `abbreviation_to_expanded` absent.
   - Zero placeholder queries.
   - `brand_generic` count ≤ 6000 and all survivors are clinical drugs (the dropped-TUI
     printout confirms the filter fired on `T109`/`T110`).
   - `generic_clinical_retrieval` family present, containing only `keyword_clinical` +
     `keyword_technical` styles sourced from the two register-shift families.
3. Print a before/after composition table (original vs cleaned) — include it in your
   final report.

---

## 6. Fan-out subagent random-sample audit (the real self-check)

Rule-based checks CANNOT see hallucination, register mismatch, or monotony — that's why
the original problems survived. You MUST replicate the human-read audit. Spawn **one
general-purpose subagent per surviving task_family** (5 families after cleaning:
`symptom_to_diagnosis, biomedical_to_clinical, lay_to_clinical, brand_generic,
generic_clinical_retrieval`), in parallel, `run_in_background=true`. Give each the card
below with `{FAMILY}` and `{PATH}` substituted.

`{PATH}` = `data/processed/private/synthetic/consolidated/mode4_teacher_filtered_multistyle_cleaned.parquet`

### Per-subagent task card
> You are auditing the `{FAMILY}` slice of the CLEANED MedColBERT synthetic dataset.
> Work in `/workspace/MedColBERT`. Use `uv run`.
>
> 1. Load `{PATH}` with pandas, filter to `task_family == "{FAMILY}"`. Report the row
>    count and `query_style` distribution.
> 2. STRATIFIED random sample, 12 rows per `query_style` present (reproducible seed):
>    `import hashlib; seed = int(hashlib.md5(b"{FAMILY}").hexdigest()[:8],16) % 2**31`
>    `sample = df.groupby("query_style", group_keys=False).sample(n=12, random_state=seed)`
>    (If a style has <12 rows, take all of them.) Print the sampled `example_id`s.
> 3. For EACH sampled row, READ `query` + `passage_text` together (+ `fact`/`alt_term`/
>    `cui_label` where relevant). Score 1–5 with a one-line justification on:
>    (a) **VOCAB SHIFT** — does the query use genuinely different surface vocabulary
>        than the passage (real substitution), not copied/reformulated passage wording?
>        Be strict. For keyword styles, judge whether the *terms* differ from the
>        passage's terms, not whether the query is a full sentence.
>    (b) **FACT GROUNDING** — is every query claim supported by the passage? Flag any
>        unsupported term/claim as HALLUCINATION.
>    (c) **RELEVANCE** — is the passage a genuine positive match for the query intent?
>    (d) **NATURALNESS** — fluent and non-templated? Flag robotic phrasing / monoculture.
>    (e) **FAMILY-SPECIFIC** — for `{FAMILY}`:
>        - `symptom_to_diagnosis`: symptom/evidence phrasing vs diagnosis target?
>        - `biomedical_to_clinical`: literature phrasing vs clinical phrasing shift?
>        - `lay_to_clinical`: layperson wording vs clinical passage (real register shift)?
>        - `brand_generic`: does the query swap a brand name for a generic/ingredient (or
>          vice versa), not just paraphrase? Are alt_terms real drugs?
>        - `generic_clinical_retrieval`: is it a legitimate clinical/medical keyword
>          retrieval query that matches its passage (this family is NOT expected to shift
>          register — judge it as generic medical search; the only failure here is
>          hallucination, non-medical content, or irrelevance)?
> 4. GUARDRAILS: low `term_overlap` (<0.5) is EXPECTED for keyword queries — NOT a
>    hallucination or bad-vocab-shift signal. Keyword queries are not full sentences;
>    don't penalize terseness. For `full_question`, flag monotony if >30% start with the
>    same word/phrase.
> 5. Return EXACTLY this JSON:
>    ```
>    {"family":"{FAMILY}","rows_in_family":<int>,"sampled_ids":[...],
>     "mean_scores":{"vocab_shift":<>,"fact_grounding":<>,"relevance":<>,"naturalness":<>,"family_specific":<>},
>     "hallucination_count":<int out of N>,"monotony_flag":<bool>,
>     "worst_rows":[{"example_id":...,"dimension":...,"problem":...}, ...up to 5],
>     "verdict":"PASS"|"FAIL"|"BORDERLINE",
>     "summary":"<2-3 sentences>"}
>    ```
>    Thresholds: PASS if vocab_shift≥3.5 AND halluc_rate<5% AND no monotony AND
>    family_specific≥3.5; FAIL if vocab_shift<3 OR halluc_rate≥10%; else BORDERLINE.

### Aggregate (after all 5 return)
Produce a table: `family | rows | vocab_shift | fact_grounding | relevance |
naturalness | family_specific | halluc_rate | monotony | verdict`. Put the Step-1
ngram_copy scores next to the sampled vocab_shift. **Overall verdict = PASS only if all
5 families PASS AND every ngram_copy family score ≥ 60/100.** List every `worst_rows`
entry. Report real numbers — if a family fails, say so plainly.

---

## 7. Definition of done

All of these must be true before you declare complete:
- [ ] `scripts/data/clean_consolidated.py` written and runs clean (`ALL ASSERTIONS PASSED`).
- [ ] `mode4_teacher_filtered_multistyle_cleaned.parquet` + `cleaned_dropped_rows.parquet` written.
- [ ] Original source parquet **untouched** (diff confirmed — you only added new files).
- [ ] `ngram_copy_audit.py --per-family` on cleaned data: every family ≥ 60/100.
- [ ] Composition table printed; `abbreviation_to_expanded` gone; `brand_generic` ≤ 6000;
      `generic_clinical_retrieval` present; 0 placeholders.
- [ ] 5 fan-out subagents returned; aggregate table written; **no FAIL verdicts** and
      overall hallucination rate < 5%.
- [ ] Final report saved to `docs/audit_cleanup_<date>.md` containing: before/after
      composition, drop-reason counts, ngram_copy table, fan-out aggregate table,
      worst_rows, and overall verdict.

**If any family FAILs the fan-out audit, do NOT mark complete** — report it with the
specific `worst_rows` so the failing rows can be inspected. Do not silently re-clean to
force a pass; surface the problem.

---

## 8. What NOT to do
- Do not overwrite the original parquet.
- Do not push to HuggingFace (that is a separate, later step; the user has not approved it).
- Do not trust `audit_quality.py`'s 88–94/100 vocab-shift number — it is the misleading
  proxy (alt_term presence + term_overlap) that let the old temp=0.0 dataset pass a fake bar.
- Do not treat low `term_overlap` as a defect on keyword rows.
- Do not "round up" a BORDERLINE to PASS or hide a FAIL.

---

## 9. Round 2 — fix the audit, then re-clean (REQUIRED)

### Why this round exists
Round 1 reached a FAIL verdict on `generic_clinical_retrieval` at **12.5% hallucination
= 3/24 rows**. That point estimate is statistically meaningless: on n=24 one row is 4.2
percentage points, so the true rate plausibly lies anywhere in ~4%–25%. The Round-1
verdict treated a noisy estimate as a precise measurement and gated the whole deliverable
on it. Round 2 fixes that. Two principles:

1. **Stop estimating rates from n=24.** Re-audit at n=80 per family and report a **95%
   Wilson confidence interval** for the hallucination rate, not a bare percentage. The
   verdict applies to the interval, never the point estimate.
2. **Stop relying on the orchestrator model's reading strictness.** Use the local
   **Gemma 4 31B** (vLLM, already running) as a *deterministic, reproducible* judge for
   the two dimensions where model judgment matters most (fact_grounding,
   family_specific). vocab_shift is already measured authoritatively by `ngram_copy_4`;
   monotony is a regex. The LLM is only needed where it adds real signal.

### 9.1 The local LLM you will use (verified working)
- Endpoint: `http://localhost:8000/v1/chat/completions` (OpenAI-compatible, vLLM).
- Model id: `gemma-4-31B` (confirmed via `GET /v1/models`).
- Call pattern — reuse this exact helper (mirrors
  `src/medcolbert/generation/direct_generator.py:434`). Use **temperature=0.0** for all
  judging so results are reproducible:

```python
import httpx, json
VLLM = "http://localhost:8000/v1"; MODEL = "gemma-4-31B"
def gemma(system: str, user: str, max_tokens: int = 300, timeout: float = 90.0) -> str:
    with httpx.Client(timeout=timeout) as c:
        r = c.post(f"{VLLM}/chat/completions",
                   json={"model": MODEL, "temperature": 0.0,
                         "max_tokens": max_tokens,
                         "messages": [{"role": "system", "content": system},
                                      {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
```
Before using it in bulk, verify the endpoint: `curl -s http://localhost:8000/v1/models`.
If it is down, STOP and report — do not fall back to the orchestrator model for judging
(that reintroduces the strictness variance Round 2 is meant to eliminate).

### 9.2 Re-cleaning steps (run in this order on the Round-1 cleaned file)

**Input** = `mode4_teacher_filtered_multistyle_cleaned.parquet` (the Round-1 output,
78,575 rows). Round 2 reads it, never overwrites it, and writes
`mode4_teacher_filtered_multistyle_cleaned_v2.parquet` + a v2 dropped-rows file.

**(a) Broader corrupted-query sweep (ALL families).** Round 1's placeholder regex
(`^N/A|^Since the provided fact`) missed rows like `symptom_to_diagnosis` example
`457ae6f8…` ("note to self"/meta-commentary as query). Add a second regex that flags
queries containing meta/self-referential phrasing, applied to **every** family:
```
META_RE = re.compile(r"\b(note to self|the provided fact|the above|this passage|"
                     r"the passage (is|does|describes)|since the (fact|passage)|"
                     r"as (an? )?(AI|language model)|I (am|cannot|don't))\b", re.I)
```
Flag matching rows `drop_reason = "corrupted_meta_query"`. Spot-check 10 flagged rows
and print them before committing the drop — confirm they are genuinely corrupt, not
legitimate clinical phrasing that happens to contain "the passage describes".

**(b) Cap `generic_clinical_retrieval` to 10,000.** It is currently 21,972 rows (~28% of
the dataset) and is a relabel dumping-ground, not a designed family — structurally wrong
for a model whose first priority is the clinical core. Keep a reproducible random sample
of 10,000 (`random_state=20260621`), `drop_reason="downsample_generic_clinical"` for the
rest. This is independent of the hallucination question; do it regardless.

**(c) UMLS term-grounding pre-filter on the capped `generic_clinical_retrieval`.** For
each surviving row, tokenize the query into content tokens (lowercase, strip
punctuation, drop a standard English stopword list and tokens <3 chars). A content token
is **grounded** if it is a case-insensitive substring of `passage_text` OR it equals (or
is a variant of) a synonym of the row's concept. Build the synonym set per row from
`concept_strings.parquet`: resolve `cui_label`/`target_cuis` → CUI (reuse the Round-1
resolver — remember the mixed CUI-or-name format), then collect all `string` values for
that CUI and their `normalized_string`. A row is a **hallucination candidate** if it has
≥1 ungrounded content token. Mark `drop_reason="umls_ungrounded_term"`.

  - This is a **pre-filter**, not the final word: valid lay synonyms absent from UMLS
    will false-positive. Do NOT drop on the UMLS flag alone. Pass every candidate to
    step (d).

**(d) Gemma grounding verifier on candidates from (c).** For each candidate row, call
Gemma with the passage + query and ask for a strict verdict. Prompt:
```
SYSTEM: You are a strict medical-data auditor. You check whether every medical concept
and claim in a search QUERY is supported by the PASSAGE. Be conservative: if a concept
in the query is NOT stated in or directly entailed by the passage, mark it ungrounded.
USER:
PASSAGE: {passage_text}
QUERY: {query}
ROW CONCEPT (the intended subject): {cui_label or alt_term}
List every query term/concept NOT supported by the passage. Then output a final line
exactly: VERDICT: GROUNDED   or   VERDICT: HALLUCINATED
```
Parse the final `VERDICT:` line (case-insensitive). Drop only rows where
`VERDICT: HALLUCINATED`. Record the ungrounded terms Gemma listed into the dropped-rows
file. **Drop reason `gemma_hallucinated`.** This catches the three real Round-1 failure
modes (statin/AY9944, vitamin A/E, "stump care") that UMLS alone may miss because they
are multi-word fabricated clinical terms, not single ungrounded tokens.

  - Batch this with `max_tokens=300`, sequential or modestly concurrent (vLLM handles
    it; ~10k rows × ~1s ≈ a few hours worst case — fine for a local GPU). Print a running
    counter every 500 rows.

**(e) Optional: further cut `brand_generic`.** Round 1 left `brand_generic` at 6,000 with
`family_specific=1.85` — the report itself notes the pipeline doesn't produce true
brand↔generic swaps. Per the user's "drugs are second-class" priority, cut to **3,000**
(`random_state=20260621`, `drop_reason="downsample_brand_generic_2"`). If the user has
said to keep 6,000, skip this and note it.

### 9.3 Re-audit at n=80 with a Gemma judge (REQUIRED — replaces Round-1's n=24)

Write `scripts/data/reaudit_gemma.py`. For each surviving family:
1. Stratified sample of **80 rows** (20 per `query_style` present; if a style has <20,
   take all and top up from other styles to reach 80 total — record the actual mix).
   Seed per family: `int(hashlib.md5(family.encode()).hexdigest()[:8],16) % 2**31`.
2. For each sampled row, call Gemma (temperature 0.0) to score **two** dimensions on a
   1–5 scale with a one-line reason: `fact_grounding` and `family_specific` (using the
   family-specific criteria from §6's task card). Also have it return a boolean
   `hallucinated` (any unsupported concept) and list `ungrounded_terms`.
3. `vocab_shift` is NOT re-scored by the LLM — use the row's existing `ngram_copy_4`
   column (1 − that value, scaled to 1–5) for consistency, and report the family's
   `ngram_copy_audit.py --per-family` score alongside.
4. `monotony` is computed by regex on `full_question` openers (no LLM): flag if the most
   common first word/phrase covers >30% of `full_question` rows in the family.
5. For each family compute:
   - `halluc_count` / `n` and the **95% Wilson interval** for the proportion
     (`scipy.stats` or a 10-line manual Wilson function — do not skip this).
   - mean `fact_grounding`, `family_specific` (1–5).
6. **Verdict from the interval, not the point estimate:**
   - PASS if upper bound of the hallucination 95% CI < 10% AND mean fact_grounding ≥ 4
     AND mean family_specific ≥ 3.5 AND ngram_copy ≥ 60 AND no monotony.
   - FAIL if lower bound of the 95% CI ≥ 10% (i.e. even the optimistic end is at/above
     the threshold).
   - BORDERLINE if the CI straddles 10% (interval contains 10%) — this is an explicit
     "uncertain" verdict, NOT a pass. Surface it; do not round up.

### 9.4 Round-2 Definition of Done (supersedes §7 for the v2 deliverable)
- [ ] `mode4_teacher_filtered_multistyle_cleaned_v2.parquet` + v2 dropped-rows written;
      original and Round-1 cleaned files untouched.
- [ ] Broader meta-query sweep applied across all families; 10 flagged rows spot-checked
      and printed.
- [ ] `generic_clinical_retrieval` ≤ 10,000; UMLS pre-filter + Gemma verifier run on it;
      dropped rows carry `gemma_hallucinated` with the ungrounded terms recorded.
- [ ] `ngram_copy_audit.py --per-family` on v2: every family ≥ 60/100.
- [ ] `reaudit_gemma.py` run at n=80/family; every family reports a **Wilson 95% CI**.
- [ ] No family FAILs by the CI rule. Any BORDERLINE (CI straddles 10%) is reported
      honestly with its worst_rows — not rounded to PASS.
- [ ] Final report appended to `docs/audit_cleanup_2026-06-21.md` (Round 2 section):
      before/after v2 composition, all drop-reason counts, ngram_copy table, the n=80
      Gemma re-audit table with CIs, worst_rows, overall verdict.

### 9.5 What NOT to do in Round 2
- Do NOT report a hallucination "rate" as a bare percentage from a small sample — always
  the Wilson CI with n stated.
- Do NOT use the orchestrator model (you, DeepSeek) as the judge for fact_grounding /
  family_specific. Use Gemma via the helper. The whole point is reproducible, model-
  independent judging.
- Do NOT drop rows on the UMLS ungrounded-term flag alone — always confirm with the
  Gemma verifier (step d). UMLS is the pre-filter; Gemma is the decision.
- Do NOT skip the endpoint check. If vLLM is down, stop and report rather than
  substituting your own judgment.
- Do NOT cap/finalize `generic_clinical_retrieval` based on the Round-1 12.5% number.
  Apply the cap (structural) and the grounding filter (real defects), then let the n=80
  CI audit decide the verdict.
