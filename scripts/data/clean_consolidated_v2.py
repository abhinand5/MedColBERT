"""Round 2 cleaning per docs/CLEANUP_PLAN.md §9.
Reads the Round-1 cleaned file, applies:
  (a) broader corrupted-query sweep with refined META_RE
  (b) cap generic_clinical_retrieval to 10,000
  (c) UMLS term-grounding pre-filter
  (d) Gemma grounding verifier on candidates
  (e) optional brand_generic cut to 3,000
Writes v2 cleaned + v2 dropped; never overwrites v1 or original."""
import json, re, time
import pandas as pd
import numpy as np
import httpx
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

ROOT = Path("/workspace/MedColBERT")
SRC = ROOT / "data/processed/private/synthetic/consolidated/mode4_teacher_filtered_multistyle_cleaned.parquet"
ONT = ROOT / "data/processed/private/ontology"
OUT_V2 = ROOT / "data/processed/private/synthetic/consolidated/mode4_teacher_filtered_multistyle_cleaned_v2.parquet"
OUT_DROPPED_V2 = ROOT / "data/processed/private/synthetic/consolidated/cleaned_v2_dropped_rows.parquet"

VLLM = "http://localhost:8000/v1"
MODEL = "gemma-4-31B"
GENERIC_CLINICAL_CAP = 10_000
BRAND_GENERIC_V2_CAP = 3_000
RANDOM_STATE = 20260621
VLLM_CONCURRENCY = 16
VLLM_TIMEOUT = 120.0

# ---- regexes ----
META_RE = re.compile(
    r"\b(note to self|the provided fact|the above|this passage|"
    r"the passage (is|does|describes)|since the (fact|passage)|"
    r"as (an? )?(AI|language model)|"
    r"I (cannot|do not) (generate|comply|answer|provide|create))\b", re.I)
SINCE_REFUSAL_RE = re.compile(
    r"(Since|Because) (no|there is no|there was no) (medical|clinical|passage|fact)", re.I)

# ---- stopwords (standard English + medical stopwords that are too generic) ----
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "i", "me", "my",
    "we", "our", "you", "your", "he", "she", "it", "they", "them",
    "this", "that", "these", "those", "in", "on", "at", "to", "for",
    "of", "with", "from", "by", "as", "but", "or", "and", "if", "so",
    "than", "then", "also", "not", "no", "yes", "just", "only", "very",
    "about", "into", "over", "after", "before", "between", "through",
    "during", "above", "below", "up", "down", "out", "off",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "any", "much", "many", "own", "same",
    "doesn", "isn", "aren", "wasn", "weren", "haven", "hasn", "hadn",
    "won", "wouldn", "shan", "shouldn", "can", "couldn", "mustn",
    "don", "didn", "needn", "mightn",
}

def content_tokens(text: str) -> list[str]:
    """Tokenize to lowercase content words, stripping punctuation, dropping stopwords and <3 char tokens."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) >= 3]

def gemma(system: str, user: str, max_tokens: int = 300, timeout: float = VLLM_TIMEOUT) -> str | None:
    """Call Gemma 4 31B via vLLM. Returns content string or None on error."""
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.post(f"{VLLM}/chat/completions",
                       headers={"Content-Type": "application/json"},
                       json={"model": MODEL, "temperature": 0.0,
                             "max_tokens": max_tokens,
                             "messages": [{"role": "system", "content": system},
                                          {"role": "user", "content": user}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [Gemma error] {e}")
        return None

# ---- load Round-1 cleaned data ----
print("Loading Round-1 cleaned file...")
df = pd.read_parquet(SRC)
print(f"  {len(df):,} rows")

# ---- (a) Broader corrupted-query sweep ----
print("\n=== (a) Corrupted meta-query sweep ===")
meta_mask = df["query"].fillna("").str.contains(META_RE)
refusal_mask = df["query"].fillna("").str.contains(SINCE_REFUSAL_RE)
corrupt_mask = meta_mask | refusal_mask
print(f"  META_RE matches: {meta_mask.sum()}")
print(f"  SINCE_REFUSAL_RE matches: {refusal_mask.sum()}")
print(f"  Combined (dedup): {corrupt_mask.sum()}")

# Spot-check 10
sample_n = min(10, corrupt_mask.sum())
if sample_n > 0:
    print(f"\n  Spot-checking {sample_n} flagged rows:")
    flagged = df[corrupt_mask].sample(sample_n, random_state=42)
    for i, (_, r) in enumerate(flagged.iterrows()):
        print(f"  {i+1}. [{r['task_family']}] {r['query'][:200]}")

# Assign drop reason
drop_reason = pd.Series([None] * len(df), index=df.index, dtype=object)
drop_reason[corrupt_mask] = "corrupted_meta_query"
print(f"\n  Dropping {corrupt_mask.sum()} rows as corrupted_meta_query")

# ---- (b) Cap generic_clinical_retrieval to 10,000 ----
print("\n=== (b) Cap generic_clinical_retrieval ===")
gcr_idx = df[(drop_reason.isna()) & (df["task_family"] == "generic_clinical_retrieval")].index
print(f"  generic_clinical_retrieval before cap: {len(gcr_idx):,}")
if len(gcr_idx) > GENERIC_CLINICAL_CAP:
    rng = np.random.default_rng(RANDOM_STATE)
    keep_gcr = rng.choice(gcr_idx, size=GENERIC_CLINICAL_CAP, replace=False)
    drop_gcr = gcr_idx.difference(keep_gcr)
    drop_reason.loc[drop_gcr] = "downsample_generic_clinical"
    print(f"  Keeping {GENERIC_CLINICAL_CAP:,}, dropping {len(drop_gcr):,}")

# ---- (c) UMLS term-grounding pre-filter on capped generic_clinical_retrieval ----
print("\n=== (c) UMLS term-grounding pre-filter ===")
print("  Loading UMLS concept_strings...")
cs = pd.read_parquet(ONT / "concept_strings.parquet",
                     columns=["cui", "string", "normalized_string"])
# Build cui -> set of synonyms (case-folded)
print("  Building synonym index...")
cs_clean = cs.dropna(subset=["normalized_string"])
cui_synonyms = defaultdict(set)
for _, row in cs_clean.iterrows():
    cui = row["cui"]
    ns = str(row["normalized_string"]).strip().lower()
    if ns:
        cui_synonyms[cui].add(ns)
    s = str(row["string"]).strip().lower()
    if s:
        cui_synonyms[cui].add(s)
print(f"  {len(cui_synonyms):,} CUIs indexed with synonyms")

# Get the surviving generic_clinical_retrieval rows (after caps but before UMLS drop)
gcr_surviving_mask = (drop_reason.isna()) & (df["task_family"] == "generic_clinical_retrieval")
gcr_rows = df[gcr_surviving_mask]
print(f"  Running UMLS grounding check on {len(gcr_rows):,} generic_clinical_retrieval rows...")

ungrounded_candidates = []
for idx in gcr_rows.index:
    row = df.loc[idx]
    query = str(row["query"]) if pd.notna(row["query"]) else ""
    passage = str(row["passage_text"]) if pd.notna(row["passage_text"]) else ""
    cui = row.get("resolved_cui")

    tokens = content_tokens(query)
    if not tokens:
        continue

    passage_lower = passage.lower()
    synonyms = cui_synonyms.get(cui, set()) if pd.notna(cui) else set()

    ungrounded = []
    for token in tokens:
        # Grounded if substring of passage OR in synonym set
        if token in passage_lower:
            continue
        if token in synonyms:
            continue
        # Check synonym set for token being a substring of any synonym
        found_in_syn = any(token in syn for syn in synonyms)
        if found_in_syn:
            continue
        ungrounded.append(token)

    if ungrounded:
        ungrounded_candidates.append((idx, ungrounded))

print(f"  UMLS pre-filter: {len(ungrounded_candidates)} candidates flagged (≥1 ungrounded token)")
if ungrounded_candidates:
    avg_ungrounded = np.mean([len(u) for _, u in ungrounded_candidates])
    print(f"  Avg ungrounded tokens per candidate: {avg_ungrounded:.1f}")
    # Print a few examples
    print("  Example candidates:")
    for i, (idx, utokens) in enumerate(ungrounded_candidates[:5]):
        r = df.loc[idx]
        print(f"    {i+1}. query='{r['query'][:120]}' | ungrounded: {utokens[:8]}")

# ---- (d) Gemma grounding verifier on candidates ----
print(f"\n=== (d) Gemma grounding verifier ({len(ungrounded_candidates)} candidates) ===")

SYSTEM_PROMPT = """You are a strict medical-data auditor. You check whether every medical concept and claim in a search QUERY is supported by the PASSAGE. Be conservative: if a concept in the query is NOT stated in or directly entailed by the passage, mark it ungrounded."""

def build_user_prompt(passage_text: str, query: str, concept_label: str) -> str:
    return f"""PASSAGE: {passage_text}
QUERY: {query}
ROW CONCEPT (the intended subject): {concept_label}
List every query term/concept NOT supported by the passage. Then output a final line exactly: VERDICT: GROUNDED   or   VERDICT: HALLUCINATED"""

gemma_drop_indices = []
gemma_ungrounded_terms = {}
counter_lock = threading.Lock()
counter = [0]

def verify_candidate(idx: int, ungrounded_pre: list[str]) -> tuple:
    """Returns (idx, verdict_str, gemma_response) or (idx, 'error', None)."""
    row = df.loc[idx]
    passage = str(row["passage_text"]) if pd.notna(row["passage_text"]) else ""
    query = str(row["query"]) if pd.notna(row["query"]) else ""
    concept = str(row.get("cui_label", "")) if pd.notna(row.get("cui_label")) else str(row.get("alt_term", ""))

    user = build_user_prompt(passage, query, concept)
    resp = gemma(SYSTEM_PROMPT, user, max_tokens=300)

    with counter_lock:
        counter[0] += 1
        if counter[0] % 100 == 0:
            print(f"    [{counter[0]}/{len(ungrounded_candidates)}]")

    if resp is None:
        return (idx, "error", None)

    # Parse VERDICT line
    verdict = "error"
    for line in resp.strip().split("\n"):
        line_clean = line.strip()
        if line_clean.upper().startswith("VERDICT:"):
            v = line_clean.split(":", 1)[1].strip().upper()
            if "HALLUCINATED" in v:
                verdict = "hallucinated"
            elif "GROUNDED" in v:
                verdict = "grounded"
            break
    # Fallback: if no VERDICT line, search whole response
    if verdict == "error":
        upper = resp.upper()
        if "VERDICT: HALLUCINATED" in upper or "VERDICT:  HALLUCINATED" in upper:
            verdict = "hallucinated"
        elif "VERDICT: GROUNDED" in upper or "VERDICT:  GROUNDED" in upper:
            verdict = "grounded"

    return (idx, verdict, resp)

if ungrounded_candidates:
    start_time = time.time()
    candidate_indices = [idx for idx, _ in ungrounded_candidates]

    with ThreadPoolExecutor(max_workers=VLLM_CONCURRENCY) as executor:
        futures = {executor.submit(verify_candidate, idx, ut): idx
                   for idx, ut in ungrounded_candidates}
        for future in as_completed(futures):
            idx, verdict, resp = future.result()
            if verdict == "hallucinated":
                gemma_drop_indices.append(idx)
                # Extract ungrounded terms from Gemma response
                terms = []
                if resp:
                    for line in resp.strip().split("\n"):
                        line = line.strip()
                        if line and not line.upper().startswith("VERDICT:") and not line.startswith("PASSAGE:") and not line.startswith("QUERY:") and not line.startswith("ROW CONCEPT"):
                            if len(line) < 300 and not line.startswith("List every"):
                                terms.append(line)
                gemma_ungrounded_terms[idx] = " | ".join(terms[:5]) if terms else "(see response)"
            elif verdict == "error":
                print(f"    [WARN] Gemma error on idx={idx}, skipping (conservatively kept)")

    elapsed = time.time() - start_time
    print(f"\n  Gemma verification complete in {elapsed:.0f}s")
    print(f"  HALLUCINATED: {len(gemma_drop_indices)}")
    print(f"  GROUNDED: {len(ungrounded_candidates) - len(gemma_drop_indices)}")

    # Assign drop reasons
    for idx in gemma_drop_indices:
        if drop_reason[idx] is None:
            drop_reason[idx] = "gemma_hallucinated"
else:
    print("  No candidates to verify — skipping Gemma step")

# ---- (e) Optional: further cut brand_generic to 3,000 ----
print(f"\n=== (e) Cut brand_generic to {BRAND_GENERIC_V2_CAP:,} ===")
bg_idx = df[(drop_reason.isna()) & (df["task_family"] == "brand_generic")].index
print(f"  brand_generic before cut: {len(bg_idx):,}")
if len(bg_idx) > BRAND_GENERIC_V2_CAP:
    rng = np.random.default_rng(RANDOM_STATE + 1)  # different seed from GCR cap
    keep_bg = rng.choice(bg_idx, size=BRAND_GENERIC_V2_CAP, replace=False)
    drop_bg = bg_idx.difference(keep_bg)
    drop_reason.loc[drop_bg] = "downsample_brand_generic_2"
    print(f"  Keeping {BRAND_GENERIC_V2_CAP:,}, dropping {len(drop_bg):,}")

# ---- Compile results ----
df["round2_drop_reason"] = drop_reason
kept = df[drop_reason.isna()].copy()
dropped = df[drop_reason.notna()].copy()

# Attach Gemma ungrounded terms to dropped rows where available
if gemma_ungrounded_terms:
    # Create a series for the ungrounded terms
    ut_series = pd.Series(gemma_ungrounded_terms, dtype=object)
    dropped["gemma_ungrounded_terms"] = dropped.index.map(lambda i: ut_series.get(i, ""))

print("\n=== Round 2 drop reasons ===")
print(dropped["round2_drop_reason"].value_counts().to_string())

# ---- Write outputs ----
kept.to_parquet(OUT_V2, index=False)
dropped.to_parquet(OUT_DROPPED_V2, index=False)
print(f"\nwrote {len(kept):,} cleaned v2 -> {OUT_V2}")
print(f"wrote {len(dropped):,} dropped v2  -> {OUT_DROPPED_V2}")

# ---- Composition report ----
print("\n=== FINAL v2 task_family x query_style ===")
xtab = pd.crosstab(kept["task_family"], kept["query_style"])
print(xtab.to_string())
print(f"\nTotal: {xtab.sum().sum():,} rows")

# ---- Integrity asserts ----
assert "abbreviation_to_expanded" not in kept["task_family"].values, "abbreviation family must be gone"
assert kept[kept["task_family"] == "generic_clinical_retrieval"].shape[0] <= GENERIC_CLINICAL_CAP, \
    f"generic_clinical_retrieval must be ≤ {GENERIC_CLINICAL_CAP}"
assert kept[kept["task_family"] == "brand_generic"].shape[0] <= BRAND_GENERIC_V2_CAP, \
    f"brand_generic must be ≤ {BRAND_GENERIC_V2_CAP}"
# Verify no corrupted meta queries remain (use the refined regex)
remaining_meta = kept["query"].fillna("").str.contains(META_RE) | kept["query"].fillna("").str.contains(SINCE_REFUSAL_RE)
assert remaining_meta.sum() == 0, f"{remaining_meta.sum()} corrupted meta queries remain in kept set"
print("\nALL ASSERTIONS PASSED")
print(f"\nBefore: {len(df):,} rows (Round 1 cleaned)")
print(f"After:  {len(kept):,} rows (Round 2 cleaned v2)")
print(f"Dropped in Round 2: {len(dropped):,} rows")
