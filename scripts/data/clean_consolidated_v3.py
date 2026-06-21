"""MedColBERT v3 deterministic cleanup.

Starts from the Round-1 cleaned file (78,575 rows) — NOT v2, whose UMLS+Gemma
grounding filter over-fired and destroyed valid generic_clinical_retrieval rows.

v3 applies ONLY deterministic, verifiable steps:
  (a) meta-query / refusal sweep (all families)
  (b) clinical-tag mismatch heuristic (ALL families — Round 1 only did generic_clinical_retrieval).
      Drops rows where the query bolts a nursing/treatment tag onto a passage with NO
      clinical context. This is the real driver of the symptom_to_diagnosis hallucinations.
  (c) cap generic_clinical_retrieval to 10,000 (relabel dumping-ground, was 21,972)
  (d) cut brand_generic 6,000 -> 3,000 (drugs are second-class; FS=1.18 says the pipeline
      doesn't produce real brand<->generic swaps)

NO UMLS+Gemma grounding filter: it over-fires (87.6% drop on generic_clinical_retrieval,
mostly false positives). Measurement of residual hallucination is the job of the calibrated
re-audit, not a destructive filter.

Never overwrites inputs. Writes _v3 parquet + labeled dropped-rows file.
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/workspace/MedColBERT")
SRC = ROOT / "data/processed/private/synthetic/consolidated/mode4_teacher_filtered_multistyle_cleaned.parquet"
OUT_CLEAN = ROOT / "data/processed/private/synthetic/consolidated/mode4_teacher_filtered_multistyle_cleaned_v3.parquet"
OUT_DROPPED = ROOT / "data/processed/private/synthetic/consolidated/cleaned_v3_dropped_rows.parquet"

GENERIC_CLINICAL_CAP = 10_000
BRAND_GENERIC_CAP = 3_000
RANDOM_STATE = 20260621

# (a) Meta/refusal queries — first-person, self-referential, or refusal-to-generate text
META_RE = re.compile(
    r"\b(note to self|the provided fact|the above|this passage|"
    r"the passage (is|does|describes)|since the (fact|passage)|"
    r"as (an? )?(AI|language model)|I (am|cannot|don't|can't|won't)|"
    r"please provide|so I can translate)\b",
    re.I,
)
SINCE_REFUSAL_RE = re.compile(r"^Since the provided fact", re.I)

# (b) Clinical-tag mismatch — the generation template bolts nursing/treatment tags onto
# basic-science passages that have no clinical-care context.
CLINICAL_TAG_RE = re.compile(
    r"\b(nursing\s*(care|interventions?|assessment|monitoring|plan|protocols?)\b"
    r"|tx\s*protocols?|treatment\s*protocols?"
    r"|NICU\s*protocols?|TPN\s*(administration|protocol)?"
    r"|discharge\s*planning|care\s*plan)\b",
    re.I,
)
# Passage has genuine clinical context -> the tag may be legitimate, keep it.
CLINICAL_CONTEXT_RE = re.compile(
    r"\b(patient|nursing|treatment|therapy|clinical|diagnosis|diagnosed|"
    r"surgical|therapeutic|monitoring|hospital|nurse|physician|ward|"
    r"administer|dosage|prognosis|symptom)\b",
    re.I,
)

print(f"Loading {SRC.name}...")
df = pd.read_parquet(SRC)
print(f"  {len(df):,} rows")

# carry forward only needed cols + keep resolved_cui/drop_reason for lineage
df = df.copy()
df["drop_reason"] = None  # reset; v3 records its own reasons

dropped_frames = []

def move_to_dropped(mask_indices, reason):
    """Move rows at given index labels from df to dropped with a reason."""
    rows = df.loc[mask_indices].copy()
    rows["drop_reason"] = reason
    dropped_frames.append(rows)
    df.drop(index=mask_indices, inplace=True)
    return len(rows)

# ---- (a) meta / refusal sweep (all families) ----
print("\n=== (a) meta/refusal sweep (all families) ===")
q = df["query"].fillna("")
meta_mask = q.str.contains(META_RE, regex=True, na=False) | q.str.contains(SINCE_REFUSAL_RE, regex=True, na=False)
n_a = move_to_dropped(df.index[meta_mask], "corrupted_meta_query")
print(f"  dropped {n_a:,} corrupted_meta_query")
print("  spot-check 8:")
for s in df if False else []:
    pass
for _, r in pd.concat(dropped_frames).head(8).iterrows() if dropped_frames else []:
    print(f"    [{r['task_family']}] {str(r['query'])[:110]}")

# ---- (b) clinical-tag mismatch (ALL families) ----
print("\n=== (b) clinical-tag mismatch (ALL families) ===")
q = df["query"].fillna("")
p = df["passage_text"].fillna("")
has_tag = q.str.contains(CLINICAL_TAG_RE, regex=True, na=False)
no_ctx = ~p.str.contains(CLINICAL_CONTEXT_RE, regex=True, na=False)
tag_mask = has_tag & no_ctx
# report per family before dropping
print("  tag_mismatch per family:")
print(pd.crosstab(df.loc[tag_mask, "task_family"], ["drop"] * tag_mask.sum()).to_string() if tag_mask.sum() else "  (none)")
n_b = move_to_dropped(df.index[tag_mask], "clinical_tag_mismatch")
print(f"  dropped {n_b:,} clinical_tag_mismatch total")

# ---- (c) cap generic_clinical_retrieval to 10,000 ----
print("\n=== (c) cap generic_clinical_retrieval ===")
gcr_idx = df.index[df["task_family"] == "generic_clinical_retrieval"]
print(f"  before cap: {len(gcr_idx):,}")
if len(gcr_idx) > GENERIC_CLINICAL_CAP:
    rng = np.random.default_rng(RANDOM_STATE)
    keep = rng.choice(gcr_idx.to_numpy(), size=GENERIC_CLINICAL_CAP, replace=False)
    keep_set = set(keep.tolist())
    drop_idx = gcr_idx[~gcr_idx.isin(keep_set)]
    n_c = move_to_dropped(drop_idx, "downsample_generic_clinical")
    print(f"  kept {GENERIC_CLINICAL_CAP:,}, dropped {n_c:,}")
else:
    print("  under cap, no action")

# ---- (d) cut brand_generic to 3,000 ----
print("\n=== (d) cut brand_generic to 3,000 ===")
bg_idx = df.index[df["task_family"] == "brand_generic"]
print(f"  before cut: {len(bg_idx):,}")
if len(bg_idx) > BRAND_GENERIC_CAP:
    rng = np.random.default_rng(RANDOM_STATE)
    keep = rng.choice(bg_idx.to_numpy(), size=BRAND_GENERIC_CAP, replace=False)
    keep_set = set(keep.tolist())
    drop_idx = bg_idx[~bg_idx.isin(keep_set)]
    n_d = move_to_dropped(drop_idx, "downsample_brand_generic_v3")
    print(f"  kept {BRAND_GENERIC_CAP:,}, dropped {n_d:,}")

# ---- write outputs ----
df.to_parquet(OUT_CLEAN, index=False)
dropped = pd.concat(dropped_frames) if dropped_frames else pd.DataFrame(columns=df.columns)
dropped.to_parquet(OUT_DROPPED, index=False)
print(f"\nwrote {len(df):,} cleaned -> {OUT_CLEAN.name}")
print(f"wrote {len(dropped):,} dropped  -> {OUT_DROPPED.name}")

# ---- report ----
print("\n=== v3 drop reasons ===")
print(dropped["drop_reason"].value_counts().to_string())
print("\n=== v3 composition (task_family x query_style) ===")
print(pd.crosstab(df["task_family"], df["query_style"], margins=True).to_string())

# ---- assertions ----
assert "abbreviation_to_expanded" not in df["task_family"].values
assert df["query"].fillna("").str.contains(META_RE, regex=True, na=False).sum() == 0, "meta queries remain"
assert df["query"].fillna("").str.contains(SINCE_REFUSAL_RE, regex=True, na=False).sum() == 0
assert (df["task_family"] == "brand_generic").sum() <= BRAND_GENERIC_CAP
assert (df["task_family"] == "generic_clinical_retrieval").sum() <= GENERIC_CLINICAL_CAP
print("\nALL ASSERTIONS PASSED")
