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

# ---- targeted fix: drop clinical-tag hallucinations in generic_clinical_retrieval ----
# Pattern: template appended "nursing care", "nursing interventions", "tx protocols"
# etc. to basic-science/lab passages that contain no clinical-care context.
CLINICAL_TAG_RE = re.compile(
    r"\b(nursing\s*(care|interventions|assessment|monitoring|plan)\b"
    r"|tx\s*protocols|treatment\s*protocols)", re.I
)
CLINICAL_CONTEXT_RE = re.compile(
    r"\b(patient|nursing|treatment|therapy|clinical|diagnosis|surgical|therapeutic|monitoring)\b", re.I
)
gcr_mask = kept["task_family"] == "generic_clinical_retrieval"
has_clinical_tags = kept.loc[gcr_mask, "query"].fillna("").str.contains(CLINICAL_TAG_RE)
no_clinical_context = ~kept.loc[gcr_mask, "passage_text"].fillna("").str.contains(CLINICAL_CONTEXT_RE)
tag_hallucination = has_clinical_tags & no_clinical_context
tag_hallucination = tag_hallucination[tag_hallucination].index
if len(tag_hallucination) > 0:
    # Move these rows from kept to dropped with the new reason
    halluc_rows = kept.loc[tag_hallucination].copy()
    halluc_rows["drop_reason"] = "clinical_tag_mismatch"
    dropped = pd.concat([dropped, halluc_rows], ignore_index=False)
    kept = kept.drop(tag_hallucination)
    print(f"dropped {len(tag_hallucination):,} generic_clinical_retrieval rows with hallucinated clinical tags")

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
