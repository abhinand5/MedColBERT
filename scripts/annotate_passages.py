"""
Annotate passages with CUI labels and alternative terms from the ontology.
Output: data/processed/private/passages/real_annotated_500.json
"""
import pandas as pd
import json
import hashlib
from collections import defaultdict

TARGET_GROUPS = [
    "disorders",
    "procedures",
    "drugs_chemicals",
    "findings_signs_symptoms",
    "anatomy",
]
TARGET_VOCAB_SHIFTS = ["consumer_to_clinical", "abbreviation_to_expanded"]
PASSAGES_PATH = "data/processed/private/passages/passages.parquet"
STRINGS_PATH = "data/processed/private/ontology/concept_strings.parquet"
PAIRS_PATH = "data/processed/private/ontology/concept_pairs.parquet"
OUTPUT_PATH = "data/processed/private/passages/real_annotated_500.json"

print("=" * 60)
print("STEP 1: Loading passages")
print("=" * 60)
passages = pd.read_parquet(PASSAGES_PATH)
print(f"Loaded {len(passages)} passages")

print("=" * 60)
print("STEP 2: Loading and filtering concept strings")
print("=" * 60)
cs = pd.read_parquet(STRINGS_PATH)
print(f"Total concept strings: {len(cs):,}")

# Filter to target semantic groups
mask = cs["semantic_group"].isin(TARGET_GROUPS)
cs = cs[mask].copy()
print(f"After semantic group filter: {len(cs):,}")

# Filter by string length 5-50 chars (raw string, after stripping)
cs = cs[cs["string"].str.strip().str.len().between(5, 50)].copy()
print(f"After length filter (5-50 chars): {len(cs):,}")

# Deduplicate by normalized_string
cs = cs.drop_duplicates(subset=["normalized_string"], keep="first")
print(f"After dedup by normalized_string: {len(cs):,}")

print("=" * 60)
print("STEP 3: Loading and filtering concept pairs")
print("=" * 60)
cp = pd.read_parquet(PAIRS_PATH)
print(f"Total concept pairs: {len(cp):,}")

# Filter to consumer_to_clinical and abbreviation_to_expanded
cp = cp[cp["vocab_shift_type"].isin(TARGET_VOCAB_SHIFTS)].copy()
print(f"After vocab shift filter: {len(cp):,}")

print("=" * 60)
print("STEP 4: Building lookup structures")
print("=" * 60)

# Build normalized_string -> (cui, semantic_group) lookup from concept_strings
# Use a dict for fast lookup
string_to_concept = {}
for _, row in cs.iterrows():
    ns = row["normalized_string"]
    if ns:  # skip empty
        string_to_concept[ns] = {
            "cui": row["cui"],
            "semantic_group": row["semantic_group"],
            "string": row["string"],
        }
print(f"String->CUI lookup entries: {len(string_to_concept):,}")

# Build string_hash -> normalized_string lookup for concept strings
# We need this to resolve left/right_string_id in concept_pairs
# But first, let's build a full hash -> normalized_string map from ALL concept_strings
# (filtering would be too restrictive here since we need to map string IDs)
# Use the already-loaded cs for the filtered set, plus we need to load all
# to resolve the right_string_id (alternative term) which may not be in the filtered set
print("Loading full concept strings for hash resolution...")
cs_all = pd.read_parquet(STRINGS_PATH)
hash_to_string = {}
for _, row in cs_all.iterrows():
    sh = row["string_hash"]
    if sh:
        hash_to_string[sh] = {"string": row["string"], "normalized_string": row["normalized_string"]}
print(f"Hash->string lookup entries: {len(hash_to_string):,}")

# Build CUI -> list of (alt_term, vocab_shift_type, semantic_group) from concept_pairs
# For each pair, the left side is typically the clinical/abbreviated form,
# the right side is the consumer/expanded form (or vice versa).
# We want to match passages on the clinical/abbreviated form and provide the consumer/expanded form as alt_term
print("Building CUI->alternative terms from concept_pairs...")
# First, create a set of CUIs that are in our filtered concept_strings
target_cuis = set(cs["cui"].unique())
print(f"Target CUIs in filtered strings: {len(target_cuis):,}")

# Filter pairs to those in our target CUIs
cp_filtered = cp[cp["cui"].isin(target_cuis)].copy()
print(f"Pairs after CUI filter: {len(cp_filtered):,}")

cui_to_alt_terms = defaultdict(list)
for _, pair in cp_filtered.iterrows():
    left_info = hash_to_string.get(pair["left_string_id"])
    right_info = hash_to_string.get(pair["right_string_id"])
    if left_info is None or right_info is None:
        continue

    # For consumer_to_clinical: left is clinical, right is consumer
    # For abbreviation_to_expanded: left is abbreviation, right is expanded
    # We want to match on the left (clinical/abbreviation) and provide the right (consumer/expanded) as alt
    cui_to_alt_terms[pair["cui"]].append({
        "match_string": left_info["normalized_string"],
        "match_display": left_info["string"],
        "alt_term": right_info["string"],
        "vocab_shift_type": pair["vocab_shift_type"],
        "semantic_group": pair["semantic_group"],
    })

print(f"CUIs with alt terms: {len(cui_to_alt_terms):,}")

# Also, for backward mapping: collect all match_strings from alt_terms for fast substring search
# normalized match string -> (cui, alt_info)
match_string_to_alt = {}
for cui, alt_list in cui_to_alt_terms.items():
    for alt_info in alt_list:
        ms = alt_info["match_string"]
        if ms:
            match_string_to_alt[ms] = {
                "cui": cui,
                "alt_term": alt_info["alt_term"],
                "vocab_shift_type": alt_info["vocab_shift_type"],
                "semantic_group": alt_info["semantic_group"],
            }
print(f"Match strings for substring search: {len(match_string_to_alt):,}")

print("=" * 60)
print("STEP 5: Annotating passages")
print("=" * 60)
results = []
seen_ids = set()

# Sort match strings by length (longest first) for better matching
match_strings_sorted = sorted(match_string_to_alt.keys(), key=len, reverse=True)
print(f"Total match strings to search: {len(match_strings_sorted):,}")

for idx, (_, passage) in enumerate(passages.iterrows()):
    passage_id = passage["passage_id"]
    if passage_id in seen_ids:
        continue
    seen_ids.add(passage_id)

    text = passage.get("text", "")
    title = passage.get("title", "")

    # Truncate text to 1200 chars for speed
    if len(text) > 1200:
        text = text[:1200]

    # Build passage text for matching (lowercase, combine title + text)
    combined = (str(title) + " " + str(text)).lower()

    matched_entry = None
    best_len = 0

    for ms in match_strings_sorted:
        if ms in combined:
            alt_info = match_string_to_alt[ms]
            # Found a match - use the first one with longest match
            matched_entry = {
                "passage_id": passage_id,
                "text": passage.get("text", ""),
                "title": passage.get("title", ""),
                "source_doc_id": passage.get("source_doc_id", ""),
                "cui_label": alt_info["cui"],
                "alt_term": alt_info["alt_term"],
                "semantic_group": alt_info["semantic_group"],
                "vocab_shift_type": alt_info["vocab_shift_type"],
            }
            break  # Take first (longest) match

    if matched_entry:
        results.append(matched_entry)
        if len(results) >= 500:
            # Keep going to see if we can get more
            pass

    if (idx + 1) % 200 == 0:
        print(f"  Processed {idx + 1}/{len(passages)} passages, annotated so far: {len(results)}")

print(f"Total annotated passages: {len(results)}")

# If fewer than 500, try fuzzy matching or expand search
if len(results) < 500:
    print(f"\nOnly found {len(results)} annotations. Need 500.")

# Sort results by passage_id for consistency
results = sorted(results, key=lambda x: x["passage_id"])

print("=" * 60)
print("STEP 6: Saving results")
print("=" * 60)
# Ensure output dir exists
import os
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved {len(results)} annotations to {OUTPUT_PATH}")

print("=" * 60)
print("COUNTS BY SEMANTIC GROUP")
print("=" * 60)
group_counts = defaultdict(int)
for r in results:
    group_counts[r["semantic_group"]] += 1
for g, cnt in sorted(group_counts.items(), key=lambda x: -x[1]):
    print(f"  {g}: {cnt}")
