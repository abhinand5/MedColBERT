#!/usr/bin/env python3
"""Prepare abbreviation→expanded passage pool for multistyle batch generation.

Query concept_pairs.parquet filtered to vocab_shift_type="abbreviation_to_expanded",
resolve string IDs to actual terms, and match passages from real_annotated_500.json
that contain abbreviated terms.

Output: A JSON array of enriched passage objects with vocab_shift_type and task_family
set to "abbreviation_to_expanded", ready for multistyle_batch_v2.py.

Usage:
  uv run python scripts/data/prepare_abbreviation_passages.py
  uv run python scripts/data/prepare_abbreviation_passages.py --max-per-abbr 3 --min-abbr-len 3
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE_DIR = Path("/workspace/MedColBERT")
CONCEPT_PAIRS_PATH = BASE_DIR / "data/processed/private/ontology/concept_pairs.parquet"
CONCEPT_STRINGS_PATH = BASE_DIR / "data/processed/private/ontology/concept_strings.parquet"
PASSAGES_PATH = BASE_DIR / "data/processed/private/passages/real_annotated_500.json"
OUTPUT_PATH = BASE_DIR / "data/processed/private/passages/abbreviation_passages.json"


# Common English words that happen to be short / all-caps and would otherwise
# pass the abbreviation filters. Used (case-insensitively) to reject alt_terms
# that are full words (LUNG, CELL, LIFE, WISH, LAMB, FEET, SEA, LEAD, ZINC, ...)
# and to EXCLUDE from the context-relevance overlap so generic words like
# "with"/"type"/"protein" don't create false homonym matches.
_COMMON_WORDS = {
    # anatomical / body
    "skin", "cell", "cells", "bone", "bones", "gene", "genes", "band", "root",
    "seed", "leaf", "hair", "nail", "lung", "lungs", "lips", "hand", "hands",
    "foot", "feet", "head", "face", "eyes", "ears", "nose", "neck", "back",
    "chest", "arm", "arms", "leg", "legs", "hip", "jaw", "rib", "ear", "eye",
    "lips", "gum", "toe", "knee", "pore", "duct", "gut", "fat", "bile", "womb",
    "cord", "lobe", "node", "flap", "sac", "sacs", "salt", "wax", "milk",
    "stem", "base", "core", "peak", "wall", "tube", "tip", "cap", "bed",
    "blood", "body", "tissue", "organ", "brain", "heart", "liver", "kidney",
    "nerve", "muscle", "vein", "artery",
    # common English (esp. 3-5 letter all-caps words seen in UMLS)
    "life", "wish", "lamb", "rays", "sea", "lead", "bind", "pain", "far",
    "bean", "band", "cast", "ring", "wave", "star", "moon", "rain", "snow",
    "wind", "fire", "wood", "iron", "gold", "zinc", "copper", "born", "dead",
    "male", "female", "man", "men", "woman", "women", "boy", "girl", "child",
    "adult", "baby", "age", "ages", "old", "new", "good", "bad", "big", "low",
    "high", "mid", "end", "top", "side", "sides", "left", "right", "front",
    "rear", "near", "far", "out", "off", "now", "then", "here", "there",
    "what", "when", "where", "which", "while", "white", "black", "red",
    "blue", "green", "yellow", "brown", "pink", "gray", "grey", "dark",
    "light", "long", "short", "tall", "wide", "thin", "thick", "deep",
    "fats", "fat", "cure", "task", "map", "maps", "race", "rest", "test",
    "tests", "work", "play", "talk", "walk", "talk", "read", "make", "made",
    "take", "took", "give", "gave", "come", "came", "look", "find", "found",
    "time", "times", "date", "week", "year", "years", "month", "day", "days",
    "fact", "acts", "core", "data", "info", "news", "view", "area", "part",
    "role", "rule", "base", "form", "line", "point", "set", "sets", "net",
    "web", "site", "step", "steps", "page", "book", "name", "names", "word",
    # full chemical/neurotransmitter names that are NOT abbreviations
    "serotonin", "dopamine", "adrenaline", "noradrenaline", "histamine",
    "glutamate", "glycine", "acetylcholine", "endorphin", "melatonin",
    "insulin", "glucagon", "heparin", "penicillin", "streptomycin",
    "chloramphenicol", "tetracycline", "oxytetracycline", "chlortetracycline",
    "dexamethasone", "phenytoin", "phenobarbital", "isoproterenol",
    "phenylpropanolamine", "phenoxybenzamine", "hemicellulose", "nitrofen",
    "phenylhydrazone", "adenosine", "chlordiazepoxide", "dicumarol",
    "cytarabine", "cycloheximide", "durapatite", "erythrocytes", "injection",
    "metabolite", "nucleotides", "oligosaccharides", "peptidoglycan",
    "polysaccharides", "protons", "pyrimidine", "streptomycin", "carbohydrates",
    "glycosaminoglycans", "hydrogen", "hydroxylamine", "phosphate",
    "phosphatidylcholines", "phosphatidylethanolamines", "phosphoenolpyruvate",
    "aminoglycosides", "antibiotics", "bromodeoxyuridine", "dinitrochlorobenzene",
    "ferricyanides", "isoflurophate", "malondialdehyde", "methylhistamines",
    "glutaral", "diclofenac", "ibuprofen", "aspirin", "warfarin",
    # generic biomedical / stopwords that must NOT count as relevance overlap
    "with", "from", "into", "onto", "upon", "over", "under", "this", "that",
    "these", "those", "type", "types", "like", "also", "than", "such", "each",
    "both", "some", "any", "all", "one", "two", "three", "first", "second",
    "factor", "factors", "protein", "proteins", "gene", "genes", "membrane",
    "nuclear", "nucleus", "acid", "level", "levels", "rate", "rates",
    "activity", "effect", "effects", "response", "result", "results", "study",
    "group", "groups", "patient", "patients", "case", "cases", "method",
    "human", "animal", "system", "disease", "syndrome", "disorder", "cell",
    "cells", "tissue", "tissues", "blood", "body", "clinical", "medical",
    "health", "treatment", "therapy", "care", "dose", "acute", "chronic",
    "primary", "secondary", "general", "specific", "particular", "certain",
    "various", "multiple", "several", "associated", "related", "following",
    "human", "product", "products", "complex", "subunit", "family", "class",
    "group", "member", "chain", "region", "domain", "site",
}


def _content_words(text: str) -> set:
    """Distinctive content words (>=4 chars, alpha, not generic) for overlap."""
    return {w for w in re.findall(r"[a-z]+", (text or "").lower())
            if len(w) >= 4 and w not in _COMMON_WORDS}


def strip_umls_formatting(term: str) -> str:
    """Strip UMLS parenthetical qualifiers and bracket artifacts from a term.

    Removes:
      - Parenthetical qualifiers: "Strychnine (substance)" → "Strychnine"
      - Bracket artifacts: "Device [medical device]" → "Device"
      - Nested/complex: "X [B X]" → "X"
    """
    if not term or not isinstance(term, str):
        return term
    # Strip parenthetical qualifiers: (substance), (finding), (diagnosis), etc.
    term = re.sub(r'\s*\([^)]*\)', '', term)
    # Strip bracket artifacts: [medical device], [brand name], [B X], etc.
    term = re.sub(r'\s*\[[^\]]*\]', '', term)
    # Clean up any double spaces
    term = re.sub(r'\s+', ' ', term).strip()
    return term


def load_abbreviation_pairs(
    min_abbr_len: int = 2,
    semantic_groups: list[str] | None = None,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    """Load abbreviation→expanded pairs from concept_pairs, resolved to string values.

    Returns DataFrame with columns: cui, abbreviated, expanded, semantic_group
    """
    print("Loading concept_pairs...")
    pairs = pd.read_parquet(CONCEPT_PAIRS_PATH)
    pairs = pairs[pairs["vocab_shift_type"] == "abbreviation_to_expanded"]
    print(f"  {len(pairs):,} abbreviation_to_expanded pairs")

    if semantic_groups:
        pairs = pairs[pairs["semantic_group"].isin(semantic_groups)]
        print(f"  {len(pairs):,} after filtering to {len(semantic_groups)} semantic groups")

    print("Loading concept_strings...")
    strings = pd.read_parquet(CONCEPT_STRINGS_PATH)
    # Only need abbreviated strings for matching
    abbr_strings = strings[strings["is_abbreviation"] == True][
        ["string_hash", "string", "normalized_string", "cui"]
    ].copy()
    expanded_strings = strings[strings["is_abbreviation"] == False][
        ["string_hash", "string", "normalized_string", "cui"]
    ].copy()
    print(f"  {len(abbr_strings):,} abbreviated strings, {len(expanded_strings):,} expanded strings")

    # Resolve left (abbreviated) side
    print("Resolving abbreviated terms (left side)...")
    pairs_with_abbr = pairs.merge(
        abbr_strings.rename(columns={
            "string_hash": "left_string_id",
            "string": "abbreviated",
            "normalized_string": "abbreviated_norm",
        }),
        on=["left_string_id", "cui"],
        how="inner",
    )
    print(f"  {len(pairs_with_abbr):,} pairs with resolved abbreviated terms")

    # Resolve right (expanded) side
    print("Resolving expanded terms (right side)...")
    full = pairs_with_abbr.merge(
        expanded_strings.rename(columns={
            "string_hash": "right_string_id",
            "string": "expanded",
            "normalized_string": "expanded_norm",
        }),
        on=["right_string_id", "cui"],
        how="inner",
    )
    print(f"  {len(full):,} fully resolved pairs")

    # Filter by abbreviation length
    if min_abbr_len > 0:
        full = full[full["abbreviated"].str.len() >= min_abbr_len]
        print(f"  {len(full):,} after min_abbr_len >= {min_abbr_len}")

    # Filter out abbreviations that are single letters or just numbers
    full = full[~full["abbreviated"].str.match(r'^[0-9\s\.\,\-\+\(\)\[\]\{\}]+$')]
    print(f"  {len(full):,} after removing purely numeric/symbol abbreviations")

    # Filter out very long abbreviations (likely not real abbreviations)
    full = full[full["abbreviated"].str.len() <= 20]
    print(f"  {len(full):,} after max abbreviation length filter")

    # ── Strict "real abbreviation" filter ─────────────────────────────────
    # Only keep abbreviations that look like real medical acronyms/initialisms:
    #   a) All uppercase (2-10 chars) — e.g., COPD, CABG, DPPC, IBMX
    #   b) Contain digits — e.g., 2,4-D, 5-FU, IL-6
    #   c) Mixed case but short (2-5 chars) with clear abbreviation pattern
    def is_real_abbreviation(s: str) -> bool:
        if not s or not isinstance(s, str):
            return False
        s_clean = s.strip()
        if len(s_clean) < 2 or len(s_clean) > 15:
            return False
        # Must contain at least one letter
        if not any(c.isalpha() for c in s_clean):
            return False

        # Reject pure numbers / numeric-led staging codes (e.g. "2", "16", "223",
        # "T2", "N0") — these are cancer-staging/numeric tokens, not abbreviations.
        # An alt_term must contain at least one alphabetic LETTER run of length >=2
        # to qualify as an abbreviation/acronym.
        alpha_runs = [r for r in re.findall(r"[A-Za-z]+", s_clean) if len(r) >= 2]
        if not alpha_runs:
            return False

        # Reject common English/anatomical words that happen to be short / all-caps
        # and would otherwise pass the acronym rule (SKIN, CELL, BONE, LUNG, GENE,
        # LIFE, WISH, LAMB, FEET, SEA, LEAD, ZINC, ...). These are full words,
        # not abbreviations. Case-insensitive check covers all-caps variants.
        if s_clean.lower() in _COMMON_WORDS:
            return False

        has_lower = any(c.islower() for c in s_clean)
        has_upper = any(c.isupper() for c in s_clean)
        has_digit = any(c.isdigit() for c in s_clean)
        is_all_alpha = s_clean.isalpha()

        # ── Reject full English/medical words disguised as abbreviations ──
        # All-caps all-alpha words ≥5 chars are almost certainly full words
        # (e.g., MORPHINE, ANGINA, OBESE, DROWNING, BUSULFAN), not abbreviations.
        if not has_lower and not has_digit and is_all_alpha and len(s_clean) >= 5:
            return False
        # Mixed-case all-alpha words ≥6 chars are full words (e.g., Morphine, Angina)
        if has_lower and not has_digit and is_all_alpha and len(s_clean) >= 6:
            return False

        # ── Accept genuine abbreviation patterns ──
        # All-caps short acronym (2-4 chars): COPD, CABG, IL-6, 2,4-D
        if not has_lower and len(s_clean) <= 4:
            return True
        # All-caps with digits (5-10 chars): 5-FU, IL-6R, H2O2
        if not has_lower and has_digit and len(s_clean) <= 10:
            return True

        # Short mixed-case / lowercase strings: accept ONLY if there is a digit
        # (Mdm2, p53, Ki67, Ca2+) OR internal capitalization — i.e. a camelCase
        # acronym where an uppercase letter appears AFTER position 0 (cGMP, hCG,
        # tPA, mRNA, aCL). A plain Title-case word (Lung, Eels, Claw, Crohn,
        # Bubo, Duct) or an all-lowercase word (zinc, and, epi) is a common word,
        # NOT an abbreviation, and must be rejected.
        if has_lower and len(s_clean) <= 8:
            if has_digit:
                return True
            if has_upper and any(c.isupper() for c in s_clean[1:]):
                return True
            return False
        # Longer mixed-case with digits (9-15 chars): rare alphanumeric forms
        if has_lower and has_digit and len(s_clean) <= 15:
            return True

        return False

    full = full[full["abbreviated"].apply(is_real_abbreviation)]
    print(f"  {len(full):,} after strict abbreviation filter (acronyms, alphanumeric, short forms)")

    # ── Strip UMLS formatting codes ─────────────────────────────────────
    print("Stripping UMLS formatting codes from terms...")
    full["abbreviated"] = full["abbreviated"].apply(strip_umls_formatting)
    full["abbreviated_norm"] = full["abbreviated_norm"].apply(strip_umls_formatting)
    full["expanded"] = full["expanded"].apply(strip_umls_formatting)
    full["expanded_norm"] = full["expanded_norm"].apply(strip_umls_formatting)
    # Drop rows where stripping emptied a term
    full = full[full["abbreviated"].str.len() > 0]
    full = full[full["expanded"].str.len() > 0]
    print(f"  {len(full):,} after stripping UMLS formatting")

    # ── Filter by expanded length: ≥2x abbreviation length ──────────────
    full = full[full["expanded"].str.len() >= 2 * full["abbreviated"].str.len()]
    print(f"  {len(full):,} after expanded-length ≥2x filter")

    # ── Filter: different root words ────────────────────────────────────
    # Expanded form should use substantially different words from the abbreviation
    def has_different_roots(abbr: str, expanded: str) -> bool:
        """Check that expanded form uses different root words from abbreviation."""
        if not abbr or not expanded:
            return False
        abbr_words = set(abbr.lower().split())
        expd_words = set(expanded.lower().split())
        # If the abbreviation is an acronym (all caps, no spaces), it won't
        # share words with the expansion at all — that's the ideal case.
        if " " not in abbr.strip():
            # For single-word abbreviations/acronyms, check substring embedding
            # and letter-by-letter matching
            abbr_clean = abbr.strip().lower()
            # Check if abbr is an initialism of the expanded form
            expd_word_list = expanded.lower().split()
            # If the abbreviation's letters match initial letters of expanded words,
            # that's a genuine initialism (e.g., COPD → chronic obstructive pulmonary disease)
            is_init = False
            if len(abbr_clean) >= 2 and all(c.isalpha() for c in abbr_clean):
                initials = "".join(w[0] for w in expd_word_list if w)
                if abbr_clean == initials[:len(abbr_clean)] or any(
                    abbr_clean == "".join(w[0] for w in expd_word_list[i:])[:len(abbr_clean)]
                    for i in range(len(expd_word_list))
                ):
                    is_init = True
            # Check substantial substring overlap — reject if abbr IS a full word
            # that appears in the expanded form (e.g., MIGRAINE → migraine headache)
            abbr_in_expanded = abbr_clean in expanded.lower()
            # If abbr is ≥4 chars and appears as a full word/substring in expanded,
            # it's not a real abbreviation — it's a repetition
            if abbr_in_expanded and len(abbr_clean) >= 4:
                # But exempt if abbr is short (2-3 chars) and appears coincidentally
                if len(abbr_clean) <= 3:
                    return is_init or not abbr_in_expanded
                return False
            return True
        # Multi-word abbreviations — check word overlap
        shared = abbr_words & expd_words
        # Allow up to 30% word overlap
        if len(shared) / max(len(abbr_words), 1) > 0.3:
            return False
        return True

    full = full[full.apply(lambda r: has_different_roots(r["abbreviated"], r["expanded"]), axis=1)]
    print(f"  {len(full):,} after different-root-words filter")

    # ── Filter out UMLS qualifier types (T080, T081) ───────────────────
    # Join with concept_strings to get semantic_types for filtering
    print("Filtering T080/T081 qualifier types...")
    # Get CUIs to check
    cuis_to_check = full["cui"].unique()
    # Build a set of CUIs that have T080 or T081 semantic types
    strings_with_bad_types = strings[
        strings["cui"].isin(cuis_to_check)
    ]
    import numpy as np
    def has_bad_semantic_type(sem_types) -> bool:
        """Check if semantic_types array contains T080 or T081."""
        if sem_types is None:
            return False
        try:
            types_list = list(sem_types) if hasattr(sem_types, '__iter__') else [sem_types]
            return any(t in ("T080", "T081") for t in types_list)
        except Exception:
            return False

    bad_cuis = set(
        strings_with_bad_types[strings_with_bad_types["semantic_types"].apply(has_bad_semantic_type)]["cui"].unique()
    )
    full = full[~full["cui"].isin(bad_cuis)]
    print(f"  {len(full):,} after filtering T080/T081 qualifier CUIs ({len(bad_cuis)} CUIs filtered)")

    if max_pairs:
        full = full.head(max_pairs)

    result = full[
        ["cui", "abbreviated", "abbreviated_norm", "expanded", "expanded_norm",
         "semantic_group", "pair_id"]
    ].drop_duplicates(subset=["abbreviated", "expanded"])
    print(f"  {len(result):,} after deduplication by (abbreviated, expanded)")
    return result


def build_abbreviation_index(pairs: pd.DataFrame) -> dict[str, list[dict]]:
    """Build a lookup from normalized abbreviation → list of (cui, expanded, semantic_group) entries.

    Defensive filter: skip any pair whose abbreviated form (or its normalized
    key) is a pure number or lacks an alphabetic run of length >=2. These are
    cancer-staging/numeric tokens (e.g. "2", "223", "T2") that are not genuine
    abbreviations; some slip through pair loading via normalized-form collisions.
    Also skip any whose abbreviated form is a common English word.
    """
    index = defaultdict(list)
    skipped = 0
    for _, row in pairs.iterrows():
        abbr = str(row["abbreviated"]).strip()
        abbr_norm = str(row["abbreviated_norm"]).lower().strip()
        # Require a letter run of length >=2 in BOTH forms
        if not (re.findall(r"[A-Za-z]{2,}", abbr) and re.findall(r"[A-Za-z]{2,}", abbr_norm)):
            skipped += 1
            continue
        if abbr_norm in _COMMON_WORDS or abbr.lower() in _COMMON_WORDS:
            skipped += 1
            continue
        index[abbr_norm].append({
            "cui": row["cui"],
            "abbreviated": row["abbreviated"],
            "expanded": row["expanded"],
            "semantic_group": row["semantic_group"],
            "pair_id": row["pair_id"],
        })
    print(f"Indexed {len(index):,} unique normalized abbreviations "
          f"(skipped {skipped:,} numeric/common-word pairs)")
    return index


def match_passages(
    passages: list[dict],
    abbr_index: dict[str, list[dict]],
    max_per_abbr: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Match passages against abbreviation index.

    For each abbreviation, find passages that contain it (case-insensitive word-boundary match).
    Cap passages per abbreviation to avoid over-representation.
    Each passage gets one abbreviation match (first found).
    """
    random.seed(seed)

    # Build a regex pattern for all abbreviations (longer first to match longest possible)
    sorted_abbrs = sorted(abbr_index.keys(), key=len, reverse=True)
    print(f"Building regex from {len(sorted_abbrs):,} abbreviations...")

    # For efficiency, use word-boundary matching
    # Escape special regex chars in abbreviations
    def escape_regex(s: str) -> str:
        return re.escape(s)

    # Track which passages matched which abbreviations
    abbr_passage_map = defaultdict(list)  # abbr_norm → [(passage_idx, passage)]
    matched_passage_indices = set()

    print(f"Matching against {len(passages):,} passages...")

    # Strategy: iterate through passages, check each abbreviation
    # This is O(passages * abbreviations) which is too large
    # Instead, use a more targeted approach

    # Alternative: tokenize each passage, look up tokens in abbr_index
    # But abbreviations can be multi-word (e.g., "acute renal failure")
    # Better: use regex OR of all abbreviations against each passage

    # For 66K passages and 350K abbreviations, full regex is impractical
    # Strategy: use a trie-like approach - build prefix index of abbreviations
    # Or: only match abbreviations that are 2+ chars and look for them as whole words

    # Practical approach: build batches of abbreviations, scan each passage
    # But this is still expensive. Let's use a simpler method:
    # For each passage, extract unique lowercase words and n-grams (1-4 words),
    # then check against abbr_index

    print("  Tokenizing passages and matching...")
    abbr_passages = []  # final output: enriched passages
    abbr_to_use_count = defaultdict(int)

    for p_idx, passage in enumerate(passages):
        if p_idx % 5000 == 0 and p_idx > 0:
            print(f"    Processed {p_idx}/{len(passages)} passages, "
                  f"found {len(abbr_passages)} matched so far")

        text = passage.get("text", "").lower()
        if not text:
            continue

        # Tokenize into words
        words = re.findall(r'[a-z0-9]+', text)
        if not words:
            continue

        # Generate n-grams (1 to 5 words) that could be abbreviations
        # Most medical abbreviations are 1-3 words
        matched_abbrs = set()
        for n in range(1, 6):
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i + n])
                if ngram in abbr_index:
                    matched_abbrs.add(ngram)

        if not matched_abbrs:
            continue

        # Pick the best abbreviation (longest match, then random among ties)
        best_abbrs = sorted(matched_abbrs, key=len, reverse=True)
        # Prefer abbreviations we haven't used much yet
        best_abbr = min(best_abbrs, key=lambda a: abbr_to_use_count[a])

        if abbr_to_use_count[best_abbr] >= max_per_abbr:
            continue  # This abbreviation has enough passages already

        # ── Context-aware expansion + relevance filtering ───────────────
        # For abbreviations with multiple expansions, pick the one whose
        # expanded terms overlap best with the passage text.
        candidates = abbr_index[best_abbr]
        passage_words_lower = set(re.findall(r'[a-z0-9]+', text))

        # Filter: kill homonym mismatches where an abbreviation's WRONG expansion
        # is chosen (e.g. "TTP"→"thrombotic thrombocytopenic purpura" in a
        # thromboglobulin/platelet passage; "DHS"→"drug-induced hypersensitivity
        # syndrome" in a "delayed hypersensitivity" passage; "AIDS"→"acquired
        # immune deficiency syndrome" in a haemolytic-anaemia passage). These
        # share ≥2 distinctive words with the wrong expansion, so word-overlap
        # alone cannot reject them.
        #
        # A candidate is accepted only if the abbreviation is used in THIS sense
        # in the passage, confirmed by EITHER:
        #   (a) DEFINITIONAL CO-OCCURRENCE: the abbreviation appears in the
        #       passage within 80 chars of ≥2 distinctive expansion words (the
        #       "expanded (ABBR)" / "ABBR, expanded" definitional pattern), OR
        #   (b) STRONG OVERLAP: ≥3 distinctive expansion content-words appear
        #       anywhere in the passage (the passage substantively spells out the
        #       expansion). This catches cases where the abbreviation is used
        #       repeatedly alongside its expansion.
        # Using _content_words (>=4 chars, alpha, EXCLUDING generic stopwords)
        # so "with"/"type"/"protein"/"gene" never count.
        passage_content = _content_words(text)
        abbr_lower = best_abbr.lower()
        # Pre-find abbreviation positions once per passage
        abbr_positions = [m.start() for m in re.finditer(re.escape(abbr_lower), text)] \
            if abbr_lower else []

        def _is_definitional(cand):
            exp_words = _content_words(cand["expanded"])
            if not exp_words or not abbr_positions:
                return False
            for pos in abbr_positions:
                window = text[max(0, pos - 80): pos + 80]
                if len(exp_words & _content_words(window)) >= 2:
                    return True
            return False

        if len(candidates) > 1 or True:  # always run relevance check
            relevant = []
            for cand in candidates:
                expanded_words = _content_words(cand["expanded"])
                if not expanded_words:
                    # No distinctive content words to verify against — skip
                    # (can't confirm the abbreviation is used in this sense).
                    continue
                overlap = expanded_words & passage_content
                definitional = _is_definitional(cand)
                strong_overlap = len(overlap) >= 3
                # Accept if the abbreviation is used in this sense: definitional
                # co-occurrence (best signal) OR strong (≥3) distinctive overlap.
                if definitional or strong_overlap:
                    # Score: definitional wins; otherwise by overlap count
                    score = 100 + len(overlap) if definitional else len(overlap)
                    relevant.append((score, cand))

            if relevant:
                # Pick the one with highest overlap
                entry = max(relevant, key=lambda x: x[0])[1]
            else:
                # No relevant expansion found — skip this abbreviation for this passage
                continue

        abbr_to_use_count[best_abbr] += 1

        # Create enriched passage
        enriched = dict(passage)  # preserve all original fields
        enriched["vocab_shift_type"] = "abbreviation_to_expanded"
        enriched["task_family"] = "abbreviation_to_expanded"
        # cui_label = term that appears in the passage (the expanded/formal form)
        # alt_term = term that should appear in the query (the abbreviation)
        enriched["cui_label"] = entry["expanded"]       # what's in the passage
        enriched["alt_term"] = entry["abbreviated"]     # what the query should use
        enriched["target_cui"] = entry["cui"]
        enriched["abbreviation_matched"] = best_abbr
        enriched["pair_id"] = entry["pair_id"]
        enriched["_original_semantic_group"] = passage.get("semantic_group", "other")

        # Update semantic_group to match the pair's semantic group if available
        if entry.get("semantic_group"):
            enriched["semantic_group"] = entry["semantic_group"]

        abbr_passages.append(enriched)
        matched_passage_indices.add(p_idx)

    print(f"  Matched {len(abbr_passages)} passages across "
          f"{len(abbr_to_use_count)} unique abbreviations")
    return abbr_passages


def main():
    parser = argparse.ArgumentParser(
        description="Prepare abbreviation→expanded passage pool for batch generation"
    )
    parser.add_argument("--min-abbr-len", type=int, default=2,
                        help="Minimum abbreviation length (chars)")
    parser.add_argument("--max-per-abbr", type=int, default=5,
                        help="Max passages per abbreviation")
    parser.add_argument("--semantic-groups", type=str, default=None,
                        help="Comma-separated semantic groups to include (default: all)")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Cap total pairs to process")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    semantic_groups = None
    if args.semantic_groups:
        semantic_groups = [sg.strip() for sg in args.semantic_groups.split(",")]

    # 1. Load abbreviation pairs
    pairs = load_abbreviation_pairs(
        min_abbr_len=args.min_abbr_len,
        semantic_groups=semantic_groups,
        max_pairs=args.max_pairs,
    )
    print(f"\nFinal pair count: {len(pairs):,}")
    print(f"Semantic groups: {pairs['semantic_group'].value_counts().to_dict()}")

    # 2. Build index
    abbr_index = build_abbreviation_index(pairs)

    # 3. Load passages
    print(f"\nLoading passages from {PASSAGES_PATH}...")
    with open(PASSAGES_PATH, "r") as f:
        passages = json.load(f)
    print(f"  {len(passages):,} passages loaded")

    # 4. Match
    print("\nMatching passages against abbreviations...")
    matched = match_passages(
        passages, abbr_index,
        max_per_abbr=args.max_per_abbr,
        seed=args.seed,
    )

    # 5. Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(matched, f, indent=2)
    print(f"\nWrote {len(matched):,} abbreviation-enriched passages to {args.output}")

    # Stats
    sg_counts = defaultdict(int)
    for p in matched:
        sg_counts[p.get("semantic_group", "other")] += 1
    print(f"\nPassages by semantic group:")
    for sg, count in sorted(sg_counts.items(), key=lambda x: -x[1]):
        print(f"  {sg}: {count:,}")


if __name__ == "__main__":
    main()
