#!/usr/bin/env python3
"""Ingest and annotate additional PubMed passages for scaling to 100K.

Streams from HuggingFace MedRAG/pubmed, annotates with CUI labels + alt terms
from the UMLS ontology, and appends to the annotated passages file.

Usage:
  uv run python scripts/data/ingest_more_passages.py --count 5000
  uv run python scripts/data/ingest_more_passages.py --count 10000 --start 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE_DIR = Path("/workspace/MedColBERT")
STRINGS_PATH = BASE_DIR / "data/processed/private/ontology/concept_strings.parquet"
PAIRS_PATH = BASE_DIR / "data/processed/private/ontology/concept_pairs.parquet"
ANNOTATED_PATH = BASE_DIR / "data/processed/private/passages/real_annotated_500.json"
OUTPUT_PATH = BASE_DIR / "data/processed/private/passages/real_annotated.json"

TARGET_GROUPS = [
    "disorders",
    "procedures",
    "drugs_chemicals",
    "findings_signs_symptoms",
    "anatomy",
]
TARGET_VOCAB_SHIFTS = ["consumer_to_clinical", "abbreviation_to_expanded"]


def load_ontology():
    """Build the match_string → alt_term lookup from ontology files."""
    print("Loading concept strings...")
    cs = pd.read_parquet(STRINGS_PATH)

    # Filter to target semantic groups
    mask = cs["semantic_group"].isin(TARGET_GROUPS)
    cs = cs[mask].copy()
    cs = cs[cs["string"].str.strip().str.len().between(5, 50)].copy()
    cs = cs.drop_duplicates(subset=["normalized_string"], keep="first")
    print(f"  {len(cs):,} filtered concept strings")

    # Full strings for hash resolution
    cs_all = pd.read_parquet(STRINGS_PATH)
    hash_to_string = {}
    for _, row in cs_all.iterrows():
        sh = row["string_hash"]
        if sh:
            hash_to_string[sh] = {
                "string": row["string"],
                "normalized_string": row["normalized_string"],
            }
    print(f"  {len(hash_to_string):,} hash→string entries")

    # Load and filter pairs
    print("Loading concept pairs...")
    cp = pd.read_parquet(PAIRS_PATH)
    cp = cp[cp["vocab_shift_type"].isin(TARGET_VOCAB_SHIFTS)].copy()
    print(f"  {len(cp):,} filtered pairs")

    target_cuis = set(cs["cui"].unique())
    cp = cp[cp["cui"].isin(target_cuis)].copy()
    print(f"  {len(cp):,} pairs for target CUIs")

    # Build match_string → alt_term lookup
    print("Building match_string → alt_term lookup...")
    match_string_to_alt = {}
    for _, pair in cp.iterrows():
        left_info = hash_to_string.get(pair["left_string_id"])
        right_info = hash_to_string.get(pair["right_string_id"])
        if left_info is None or right_info is None:
            continue
        ms = left_info["normalized_string"]
        if ms and ms not in match_string_to_alt:
            match_string_to_alt[ms] = {
                "cui": pair["cui"],
                "match_display": left_info["string"],
                "alt_term": right_info["string"],
                "vocab_shift_type": pair["vocab_shift_type"],
                "semantic_group": pair["semantic_group"],
            }

    # Sort match strings by length (longest first) for best matching
    match_strings_sorted = sorted(match_string_to_alt.keys(), key=len, reverse=True)
    print(f"  {len(match_strings_sorted):,} match strings ready for annotation")
    return match_string_to_alt, match_strings_sorted


def annotate_passage(text: str, title: str, match_string_to_alt: dict,
                     match_strings_sorted: list) -> dict | None:
    """Annotate a single passage with CUI label and alt term."""
    combined = (str(title) + " " + str(text)[:1200]).lower()

    for ms in match_strings_sorted:
        if ms in combined:
            alt_info = match_string_to_alt[ms]
            return {
                "cui_label": alt_info["cui"],
                "alt_term": alt_info["alt_term"],
                "semantic_group": alt_info["semantic_group"],
                "vocab_shift_type": alt_info["vocab_shift_type"],
            }
    return None


def ingest_from_pubmed(target_count: int, start: int = 0,
                       match_string_to_alt: dict | None = None,
                       match_strings_sorted: list | None = None) -> list[dict]:
    """Stream from MedRAG/pubmed and annotate passages."""
    from datasets import load_dataset

    print(f"\nStreaming MedRAG/pubmed (starting at {start}, target {target_count})...")
    ds = load_dataset("MedRAG/pubmed", split="train", streaming=True)

    results = []
    seen_doc_ids = set()
    annotated = 0
    skipped_no_match = 0
    skipped_too_short = 0
    start_time = time.time()

    for i, example in enumerate(ds):
        if i < start:
            if i % 10000 == 0:
                print(f"  Skipping... {i}/{start}")
            continue

        doc_id = str(example.get("id", example.get("pmid", str(i))))
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)

        title = example.get("title", "") or ""
        content = example.get("content", example.get("abstract", "")) or ""
        text = f"{title}\n{content}".strip() if title else content

        if not text or len(text) < 80:
            skipped_too_short += 1
            continue

        annotation = annotate_passage(text, title, match_string_to_alt, match_strings_sorted)

        if annotation:
            results.append({
                "passage_id": f"pubmed_{doc_id}_0",
                "text": text,
                "title": title,
                "source_doc_id": doc_id,
                **annotation,
            })
            annotated += 1
        else:
            skipped_no_match += 1

        if annotated >= target_count:
            break

        if (annotated + skipped_no_match) % 200 == 0:
            elapsed = time.time() - start_time
            rate = annotated / elapsed if elapsed > 0 else 0
            print(f"  Processed {annotated + skipped_no_match + skipped_too_short:,} docs, "
                  f"annotated={annotated} ({rate:.1f}/s), "
                  f"no_match={skipped_no_match}, too_short={skipped_too_short}")

    elapsed = time.time() - start_time
    print(f"\nDone. {annotated} annotated, {skipped_no_match} no match, "
          f"{skipped_too_short} too short in {elapsed:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="Ingest and annotate PubMed passages")
    parser.add_argument("--count", type=int, default=5000,
                        help="Number of annotated passages to ingest")
    parser.add_argument("--start", type=int, default=0,
                        help="Start offset in MedRAG/pubmed stream")
    parser.add_argument("--append", action="store_true", default=True,
                        help="Append to existing annotated passages (default: True)")
    parser.add_argument("--no-append", action="store_true",
                        help="Start fresh (don't append to existing)")
    args = parser.parse_args()

    # Load ontology
    match_string_to_alt, match_strings_sorted = load_ontology()

    # Load existing annotated passages
    existing = []
    if not args.no_append and ANNOTATED_PATH.exists():
        with open(ANNOTATED_PATH) as f:
            existing = json.load(f)
        print(f"\nExisting annotated passages: {len(existing)}")
        existing_ids = {p["passage_id"] for p in existing}
        print(f"  Unique passage IDs: {len(existing_ids)}")

    # Ingest new passages
    new_passages = ingest_from_pubmed(
        target_count=args.count,
        start=args.start,
        match_string_to_alt=match_string_to_alt,
        match_strings_sorted=match_strings_sorted,
    )

    # Deduplicate against existing
    existing_ids = {p["passage_id"] for p in existing}
    new_unique = [p for p in new_passages if p["passage_id"] not in existing_ids]
    print(f"\nNew unique passages: {len(new_unique)} (removed {len(new_passages) - len(new_unique)} duplicates)")

    # Merge
    all_passages = existing + new_unique
    print(f"Total passages: {len(all_passages)}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_passages, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")

    # Also update the canonical file
    with open(ANNOTATED_PATH, "w") as f:
        json.dump(all_passages, f, indent=2)
    print(f"Updated {ANNOTATED_PATH}")

    # Print group distribution
    print("\nSemantic group distribution:")
    group_counts = defaultdict(int)
    for p in all_passages:
        group_counts[p.get("semantic_group", "unknown")] += 1
    for g, cnt in sorted(group_counts.items(), key=lambda x: -x[1]):
        print(f"  {g}: {cnt}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
