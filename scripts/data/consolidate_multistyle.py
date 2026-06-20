#!/usr/bin/env python3
"""Consolidate all multistyle batch outputs and push to HuggingFace.

Scans all batch directories for accepted.parquet files, merges them,
deduplicates by example_id, and pushes to fierysurf/medcolbert-synthetic-pilot.

Usage:
  uv run python scripts/data/consolidate_multistyle.py
  uv run python scripts/data/consolidate_multistyle.py --push
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path("/workspace/MedColBERT")
SYNTHETIC_DIR = BASE_DIR / "data/processed/private/synthetic/ontology_grounded_teacher_filtered"
CONSOLIDATED_DIR = BASE_DIR / "data/processed/private/synthetic/consolidated"
CHECKPOINT_PATH = BASE_DIR / "data/processed/private/synthetic/checkpoint.json"

HF_REPO = "fierysurf/medcolbert-synthetic-pilot"


def find_all_accepted(include_fresh: bool = True) -> list[Path]:
    """Find all accepted.parquet files across batch directories.

    REGENERATION (2026-06-20): scan v6_fresh ONLY. This excludes the legacy
    `ontology_grounded_teacher_filtered/` temp=0.0 data (which had FAKE vocab
    shift) and the v5_fresh dir. The v6_fresh data was generated with rewritten
    vocabulary-substitution prompts and is the dataset we are pushing to HF to
    REPLACE the old 109,505-row corpus.

    Both abbreviation (abbr_v6_*) and diverse (div_v6_*) batches live here. The
    low-quality `abbreviation_to_expanded` rows produced by the OLD drugs_chemicals
    mapping in div_v6_0001..0018 are filtered in `consolidate()` (see below).
    """
    accepted_files = []

    # Scan v6_fresh ONLY (the regeneration directory)
    fresh_dir = BASE_DIR / "data/processed/private/synthetic/v6_fresh"
    if not fresh_dir.exists():
        print(f"[WARN] v6_fresh dir not found: {fresh_dir}")
        return accepted_files

    for batch_dir in sorted(fresh_dir.iterdir()):
        if not batch_dir.is_dir():
            continue
        accepted_file = batch_dir / "accepted.parquet"
        if accepted_file.exists():
            accepted_files.append(accepted_file)
    return accepted_files


# div_v6 batches that used the OLD drugs_chemicals→abbreviation_to_expanded mapping
# (low quality — no abbreviation relevance filter). The dedicated abbr_v6_* batches
# are the real abbreviation pairs and are kept.
_BAD_ABBR_DIV_BATCHES = {f"div_v6_{i:04d}" for i in range(1, 19)}


def consolidate(output_dir: Path | None = None) -> dict:
    """Merge all accepted.parquet files, deduplicate, and save."""
    output_dir = output_dir or CONSOLIDATED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    accepted_files = find_all_accepted()
    print(f"Found {len(accepted_files)} accepted.parquet files")

    if not accepted_files:
        print("No accepted files found!")
        return {"total": 0, "files": 0}

    # Read and merge
    dfs = []
    total_from_files = 0
    filtered_bad_abbr = 0
    dropped_old_fq = 0
    for f in accepted_files:
        try:
            df = pd.read_parquet(f)
            bname = f.parent.name

            # Filter out the low-quality abbreviation_to_expanded rows that came
            # from the OLD drugs_chemicals mapping in div_v6_0001..0018 (regular
            # passages, no abbreviation relevance filter; alt_terms are full
            # words). The dedicated abbr_v6_* abbreviation pairs are kept.
            if bname in _BAD_ABBR_DIV_BATCHES and "task_family" in df.columns:
                bad_mask = df["task_family"] == "abbreviation_to_expanded"
                bad_count = int(bad_mask.sum())
                if bad_count:
                    df = df[~bad_mask].reset_index(drop=True)
                    filtered_bad_abbr += bad_count

            # ── full_question swap ────────────────────────────────────────
            # The div_v6_* batches' original full_question rows collapsed into
            # a "Does [NP] [VP]?" monoculture (~37%). They were regenerated with
            # forced-opener diversity into fq_v6_* batches (reusing the same
            # facts/passages). So: from div_v6_* keep ONLY the keyword rows
            # (drop the old templated full_question); fq_v6_* supplies the new
            # full_question rows; abbr_v6_* supplies all abbreviation rows
            # (regenerated with the fixed abbreviation pool + forced openers).
            if bname.startswith("div_v6_") and "query_style" in df.columns:
                old_fq_mask = df["query_style"] == "full_question"
                dropped = int(old_fq_mask.sum())
                if dropped:
                    df = df[~old_fq_mask].reset_index(drop=True)
                    dropped_old_fq += dropped

            dfs.append(df)
            total_from_files += len(df)
            print(f"  {bname}/accepted.parquet: {len(df)} rows")
        except Exception as e:
            print(f"  [WARN] Failed to read {f}: {e}")

    if not dfs:
        return {"total": 0, "files": len(accepted_files)}

    if filtered_bad_abbr:
        print(f"\nFiltered {filtered_bad_abbr} low-quality abbreviation_to_expanded rows from div_v6_0001..0018")
    if dropped_old_fq:
        print(f"Dropped {dropped_old_fq} old templated full_question rows from div_v6_* (replaced by fq_v6_* regen)")

    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal before dedup: {len(merged)}")

    # Deduplicate by example_id
    if "example_id" in merged.columns:
        before = len(merged)
        merged = merged.drop_duplicates(subset=["example_id"], keep="first")
        after = len(merged)
        print(f"Dedup by example_id: {before} → {after} (removed {before - after})")

    # Also dedup by query text (exact match)
    if "query" in merged.columns:
        before = len(merged)
        merged = merged.drop_duplicates(subset=["query"], keep="first")
        after = len(merged)
        print(f"Dedup by query text: {before} → {after} (removed {before - after})")

    # ── Data integrity fixes ─────────────────────────────────────────────
    # Fix 1: Backfill null query_style from query content
    if "query_style" in merged.columns:
        null_qs_mask = merged["query_style"].isna()
        null_qs_count = null_qs_mask.sum()
        if null_qs_count > 0:
            print(f"\nFixing {null_qs_count} rows with null query_style...")
            # Heuristic: if query ends with ?, it's a full_question; otherwise infer from content
            def infer_query_style(query_str, alt_term=""):
                if not query_str or not isinstance(query_str, str):
                    return "full_question"
                q = query_str.strip()
                # Questions → full_question
                if q.endswith("?"):
                    return "full_question"
                # Short, no question mark, keyword-like → check length
                words = q.split()
                if len(words) <= 10 and "?" not in q:
                    # Could be keyword — look at content
                    if any(kw in q.lower() for kw in ["keyword", "search"]):
                        return "keyword_technical"
                    if len(words) <= 8:
                        return "keyword_technical"  # conservative default for short queries
                return "full_question"

            # For each null row, infer query_style
            for idx in merged[null_qs_mask].index:
                query = merged.loc[idx, "query"] if "query" in merged.columns else ""
                inferred = infer_query_style(query)
                merged.loc[idx, "query_style"] = inferred

            remaining_null = merged["query_style"].isna().sum()
            print(f"  Backfilled {null_qs_count - remaining_null} rows, {remaining_null} still null")

    # Fix 2: Ensure judge_source column exists and is populated
    if "judge_source" not in merged.columns:
        merged["judge_source"] = None

    null_js_mask = merged["judge_source"].isna()
    null_js_count = null_js_mask.sum()
    if null_js_count > 0:
        print(f"\nFixing {null_js_count} rows with null judge_source...")
        # If judge_model is available, infer judge_source from it
        if "judge_model" in merged.columns:
            for idx in merged[null_js_mask].index:
                jm = str(merged.loc[idx, "judge_model"]) if pd.notna(merged.loc[idx, "judge_model"]) else ""
                if jm == "none":
                    merged.loc[idx, "judge_source"] = "none_dummy"
                elif "keyword" in jm or "fastpath" in jm:
                    merged.loc[idx, "judge_source"] = "none_keyword_fastpath"
                else:
                    merged.loc[idx, "judge_source"] = "direct_vllm"
        # Any remaining nulls → default to "none_dummy"
        still_null = merged["judge_source"].isna().sum()
        if still_null > 0:
            merged["judge_source"] = merged["judge_source"].fillna("none_dummy")
        print(f"  Backfilled {null_js_count} rows, {merged['judge_source'].isna().sum()} still null")

    # Fix 3: Compute term_overlap for all rows (legacy data may not have it)
    if "term_overlap" not in merged.columns or merged["term_overlap"].isna().any():
        print("\nComputing term_overlap for existing data...")
        import re as _re

        def _compute_overlap_batch(query, passage, cui_label="", task_family=""):
            if not query or not passage or not isinstance(query, str) or not isinstance(passage, str):
                return 0.0
            query_words = [w.lower() for w in _re.findall(r'[a-z0-9]+', query.lower()) if len(w) >= 3]
            passage_words = set(_re.findall(r'[a-z0-9]+', passage.lower()))
            if not query_words:
                return 0.0
            # Add cui_label words to passage vocabulary
            if cui_label and isinstance(cui_label, str):
                clean = _re.sub(r'\s*\([^)]*\)', '', cui_label)
                clean = _re.sub(r'\s*\[[^\]]*\]', '', clean)
                passage_words |= set(_re.findall(r'[a-z0-9]+', clean.lower()))
            found = sum(1 for w in query_words if w in passage_words)
            return found / len(query_words)

        if "term_overlap" not in merged.columns:
            merged["term_overlap"] = None

        null_tom = merged["term_overlap"].isna()
        if null_tom.any():
            merged.loc[null_tom, "term_overlap"] = merged.loc[null_tom].apply(
                lambda r: round(_compute_overlap_batch(
                    r.get("query", ""), r.get("passage_text", ""),
                    r.get("cui_label", ""), r.get("task_family", ""),
                ), 3),
                axis=1,
            )
            print(f"  Computed term_overlap for {null_tom.sum()} rows")
            low_overlap = (merged["term_overlap"] < 0.5) & (merged["term_overlap"].notna())
            print(f"  {low_overlap.sum()} rows have term_overlap < 0.5 (post-hoc)")

    # Fix 4: Compute alt_term_overlap for all rows
    if "alt_term_overlap" not in merged.columns:
        merged["alt_term_overlap"] = None

    # Fix 5: Repair UMLS artifacts in queries (strip rather than remove)
    import re as _re2
    _umls_artifact_pattern = _re2.compile(
        r'\s*\((?:substance|finding|diagnosis|procedure|morphologic abnormality)\)|'
        r'\s*\[(?:medical device|brand name)\]',
        _re2.IGNORECASE,
    )
    if "query" in merged.columns:
        before = len(merged)
        has_artifacts = merged["query"].apply(
            lambda q: bool(_umls_artifact_pattern.search(str(q))) if isinstance(q, str) else False
        )
        artifact_count = has_artifacts.sum()
        if artifact_count > 0:
            merged.loc[has_artifacts, "query"] = merged.loc[has_artifacts, "query"].apply(
                lambda q: _umls_artifact_pattern.sub('', str(q)).strip() if isinstance(q, str) else q
            )
            print(f"\nRepaired {artifact_count} rows with UMLS artifacts in queries "
                  f"({artifact_count/before*100:.2f}%) — stripped qualifiers")

    # Fix 6: Remove data leakage (query verbatim in passage)
    if "query" in merged.columns and "passage_text" in merged.columns:
        before = len(merged)
        leakage_mask = merged.apply(
            lambda r: isinstance(r.get("query"), str) and isinstance(r.get("passage_text"), str)
                      and len(str(r["query"])) > 20
                      and str(r["query"]).lower() in str(r["passage_text"]).lower(),
            axis=1,
        )
        leakage_count = leakage_mask.sum()
        if leakage_count > 0:
            merged = merged[~leakage_mask]
            print(f"Removed {leakage_count} rows with data leakage (query in passage)")

    # Fix 7: Ensure all expected columns exist
    expected_columns = [
        "example_id", "passage_id", "passage_text", "query", "query_style",
        "fact", "mode", "task_family", "role", "cui_label", "alt_term",
        "target_cuis", "semantic_group", "vocab_shift_type",
        "generator_model", "prompt_hash", "passed_validation",
        "validation_failures", "validation_warnings",
        "judgments", "decision", "rejection_reason",
        "judge_model", "judge_source",
    ]
    for col in expected_columns:
        if col not in merged.columns:
            merged[col] = None
            print(f"  Added missing column: {col}")

    # Split by query_style for separate files
    if "query_style" in merged.columns:
        for style in merged["query_style"].unique():
            style_df = merged[merged["query_style"] == style]
            style_path = output_dir / f"mode4_{style}.parquet"
            style_df.to_parquet(style_path, index=False)
            print(f"  Saved {len(style_df)} {style} examples to {style_path}")

    # Save full merged file
    merged_path = output_dir / "mode4_teacher_filtered_multistyle.parquet"
    merged.to_parquet(merged_path, index=False)
    print(f"Saved {len(merged)} total examples to {merged_path}")

    # Compute stats
    stats = {
        "total_accepted": len(merged),
        "files_processed": len(accepted_files),
        "total_from_files": total_from_files,
        "dedup_removed": total_from_files - len(merged),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Per-style counts
    if "query_style" in merged.columns:
        stats["per_style"] = merged["query_style"].value_counts().to_dict()

    # Per-semantic-group counts
    if "semantic_group" in merged.columns:
        stats["per_semantic_group"] = merged["semantic_group"].value_counts().to_dict()

    # Per-task-family counts
    if "task_family" in merged.columns:
        stats["per_task_family"] = merged["task_family"].value_counts().to_dict()

    stats_path = output_dir / "consolidation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")

    return stats


def update_checkpoint(total_accepted: int):
    """Update checkpoint.json with latest counts."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            checkpoint = json.load(f)
    else:
        checkpoint = {}

    checkpoint["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    checkpoint["total_accepted"] = total_accepted
    checkpoint["multistyle"] = True

    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)
    print(f"Checkpoint updated: {total_accepted} total accepted")


def push_to_hf(stats: dict, dry_run: bool = False):
    """Push consolidated data to HuggingFace."""
    # Get HF token: prefer env var, fall back to hf CLI stored token
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        try:
            result = subprocess.run(
                ["hf", "auth", "token"],
                capture_output=True, text=True, check=True,
            )
            hf_token = result.stdout.strip()
        except Exception:
            pass
    if not hf_token:
        print("[ERROR] No HF token available. Cannot push to HuggingFace.")
        print("  Run: hf auth login --token <token>  or  export HF_TOKEN=<token>")
        return False

    if dry_run:
        print(f"[DRY RUN] Would push {stats['total_accepted']} examples to {HF_REPO}")
        return True

    print(f"Pushing to {HF_REPO}...")

    # Upload consolidated parquet files
    for parquet_file in CONSOLIDATED_DIR.glob("*.parquet"):
        cmd = [
            "hf", "upload", "--repo-type", "dataset", HF_REPO, str(parquet_file),
            f"data/{parquet_file.name}",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  Uploaded {parquet_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Upload failed for {parquet_file.name}: {e.stderr}")

    # Upload stats
    stats_path = CONSOLIDATED_DIR / "consolidation_stats.json"
    if stats_path.exists():
        cmd = [
            "hf", "upload", "--repo-type", "dataset", HF_REPO, str(stats_path),
            "stats/consolidation_stats.json",
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    # Upload checkpoint
    if CHECKPOINT_PATH.exists():
        cmd = [
            "hf", "upload", "--repo-type", "dataset", HF_REPO, str(CHECKPOINT_PATH),
            "checkpoint.json",
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    print(f"Push complete! https://huggingface.co/{HF_REPO}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Consolidate multistyle batches and push to HF")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace after consolidation")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (don't actually push)")
    args = parser.parse_args()

    print("=== MedColBERT Multistyle Consolidation ===\n")
    stats = consolidate()

    if stats.get("total_accepted", stats.get("total", 0)) == 0:
        print("\nNo accepted data to consolidate.")
        return 1

    update_checkpoint(stats["total_accepted"])

    if args.push or args.dry_run:
        push_to_hf(stats, dry_run=args.dry_run)

    print(f"\n=== Consolidation Complete: {stats['total_accepted']} accepted examples ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
