#!/usr/bin/env python3
"""Consolidate all batch accepted.parquet files and push to HuggingFace.

Reads all accepted.parquet from mode4 batch directories, deduplicates by
example_id, merges with existing consolidated data, and uploads to HF.

Usage:
  uv run python scripts/data/consolidate_and_push.py --push
  uv run python scripts/data/consolidate_and_push.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path("/workspace/MedColBERT")
SYNTHETIC_DIR = BASE_DIR / "data/processed/private/synthetic"
MODE4_DIR = SYNTHETIC_DIR / "ontology_grounded_teacher_filtered"
CONSOLIDATED_DIR = SYNTHETIC_DIR / "consolidated"

HF_REPO = "abhinand/medcolbert-synthetic-pilot"
HF_REPO_TYPE = "dataset"


def count_parquet(path: Path) -> int:
    """Count rows in a parquet file, returning 0 on error or empty."""
    try:
        df = pd.read_parquet(path)
        return len(df)
    except Exception:
        return 0


def collect_all_accepted() -> pd.DataFrame:
    """Collect all accepted.parquet files from batch directories."""
    frames = []
    if not MODE4_DIR.exists():
        return pd.DataFrame()

    for parquet_path in sorted(MODE4_DIR.rglob("accepted.parquet")):
        try:
            df = pd.read_parquet(parquet_path)
            if len(df) > 0:
                frames.append(df)
                print(f"  Read {len(df):4d} rows from {parquet_path.parent.name}/{parquet_path.name}")
        except Exception as e:
            print(f"  ERROR reading {parquet_path}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Combined: {len(combined)} rows from {len(frames)} files")
    return combined


def consolidate() -> dict:
    """Consolidate all accepted examples, deduplicate, and save.

    Returns:
        dict with stats about the consolidation.
    """
    CONSOLIDATED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MedColBERT Dataset Consolidation")
    print("=" * 60)

    # 1. Collect all batch accepted.parquet
    print("\n[1/4] Collecting batch accepted.parquet files...")
    batch_df = collect_all_accepted()

    if len(batch_df) == 0:
        print("  No batch data found!")
        return {"total": 0, "new": 0}

    # 2. Load existing consolidated
    consolidated_path = CONSOLIDATED_DIR / "mode4_teacher_filtered.parquet"
    existing_df = pd.DataFrame()
    if consolidated_path.exists():
        existing_df = pd.read_parquet(consolidated_path)
        print(f"\n[2/4] Existing consolidated: {len(existing_df)} rows")

    # 3. Merge and deduplicate
    print(f"\n[3/4] Merging and deduplicating...")
    if len(existing_df) > 0:
        all_df = pd.concat([existing_df, batch_df], ignore_index=True)
    else:
        all_df = batch_df

    before_dedup = len(all_df)

    # Deduplicate by example_id
    if "example_id" in all_df.columns:
        # Parse JSON-serialized example_ids if needed
        all_df = all_df.drop_duplicates(subset=["example_id"], keep="first")
    else:
        # Fallback: dedup by query + passage_id
        all_df = all_df.drop_duplicates(subset=["query", "passage_id"], keep="first")

    after_dedup = len(all_df)
    dupes_removed = before_dedup - after_dedup
    print(f"  Before dedup: {before_dedup}")
    print(f"  After dedup:  {after_dedup}")
    print(f"  Removed:      {dupes_removed} duplicates")

    # 4. Save consolidated
    print(f"\n[4/4] Saving consolidated dataset...")

    # Ensure list/dict columns are JSON-serialized (CRITICAL FIX #4)
    for col in all_df.columns:
        if all_df[col].dtype == object:
            try:
                has_nested = all_df[col].apply(lambda x: isinstance(x, (list, dict))).any()
                if has_nested:
                    all_df[col] = all_df[col].apply(json.dumps)
            except Exception:
                pass

    all_df.to_parquet(consolidated_path, index=False)
    print(f"  Saved {len(all_df)} rows to {consolidated_path}")

    # Also save a mode3 + mode4 combined total
    mode3_path = CONSOLIDATED_DIR / "mode3_ontology_grounded.parquet"
    total = len(all_df)
    if mode3_path.exists():
        mode3_count = count_parquet(mode3_path)
        total = mode3_count + len(all_df)
        print(f"  Mode 3 (existing): {mode3_count}")
        print(f"  Mode 4 (new):      {len(all_df)}")
        print(f"  Combined total:    {total}")

    # Save stats
    stats = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "mode4_accepted": len(all_df),
        "total_accepted": total,
        "duplicates_removed": dupes_removed,
        "num_batch_files": len(list(MODE4_DIR.rglob("accepted.parquet"))) if MODE4_DIR.exists() else 0,
    }
    with open(CONSOLIDATED_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Final stats: {json.dumps(stats, indent=2)}")
    return stats


def push_to_hf(stats: dict, dry_run: bool = False) -> bool:
    """Push consolidated parquet files to HuggingFace.

    Uses `hf upload` to push the consolidated directory.
    """
    print(f"\n{'=' * 60}")
    print(f"Pushing to HuggingFace: {HF_REPO}")
    print(f"{'=' * 60}")

    if dry_run:
        print("[DRY RUN] Would push the following files:")
        for f in sorted(CONSOLIDATED_DIR.rglob("*.parquet")):
            print(f"  {f}")
        for f in sorted(CONSOLIDATED_DIR.rglob("*.json")):
            print(f"  {f}")
        return True

    # Push consolidated files
    files_to_push = []
    for f in sorted(CONSOLIDATED_DIR.rglob("*")):
        if f.is_file() and f.suffix in (".parquet", ".json"):
            files_to_push.append(f)

    for f in files_to_push:
        rel_path = f"consolidated/{f.name}"
        print(f"  Uploading {f} -> {rel_path}...")
        cmd = [
            "hf", "upload",
            HF_REPO,
            str(f),
            rel_path,
            "--type", HF_REPO_TYPE,
            "--commit-message", f"consolidate: {stats['mode4_accepted']} mode4 accepted (total: {stats['total_accepted']})",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                print(f"    ✅ {rel_path}")
            else:
                print(f"    ❌ {rel_path}: {result.stderr[-200:]}")
        except Exception as e:
            print(f"    ❌ {rel_path}: {e}")

    print(f"\n  Pushed {len(files_to_push)} files to {HF_REPO}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Consolidate and push MedColBERT dataset")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace after consolidation")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be pushed without pushing")
    args = parser.parse_args()

    # Consolidate
    stats = consolidate()

    if stats["total"] == 0:
        print("\nNo data to push.")
        return 1

    # Push if requested
    if args.push or args.dry_run:
        push_to_hf(stats, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
