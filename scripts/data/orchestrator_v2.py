#!/usr/bin/env python3
"""Continuous multistyle batch orchestrator — maximizes vLLM throughput.

Runs batches SEQUENTIALLY (not parallel) to avoid vLLM contention.
Each batch: 100 passages → 4 queries each → judge (hybrid) → save.
Tracks total accepted and stops when target is reached.

Key design: vLLM is serial — parallel batches compete and slow each other down.
This orchestrator runs one batch at a time for maximum GPU utilization.

Usage:
  uv run python scripts/data/orchestrator_v2.py --target 100000
  uv run python scripts/data/orchestrator_v2.py --target 50000 --start-seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path("/workspace/MedColBERT")
CHECKPOINT_PATH = BASE_DIR / "data/processed/private/synthetic/checkpoint.json"
PASSAGES_PATH = BASE_DIR / "data/processed/private/passages/real_annotated_500.json"
OUTPUT_BASE = BASE_DIR / "data/processed/private/synthetic/ontology_grounded_teacher_filtered"


def count_existing_accepted() -> int:
    """Count unique deduplicated accepted examples from the consolidated merged file.

    Only counts the main merged parquet to avoid double-counting per-style subsets.
    """
    # Count from the deduplicated merged file (avoids double-counting)
    consolidated = BASE_DIR / "data/processed/private/synthetic/consolidated"
    merged_file = consolidated / "mode4_teacher_filtered_multistyle.parquet"
    if merged_file.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(merged_file)
            return len(df)
        except Exception:
            pass

    # Also count mode 3 legacy
    mode3_file = consolidated / "mode3_ontology_grounded.parquet"
    mode3_count = 0
    if mode3_file.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(mode3_file)
            mode3_count = len(df)
        except Exception:
            pass

    # Fall back: count unique from individual batch stats
    total = mode3_count
    for batch_dir in sorted(OUTPUT_BASE.iterdir()):
        if not batch_dir.is_dir():
            continue
        stats_file = batch_dir / "stats.json"
        if stats_file.exists():
            try:
                with open(stats_file) as f:
                    stats = json.load(f)
                total += stats.get("accepted", 0)
            except Exception:
                pass
    return total


def update_checkpoint(total: int, current_batch: str = ""):
    """Update checkpoint with progress."""
    checkpoint = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_accepted": total,
        "current_batch": current_batch,
        "mode": "multistyle_v2_orchestrated",
    }
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, indent=2)


def run_orchestrator(target: int = 100000, start_seed: int = 1000, batch_count: int = 100,
                     passages_path: Path | None = None, batch_prefix: str = "cont_w2"):
    """Run batches sequentially until target reached."""
    import subprocess

    # Count existing
    existing = count_existing_accepted()
    print(f"Existing accepted examples: {existing:,}")
    print(f"Target: {target:,}")
    print(f"Need: {max(0, target - existing):,}")
    if passages_path:
        print(f"Passages: {passages_path}")
    print()

    if existing >= target:
        print("Target already reached!")
        return existing

    total_accepted = existing
    batch_num = 0
    seed = start_seed
    start_time = time.time()

    while total_accepted < target:
        batch_num += 1
        batch_id = f"{batch_prefix}_{batch_num:04d}"
        seed += 1

        print(f"\n{'='*60}")
        print(f"Batch {batch_num}: {batch_id} (seed={seed})")
        print(f"Progress: {total_accepted:,}/{target:,} ({total_accepted/target*100:.1f}%)")
        elapsed = time.time() - start_time
        if total_accepted > existing:
            rate = (total_accepted - existing) / elapsed * 3600
            remaining = (target - total_accepted) / rate if rate > 0 else float('inf')
            print(f"Rate: {rate:.0f}/hr | ETA: {remaining:.1f} hrs")
        print(f"{'='*60}")

        # Run batch
        cmd = [
            "uv", "run", "python",
            str(BASE_DIR / "scripts/data/multistyle_batch_v2.py"),
            "--seed", str(seed),
            "--batch-id", batch_id,
            "--count", str(batch_count),
            "--judge-mode", "none",
        ]
        if passages_path:
            cmd.extend(["--passages", str(passages_path)])

        try:
            result = subprocess.run(cmd, capture_output=False, text=True, timeout=7200)
            if result.returncode != 0:
                print(f"[ERROR] Batch {batch_id} failed with exit code {result.returncode}")
                # Try with a new seed
                seed += 10
                continue
        except subprocess.TimeoutExpired:
            print(f"[ERROR] Batch {batch_id} timed out after 2 hours")
            continue
        except KeyboardInterrupt:
            print(f"\n[INTERRUPTED] Stopping after batch {batch_num}")
            break

        # Read stats
        stats_file = OUTPUT_BASE / batch_id / "stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
            batch_accepted = stats.get("accepted", 0)
            total_accepted += batch_accepted
            update_checkpoint(total_accepted, batch_id)
            print(f"[{batch_id}] +{batch_accepted} accepted → total={total_accepted:,}")
        else:
            print(f"[WARN] No stats.json found for {batch_id}")

        # Every 10 batches, consolidate and push
        if batch_num % 10 == 0:
            print(f"\n[ORCHESTRATOR] Running consolidation (batch {batch_num})...")
            subprocess.run([
                "uv", "run", "python",
                str(BASE_DIR / "scripts/data/consolidate_multistyle.py"),
            ], capture_output=False, timeout=300)
            update_checkpoint(total_accepted, batch_id)

    # Final consolidation
    print(f"\n{'='*60}")
    print(f"ORCHESTRATOR COMPLETE")
    elapsed = time.time() - start_time
    print(f"Total accepted: {total_accepted:,}")
    print(f"Elapsed: {elapsed/3600:.1f} hrs")
    print(f"Rate: {(total_accepted - existing) / elapsed * 3600:.0f}/hr")
    print(f"{'='*60}")

    # Final consolidation and push
    print("\nRunning final consolidation...")
    subprocess.run([
        "uv", "run", "python",
        str(BASE_DIR / "scripts/data/consolidate_multistyle.py"),
        "--push",
    ], capture_output=False, timeout=600)

    return total_accepted


def main():
    parser = argparse.ArgumentParser(description="Continuous multistyle batch orchestrator")
    parser.add_argument("--target", type=int, default=100000,
                        help="Target number of accepted examples")
    parser.add_argument("--start-seed", type=int, default=1000,
                        help="Starting random seed")
    parser.add_argument("--batch-count", type=int, default=100,
                        help="Passages per batch")
    parser.add_argument("--passages", type=Path, default=None,
                        help="Custom passages JSON file (default: real_annotated_500.json)")
    parser.add_argument("--batch-prefix", type=str, default="cont_w2",
                        help="Batch ID prefix (default: cont_w2)")
    args = parser.parse_args()

    print("=== MedColBERT Continuous Orchestrator v2 ===")
    print(f"Target: {args.target:,} accepted examples")
    print(f"Batch size: {args.batch_count} passages")
    print(f"Start seed: {args.start_seed}")
    if args.passages:
        print(f"Passages: {args.passages}")
    print(f"Batch prefix: {args.batch_prefix}")
    print()

    try:
        total = run_orchestrator(
            target=args.target,
            start_seed=args.start_seed,
            batch_count=args.batch_count,
            passages_path=args.passages,
            batch_prefix=args.batch_prefix,
        )
        print(f"\nFinal: {total:,} accepted examples")
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Orchestrator stopped.")
        update_checkpoint(count_existing_accepted(), "interrupted")


if __name__ == "__main__":
    sys.exit(main())
