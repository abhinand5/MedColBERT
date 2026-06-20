#!/usr/bin/env python3
"""MedColBERT Production Orchestrator — auto-launch batches until 100K accepted.

Monitors total accepted count across all batch directories, launches concurrent
batches in waves, and stops when target is reached.

Usage:
  uv run python scripts/data/orchestrator.py --target 100000 --workers-per-batch 16 --concurrent-batches 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path("/workspace/MedColBERT")
SYNTHETIC_DIR = BASE_DIR / "data/processed/private/synthetic"
MODE4_DIR = SYNTHETIC_DIR / "ontology_grounded_teacher_filtered"
CURRENT_TOTAL = 2301  # From checkpoint: 1099 mode3 + 1202 mode4

BATCH_SCRIPT = BASE_DIR / "scripts/data/concurrent_batch.py"


def count_all_accepted() -> int:
    """Count total accepted examples across all batch directories."""
    import pandas as pd

    total = 0
    if not MODE4_DIR.exists():
        return total

    for accepted_file in MODE4_DIR.rglob("accepted.parquet"):
        # Skip old batch_1 through batch_4 (already counted in consolidated)
        parent = accepted_file.parent.name
        try:
            df = pd.read_parquet(accepted_file)
            total += len(df)
        except Exception:
            pass
    return total


def get_next_seed() -> int:
    """Get the next available seed by checking existing batch directories."""
    existing = []
    if MODE4_DIR.exists():
        for d in MODE4_DIR.iterdir():
            if d.is_dir() and d.name.startswith("conc_"):
                existing.append(d.name)
    # Use timestamp-based seed to avoid collisions
    return int(time.time() * 1000) % 100000 + 4000


def run_batch(seed: int, batch_id: str, count: int = 1008, workers: int = 16) -> bool:
    """Run a single concurrent batch. Returns True if successful."""
    cmd = [
        "uv", "run", "python", str(BATCH_SCRIPT),
        "--seed", str(seed),
        "--batch-id", batch_id,
        "--count", str(count),
        "--workers", str(workers),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        print(result.stdout)
        if result.returncode != 0:
            print(f"[ORCH] Batch {batch_id} failed (exit {result.returncode}):", file=sys.stderr)
            print(result.stderr[-500:], file=sys.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[ORCH] Batch {batch_id} timed out after 1 hour", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[ORCH] Batch {batch_id} exception: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="MedColBERT Production Orchestrator")
    parser.add_argument("--target", type=int, default=100000, help="Target accepted count")
    parser.add_argument("--workers-per-batch", type=int, default=16,
                        help="Max concurrent workers per batch")
    parser.add_argument("--concurrent-batches", type=int, default=3,
                        help="Number of batches to run in parallel per wave")
    parser.add_argument("--passages-per-batch", type=int, default=1008,
                        help="Number of passages per batch")
    parser.add_argument("--max-waves", type=int, default=500,
                        help="Maximum number of waves to run")
    args = parser.parse_args()

    print("=" * 70)
    print("MedColBERT Production Orchestrator")
    print(f"Target: {args.target:,} accepted examples")
    print(f"Concurrent batches per wave: {args.concurrent_batches}")
    print(f"Passages per batch: {args.passages_per_batch}")
    print(f"Workers per batch: {args.workers_per_batch}")
    print("=" * 70)

    wave = 0
    consecutive_failures = 0

    while wave < args.max_waves:
        wave += 1

        # Count current total
        batch_accepted = count_all_accepted()
        total_accepted = CURRENT_TOTAL + batch_accepted
        remaining = args.target - total_accepted

        print(f"\n{'=' * 70}")
        print(f"WAVE {wave} | Total accepted: {total_accepted:,}/{args.target:,} "
              f"({total_accepted/args.target*100:.1f}%) | Remaining: {remaining:,}")
        print(f"{'=' * 70}")

        if total_accepted >= args.target:
            print(f"\n🎉 TARGET REACHED: {total_accepted:,} accepted examples!")
            break

        # Estimate batches needed this wave
        avg_acceptance = 0.65  # Conservative estimate
        expected_per_batch = int(args.passages_per_batch * avg_acceptance)
        print(f"Expected ~{expected_per_batch} accepted per batch (at {avg_acceptance:.0%} acceptance)")

        # Launch concurrent batches
        import concurrent.futures

        seeds = [get_next_seed() + i for i in range(args.concurrent_batches)]
        batch_ids = [f"conc_w{wave:03d}_{i+1}" for i in range(args.concurrent_batches)]

        print(f"Launching {args.concurrent_batches} batches: {batch_ids}")
        print(f"Seeds: {seeds}")

        t_wave_start = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent_batches) as executor:
            futures = {
                executor.submit(
                    run_batch, seed, batch_id, args.passages_per_batch, args.workers_per_batch
                ): batch_id
                for seed, batch_id in zip(seeds, batch_ids)
            }
            for future in concurrent.futures.as_completed(futures):
                batch_id = futures[future]
                try:
                    success = future.result()
                    status = "✅" if success else "❌"
                    print(f"[ORCH] {batch_id}: {status}")
                    if not success:
                        consecutive_failures += 1
                    else:
                        consecutive_failures = 0
                except Exception as e:
                    print(f"[ORCH] {batch_id}: ❌ exception: {e}")
                    consecutive_failures += 1

        wave_elapsed = time.time() - t_wave_start
        print(f"[ORCH] Wave {wave} completed in {wave_elapsed/60:.1f} minutes")

        # If too many consecutive failures, stop
        if consecutive_failures >= 3:
            print(f"[ORCH] {consecutive_failures} consecutive failures — stopping")
            break

    # Final count
    final_accepted = count_all_accepted()
    final_total = CURRENT_TOTAL + final_accepted
    print(f"\n{'=' * 70}")
    print(f"FINAL: {final_total:,} total accepted examples")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
