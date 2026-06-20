#!/usr/bin/env python3
"""Judge synthetic query candidates across all 5 quality dimensions.

Thin CLI wrapper — reads candidates.parquet, runs judge pipeline,
writes judged candidates with accept/reject decisions.

Usage:
    uv run python scripts/data/04_judge_candidates.py \
        --candidates data/processed/private/synthetic/mode/batch_N/candidates.parquet \
        --output data/processed/private/synthetic/mode/batch_N/judged.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from medcolbert.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser(description="Judge synthetic query candidates.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.parquet")
    parser.add_argument("--output", required=True, help="Path for judged output parquet")
    parser.add_argument("--config", default="configs/generation.yaml", help="Generation config")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"Error: candidates file not found: {candidates_path}")
        return

    df = pd.read_parquet(candidates_path)
    candidates = df.to_dict("records")
    print(f"Loaded {len(candidates)} candidates from {candidates_path}")

    config = load_yaml(args.config)

    try:
        from medcolbert.generation.dspy_programs import GenerationPipeline
        from medcolbert.generation.teacher_filter import compute_batch_stats

        pipeline = GenerationPipeline.from_config(config)

        judged = pipeline.judge_candidates(candidates)

        accepted = [c for c in judged if c.get("decision") == "ACCEPT"]
        rejected = [c for c in judged if c.get("decision") == "REJECT"]

        print(f"Accepted: {len(accepted)}, Rejected: {len(rejected)}")

        # Save all judged
        df_judged = pd.DataFrame(judged)
        # Serialize dict columns to JSON strings for parquet compatibility
        for col in df_judged.columns:
            if df_judged[col].apply(lambda x: isinstance(x, dict)).any():
                df_judged[col] = df_judged[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_judged.to_parquet(output_path, index=False)

        # Save accepted and rejected separately
        if accepted:
            pd.DataFrame(accepted).to_parquet(
                output_path.parent / "accepted.parquet", index=False
            )
        if rejected:
            pd.DataFrame(rejected).to_parquet(
                output_path.parent / "rejected.parquet", index=False
            )

        # Save stats
        stats = compute_batch_stats(accepted, rejected)
        stats_path = output_path.parent / "judge_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"Results saved to {output_path}")
        for dim, rate in stats.get("per_dimension_pass_rates", {}).items():
            print(f"  {dim}: {rate:.2%}")

    except Exception as e:
        print(f"Error during judging: {e}")
        raise


if __name__ == "__main__":
    main()
