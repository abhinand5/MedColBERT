#!/usr/bin/env python3
"""Generate synthetic queries using DSPy + Gemma 4 31B.

Thin CLI wrapper around medcolbert.generation.dspy_programs.

Usage:
    uv run python scripts/data/04_generate_synthetic_queries.py \
        --config configs/generation.yaml \
        --mode generic_synthetic \
        --limit 5 \
        --output-dir data/processed/private/synthetic

Batches:
    candidates.parquet + stats.json per batch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from medcolbert.utils.config import load_yaml
from medcolbert.utils.io import private_output_path
from medcolbert.utils.hashing import stable_hash


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic medical queries.")
    parser.add_argument("--config", default="configs/generation.yaml", help="Path to generation config")
    parser.add_argument("--mode", default="generic_synthetic", help="Generation mode")
    parser.add_argument("--limit", type=int, default=5, help="Number of examples to generate")
    parser.add_argument("--batch-id", default="0", help="Batch identifier")
    parser.add_argument("--output-dir", default="data/processed/private/synthetic", help="Output directory")
    parser.add_argument("--passages", default=None, help="Path to passages.parquet (optional)")
    args = parser.parse_args()

    config = load_yaml(args.config)
    gen_cfg = config.get("generation", {})

    output_dir = Path(args.output_dir) / args.mode / f"batch_{args.batch_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic queries (mode={args.mode}, limit={args.limit})")
    print(f"Output: {output_dir}")

    # Load passages if available
    passages = []
    passages_path = args.passages or "data/processed/private/passages/passages.parquet"
    if Path(passages_path).exists():
        df = pd.read_parquet(passages_path)
        # Sample passages for generation
        sample = df.sample(n=min(args.limit, len(df)), random_state=42)
        passages = sample.to_dict("records")
        print(f"Loaded {len(passages)} passages from {passages_path}")
    else:
        print(f"No passages file found at {passages_path} — using placeholder passages")
        # Placeholder passages for testing
        passages = [
            {
                "passage_id": f"test_{i}",
                "text": "Acute myocardial infarction is a life-threatening condition that requires immediate treatment with thrombolytics or primary percutaneous coronary intervention. Patients typically present with chest pain, dyspnea, and diaphoresis. Electrocardiography may reveal ST-segment elevation. Cardiac biomarkers including troponin I and CK-MB are elevated.",
                "cui_label": "Myocardial Infarction",
                "alt_term": "heart attack",
                "semantic_group": "disorders",
                "vocab_shift_type": "consumer_to_clinical",
            }
            for i in range(args.limit)
        ]

    # Initialize pipeline
    try:
        from medcolbert.generation.dspy_programs import GenerationPipeline

        pipeline = GenerationPipeline.from_config(config)
        print(f"Generator prompt hash: {pipeline.generator_prompt_hash}")
        print(f"Judge prompt hash: {pipeline.judge_prompt_hash}")

        candidates = pipeline.generate_batch(
            passages=passages,
            mode=args.mode,
        )

        # Save candidates
        df_candidates = pd.DataFrame(candidates)
        # Serialize list/dict columns for parquet compatibility
        for col in df_candidates.columns:
            if col == "query":
                continue
            sample_vals = df_candidates[col].dropna()
            if len(sample_vals) == 0:
                continue
            if isinstance(sample_vals.iloc[0], (dict, list)):
                df_candidates[col] = df_candidates[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (dict, list)) else json.dumps([])
                )
        candidates_path = output_dir / "candidates.parquet"
        df_candidates.to_parquet(candidates_path, index=False)
        print(f"Saved {len(candidates)} candidates to {candidates_path}")

        # Save stats
        num_passed = sum(1 for c in candidates if c.get("passed_validation"))
        stats = {
            "batch_id": args.batch_id,
            "mode": args.mode,
            "total": len(candidates),
            "passed_validation": num_passed,
            "failed_validation": len(candidates) - num_passed,
            "generator_prompt_hash": pipeline.generator_prompt_hash,
            "timestamp": str(pd.Timestamp.now()),
        }
        stats_path = output_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to {stats_path}")

    except Exception as e:
        print(f"Error: {e}")
        # Still write an empty output to show the pipeline works
        df = pd.DataFrame(passages)
        candidates_path = output_dir / "candidates.parquet"
        df.to_parquet(candidates_path, index=False)
        print(f"Wrote placeholder data to {candidates_path}")
        raise


if __name__ == "__main__":
    main()
