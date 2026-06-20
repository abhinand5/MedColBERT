#!/usr/bin/env python3
"""Create a stratified audit sample for Claude (human) review.

Reads judged candidates, creates a stratified sample of 30 examples:
- 10 random ACCEPT (by judge)
- 10 random REJECT (by judge)
- 5 from the most marginal quality dimension
- 5 from the lowest-performing task family

Usage:
    uv run python scripts/data/04_audit_sample.py \
        --judged data/processed/private/synthetic/mode/batch_N/judged.parquet \
        --output data/processed/private/synthetic/mode/batch_N/audit_sample.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def find_marginal_examples(
    df: pd.DataFrame,
    decisions: dict[str, str],
    n: int = 5,
) -> pd.DataFrame:
    """Find the N most marginal examples based on judge answers.

    An example is "marginal" if it has borderline judge answers
    (e.g., MODERATE_SHIFT vs STRONG_SHIFT, or barely passed).
    """
    marginal_rows: list[int] = []

    for idx, row in df.iterrows():
        judgments = row.get("judgments", {})
        if isinstance(judgments, str):
            try:
                judgments = json.loads(judgments)
            except json.JSONDecodeError:
                judgments = {}

        # Marginal vocab shift (MODERATE, close to MINIMAL)
        if judgments.get("vocab_shift") == "MODERATE_SHIFT":
            marginal_rows.append(idx)

    # Return up to n most marginal
    if len(marginal_rows) > n:
        marginal_rows = marginal_rows[:n]

    if marginal_rows:
        return df.iloc[marginal_rows]
    return pd.DataFrame()


def find_lowest_family(
    df: pd.DataFrame,
    n: int = 5,
) -> tuple[pd.DataFrame, str]:
    """Find examples from the lowest-performing task family."""
    if "task_family" not in df.columns:
        return pd.DataFrame(), ""

    family_acceptance: dict[str, float] = {}
    for family in df["task_family"].unique():
        family_df = df[df["task_family"] == family]
        accepted = (family_df["decision"] == "ACCEPT").sum()
        total = len(family_df)
        family_acceptance[family] = accepted / total if total > 0 else 0.0

    if not family_acceptance:
        return pd.DataFrame(), ""

    worst_family = min(family_acceptance, key=lambda k: family_acceptance[k])
    worst_df = df[df["task_family"] == worst_family].head(n)
    return worst_df, worst_family


def main():
    parser = argparse.ArgumentParser(description="Create stratified audit sample.")
    parser.add_argument("--judged", required=True, help="Path to judged.parquet")
    parser.add_argument("--output", required=True, help="Path for audit_sample.parquet")
    parser.add_argument("--sample-size", type=int, default=30, help="Total sample size")
    args = parser.parse_args()

    judged_path = Path(args.judged)
    if not judged_path.exists():
        print(f"Error: judged file not found: {judged_path}")
        return

    df = pd.read_parquet(judged_path)
    print(f"Loaded {len(df)} judged examples")

    if "decision" not in df.columns:
        print("Warning: no 'decision' column — all examples unjudged")
        df["decision"] = "UNJUDGED"

    accepted = df[df["decision"] == "ACCEPT"]
    rejected = df[df["decision"] == "REJECT"]

    # 10 random accepted
    n_accept = min(10, len(accepted))
    accept_sample = accepted.sample(n=n_accept, random_state=42) if n_accept > 0 else pd.DataFrame()

    # 10 random rejected
    n_reject = min(10, len(rejected))
    reject_sample = rejected.sample(n=n_reject, random_state=42) if n_reject > 0 else pd.DataFrame()

    # 5 from most marginal dimension
    marginal_sample = find_marginal_examples(df, {}, n=5)

    # 5 from lowest-performing task family
    family_sample, worst_family = find_lowest_family(df, n=5)

    # Combine
    samples = [accept_sample, reject_sample, marginal_sample, family_sample]
    audit_df = pd.concat([s for s in samples if len(s) > 0], ignore_index=True)
    audit_df = audit_df.drop_duplicates(subset=["example_id"] if "example_id" in audit_df.columns else audit_df.columns)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_parquet(output_path, index=False)

    print(f"Audit sample: {len(audit_df)} examples → {output_path}")
    if worst_family:
        print(f"Worst-performing task family: {worst_family}")


if __name__ == "__main__":
    main()
