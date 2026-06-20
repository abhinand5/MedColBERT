#!/usr/bin/env python
"""Authoritative vocabulary-shift audit via n-gram copy rate.

The proxy metric in audit_quality.py (alt_term presence + term_overlap) is
misleading (88-94/100 trivially). The REAL measure of vocabulary shift is how
much of the query's n-gram sequence is copied verbatim from the source passage.
Low copy rate = high vocabulary shift (the query re-expresses the passage in
different words, which is what ColBERT contrastive training needs).

Usage:
    uv run python scripts/data/ngram_copy_audit.py [--parquet PATH] [--n 3,4]
        [--sample N] [--per-family]

Vocab-shift score (per family) = 100 * (1 - mean_4gram_copy_rate).
Target: >= 60/100 for ALL task families (i.e. mean 4-gram copy rate <= 0.40).
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

DEFAULT_PARQUET = Path(
    "/workspace/MedColBERT/data/processed/private/synthetic/consolidated/"
    "mode4_teacher_filtered_multistyle.parquet"
)


def ngrams(tokens, n):
    return set(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)) if len(tokens) >= n else set()


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) >= 2]


def copy_rate(query, passage, n):
    qt = tokenize(query)
    pt = tokenize(passage)
    if not qt or len(qt) < n:
        return 0.0
    qg = ngrams(qt, n)
    pg = ngrams(pt, n)
    if not qg:
        return 0.0
    return len(qg & pg) / len(qg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--n", type=str, default="3,4")
    ap.add_argument("--sample", type=int, default=0, help="0 = all rows")
    ap.add_argument("--per-family", action="store_true")
    args = ap.parse_args()

    ns = [int(x) for x in args.n.split(",")]
    df = pd.read_parquet(args.parquet)
    print(f"Loaded {len(df)} rows from {args.parquet}")
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"Sampled {len(df)} rows")

    # Global
    for n in ns:
        rates = df.apply(
            lambda r: copy_rate(r.get("query", ""), r.get("passage_text", ""), n), axis=1
        )
        mean = rates.mean()
        score = 100 * (1 - mean)
        print(f"\n[global] {n}-gram copy rate: mean={mean:.4f} median={rates.median():.4f} "
              f"| >0.30 share={ (rates>0.30).mean():.2%} | vocab_shift_score={score:.1f}/100")

    if args.per_family and "task_family" in df.columns:
        fam = df["task_family"].dropna().unique()
        print("\n=== Per task_family (4-gram copy) ===")
        results = []
        for f in sorted(fam):
            sub = df[df["task_family"] == f]
            rates = sub.apply(
                lambda r: copy_rate(r.get("query", ""), r.get("passage_text", ""), 4), axis=1
            )
            mean = rates.mean()
            score = 100 * (1 - mean)
            results.append((f, len(sub), mean, score))
            print(f"  {f:35s} n={len(sub):6d} copy={mean:.4f} score={score:5.1f}/100 "
                  f"{'PASS' if score >= 60 else 'FAIL'}")
        all_pass = all(s >= 60 for _, _, _, s in results)
        print(f"\nALL FAMILIES >= 60/100: {'YES' if all_pass else 'NO'}")


if __name__ == "__main__":
    main()
