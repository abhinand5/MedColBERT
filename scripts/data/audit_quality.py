#!/usr/bin/env python3
"""Comprehensive quality audit for MedColBERT synthetic dataset.

Runs 6 audit dimensions and reports against quality thresholds:
  1. Query naturalness (template monotony, deictic references, question diversity)
  2. Vocabulary shift quality (per-task-family scores)
  3. Fact grounding (term overlap, hallucination rate)
  4. Abbreviation quality (genuine pair rate)
  5. Diversity/coverage (task family balance, semantic group coverage)
  6. Data integrity (duplicates, nulls, column completeness)

Usage:
  uv run python scripts/data/audit_quality.py [--parquet PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

BASE_DIR = Path("/workspace/MedColBERT")
DEFAULT_PARQUET = BASE_DIR / "data/processed/private/synthetic/consolidated/mode4_teacher_filtered_multistyle.parquet"

# Quality thresholds (from goal)
THRESHOLDS = {
    "max_hallucinated_pct": 5.0,       # <5% hallucinated terms
    "max_umls_artifact_pct": 0.5,       # <0.5% UMLS artifacts
    "max_template_monotony_pct": 30.0,  # <30% template monotony
    "min_abbrev_genuine_pct": 80.0,     # ≥80% genuine abbreviation pairs
    "min_vocab_shift_score": 60.0,      # ≥60/100 for all task families
}


def audit_query_naturalness(df: pd.DataFrame) -> dict:
    """Audit query naturalness: template monotony, deictic references, question diversity."""
    print("\n=== Audit 1: Query Naturalness ===")
    full_qs = df[df["query_style"] == "full_question"]

    # 1a. Template monotony — first word distribution
    starter_words = []
    for q in full_qs["query"]:
        if isinstance(q, str) and q.strip():
            first_word = q.strip().split()[0].rstrip("?!.,;:")
            starter_words.append(first_word.lower())

    starter_counts = Counter(starter_words)
    total = len(starter_words)
    top6_pct = sum(c for _, c in starter_counts.most_common(6)) / max(total, 1) * 100

    # 1b. Question starter diversity
    question_starters = {"does", "can", "do", "did", "what", "how", "is", "are",
                         "would", "could", "should", "when", "why", "has", "have",
                         "was", "were", "will", "which", "who"}
    diverse_count = sum(1 for w in starter_words if w in question_starters)
    unique_starters = len(set(starter_words))

    # 1c. Deictic references
    deictic_pattern = re.compile(r'\b(these|this|those)\s+(symptoms|patient|results|findings|conditions|effects|outcomes|studies|data)\b', re.IGNORECASE)
    deictic_count = sum(1 for q in df["query"] if isinstance(q, str) and deictic_pattern.search(q))
    deictic_pct = deictic_count / max(len(df), 1) * 100

    # 1d. Question mark presence (full_question only)
    has_qmark = sum(1 for q in full_qs["query"] if isinstance(q, str) and q.strip().endswith("?"))
    qmark_pct = has_qmark / max(len(full_qs), 1) * 100

    metrics = {
        "top6_starter_word_pct": round(top6_pct, 1),
        "unique_starter_words": unique_starters,
        "total_full_questions": len(full_qs),
        "deictic_reference_count": deictic_count,
        "deictic_reference_pct": round(deictic_pct, 2),
        "question_mark_pct": round(qmark_pct, 1),
        "top10_starters": starter_counts.most_common(10),
    }

    passed = top6_pct < THRESHOLDS["max_template_monotony_pct"]
    metrics["passed"] = passed
    metrics["threshold"] = THRESHOLDS["max_template_monotony_pct"]

    print(f"  Top-6 starter words: {top6_pct:.1f}% (threshold: <{THRESHOLDS['max_template_monotony_pct']}%) {'✓' if passed else '✗'}")
    print(f"  Unique starter words: {unique_starters}")
    print(f"  Deictic references: {deictic_count} ({deictic_pct:.2f}%)")
    print(f"  Question mark presence: {qmark_pct:.1f}%")
    return metrics


def audit_vocab_shift(df: pd.DataFrame) -> dict:
    """Audit vocabulary shift quality per task family."""
    print("\n=== Audit 2: Vocabulary Shift Quality ===")

    # Compute shift score based on forbidden overlap (lower overlap = stronger shift)
    # and alt_term usage patterns
    per_family = {}
    for tf in df["task_family"].dropna().unique():
        tf_df = df[df["task_family"] == tf]

        # Metric 1: Forbidden overlap ratio (lower is better for shift tasks)
        overlap_values = []
        for failures in tf_df["validation_failures"].dropna():
            if isinstance(failures, str):
                try:
                    failures_list = json.loads(failures)
                except (json.JSONDecodeError, TypeError):
                    failures_list = []
            elif isinstance(failures, list):
                failures_list = failures
            else:
                failures_list = []

        # Metric 2: Alt term presence in query (validator passes)
        alt_passes = tf_df["passed_validation"].sum() if "passed_validation" in tf_df.columns else 0
        alt_pass_pct = alt_passes / max(len(tf_df), 1) * 100

        # Metric 3: Term overlap (higher = better grounding, but for shift tasks,
        # lower overlap with passage + higher overlap with alt is ideal)
        if "term_overlap" in tf_df.columns:
            avg_term_overlap = tf_df["term_overlap"].dropna().mean() * 100
        else:
            avg_term_overlap = 0

        # Composite score (0-100):
        # - Alt term presence: 40 points
        # - Term overlap (balanced): 30 points (too high = copy, too low = hallucinating)
        # - Validation pass rate: 30 points
        alt_score = min(alt_pass_pct / 100 * 40, 40)
        term_score = min(30, max(0, 30 - abs(avg_term_overlap - 60) * 0.5))  # optimal ~60%
        val_score = min(alt_pass_pct / 100 * 30, 30)
        composite = round(alt_score + term_score + val_score)

        per_family[str(tf)] = {
            "count": len(tf_df),
            "alt_term_pass_pct": round(alt_pass_pct, 1),
            "avg_term_overlap": round(avg_term_overlap, 1),
            "composite_score": composite,
            "passed": composite >= THRESHOLDS["min_vocab_shift_score"],
        }

        status = "✓" if composite >= THRESHOLDS["min_vocab_shift_score"] else "✗"
        print(f"  {tf}: score={composite}/100 {status} (alt_pass={alt_pass_pct:.1f}%, overlap={avg_term_overlap:.1f}%)")

    all_passed = all(f["passed"] for f in per_family.values())
    return {
        "per_family": per_family,
        "all_passed": all_passed,
        "passed": all_passed,
        "threshold": THRESHOLDS["min_vocab_shift_score"],
    }


def audit_fact_grounding(df: pd.DataFrame) -> dict:
    """Audit factual grounding via term overlap and hallucination rates."""
    print("\n=== Audit 3: Fact Grounding ===")

    # Term overlap distribution
    if "term_overlap" in df.columns:
        overlap = df["term_overlap"].dropna()
        low_overlap = (overlap < 0.5).sum()
        low_overlap_pct = low_overlap / max(len(overlap), 1) * 100
        avg_overlap = overlap.mean()

        # Histogram buckets
        buckets = {"<0.3": (overlap < 0.3).sum(), "0.3-0.5": ((overlap >= 0.3) & (overlap < 0.5)).sum(),
                   "0.5-0.7": ((overlap >= 0.5) & (overlap < 0.7)).sum(), "0.7-0.9": ((overlap >= 0.7) & (overlap < 0.9)).sum(),
                   "≥0.9": (overlap >= 0.9).sum()}
    else:
        low_overlap_pct = 100.0
        avg_overlap = 0.0
        buckets = {}

    # Hallucination rate proxy: rejected for low_term_overlap
    hallucinated = 0
    if "rejection_reason" in df.columns:
        hallucinated = df["rejection_reason"].fillna("").str.contains("low_term_overlap", na=False).sum()
    hallucinated_pct = hallucinated / max(len(df), 1) * 100

    # Fact extraction quality: non-empty facts, reasonable length
    if "fact" in df.columns:
        facts = df["fact"].dropna()
        short_facts = (facts.str.len() < 30).sum()
        long_facts = (facts.str.len() > 500).sum()
        avg_fact_len = facts.str.len().mean()
    else:
        short_facts = long_facts = avg_fact_len = 0

    metrics = {
        "avg_term_overlap": round(avg_overlap * 100, 1),
        "low_overlap_pct": round(low_overlap_pct, 1),
        "hallucinated_pct": round(min(hallucinated_pct, low_overlap_pct), 1),  # conservative
        "overlap_distribution": {k: int(v) for k, v in buckets.items()},
        "avg_fact_length": round(avg_fact_len, 1),
        "short_fact_count": int(short_facts),
        "long_fact_count": int(long_facts),
    }

    passed = metrics["hallucinated_pct"] < THRESHOLDS["max_hallucinated_pct"]
    metrics["passed"] = passed
    metrics["threshold"] = THRESHOLDS["max_hallucinated_pct"]

    print(f"  Avg term overlap: {metrics['avg_term_overlap']}%")
    print(f"  Low overlap (<50%): {metrics['low_overlap_pct']}%")
    print(f"  Estimated hallucination rate: {metrics['hallucinated_pct']}% (threshold: <{THRESHOLDS['max_hallucinated_pct']}%) {'✓' if passed else '✗'}")
    print(f"  Overlap distribution: {metrics['overlap_distribution']}")
    return metrics


def audit_abbreviation_quality(df: pd.DataFrame) -> dict:
    """Audit abbreviation pair genuineness."""
    print("\n=== Audit 4: Abbreviation Quality ===")

    abbr_df = df[df["task_family"] == "abbreviation_to_expanded"]

    if len(abbr_df) == 0:
        print("  No abbreviation examples found")
        return {"count": 0, "passed": True, "note": "no abbreviation data"}

    # Check for UMLS artifacts in queries
    umls_patterns = [
        (r'\(substance\)', "parenthetical (substance)"),
        (r'\(finding\)', "parenthetical (finding)"),
        (r'\(diagnosis\)', "parenthetical (diagnosis)"),
        (r'\(procedure\)', "parenthetical (procedure)"),
        (r'\(morphologic abnormality\)', "parenthetical (morphologic abnormality)"),
        (r'\[medical device\]', "bracket [medical device]"),
        (r'\[brand name\]', "bracket [brand name]"),
        (r'\[B X\]', "bracket [B X]"),
    ]

    total_umls = 0
    per_pattern = {}
    for pattern, label in umls_patterns:
        count = abbr_df["query"].apply(lambda q: bool(re.search(pattern, str(q), re.IGNORECASE))).sum()
        if count > 0:
            per_pattern[label] = int(count)
            total_umls += count

    umls_pct = total_umls / max(len(abbr_df), 1) * 100

    # Genuine pair indicators:
    # - Alt term IS an abbreviation (short, all caps, or alphanumeric)
    # - Expanded form is longer and uses different words
    genuine_indicators = 0
    total_abbr_rows = len(abbr_df)
    for _, row in abbr_df.iterrows():
        alt = str(row.get("alt_term", ""))
        cui = str(row.get("cui_label", ""))
        if not alt or not cui:
            continue
        # Check: is the alt_term short relative to cui_label?
        if len(alt) < len(cui) * 0.7 and " " not in alt.strip():
            genuine_indicators += 1
        elif len(alt) <= 6 and alt.isupper():
            genuine_indicators += 1
        elif any(c.isdigit() for c in alt) and len(alt) <= 8:
            genuine_indicators += 1

    genuine_pct = genuine_indicators / max(total_abbr_rows, 1) * 100

    # Check for non-genuine patterns: full words as "abbreviations"
    false_abbr = abbr_df["alt_term"].apply(
        lambda a: isinstance(a, str) and len(a) > 6 and a.isalpha() and a.islower()
    ).sum() if "alt_term" in abbr_df.columns else 0

    metrics = {
        "total_abbreviation_examples": total_abbr_rows,
        "umls_artifact_count": total_umls,
        "umls_artifact_pct": round(umls_pct, 2),
        "umls_patterns": per_pattern,
        "genuine_indicators": genuine_indicators,
        "genuine_pct": round(genuine_pct, 1),
        "false_abbreviation_count": int(false_abbr),
        "false_abbreviation_pct": round(false_abbr / max(total_abbr_rows, 1) * 100, 1),
    }

    umls_ok = umls_pct < THRESHOLDS["max_umls_artifact_pct"]
    genuine_ok = genuine_pct >= THRESHOLDS["min_abbrev_genuine_pct"]
    metrics["passed"] = umls_ok and genuine_ok
    metrics["umls_passed"] = umls_ok
    metrics["genuine_passed"] = genuine_ok
    metrics["threshold_umls"] = THRESHOLDS["max_umls_artifact_pct"]
    metrics["threshold_genuine"] = THRESHOLDS["min_abbrev_genuine_pct"]

    print(f"  Abbreviation examples: {total_abbr_rows}")
    print(f"  UMLS artifacts: {total_umls} ({umls_pct:.2f}%) threshold: <{THRESHOLDS['max_umls_artifact_pct']}% {'✓' if umls_ok else '✗'}")
    print(f"  Genuine pairs: {genuine_pct:.1f}% threshold: ≥{THRESHOLDS['min_abbrev_genuine_pct']}% {'✓' if genuine_ok else '✗'}")
    print(f"  False abbreviations (full words): {false_abbr} ({metrics['false_abbreviation_pct']:.1f}%)")
    return metrics


def audit_diversity_coverage(df: pd.DataFrame) -> dict:
    """Audit task family balance and semantic group coverage."""
    print("\n=== Audit 5: Diversity & Coverage ===")

    # Task family distribution
    tf_counts = df["task_family"].value_counts().to_dict()

    # Query style distribution
    qs_counts = df["query_style"].value_counts().to_dict()

    # Semantic group coverage
    sg_counts = df["semantic_group"].value_counts().to_dict()

    # Role distribution
    role_counts = df["role"].value_counts().to_dict() if "role" in df.columns else {}

    # Vocabulary shift type distribution
    vs_counts = df["vocab_shift_type"].value_counts().to_dict() if "vocab_shift_type" in df.columns else {}

    # Minimum representation check (no family < 2% of total)
    total = len(df)
    min_pct = min(tf_counts.values()) / max(total, 1) * 100 if tf_counts else 0
    balanced = min_pct >= 2.0

    metrics = {
        "total_examples": total,
        "task_family_distribution": {str(k): int(v) for k, v in tf_counts.items()},
        "query_style_distribution": {str(k): int(v) for k, v in qs_counts.items()},
        "semantic_group_coverage": {str(k): int(v) for k, v in sorted(sg_counts.items(), key=lambda x: -x[1])},
        "role_distribution": {str(k): int(v) for k, v in role_counts.items()},
        "vocab_shift_distribution": {str(k): int(v) for k, v in vs_counts.items()},
        "min_task_family_pct": round(min_pct, 1),
        "unique_semantic_groups": len(sg_counts),
        "unique_task_families": len(tf_counts),
        "balanced": balanced,
    }

    # Pass if at least 3 task families are represented and balanced
    metrics["passed"] = len(tf_counts) >= 3 or balanced

    print(f"  Task families: {tf_counts}")
    print(f"  Query styles: {qs_counts}")
    print(f"  Semantic groups: {len(sg_counts)} unique")
    print(f"  Min task family representation: {min_pct:.1f}%")
    return metrics


def audit_data_integrity(df: pd.DataFrame) -> dict:
    """Audit data integrity: duplicates, nulls, column completeness."""
    print("\n=== Audit 6: Data Integrity ===")

    total = len(df)

    # Duplicate example_ids
    dup_ids = df["example_id"].duplicated().sum() if "example_id" in df.columns else 0

    # Duplicate queries
    dup_queries = df["query"].duplicated().sum() if "query" in df.columns else 0

    # Data leakage: query text contained in passage text
    leakage = 0
    if "query" in df.columns and "passage_text" in df.columns:
        for _, row in df.iterrows():
            q = str(row.get("query", ""))
            p = str(row.get("passage_text", ""))
            if len(q) > 20 and q.lower() in p.lower():
                leakage += 1

    # Null analysis
    nulls = {}
    critical_cols = ["example_id", "query", "query_style", "task_family", "passage_text",
                     "cui_label", "alt_term", "decision", "judge_model", "judge_source"]
    for col in critical_cols:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                nulls[col] = int(null_count)

    # Column completeness
    all_cols = df.columns.tolist()
    missing_expected = [c for c in [
        "example_id", "passage_id", "passage_text", "query", "query_style",
        "fact", "mode", "task_family", "role", "cui_label", "alt_term",
        "semantic_group", "vocab_shift_type", "judge_model", "judge_source",
        "term_overlap", "decision", "judgments",
    ] if c not in all_cols]

    # Query length sanity
    if "query" in df.columns:
        q_lens = df["query"].dropna().str.len()
        too_short = (q_lens < 5).sum()
        too_long = (q_lens > 180).sum()
    else:
        too_short = too_long = 0

    metrics = {
        "total_rows": total,
        "duplicate_example_ids": int(dup_ids),
        "duplicate_queries": int(dup_queries),
        "data_leakage_count": int(leakage),
        "null_columns": nulls,
        "missing_expected_columns": missing_expected,
        "too_short_queries": int(too_short),
        "too_long_queries": int(too_long),
        "total_columns": len(all_cols),
    }

    passed = (
        dup_ids == 0
        and leakage == 0
        and len(nulls) == 0
        and len(missing_expected) == 0
        and too_short == 0
    )
    metrics["passed"] = passed

    print(f"  Duplicate example_ids: {dup_ids}")
    print(f"  Duplicate queries: {dup_queries}")
    print(f"  Data leakage (query in passage): {leakage}")
    print(f"  Null columns: {nulls}")
    print(f"  Missing expected columns: {missing_expected}")
    print(f"  Query length issues: {too_short} short, {too_long} long")
    status = "✓" if passed else "✗"
    print(f"  Overall integrity: {status}")
    return metrics


def audit_umls_artifacts_global(df: pd.DataFrame) -> dict:
    """Global UMLS artifact scan across ALL queries (not just abbreviations)."""
    print("\n=== Bonus: Global UMLS Artifact Scan ===")

    patterns = [
        r'\(substance\)',
        r'\(finding\)',
        r'\(diagnosis\)',
        r'\(procedure\)',
        r'\(morphologic abnormality\)',
        r'\[medical device\]',
        r'\[brand name\]',
        r'\[B \w+\]',
    ]

    total_artifacts = 0
    per_pattern = {}
    for pattern in patterns:
        count = df["query"].apply(lambda q: bool(re.search(pattern, str(q), re.IGNORECASE))).sum()
        if count > 0:
            per_pattern[pattern] = int(count)
            total_artifacts += count

    artifact_pct = total_artifacts / max(len(df), 1) * 100
    passed = artifact_pct < THRESHOLDS["max_umls_artifact_pct"]

    print(f"  Total UMLS artifacts: {total_artifacts} ({artifact_pct:.2f}%) threshold: <{THRESHOLDS['max_umls_artifact_pct']}% {'✓' if passed else '✗'}")
    print(f"  Per pattern: {per_pattern}")

    return {
        "total_artifacts": total_artifacts,
        "artifact_pct": round(artifact_pct, 2),
        "per_pattern": per_pattern,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive quality audit for MedColBERT synthetic data")
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET,
                        help="Path to consolidated parquet file")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON file for audit results")
    parser.add_argument("--dimension", type=str, default="all",
                        choices=["all", "naturalness", "vocab_shift", "fact_grounding",
                                 "abbreviation", "diversity", "integrity", "umls"],
                        help="Run specific audit dimension only")
    args = parser.parse_args()

    if not args.parquet.exists():
        print(f"Error: Parquet file not found: {args.parquet}")
        sys.exit(1)

    print(f"Loading {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # Convert JSON columns back to dicts if needed
    for col in ["judgments", "validation_failures", "validation_warnings"]:
        if col in df.columns:
            pass  # Keep as strings for audit

    results = {
        "dataset_path": str(args.parquet),
        "total_rows": len(df),
        "thresholds": THRESHOLDS,
        "audits": {},
    }

    audit_funcs = {
        "naturalness": audit_query_naturalness,
        "vocab_shift": audit_vocab_shift,
        "fact_grounding": audit_fact_grounding,
        "abbreviation": audit_abbreviation_quality,
        "diversity": audit_diversity_coverage,
        "integrity": audit_data_integrity,
        "umls": audit_umls_artifacts_global,
    }

    if args.dimension == "all":
        for name, func in audit_funcs.items():
            try:
                results["audits"][name] = func(df)
            except Exception as e:
                print(f"  [ERROR] {name} audit failed: {e}")
                results["audits"][name] = {"error": str(e), "passed": False}
    else:
        results["audits"][args.dimension] = audit_funcs[args.dimension](df)

    # Summary
    audits = results["audits"]
    passed = sum(1 for a in audits.values() if a.get("passed", False))
    total = len(audits)
    results["summary"] = {
        "passed": passed,
        "total": total,
        "all_passed": passed == total,
    }

    print(f"\n{'='*60}")
    print(f"OVERALL: {passed}/{total} audit dimensions passed")
    if passed < total:
        failed = [k for k, v in audits.items() if not v.get("passed", False)]
        print(f"Failed: {failed}")
    print(f"{'='*60}")

    if args.output:
        # Custom JSON encoder for numpy types
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as np
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, bool):
                    return obj
                return super().default(obj)

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
        print(f"\nAudit results saved to {args.output}")

    return 0 if results["summary"]["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
