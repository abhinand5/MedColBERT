#!/usr/bin/env python
"""Regenerate full_question queries for the diverse (div_v6) batches with the
forced-opener prompt, REUSING the already-extracted facts/alt_terms/passages.

Why: the v6_fresh diverse batches' full_question rows collapsed into a single
"Does [NP] [VP]?" template (~37% of full_question). The keyword rows are fine
and the facts/passages are good, so we only need to re-emit the QUESTION with a
forced diverse opener — one vLLM call per row (no fact re-extraction), which is
~5x cheaper than re-running whole batches.

Output: fq_v6_NNNN batch dirs under v6_fresh, each holding ~BATCH rows of
regenerated full_question examples (all columns preserved, query/term_overlap/
ngram_copy_4/alt_term_overlap recomputed). Consolidation then keeps keyword rows
from div_v6_* and full_question rows from fq_v6_*.

Usage:
    uv run python scripts/data/regen_full_questions.py [--batch-size 400] [--workers 8]
"""
import argparse
import glob
import hashlib
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

BASE = Path("/workspace/MedColBERT")
V6 = BASE / "data/processed/private/synthetic/v6_fresh"

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BASE / "src"))
from multistyle_batch_v2 import (  # noqa: E402
    generate_full_question,
    compute_term_overlap,
    compute_alt_term_overlap,
)
from medcolbert.utils.hashing import stable_hash  # noqa: E402


def _ngram_copy(query: str, passage: str, n: int = 4) -> float:
    q_words = re.findall(r"[a-z0-9]+", query.lower())
    p_text = passage.lower()
    if len(q_words) < n:
        return 0.0
    ngrams = [" ".join(q_words[i : i + n]) for i in range(len(q_words) - n + 1)]
    copied = sum(1 for ng in ngrams if ng in p_text)
    return copied / len(ngrams) if ngrams else 0.0


def regen_one(row, batch_id, idx):
    """Regenerate the full_question query for one row. Returns (row_dict, ok)."""
    fact = row.get("fact", "")
    role = row.get("role", "patient")
    tf = row.get("task_family", "")
    cui_label = row.get("cui_label", "")
    alt_term = row.get("alt_term", "")
    passage = row.get("passage_text", "")
    try:
        query = generate_full_question(fact, role, tf, cui_label, alt_term)
    except Exception as e:
        return None, f"gen_error: {e}"
    if not query or len(query.strip()) < 10:
        return None, "empty/short"
    query = query.strip().strip('"').strip("'")

    # Validation: n-gram copy + term overlap (same rules as the batch post-filter)
    nc4 = _ngram_copy(query, passage, n=4)
    nc3 = _ngram_copy(query, passage, n=3)
    if nc4 > 0.30 or nc3 > 0.50:
        return None, f"high_ngram_copy 4={nc4:.2f} 3={nc3:.2f}"
    overlap = compute_term_overlap(query, passage, cui_label, tf)
    if overlap < 0.10:
        return None, f"no_term_overlap {overlap:.2f}"

    out = row.to_dict()
    out["query"] = query
    out["query_style"] = "full_question"
    out["term_overlap"] = round(overlap, 3)
    out["ngram_copy_4"] = round(nc4, 3)
    out["alt_term_overlap"] = round(compute_alt_term_overlap(query, alt_term), 3)
    out["generator_model"] = "gemma-4-31B"
    out["judge_source"] = "none_forced_opener_regen"
    out["example_id"] = stable_hash(
        row.get("passage_id", ""), query, "full_question", batch_id, str(idx)
    )
    return out, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=400)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    # Load all div_v6 full_question rows (the ones we're replacing)
    div_files = sorted(glob.glob(str(V6 / "div_v6_*" / "accepted.parquet")))
    print(f"Loading full_question rows from {len(div_files)} div_v6 batches...")
    frames = []
    for f in div_files:
        df = pd.read_parquet(f)
        fq = df[df.get("query_style", "") == "full_question"]
        if len(fq):
            frames.append(fq)
    if not frames:
        print("No full_question rows found!"); return
    all_fq = pd.concat(frames, ignore_index=True)
    print(f"Total full_question rows to regenerate: {len(all_fq)}")

    # Regenerate in parallel
    n = len(all_fq)
    results = []
    failures = []
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(regen_one, all_fq.iloc[i], "fq_v6", i): i
            for i in range(n)
        }
        for fut in as_completed(futs):
            row_dict, err = fut.result()
            done += 1
            if row_dict is not None:
                results.append(row_dict)
            else:
                failures.append(err)
            if done % 500 == 0:
                rate = done / (time.time() - t0)
                print(f"  {done}/{n} done, {len(results)} accepted, "
                      f"{len(failures)} rejected ({rate:.0f}/min)")

    print(f"\nRegen complete: {len(results)} accepted, {len(failures)} rejected "
          f"in {(time.time()-t0)/60:.1f} min")
    if failures:
        from collections import Counter
        print("  rejection reasons:", Counter(failures).most_common(8))

    if not results:
        print("No accepted regen rows!"); return

    # Write to fq_v6_NNNN batch dirs
    res_df = pd.DataFrame(results)
    n_batches = (len(res_df) + args.batch_size - 1) // args.batch_size
    written = 0
    for bi in range(n_batches):
        chunk = res_df.iloc[bi * args.batch_size : (bi + 1) * args.batch_size]
        bid = f"fq_v6_{bi + 1:04d}"
        out_dir = V6 / bid
        out_dir.mkdir(parents=True, exist_ok=True)
        chunk.to_parquet(out_dir / "accepted.parquet", index=False)
        stats = {
            "total": int(len(chunk)), "accepted": int(len(chunk)),
            "acceptance_rate": 1.0, "batch_id": bid,
            "source": "regen_full_questions", "rows": int(len(chunk)),
        }
        import json
        with open(out_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        written += len(chunk)
    print(f"Wrote {written} rows across {n_batches} fq_v6_* batch dirs")


if __name__ == "__main__":
    main()
