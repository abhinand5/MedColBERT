"""Real-corpus retrieval eval CLI — the train-on-synthetic quality gate.

Builds a ColBERT PLAID index over the 66k real PubMed corpus, retrieves the
5,967 dev queries, and reports MRR@10, Recall@10/100, nDCG@10. The trained
model MUST beat the BM25 baseline (Recall@100 ≈ 0.63).

Run after stage2:
    uv run python scripts/eval/run_real_corpus.py \\
        --model runs/base_stage2/final \\
        --corpus data/processed/private/passages/real_annotated_500.json \\
        --index-dir runs/base_stage2/eval_index
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from medcolbert.eval.real_corpus import (  # noqa: E402
    load_dev_queries,
    load_real_corpus,
    run_real_corpus_eval,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MedColBERT real-corpus retrieval eval")
    p.add_argument("--model", default=None,
                   help="Path/Hub id of the trained ColBERT model "
                        "(required unless --dev-rows-only).")
    p.add_argument(
        "--corpus",
        default="data/processed/private/passages/real_annotated_500.json",
        help="Real PubMed corpus JSON.",
    )
    p.add_argument("--index-dir", default="runs/base_stage2/eval_index")
    p.add_argument("--index-name", default="real_corpus")
    p.add_argument("--k", type=int, default=100,
                   help="Top-k to retrieve (>=100 for Recall@100).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--no-override-index", action="store_true",
                   help="Reuse an existing PLAID index.")
    p.add_argument("--out", default=None,
                   help="Optional path to write metrics JSON.")
    p.add_argument("--dev-rows-only", action="store_true",
                   help="Only load and print dev row count (no model/index).")
    p.add_argument("--max-queries", type=int, default=None,
                   help="Subset dev queries to this many (for fast CPU eval). "
                        "Keeps only queries whose positive is in the corpus subset.")
    p.add_argument("--max-corpus", type=int, default=None,
                   help="Subset corpus to this many passages. All positives for "
                        "the kept queries are guaranteed included; the rest are "
                        "randomly sampled. For fast CPU eval.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    corpus_path = REPO_ROOT / args.corpus if not Path(args.corpus).is_absolute() else Path(args.corpus)

    if args.dev_rows_only:
        rows = load_dev_queries()
        print(f"[eval] dev rows: {len(rows)}")
        return

    if args.model is None:
        raise SystemExit("--model is required unless --dev-rows-only is set.")

    dev_rows = None
    corpus_path = corpus_path

    # Subset for fast CPU eval: ensure positives are in the corpus subset.
    if args.max_queries is not None or args.max_corpus is not None:
        import random

        full_corpus = load_real_corpus(corpus_path)
        dev_rows = load_dev_queries()
        corpus_by_id = {r["passage_id"]: r for r in full_corpus}

        # Filter dev rows to those whose positive is in the corpus.
        dev_rows = [r for r in dev_rows if r["positive_passage_id"] in corpus_by_id]
        print(f"[eval] dev rows with positive in corpus: {len(dev_rows)}")

        # Subset queries.
        max_q = args.max_queries or len(dev_rows)
        if max_q < len(dev_rows):
            random.seed(42)
            dev_rows = random.sample(dev_rows, max_q)
        print(f"[eval] using {len(dev_rows)} queries")

        # Build corpus subset: all positives + random fill.
        positive_ids = {r["positive_passage_id"] for r in dev_rows}
        positive_passages = [corpus_by_id[pid] for pid in positive_ids]
        other_passages = [r for r in full_corpus if r["passage_id"] not in positive_ids]
        max_c = args.max_corpus or len(full_corpus)
        if max_c < len(full_corpus):
            random.seed(42)
            need = max(0, max_c - len(positive_passages))
            other_passages = random.sample(other_passages, min(need, len(other_passages)))
        corpus_subset = positive_passages + other_passages
        print(f"[eval] corpus subset: {len(corpus_subset)} passages "
              f"({len(positive_passages)} positives + {len(other_passages)} others)")

        # Write subset corpus to a temp file.
        subset_path = REPO_ROOT / args.index_dir / "corpus_subset.json"
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        with open(subset_path, "w", encoding="utf-8") as f:
            json.dump(corpus_subset, f)
        corpus_path = subset_path
        print(f"[eval] wrote subset corpus to {subset_path}")
    else:
        dev_rows = None

    metrics = run_real_corpus_eval(
        model_path=args.model,
        corpus_path=corpus_path,
        index_dir=REPO_ROOT / args.index_dir,
        index_name=args.index_name,
        dev_rows=dev_rows,
        k=args.k,
        batch_size=args.batch_size,
        override_index=not args.no_override_index,
    )
    d = metrics.as_dict()
    print("[eval] real-corpus retrieval metrics:")
    for k, v in d.items():
        if k != "per_query":
            print(f"  {k}: {v}")
    print("[eval] BM25 baseline Recall@100 ≈ 0.63 — model must beat this.")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        print(f"[eval] metrics written to {out_path}")


if __name__ == "__main__":
    main()
