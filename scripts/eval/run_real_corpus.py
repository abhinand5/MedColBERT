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

    metrics = run_real_corpus_eval(
        model_path=args.model,
        corpus_path=corpus_path,
        index_dir=REPO_ROOT / args.index_dir,
        index_name=args.index_name,
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
