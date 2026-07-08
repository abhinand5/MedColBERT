"""Stage 2 — dense (single-vector) MedEmbed finetune on the v1 hard-negative
triplets.

Loads the full ``long`` config from ``fierysurf/medcolbert-training-v1``
(one row per kept negative, ~711k rows), builds a dense SentenceTransformer
from BioClinical-ModernBERT-base, and trains with
CachedMultipleNegativesRankingLoss (GradCache) so the effective batch can be
large without OOM, since the in-batch-negatives loss is incompatible with
gradient accumulation.

The in-loop evaluator uses a held-out slice of the training data as a triplet
monitor (TripletEvaluator). The real quality gate is the separate real-corpus
retrieval eval (a dense index against the same 66k passages / 5,967 dev
queries), which must be run after this stage.

Run (single L40S):
    uv run python scripts/training/dense_finetune.py \\
        --train-config configs/train_dense_base.yaml \\
        --model-config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from medcolbert.training.sbert_train import run_finetune  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MedEmbed dense finetune")
    p.add_argument("--train-config", default="configs/train_dense_base.yaml")
    p.add_argument("--model-config", default="configs/base.yaml")
    p.add_argument("--stage", default="stage2_dense")
    p.add_argument("--config-name", default="long",
                   help="HF dataset config (long=one negative per row).")
    p.add_argument("--split", default="train")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Cap training rows (default: full dataset).")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override max_steps.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override per_device_train_batch_size (effective batch).")
    p.add_argument("--mini-batch-size", type=int, default=None,
                   help="Override GradCache mini_batch_size (forces grad_accum=1).")
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--no-gradient-checkpointing", action="store_true",
                   help="Disable gradient checkpointing (use more VRAM, run faster). "
                        "Recommended on a 48GB L40S.")
    p.add_argument("--save-strategy", choices=["steps", "epoch"], default=None,
                   help="Checkpoint save strategy. Default: from config (steps).")
    p.add_argument("--save-total-limit", type=int, default=None,
                   help="Max checkpoints to keep (oldest deleted). Default: 3.")
    p.add_argument("--exclude-weak", action="store_true",
                   help="Drop negatives flagged weak by the audit.")
    p.add_argument("--min-relevant-looks", type=int, default=None,
                   help="Minimum audit_relevant_looks to keep a negative.")
    p.add_argument("--eval-holdout", type=int, default=500,
                   help="Rows held out from train for the triplet evaluator.")
    p.add_argument("--output-dir", default=None,
                   help="Output directory. Default: config's outputs.run_dir "
                        "(runs/dense_base_stage2 / runs/dense_large_stage2).")
    p.add_argument("--run-name", default="medembed-dense-base-stage2")
    p.add_argument("--save-final", action="store_true", default=True,
                   help="Save the final model to <output_dir>/final.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    run_finetune(parse_args())


if __name__ == "__main__":
    main()
