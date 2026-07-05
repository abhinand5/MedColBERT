"""Stage 2 — main ColBERT finetune on the v1 hard-negative triplets.

Loads the full ``long`` config from ``fierysurf/medcolbert-training-v1`` (one
row per kept negative, ~711k rows), builds the ColBERT model from
BioClinical-ModernBERT-base, and trains with the cached contrastive loss
(GradCache) so the effective batch (per_device_train_batch_size) can be large
without OOM, since the contrastive loss is incompatible with gradient
accumulation.

The in-loop evaluator uses a held-out slice of the training data as a triplet
monitor. The real quality gate is the separate real-corpus retrieval eval
(``scripts/eval/run_real_corpus.py``), which must be run after this stage.

Run (single L40S):
    uv run python scripts/training/stage2_colbert_finetune.py \\
        --train-config configs/train_base.yaml \\
        --model-config configs/base.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from medcolbert.training.datasets import (  # noqa: E402
    prepare_training_dataset,
)
from medcolbert.training.pylate_train import (  # noqa: E402
    build_evaluator,
    build_loss,
    build_model,
    build_trainer,
    enable_wandb_reporting,
    resolve_stage,
    wandb_enabled,
)
from medcolbert.utils.config import load_configs  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MedColBERT stage2 ColBERT finetune")
    p.add_argument("--train-config", default="configs/train_base.yaml")
    p.add_argument("--model-config", default="configs/base.yaml")
    p.add_argument("--stage", default="stage2_colbert")
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
                   help="Override GradCache mini_batch_size.")
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--no-gradient-checkpointing", action="store_true",
                   help="Disable gradient checkpointing (use more VRAM, run "
                        "faster). Recommended on a 48GB L40S.")
    p.add_argument("--save-strategy", choices=["steps", "epoch"],
                   default=None,
                   help="Checkpoint save strategy. 'epoch' saves at the end of "
                        "each epoch (auto-detected from dataset size + batch). "
                        "Default: from config (steps).")
    p.add_argument("--save-total-limit", type=int, default=None,
                   help="Max checkpoints to keep (oldest deleted). Default: 3.")
    p.add_argument("--exclude-weak", action="store_true",
                   help="Drop negatives flagged weak by the audit.")
    p.add_argument("--min-relevant-looks", type=int, default=None,
                   help="Minimum audit_relevant_looks to keep a negative.")
    p.add_argument("--eval-holdout", type=int, default=500,
                   help="Rows held out from train for the triplet evaluator.")
    p.add_argument("--output-dir", default="runs/base_stage2")
    p.add_argument("--run-name", default="medcolbert-base-stage2")
    p.add_argument("--save-final", action="store_true", default=True,
                   help="Save the final model to <output_dir>/final.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_configs(args.model_config, args.train_config)

    stage = resolve_stage(cfg, args.stage)
    if args.max_steps is not None:
        stage.max_steps = args.max_steps
    if args.batch_size is not None:
        stage.per_device_train_batch_size = args.batch_size
    if args.mini_batch_size is not None:
        stage.mini_batch_size = args.mini_batch_size
        # CachedContrastive is incompatible with gradient accumulation;
        # when mini_batch_size is set (esp. via CLI), force grad_accum=1 so
        # the configured accum doesn't silently inflate the logical batch.
        stage.gradient_accumulation_steps = 1
    if args.learning_rate is not None:
        stage.learning_rate = args.learning_rate
    if args.temperature is not None:
        stage.temperature = args.temperature
    if args.no_gradient_checkpointing:
        stage.gradient_checkpointing = False
    if args.save_strategy is not None:
        stage.save_strategy = args.save_strategy
    if args.save_total_limit is not None:
        stage.save_total_limit = args.save_total_limit

    # Single-GPU: never gather across devices (no all-gather partner).
    if torch.cuda.device_count() <= 1:
        stage.gather_across_devices = False

    if wandb_enabled():
        enable_wandb_reporting(stage)
        print(f"[stage2] wandb reporting enabled "
              f"(project={os.environ.get('WANDB_PROJECT', '<unset>')})")
    else:
        print("[stage2] wandb not enabled (set WANDB_API_KEY to enable)")

    print(f"[stage2] loading {args.config_name}/{args.split} from HF...")
    train_ds, reports = prepare_training_dataset(
        config=args.config_name,
        split=args.split,
        keep_audit=False,
        min_relevant_looks=args.min_relevant_looks,
        exclude_weak=args.exclude_weak,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    for r in reports:
        print(f"[stage2] filter report: {r.as_dict()}")
    print(f"[stage2] train rows: {len(train_ds)}")

    # Hold out a small slice for the in-loop triplet evaluator.
    holdout = min(args.eval_holdout, max(1, len(train_ds) // 100))
    split = train_ds.train_test_split(test_size=holdout, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"[stage2] eval holdout: {len(eval_ds)} rows")

    print(f"[stage2] building ColBERT from {cfg['model']}...")
    model = build_model(cfg, stage)

    print(f"[stage2] loss={stage.loss_type} "
          f"batch={stage.per_device_train_batch_size} "
          f"mini={stage.mini_batch_size} temp={stage.temperature} "
          f"effective={stage.effective_batch} "
          f"steps={stage.max_steps} lr={stage.learning_rate} "
          f"save={stage.save_strategy} limit={stage.save_total_limit}")
    loss = build_loss(model, stage)

    evaluator = build_evaluator(eval_ds, name="stage2_eval")

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = build_trainer(
        model=model,
        stage_cfg=stage,
        train_dataset=train_ds,
        loss=loss,
        output_dir=output_dir,
        run_name=args.run_name,
        eval_dataset=eval_ds,
        evaluator=evaluator,
    )

    print("[stage2] starting training...")
    trainer.train()

    if args.save_final:
        final_dir = output_dir / "final"
        model.save_pretrained(str(final_dir))
        print(f"[stage2] done. final model saved to {final_dir}")
        print("[stage2] NEXT: run scripts/eval/run_real_corpus.py "
              "--model <final_dir> (the real-corpus retrieval gate).")
    else:
        print("[stage2] done.")


if __name__ == "__main__":
    main()
