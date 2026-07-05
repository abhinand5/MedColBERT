"""Stage 0 smoke test — verify the full training stack works on a tiny subset.

Loads a 1000-row subset of the ``long`` triplets, builds the ColBERT model from
BioClinical-ModernBERT-base, runs ~1000 steps with the cached contrastive loss,
and confirms the loss decreases and the model fits in VRAM.

Run:
    uv run python scripts/training/stage0_smoke.py \\
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
    p = argparse.ArgumentParser(description="MedColBERT stage0 smoke test")
    p.add_argument("--train-config", default="configs/train_base.yaml")
    p.add_argument("--model-config", default="configs/base.yaml")
    p.add_argument("--stage", default="stage0_smoke")
    p.add_argument("--max-examples", type=int, default=None,
                   help="Override max_examples (default: from config).")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override max_steps (default: from config).")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override per_device_train_batch_size.")
    p.add_argument("--output-dir", default="runs/base_smoke")
    p.add_argument("--run-name", default="medcolbert-stage0-smoke")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_configs(args.model_config, args.train_config)

    stage = resolve_stage(cfg, args.stage)
    if args.max_examples is not None:
        cfg.setdefault(args.stage, {})["max_examples"] = args.max_examples
    max_examples = args.max_examples or cfg.get(args.stage, {}).get("max_examples", 1000)
    if args.max_steps is not None:
        stage.max_steps = args.max_steps
    if args.batch_size is not None:
        stage.per_device_train_batch_size = args.batch_size

    # Single-GPU: never gather across devices.
    if torch.cuda.device_count() <= 1:
        stage.gather_across_devices = False

    if wandb_enabled():
        enable_wandb_reporting(stage)
        print(f"[stage0] wandb reporting enabled "
              f"(project={os.environ.get('WANDB_PROJECT', '<unset>')})")
    else:
        print("[stage0] wandb not enabled (set WANDB_API_KEY to enable)")

    print(f"[stage0] loading up to {max_examples} triplets from HF long config...")
    train_ds, reports = prepare_training_dataset(
        config="long",
        split="train",
        keep_audit=False,
        max_examples=max_examples,
        seed=args.seed,
    )
    for r in reports:
        print(f"[stage0] filter report: {r.as_dict()}")
    print(f"[stage0] train rows: {len(train_ds)}")

    # Hold out a tiny slice for the in-loop triplet evaluator.
    split = train_ds.train_test_split(test_size=min(64, max(1, len(train_ds) // 10)),
                                      seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]

    print(f"[stage0] building ColBERT from {cfg['model']}...")
    model = build_model(cfg, stage)

    print(f"[stage0] loss={stage.loss_type} batch={stage.per_device_train_batch_size} "
          f"mini={stage.mini_batch_size} temp={stage.temperature} "
          f"effective={stage.effective_batch}")
    loss = build_loss(model, stage)

    evaluator = build_evaluator(eval_ds, name="stage0_eval")

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

    print("[stage0] starting training...")
    trainer.train()

    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    print(f"[stage0] done. model saved to {final_dir}")


if __name__ == "__main__":
    main()
