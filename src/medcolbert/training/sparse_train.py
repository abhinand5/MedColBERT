"""Construction of the sparse (SPLADE) SparseEncoder model, loss, and trainer.

Mirrors ``pylate_train`` for a single-vector *sparse* embedding (SPLADE)
instead of ColBERT late interaction. The training loop is delegated to
``SparseEncoderTrainer`` (from sentence-transformers, which wraps the
HuggingFace ``Trainer``). This module only assembles the pieces from config
dicts so scripts stay thin.

Loss: ``SpladeLoss`` wrapping ``SparseMultipleNegativesRankingLoss`` is the
canonical SPLADE training recipe — it adds the FLOPS regularizer that keeps
the sparse activation budget bounded. Plain ``SparseMultipleNegativesRankingLoss``
(no regularizer) is also supported via ``loss_type: sparse_mnrl``. Unlike the
dense/ColBERT GradCache losses, sparse in-batch negatives *are* compatible
with gradient accumulation; duplicates within a batch are avoided via the
``NO_DUPLICATES`` batch sampler.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentence_transformers import (
    SparseEncoder,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.sparse_encoder.evaluation import SparseTripletEvaluator
from sentence_transformers.sparse_encoder.losses import (
    SparseMultipleNegativesRankingLoss,
    SpladeLoss,
)
from sentence_transformers.training_args import BatchSamplers

from medcolbert.training.datasets import prepare_training_dataset, to_sbert_columns
from medcolbert.utils.config import load_configs

_BATCH_SAMPLERS = {
    "no_duplicates": BatchSamplers.NO_DUPLICATES,
    "batch_sampler": BatchSamplers.BATCH_SAMPLER,
}


def wandb_enabled() -> bool:
    """True if wandb should report (WANDB_API_KEY set)."""
    return bool(os.environ.get("WANDB_API_KEY"))


def enable_wandb_reporting(stage_cfg: SparseStageConfig) -> SparseStageConfig:
    """Flip ``report_to`` to wandb if WANDB_API_KEY is set; else leave as-is."""
    if wandb_enabled():
        stage_cfg.report_to = ["wandb"]
    return stage_cfg


@dataclass
class SparseStageConfig:
    """Resolved hyperparameters for one sparse (SPLADE) training stage."""

    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    bf16: bool
    fp16: bool
    max_steps: int | None
    num_train_epochs: float | None
    warmup_ratio: float
    gradient_checkpointing: bool
    loss_type: str  # "splade_mnr" | "sparse_mnrl"
    query_regularizer_weight: float
    document_regularizer_weight: float
    query_length: int
    document_length: int
    batch_sampler: str  # "no_duplicates" | "batch_sampler"
    save_steps: int
    eval_steps: int
    logging_steps: int
    report_to: list[str]
    save_strategy: str  # "steps" | "epoch"
    save_total_limit: int

    @property
    def effective_batch(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


def resolve_stage(
    merged_cfg: dict[str, Any],
    stage: str,
) -> SparseStageConfig:
    """Pull a stage block out of the merged config and normalise it.

    ``stage`` is a key like ``"stage2_sparse"``. Falls back to base-model
    lengths for query/document when the stage block does not override them.
    """
    block = merged_cfg.get(stage, {}) or {}
    base = merged_cfg

    max_steps = block.get("max_steps")
    num_train_epochs = block.get("num_train_epochs")
    # If neither max_steps nor epochs set, default to 1 epoch.
    if max_steps is None and num_train_epochs is None:
        num_train_epochs = 1

    loss_type = block.get("loss_type", "splade_mnr")

    outputs = merged_cfg.get("outputs", {}) or {}
    report_to = block.get("report_to", ["none"])
    return SparseStageConfig(
        learning_rate=block.get("learning_rate", 2e-5),
        per_device_train_batch_size=block.get("per_device_train_batch_size", 16),
        gradient_accumulation_steps=block.get("gradient_accumulation_steps", 1),
        bf16=block.get("bf16", True),
        fp16=block.get("fp16", False),
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        warmup_ratio=block.get("warmup_ratio", 0.0),
        gradient_checkpointing=block.get("gradient_checkpointing", False),
        loss_type=loss_type,
        query_regularizer_weight=block.get("query_regularizer_weight", 5e-5),
        document_regularizer_weight=block.get("document_regularizer_weight", 3e-5),
        query_length=block.get("query_length", base.get("query_maxlen", 64)),
        document_length=block.get("document_length", base.get("doc_maxlen", 256)),
        batch_sampler=block.get("batch_sampler", "no_duplicates"),
        save_steps=block.get("save_steps", outputs.get("save_steps", 10000)),
        eval_steps=block.get("eval_steps", outputs.get("eval_steps", 5000)),
        logging_steps=block.get("logging_steps", 20),
        report_to=report_to,
        save_strategy=block.get("save_strategy", "steps"),
        save_total_limit=block.get("save_total_limit", 3),
    )


def build_model(merged_cfg: dict[str, Any], stage_cfg: SparseStageConfig) -> SparseEncoder:
    """Instantiate a SparseEncoder from the backbone in config.

    ``SparseEncoder`` builds the SPLADE pooling head on top of the backbone
    transformer. Loading in fp32 is preferred for training stability (per the
    sbert docs); override via the model config's ``model_kwargs`` if needed.
    """
    model_name = merged_cfg.get("model")
    if not model_name:
        raise ValueError("merged_cfg['model'] is required (backbone name).")
    model_kwargs = merged_cfg.get("model_kwargs") or {"torch_dtype": "float32"}
    model = SparseEncoder(model_name, model_kwargs=model_kwargs)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = stage_cfg.document_length
    return model


def build_loss(
    model: SparseEncoder,
    stage_cfg: SparseStageConfig,
):
    """Build the sparse in-batch-negatives loss.

    ``splade_mnr`` wraps ``SparseMultipleNegativesRankingLoss`` in
    ``SpladeLoss`` (the FLOPS-regularized canonical recipe).
    ``sparse_mnrl`` is the unregularized variant.
    """
    if stage_cfg.loss_type == "sparse_mnrl":
        return SparseMultipleNegativesRankingLoss(model=model)
    if stage_cfg.loss_type == "splade_mnr":
        base = SparseMultipleNegativesRankingLoss(model=model)
        return SpladeLoss(
            model=model,
            loss=base,
            query_regularizer_weight=stage_cfg.query_regularizer_weight,
            document_regularizer_weight=stage_cfg.document_regularizer_weight,
        )
    raise ValueError(f"Unknown loss_type: {stage_cfg.loss_type}")


def build_training_args(
    stage_cfg: SparseStageConfig,
    output_dir: Path | str,
    run_name: str,
    eval_strategy: str = "steps",
) -> SparseEncoderTrainingArguments:
    """Build HF-style training arguments from a resolved stage config.

    ``batch_sampler=NO_DUPLICATES`` prevents duplicate anchors in a batch,
    which strengthens the in-batch-negative signal for
    ``SparseMultipleNegativesRankingLoss``.
    """
    batch_sampler = _BATCH_SAMPLERS.get(
        stage_cfg.batch_sampler, BatchSamplers.NO_DUPLICATES
    )
    kwargs: dict[str, Any] = dict(
        output_dir=str(output_dir),
        run_name=run_name,
        learning_rate=stage_cfg.learning_rate,
        per_device_train_batch_size=stage_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=stage_cfg.gradient_accumulation_steps,
        warmup_ratio=stage_cfg.warmup_ratio,
        bf16=stage_cfg.bf16,
        fp16=stage_cfg.fp16,
        gradient_checkpointing=stage_cfg.gradient_checkpointing,
        eval_strategy=eval_strategy,
        eval_steps=stage_cfg.eval_steps,
        logging_steps=stage_cfg.logging_steps,
        save_total_limit=stage_cfg.save_total_limit,
        report_to=stage_cfg.report_to,
        batch_sampler=batch_sampler,
        dataloader_drop_last=True,
    )
    if stage_cfg.save_strategy == "epoch":
        kwargs["save_strategy"] = "epoch"
    else:
        kwargs["save_strategy"] = "steps"
        kwargs["save_steps"] = stage_cfg.save_steps
    if stage_cfg.max_steps is not None:
        kwargs["max_steps"] = stage_cfg.max_steps
        kwargs.pop("num_train_epochs", None)
    else:
        kwargs["num_train_epochs"] = stage_cfg.num_train_epochs or 1
    return SparseEncoderTrainingArguments(**kwargs)


def build_evaluator(
    eval_dataset,
    name: str = "dev",
) -> SparseTripletEvaluator:
    """Build a sparse triplet evaluator from a dev set with anchor/positive/negative."""
    return SparseTripletEvaluator(
        anchors=eval_dataset["anchor"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name=name,
    )


def build_trainer(
    model: SparseEncoder,
    stage_cfg: SparseStageConfig,
    train_dataset,
    loss,
    output_dir: Path | str,
    run_name: str,
    eval_dataset=None,
    evaluator=None,
) -> SparseEncoderTrainer:
    """Assemble the sparse trainer."""
    args = build_training_args(stage_cfg, output_dir, run_name)
    return SparseEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )


def run_finetune(args) -> None:
    """Run the sparse finetune end-to-end from a namespace of CLI args.

    ``args`` is any object with the attributes: ``train_config``,
    ``model_config``, ``stage``, ``config_name``, ``split``, ``max_examples``,
    ``max_steps``, ``batch_size``, ``learning_rate``,
    ``query_regularizer_weight``, ``document_regularizer_weight``,
    ``no_gradient_checkpointing``, ``save_strategy``, ``save_total_limit``,
    ``exclude_weak``, ``min_relevant_looks``, ``eval_holdout``,
    ``output_dir`` (optional; ``None`` falls back to the config's
    ``outputs.run_dir``), ``run_name``, ``save_final``, ``seed``.
    """
    cfg = load_configs(args.model_config, args.train_config)
    stage = resolve_stage(cfg, args.stage)

    if args.max_steps is not None:
        stage.max_steps = args.max_steps
    if args.batch_size is not None:
        stage.per_device_train_batch_size = args.batch_size
    if args.learning_rate is not None:
        stage.learning_rate = args.learning_rate
    if args.query_regularizer_weight is not None:
        stage.query_regularizer_weight = args.query_regularizer_weight
    if args.document_regularizer_weight is not None:
        stage.document_regularizer_weight = args.document_regularizer_weight
    if args.no_gradient_checkpointing:
        stage.gradient_checkpointing = False
    if args.save_strategy is not None:
        stage.save_strategy = args.save_strategy
    if args.save_total_limit is not None:
        stage.save_total_limit = args.save_total_limit

    if wandb_enabled():
        enable_wandb_reporting(stage)
        print(f"[sparse] wandb reporting enabled "
              f"(project={os.environ.get('WANDB_PROJECT', '<unset>')})")
    else:
        print("[sparse] wandb not enabled (set WANDB_API_KEY to enable)")

    print(f"[sparse] loading {args.config_name}/{args.split} from HF...")
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
        print(f"[sparse] filter report: {r.as_dict()}")
    train_ds = to_sbert_columns(train_ds)
    print(f"[sparse] train rows: {len(train_ds)}")

    holdout = min(args.eval_holdout, max(1, len(train_ds) // 100))
    split = train_ds.train_test_split(test_size=holdout, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"[sparse] eval holdout: {len(eval_ds)} rows")

    print(f"[sparse] building SparseEncoder from {cfg['model']}...")
    model = build_model(cfg, stage)

    print(f"[sparse] loss={stage.loss_type} "
          f"batch={stage.per_device_train_batch_size} "
          f"effective={stage.effective_batch} steps={stage.max_steps} "
          f"lr={stage.learning_rate} q_reg={stage.query_regularizer_weight} "
          f"d_reg={stage.document_regularizer_weight} "
          f"sampler={stage.batch_sampler} save={stage.save_strategy} "
          f"limit={stage.save_total_limit}")
    loss = build_loss(model, stage)
    evaluator = build_evaluator(eval_ds, name="sparse_eval")

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        cfg.get("outputs", {}).get("run_dir", "runs/sparse_stage2")
    )
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
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

    print("[sparse] starting training...")
    trainer.train()

    if args.save_final:
        final_dir = output_dir / "final"
        model.save_pretrained(str(final_dir))
        print(f"[sparse] done. final model saved to {final_dir}")
        print("[sparse] NEXT: run a sparse real-corpus retrieval eval against "
              "<final_dir> (extend scripts/eval/run_real_corpus.py for "
              "SparseEncoder encode + sparse top-k).")
    else:
        print("[sparse] done.")
