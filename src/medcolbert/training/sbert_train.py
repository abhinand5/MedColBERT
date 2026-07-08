"""Construction of the dense (single-vector) SentenceTransformer model, loss,
and trainer.

Mirrors ``pylate_train`` but targets a single-vector dense embedding instead
of ColBERT token-level interaction. The training loop is delegated to
``SentenceTransformerTrainer`` (from sentence-transformers, which wraps the
HuggingFace ``Trainer``). This module only assembles the pieces from config
dicts so scripts stay thin.

Key constraint: ``CachedMultipleNegativesRankingLoss`` (GradCache) is
**not compatible with gradient accumulation** — in-batch negatives require
the full batch in one backward. To emulate a large effective batch on a
single GPU it encodes in ``mini_batch_size`` chunks but computes the loss
over the full ``per_device_train_batch_size``. ``gradient_accumulation_steps``
is forced to 1 when the cached loss is used, exactly like PyLate's
``CachedContrastive``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    evaluation,
    losses,
)
from sentence_transformers import models as st_models

from medcolbert.training.datasets import prepare_training_dataset, to_sbert_columns
from medcolbert.utils.config import load_configs


def wandb_enabled() -> bool:
    """True if wandb should report (WANDB_API_KEY set)."""
    return bool(os.environ.get("WANDB_API_KEY"))


def enable_wandb_reporting(stage_cfg: DenseStageConfig) -> DenseStageConfig:
    """Flip ``report_to`` to wandb if WANDB_API_KEY is set; else leave as-is.

    The wandb project is read from ``WANDB_PROJECT`` by the HF Trainer. The
    run name is the stage's ``run_name`` passed to ``build_trainer``.
    """
    if wandb_enabled():
        stage_cfg.report_to = ["wandb"]
    return stage_cfg


@dataclass
class DenseStageConfig:
    """Resolved hyperparameters for one dense training stage."""

    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    bf16: bool
    fp16: bool
    max_steps: int | None
    num_train_epochs: float | None
    warmup_ratio: float
    gradient_checkpointing: bool
    loss_type: str  # "mnrl" | "cached_mnrl"
    mini_batch_size: int | None
    query_length: int
    document_length: int
    pooling: str  # "mean" | "cls" | "max"
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
) -> DenseStageConfig:
    """Pull a stage block out of the merged config and normalise it.

    ``stage`` is a key like ``"stage2_dense"``. Falls back to base-model
    lengths for query/document and to the model config's ``pooling`` when the
    stage block does not override them.
    """
    block = merged_cfg.get(stage, {}) or {}
    base = merged_cfg

    max_steps = block.get("max_steps")
    num_train_epochs = block.get("num_train_epochs")
    # If neither max_steps nor epochs set, default to 1 epoch.
    if max_steps is None and num_train_epochs is None:
        num_train_epochs = 1

    loss_type = block.get("loss_type", "cached_mnrl")
    mini_batch_size = block.get("mini_batch_size")

    # When using the cached (GradCache) loss, gradient accumulation must be 1.
    grad_accum = block.get("gradient_accumulation_steps", 1)
    if loss_type == "cached_mnrl" and mini_batch_size is not None:
        grad_accum = 1

    outputs = merged_cfg.get("outputs", {}) or {}
    report_to = block.get("report_to", ["none"])
    return DenseStageConfig(
        learning_rate=block.get("learning_rate", 1e-5),
        per_device_train_batch_size=block.get("per_device_train_batch_size", 16),
        gradient_accumulation_steps=grad_accum,
        bf16=block.get("bf16", True),
        fp16=block.get("fp16", False),
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        warmup_ratio=block.get("warmup_ratio", 0.0),
        gradient_checkpointing=block.get("gradient_checkpointing", False),
        loss_type=loss_type,
        mini_batch_size=mini_batch_size,
        query_length=block.get("query_length", base.get("query_maxlen", 64)),
        document_length=block.get("document_length", base.get("doc_maxlen", 256)),
        pooling=block.get("pooling", base.get("pooling", "mean")),
        save_steps=block.get("save_steps", outputs.get("save_steps", 10000)),
        eval_steps=block.get("eval_steps", outputs.get("eval_steps", 5000)),
        logging_steps=block.get("logging_steps", 20),
        report_to=report_to,
        save_strategy=block.get("save_strategy", "steps"),
        save_total_limit=block.get("save_total_limit", 3),
    )


def build_model(merged_cfg: dict[str, Any], stage_cfg: DenseStageConfig) -> SentenceTransformer:
    """Instantiate a dense SentenceTransformer from the backbone in config.

    The backbone is wrapped with an explicit ``Transformer`` + ``Pooling``
    stack so the pooling mode (mean/cls/max) is deterministic. ModernBERT has
    no native pooling head, so this must be specified. ``max_seq_length`` is
    set to the document length (queries are shorter and fit within it).
    """
    model_name = merged_cfg.get("model")
    if not model_name:
        raise ValueError("merged_cfg['model'] is required (backbone name).")
    transformer = st_models.Transformer(
        model_name_or_path=model_name,
        max_seq_length=stage_cfg.document_length,
    )
    pooling = st_models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode=stage_cfg.pooling,
    )
    return SentenceTransformer(modules=[transformer, pooling])


def build_loss(
    model: SentenceTransformer,
    stage_cfg: DenseStageConfig,
):
    """Build the in-batch-negatives loss (plain or cached/GradCache).

    ``MultipleNegativesRankingLoss`` uses the optional ``negative`` column as
    a hard negative alongside in-batch negatives. ``cached_mnrl`` encodes in
    ``mini_batch_size`` chunks but scores the full batch.
    """
    if stage_cfg.loss_type == "mnrl":
        return losses.MultipleNegativesRankingLoss(model)
    if stage_cfg.loss_type == "cached_mnrl":
        mini = stage_cfg.mini_batch_size or max(1, stage_cfg.per_device_train_batch_size // 4)
        return losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini)
    raise ValueError(f"Unknown loss_type: {stage_cfg.loss_type}")


def build_training_args(
    stage_cfg: DenseStageConfig,
    output_dir: Path | str,
    run_name: str,
    eval_strategy: str = "steps",
) -> SentenceTransformerTrainingArguments:
    """Build HF-style training arguments from a resolved stage config."""
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
    return SentenceTransformerTrainingArguments(**kwargs)


def build_evaluator(
    eval_dataset,
    name: str = "dev",
) -> evaluation.TripletEvaluator:
    """Build a triplet evaluator from a dev dataset with anchor/positive/negative."""
    return evaluation.TripletEvaluator(
        anchors=eval_dataset["anchor"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name=name,
    )


def build_trainer(
    model: SentenceTransformer,
    stage_cfg: DenseStageConfig,
    train_dataset,
    loss,
    output_dir: Path | str,
    run_name: str,
    eval_dataset=None,
    evaluator=None,
) -> SentenceTransformerTrainer:
    """Assemble the trainer.

    No custom collator is needed: ``SentenceTransformerTrainer`` defaults to
    the column-wise collator that routes ``anchor``/``positive``/``negative``
    dataset columns into the loss.
    """
    args = build_training_args(stage_cfg, output_dir, run_name)
    return SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )


def run_finetune(args) -> None:
    """Run the dense finetune end-to-end from a namespace of CLI args.

    ``args`` is any object with the attributes: ``train_config``,
    ``model_config``, ``stage``, ``config_name``, ``split``, ``max_examples``,
    ``max_steps``, ``batch_size``, ``mini_batch_size``, ``learning_rate``,
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
    if args.mini_batch_size is not None:
        stage.mini_batch_size = args.mini_batch_size
        # CachedMultipleNegativesRankingLoss is incompatible with gradient
        # accumulation; when mini_batch_size is set, force grad_accum=1 so the
        # configured accum does not silently inflate the logical batch.
        stage.gradient_accumulation_steps = 1
    if args.learning_rate is not None:
        stage.learning_rate = args.learning_rate
    if args.no_gradient_checkpointing:
        stage.gradient_checkpointing = False
    if args.save_strategy is not None:
        stage.save_strategy = args.save_strategy
    if args.save_total_limit is not None:
        stage.save_total_limit = args.save_total_limit

    if wandb_enabled():
        enable_wandb_reporting(stage)
        print(f"[dense] wandb reporting enabled "
              f"(project={os.environ.get('WANDB_PROJECT', '<unset>')})")
    else:
        print("[dense] wandb not enabled (set WANDB_API_KEY to enable)")

    print(f"[dense] loading {args.config_name}/{args.split} from HF...")
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
        print(f"[dense] filter report: {r.as_dict()}")
    train_ds = to_sbert_columns(train_ds)
    print(f"[dense] train rows: {len(train_ds)}")

    holdout = min(args.eval_holdout, max(1, len(train_ds) // 100))
    split = train_ds.train_test_split(test_size=holdout, seed=args.seed)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"[dense] eval holdout: {len(eval_ds)} rows")

    print(f"[dense] building dense SentenceTransformer from {cfg['model']}...")
    model = build_model(cfg, stage)

    print(f"[dense] loss={stage.loss_type} "
          f"batch={stage.per_device_train_batch_size} "
          f"mini={stage.mini_batch_size} effective={stage.effective_batch} "
          f"steps={stage.max_steps} lr={stage.learning_rate} "
          f"pooling={stage.pooling} save={stage.save_strategy} "
          f"limit={stage.save_total_limit}")
    loss = build_loss(model, stage)
    evaluator = build_evaluator(eval_ds, name="dense_eval")

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        cfg.get("outputs", {}).get("run_dir", "runs/dense_stage2")
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

    print("[dense] starting training...")
    trainer.train()

    if args.save_final:
        final_dir = output_dir / "final"
        model.save_pretrained(str(final_dir))
        print(f"[dense] done. final model saved to {final_dir}")
        print("[dense] NEXT: run a dense real-corpus retrieval eval against "
              "<final_dir> (extend scripts/eval/run_real_corpus.py for "
              "SentenceTransformer encode + dot-product top-k).")
    else:
        print("[dense] done.")
