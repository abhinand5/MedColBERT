"""Construction of PyLate ColBERT model, loss, and HF/sentence-transformers trainer.

The training loop is delegated to ``SentenceTransformerTrainer`` (from
sentence-transformers, which wraps HuggingFace ``Trainer``). This module only
assembles the pieces from config dicts so scripts stay thin.

Key constraint: PyLate's contrastive loss is **not compatible with gradient
accumulation** (in-batch negatives require the full batch in one backward). To
emulate a large effective batch on a single GPU we use
``losses.CachedContrastive`` (GradCache), which encodes in
``mini_batch_size`` chunks but computes the loss over the full
``per_device_train_batch_size``. ``gradient_accumulation_steps`` is forced to 1
when the cached loss is used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils


def wandb_enabled() -> bool:
    """True if wandb should report (WANDB_API_KEY set and wandb importable)."""
    return bool(os.environ.get("WANDB_API_KEY"))


def enable_wandb_reporting(stage_cfg: StageConfig) -> StageConfig:
    """Flip ``report_to`` to wandb if WANDB_API_KEY is set; else leave as-is.

    The wandb project is read from ``WANDB_PROJECT`` by the HF Trainer. The
    run name is the stage's ``run_name`` passed to ``build_trainer``.
    """
    if wandb_enabled():
        stage_cfg.report_to = ["wandb"]
    return stage_cfg


@dataclass
class StageConfig:
    """Resolved hyperparameters for one training stage."""

    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    bf16: bool
    fp16: bool
    max_steps: int | None
    num_train_epochs: float | None
    warmup_ratio: float
    gradient_checkpointing: bool
    gather_across_devices: bool
    temperature: float
    loss_type: str  # "contrastive" | "cached_contrastive"
    mini_batch_size: int | None
    query_length: int
    document_length: int
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
) -> StageConfig:
    """Pull a stage block out of the merged config and normalise it.

    ``stage`` is a key like ``"stage0_smoke"`` or ``"stage2_colbert"``. Falls
    back to base-model lengths for query/document when the stage block does
    not override them.
    """
    block = merged_cfg.get(stage, {}) or {}
    base = merged_cfg

    max_steps = block.get("max_steps")
    num_train_epochs = block.get("num_train_epochs")
    # If neither max_steps nor epochs set, default to 1 epoch.
    if max_steps is None and num_train_epochs is None:
        num_train_epochs = 1

    loss_type = block.get("loss_type", "cached_contrastive")
    mini_batch_size = block.get("mini_batch_size")

    # When using the cached (GradCache) loss, gradient accumulation must be 1.
    grad_accum = block.get("gradient_accumulation_steps", 1)
    if loss_type == "cached_contrastive" and mini_batch_size is not None:
        grad_accum = 1

    outputs = merged_cfg.get("outputs", {}) or {}
    report_to = block.get("report_to", ["none"])
    return StageConfig(
        learning_rate=block.get("learning_rate", 1e-5),
        per_device_train_batch_size=block.get("per_device_train_batch_size", 16),
        gradient_accumulation_steps=grad_accum,
        bf16=block.get("bf16", True),
        fp16=block.get("fp16", False),
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        warmup_ratio=block.get("warmup_ratio", 0.0),
        gradient_checkpointing=block.get("gradient_checkpointing", False),
        gather_across_devices=block.get("gather_across_devices", False),
        temperature=block.get("temperature", 0.02),
        loss_type=loss_type,
        mini_batch_size=mini_batch_size,
        query_length=block.get("query_length", base.get("query_maxlen", 64)),
        document_length=block.get("document_length", base.get("doc_maxlen", 256)),
        save_steps=block.get("save_steps", outputs.get("save_steps", 10000)),
        eval_steps=block.get("eval_steps", outputs.get("eval_steps", 5000)),
        logging_steps=block.get("logging_steps", 20),
        report_to=report_to,
        save_strategy=block.get("save_strategy", "steps"),
        save_total_limit=block.get("save_total_limit", 3),
    )


def build_model(merged_cfg: dict[str, Any], stage_cfg: StageConfig) -> models.ColBERT:
    """Instantiate the ColBERT model from the backbone in config.

    ``embedding_size`` is the ColBERT projection dimension (``colbert_dim``).
    ``query_length`` / ``document_length`` set the max token lengths.
    """
    model_name = merged_cfg.get("model")
    if not model_name:
        raise ValueError("merged_cfg['model'] is required (backbone name).")
    return models.ColBERT(
        model_name_or_path=model_name,
        embedding_size=merged_cfg.get("colbert_dim"),
        query_length=stage_cfg.query_length,
        document_length=stage_cfg.document_length,
        trust_remote_code=True,
    )


def build_loss(
    model: models.ColBERT,
    stage_cfg: StageConfig,
):
    """Build the contrastive loss (plain or cached/GradCache).

    ``gather_across_devices`` should be False on a single GPU.
    """
    common = dict(
        model=model,
        temperature=stage_cfg.temperature,
        gather_across_devices=stage_cfg.gather_across_devices,
    )
    if stage_cfg.loss_type == "contrastive":
        return losses.Contrastive(**common)
    if stage_cfg.loss_type == "cached_contrastive":
        mini = stage_cfg.mini_batch_size or max(1, stage_cfg.per_device_train_batch_size // 4)
        return losses.CachedContrastive(
            mini_batch_size=mini,
            show_progress_bar=False,
            **common,
        )
    raise ValueError(f"Unknown loss_type: {stage_cfg.loss_type}")


def build_training_args(
    stage_cfg: StageConfig,
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
) -> evaluation.ColBERTTripletEvaluator:
    """Build a triplet evaluator from a dev dataset with query/positive/negative."""
    return evaluation.ColBERTTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        name=name,
    )


def build_trainer(
    model: models.ColBERT,
    stage_cfg: StageConfig,
    train_dataset,
    loss,
    output_dir: Path | str,
    run_name: str,
    eval_dataset=None,
    evaluator=None,
) -> SentenceTransformerTrainer:
    """Assemble the trainer with the ColBERT collator."""
    args = build_training_args(stage_cfg, output_dir, run_name)
    collator = utils.ColBERTCollator(tokenize_fn=model.tokenize)
    return SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
        data_collator=collator,
    )
