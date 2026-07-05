"""Tests for trainer config resolution (pure, no GPU/network)."""

from __future__ import annotations


from medcolbert.training.pylate_train import (
    build_training_args,
    enable_wandb_reporting,
    resolve_stage,
    wandb_enabled,
)


def _merged_cfg() -> dict:
    return {
        "model": "thomas-sounack/BioClinical-ModernBERT-base",
        "colbert_dim": 128,
        "query_maxlen": 64,
        "doc_maxlen": 256,
        "outputs": {"save_steps": 10000, "eval_steps": 5000},
        "stage2_colbert": {
            "learning_rate": 0.00001,
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 16,
            "bf16": True,
            "gradient_checkpointing": True,
            "gather_across_devices": True,
            "warmup_ratio": 0.03,
            "max_steps": 100000,
            "loss_type": "cached_contrastive",
            "mini_batch_size": 16,
        },
        "stage0_smoke": {
            "max_examples": 1000,
            "max_steps": 1000,
            "per_device_train_batch_size": 8,
        },
    }


def test_resolve_stage2_reads_block_and_forces_grad_accum_to_1():
    cfg = _merged_cfg()
    s = resolve_stage(cfg, "stage2_colbert")
    assert s.learning_rate == 1e-5
    assert s.per_device_train_batch_size == 16
    # cached_contrastive + mini_batch_size => grad_accum forced to 1
    assert s.gradient_accumulation_steps == 1
    assert s.loss_type == "cached_contrastive"
    assert s.mini_batch_size == 16
    assert s.max_steps == 100000
    assert s.query_length == 64
    assert s.document_length == 256
    assert s.effective_batch == 16  # per_device * grad_accum(1)


def test_resolve_stage0_smoke_defaults_loss_to_cached():
    cfg = _merged_cfg()
    s = resolve_stage(cfg, "stage0_smoke")
    assert s.max_steps == 1000
    assert s.per_device_train_batch_size == 8
    assert s.loss_type == "cached_contrastive"  # default
    # no mini_batch_size set => grad_accum stays at default 1
    assert s.gradient_accumulation_steps == 1
    # max_steps set => epochs left None (HF uses max_steps)
    assert s.num_train_epochs is None


def test_resolve_plain_contrastive_keeps_grad_accum():
    cfg = _merged_cfg()
    cfg["stage2_colbert"]["loss_type"] = "contrastive"
    cfg["stage2_colbert"]["mini_batch_size"] = None
    s = resolve_stage(cfg, "stage2_colbert")
    # plain contrastive keeps the configured accumulation (caller's responsibility
    # that it's actually compatible)
    assert s.gradient_accumulation_steps == 16
    assert s.effective_batch == 16 * 16


def test_resolve_unknown_stage_returns_defaults():
    s = resolve_stage({}, "stage_does_not_exist")
    assert s.per_device_train_batch_size == 16
    assert s.loss_type == "cached_contrastive"
    assert s.num_train_epochs == 1


def test_build_training_args_has_max_steps_when_set(tmp_path):
    cfg = _merged_cfg()
    s = resolve_stage(cfg, "stage2_colbert")
    args = build_training_args(s, tmp_path / "out", "run-x")
    assert args.max_steps == 100000
    assert args.per_device_train_batch_size == 16
    assert args.gradient_accumulation_steps == 1
    assert args.bf16 is True
    assert args.gradient_checkpointing is True
    assert args.learning_rate == 1e-5
    assert args.dataloader_drop_last is True
    assert args.save_strategy == "steps"
    assert args.save_total_limit == 3


def test_build_training_args_uses_epochs_when_no_max_steps(tmp_path):
    cfg = _merged_cfg()
    cfg["stage0_smoke"]["max_steps"] = None
    cfg["stage0_smoke"]["num_train_epochs"] = 2
    s = resolve_stage(cfg, "stage0_smoke")
    args = build_training_args(s, tmp_path / "out", "run-smoke")
    assert args.max_steps == -1  # HF sentinel for "not set"
    assert args.num_train_epochs == 2


def test_save_strategy_epoch():
    cfg = {"stage2_colbert": {"save_strategy": "epoch", "save_total_limit": 5}}
    s = resolve_stage(cfg, "stage2_colbert")
    assert s.save_strategy == "epoch"
    assert s.save_total_limit == 5
    args = build_training_args(s, "/tmp/out", "run-x")
    assert args.save_strategy == "epoch"
    assert args.save_total_limit == 5


def test_save_strategy_steps_passes_save_steps():
    cfg = {"stage2_colbert": {"save_strategy": "steps", "save_steps": 5000}}
    s = resolve_stage(cfg, "stage2_colbert")
    args = build_training_args(s, "/tmp/out", "run-x")
    assert args.save_strategy == "steps"
    assert args.save_steps == 5000


def test_default_report_to_is_none():
    s = resolve_stage({}, "stage_x")
    assert s.report_to == ["none"]


def test_report_to_from_config():
    cfg = {"stage2_colbert": {"report_to": ["wandb"]}}
    s = resolve_stage(cfg, "stage2_colbert")
    assert s.report_to == ["wandb"]
    args = build_training_args(s, "/tmp/out", "run-x")
    assert args.report_to == ["wandb"]


def test_enable_wandb_reporting_flips_when_key_set(monkeypatch):
    s = resolve_stage({}, "stage_x")
    assert s.report_to == ["none"]
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    assert wandb_enabled() is True
    enable_wandb_reporting(s)
    assert s.report_to == ["wandb"]


def test_enable_wandb_reporting_noop_without_key(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    s = resolve_stage({}, "stage_x")
    assert wandb_enabled() is False
    enable_wandb_reporting(s)
    assert s.report_to == ["none"]
