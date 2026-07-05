# MedColBERT-large Training Plan

**Status:** awaiting approval
**Backbone:** `thomas-sounack/BioClinical-ModernBERT-large` (396M params)
**Hardware:** single NVIDIA L40S (48 GB)
**Goal:** train `medcolbert-large-v1-alpha` on the same v1 ontology-controlled
triplets and run the real-corpus gate, mirroring the base run so the two are
directly comparable ablations.

## 1. What the base run told us

The base run (`medcolbert-base-run0`, BioClinical-ModernBERT-base, 150M) is our
only empirical anchor. Key observations that drive the large hyperparameters:

| Observation | Implication for large |
|---|---|
| Loss plateaued at epoch 3-4 (avg 0.023 → 0.020); epoch 5 added <0.001 retrieval gain | 5 epochs is enough; no need to go longer. Keep `max_steps=14000`. |
| `batch=256` with GradCache gave healthy in-batch signal; ~46 same-CUI / 2.5 same-positive collisions per batch, 99.86% genuine negatives | Keep `batch=256`. The stronger backbone should not need a larger pool. |
| `mini_batch_size=160` (no GC) used 29.5 GB / 48 GB | Large's model+optimizer is ~4 GB heavier; same mini-batch won't fit without GC. Must drop `mini_batch_size` or enable GC. |
| `lr=1e-5` was stable; one warmup spike to loss 193 recovered in 1 step | Large is more instability-prone — lower to `5e-6` (ModernBERT-large convention for fine-tuning). |
| `temperature=0.02` stable; gets absorbed into the learned projection scale | Keep `0.02`. |
| `bf16` clean, no overflow | Keep. |
| `gradient_checkpointing=False` halved step time vs the default config | Prefer keeping GC off if memory allows; large's extra optimizer state eats into the budget. |
| Smoke test (1000 steps) loss 62.48 → 0.22 was the right gate before the full run | Run an identical smoke test on large before committing to 14k steps. |

## 2. Backbone comparison

| | base | large | ratio |
|---|---|---|---|
| Parameters | 150M | 396M | 2.64× |
| Hidden | 768 | 1024 | 1.33× |
| Layers | 22 | 28 | 1.27× |
| Heads | 12 | 16 | 1.33× |
| Intermediate | 1152 | 2624 | 2.28× |
| Vocab | 50368 | 50368 | 1.00× |
| Max pos | 8192 | 8192 | 1.00× |

Per-token forward+backward FLOPs scale roughly with `hidden × layers`
→ 1.33 × 1.27 ≈ **1.69× base compute**.

## 3. Memory budget (L40S 48 GB)

Static (always resident, bf16 weights + fp32 AdamW):
- base: ~3.0 GB (weights 0.3 GB bf16 + AdamW states 2.4 GB fp32)
- large: ~6.3 GB (weights 0.8 GB bf16 + AdamW states 4.7 GB fp32)

Activation peak is dominated by `mini_batch_size × seq_len × hidden × layers`
for the largest of {query, positive, negative} encodings inside GradCache. Doc
side uses `doc_maxlen=256`; query side uses `query_maxlen=64`. Negatives are
docs, so the doc encoder is the peak.

Base run, `mini=160`, no GC: 29.5 GB total ⇒ ~26.5 GB activation peak.
For large at the same mini-batch, activation scales ~1.69× ⇒ ~45 GB peak +
6.3 GB static = **~51 GB** — does not fit.

Two viable configs:

| Config | `mini_batch_size` | GC | Est. VRAM | Est. step time |
|---|---|---|---|---|
| **A (recommended)** | 80 | off | ~22 GB | ~6.0 s |
| B (fallback) | 128 | on | ~25 GB | ~8.5 s |

Config A keeps GC off (no recomputation), drops `mini_batch_size` to 80 so the
activation peak is roughly halved vs the base run. Static ~6.3 GB + peak
~16 GB = ~22 GB, leaving generous headroom on the 48 GB card. GradCache still
gives us the full `batch=256` loss; only the chunking granularity changes.

Config B is the fallback if A shows any OOM during the smoke test. GC trades
~30% extra compute for ~40% less activation memory.

## 4. Proposed hyperparameters

```yaml
# configs/train_large.yaml
model_config: configs/large.yaml
framework: pylate

stage0_smoke:
  enabled: true
  max_examples: 1000
  max_steps: 1000
  learning_rate: 5e-6
  per_device_train_batch_size: 64
  mini_batch_size: 32
  bf16: true
  gradient_checkpointing: false
  temperature: 0.02
  warmup_ratio: 0.03

stage2_colbert:
  enabled: true
  train_path: data/processed/private/triplets_v1.parquet
  eval_path: data/processed/private/triplets_dev.parquet
  max_steps: 14000           # ~5 epochs, same as base
  learning_rate: 5e-6         # half of base; larger models prefer lower LR
  warmup_ratio: 0.03
  per_device_train_batch_size: 256   # in-batch pool unchanged
  mini_batch_size: 80         # GradCache chunk; tuned for 48 GB without GC
  gradient_accumulation_steps: 1     # forced by cached loss
  bf16: true
  gradient_checkpointing: false
  gather_across_devices: false   # single GPU
  temperature: 0.02           # ColBERTv2 convention
  save_strategy: epoch
  save_total_limit: 3

outputs:
  checkpoint_dir: runs/large_stage2
  run_dir: runs/large_stage2
  save_steps: 10000
  eval_steps: 5000
```

### Rationale per knob

- **`learning_rate=5e-6`** (base used `1e-5`). Larger models fine-tune with
  lower LR; the ModernBERT-large recipe uses `5e-5` for continued pretraining
  and lower for downstream. Half the base LR is conservative and almost
  certainly safe; the smoke test will confirm before the long run.
- **`per_device_train_batch_size=256`** — same as base. Keeps the in-batch
  negative pool comparable so any retrieval gain is attributable to backbone
  capacity, not batch size. GradCache means this is the *logical* batch
  (contrastive loss sees all 256 negatives); `mini_batch_size` only controls
  VRAM chunking.
- **`mini_batch_size=80`** — chosen so the activation peak fits in 48 GB
  without gradient checkpointing. Drops from base's 160 in proportion to the
  ~1.7× compute increase (160 / 1.7 ≈ 94; round to 80 for safety margin).
- **`gradient_checkpointing=false`** — the L40S has 48 GB; large's static
  cost is 6.3 GB and post-chunking activations at mini=80 are ~16 GB,
  leaving ~25 GB headroom. Turning GC off saves ~30% wall-clock. If the
  smoke test OOMs, flip to fallback Config B (GC on, mini=128).
- **`max_steps=14000`** — same as base (~5 epochs over 711k rows at
  batch=256 ⇒ 2777 steps/epoch × 5 = 13,885). The base plateaued at epoch 3-4;
  the large model is likely to plateau at the same point or earlier. Keeping
  the same step budget makes the epoch-by-epoch comparison meaningful.
- **`save_strategy=epoch`, `save_total_limit=3`** — same as base. Kept the
  last 3 epoch boundaries; user persisted the first two externally for
  base. We'll do the same.
- **`temperature=0.02`** — unchanged. Smoke test on base confirmed this was
  stable; gets absorbed into the learned projection scale.
- **`warmup_ratio=0.03`** — unchanged (3% of 14000 = 420 steps).

## 5. Smoke test protocol (mandatory before full run)

```bash
uv run python scripts/training/stage0_smoke.py \
  --train-config configs/train_large.yaml \
  --model-config configs/large.yaml \
  --max-examples 1000 \
  --max-steps 1000
```

Gate criteria (same as base smoke):
1. No NaN/Inf in loss.
2. Loss decreases by ≥100× from peak (base: 62.48 → 0.22, ratio 285×).
3. Peak VRAM ≤ 40 GB (leaves 8 GB safety margin).
4. No CUDA OOM.
5. Step time ≤ 8 s (so 14k steps ≤ 31 h worst-case).

If smoke passes at `mini_batch_size=80`, proceed to full run.
If smoke OOMs, switch to fallback Config B (GC on, `mini_batch_size=128`)
and re-run smoke.
If loss diverges (spike does not recover within 20 steps), lower LR to
`3e-6` and re-run.

## 6. Full run command (proposed, after smoke passes)

```bash
WANDB_API_KEY=... WANDB_PROJECT=MedColBERT \
.venv/bin/python scripts/training/stage2_colbert_finetune.py \
  --train-config configs/train_large.yaml \
  --model-config configs/large.yaml \
  --stage stage2_colbert \
  --batch-size 256 \
  --mini-batch-size 80 \
  --no-gradient-checkpointing \
  --max-steps 14000 \
  --save-strategy epoch \
  --save-total-limit 3 \
  --output-dir runs/large_stage2 \
  --run-name medcolbert-large-run0
```

Estimated runtime: base was 14h40m at 3.57 s/step. Large at 1.7× compute
≈ 6 s/step ⇒ 14000 × 6 = **~23 h**. With GC fallback it would be ~33 h.

## 7. Eval plan (after training)

Re-run the same `scripts/eval/eval_all_epochs.sh` against
`runs/large_stage2/checkpoint-*` and `runs/large_stage2/final`. To compare
base vs large directly:

```bash
# Append large rows to the comparison table
for ckpt in epoch1 epoch2 epoch4 epoch5 final; do
  cp runs/large_stage2/eval_${ckpt}.json runs/base_stage2/eval_large_${ckpt}.json
done
```

Gate: MedColBERT-large must beat BM25 (Recall@100 ≥ 0.626 — already
trivially cleared by base) **and** match or beat MedColBERT-base on every
metric. If large underperforms base, the synthetic signal has saturated the
backbone and the project's value claim shifts to "the
ontology-controlled recipe transfers across backbones" rather than "larger
backbones exploit it more" — still a publishable result, but the framing
changes.

## 8. What we are NOT changing vs base

To keep this a clean backbone ablation:
- Same training data (`fierysurf/medcolbert-training-v1`, `long` config, 711k rows).
- Same loss (`CachedContrastive`), temperature, batch size.
- Same epoch budget (5), save strategy, and total limit.
- Same eval corpus and dev queries.
- Same eval script and metrics.

The only deliberate differences are `model`, `learning_rate` (halved), and
`mini_batch_size` (memory-compensating). The LR change is the one knob that
could confound the comparison; we accept the risk because a `1e-5` LR on a
396M model is likely to spike, and a divergence is worse than a slightly
conservative LR for a controlled ablation.

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| OOM at `mini=80` no-GC | Low | Fallback to Config B (GC on, mini=128). Smoke test catches this before the long run. |
| Loss diverges at `lr=5e-6` | Low | Smoke test gates this. Fallback: `lr=3e-6`. |
| Large underperforms base | Medium | Not a project risk — still publishable as a backbone-saturation finding. Decision deferred to eval results. |
| Run takes >24 h | Medium | Acceptable. If GPU contention, can pause via checkpoint resume (HF Trainer supports). |
| Epoch boundary mismatch with base (2777 vs ~2777 steps/epoch) | None | Same dataset, same batch ⇒ identical epoch boundaries. |

## 10. Artifacts

After approval and successful run:
- `runs/large_stage2/checkpoint-{2776,5552,11104,13880}/` — epoch boundaries
- `runs/large_stage2/final/` — canonical `medcolbert-large-v1-alpha`
- `runs/large_stage2/eval_epoch{N}.json` + `eval_comparison.txt`
- Update `docs/phase6-training-results.md` with a base-vs-large table
- Push to `fierysurf/MedColBERT-v1-alpha-internal` under a `large/` prefix
  (script will need a small path tweak — defer until run succeeds)

## 11. No new training script needed

`scripts/training/stage2_colbert_finetune.py` already accepts
`--model-config` and `--train-config`. The only new artifacts are
`configs/train_large.yaml` and this plan. The CLI overrides
(`--batch-size`, `--mini-batch-size`, `--no-gradient-checkpointing`,
`--save-strategy`, `--save-total-limit`, `--output-dir`, `--run-name`) cover
everything else.