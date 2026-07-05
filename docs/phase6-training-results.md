# Phase 6 — Training Results (medcolbert-base-run0)

## Run

| Field | Value |
|---|---|
| Run id | `medcolbert-base-run0` |
| Wandb | https://wandb.ai/abhinandb/MedColBERT/runs/zhepk234 |
| Branch | `dev` |
| Backbone | `thomas-sounack/BioClinical-ModernBERT-base` |
| Training data | `fierysurf/medcolbert-training-v1`, `long` config (711,301 rows, 4,273 CUIs) |
| Loss | `CachedContrastive` (GradCache), temperature 0.02 |
| Batch | logical 256, mini 160, gradient_accumulation_steps 1 |
| Steps / epochs | 14,000 / ~5.04 |
| LR | 1e-5 linear, bf16, no gradient checkpointing |
| Save | `save_strategy=epoch`, `save_total_limit=3` |
| Hardware | single L40S (48 GB), ~29.5 GB VRAM, 100% util |
| Runtime | 14h40m, 67.84 samples/s |
| Final `train_loss` | 0.9872 (warmup-weighted average) |

## Loss Trajectory (per epoch, from `trainer_state`)

| Epoch | Avg Loss | Min | Max |
|---|---|---|---|
| 0 | 4.8708 | 0.087 | 193.7 |
| 1 | 0.0609 | 0.034 | 0.24 |
| 2 | 0.0302 | 0.021 | 0.044 |
| 3 | 0.0232 | 0.016 | 0.035 |
| 4 | 0.0206 | 0.012 | 0.037 |
| 5 | 0.0202 | 0.016 | 0.024 |

Loss plateaus at epoch 3-4. Grad-norm spikes recovered within a few
steps; no divergence.

## Checkpoints

`save_total_limit=3` kept the last three epoch boundaries on disk:

| Checkpoint | Epoch | Location |
|---|---|---|
| checkpoint-2776 | 1 | `/workspace/models/checkpoint-2776` (user-persisted) |
| checkpoint-5552 | 2 | `/workspace/models/checkpoint-5552` (user-persisted) |
| checkpoint-8328 | 3 | pruned (save_total_limit) |
| checkpoint-11104 | 4 | `runs/base_stage2/checkpoint-11104` |
| checkpoint-13880 | 5 | `runs/base_stage2/checkpoint-13880` |
| final | 5.04 | `runs/base_stage2/final` |

## Real-Corpus Eval — Full 66,108-passage PubMed Corpus / 5,967 Dev Queries

Gate: beat BM25 Recall@100 ≈ 0.626.

| Checkpoint | MRR@10 | R@10 | R@100 | nDCG@10 |
|---|---|---|---|---|
| **BM25 baseline** | — | — | **0.626** | — |
| epoch1 | 0.66895 | 0.84733 | 0.95743 | 0.71200 |
| epoch2 | 0.69279 | 0.86409 | 0.96581 | 0.73439 |
| epoch4 | 0.71525 | 0.87615 | 0.97017 | 0.75441 |
| epoch5 | 0.71384 | 0.87649 | 0.97117 | 0.75341 |
| **final** | **0.71430** | **0.87766** | **0.97117** | 0.75396 |

Saved metrics: `runs/base_stage2/eval_epoch{1,2,4,5}.json`,
`runs/base_stage2/eval_final.json`, and the comparison table at
`runs/base_stage2/eval_comparison.txt`. Per-checkpoint PLAID indexes at
`runs/base_stage2/eval_index_epoch{1,2,4,5,final}/`.

## Conclusions

- **Gate passed.** Recall@100 = 0.971 vs BM25 0.626 (+55% relative).
  Ontology-controlled synthetic supervision generalizes to real PubMed
  retrieval — the model did not just memorize the synthetic negatives.
- **Best checkpoint:** `final` and `epoch4` are statistically tied
  (differences <0.002 on 5,967 queries). `final` is the canonical
  "last token" checkpoint and is selected as `medcolbert-base-v1`.
- **Diminishing returns after epoch 2.** Largest jumps are 1→2 (MRR
  +0.024) and 2→4 (MRR +0.022). Epoch 4→5 adds <0.001 across all
  metrics — the LR schedule tail (5e-8) cannot move the model.
- **No overfitting signal.** Metrics keep improving slightly even as
  loss flattens, and epoch 5 does not degrade. The 5-epoch length was
  safe; 4 epochs would also have been sufficient.

## Decisions

- `--mini-batch-size` (CLI) forces `gradient_accumulation_steps=1`. The
  cached contrastive loss is incompatible with gradient accumulation;
  without this override the config's `gradient_accumulation_steps=16`
  silently inflated the logical batch from 256 to 4096.
- `--no-gradient-checkpointing` was used on the 48 GB L40S. This raised
  VRAM from ~7 GB to ~29.5 GB and roughly halved per-step time.
- `save_strategy=epoch` (not steps) so each epoch boundary is a clean
  comparable checkpoint.
- In-batch same-CUI / same-positive collisions (~46 / ~2.5 per batch)
  were accepted for this run. A dedup-by-positive sampler is deferred to
  a v2 run only if a later gate underperforms.

## Reproduction

```bash
# Training (single L40S)
WANDB_API_KEY=... WANDB_PROJECT=MedColBERT \
.venv/bin/python scripts/training/stage2_colbert_finetune.py \
  --config configs/train_base.yaml \
  --config-name stage2_colbert \
  --batch-size 256 \
  --mini-batch-size 160 \
  --no-gradient-checkpointing \
  --max-steps 14000 \
  --save-strategy epoch \
  --save-total-limit 3

# Eval (all checkpoints)
bash scripts/eval/eval_all_epochs.sh
```

## What's Not Done

- Required baselines (Phase 7): especially the
  `bioClinical-no-ontology` ColBERT control — the mandatory ablation
  that isolates the value of ontology-controlled supervision from the
  value of the backbone. See
  [docs/goals/07-evaluation-and-claims.md](goals/07-evaluation-and-claims.md).
- vocab-shift slice eval (lay→clinical, abbreviation→expanded,
  brand→generic, symptom→diagnosis, biomedical→clinical).
- Model card, provenance summary, license/memorization review (Phase 8).
- Decontamination report for the eval corpus against the training set.

This run clears the **Training gate** (Phase 6). The **Evaluation gate**
(Phase 7) remains pending — passing the real-corpus retrieval gate is
necessary but not sufficient for the project's research claim.