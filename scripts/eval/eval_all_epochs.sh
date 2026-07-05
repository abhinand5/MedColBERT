#!/usr/bin/env bash
# eval_all_epochs.sh — Run full real-corpus retrieval eval on all saved
# MedColBERT checkpoints (epochs 1, 2, 4, 5, final) and write a comparison
# table. Every checkpoint gets its own PLAID index dir and metrics JSON.
#
# Usage:
#   bash scripts/eval/eval_all_epochs.sh
#
# Optional env overrides:
#   BATCH_SIZE   — encoding batch size (default 128)
#   PYTHON       — python binary (default .venv/bin/python)
#   HF_TOKEN     — HuggingFace token for private dev dataset
#
# Output:
#   logs/eval_all_epochs.log          — combined log for all runs
#   runs/base_stage2/eval_epoch{N}.json — per-checkpoint metrics
#   runs/base_stage2/eval_comparison.txt — final comparison table

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

BATCH_SIZE="${BATCH_SIZE:-128}"
PYTHON="${PYTHON:-.venv/bin/python}"
LOG="logs/eval_all_epochs.log"
METRICS_DIR="runs/base_stage2"
COMPARISON="$METRICS_DIR/eval_comparison.txt"

mkdir -p logs "$METRICS_DIR"

# Checkpoints: name | path | epoch label
# Epoch 3 was pruned by save_total_limit=3; user persisted epochs 1-2 externally.
CHECKPOINTS=(
  "epoch1|/workspace/models/checkpoint-2776"
  "epoch2|/workspace/models/checkpoint-5552"
  "epoch4|runs/base_stage2/checkpoint-11104"
  "epoch5|runs/base_stage2/checkpoint-13880"
  "final|runs/base_stage2/final"
)

CORPUS="data/processed/private/passages/real_annotated_500.json"
K=100

# ─── Helpers ─────────────────────────────────────────────────────────────────
ts() { date "+%Y-%m-%d %H:%M:%S"; }

log() { echo "[$(ts)] $*" | tee -a "$LOG"; }

section() {
  echo "" | tee -a "$LOG"
  echo "════════════════════════════════════════════════════════════════" | tee -a "$LOG"
  log "$1"
  echo "════════════════════════════════════════════════════════════════" | tee -a "$LOG"
}

# ─── Pre-flight checks ───────────────────────────────────────────────────────
section "PRE-FLIGHT CHECKS"

if [ ! -f "$PYTHON" ]; then
  log "ERROR: Python binary not found at $PYTHON"
  exit 1
fi

if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>>"$LOG"; then
  log "ERROR: CUDA not available. GPU required for full-corpus eval."
  exit 1
fi
log "GPU available: $("$PYTHON" -c "import torch; print(torch.cuda.get_device_name(0))")"

if [ ! -f "$CORPUS" ]; then
  log "ERROR: Corpus not found at $CORPUS"
  exit 1
fi
log "Corpus: $CORPUS"

MISSING=0
for entry in "${CHECKPOINTS[@]}"; do
  IFS='|' read -r name path <<< "$entry"
  if [ ! -f "$path/config.json" ]; then
    log "WARNING: checkpoint '$name' missing config.json at $path"
    MISSING=$((MISSING + 1))
  else
    log "  OK: $name → $path"
  fi
done

if [ "$MISSING" -eq "${#CHECKPOINTS[@]}" ]; then
  log "ERROR: No valid checkpoints found."
  exit 1
fi
log "Pre-flight passed ($(( ${#CHECKPOINTS[@]} - MISSING )) / ${#CHECKPOINTS[@]} checkpoints valid)"

# ─── Run eval per checkpoint ─────────────────────────────────────────────────
log ""
log "Starting eval loop: batch_size=$BATCH_SIZE, k=$K, corpus=66,108 passages, 5,967 queries"

for entry in "${CHECKPOINTS[@]}"; do
  IFS='|' read -r name ckpt_path <<< "$entry"

  if [ ! -f "$ckpt_path/config.json" ]; then
    log "SKIP: $name (missing config.json)"
    continue
  fi

  section "EVAL: $name ($ckpt_path)"

  index_dir="$METRICS_DIR/eval_index_${name}"
  metrics_json="$METRICS_DIR/eval_${name}.json"

  log "index_dir:  $index_dir"
  log "metrics:    $metrics_json"

  START=$(date +%s)

  "$PYTHON" scripts/eval/run_real_corpus.py \
    --model "$ckpt_path" \
    --corpus "$CORPUS" \
    --index-dir "$index_dir" \
    --index-name "real_corpus" \
    --k "$K" \
    --batch-size "$BATCH_SIZE" \
    --out "$metrics_json" \
    2>&1 | tee -a "$LOG"

  STATUS=${PIPESTATUS[0]}
  END=$(date +%s)
  ELAPSED=$((END - START))
  MINS=$((ELAPSED / 60))
  SECS=$((ELAPSED % 60))

  if [ "$STATUS" -ne 0 ]; then
    log "FAILED: $name (exit=$STATUS, ${MINS}m${SECS}s)"
  else
    log "DONE: $name (${MINS}m${SECS}s)"
    if [ -f "$metrics_json" ]; then
      log "  metrics: $(cat "$metrics_json" | tr -d '\n')"
    fi
  fi
done

# ─── Build comparison table ──────────────────────────────────────────────────
section "COMPARISON TABLE"

{
  echo "MedColBERT Real-Corpus Eval — $(ts)"
  echo "Corpus: 66,108 passages | Queries: 5,967 | k=$K"
  echo "BM25 baseline: Recall@100 ≈ 0.626"
  echo ""
  printf "%-10s  %8s  %8s  %8s  %8s\n" "checkpoint" "MRR@10" "R@10" "R@100" "nDCG@10"
  printf "%-10s  %8s  %8s  %8s  %8s\n" "----------" "--------" "------" "------" "--------"
  for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r name _ <<< "$entry"
    jf="$METRICS_DIR/eval_${name}.json"
    if [ -f "$jf" ]; then
      "$PYTHON" -c "
import json
d = json.load(open('$jf'))
print(f\"{'$name':<10}  {d['mrr_at_10']:8.4f}  {d['recall_at_10']:8.4f}  {d['recall_at_100']:8.4f}  {d['ndcg_at_10']:8.4f}\")
"
    else
      printf "%-10s  %8s  %8s  %8s  %8s\n" "$name" "—" "—" "—" "—"
    fi
  done
} | tee "$COMPARISON" | tee -a "$LOG"

log ""
log "Comparison table written to $COMPARISON"
log "Full log: $LOG"
log "All done."
