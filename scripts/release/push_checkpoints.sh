#!/usr/bin/env bash
# push_checkpoints.sh — Push all MedColBERT-v1-alpha checkpoints to a private
# Hugging Face model repo for internal provenance.
#
# Repo layout after upload:
#   /                                ← final (canonical v1-alpha weights)
#   /checkpoints/checkpoint-2776/    ← epoch 1 (full state incl. optimizer)
#   /checkpoints/checkpoint-5552/    ← epoch 2
#   /checkpoints/checkpoint-11104/   ← epoch 4
#   /checkpoints/checkpoint-13880/   ← epoch 5
#   /checkpoints/checkpoint-14000/   ← epoch 5 (final step, with optimizer)
#
# Each checkpoint folder includes optimizer.pt + scheduler.pt + rng_state.pth
# (~1.2 GB extra) so training can resume from any boundary. Total ~10 GB.
#
# Usage:
#   bash scripts/release/push_checkpoints.sh
#
# Env:
#   HF_TOKEN   — required (Hugging Face token with write access)
#   REPO_ID    — destination repo (default fierysurf/MedColBERT-v1-alpha-internal)
#   DRY_RUN    — set to 1 to skip actual uploads (preflight only)

set -euo pipefail

REPO_ID="${REPO_ID:-fierysurf/MedColBERT-v1-alpha-internal}"
DRY_RUN="${DRY_RUN:-0}"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

# ─── Pre-flight ───────────────────────────────────────────────────────────────
log "PRE-FLIGHT"

if [ -z "${HF_TOKEN:-}" ]; then
  log "ERROR: HF_TOKEN env var required"
  exit 1
fi

if ! hf auth whoami --format quiet 2>&1 | grep -q "fierysurf"; then
  log "ERROR: not logged in as fierysurf. Run: hf auth login"
  exit 1
fi
log "  logged in: $(hf auth whoami --format quiet 2>&1)"

declare -a SOURCES=(
  "checkpoint-2776|/workspace/models/checkpoint-2776|epoch 1"
  "checkpoint-5552|/workspace/models/checkpoint-5552|epoch 2"
  "checkpoint-11104|runs/base_stage2/checkpoint-11104|epoch 4"
  "checkpoint-13880|runs/base_stage2/checkpoint-13880|epoch 5"
  "checkpoint-14000|runs/base_stage2/checkpoint-14000|epoch 5 (final step)"
  "final|runs/base_stage2/final|canonical v1-alpha"
)

MISSING=0
for entry in "${SOURCES[@]}"; do
  IFS='|' read -r name path label <<< "$entry"
  if [ ! -f "$path/config.json" ]; then
    log "  MISSING: $name at $path"
    MISSING=$((MISSING + 1))
  else
    sz=$(du -sh "$path" | cut -f1)
    log "  OK: $name ($sz, $label) ← $path"
  fi
done

if [ "$MISSING" -gt 0 ]; then
  log "ERROR: $MISSING source checkpoint(s) missing"
  exit 1
fi
log "  all ${#SOURCES[@]} sources present"

# ─── Create private repo ─────────────────────────────────────────────────────
log "Creating private repo (if missing): $REPO_ID"
if [ "$DRY_RUN" -eq "0" ]; then
  hf repos create "$REPO_ID" --type model --private --exist-ok 2>&1 | sed 's/^/  /' || \
    log "  (repo may already exist; continuing)"
fi

# ─── Upload final to repo root (canonical weights) ──────────────────────────
log "UPLOAD: final → $REPO_ID:/ (canonical v1-alpha)"
if [ "$DRY_RUN" -eq "0" ]; then
  hf upload "$REPO_ID" "runs/base_stage2/final" "." \
    --type model \
    --commit-message "Upload MedColBERT-v1-alpha (final, canonical weights)" \
    --commit-description "BioClinical-ModernBERT + ColBERT, 5 epochs CachedContrastive on 711k ontology-controlled triplets. Recall@100=0.971 vs BM25 0.626 on PubMed 66k corpus." \
    2>&1 | sed 's/^/  /'
else
  log "  [dry-run] would upload runs/base_stage2/final to repo root"
fi

# ─── Upload each checkpoint folder to /checkpoints/<name>/ ────────────────────
for entry in "${SOURCES[@]}"; do
  IFS='|' read -r name path label <<< "$entry"
  [ "$name" = "final" ] && continue
  dst="checkpoints/$name/"
  log "UPLOAD: $name ($label) → $REPO_ID:/$dst"
  if [ "$DRY_RUN" -eq "0" ]; then
    hf upload "$REPO_ID" "$path" "$dst" \
      --type model \
      --commit-message "Upload checkpoint: $name ($label)" \
      2>&1 | sed 's/^/  /'
  else
    log "  [dry-run] would upload $path to $dst"
  fi
done

# ─── Done ────────────────────────────────────────────────────────────────────
log ""
log "DONE. Repo: https://huggingface.co/$REPO_ID"
log "Layout:"
log "  /                                ← final (canonical v1-alpha)"
log "  /checkpoints/checkpoint-2776/    ← epoch 1"
log "  /checkpoints/checkpoint-5552/    ← epoch 2"
log "  /checkpoints/checkpoint-11104/   ← epoch 4"
log "  /checkpoints/checkpoint-13880/   ← epoch 5"
log "  /checkpoints/checkpoint-14000/   ← epoch 5 (final step, with optimizer)"
log ""
log "Load canonical model:"
log "  from pylate import models"
log "  model = models.ColBERT('$REPO_ID')"
log ""
log "Load a specific checkpoint:"
log "  from huggingface_hub import snapshot_download"
log "  path = snapshot_download('$REPO_ID', allow_patterns='checkpoints/checkpoint-2776/*')"
log "  model = models.ColBERT(path + '/checkpoints/checkpoint-2776')"