#!/usr/bin/env bash
# push_checkpoints.sh — Push checkpoint directories to a Hugging Face model repo.
#
# Scans a local run directory for checkpoint-* subfolders and a final/ subfolder,
# then uploads them to the given HF repo.
#
# Usage:
#   bash scripts/release/push_checkpoints.sh --local-dir <PATH> --repo-id <REPO> [--prefix <PREFIX>]
#
# Examples:
#   # Push base model to the internal repo (final → repo root)
#   bash scripts/release/push_checkpoints.sh \
#     --local-dir runs/base_stage2 \
#     --repo-id fierysurf/MedColBERT-v1-alpha-internal
#
#   # Push large model under a subfolder
#   bash scripts/release/push_checkpoints.sh \
#     --local-dir runs/large_stage2 \
#     --repo-id fierysurf/MedColBERT-v1-alpha-internal \
#     --prefix large/
#
#   # Dry-run to see what would be uploaded
#   bash scripts/release/push_checkpoints.sh \
#     --local-dir runs/large_stage2 \
#     --repo-id fierysurf/MedColBERT-v1-alpha-internal \
#     --prefix large/ \
#     --dry-run
#
# Env:
#   HF_TOKEN   — required (Hugging Face token with write access)
#
# Flags:
#   --local-dir <PATH>   Path to the run directory containing checkpoints
#   --repo-id <REPO>     Destination Hugging Face repo
#   --prefix <PREFIX>    Subfolder prefix in the repo (e.g. "large/")
#   --dry-run            Preflight only, skip actual uploads
#
# The "Load model" hint is auto-detected from modules.json: ColBERT (pylate),
# dense (SentenceTransformer), or sparse (SparseEncoder).

set -euo pipefail

ts() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*"; }

# ─── Parse args ─────────────────────────────────────────────────────────────
LOCAL_DIR=""
REPO_ID=""
PREFIX=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local-dir) LOCAL_DIR="$2"; shift 2 ;;
    --repo-id)   REPO_ID="$2";   shift 2 ;;
    --prefix)    PREFIX="$2";    shift 2 ;;
    --dry-run)   DRY_RUN=1;      shift   ;;
    *)
      log "ERROR: unknown argument: $1"
      log "Usage: $0 --local-dir <PATH> --repo-id <REPO> [--prefix <PREFIX>] [--dry-run]"
      exit 1
      ;;
  esac
done

if [ -z "$LOCAL_DIR" ]; then
  log "ERROR: --local-dir is required"
  exit 1
fi

if [ -z "$REPO_ID" ]; then
  log "ERROR: --repo-id is required"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Normalise LOCAL_DIR to absolute
LOCAL_DIR="$(cd "$LOCAL_DIR" 2>/dev/null && pwd)" || {
  log "ERROR: --local-dir '$LOCAL_DIR' does not exist or is not accessible"
  exit 1
}

log "local-dir: $LOCAL_DIR"
log "repo-id:   $REPO_ID"
log "prefix:    '${PREFIX:-}(none)'"
log "dry-run:   $([ "$DRY_RUN" -eq 1 ] && echo yes || echo no)"

# ─── Auth ────────────────────────────────────────────────────────────────────
log "AUTH CHECK"

if [ -z "${HF_TOKEN:-}" ]; then
  log "ERROR: HF_TOKEN env var required"
  exit 1
fi

HF_USER="$(hf auth whoami --format quiet 2>&1)"
if [ -z "$HF_USER" ]; then
  log "ERROR: not logged in. Run: hf auth login"
  exit 1
fi
log "  logged in: $HF_USER"

# ─── Discover checkpoints ────────────────────────────────────────────────────
log "SCAN: $LOCAL_DIR"

declare -a CHECKPOINTS=()
FINAL_DIR=""

for item in "$LOCAL_DIR"/*/; do
  name="$(basename "$item")"
  if [[ "$name" == checkpoint-* ]]; then
    if [ -f "$item/config.json" ]; then
      sz=$(du -sh "$item" | cut -f1)
      log "  checkpoint: $name ($sz)"
      CHECKPOINTS+=("$name|$item")
    else
      log "  SKIP: $name (no config.json)"
    fi
  elif [ "$name" = "final" ]; then
    if [ -f "$item/config.json" ]; then
      sz=$(du -sh "$item" | cut -f1)
      log "  final: $item ($sz)"
      FINAL_DIR="$item"
    else
      log "  SKIP: final (no config.json)"
    fi
  fi
done

if [ -z "$FINAL_DIR" ] && [ ${#CHECKPOINTS[@]} -eq 0 ]; then
  log "ERROR: no checkpoints or final/ found in $LOCAL_DIR"
  exit 1
fi

# ─── Create private repo ─────────────────────────────────────────────────────
log "Creating private repo (if missing): $REPO_ID"
if [ "$DRY_RUN" -eq "0" ]; then
  hf repos create "$REPO_ID" --type model --private --exist-ok 2>&1 | sed 's/^/  /' || \
    log "  (repo may already exist; continuing)"
fi

# ─── Upload final → <prefix> (canonical weights) ─────────────────────────────
if [ -n "$FINAL_DIR" ]; then
  # If prefix is empty, we upload final to repo root (.)
  # If prefix is set, we upload final to that subfolder
  hf_dst="${PREFIX:-.}"
  log "UPLOAD: final → $REPO_ID:/$hf_dst"
  if [ "$DRY_RUN" -eq "0" ]; then
    hf upload "$REPO_ID" "$FINAL_DIR" "$hf_dst" \
      --type model \
      --commit-message "Upload final (canonical weights)" \
      2>&1 | sed 's/^/  /'
  else
    log "  [dry-run] would upload $FINAL_DIR → $hf_dst"
  fi
fi

# ─── Upload each checkpoint to <prefix>checkpoints/<name>/ ───────────────────
for entry in "${CHECKPOINTS[@]}"; do
  IFS='|' read -r name path <<< "$entry"
  dst="${PREFIX}checkpoints/$name/"
  log "UPLOAD: $name → $REPO_ID:/$dst"
  if [ "$DRY_RUN" -eq "0" ]; then
    hf upload "$REPO_ID" "$path" "$dst" \
      --type model \
      --commit-message "Upload checkpoint: $name" \
      2>&1 | sed 's/^/  /'
  else
    log "  [dry-run] would upload $path → $dst"
  fi
done

# ─── Detect architecture (for the load hint) ─────────────────────────────────
# modules.json is the SentenceTransformer/PyLate manifest; the module types
# distinguish the three paradigms:
#   SpladePooling / sparse_encoder        -> SparseEncoder
#   pylate.models.*                       -> pylate ColBERT
#   sentence_transformers.models.Pooling  -> SentenceTransformer (dense)
# Order matters: SpladePooling contains "Pooling", so check sparse first.
ARCH="colbert"
SAMPLE_DIR=""
if [ -n "$FINAL_DIR" ]; then
  SAMPLE_DIR="$FINAL_DIR"
elif [ ${#CHECKPOINTS[@]} -gt 0 ]; then
  SAMPLE_DIR="${CHECKPOINTS[0]#*|}"
fi
if [ -n "$SAMPLE_DIR" ] && [ -f "${SAMPLE_DIR%/}/modules.json" ]; then
  if grep -qE "SpladePooling|sparse_encoder" "${SAMPLE_DIR%/}/modules.json"; then
    ARCH="sparse"
  elif grep -q "pylate" "${SAMPLE_DIR%/}/modules.json"; then
    ARCH="colbert"
  elif grep -q "Pooling" "${SAMPLE_DIR%/}/modules.json"; then
    ARCH="dense"
  fi
fi
log "architecture: $ARCH"

# ─── Done ────────────────────────────────────────────────────────────────────
log ""
log "DONE. Repo: https://huggingface.co/$REPO_ID"
log ""
log "Load model ($ARCH):"
case "$ARCH" in
  colbert)
    if [ -n "$PREFIX" ]; then
      log "  from pylate import models"
      log "  model = models.ColBERT('$REPO_ID', subfolder='${PREFIX%/}')"
    else
      log "  from pylate import models"
      log "  model = models.ColBERT('$REPO_ID')"
    fi
    ;;
  sparse)
    if [ -n "$PREFIX" ]; then
      log "  from huggingface_hub import snapshot_download"
      log "  from sentence_transformers import SparseEncoder"
      log "  p = snapshot_download('$REPO_ID', allow_patterns='${PREFIX}*')"
      log "  model = SparseEncoder(p + '/${PREFIX%/}')"
    else
      log "  from sentence_transformers import SparseEncoder"
      log "  model = SparseEncoder('$REPO_ID')"
    fi
    ;;
  dense)
    if [ -n "$PREFIX" ]; then
      log "  from huggingface_hub import snapshot_download"
      log "  from sentence_transformers import SentenceTransformer"
      log "  p = snapshot_download('$REPO_ID', allow_patterns='${PREFIX}*')"
      log "  model = SentenceTransformer(p + '/${PREFIX%/}')"
    else
      log "  from sentence_transformers import SentenceTransformer"
      log "  model = SentenceTransformer('$REPO_ID')"
    fi
    ;;
esac

if [ ${#CHECKPOINTS[@]} -gt 0 ]; then
  first_ckpt="${CHECKPOINTS[0]%%|*}"
  log ""
  log "Load a specific checkpoint:"
  log "  from huggingface_hub import snapshot_download"
  case "$ARCH" in
    colbert)
      log "  from pylate import models"
      log "  p = snapshot_download('$REPO_ID', allow_patterns='${PREFIX}checkpoints/${first_ckpt}/*')"
      log "  model = models.ColBERT(p + '/${PREFIX}checkpoints/${first_ckpt}')"
      ;;
    sparse)
      log "  from sentence_transformers import SparseEncoder"
      log "  p = snapshot_download('$REPO_ID', allow_patterns='${PREFIX}checkpoints/${first_ckpt}/*')"
      log "  model = SparseEncoder(p + '/${PREFIX}checkpoints/${first_ckpt}')"
      ;;
    dense)
      log "  from sentence_transformers import SentenceTransformer"
      log "  p = snapshot_download('$REPO_ID', allow_patterns='${PREFIX}checkpoints/${first_ckpt}/*')"
      log "  model = SentenceTransformer(p + '/${PREFIX}checkpoints/${first_ckpt}')"
      ;;
  esac
fi
