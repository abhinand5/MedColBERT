#!/usr/bin/env bash
# Regenerate abbreviation batches with fixed prep + forced-opener prompt.
# Usage: _resume_abbr_loop.sh <start> <count>
set -u
START="$1"
COUNT="$2"
cd /workspace/MedColBERT
OUT=/workspace/MedColBERT/data/processed/private/synthetic/v6_fresh
PASS=/workspace/MedColBERT/data/processed/private/passages/abbreviation_passages.json
LOG=/workspace/MedColBERT/data/processed/private/synthetic/v6_fresh/_abbr_loop.log
echo "ABBR LOOP start=$START count=$COUNT $(date -u +%FT%TZ)" >> "$LOG"
b=$START
n=0
while [ "$n" -lt "$COUNT" ]; do
  bid=$(printf "abbr_v6_%04d" "$b")
  seed=$((90000 + b))
  if [ -f "$OUT/$bid/stats.json" ]; then
    echo "SKIP $bid $(date -u +%T)" >> "$LOG"; b=$((b+1)); n=$((n+1)); continue
  fi
  rm -rf "$OUT/$bid"
  echo "RUN $bid seed=$seed $(date -u +%T)" >> "$LOG"
  uv run python /workspace/MedColBERT/scripts/data/multistyle_batch_v2.py \
    --seed "$seed" --batch-id "$bid" --count 50 \
    --judge-mode none \
    --passages "$PASS" \
    --output-base "$OUT" >> "$LOG" 2>&1
  rc=$?
  if [ ! -f "$OUT/$bid/stats.json" ]; then
    echo "FAILED $bid rc=$rc $(date -u +%T)" >> "$LOG"; rm -rf "$OUT/$bid"
  else
    acc=$(python3 -c "import json;print(json.load(open('$OUT/$bid/stats.json')).get('accepted',0))" 2>/dev/null || echo 0)
    echo "DONE $bid accepted=$acc rc=$rc $(date -u +%T)" >> "$LOG"
  fi
  b=$((b+1)); n=$((n+1))
done
echo "ABBR LOOP END $(date -u +%FT%TZ)" >> "$LOG"
