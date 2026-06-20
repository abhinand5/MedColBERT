#!/usr/bin/env bash
# Resume diverse generation in v6_fresh. Usage: _resume_div_loop.sh <start_batch> <stride> <max_batches>
set -u
START="$1"
STRIDE="$2"
MAXB="$3"
cd /workspace/MedColBERT
OUT=/workspace/MedColBERT/data/processed/private/synthetic/v6_fresh
LOG=/workspace/MedColBERT/data/processed/private/synthetic/v6_fresh/_loop_${START}.log
echo "LOOP start=$START stride=$STRIDE maxb=$MAXB $(date -u +%FT%TZ)" >> "$LOG"
b=$START
n=0
while [ "$n" -lt "$MAXB" ]; do
  bid=$(printf "div_v6_%04d" "$b")
  seed=$((80000 + b))
  if [ -f "$OUT/$bid/stats.json" ]; then
    echo "SKIP $bid (already complete) $(date -u +%T)" >> "$LOG"
    b=$((b + STRIDE)); n=$((n + 1)); continue
  fi
  # clean any incomplete dir for this batch
  rm -rf "$OUT/$bid"
  echo "RUN $bid seed=$seed $(date -u +%T)" >> "$LOG"
  uv run python /workspace/MedColBERT/scripts/data/multistyle_batch_v2.py \
    --seed "$seed" --batch-id "$bid" --count 100 \
    --judge-mode none \
    --output-base "$OUT" >> "$LOG" 2>&1
  rc=$?
  if [ ! -f "$OUT/$bid/stats.json" ]; then
    echo "FAILED $bid rc=$rc (no stats.json) $(date -u +%T)" >> "$LOG"
    rm -rf "$OUT/$bid"
  else
    acc=$(python -c "import json;print(json.load(open('$OUT/$bid/stats.json')).get('accepted',0))" 2>/dev/null || echo 0)
    echo "DONE $bid accepted=$acc rc=$rc $(date -u +%T)" >> "$LOG"
  fi
  b=$((b + STRIDE))
  n=$((n + 1))
done
echo "LOOP END start=$START $(date -u +%FT%TZ)" >> "$LOG"
