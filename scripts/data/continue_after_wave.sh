#!/bin/bash
# Wait for current concurrent batches (conc_001-003) to complete,
# consolidate results, then start auto_scale.sh for continuous operation.
set -euo pipefail

BASE_DIR="/workspace/MedColBERT"
OUTPUT_DIR="${BASE_DIR}/data/processed/private/synthetic/ontology_grounded_teacher_filtered"

echo "=== MedColBERT Continuation ==="
echo "Waiting for current wave (conc_001, conc_002, conc_003) to complete..."

# Wait for all 3 to finish
BATCHES_DONE=0
while [ $BATCHES_DONE -lt 3 ]; do
    BATCHES_DONE=0
    for bid in conc_001 conc_002 conc_003; do
        if [ -f "${OUTPUT_DIR}/${bid}/accepted.parquet" ]; then
            BATCHES_DONE=$((BATCHES_DONE + 1))
        fi
    done
    if [ $BATCHES_DONE -lt 3 ]; then
        # Check if processes are still running
        RUNNING=$(ps aux | grep 'concurrent_batch.py' | grep -v grep | wc -l)
        echo "  $(date +%H:%M:%S) ${BATCHES_DONE}/3 batch outputs ready, ${RUNNING} processes still running..."
        sleep 30
    fi
done

echo ""
echo "All 3 batches complete! Consolidating..."

# Consolidate
cd "$BASE_DIR"
uv run python scripts/data/consolidate_and_push.py

echo ""
echo "Starting auto-scale for continuous generation..."

# Start auto-scale with seed after current batches
export TARGET=100000
export CONCURRENT=3
export PASSAGES_PER_BATCH=1008
export WORKERS=16
export START_SEED=4000

bash scripts/data/auto_scale.sh
