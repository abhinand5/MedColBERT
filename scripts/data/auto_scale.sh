#!/bin/bash
# MedColBERT Auto-Scale: Continuously launch concurrent batches until 100K reached.
#
# Usage:
#   bash scripts/data/auto_scale.sh
#
# Config:
#   TARGET=100000
#   CONCURRENT=2          # batches per wave
#   PASSAGES_PER_BATCH=1008
#   WORKERS=16
#   START_SEED=4000

set -euo pipefail

TARGET="${TARGET:-100000}"
CONCURRENT="${CONCURRENT:-2}"
PASSAGES_PER_BATCH="${PASSAGES_PER_BATCH:-1008}"
WORKERS="${WORKERS:-16}"
START_SEED="${START_SEED:-4000}"

BASE_DIR="/workspace/MedColBERT"
OUTPUT_BASE="${BASE_DIR}/data/processed/private/synthetic/ontology_grounded_teacher_filtered"
BATCH_SCRIPT="${BASE_DIR}/scripts/data/concurrent_batch.py"
LOG_DIR="/tmp/medcolbert_batches"
mkdir -p "$LOG_DIR"

CURRENT_TOTAL=2301  # baseline
WAVE=0
SEED=$START_SEED

echo "============================================"
echo "MedColBERT Auto-Scale"
echo "Target: $TARGET accepted examples"
echo "Concurrent batches/wave: $CONCURRENT"
echo "Passages/batch: $PASSAGES_PER_BATCH"
echo "============================================"

while true; do
    WAVE=$((WAVE + 1))

    # Count current accepted
    BATCH_ACCEPTED=0
    if [ -d "$OUTPUT_BASE" ]; then
        BATCH_ACCEPTED=$(cd "$OUTPUT_BASE" && find . -name "accepted.parquet" -type f | while read f; do
            uv run python -c "import pandas as pd; df=pd.read_parquet('$f'); print(len(df))" 2>/dev/null || echo 0
        done | paste -sd+ | bc 2>/dev/null || echo 0)
    fi
    TOTAL=$((CURRENT_TOTAL + BATCH_ACCEPTED))
    REMAINING=$((TARGET - TOTAL))

    echo ""
    echo "============================================"
    echo "WAVE $WAVE | Total: $TOTAL/$TARGET | Remaining: $REMAINING"
    echo "============================================"

    if [ "$TOTAL" -ge "$TARGET" ]; then
        echo "🎉 TARGET REACHED: $TOTAL accepted!"
        break
    fi

    # Estimate how many accepted this wave should produce
    ESTIMATED=$((PASSAGES_PER_BATCH * 65 / 100 * CONCURRENT))
    echo "Estimated this wave: ~$ESTIMATED accepted"

    # Launch concurrent batches
    PIDS=()
    for i in $(seq 1 $CONCURRENT); do
        BATCH_ID="auto_w${WAVE}_${i}"
        LOGFILE="${LOG_DIR}/${BATCH_ID}.log"
        echo "  Launching $BATCH_ID (seed=$SEED)..."
        PYTHONUNBUFFERED=1 uv run python "$BATCH_SCRIPT" \
            --seed "$SEED" \
            --batch-id "$BATCH_ID" \
            --count "$PASSAGES_PER_BATCH" \
            --workers "$WORKERS" \
            > "$LOGFILE" 2>&1 &
        PIDS+=($!)
        SEED=$((SEED + 1))
    done

    echo "  Waiting for ${#PIDS[@]} batches to complete (PIDs: ${PIDS[*]})..."

    # Wait for all batches in this wave
    FAILED=0
    for pid in "${PIDS[@]}"; do
        if wait "$pid"; then
            echo "  PID $pid: ✅ done"
        else
            echo "  PID $pid: ❌ failed"
            FAILED=$((FAILED + 1))
        fi
    done

    # Quick stats from this wave
    NEW_TOTAL_BATCH=0
    if [ -d "$OUTPUT_BASE" ]; then
        NEW_TOTAL_BATCH=$(cd "$OUTPUT_BASE" && find . -name "accepted.parquet" -type f | while read f; do
            uv run python -c "import pandas as pd; df=pd.read_parquet('$f'); print(len(df))" 2>/dev/null || echo 0
        done | paste -sd+ | bc 2>/dev/null || echo 0)
    fi
    NEW_ACCEPTED=$((NEW_TOTAL_BATCH - BATCH_ACCEPTED))
    echo "  Wave $WAVE produced: ~$NEW_ACCEPTED accepted"

    if [ "$FAILED" -ge "$CONCURRENT" ]; then
        echo "  All batches failed — stopping"
        break
    fi
done

echo ""
echo "Final total: $(cd "$OUTPUT_BASE" && find . -name 'accepted.parquet' -type f | while read f; do uv run python -c "import pandas as pd; df=pd.read_parquet('$f'); print(len(df))" 2>/dev/null || echo 0; done | paste -sd+ | bc)"
echo "Done!"
