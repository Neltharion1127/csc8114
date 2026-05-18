#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

SESSION="REPLACE_WITH_NEW_SESSION_ID"
DEV="cpu"

run_eval() {
    local scenario=$1
    echo "=== $scenario ($SESSION) ==="
    uv run python src/data/run_evaluation.py \
        --session "$SESSION" \
        --scenario "$scenario" \
        --device "$DEV" 2>&1 | grep -E "Saved|ERROR|WARNING|Client [0-9]+ \|.*AUPRC"
    echo ""
}

for sc in N01 N02 N03 N04 L05 L06 L07 L08 L09 L10 H11 H12 H13 H14 H15 H16 M17; do
    for seed in 42 52 62; do
        run_eval "${sc}_seed${seed}"
    done
done

echo "=== All done ==="
