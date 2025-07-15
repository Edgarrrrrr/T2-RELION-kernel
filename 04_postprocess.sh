#!/usr/bin/env bash
set -euo pipefail

CSV="results/kernel_results.csv"
FIG7="results/Fig7.pdf"
FIG8="results/Fig8.pdf"

[[ $# -ne 0 ]] && { echo "Usage: $0"; exit 1; }

START_TIME=$(date +%s)

python "scripts/draw_performance.py" \
       --input "${CSV}" \
       --fig7  "${FIG7}" \
       --fig8  "${FIG8}"


END_TIME=$(date +%s)
echo "Elapsed time: $((END_TIME - START_TIME)) seconds"
echo "T4 done - figures stored as ${FIG7}, ${FIG8}"