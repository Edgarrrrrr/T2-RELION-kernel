#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data/diff_cal_kernel_data"
BUILD_DIR="build"
OUT_DIR="results"
RESULT_CSV="${OUT_DIR}/kernel_results.csv"

DATA_NUM_PER_ITER=1000
TEST_TIMES=20

START_TIME=$(date +%s)

[[ $# -ne 0 ]] && { echo "Usage: $0"; exit 1; }
mkdir -p "${OUT_DIR}"

"${BUILD_DIR}/test_coarse" \
    --input_dir          "${DATA_DIR}" \
    --result_file        "${RESULT_CSV}" \
    --data_num_per_iter  "${DATA_NUM_PER_ITER}" \
    --test_times         "${TEST_TIMES}"


END_TIME=$(date +%s)
echo "Elapsed time: $((END_TIME - START_TIME)) seconds"
echo "T3 done - results saved to ${RESULT_CSV}"
