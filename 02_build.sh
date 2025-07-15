#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="build"

[[ $# -ne 0 ]] && { echo "Usage: $0"; exit 1; }

START_TIME=$(date +%s)

mkdir -p "${BUILD_DIR}"
cd       "${BUILD_DIR}"

cmake ..
cmake --build . --config Release


END_TIME=$(date +%s)
echo "Elapsed time: $((END_TIME - START_TIME)) seconds"
echo "T2 done - executable at ${BUILD_DIR}/test_coarse"
