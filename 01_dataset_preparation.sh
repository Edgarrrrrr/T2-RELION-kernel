#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="data"
ZENODO_ID="15283628"

[[ $# -ne 0 ]] && { echo "Usage: $0"; exit 1; }

START_TIME=$(date +%s)

echo ">> Fetching dataset from Zenodo record ${ZENODO_ID} …"
python "./scripts/zenodo_fetch_and_extract.py" \
       --record_id "${ZENODO_ID}" \
       --target_dir "${DATA_DIR}" \

END_TIME=$(date +%s)
echo "Elapsed time: $((END_TIME - START_TIME)) seconds"
echo "T1 done - data in ${DATA_DIR}/"
