#!/usr/bin/env bash
set -uo pipefail
cd /path/to/SynVA-A1
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
MASTER=logs/methods_CD_${STAMP}.log
echo "=== Methods C,D rerun @ ${STAMP} ===" | tee -a ${MASTER}
for M in C_fsq_ar D_vq_transformer; do
  echo "" | tee -a ${MASTER}
  echo "### START ${M} $(date -Is) ###" | tee -a ${MASTER}
  bash methods/${M}/run.sh 1 2>&1 | tee -a ${MASTER}
  rc=${PIPESTATUS[0]}
  echo "### END   ${M} rc=${rc} $(date -Is) ###" | tee -a ${MASTER}
done
echo "=== ALL DONE $(date -Is) ===" | tee -a ${MASTER}
