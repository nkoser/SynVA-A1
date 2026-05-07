#!/usr/bin/env bash
# Re-run B, C, D after canonical_mesh + early-stop fixes.
set -uo pipefail
cd /path/to/SynVA-A1
mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
MASTER=logs/methods_BCD_${STAMP}.log
echo "=== Methods B,C,D rerun @ ${STAMP} ===" | tee -a ${MASTER}
for M in B_mog_prior_cvae C_fsq_ar D_vq_transformer; do
  echo "" | tee -a ${MASTER}
  echo "### START ${M} $(date -Is) ###" | tee -a ${MASTER}
  bash methods/${M}/run.sh 1 2>&1 | tee -a ${MASTER}
  rc=${PIPESTATUS[0]}
  echo "### END   ${M} rc=${rc} $(date -Is) ###" | tee -a ${MASTER}
done
echo "=== ALL DONE $(date -Is) ===" | tee -a ${MASTER}
