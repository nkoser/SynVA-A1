#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/fair_reference_seams.sh <input_reference_run> <output_fair_run> [max_cases]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INPUT_ROOT="$1"
OUT_ROOT="$2"
MAX_CASES="${3:-0}"

cd "${REPO_ROOT}/code/synva_method"

conda run --no-capture-output -n unified_env python tools/fair_reference_stitch_seams.py \
  --input_root "${INPUT_ROOT}" \
  --out_root "${OUT_ROOT}" \
  --method harmonic \
  --hops 7 \
  --iterations 80 \
  --relax 0.70 \
  --blend 0.85 \
  --anchor_min 0.00 \
  --anchor_max 0.30 \
  --label0_mobility 0.75 \
  --label1_mobility 0.90 \
  --label2_mobility 1.0 \
  --max_cases "${MAX_CASES}"

