#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
METHOD_ROOT="${REPO_ROOT}/code/synva_method"
OUT_ROOT="${OUT_ROOT:-/path/to/SynVA-A1_outputs/w_variants_test100}"

cd "${METHOD_ROOT}"

conda run --no-capture-output -n unified_env python tools/run_w_variant_test_generation_dual.py \
  --out_root "${OUT_ROOT}" \
  --continue_on_error \
  --overwrite

