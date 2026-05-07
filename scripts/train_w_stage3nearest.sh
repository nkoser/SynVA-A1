#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}/code/synva_method"

bash methods/W_cond_ghd_vae/run_vessel_stage3nearest_aneug_ghds.sh "${1:-1}"

