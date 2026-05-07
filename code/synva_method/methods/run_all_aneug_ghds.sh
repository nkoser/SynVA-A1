#!/usr/bin/env bash
# Retrain methods on the external /path/to/aneug-ghds fitting tokens.
set -euo pipefail
cd /path/to/SynVA-A1

DATASET_ROOT=${DATASET_ROOT:-/path/to/aneug-ghds/data}
CSV_SPLIT=${CSV_SPLIT:-/path/to/data_split_real.csv}
SPLIT=${ANEUG_SPLIT:-}
SEEDS=${SEEDS:-1}
METHODS=${METHODS:-"A B C D E_D E_C W"}

if [[ -z "${SPLIT}" ]]; then
  SPLIT=$(conda run --no-capture-output -n unified_env python tools/build_aneug_ghds_split.py \
    --data_root "${DATASET_ROOT}" --csv "${CSV_SPLIT}" --seed 42 --val_fraction 0.20 --path_only)
fi

export GHD_ROOT="${DATASET_ROOT}/ghd_fitting"
export GHD_RUN="vanilla"
export GHD_CHK_NAME="ghb_fitting_checkpoint.pkl"
export DATA_ROOT="__alignment_vessel__"
export ALIGNED_DATA_ROOT="${DATASET_ROOT}/alignment"
export CONDITION_SPACE="raw"
export CONDITION_DATA_MODE="alignment_vessel"
export CANONICAL_MESH="${DATASET_ROOT}/alignment/canonical_model/part_aligned.obj"
export EIGEN_CHK="${DATASET_ROOT}/alignment/canonical_model/canonical_model_144_normed.pkl"
export CANONICAL_OPA_CHECKPOINT="${DATASET_ROOT}/alignment/canonical_model/opa_checkpoint.pkl"

export SAVE_ROOT_A="checkpoints/methods_aneug_ghds/A_pca_flow_matching"
export SAVE_ROOT_B="checkpoints/methods_aneug_ghds/B_mog_prior_cvae"
export SAVE_ROOT_C="checkpoints/methods_aneug_ghds/C_fsq_ar"
export SAVE_ROOT_D="checkpoints/methods_aneug_ghds/D_vq_transformer"
export SAVE_ROOT_ED="checkpoints/methods_aneug_ghds/E_collision/D"
export SAVE_ROOT_EC="checkpoints/methods_aneug_ghds/E_collision/C"
export SAVE_ROOT_W="checkpoints/methods_aneug_ghds/W_cond_ghd_vae"

export NO_VESSEL_ARG="${NO_VESSEL_ARG:-}"
export W_COL="${W_COL:-50.0}"

mkdir -p logs checkpoints/methods_aneug_ghds

echo "Using split: ${SPLIT}"
echo "CSV split: ${CSV_SPLIT}"
echo "Dataset root: ${DATASET_ROOT}"
echo "GHD root: ${GHD_ROOT}"
echo "OPA root: ${ALIGNED_DATA_ROOT}"
echo "Canonical mesh: ${CANONICAL_MESH}"
echo "Eigen chk: ${EIGEN_CHK}"
echo "Condition data mode: ${CONDITION_DATA_MODE}"
echo "Methods: ${METHODS}"
echo "Seeds: ${SEEDS}"

for seed in ${SEEDS}; do
  for method in ${METHODS}; do
    case "${method}" in
      A)
        SPLIT="${SPLIT}" bash methods/A_pca_flow_matching/run.sh
        ;;
      B)
        SPLIT="${SPLIT}" bash methods/B_mog_prior_cvae/run.sh "${seed}"
        ;;
      C)
        SPLIT="${SPLIT}" bash methods/C_fsq_ar/run.sh "${seed}"
        ;;
      D)
        SPLIT="${SPLIT}" bash methods/D_vq_transformer/run.sh "${seed}"
        ;;
      E_D)
        SPLIT="${SPLIT}" bash methods/E_collision/run.sh "${seed}" D
        ;;
      E_C)
        SPLIT="${SPLIT}" bash methods/E_collision/run.sh "${seed}" C
        ;;
      W)
        SPLIT="${SPLIT}" bash methods/W_cond_ghd_vae/run.sh "${seed}"
        ;;
      *)
        echo "Unknown method: ${method}" >&2
        exit 2
        ;;
    esac
  done
done
