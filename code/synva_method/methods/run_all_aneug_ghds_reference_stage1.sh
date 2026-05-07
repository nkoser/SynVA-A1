#!/usr/bin/env bash
# Reference-stage retraining on /path/to/aneug-ghds:
# - same data split as /path/to/data_split_real.csv
# - same Stage-1 target convention as vessel-mesh-editing-master: GHD + scale
# - same canonical normalization factor used by that fitting pipeline: 1.10 * 2.50 = 2.75
set -euo pipefail
cd /path/to/SynVA-A1

DATASET_ROOT=${DATASET_ROOT:-/path/to/aneug-ghds/data}
CSV_SPLIT=${CSV_SPLIT:-/path/to/data_split_real.csv}
SPLIT=${ANEUG_SPLIT:-checkpoints/aneug_ghds/splits/aneug_ghds_realcsv_opa20_seed42_20260502_013432}
SEEDS=${SEEDS:-1}
METHODS=${METHODS:-"A B C D E_D E_C W"}

if [[ ! -d "${SPLIT}" ]]; then
  SPLIT=$(conda run --no-capture-output -n unified_env python tools/build_aneug_ghds_split.py \
    --data_root "${DATASET_ROOT}" --csv "${CSV_SPLIT}" --seed 42 --val_fraction 0.20 --path_only)
fi

export GHD_ROOT="${DATASET_ROOT}/ghd_fitting"
export GHD_RUN="vanilla"
export GHD_CHK_NAME="ghb_fitting_checkpoint.pkl"

# The reference baseline conditions on the ordered OPA ring in raw/reference coordinates.
# We intentionally avoid alignment_vessel here because those files are aligned aneurysm parts,
# not the true vessel surface. Collision is therefore disabled by default below.
export DATA_ROOT="${DATA_ROOT:-/path/to/prepared_meshes_3}"
export ALIGNED_DATA_ROOT="${DATASET_ROOT}/alignment"
export CONDITION_SPACE="raw"
export CONDITION_DATA_MODE="opa_only"
export CANONICAL_MESH="${DATASET_ROOT}/alignment/canonical_model/part_aligned.obj"
export EIGEN_CHK="${DATASET_ROOT}/alignment/canonical_model/canonical_model_144_normed.pkl"
export CANONICAL_OPA_CHECKPOINT="${DATASET_ROOT}/alignment/canonical_model/opa_checkpoint.pkl"
export CANONICAL_NORM_FACTOR="${CANONICAL_NORM_FACTOR:-2.75}"
export WITHSCALE_ARG="${WITHSCALE_ARG:---withscale}"

export SAVE_ROOT_A="checkpoints/methods_aneug_ghds_refstage1/A_pca_flow_matching"
export SAVE_ROOT_B="checkpoints/methods_aneug_ghds_refstage1/B_mog_prior_cvae"
export SAVE_ROOT_C="checkpoints/methods_aneug_ghds_refstage1/C_fsq_ar"
export SAVE_ROOT_D="checkpoints/methods_aneug_ghds_refstage1/D_vq_transformer"
export SAVE_ROOT_ED="checkpoints/methods_aneug_ghds_refstage1/E_collision/D"
export SAVE_ROOT_EC="checkpoints/methods_aneug_ghds_refstage1/E_collision/C"
export SAVE_ROOT_W="checkpoints/methods_aneug_ghds_refstage1/W_cond_ghd_vae"

export NO_VESSEL_ARG="${NO_VESSEL_ARG:---no_vessel_pts}"
export W_COL="${W_COL:-0.0}"

mkdir -p logs checkpoints/methods_aneug_ghds_refstage1

echo "Using split: ${SPLIT}"
echo "CSV split: ${CSV_SPLIT}"
echo "Dataset root: ${DATASET_ROOT}"
echo "GHD root: ${GHD_ROOT}"
echo "GHD run: ${GHD_RUN}"
echo "Canonical mesh: ${CANONICAL_MESH}"
echo "Eigen chk: ${EIGEN_CHK}"
echo "OPA checkpoint: ${CANONICAL_OPA_CHECKPOINT}"
echo "Condition space/mode: ${CONDITION_SPACE}/${CONDITION_DATA_MODE}"
echo "Canonical norm factor: ${CANONICAL_NORM_FACTOR}"
echo "With scale arg: ${WITHSCALE_ARG}"
echo "Collision weight: ${W_COL}"
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
