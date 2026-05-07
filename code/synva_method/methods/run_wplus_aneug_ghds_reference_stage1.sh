#!/usr/bin/env bash
# Train W+ on /path/to/aneug-ghds with the reference Stage-1 representation.
# W+ = vessel-mesh-editing-style ConditionalGHDVAE + optional mesh/ring losses.
set -euo pipefail
cd /path/to/SynVA-A1

DATASET_ROOT=${DATASET_ROOT:-/path/to/aneug-ghds/data}
CSV_SPLIT=${CSV_SPLIT:-/path/to/data_split_real.csv}
SPLIT=${ANEUG_SPLIT:-checkpoints/aneug_ghds/splits/aneug_ghds_realcsv_opa20_seed42_20260502_013432}
SEED=${SEED:-1}

if [[ ! -d "${SPLIT}" ]]; then
  SPLIT=$(conda run --no-capture-output -n unified_env python tools/build_aneug_ghds_split.py \
    --data_root "${DATASET_ROOT}" --csv "${CSV_SPLIT}" --seed 42 --val_fraction 0.20 --path_only)
fi

export GHD_ROOT="${DATASET_ROOT}/ghd_fitting"
export GHD_RUN="vanilla"
export GHD_CHK_NAME="ghb_fitting_checkpoint.pkl"
export DATA_ROOT="${DATA_ROOT:-/path/to/prepared_meshes_3}"
export ALIGNED_DATA_ROOT="${DATASET_ROOT}/alignment"
export CONDITION_SPACE="raw"
export CONDITION_DATA_MODE="opa_only"
export CANONICAL_MESH="${DATASET_ROOT}/alignment/canonical_model/part_aligned.obj"
export EIGEN_CHK="${DATASET_ROOT}/alignment/canonical_model/canonical_model_144_normed.pkl"
export CANONICAL_OPA_CHECKPOINT="${DATASET_ROOT}/alignment/canonical_model/opa_checkpoint.pkl"
export CANONICAL_NORM_FACTOR="${CANONICAL_NORM_FACTOR:-2.75}"
export WITHSCALE_ARG="${WITHSCALE_ARG:---withscale}"

export SAVE_ROOT_W="${SAVE_ROOT_W:-checkpoints/methods_aneug_ghds_refstage1/W_plus_meshring}"
export META="${META:-W_plus_meshring_seed${SEED}_$(date +%Y%m%d_%H%M%S)}"

# Conservative first W+ recipe: keep coefficient MSE dominant, add geometry guidance.
export W_VERT="${W_VERT:-100.0}"
export W_NORMAL="${W_NORMAL:-5.0}"
export W_RING="${W_RING:-0.0}"
export W_RING_CHAMFER="${W_RING_CHAMFER:-1.0}"
export EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-val_vert_mse}"

mkdir -p logs "${SAVE_ROOT_W}"

echo "Using split: ${SPLIT}"
echo "Dataset root: ${DATASET_ROOT}"
echo "Save root: ${SAVE_ROOT_W}"
echo "Meta: ${META}"
echo "Weights: W_VERT=${W_VERT} W_NORMAL=${W_NORMAL} W_RING=${W_RING} W_RING_CHAMFER=${W_RING_CHAMFER}"
echo "Early-stop metric: ${EARLY_STOP_METRIC}"

SPLIT="${SPLIT}" bash methods/W_cond_ghd_vae/run.sh "${SEED}"
