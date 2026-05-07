#!/usr/bin/env bash
# Sequential ordered-ostium-ring retraining launcher for methods A/B/C/D/E.
set -euo pipefail
cd /path/to/SynVA-A1

SPLIT=${SPLIT:-checkpoints/vessel_aware_cvae/splits_real_csv_ordered_ring_20260501_234236}
CSV_SPLIT=${CSV_SPLIT:-/path/to/data_split_real.csv}
GHD_ROOT=${GHD_ROOT:-/path/to/SynVA-A1/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999}
GHD_RUN=${GHD_RUN:-prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3}
GHD_CHK_NAME=${GHD_CHK_NAME:-ghb_fitting_checkpoint.pkl}
DATA_ROOT=${DATA_ROOT:-/path/to/prepared_meshes_3}
ALIGNED_DATA_ROOT=${ALIGNED_DATA_ROOT:-/path/to/ghd_prepared_meshes_3_aneurysm_1op_new}
CONDITION_SPACE=${CONDITION_SPACE:-ghd_local}
CONDITION_DATA_MODE=${CONDITION_DATA_MODE:-prepared}
CANONICAL_OPA_CHECKPOINT=${CANONICAL_OPA_CHECKPOINT:-/path/to/SynVA-A1/checkpoints/canonical_average/opa_checkpoint_1op.pkl}
SEEDS=${SEEDS:-1}
METHODS=${METHODS:-"A B C D E_D E_C W"}

echo "Using split: ${SPLIT}"
echo "Split CSV: ${CSV_SPLIT}"
echo "GHD root: ${GHD_ROOT}"
echo "GHD run: ${GHD_RUN}"
echo "Data root: ${DATA_ROOT}"
echo "Aligned data root: ${ALIGNED_DATA_ROOT}"
echo "Condition space: ${CONDITION_SPACE}"
echo "Condition data mode: ${CONDITION_DATA_MODE}"
echo "Seeds: ${SEEDS}"
echo "Methods: ${METHODS}"
export GHD_ROOT GHD_RUN GHD_CHK_NAME DATA_ROOT ALIGNED_DATA_ROOT
export CONDITION_SPACE CONDITION_DATA_MODE CANONICAL_OPA_CHECKPOINT

for seed in ${SEEDS}; do
  for method in ${METHODS}; do
    case "${method}" in
      A)
        echo "[run] Method A ordered-ring PCA flow"
        SPLIT="${SPLIT}" bash methods/A_pca_flow_matching/run.sh
        ;;
      B)
        echo "[run] Method B ordered-ring MoG CVAE seed ${seed}"
        SPLIT="${SPLIT}" bash methods/B_mog_prior_cvae/run.sh "${seed}"
        ;;
      C)
        echo "[run] Method C ordered-ring FSQ+AR seed ${seed}"
        SPLIT="${SPLIT}" bash methods/C_fsq_ar/run.sh "${seed}"
        ;;
      D)
        echo "[run] Method D ordered-ring VQ+AR seed ${seed}"
        SPLIT="${SPLIT}" bash methods/D_vq_transformer/run.sh "${seed}"
        ;;
      E_D)
        echo "[run] Method E-D ordered-ring collision VQ seed ${seed}"
        SPLIT="${SPLIT}" bash methods/E_collision/run.sh "${seed}" D
        ;;
      E_C)
        echo "[run] Method E-C ordered-ring collision FSQ seed ${seed}"
        SPLIT="${SPLIT}" bash methods/E_collision/run.sh "${seed}" C
        ;;
      W)
        echo "[run] Method W works ConditionalGHDVAE seed ${seed}"
        SPLIT="${SPLIT}" bash methods/W_cond_ghd_vae/run.sh "${seed}"
        ;;
      *)
        echo "Unknown method: ${method}" >&2
        exit 2
        ;;
    esac
  done
done
