#!/usr/bin/env bash
# Method A — PCA95 + Whitened Flow Matching, single seed
set -euo pipefail
cd /path/to/SynVA-A1
SPLIT=${SPLIT:-checkpoints/vessel_aware_cvae/splits_real_csv_ordered_ring_20260501_234236}
GHD_ROOT=${GHD_ROOT:-/path/to/SynVA-A1/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999}
GHD_RUN=${GHD_RUN:-prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3}
GHD_CHK_NAME=${GHD_CHK_NAME:-ghb_fitting_checkpoint.pkl}
DATA_ROOT=${DATA_ROOT:-/path/to/prepared_meshes_3}
ALIGNED_DATA_ROOT=${ALIGNED_DATA_ROOT:-/path/to/ghd_prepared_meshes_3_aneurysm_1op_new}
CONDITION_SPACE=${CONDITION_SPACE:-ghd_local}
CONDITION_DATA_MODE=${CONDITION_DATA_MODE:-prepared}
CANONICAL_MESH=${CANONICAL_MESH:-/path/to/SynVA-A1/checkpoints/canonical_average/part_aligned.obj}
EIGEN_CHK=${EIGEN_CHK:-/path/to/SynVA-A1/checkpoints/canonical_average/eigen_chk_144.pkl}
CANONICAL_OPA_CHECKPOINT=${CANONICAL_OPA_CHECKPOINT:-/path/to/SynVA-A1/checkpoints/canonical_average/opa_checkpoint_1op.pkl}
CANONICAL_NORM_FACTOR=${CANONICAL_NORM_FACTOR:-1.10}
WITHSCALE_ARG=${WITHSCALE_ARG:-}
SAVE_ROOT_A=${SAVE_ROOT_A:-checkpoints/methods/A_pca_flow_matching}
META=A_pca95_flow_oring_$(date +%Y%m%d_%H%M%S)
LOG=logs/${META}.log
conda run --no-capture-output -n unified_env python train_vessel_pca_flow_matching.py \
  --ghd_chk_root ${GHD_ROOT} \
  --ghd_run ${GHD_RUN} \
  --ghd_chk_name ${GHD_CHK_NAME} \
  --data_root ${DATA_ROOT} \
  --aligned_data_root ${ALIGNED_DATA_ROOT} \
  --condition_space ${CONDITION_SPACE} \
  --condition_data_mode ${CONDITION_DATA_MODE} \
  --canonical_mesh ${CANONICAL_MESH} \
  --canonical_mesh_obj ${CANONICAL_MESH} \
  --eigen_chk_obj ${EIGEN_CHK} \
  --canonical_opa_checkpoint ${CANONICAL_OPA_CHECKPOINT} \
  --canonical_norm_factor ${CANONICAL_NORM_FACTOR} \
  ${WITHSCALE_ARG} \
  --train_cases_file ${SPLIT}/cases_train.json \
  --val_cases_file   ${SPLIT}/cases_val.json \
  --save_root ${SAVE_ROOT_A} \
  --meta ${META} --log_file ${LOG} \
  --pca_var 0.95 --whiten_pca \
  --ostium_source opa_checkpoint --use_ordered_ring --ring_points 20 \
  --hidden_dim 256 --time_dim 64 --flow_blocks 4 \
  --lr 1.5e-3 --weight_decay 1e-4 --lr_step 1500 --lr_gamma 0.5 \
  --epochs 4000 --batch_size 128 \
  --condition_dropout 0.1 --w_velocity 1.0 --w_endpoint 0.25 \
  --seed 1
