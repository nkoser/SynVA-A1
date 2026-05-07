#!/usr/bin/env bash
# Method C — FSQ-VAE + AR Prior, single seed
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
SAVE_ROOT_C=${SAVE_ROOT_C:-checkpoints/methods/C_fsq_ar}
SEED=${1:-1}
META=C_fsq_oring_seed${SEED}_$(date +%Y%m%d_%H%M%S)
LOG=logs/${META}.log
mkdir -p logs

conda run --no-capture-output -n unified_env python methods/C_fsq_ar/train.py \
  --ghd_chk_root ${GHD_ROOT} \
  --ghd_run ${GHD_RUN} \
  --ghd_chk_name ${GHD_CHK_NAME} \
  --data_root ${DATA_ROOT} \
  --aligned_data_root ${ALIGNED_DATA_ROOT} \
  --condition_space ${CONDITION_SPACE} \
  --condition_data_mode ${CONDITION_DATA_MODE} \
  --canonical_mesh ${CANONICAL_MESH} \
  --canonical_mesh_obj ${CANONICAL_MESH} \
  --eigen_chk ${EIGEN_CHK} \
  --canonical_opa_checkpoint ${CANONICAL_OPA_CHECKPOINT} \
  --canonical_norm_factor ${CANONICAL_NORM_FACTOR} \
  ${WITHSCALE_ARG} \
  --train_cases_file ${SPLIT}/cases_train.json \
  --val_cases_file   ${SPLIT}/cases_val.json \
  --save_root ${SAVE_ROOT_C} \
  --meta ${META} --log_file ${LOG} \
  --ostium_source opa_checkpoint --use_ordered_ring --ring_points 20 \
  --hidden_dim 384 --encoder_blocks 3 --decoder_blocks 6 \
  --num_tokens 8 --levels 8,8,5,5,5 --dropout 0.05 \
  --vessel_cond_dim 32 --no_vessel_pts \
  --batch_size 64 \
  --stage1_epochs 2000 --stage1_lr 7e-4 --stage1_wd 1e-4 \
  --ar_dim 256 --ar_depth 4 --ar_heads 4 --ar_dropout 0.10 \
  --stage2_epochs 2000 --stage2_lr 5e-4 --stage2_wd 1e-4 \
  --cond_dropout 0.10 --seed ${SEED}
