#!/usr/bin/env bash
# Method W — ConditionalGHDVAE from vessel-mesh-editing-master, ordered OPA ring condition.
set -euo pipefail
cd "0 65534 65534 0dirname "-e")/../.."

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
W_VERT=${W_VERT:-0.0}
W_NORMAL=${W_NORMAL:-0.0}
W_RING=${W_RING:-0.0}
W_RING_CHAMFER=${W_RING_CHAMFER:-0.0}
EARLY_STOP_METRIC=${EARLY_STOP_METRIC:-val_mse}
EPOCHS=${EPOCHS:-4000}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-7e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
KL_WARMUP=${KL_WARMUP:-400}
CONDITION_DROPOUT=${CONDITION_DROPOUT:-0.10}
CONDITION_MODE=${CONDITION_MODE:-ring}
SAVE_ROOT_W=${SAVE_ROOT_W:-checkpoints/methods/W_cond_ghd_vae}
SEED=${1:-1}
META=${META:-W_cond_ghd_vae_oring_seed${SEED}_$(date +%Y%m%d_%H%M%S)}
LOG=logs/${META}.log
mkdir -p logs

conda run --no-capture-output -n unified_env python methods/W_cond_ghd_vae/train.py \
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
  --save_root ${SAVE_ROOT_W} \
  --meta ${META} --log_file ${LOG} \
  --ostium_source opa_checkpoint --use_ordered_ring --ring_points 20 \
  --condition_mode ${CONDITION_MODE} \
  --hidden_dim 384 --latent_dim 64 --cond_embed_dim 128 \
  --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --lr ${LR} --weight_decay ${WEIGHT_DECAY} \
  --w_mse 1.0 --w_kl 1.0 --w_vert ${W_VERT} --w_normal ${W_NORMAL} \
  --w_ring ${W_RING} --w_ring_chamfer ${W_RING_CHAMFER} \
  --early_stop_metric ${EARLY_STOP_METRIC} \
  --kl_cap 30.0 --free_bits 0.5 \
  --kl_warmup ${KL_WARMUP} --condition_dropout ${CONDITION_DROPOUT} --seed ${SEED}
