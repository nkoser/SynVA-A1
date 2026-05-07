#!/usr/bin/env bash
# W+Vessel: train the W-style ConditionalGHDVAE with real vessel/ostium/ring context.
# The checkpoint is still consumable by the official vessel-mesh-editing-master
# inference pipeline through --external-method-type W.
set -euo pipefail
cd "0 65534 65534 0dirname "-e")/../.."

SPLIT=${SPLIT:-checkpoints/aneug_ghds/splits/aneug_ghds_realcsv_opa20_seed42_20260502_013432}
GHD_ROOT=${GHD_ROOT:-/path/to/aneug-ghds/data/ghd_fitting}
GHD_RUN=${GHD_RUN:-vanilla}
GHD_CHK_NAME=${GHD_CHK_NAME:-ghb_fitting_checkpoint.pkl}
DATA_ROOT=${DATA_ROOT:-/path/to/prepared_meshes_3}
ALIGNED_DATA_ROOT=${ALIGNED_DATA_ROOT:-/path/to/aneug-ghds/data/alignment}
CANONICAL_ROOT=${CANONICAL_ROOT:-/path/to/aneug-ghds/data/alignment/canonical_model}
CANONICAL_MESH=${CANONICAL_MESH:-${CANONICAL_ROOT}/part_aligned.obj}
EIGEN_CHK=${EIGEN_CHK:-${CANONICAL_ROOT}/canonical_model_144_normed.pkl}
CANONICAL_OPA_CHECKPOINT=${CANONICAL_OPA_CHECKPOINT:-${CANONICAL_ROOT}/opa_checkpoint.pkl}
CANONICAL_NORM_FACTOR=${CANONICAL_NORM_FACTOR:-2.75}

SEED=${1:-1}
EPOCHS=${EPOCHS:-1600}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-5e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
KL_WARMUP=${KL_WARMUP:-300}
CONDITION_DROPOUT=${CONDITION_DROPOUT:-0.05}

W_VERT=${W_VERT:-50.0}
W_NORMAL=${W_NORMAL:-2.0}
W_RING=${W_RING:-10.0}
W_RING_CHAMFER=${W_RING_CHAMFER:-2.0}
EARLY_STOP_METRIC=${EARLY_STOP_METRIC:-val_ring_chamfer}

HIDDEN_DIM=${HIDDEN_DIM:-256}
LATENT_DIM=${LATENT_DIM:-108}
COND_EMBED_DIM=${COND_EMBED_DIM:-64}
VESSEL_COND_DIM=${VESSEL_COND_DIM:-96}
VESSEL_FEAT_DIM=${VESSEL_FEAT_DIM:-96}
ORDERED_RING_FEAT_DIM=${ORDERED_RING_FEAT_DIM:-64}

SAVE_ROOT_W=${SAVE_ROOT_W:-checkpoints/methods_aneug_ghds_refstage1/W_vessel}
META=${META:-W_vessel_opa20_seed${SEED}_$(date +%Y%m%d_%H%M%S)}
LOG=logs/${META}.log
mkdir -p logs "${SAVE_ROOT_W}"

conda run --no-capture-output -n unified_env python methods/W_cond_ghd_vae/train.py \
  --ghd_chk_root "${GHD_ROOT}" \
  --ghd_run "${GHD_RUN}" \
  --ghd_chk_name "${GHD_CHK_NAME}" \
  --data_root "${DATA_ROOT}" \
  --aligned_data_root "${ALIGNED_DATA_ROOT}" \
  --condition_space raw \
  --condition_data_mode alignment_vessel \
  --canonical_mesh "${CANONICAL_MESH}" \
  --canonical_mesh_obj "${CANONICAL_MESH}" \
  --eigen_chk "${EIGEN_CHK}" \
  --canonical_opa_checkpoint "${CANONICAL_OPA_CHECKPOINT}" \
  --canonical_norm_factor "${CANONICAL_NORM_FACTOR}" \
  --withscale \
  --train_cases_file "${SPLIT}/cases_train.json" \
  --val_cases_file "${SPLIT}/cases_val.json" \
  --save_root "${SAVE_ROOT_W}" \
  --meta "${META}" \
  --log_file "${LOG}" \
  --ostium_source opa_checkpoint \
  --use_ordered_ring \
  --ring_points 20 \
  --condition_mode vessel \
  --vessel_cond_dim "${VESSEL_COND_DIM}" \
  --vessel_feat_dim "${VESSEL_FEAT_DIM}" \
  --ordered_ring_feat_dim "${ORDERED_RING_FEAT_DIM}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --latent_dim "${LATENT_DIM}" \
  --cond_embed_dim "${COND_EMBED_DIM}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --w_mse 1.0 \
  --w_kl 1.0 \
  --w_vert "${W_VERT}" \
  --w_normal "${W_NORMAL}" \
  --w_ring "${W_RING}" \
  --w_ring_chamfer "${W_RING_CHAMFER}" \
  --early_stop_metric "${EARLY_STOP_METRIC}" \
  --kl_cap 30.0 \
  --free_bits 0.5 \
  --kl_warmup "${KL_WARMUP}" \
  --condition_dropout "${CONDITION_DROPOUT}" \
  --val_freq 25 \
  --save_freq 200 \
  --seed "${SEED}" 2>&1 | tee "${LOG}"
