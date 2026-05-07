#!/usr/bin/env bash
# Method A — PCA95 + Whitened Flow Matching, single seed
set -euo pipefail
cd /path/to/SynVA-A1
SPLIT=${SPLIT:-checkpoints/vessel_aware_cvae/splits_finish_v5_only3999_full_20260429_171901}
META=A_pca95_flow_$(date +%Y%m%d_%H%M%S)
LOG=logs/${META}.log
conda run --no-capture-output -n unified_env python train_vessel_pca_flow_matching.py \
  --train_cases_file ${SPLIT}/cases_train.json \
  --val_cases_file   ${SPLIT}/cases_val.json \
  --save_root checkpoints/methods/A_pca_flow_matching \
  --meta ${META} --log_file ${LOG} \
  --pca_var 0.95 --whiten_pca \
  --hidden_dim 256 --time_dim 64 --flow_blocks 4 \
  --lr 1.5e-3 --weight_decay 1e-4 --lr_step 1500 --lr_gamma 0.5 \
  --epochs 4000 --batch_size 128 \
  --condition_dropout 0.1 --w_velocity 1.0 --w_endpoint 0.25 \
  --seed 1
