#!/usr/bin/env bash
# Method D — VQ-VAE + AR Prior, single seed
set -euo pipefail
cd /path/to/SynVA-A1
SPLIT=${SPLIT:-checkpoints/vessel_aware_cvae/splits_finish_v5_only3999_full_20260429_171901}
SEED=${1:-1}
META=D_vq_seed${SEED}_$(date +%Y%m%d_%H%M%S)
LOG=logs/${META}.log
mkdir -p logs

conda run --no-capture-output -n unified_env python methods/D_vq_transformer/train.py \
  --train_cases_file ${SPLIT}/cases_train.json \
  --val_cases_file   ${SPLIT}/cases_val.json \
  --save_root checkpoints/methods/D_vq_transformer \
  --meta ${META} --log_file ${LOG} \
  --hidden_dim 384 --encoder_blocks 3 --decoder_blocks 6 \
  --num_tokens 8 --code_dim 32 --num_codes 256 \
  --ema_decay 0.99 --commitment_beta 0.25 --dead_reset_every 50 \
  --vessel_cond_dim 32 --no_vessel_pts \
  --batch_size 64 \
  --stage1_epochs 2000 --stage1_lr 7e-4 --stage1_wd 1e-4 \
  --ar_dim 256 --ar_depth 4 --ar_heads 4 --ar_dropout 0.10 \
  --stage2_epochs 2000 --stage2_lr 5e-4 --stage2_wd 1e-4 \
  --cond_dropout 0.10 --seed ${SEED}
