#!/usr/bin/env bash
# Method C — FSQ-VAE + AR Prior, single seed
set -euo pipefail
cd /path/to/SynVA-A1
SPLIT=${SPLIT:-checkpoints/vessel_aware_cvae/splits_finish_v5_only3999_full_20260429_171901}
SEED=${1:-1}
META=C_fsq_seed${SEED}_$(date +%Y%m%d_%H%M%S)
LOG=logs/${META}.log
mkdir -p logs

conda run --no-capture-output -n unified_env python methods/C_fsq_ar/train.py \
  --train_cases_file ${SPLIT}/cases_train.json \
  --val_cases_file   ${SPLIT}/cases_val.json \
  --save_root checkpoints/methods/C_fsq_ar \
  --meta ${META} --log_file ${LOG} \
  --hidden_dim 384 --encoder_blocks 3 --decoder_blocks 6 \
  --num_tokens 8 --levels 8,8,5,5,5 --dropout 0.05 \
  --vessel_cond_dim 32 --no_vessel_pts \
  --batch_size 64 \
  --stage1_epochs 2000 --stage1_lr 7e-4 --stage1_wd 1e-4 \
  --ar_dim 256 --ar_depth 4 --ar_heads 4 --ar_dropout 0.10 \
  --stage2_epochs 2000 --stage2_lr 5e-4 --stage2_wd 1e-4 \
  --cond_dropout 0.10 --seed ${SEED}
