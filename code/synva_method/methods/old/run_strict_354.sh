#!/usr/bin/env bash
# Retrain everything (methods A/B/C/D + baseline 5-seed ensemble) on the
# strict 354-case split (only fully converged GHD fits, epoch >= 3999).
set -uo pipefail
cd /path/to/SynVA-A1
mkdir -p logs

SPLIT_DIR=${SPLIT:-checkpoints/vessel_aware_cvae/splits_finish_v5_only3999_full_20260429_171901}
export SPLIT="$SPLIT_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
MASTER=logs/strict354_all_${STAMP}.log
echo "=== Strict 354-case retraining @ ${STAMP} ===" | tee -a "$MASTER"
echo "SPLIT=$SPLIT_DIR" | tee -a "$MASTER"

# 1) Methods A, B, C, D
for M in A_pca_flow_matching B_mog_prior_cvae C_fsq_ar D_vq_transformer; do
  echo "" | tee -a "$MASTER"
  echo "### START ${M} $(date -Is) ###" | tee -a "$MASTER"
  bash methods/${M}/run.sh 1 2>&1 | tee -a "$MASTER"
  rc=${PIPESTATUS[0]}
  echo "### END   ${M} rc=${rc} $(date -Is) ###" | tee -a "$MASTER"
done

# 2) Baseline PCA-CVAE ensemble (5 seeds, same hyperparams as the 490er run)
echo "" | tee -a "$MASTER"
echo "### START baseline_pca_cvae_5seed $(date -Is) ###" | tee -a "$MASTER"
bash run_cvae_seed_ensemble.sh \
  cvae_Fpca95_strict354_ostiumonly \
  "$SPLIT_DIR" 5 \
  --model_type v8_resnet --use_conditional_prior --no_vessel_pts \
  --hidden_dim 160 --latent_dim 24 --encoder_blocks 2 --decoder_blocks 4 \
  --dropout 0.10 --condition_dropout 0.20 \
  --lr 0.0006 --weight_decay 0.0005 \
  --w_kl 0.004 --kl_schedule linear --kl_warmup 100 --free_bits 0.05 \
  --pca_var 0.95 --batch_size 16 --epochs 1000 \
  --early_stop_patience 12 --early_stop_metric val_total \
  2>&1 | tee -a "$MASTER"
rc=${PIPESTATUS[0]}
echo "### END   baseline_pca_cvae_5seed rc=${rc} $(date -Is) ###" | tee -a "$MASTER"

echo "=== ALL DONE $(date -Is) ===" | tee -a "$MASTER"
