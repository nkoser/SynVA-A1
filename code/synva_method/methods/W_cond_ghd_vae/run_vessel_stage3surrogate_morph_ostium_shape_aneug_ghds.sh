#!/usr/bin/env bash
# Shape-focused ostium-morphology W run.  This keeps the reference-stage3
# objective, but gives the sac surface/extent/moment proxies explicit weight.
set -euo pipefail

SEED=${1:-1}

export SAVE_ROOT_W=${SAVE_ROOT_W:-checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3surrogate_morph_ostium_shape}
export META=${META:-W_vessel_stage3surrogate_morph_ostium_shape_seed${SEED}_$(date +%Y%m%d_%H%M%S)}

export W_VERT=${W_VERT:-750.0}
export W_NORMAL=${W_NORMAL:-5.0}
export W_STAGE3_LABEL2=${W_STAGE3_LABEL2:-3.0}
export W_STAGE3_CENTER=${W_STAGE3_CENTER:-5.0}
export W_STAGE3_SIDE=${W_STAGE3_SIDE:-1.0}
export W_STAGE3_OPENING=${W_STAGE3_OPENING:-8.0}

export W_SHAPE_EXTENT=${W_SHAPE_EXTENT:-8.0}
export W_SHAPE_AREA=${W_SHAPE_AREA:-1.0}
export W_SHAPE_VOLUME=${W_SHAPE_VOLUME:-4.0}
export W_SHAPE_MOMENT=${W_SHAPE_MOMENT:-8.0}

export W_KL=${W_KL:-0.0001}
export KL_WARMUP=${KL_WARMUP:-1500}

exec bash methods/W_cond_ghd_vae/run_vessel_stage3surrogate_morph_ostium_aneug_ghds.sh "${SEED}"
