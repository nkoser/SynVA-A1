#!/usr/bin/env bash
# Shape-focused W run with direct sampled pouch-surface Chamfer supervision.
# This replaces the coarse morphology-proxy losses with a surface loss on the
# fitted target pouch mesh while keeping the reference Stage-3 attachment terms.
set -euo pipefail

SEED=${1:-1}

export SAVE_ROOT_W=${SAVE_ROOT_W:-checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3surrogate_morph_ostium_surface}
export META=${META:-W_vessel_stage3surrogate_morph_ostium_surface_seed${SEED}_$(date +%Y%m%d_%H%M%S)}

export BATCH_SIZE=${BATCH_SIZE:-32}
export W_VERT=${W_VERT:-400.0}
export W_NORMAL=${W_NORMAL:-5.0}

export W_STAGE3_LABEL2=${W_STAGE3_LABEL2:-3.0}
export W_STAGE3_CENTER=${W_STAGE3_CENTER:-5.0}
export W_STAGE3_SIDE=${W_STAGE3_SIDE:-1.0}
export W_STAGE3_OPENING=${W_STAGE3_OPENING:-8.0}

export W_SHAPE_EXTENT=${W_SHAPE_EXTENT:-0.0}
export W_SHAPE_AREA=${W_SHAPE_AREA:-0.0}
export W_SHAPE_VOLUME=${W_SHAPE_VOLUME:-0.0}
export W_SHAPE_MOMENT=${W_SHAPE_MOMENT:-0.0}

export W_SURFACE_CHAMFER=${W_SURFACE_CHAMFER:-200.0}
export W_SURFACE_NORMAL=${W_SURFACE_NORMAL:-0.5}
export SURFACE_SAMPLES=${SURFACE_SAMPLES:-1024}

export W_KL=${W_KL:-0.001}
export KL_WARMUP=${KL_WARMUP:-1000}
export FREE_BITS=${FREE_BITS:-0.01}

exec bash methods/W_cond_ghd_vae/run_vessel_stage3surrogate_morph_ostium_aneug_ghds.sh "${SEED}"
