# Checkpoints

This repository documents the compact CVAE checkpoints needed for the
paper-facing comparison and the ablations described in
`code/synva_method/docs/w_priorcalib_method_ablation.tex`.

Large artifacts are intentionally excluded from Git: CVAE weights, GHD fitting
checkpoints, canonical mesh assets, sample meshes, and intermediate training
snapshots.

## Reference baseline

The original `vessel-mesh-editing-master` W checkpoint is expected at:

```text
code/AneuG-Own-edit/checkpoints-new/first_stage_ostium_conditional/models_epoch_2000.pth
```

In the local research environment it can be copied from:

```text
/path/to/reference-vessel-mesh-editing/code/AneuG-Own-edit/checkpoints-new/first_stage_ostium_conditional/models_epoch_2000.pth
```

## SynVA W checkpoints

SynVA checkpoint configs and expected weight locations are stored under:

```text
code/synva_method/checkpoints/methods_aneug_ghds_refstage1/
```

Named paper variants:

```text
W_cond_ghd_vae/W_cond_ghd_vae_oring_seed1_20260502_160251/models_best_val.pth
W_vessel_stage3surrogate/W_vessel_stage3surrogate_seed1_20260502_222924/models_best_val.pth
W_vessel_stage3surrogate_morph/W_vessel_stage3surrogate_morph_seed1_20260503_154326/models_best_val.pth
W_vessel_stage3surrogate_priorcalib/W_vessel_stage3surrogate_priorcalib_seed1_20260504_010917/models_best_val.pth
W_vessel_stage3surrogate_morph_priorcalib/W_vessel_stage3surrogate_morph_priorcalib_seed1_20260503_203915/models_best_val.pth
W_vessel_stage3surrogate_morph_priorcalib_ostiumstrong/W_vessel_stage3surrogate_morph_priorcalib_ostiumstrong_seed1_20260503_212312/models_best_val.pth
```

Included checkpoint families:

```text
W_cond_ghd_vae/
W_vessel_stage3surrogate/
W_vessel_stage3surrogate_morph/
W_vessel_stage3surrogate_priorcalib/
W_vessel_stage3surrogate_morph_priorcalib/
W_vessel_stage3surrogate_morph_priorcalib_ostiumstrong/
```

Paper variant mapping:

```text
AneuG-Base        -> W_cond_ghd_vae/
AneuG-Cond        -> W_vessel_stage3surrogate/
AneuG-Cond-Morph  -> W_vessel_stage3surrogate_morph/
AneuG-Cond-Prior  -> W_vessel_stage3surrogate_priorcalib/
SynVA-A1          -> W_vessel_stage3surrogate_morph_priorcalib/
SynVA-A1-Ostium   -> W_vessel_stage3surrogate_morph_priorcalib_ostiumstrong/
```

The split used by all current W runs is:

```text
code/synva_method/checkpoints/aneug_ghds/splits/aneug_ghds_realcsv_opa20_seed42_20260502_013432/
```
