# SynVA-A1: Vessel-conditioned Aneurysm Generation (preliminary)

SynVA-A1 is a vessel- and ostium-conditioned mesh generative model for the
synthesis of aneurysm sacs on existing vascular structures. The method operates
in a Graph Harmonic Deformation (GHD) coefficient space and uses conditional
CVAE variants to generate aneurysm geometry from local vessel/ostium context,
ordered ostium ring information, and optional morphology/prior-calibration
conditions. Generated sacs are decoded through the reference GHD representation
and attached to the target vessel with the reference stitching pipeline.

This repository combines the GHD fitting, canonical alignment, OPA/ostium
handling, inference, and stitching code from `vessel-mesh-editing-master` with
the SynVA-A1 model implementations, training scripts, ablations, and evaluation
wrappers developed in AneuG.

## Paper review / preliminary repository

This repository is **only** the preliminary companion codebase for **paper
review**. It is **not** intended as the final, cleaned-up release, for example
with stable APIs, packaging, complete artifact hosting, or full public
reproducibility.

The **final repository** including a refined structure, documentation, model
artifact handling, and release instructions will be published **after
acceptance**.

The intended separation is simple:

1. **Reference infrastructure** remains under `code/AneuG-Own-edit` and
   `code/inference`. This includes GHD fitting, canonical alignment,
   OPA/ostium handling, reference Stage-1 utilities, and reference Step-3
   stitching.
2. **SynVA methods** live under `code/synva_method`. This contains our
   reference-compatible W variants, ablation scripts, model loaders, metrics,
   and reproducibility runners.
3. **Data, GHD fitting checkpoints, and canonical mesh assets** are not copied
   into this repository. The default scripts use the existing `/path/to/aneug-ghds`,
   `/path/to/prepared_meshes_3`, and reference canonical assets.

## Repository Layout

```text
SynVA-A1/
  code/
    inference/                 # reference Step-3 inference/stitching
    AneuG-Own-edit/             # reference GHD fitting + Stage-1 utilities
    synva_method/               # our W variants, ablations, wrappers, metrics
  checkpoints/
    manifests/                  # checkpoint/data notes
  docs/
    REFERENCE_INFERENCE.md      # copied reference inference notes
  scripts/
    generate_test100_w_variants.sh
    train_w_stage3surrogate.sh
    train_w_stage3nearest.sh
```

## Checkpoint Artifacts

Model weights are **not committed** to this preliminary GitHub-ready repository.
The repository keeps code, configs, split files, manifests, and expected
checkpoint paths only.

The paper-facing W-family CVAE checkpoints named in
`docs/w_priorcalib_method_ablation.tex` are expected under:

```text
code/synva_method/checkpoints/methods_aneug_ghds_refstage1/
```

The paper-facing variants map to these checkpoint families:

```text
AneuG-Base:        W_cond_ghd_vae/
AneuG-Cond:        W_vessel_stage3surrogate/
AneuG-Cond-Morph:  W_vessel_stage3surrogate_morph/
AneuG-Cond-Prior:  W_vessel_stage3surrogate_priorcalib/
SynVA-A1:          W_vessel_stage3surrogate_morph_priorcalib/
SynVA-A1-Ostium:   W_vessel_stage3surrogate_morph_priorcalib_ostiumstrong/
```

Other exploratory W runs may still have training scripts in
`code/synva_method/methods/W_cond_ghd_vae/`, but their checkpoints are not
bundled in this release unless the variant is named in the paper ablation text.
For each SynVA variant, `config.json` is kept where available, but
`models_best_val.pth` and intermediate `models_epoch_*.pth` training snapshots
are excluded from Git.

The reference baseline checkpoint is expected at:

```text
code/AneuG-Own-edit/checkpoints-new/first_stage_ostium_conditional/models_epoch_2000.pth
```

In the local research environment this file comes from:

```text
/path/to/reference-vessel-mesh-editing/code/AneuG-Own-edit/checkpoints-new/first_stage_ostium_conditional/models_epoch_2000.pth
```

## Generate the 100-Case Test Set

From the repository root:

```bash
bash scripts/generate_test100_w_variants.sh
```

This calls:

```text
code/synva_method/tools/run_w_variant_test_generation_dual.py
```

and generates:

```text
/path/to/SynVA-A1_outputs/w_variants_test100
```

The strict reference path uses `code/inference/run_inference_pipeline.py` and
keeps the reference GHD decode and bridge stitching fixed. Only the Stage-1
sampler is swapped for SynVA variants.

## Train Main SynVA Variants

```bash
bash scripts/train_w_stage3surrogate.sh
bash scripts/train_w_stage3nearest.sh
```

Both scripts delegate to `code/synva_method/methods/W_cond_ghd_vae/` and use
the same `/path/to/aneug-ghds` alignment/GHD roots as the reference repository.

## GHD Fitting and Stitching

Use the reference code directly:

```text
code/AneuG-Own-edit/ghd_fitting.py
code/AneuG-Own-edit/utils/test-skripts/*
code/inference/run_inference_pipeline.py
```

SynVA does not replace these parts. The model contribution is the conditional
Stage-1 generator and its ablations.

## Provenance and License Notes

This repository is a merged preliminary research artifact for paper review.
It intentionally preserves source attribution while removing local user paths,
personal identifiers, datasets, meshes, and model weights.

We gratefully acknowledge the AneuG project, whose GHD fitting, canonical
alignment, ostium/OPA utilities, Stage-1 modeling code, and stitching pipeline
provide the reference infrastructure on which this preliminary SynVA-A1 review
repository builds. SynVA-A1 adds the vessel-conditioned W-family model variants,
ablation scripts, generation wrappers, and evaluation tools.

Baseline code that was already present in the copied AneuG/reference tree is
kept for provenance and compatibility. Subcomponents may carry their own
upstream licenses; for example, the bundled Michelangelo baseline declares
GPL-3.0 in its README and includes the corresponding license file under
`code/AneuG-Own-edit/baselines/Michelangelo/LICENSE`. External Python packages
are not vendored; users must follow the licenses of their installed
dependencies. See `NOTICE.md` for the current provenance checklist.

