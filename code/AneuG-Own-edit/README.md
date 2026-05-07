# AneuG-Own-edit

This repository contains the adapted AneuG pipeline for pouch-only GHD fitting and ostium-conditional Stage-1 VAE training.

The usual workflow is:

1. Prepare aligned mesh folders.
2. Run GHD fitting for every case.
3. Train the ostium-conditional Stage-1 VAE from the fitted GHD checkpoints.

## Environment

Create the Conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate aneug
```

The environment requires Python 3.10, PyTorch, PyTorch3D, Open3D, trimesh, pymeshlab, torch-geometric, and the other packages listed in `environment.yml`.

## Required Data Structure

All paths below are relative to the repository root.

### Alignment Data

The alignment root must contain one canonical folder and one folder per case:

```text
<alignment_root>/
├── <canonical_name>/
│   ├── part_aligned.obj
│   └── opa_checkpoint.pkl
├── <case_id_1>/
│   ├── part_aligned.obj
│   └── opa_checkpoint.pkl
├── <case_id_2>/
│   ├── part_aligned.obj
│   └── opa_checkpoint.pkl
└── ...
```

Required files:

- `part_aligned.obj`: aligned mesh.
- `opa_checkpoint.pkl`: ostium/opening checkpoint.

If you use a precomputed canonical eigen checkpoint, it can be stored anywhere, for example:

```text
<checkpoint_root>/<ghd_run_name>/canonical_model_144_normed.pkl
```

### GHD Fitting Output

After GHD fitting, the required output for each case is:

```text
<ghd_checkpoint_root>/
├── <case_id_1>/
│   └── <ghd_meta>/
│       └── ghb_fitting_checkpoint.pkl
├── <case_id_2>/
│   └── <ghd_meta>/
│       └── ghb_fitting_checkpoint.pkl
└── ...
```

By default, the Stage-1 script expects:

```text
<ghd_checkpoint_root>/<case_id>/vanilla/ghb_fitting_checkpoint.pkl
```

### Stage-1 Condition Data

For Stage-1 training, every case needs a matching condition checkpoint:

```text
<condition_root>/
├── <case_id_1>/
│   └── opa_checkpoint.pkl
├── <case_id_2>/
│   └── opa_checkpoint.pkl
└── ...
```

The case IDs in `<ghd_checkpoint_root>` and `<condition_root>` must match.

### Dataset Split

The split file is optional if you let the script create one, but if provided it must have this structure:

```json
{
  "train": ["<case_id_1>", "<case_id_2>"],
  "val": ["<case_id_3>"],
  "test": ["<case_id_4>"]
}
```

All listed case IDs must exist in both the GHD checkpoint root and the condition root.

## 1. GHD Fitting

Run `ghd_fitting.py` from the repository root:

```bash
python ghd_fitting.py \
    --device <gpu_ids> \
    --root_template <alignment_root> \
    --root_target <alignment_root> \
    --name_canonical <canonical_name> \
    --save_root <ghd_checkpoint_root> \
    --meta vanilla \
    --epochs 3000 \
    --chk_num 4 \
    --chk_freq 750 \
    --viz_freq 200 \
    --log_freq 100 \
    --canonical_eigen_chk <path_to_canonical_eigen_checkpoint>
```

Example:

```bash
python ghd_fitting.py \
    --device 1,2,3,4,5,6,7 \
    --root_template <alignment_root> \
    --root_target <alignment_root> \
    --name_canonical <canonical_name> \
    --save_root <ghd_checkpoint_root> \
    --meta vanilla \
    --epochs 3000 \
    --chk_num 4 \
    --chk_freq 750 \
    --viz_freq 200 \
    --log_freq 100 \
    --canonical_eigen_chk <ghd_checkpoint_root>/canonical_model_144_normed.pkl
```

Important arguments:

- `--device`: GPU IDs, for example `0` or `0,1,2,3`.
- `--root_template`: alignment root containing the canonical folder.
- `--root_target`: alignment root containing the target case folders.
- `--name_canonical`: name of the canonical folder. This folder is skipped as a target case.
- `--save_root`: output root for the GHD checkpoints.
- `--meta`: run name below each case folder. Stage-1 currently expects `vanilla`.
- `--canonical_eigen_chk`: path to the canonical eigen checkpoint. If the file does not exist, the fitting code can create it.

The required result per case is:

```text
<ghd_checkpoint_root>/<case_id>/vanilla/ghb_fitting_checkpoint.pkl
```

## 2. Stage-1 Ostium-Conditional VAE

Run `first_stage_ostium_conditional.py` after GHD fitting:

```bash
python first_stage_ostium_conditional.py \
    --stage1-objective mesh_vae \
    --ghd-chk-root <ghd_checkpoint_root> \
    --condition-root <condition_root> \
    --alignment-root <alignment_root> \
    --canonical-root <alignment_root>/<canonical_name> \
    --prepare-condition-from-ghd 0 \
    --force-prepare-condition-from-ghd 0 \
    --split-file <split_file.json> \
    --force-resplit 0 \
    --epochs 2000 \
    --batch-size 4 \
    --train-subset-limit 8 \
    --hidden-dim 2048 \
    --latent-dim 512 \
    --cond-embed-dim 256 \
    --norm-type layer \
    --lr 1e-4 \
    --posterior-noise-scale 0.1 \
    --max-grad-norm 1.0 \
    --target-clamp 8.0 \
    --scale-clamp 6.0 \
    --w-vert 250 \
    --w-target 1.0 \
    --w-scale 1.0 \
    --w-kl-max 0.0002 \
    --kl-warmup-epochs 300 \
    --kl-free-bits 0.01 \
    --use-reg 0 \
    --w-reg 0 \
    --w-rigid 0 \
    --w-trumpet 0 \
    --w-smooth 0 \
    --w-normal 0 \
    --w-consistency 0 \
    --w-spectral 0 \
    --w-cond 0 \
    --ring-points 20 \
    --num-workers 0 \
    --log-wandb 0 \
    --log-every 50 \
    --val-every 100 \
    --run-checkpoint-inference 0 \
    --meta <stage1_run_name>
```

Important arguments:

- `--ghd-chk-root`: root containing `<case_id>/vanilla/ghb_fitting_checkpoint.pkl`.
- `--condition-root`: root containing `<case_id>/opa_checkpoint.pkl`.
- `--alignment-root`: alignment data root.
- `--canonical-root`: canonical folder containing `part_aligned.obj` and `opa_checkpoint.pkl`.
- `--prepare-condition-from-ghd 0`: assumes condition checkpoints already exist.
- `--split-file`: JSON split file with `train`, `val`, and `test`.
- `--train-subset-limit`: limits the number of training cases. Remove it or increase it for full training.
- `--ring-points`: number of ostium ring points used as condition.
- `--meta`: Stage-1 run name.

The trained checkpoints are written to:

```text
checkpoint-v2/first_stage_ostium_conditional/<stage1_run_name>/models_epoch_*.pth
```

## Minimal Checklist

Before running Stage-1, verify that every training case has:

```text
<ghd_checkpoint_root>/<case_id>/vanilla/ghb_fitting_checkpoint.pkl
<condition_root>/<case_id>/opa_checkpoint.pkl
```

Also verify that the canonical folder has:

```text
<alignment_root>/<canonical_name>/part_aligned.obj
<alignment_root>/<canonical_name>/opa_checkpoint.pkl
```

If the split file contains cases that are missing from either root, Stage-1 will fail.
