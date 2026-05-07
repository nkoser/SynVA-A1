# Inference

This folder contains the case-level inference pipeline for generating an aneurysm/pouch from an ostium region and composing it back onto a vessel mesh.

There are two entry points:

- `run_inference_pipeline.py`: run one case.
- `run_inference_all_cases.py`: run the same pipeline for all cases in one or more splits.

## Required Case Structure

Each inference case must live below a split folder:

```text
cases/
└── <split_name>/
    └── <case_id>/
        ├── 04_subpointclouds/
        │   └── subpointcloud_label_2.ply
        ├── 05_submeshes/
        │   └── vessel_submesh.obj
        └── 07_other/
            ├── centroid_ostium.npy
            └── normal_vector.npy
```

Required files:

- `04_subpointclouds/subpointcloud_label_2.ply`: ostium point cloud.
- `05_submeshes/vessel_submesh.obj`: vessel mesh without generated aneurysm.
- `07_other/centroid_ostium.npy`: ostium centroid.
- `07_other/normal_vector.npy`: ostium normal vector.

The pipeline writes generated files into the case folder:

```text
cases/<split_name>/<case_id>/
├── _runtime/
└── outputs/
```

These folders are generated and are not required before running inference.

## Required Stage-1 Assets

The inference pipeline calls the trained Stage-1 model from the AneuG folder. The default location is:

```text
../AneuG-Own-edit/
```

The following assets are required:

```text
<aneug_root>/
├── infer_stage1_ostium_conditional.py
├── checkpoint-v2/
│   ├── first_stage_ostium_conditional/
│   │   └── <stage1_run_name>/
│   │       └── models_epoch_<epoch>.pth
│   └── <ghd_checkpoint_root>/
│       ├── canonical_model_144_normed.pkl
│       └── <case_id>/
│           ├── opa_checkpoint.pkl
│           └── vanilla/
│               └── ghb_fitting_checkpoint.pkl
└── <alignment_root>/
    ├── <canonical_name>/
    │   ├── part_aligned.obj
    │   ├── opa_checkpoint.pkl
    │   └── canonical_model_144_normed.pkl
    └── <case_id>/
        └── opa_checkpoint.pkl
```

Notes:

- The `<case_id>` used for inference must also exist in the Stage-1 reference roots.
- `canonical_model_144_normed.pkl` can be found either in `<alignment_root>/<canonical_name>/` or in `<ghd_checkpoint_root>/`.
- The split file used during Stage-1 training can be passed with `--stage1-split-file`, but it is only used if the file exists.

## Single-Case Inference

Run from the `inference` folder:

```bash
python run_inference_pipeline.py all \
    --case-name <case_id> \
    --case-split <split_name> \
    --num-samples 1 \
    --ring-points 20 \
    --seed <seed> \
    --resample-aneurysm-to-vessel-resolution \
    --stitch \
    --stitch-method bridge \
    --overwrite \
    --smooth-ostium-transition
```

Example:

```bash
/path/to/env/bin/python run_inference_pipeline.py all \
    --case-name <case_id> \
    --case-split <split_name> \
    --num-samples 1 \
    --ring-points 20 \
    --seed 20260429 \
    --resample-aneurysm-to-vessel-resolution \
    --stitch \
    --stitch-method bridge \
    --overwrite \
    --smooth-ostium-transition
```

Important arguments:

- `all`: runs `step1`, `step2`, and `step3`.
- `--case-name`: case folder name below `cases/<split_name>/`.
- `--case-split`: split folder, for example `train` or `test`.
- `--num-samples`: number of generated Stage-1 samples.
- `--ring-points`: number of ostium ring points. Use the same value as in Stage-1 training.
- `--seed`: random seed for sampling.
- `--resample-aneurysm-to-vessel-resolution`: remeshes the generated aneurysm to the vessel mesh resolution.
- `--stitch`: exports a stitched vessel+aneurysm mesh.
- `--stitch-method bridge`: uses bridge stitching.
- `--smooth-ostium-transition`: smooths the local stitched transition. Requires `--stitch`.
- `--overwrite`: removes existing runtime/output folders for the selected steps.

If your Stage-1 assets are not in the default locations, pass them explicitly:

```bash
python run_inference_pipeline.py all \
    --case-name <case_id> \
    --case-split <split_name> \
    --aneug-root <aneug_root> \
    --stage1-checkpoint <path_to_models_epoch.pth> \
    --stage1-ghd-root <ghd_checkpoint_root> \
    --stage1-alignment-root <alignment_root> \
    --stage1-canonical-root <alignment_root>/<canonical_name>
```

## Run All Cases

Run the pipeline for every case under the selected splits:

```bash
python run_inference_all_cases.py --overwrite
```

By default, this script searches below:

```text
cases/train/
cases/test/
```

It forwards the following options to each single-case run:

```text
--ring-points 20
--resample-aneurysm-to-vessel-resolution
--stitch
--stitch-method bridge
--smooth-ostium-transition
```

Useful options:

```bash
python run_inference_all_cases.py all --splits test --jobs 1 --overwrite
python run_inference_all_cases.py all --splits train test --max-cases 5 --overwrite
python run_inference_all_cases.py step2 --splits test --continue-on-error
```

Additional unknown arguments are forwarded to `run_inference_pipeline.py`. For example:

```bash
python run_inference_all_cases.py all \
    --splits test \
    --overwrite \
    --num-samples 2 \
    --seed 20260429
```

## Output Files

The final outputs are written to:

```text
cases/<split_name>/<case_id>/outputs/final/
```

Important files:

```text
<case_id>_generated_aneurysm_world.obj
<case_id>_vessel_with_generated_aneurysm_unstitched.obj
<case_id>_vessel_with_generated_aneurysm_stitched.obj
<case_id>_vessel_with_generated_aneurysm_stitched_labels.npy
<case_id>_sample_with_ostium_colored.ply
```

The summary files are:

```text
cases/<split_name>/<case_id>/outputs/step1_opa_summary.json
cases/<split_name>/<case_id>/outputs/step2_infer_summary.json
cases/<split_name>/<case_id>/outputs/step3_compose_summary.json
shared/last_all_cases_run_summary.json
```

## Minimal Checklist

Before running one case, verify:

```text
cases/<split_name>/<case_id>/04_subpointclouds/subpointcloud_label_2.ply
cases/<split_name>/<case_id>/05_submeshes/vessel_submesh.obj
cases/<split_name>/<case_id>/07_other/centroid_ostium.npy
cases/<split_name>/<case_id>/07_other/normal_vector.npy
<stage1_checkpoint>
<ghd_checkpoint_root>/<case_id>/vanilla/ghb_fitting_checkpoint.pkl
<ghd_checkpoint_root>/<case_id>/opa_checkpoint.pkl
<alignment_root>/<canonical_name>/part_aligned.obj
<alignment_root>/<canonical_name>/opa_checkpoint.pkl
<alignment_root>/<case_id>/opa_checkpoint.pkl
```

If any of these files are missing, the pipeline will stop before the corresponding step runs.
