# Vessel-Conditioned Aneurysm Sac Generation with Reference-Compatible Stitching

**Methods and experimental protocol, current draft (May 2026).**

This note documents the current experimental setup used in AneuG after the
integration with `vessel-mesh-editing-master`. It supersedes the earlier
methods draft that described the A/C/D/E architecture sweep as the active
comparison. The current paper-facing comparison is centered on the reference
conditional GHD-VAE (`W`) from `vessel-mesh-editing-master` and on our
reference-compatible W variants. The guiding constraint is that the full
inference and stitching pipeline is kept identical to the reference repository
wherever possible; only the Stage-1 generative model is exchanged.

---

## 1. Task Definition

Given a parent vessel mesh and an ostium region, the task is to generate a
plausible aneurysm sac that can be attached to the vessel at the prescribed
opening. The generated sac is represented in the Graph Harmonic Deformation
(GHD) coefficient space of a canonical aneurysm template. At inference time,
the GHD coefficients are decoded to a triangular sac mesh, aligned to the
patient-specific ostium, and stitched into the patient vessel.

The current comparison isolates three components:

1. the Stage-1 conditional sac generator,
2. the transformation from generated GHD coefficients to a patient-space sac,
3. the final vessel-sac stitching and local seam smoothing.

For fair comparison against `vessel-mesh-editing-master`, component 2 and 3
are kept identical to the reference pipeline in the strict experiments. Our
model changes only component 1.

---

## 2. Data, Split, and Representation

### Dataset

All current W experiments use the data package from the reference repository:

```text
/path/to/aneug-ghds/data/ghd_fitting
/path/to/aneug-ghds/data/alignment
/path/to/prepared_meshes_3
```

The canonical model is:

```text
/path/to/aneug-ghds/data/alignment/canonical_model/part_aligned.obj
/path/to/aneug-ghds/data/alignment/canonical_model/canonical_model_144_normed.pkl
/path/to/aneug-ghds/data/alignment/canonical_model/opa_checkpoint.pkl
```

The active split is fixed by the real CSV split:

```text
checkpoints/aneug_ghds/splits/aneug_ghds_realcsv_opa20_seed42_20260502_013432
```

| Split | Cases |
| --- | ---: |
| Train | 530 |
| Validation | 132 |
| Test | 100 |

The exported full-test meshes use exactly `cases_test.json` from this split.

### Shape representation

Each aneurysm sac is encoded by GHD coefficients on a fixed canonical mesh.
The current W experiments use the reference GHD fitting run:

```text
ghd_run = vanilla
ghd_chk_name = ghb_fitting_checkpoint.pkl
```

The model input includes the GHD coefficients and, when `--withscale` is
enabled, an additional scale channel. Thus the learned vector is

```text
x = [ghd_coefficients, scale]
```

where the GHD part has 432 dimensions. Dataset statistics are stored in the
checkpoint and restored during sampling.

### Conditioning

The reference W baseline conditions on an ordered ostium ring. Our W variants
replace this with a richer vessel-aware condition while keeping the same
reference GHD and Stage-3 infrastructure. The active condition mode is:

```text
condition_space = raw
condition_data_mode = alignment_vessel
ostium_source = opa_checkpoint
use_ordered_ring = true
ring_points = 20
num_vessel_pts = 256
num_label2_pts = 256
condition_mode = vessel
```

The conditioner receives patient-space vessel information from the aligned
`part_aligned.obj`, the OPA-derived ordered ostium ring, ostium parameters, and
label-2 ostium point samples. It outputs a 96-dimensional condition embedding.

---

## 3. Reference Model W

The reference baseline is the conditional GHD-VAE from
`vessel-mesh-editing-master`. We refer to it as `W_ref`.

Reference checkpoint:

```text
/path/to/SynVA-A1/code/AneuG-Own-edit/checkpoints-new/first_stage_ostium_conditional/models_epoch_2000.pth
```

The model is used through the reference Stage-1 inference script:

```text
/path/to/SynVA-A1/code/AneuG-Own-edit/utils/test-skripts/infer_stage1_ostium_conditional.py
```

and the reference Stage-3 composition/stitching script:

```text
/path/to/SynVA-A1/code/inference/run_inference_pipeline.py
```

This baseline is only generated in the strict reference pipeline, because it
does not have a separate AneuG-native sampler path. In the final export,
`normal_stitching/W_ref` is therefore intentionally skipped.

---

## 4. Our Reference-Compatible W Variants

Our W variants retain the reference ConditionalGHDVAE architecture but change
the condition model and loss terms. The implementation lives in:

```text
methods/W_cond_ghd_vae/model.py
methods/W_cond_ghd_vae/train.py
methods/_common/stage3_surrogate_loss.py
```

### Selected method: W_stage3surrogate_morph_priorcalib

Paper figure source:

```text
docs/figures/w_priorcalib_model.drawio
docs/figures/w_priorcalib_final_method_detail.drawio
```

The current best overall method is
`W_stage3surrogate_morph_priorcalib`. It is selected as the main proposed
method because it improves the held-out morphology distribution over `W_ref`
while retaining the exact reference Stage-3 and stitching pipeline. In other
words, the comparison changes only the conditional Stage-1 generator; alignment,
bridge stitching, and ostium smoothing remain those of
`vessel-mesh-editing-master`.

Checkpoint:

```text
checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3surrogate_morph_priorcalib/W_vessel_stage3surrogate_morph_priorcalib_seed1_20260503_203915/models_best_val.pth
```

Training log:

```text
logs/W_vessel_stage3surrogate_morph_priorcalib_seed1_20260503_203915.log
```

The best checkpoint was selected by validation total loss:

| Run | best epoch | best val total |
| --- | ---: | ---: |
| `W_stage3surrogate_morph_priorcalib_seed1_20260503_203915` | 2550 | 1.327942 |

Methodically, the model combines five ingredients:

1. the reference W conditional GHD-VAE backbone;
2. vessel-aware conditioning from the aligned vessel, ordered OPA ostium ring,
   label-2 ostium samples, and ostium parameters;
3. morphology conditioning from `/path/to/synva_real_data`;
4. differentiable Stage-3 surrogate losses so the generated sac is trained in
   the same world-space frame where the reference stitcher attaches it;
5. prior-path calibration, which explicitly trains the inference path
   `z ~ N(0,I) -> decoder(z,c)` instead of only the posterior reconstruction
   path `x -> encoder -> decoder`.

This last point is the key methodological difference. The early morphology
models reconstructed well but generated overly compact sacs at sampling time.
That happened because training optimized mostly the posterior path, while test
generation samples from the prior. `W_stage3surrogate_morph_priorcalib`
therefore decodes an additional prior sample during training and matches both
the target sample and the batch-level mean/std of the target GHD distribution.

### Final method in detail

The final method is best understood as a reference-compatible replacement for
only the first stage of the `vessel-mesh-editing-master` pipeline. The reference
repository provides a conditional GHD-VAE that maps an ostium condition to GHD
coefficients of an aneurysm sac. We keep this representation and the downstream
reference geometry processing, but replace the condition model and training
objective.

For each case, the model receives a condition vector

```text
c = f_cond(vessel, ostium_ring, label2_points, ostium_params, morphology)
```

where:

| Input | Role in the final method |
| --- | --- |
| aligned vessel mesh | Gives the local parent-vessel context in the same aligned frame used by the reference data. |
| OPA ordered ostium ring | Defines the target opening geometry and provides a stable ordered ring condition. |
| label-2 ostium samples | Provide a denser geometric description of the opening surface used by the Stage-3 surrogate. |
| ostium parameters | Encode opening center, orientation, and scale-related ostium information. |
| morphology vector | Conditions the sac on case-level anatomical shape descriptors from `/path/to/synva_real_data`. |

The target predicted by the model is:

```text
x = [GHD coefficients, scale]
```

The GHD coefficients deform the fixed canonical aneurysm template from
`/path/to/aneug-ghds/data/alignment/canonical_model`, and the scale channel keeps
the generated sac at the correct global size. After prediction, the GHD vector
is decoded back to a sac mesh before the reference Stage-3 alignment and
stitching steps.

The final model has two training paths but only one inference path:

| Path | Used when | Purpose |
| --- | --- | --- |
| posterior path `x,c -> q(z|x,c) -> decoder(z,c)` | training | Learns reconstruction, KL structure, and mesh-space consistency. |
| prior path `z~N(0,I),c -> decoder(z,c)` | training and inference | Matches the exact sampling path used at test time. |

This distinction is central. A standard VAE can obtain good validation
reconstruction while still producing weak samples, because the decoder sees
posterior latents during training but prior latents during generation. The
final method explicitly trains the prior path, so the generated samples are not
only plausible reconstructions but are also better calibrated as unconditional
samples for a fixed condition.

The full objective is:

```text
L_total =
    L_coeff
  + L_scale
  + L_vertex
  + L_KL
  + L_stage3
  + L_prior
```

with:

```text
L_coeff  = MSE(x_hat, x)
L_scale  = Huber(scale_hat, scale)
L_vertex = MSE(V_hat, V)
L_KL     = KL(q(z|x,c) || N(0,I))
```

`L_vertex` decodes the predicted GHD coefficients through the canonical
eigenbasis and supervises the resulting mesh vertices. This avoids a failure
mode where coefficient MSE looks acceptable but the decoded sac is poorly
positioned or distorted in mesh space.

The Stage-3 surrogate loss is:

```text
L_stage3 =
    w_label2  * d(predicted_sac, label2_ostium)
  + w_center  * d(predicted_center, target_center)
  + w_side    * side_penalty(predicted_sac, ostium_plane)
  + w_opening * d(predicted_opening_center, ostium_center)
```

It approximates the geometry that the reference Stage-3 pipeline later uses for
alignment and stitching. The purpose is not to replace the reference Stage-3
code, but to make the Stage-1 generator produce sacs that are already easier
for that fixed reference pipeline to place and attach.

The prior calibration term is:

```text
z_prior ~ N(0,I)
x_prior = decoder(z_prior, c)

L_prior =
    w_prior_mse        * Huber(x_prior, x)
  + w_prior_batch_mean * MSE(mean_batch(x_prior), mean_batch(x))
  + w_prior_batch_std  * MSE(std_batch(x_prior),  std_batch(x))
```

The first term keeps prior samples case-compatible. The batch-mean term keeps
the generated population centered on the training distribution. The batch-std
term is the important diversity/realism term: it discourages collapse toward a
single compact conditional mean and was added because earlier morphology runs
looked too small and too concentrated in morphology space.

The selected weights are:

| Component | Weight |
| --- | ---: |
| coefficient MSE | 1.0 |
| scale Huber | 1.0 |
| vertex MSE | 250.0 |
| KL | 0.01 |
| Stage-3 label2 | 5.0 |
| Stage-3 center | 10.0 |
| Stage-3 side | 2.0 |
| Stage-3 opening | 10.0 |
| prior Huber | 0.25 |
| prior batch mean | 0.25 |
| prior batch std | 8.0 |

This balance is the reason the method is currently preferred. It improves the
global morphology distribution compared with `W_ref`, especially in MMD and
Frechet distance, while keeping the exact reference stitching path unchanged.
The later `ostiumstrong` ablation increased the Stage-3/ostium weights, but it
did not become the final method: it improved `opening_center` slightly while
degrading ring/label-2 attachment and giving back part of the morphology
distribution gain. The final paper-facing method is therefore
`W_stage3surrogate_morph_priorcalib`, with `ostiumstrong` reported as an
ablation.

### Comparison to W_ref

The final method is not uniformly better than `W_ref` on every metric. The
main improvement is in aneurysm-sac shape realism and morphology distribution,
not in every local attachment proxy. This distinction is important for the
paper claim.

The metrics used in this comparison are:

| Metric | Definition | Interpretation |
| --- | --- | --- |
| Chamfer mean | Mean symmetric Chamfer distance between the generated aneurysm sac mesh and the held-out GT aneurysm submesh, computed from 10k sampled surface points per mesh. | Lower means the generated sac surface is closer to the paired GT sac surface on average. |
| `D_max` abs | Mean absolute error of maximum aneurysm diameter between generated sac and paired GT morphology. | Lower means the generated sac better matches the GT maximum diameter. |
| `H_max` abs | Mean absolute error of maximum aneurysm height between generated sac and paired GT morphology. | Lower means the generated sac better matches the GT height. |
| `W_max` abs | Mean absolute error of maximum aneurysm width between generated sac and paired GT morphology. | Lower means the generated sac better matches the GT width. |
| paired morphology z-L2 | Case-paired Euclidean distance between generated and GT morphology vectors after standardizing each morphology feature by the GT test-set mean and standard deviation. | Lower means the generated sac is closer to its paired GT case in normalized morphology space. |
| coverage | Fraction of GT morphology samples that lie within the 95th-percentile GT leave-one-out nearest-neighbor radius of at least one generated sample. | Higher means the generated population covers more of the GT morphology distribution. |
| MMD | Multi-bandwidth RBF maximum mean discrepancy between generated and GT morphology-feature distributions. | Lower means the generated and GT morphology distributions are globally more similar. |
| Frechet | Gaussian Frechet distance between generated and GT morphology distributions, computed from their feature means and covariances. | Lower means generated morphology has a closer global mean/covariance structure to GT. |
| `opening_center` | Distance between the generated opening center and the target ostium center after reference Stage-3 alignment. | Lower means the generated neck is centered more accurately on the ostium. |
| `ring_to_label2` | Mean distance from generated opening-ring points to the label-2 ostium point cloud. | Lower means the generated opening ring better follows the ostium surface. |
| `label2_to_pouch` | Mean distance from label-2 ostium points to the generated pouch surface. | Lower means the generated pouch reaches the prescribed ostium region more closely. |
| `nearest_vertex` | Distance from the ostium center to the nearest generated sac vertex after alignment. | Lower means some part of the sac is closer to the ostium center. |
| `pouch_center` | Distance between the generated sac center and the ostium center after alignment. | Lower means the global sac placement is closer to the ostium center. |

`W_stage3surrogate_morph_priorcalib` improves over `W_ref` on the sac-shape and
morphology metrics:

| Metric | `W_ref` | Final method | Better |
| --- | ---: | ---: | --- |
| Chamfer mean | 0.094099 | **0.086362** | final |
| `D_max` abs | 0.113297 | **0.084209** | final |
| `H_max` abs | 0.119853 | **0.079572** | final |
| `W_max` abs | 0.074605 | **0.057382** | final |
| paired morphology z-L2 | 1.284607 | **0.980470** | final |
| coverage | 0.84 | **0.85** | final |
| MMD | 0.006438 | **0.005519** | final |
| Frechet | 1.092620 | **1.017597** | final |

However, `W_ref` remains slightly stronger on several local ostium attachment
metrics:

| Metric | `W_ref` | Final method | Better |
| --- | ---: | ---: | --- |
| `opening_center` | **0.011645** | 0.013155 | `W_ref` |
| `ring_to_label2` | **0.018527** | 0.019024 | `W_ref` |
| `label2_to_pouch` | **0.015175** | 0.015538 | `W_ref` |
| `nearest_vertex` | **0.127311** | 0.128361 | `W_ref` |
| `pouch_center` | **0.118194** | 0.130863 | `W_ref` |

The recommended paper claim is therefore:

```text
The proposed prior-calibrated vessel- and morphology-conditioned W model
improves aneurysm-sac morphology and distributional realism over the reference
W baseline while preserving the exact reference stitching pipeline. The
reference baseline remains slightly stronger on some local ostium attachment
metrics.
```

This is the fair interpretation of the current results: we beat `W_ref` on
the generated aneurysm sac and morphology distribution, while `W_ref` still has
a small advantage in local neck/attachment placement.

### Architecture

The generator is a conditional VAE matching the reference W backbone:

```text
encoder: x -> Linear(hidden) -> residual block -> mu, logvar
decoder: [z, condition] -> Linear(hidden) -> residual block -> x_hat
```

Current hyperparameters for the successful W variants:

| Parameter | Value |
| --- | ---: |
| hidden_dim | 256 |
| latent_dim | 108 |
| cond_embed_dim | 64 |
| norm_type | batch |
| batch_size | 96 |
| epochs | 4000 |
| lr | 1e-4 |
| weight_decay | 1e-4 |
| kl_warmup | 1000 |
| free_bits | 0.01 |
| w_kl | 0.0002 |
| w_mse | 1.0 |
| w_scale | 1.0 |
| w_vert | 250.0 |
| w_normal | 0.0 |

The later morphology/prior-calibrated variants intentionally use stronger KL
regularization:

| Parameter | Value |
| --- | ---: |
| w_kl | 0.01 |
| free_bits | 0.02 |
| w_prior_mse | 0.25 |
| w_prior_batch_mean | 0.25 |
| w_prior_batch_std | 8.0 |

### Base loss

The base objective combines coefficient reconstruction, KL regularization,
scale reconstruction, and decoded vertex reconstruction:

```text
L_base =
    w_mse   * MSE(x_hat, x)
  + w_kl    * KL(q(z|x) || N(0,I))
  + w_scale * Huber(scale_hat, scale)
  + w_vert  * MSE(V_hat, V)
```

The mesh-space term is computed by decoding the predicted GHD coefficients
through the canonical mesh and eigenbasis. This was important because good
coefficient MSE alone did not reliably produce well-placed sacs after the
reference Stage-3 transformation.

### Stage-3 surrogate losses

The key addition is a differentiable approximation of the reference Stage-3
alignment. The surrogate first performs a differentiable similarity alignment
between the predicted opening ring and the target OPA/label-2 ring. It then
measures geometry in the same world-space frame used by the reference pipeline.

Implemented terms:

| Term | Purpose |
| --- | --- |
| `stage3_label2` | Pulls the generated sac surface toward the label-2 ostium point cloud after alignment. |
| `stage3_center` | Matches the patient-space sac center to the fitted target sac center. |
| `stage3_side` | Penalizes sacs appearing on the wrong side of the ostium plane. |
| `stage3_opening` | Keeps the generated opening center close to the target ostium center. |
| `stage3_nearest` | Explicitly supervises the nearest sac-to-ostium distance after alignment. |

The two successful variants are:

| Variant | Checkpoint | Additional stage3 terms |
| --- | --- | --- |
| `W_stage3surrogate` | `checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3surrogate/W_vessel_stage3surrogate_seed1_20260502_222924/models_best_val.pth` | label2=5, center=10, side=2, opening=10 |
| `W_stage3nearest` | `checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3nearest/W_vessel_stage3nearest_seed1_20260502_230124/models_best_val.pth` | label2=5, center=10, nearest=10, side=2, opening=10 |

The nearest variant was selected for the final strict visual/export pass
because it gave the best recent trade-off on the 20-case smoke comparison:
stronger ostium/ring alignment than `W_ref` and better nearest/neck behavior
than the pure surrogate version.

### Morphology conditioning and prior calibration

The first morphology-conditioned variants appended scalar aneurysm morphology
parameters from `/path/to/synva_real_data` to the vessel condition. The most useful
condition set for the selected model was the full default morphology vector
available in the morphology file. Additional ablations tested selected
ostium/pouch subsets such as:

```text
A_A, V_A, A_CH, V_CH, D_max, H_max, W_max, H_ortho, W_ortho
```

The paired GT metrics showed that morphology conditioning improved
case-specific shape matching, but the generated population still tended to
regress toward compact sacs. This was visible in the morphology distribution:
`D_max`, `H_max`, and `W_ortho` were systematically below the held-out GT
means. The reason is that the VAE training objective mainly optimizes the
posterior reconstruction path

```text
x -> encoder -> z -> decoder
```

while inference uses the prior path

```text
z ~ N(0,I) -> decoder(z, condition)
```

To reduce this mismatch, `W_stage3surrogate_morph_priorcalib` adds explicit
prior-path training losses. During training, an additional prior sample
`z_prior ~ N(0,I)` is decoded with the same condition that will be used at
inference. The loss is:

```text
L_prior =
    w_prior_mse        * Huber(decode(z_prior,c), x)
  + w_prior_batch_mean * MSE(mean_batch(x_prior), mean_batch(x))
  + w_prior_batch_std  * MSE(std_batch(x_prior),  std_batch(x))
```

The batch-standard-deviation term is important: it discourages the decoder from
matching only the conditional mean and helps preserve the real morphology
spread. On the 100-case test set, this variant improved both morphology errors
and unpaired distribution metrics versus `W_ref`, but its ostium attachment
metrics were slightly worse than the earlier stage3-surrogate models.

The final training objective for the selected method is:

```text
L_total =
    L_base
  + w_stage3_label2  * L_stage3_label2
  + w_stage3_center  * L_stage3_center
  + w_stage3_side    * L_stage3_side
  + w_stage3_opening * L_stage3_opening
  + L_prior
```

with:

| Loss weight | Value |
| --- | ---: |
| `w_mse` | 1.0 |
| `w_kl` | 0.01 |
| `w_scale` | 1.0 |
| `w_vert` | 250.0 |
| `w_stage3_label2` | 5.0 |
| `w_stage3_center` | 10.0 |
| `w_stage3_side` | 2.0 |
| `w_stage3_opening` | 10.0 |
| `w_prior_mse` | 0.25 |
| `w_prior_batch_mean` | 0.25 |
| `w_prior_batch_std` | 8.0 |

The corresponding training command is:

```bash
conda run --no-capture-output -n unified_env \
  bash methods/W_cond_ghd_vae/run_vessel_stage3surrogate_morph_priorcalib_aneug_ghds.sh 1
```

### No-morphology prior-calibration ablation

To isolate the contribution of morphology conditioning, we add the ablation
`W_stage3surrogate_priorcalib`. It keeps the same reference-compatible W
backbone, vessel-aware condition, Stage-3 surrogate weights, KL/free-bits
setting, and prior-path calibration as the final method, but removes the
morphology vector from the condition:

```text
final method:
  c = f_cond(vessel, ostium_ring, label2_points, ostium_params, morphology)

no-morph ablation:
  c = f_cond(vessel, ostium_ring, label2_points, ostium_params)
```

This ablation answers a clean methodological question: are the gains of the
final method primarily caused by prior-path calibration alone, or by the
combination of prior-path calibration and explicit morphology conditioning?

Training command:

```bash
conda run --no-capture-output -n unified_env \
  bash methods/W_cond_ghd_vae/run_vessel_stage3surrogate_priorcalib_aneug_ghds.sh 1
```

Completed run:

| Run | best epoch | best val total | Checkpoint |
| --- | ---: | ---: | --- |
| `W_vessel_stage3surrogate_priorcalib_seed1_20260504_010917` | 3200 | 1.404536 | `checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3surrogate_priorcalib/W_vessel_stage3surrogate_priorcalib_seed1_20260504_010917/models_best_val.pth` |

For comparison, the selected morphology-conditioned prior-calibrated run had
`best_val_total = 1.327942` at epoch 2550. The no-morph ablation therefore
looks worse on validation loss, but the decisive comparison is still the
strict-reference 100-case generation and the downstream sac-shape,
morphology-distribution, attachment, and diversity metrics.

The strict-reference 100-case generation finished successfully:

```text
/path/to/aneug_w_priorcalib_nomorph_exact_ref_all_test100_20260504_012305/reference_stitching/W_stage3surrogate_priorcalib
```

The result is informative rather than one-sided. The no-morph variant is
slightly worse than the morphology-conditioned prior-calibrated model in
direct paired Chamfer and validation loss, but it is best among the tested
variants in the unpaired global morphology-distribution metrics
(`coverage`, `MMD`, and `Frechet`). This means prior-path calibration is the
dominant driver for distribution-level realism, while morphology conditioning
still helps the paired reconstruction/conditioning objective.

The ablation
`W_stage3surrogate_morph_priorcalib_ostiumstrong` kept the prior calibration
unchanged but increased the Stage-3/ostium losses:

| Term | priorcalib | priorcalib_ostiumstrong |
| --- | ---: | ---: |
| stage3_label2 | 5 | 8 |
| stage3_center | 10 | 12 |
| stage3_nearest | 0 | 10 |
| stage3_side | 2 | 4 |
| stage3_opening | 10 | 25 |

This was tested to see whether stronger opening supervision could recover
`W_ref`-level attachment without losing the morphology distribution. It
improved `opening_center` slightly versus `priorcalib`, but worsened the
ring/label-2 attachment terms and degraded the distribution metrics. We
therefore keep it as an ablation, not as the selected method.

---

## 5. Inference Pipelines

### Strict reference stitching

The strict comparison uses the reference pipeline end-to-end except for the
Stage-1 sampler. The wrapper is:

```text
tools/run_strict_reference_stage3_batch.py
```

For `W_ref`, the wrapper calls the reference Stage-1 checkpoint directly. For
our W variants, it calls the same reference Stage-1 script but injects our
checkpoint through:

```text
--external_method_type W
--external_method_checkpoint <our checkpoint>
```

The strict wrapper calls the reference pipeline as:

```text
run_inference_pipeline.py all
--resample-aneurysm-to-vessel-resolution
--stitch
--stitch-method bridge
--stitch-bridge-steps 1
--smooth-ostium-transition
--smooth-ostium-iterations 10
--smooth-ostium-hops 2
```

For external W variants we pass `--skip-reconstruct`, because the reference
`reconstruct` branch requires the original CVAE encoder/posterior interface.
The stitched mesh is always produced from `outputs/stage1_sample/*_raw.obj`,
so skipping reconstruction does not change the generated-sample stitching
path.

This is the primary paper comparison because it follows
`vessel-mesh-editing-master` most closely.

### Normal AneuG stitching path

As a control, we also export our W variants through the AneuG sampler wrapper:

```text
tools/generate_and_stitch_reference.py
```

This path samples with AneuG's method loader, decodes through the reference
GHD fitter (`--decode_backend reference`), and then calls the same reference
Step-3 script. It is useful for debugging whether differences arise from
Stage-1 sampling/decode or from the final reference stitcher.

### Seam fairing extension

For qualitative seam inspection, we added a post-Reference optional fairing
step:

```text
tools/fair_reference_stitch_seams.py
```

This script does not replace the reference stitching. It copies the reference
output and applies constrained local fairing only in the ostium/bridge band
identified by `*_stitched_labels.npy` (`label == 2`). Two modes were tested:

1. local Taubin fairing, which was too conservative;
2. constrained harmonic band fairing, which reduced seam dihedral outliers.

On the 20-case visual run, the harmonic setting improved the seam-band
dihedral statistics:

| Setting | mean dihedral | p95 dihedral | mean max dihedral |
| --- | ---: | ---: | ---: |
| reference bridge/smooth | 5.87 deg | 17.57 deg | 138.45 deg |
| harmonic seam fairing | 5.21 deg | 15.40 deg | 117.36 deg |

The fairing output used for visual inspection was:

```text
outputs/strict_reference_Wnearest_bridge3_harmonic_h7_i80_test20_20260503_0015
```

This extension should be reported separately from the strict quantitative
comparison because it is an additional post-processing step.

---

## 6. Final 100-Case Test Export

The first strict full-test export for the reference baseline and the early W
variants was generated under:

```text
/path/to/aneug_w_variants_exact_ref_all_test100_20260503_122120
```

Manifest:

```text
/path/to/aneug_w_variants_exact_ref_all_test100_20260503_122120/manifest.json
```

Run summary:

```text
/path/to/aneug_w_variants_exact_ref_all_test100_20260503_122120/reference_stitching/<variant>/summary.json
```

The export contains:

| Pipeline | Variant | Cases | Status |
| --- | --- | ---: | --- |
| reference_stitching | W_ref | 100 | complete |
| reference_stitching | W_stage3surrogate | 100 | complete |
| reference_stitching | W_stage3nearest | 100 | complete |

The selected morphology/prior-calibrated method was exported separately after
the morphology experiments:

```text
/path/to/aneug_w_morph_priorcalib_exact_ref_all_test100_20260503_204853/reference_stitching/W_stage3surrogate_morph_priorcalib
```

The ostium-strong ablation was exported under:

```text
/path/to/aneug_w_morph_priorcalib_ostiumstrong_exact_ref_all_test100_20260503_213449/reference_stitching/W_stage3surrogate_morph_priorcalib_ostiumstrong
```

The earlier `normal_stitching` export is retained only as a debugging control.
It is not the primary fair comparison because it bypassed part of the
reference `all` pipeline setup.

Example output location:

```text
/path/to/aneug_w_variants_exact_ref_all_test100_20260503_122120/reference_stitching/W_stage3nearest/cases/test/C0003/outputs/final/
```

Important per-case files:

```text
<case>_generated_aneurysm_world.obj
<case>_vessel_with_generated_aneurysm_unstitched.obj
<case>_vessel_with_generated_aneurysm_stitched.obj
<case>_vessel_with_generated_aneurysm_stitched_labels.npy
<case>_sample_with_ostium_colored.ply
```

The export was produced by:

```text
tools/run_strict_reference_stage3_batch.py
```

with the reference baseline command:

```bash
conda run --no-capture-output -n unified_env python tools/run_strict_reference_stage3_batch.py \
  --cases_file checkpoints/aneug_ghds/splits/aneug_ghds_realcsv_opa20_seed42_20260502_013432/cases_test.json \
  --out_root /path/to/aneug_w_variants_exact_ref_all_test100_20260503_122120/reference_stitching/W_ref \
  --seed 1 \
  --num_samples 1 \
  --continue_on_error \
  --overwrite
```

The selected method was generated with the same strict reference wrapper, but
with our Stage-1 checkpoint injected into the reference pipeline:

```bash
conda run --no-capture-output -n unified_env python tools/run_strict_reference_stage3_batch.py \
  --cases_file checkpoints/aneug_ghds/splits/aneug_ghds_realcsv_opa20_seed42_20260502_013432/cases_test.json \
  --out_root /path/to/aneug_w_morph_priorcalib_exact_ref_all_test100_20260503_204853/reference_stitching/W_stage3surrogate_morph_priorcalib \
  --external_method_type W \
  --external_method_checkpoint checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3surrogate_morph_priorcalib/W_vessel_stage3surrogate_morph_priorcalib_seed1_20260503_203915/models_best_val.pth \
  --external_aneug_root /path/to/SynVA-A1/code/synva_method \
  --seed 1 \
  --num_samples 1 \
  --skip_reconstruct \
  --continue_on_error
```

This command writes the generated sac before stitching as:

```text
<case>_generated_aneurysm_world.obj
```

and the final strict-reference stitched mesh as:

```text
<case>_vessel_with_generated_aneurysm_stitched.obj
```

---

## 7. Quantitative Test Metrics

All metrics below are aggregated over the 100 held-out test cases from
`cases_test.json`. Values are read from each case's
`outputs/step3_compose_summary.json`; lower is better for all distances.
Distances are reported in the normalized patient-space units used by the
reference Stage-3 pipeline.

Metric definitions:

| Metric | Meaning |
| --- | --- |
| `ring_to_target` | Mean distance from the generated opening ring to the fitted target opening ring after Stage-3 alignment. |
| `ring_to_label2` | Mean distance from the generated opening ring to the label-2 ostium point cloud. |
| `label2_to_pouch` | Mean distance from the label-2 ostium points to the generated pouch surface. |
| `opening_center` | Distance between generated opening center and ostium center. |
| `nearest_vertex` | Distance from the ostium center to the nearest generated sac vertex. |
| `pouch_center` | Distance between generated sac center and ostium center. |

### Mean over 100 test cases

| Pipeline | Variant | n | ring_to_target | ring_to_label2 | label2_to_pouch | opening_center | nearest_vertex | pouch_center |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference-all | `W_ref` | 100 | 0.023541 | 0.018527 | 0.015175 | 0.011645 | **0.127311** | **0.118194** |
| Reference-all | `W_stage3surrogate` | 100 | **0.022862** | **0.017628** | **0.014606** | 0.011382 | 0.130102 | 0.123471 |
| Reference-all | `W_stage3nearest` | 100 | 0.022869 | 0.017760 | 0.014839 | **0.010513** | 0.130126 | 0.127953 |
| Reference-all | `W_stage3surrogate_morph` | 100 | 0.023618 | 0.018626 | 0.015125 | **0.010702** | 0.128546 | 0.123495 |
| Reference-all | `W_stage3surrogate_morph_priorcalib` | 100 | 0.023833 | 0.019024 | 0.015538 | 0.013155 | 0.128361 | 0.130863 |
| Reference-all | `W_stage3surrogate_morph_priorcalib_ostiumstrong` | 100 | 0.024033 | 0.019164 | 0.015776 | 0.012296 | 0.127645 | 0.130350 |
| Reference-all | `W_stage3surrogate_priorcalib` | 100 | 0.023974 | 0.019123 | 0.015564 | 0.013350 | **0.126316** | 0.131311 |

### Median over 100 test cases

| Pipeline | Variant | n | ring_to_target | ring_to_label2 | label2_to_pouch | opening_center | nearest_vertex | pouch_center |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference-all | `W_ref` | 100 | 0.013148 | 0.009397 | 0.007904 | **0.006374** | **0.096281** | **0.093912** |
| Reference-all | `W_stage3surrogate` | 100 | **0.012711** | **0.007829** | 0.007343 | 0.006659 | 0.099111 | 0.099465 |
| Reference-all | `W_stage3nearest` | 100 | 0.012715 | 0.008212 | **0.007163** | 0.006460 | 0.098234 | 0.101240 |
| Reference-all | `W_stage3surrogate_morph` | 100 | 0.013176 | 0.008439 | 0.007389 | 0.006395 | 0.097986 | 0.101281 |
| Reference-all | `W_stage3surrogate_morph_priorcalib` | 100 | 0.013106 | 0.008804 | 0.007601 | 0.007686 | 0.097455 | 0.106779 |
| Reference-all | `W_stage3surrogate_morph_priorcalib_ostiumstrong` | 100 | 0.013439 | 0.009615 | 0.008038 | 0.006974 | 0.098906 | 0.106885 |
| Reference-all | `W_stage3surrogate_priorcalib` | 100 | 0.013025 | 0.009377 | 0.007637 | 0.007787 | 0.096731 | 0.106669 |

### Pouch-to-GT shape and morphology metrics

For shape realism we additionally compare the generated sac mesh
`*_generated_aneurysm_world.obj` against the held-out GT aneurysm submesh from
`/path/to/prepared_meshes_3/<case>/05_submeshes/aneurysm_submesh.obj`. The
reported Chamfer uses 10k sampled surface points and symmetric nearest-neighbor
distances. Morphology errors are absolute errors from the robust morphology
backend.

| Variant | n | Chamfer mean | Chamfer median | D_max abs | H_max abs | W_max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `W_ref` | 100 | 0.094099 | 0.076495 | 0.113297 | 0.119853 | 0.074605 |
| `W_stage3surrogate_morph` | 100 | **0.082113** | **0.063320** | 0.107940 | 0.103746 | 0.061546 |
| `W_stage3surrogate_morph_priorcalib` | 100 | 0.086362 | 0.075390 | **0.084209** | **0.079572** | **0.057382** |
| `W_stage3surrogate_morph_priorcalib_ostiumstrong` | 100 | 0.086228 | 0.074650 | 0.088093 | 0.082618 | **0.056933** |
| `W_stage3surrogate_priorcalib` | 100 | 0.087192 | 0.069639 | 0.084195 | 0.081362 | 0.059200 |
| `W_stage3surrogate_morph_ostium` | 100 | 0.087998 | 0.068723 | 0.109904 | 0.112867 | 0.066577 |
| `W_stage3surrogate_morph_ostium_surface` | 100 | 0.095285 | 0.075525 | 0.107533 | 0.107045 | 0.065024 |

Interpretation: the original morphology variant remains best in direct
surface Chamfer, but prior calibration substantially improves the anatomical
size descriptors. It reduces the `D_max`, `H_max`, and `W_max` errors below
both `W_ref` and the previous morphology variants. The ostium-strong ablation
keeps the Chamfer close to prior calibration and slightly improves `W_max`, but
does not improve the main size descriptors. The no-morph prior-calibrated
ablation is close to the selected method on `D_max` and remains clearly better
than `W_ref` on the main size descriptors, but it is weaker than the
morphology-conditioned model in paired Chamfer and validation loss.

### Unpaired morphology-distribution metrics

To test the t-SNE observation quantitatively, we compare generated and GT
populations in standardized morphology-feature space using:

| Metric | Meaning |
| --- | --- |
| paired_z_l2 | Case-paired L2 distance in GT-standardized morphology space. |
| gen_to_gt_nn | Nearest-neighbor distance from each generated sac to the GT population. |
| gt_to_gen_nn | Nearest-neighbor distance from each GT sac to the generated population. |
| coverage | Fraction of GT sacs within the 95th-percentile GT leave-one-out NN radius of a generated sac. |
| MMD | Multi-bandwidth RBF maximum mean discrepancy; lower means closer distributions. |
| Frechet | Gaussian Frechet distance in morphology space; lower means closer mean/covariance. |

| Variant | paired_z_l2 ↓ | gen_to_gt_nn ↓ | gt_to_gen_nn ↓ | coverage ↑ | MMD ↓ | Frechet ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `W_ref` | 1.284607 | **0.256784** | **0.579981** | 0.84 | 0.006438 | 1.092620 |
| `W_stage3surrogate` | 1.217317 | 0.290853 | 0.663652 | 0.81 | 0.009033 | 1.111498 |
| `W_stage3surrogate_morph` | 1.127876 | 0.273468 | 0.669268 | 0.83 | 0.010141 | 1.254649 |
| `W_stage3surrogate_morph_priorcalib` | **0.980470** | 0.309438 | 0.601094 | **0.85** | **0.005519** | **1.017597** |
| `W_stage3surrogate_morph_priorcalib_ostiumstrong` | 0.996746 | 0.305389 | 0.613208 | 0.83 | 0.006162 | 1.061131 |
| `W_stage3surrogate_priorcalib` | 1.008820 | 0.313364 | 0.594569 | **0.87** | **0.005170** | **0.984928** |

Interpretation: `W_ref` indeed looks strong in t-SNE because it is close to
the GT population in unpaired nearest-neighbor space. Prior calibration
improves the more global distribution measures (`MMD`, `Frechet`, coverage)
and the paired morphology distance. The no-morph prior-calibrated ablation is
even stronger on the unpaired distribution metrics (`coverage`, `MMD`,
`Frechet`), which suggests that explicit morphology conditioning is not the
only source of distribution realism. However, because no-morph is worse in
validation loss and paired Chamfer, we report it as an important ablation
rather than replacing the selected morphology-conditioned method. The
ostium-strong follow-up improves `opening_center` versus prior calibration, but
worsens the ring/label-2 attachment terms and partially gives back the
distribution gains. This makes it a useful negative ablation rather than the
new default.

### Seam dihedral statistics over 100 test cases

The seam metrics are computed from each stitched OBJ and its
`*_stitched_labels.npy` labels. A seam edge is a face-adjacency edge whose
majority face labels differ. Lower is better.

| Variant | n | seam mean deg | seam p95 deg | seam max deg | seam >45 deg | seam >80 deg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `W_ref` | 100 | 5.205 | 14.551 | 34.339 | 0.56 | 0.17 |
| `W_stage3surrogate` | 100 | **4.857** | **13.970** | **33.129** | **0.47** | **0.09** |
| `W_stage3nearest` | 100 | 5.084 | 14.494 | 33.334 | 0.60 | 0.19 |

### Sac-only multi-sample diversity

To measure generative ability independently of the expensive final stitching
step, we exported sac-only samples with reference Step 1 and Step 2, but
without Step 3. This uses the same reference OPA/condition construction and
the same Stage-1 sampling script as the strict pipeline. For each of the 100
test cases, eight samples were generated.

The sac-only outputs are:

```text
/path/to/aneug_reference_sac_multisample_Wref_Wsurrogate_8x100_20260503_141954
/path/to/aneug_reference_sac_multisample_Wsurrogate_8x100_parallel_20260503_143800
/path/to/aneug_reference_sac_multisample_Wpriorcalib_8x100_20260503
```

The aggregate JSON is:

```text
outputs/sac_multisample_metrics_Wref_Wsurrogate_8x100_20260503.json
outputs/real_vs_generated_sac_diversity_with_priorcalib_20260503.json
```

Metrics:

| Metric | Meaning |
| --- | --- |
| `within-case vertex RMS diversity` | Mean pairwise RMS vertex distance between the eight sacs generated for the same case. |
| `max pair diversity per case` | Mean over cases of the maximum pairwise RMS vertex distance among the eight samples. |
| `centroid diversity` | Mean pairwise distance between sac centroids within each case. |
| `relative area diversity` | Mean pairwise relative surface-area difference within each case. |
| `area CV` | Coefficient of variation of sac surface area within each case. |
| `extent CV` | Coefficient of variation of the sac extent norm within each case. |

Metric computation details:

For each model variant and each test case, the sac-only sampler writes eight
raw meshes:

```text
<case>_sample_000_raw.obj
...
<case>_sample_007_raw.obj
```

All sacs are decoded from the same canonical GHD template and therefore share
the same vertex ordering and face topology (`8625` vertices, `17109` faces).
This fixed correspondence allows direct vertex-wise distances without nearest
neighbor matching. For one case with samples \(S_1,\dots,S_8\), each sample is
a vertex matrix \(V_i \in R^{N \times 3}\), \(N=8625\).

The pairwise vertex RMS between two samples is:

```text
RMS(i,j) = sqrt(mean_k ||V_i[k] - V_j[k]||_2^2)
```

The reported `within-case vertex RMS diversity` is the mean of this value over
all \(\binom{8}{2}=28\) sample pairs, then averaged over the 100 test cases.
`max pair diversity per case` is the maximum RMS among the 28 pairs for each
case, again averaged over cases.

For every sample we also compute:

```text
centroid = mean_k V[k]
area     = trimesh surface area
extent   = bounding_box_max - bounding_box_min
extent_norm = ||extent||_2
```

`centroid diversity` is the mean pairwise centroid distance over the 28 pairs
within each case, averaged over cases. `relative area diversity` is the mean
pairwise relative area difference:

```text
area_rel(i,j) = |area_i - area_j| / (0.5 * (area_i + area_j))
```

`area CV` and `extent CV` are computed within each case as:

```text
CV = std(samples) / mean(samples)
```

and then averaged across cases. These CV values quantify how much the eight
samples for the same ostium vary in global size.

| Variant | n cases | n samples | vertex RMS diversity | max pair diversity | centroid diversity | relative area diversity | area CV | extent CV | mean area | mean extent norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `W_ref` | 100 | 800 | 0.0187 | 0.0322 | 0.00765 | 0.0404 | 0.0318 | 0.0171 | 2.0325 | 1.5139 |
| `W_stage3surrogate` | 100 | 800 | **0.0383** | **0.0519** | **0.01151** | **0.0492** | **0.0384** | **0.0209** | 2.3359 | 1.5884 |
| `W_stage3surrogate_morph_priorcalib` | 100 | 800 | 0.0293 | 0.0463 | 0.01108 | **0.0521** | **0.0413** | **0.0214** | 2.3188 | 1.5840 |

Interpretation: both learned W variants are more diverse than `W_ref`.
`W_stage3surrogate` has the largest vertex-level diversity, while the final
`W_stage3surrogate_morph_priorcalib` sits between `W_ref` and
`W_stage3surrogate` in vertex RMS but has the largest area/extent variation.
This is consistent with the prior-calibration objective: it does not maximize
arbitrary vertex displacement, but it restores population-level size and shape
spread.

Scale interpretation:

The sac-only RMS values are measured in the same normalized canonical/world
coordinate frame as the exported Stage-1 sacs. They should be interpreted
relative to the typical generated sac size, not as millimeters. The mean sac
extent norm is 1.5139 for `W_ref` and 1.5884 for `W_stage3surrogate`.
Therefore:

| Variant | vertex RMS / mean extent | max pair RMS / mean extent | centroid diversity / mean extent | relative area diversity |
| --- | ---: | ---: | ---: | ---: |
| `W_ref` | 1.23% | 2.13% | 0.51% | 4.04% |
| `W_stage3surrogate` | **2.41%** | **3.27%** | **0.72%** | **4.92%** |
| `W_stage3surrogate_morph_priorcalib` | 1.85% | 2.92% | 0.70% | **5.21%** |

Thus the diversity is visible but not chaotic: the generated samples remain in
the same ostium-conditioned shape family. `W_stage3surrogate` explores roughly
twice as much vertex-level variation as `W_ref` relative to sac size, while the
final prior-calibrated model remains more moderate in vertex displacement but
shows the strongest relative area variation. This is useful for conditional
synthesis because the model adds sample variability without leaving the
case-conditioned sac family.

### Real held-out sac scale

We also compared the generated sac-only samples against the 100 real held-out
test sacs exported as `stage1_reconstruct/*_target_raw.obj` in the corrected
strict-reference `W_ref` run. These targets have the same template
correspondence as the generated sacs, so pairwise vertex RMS distances are
directly comparable.

The real target meshes are produced by the reference inference script in
`mode reconstruct`, but the file named `*_target_raw.obj` is the ground-truth
test sac decoded through the case fitter, not the model reconstruction. We use
these target meshes only as a scale and population-variation reference.

The aggregate JSON is:

```text
outputs/real_vs_generated_sac_diversity_20260503.json
outputs/real_vs_generated_sac_diversity_with_priorcalib_20260503.json
```

The current reproducible evaluator is:

```text
tools/evaluate_sac_multisample_diversity.py
```

Real held-out test sac statistics:

| Real test sacs | n | vertex RMS across cases | centroid distance across cases | relative area difference | mean area | mean extent norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| targets | 100 | 0.1885 | 0.1296 | 0.3861 | 2.2569 | 1.5817 |

For the real test sacs, the same quantities are computed across cases instead
of within a single case. Let \(T_a\) and \(T_b\) be two different real test
sacs in the aligned template correspondence. `vertex RMS across cases` is the
mean pairwise RMS over all \(\binom{100}{2}=4950\) pairs:

```text
real_RMS(a,b) = sqrt(mean_k ||T_a[k] - T_b[k]||_2^2)
```

The real centroid and area statistics are computed analogously over all
4950 cross-case pairs. The ratios in the next table divide generated
within-case diversity by this real cross-case diversity:

```text
within-case vertex RMS / real cross-case RMS
  = mean_case mean_sample_pairs RMS(S_i, S_j)
    / mean_real_case_pairs RMS(T_a, T_b)
```

The area and extent ratios compare generated sample means to the real target
means:

```text
mean area / real mean area
mean extent / real mean extent
```

These ratios answer two different questions:

1. how much variation the model generates for one fixed ostium relative to
   the anatomical variation across different real patients;
2. whether the generated sacs live at the correct global size scale.

Generated samples relative to the real test-sac scale:

| Variant | within-case vertex RMS / real cross-case RMS | centroid diversity / real centroid diversity | area diversity / real area diversity | mean area / real mean area | mean extent / real mean extent |
| --- | ---: | ---: | ---: | ---: | ---: |
| `W_ref` | 9.92% | 5.91% | 10.47% | 90.06% | 95.71% |
| `W_stage3surrogate` | **20.31%** | **8.88%** | **12.73%** | **103.50%** | **100.42%** |
| `W_stage3surrogate_morph_priorcalib` | 15.56% | 8.55% | **13.50%** | 102.75% | 100.15% |

Interpretation: `W_stage3surrogate` covers roughly one fifth of the real
cross-case vertex-level variation within a fixed ostium condition, while the
final prior-calibrated model covers about 15.6%. This is still clearly more
diverse than `W_ref` (9.9%), but less aggressive than the pure stage3-surrogate
model. The final model matches the real global scale particularly well:
mean area is 102.75% of the real mean and mean extent is 100.15% of the real
mean. This supports the main conclusion from the morphology-distribution
metrics: prior calibration improves realism and scale calibration, while
keeping conditional diversity in a moderate range.

Interpretation:

1. With the corrected reference `all` pipeline, all ring/label2 metrics are
   much tighter than in the earlier non-strict export.
2. `W_stage3surrogate` gives the best mean ring-to-target, ring-to-label2,
   label2-to-pouch, and seam-dihedral metrics.
3. `W_stage3nearest` gives the best mean opening-center distance and the best
   median label2-to-pouch distance, but it is not consistently better at the
   seam.
4. The original `W_ref` remains slightly better on sac-center/nearest-center
   distance metrics. These are global placement proxies and should be
   interpreted together with ring and seam metrics.

For paper reporting, the primary quantitative table should use the three
`Reference-all` rows. The old non-strict `normal_stitching` rows should not be
used as the main comparison.

### World-space aneurysm Chamfer and morphology comparison

The final geometry comparison against real held-out aneurysm sacs uses
`/path/to/synva_real_data` as ground truth. Each generated sac is compared only to
the corresponding real aneurysm submesh, not to the full stitched vessel:

```text
GT:   /path/to/synva_real_data/<matched_case>/05_submeshes/aneurysm_submesh.obj
Pred: <reference_stitching>/<variant>/cases/test/<case>/outputs/final/<case>_generated_aneurysm_world.obj
```

The generated `*_generated_aneurysm_world.obj` files and the real
`aneurysm_submesh.obj` files were checked by bounding boxes and centroids and
are in the same patient/world coordinate frame. Chamfer is computed from 10,000
surface samples per mesh with the exact symmetric `torch.cdist` protocol:

```python
vertices_pred = transformed_mesh.sample(10000)
vertices_pred = torch.tensor(vertices_pred, device=device, dtype=torch.float32)
vertices_gt = aneurysm_submesh.sample(10000)
vertices_gt = torch.tensor(vertices_gt, device=device, dtype=torch.float32)
dist1 = torch.cdist(vertices_gt, vertices_pred, p=2).min(dim=1)[0]
dist2 = torch.cdist(vertices_pred, vertices_gt, p=2).min(dim=1)[0]
chamfer_loss = dist1.mean() + dist2.mean()
```

Morphological parameters are computed through `/path/to/compute_morphology.py`
with `morphology-backend=exact`. For generated sacs, the stitched mesh labels
are used only to recover the generated aneurysm and ostium labels needed by
the morphology code; Chamfer itself is always computed on the standalone
generated aneurysm world mesh.

Combined summary files are stored without overwriting the per-run outputs:

```text
analysis_results/combined_model_summaries_20260503/combined_summary_by_variant.csv
analysis_results/combined_model_summaries_20260503/chamfer_summary.md
```

The current 100-case held-out summaries, sorted by mean Chamfer, are:

| Rank | Variant | n | Mean Chamfer 10k | Median Chamfer 10k | Source |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `W_stage3surrogate_morph_priorcalib` | 100 | 0.0770447481 | 0.0654586032 | `reference_stitching_W_stage3surrogate_morph_priorcalib_vs_synva_real_20260503` |
| 2 | `W_stage3surrogate_priorcalib` | 100 | 0.0778353974 | 0.0589322802 | `reference_stitching_W_stage3surrogate_priorcalib_vs_synva_real_20260504` |
| 3 | `W_stage3surrogate_morph` | 100 | 0.0821132694 | 0.0633201487 | `reference_stitching_W_stage3surrogate_morph_vs_synva_real_20260503` |
| 4 | `W_stage3surrogate_morph_priorcalib_ostiumstrong` | 100 | 0.0862282205 | 0.0746500292 | `reference_stitching_gt_compare_morph_priorcalib_ostiumstrong_20260503` |
| 5 | `W_stage3surrogate_morph_ostium_surface` | 100 | 0.0863451694 | 0.0673070475 | `reference_stitching_W_stage3surrogate_morph_ostium_surface_vs_synva_real_20260503` |
| 6 | `W_stage3surrogate_morph_priorcalib` | 100 | 0.0863619591 | 0.0753899849 | `reference_stitching_gt_compare_morph_priorcalib_20260503` |
| 7 | `W_stage3surrogate_morph_ostium_shape` | 100 | 0.0875795295 | 0.0693737902 | `reference_stitching_W_stage3surrogate_morph_ostium_shape_vs_synva_real_20260503` |
| 8 | `W_stage3surrogate_morph_ostium` | 100 | 0.0879977285 | 0.0687231235 | `reference_stitching_W_stage3surrogate_morph_ostium_vs_synva_real_20260503` |
| 9 | `W_stage3surrogate` | 100 | 0.0891478563 | 0.0696162730 | `reference_stitching_vs_synva_real_20260503` |
| 10 | `W_stage3nearest` | 100 | 0.0896235806 | 0.0711113438 | `reference_stitching_vs_synva_real_20260503` |
| 11 | `W_ref` | 100 | 0.0940989243 | 0.0764946267 | `reference_stitching_vs_synva_real_20260503` |
| 12 | `W_stage3surrogate_morph_ostium_surface` | 100 | 0.0952847327 | 0.0755254961 | `reference_stitching_gt_compare_morph_ostium_surface_20260503` |
| 13 | `W_stage3surrogate_morph_ostium_shape` | 100 | 0.0964514540 | 0.0792780095 | `reference_stitching_gt_compare_morph_ostium_shape_20260503` |

The repeated `priorcalib`, `ostium_surface`, and `ostium_shape` entries come
from separate completed result folders and are intentionally kept separate by
`Source` rather than merged.

---

## 8. What Changed Relative to the Earlier A/C/D/E Experiments

The earlier method suite in `methods/` still exists and remains useful for
architecture exploration:

| Tag | Method family |
| --- | --- |
| A | PCA latent + conditional flow matching |
| B | Mixture-prior conditional VAE |
| C | FSQ-VAE + autoregressive transformer |
| D | VQ-VAE + autoregressive transformer |
| E_C / E_D | C/D with collision-aware terms |

However, those runs mixed our own decode/stitching assumptions with the
reference pipeline and were not the cleanest basis for comparison against
`vessel-mesh-editing-master`. The current paper-facing protocol therefore
uses the reference-compatible W setup:

1. use the same `/path/to/aneug-ghds` GHD tokens and alignment files as the
   reference repository;
2. use the same real CSV split;
3. keep the reference Stage-3 alignment and bridge stitching fixed;
4. replace only the Stage-1 generative model;
5. report optional seam fairing as an explicit post-processing extension.

This removes the main confound discovered during visual debugging: poor sac
appearance could be caused either by the model or by decode/stitch coordinate
mismatch. The strict reference path makes that separation explicit.

---

## 9. Reporting Recommendation

For the paper methods section, the clean narrative is:

1. Define `W_ref` as the external reference baseline.
2. Introduce `W_stage3surrogate` as the first reference-compatible extension:
   same W architecture, richer vessel conditioning, plus differentiable
   Stage-3 alignment losses.
3. Introduce morphology conditioning as the case-conditioned shape extension.
4. Introduce prior calibration as the generative-distribution fix: it trains
   the inference-time prior path directly and matches batch-level morphology
   spread.
5. Report `W_stage3surrogate_morph_priorcalib` as the selected
   morphology-conditioned method: it improves paired sac shape and the main
   size descriptors versus `W_ref`, while keeping the strict reference
   stitcher fixed.
6. Report `W_stage3surrogate_priorcalib` as the no-morph ablation: it is the
   strongest tested variant on unpaired distribution metrics, but does not
   replace the selected method because it is worse in validation loss and
   paired Chamfer.
7. Report `W_stage3surrogate_morph_priorcalib_ostiumstrong` as an ablation:
   stronger ostium losses help `opening_center`, but do not recover W_ref-level
   attachment and slightly degrade the prior-calibrated morphology distribution.
8. Treat harmonic seam fairing as a qualitative post-processing extension,
   not as part of the strict model comparison.

This is the current fair comparison setup.
