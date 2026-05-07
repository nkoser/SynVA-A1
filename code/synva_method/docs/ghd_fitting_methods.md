# Graph Harmonic Deformation Fitting

## Methods-Style Description

We represent each aneurysm pouch as a smooth deformation of a fixed canonical mesh. Given a canonical mesh \(M_c=(V_c,F_c)\) and a target mesh \(M_t=(V_t,F_t)\), the fitting procedure estimates a low-dimensional graph harmonic deformation (GHD) coefficient matrix together with a global similarity transform. The fitted parameters are later stored as `GHD_coefficient`, rotation `R`, scale `s`, and translation `T` in `ghb_fitting_checkpoint.pkl`.

### Preprocessing and Opening Alignment

For each case, the pipeline assumes an aligned mesh `part_aligned.obj` and an opening checkpoint `opa_checkpoint.pkl`. The checkpoint stores the selected ostium boundary vertices, reconstructed triangulated opening patches, average opening normals, and maps from reconstructed opening vertices back to mesh vertex indices. During fitting, both canonical and target cases are loaded through `RegistrationwOpeningAlignmentwDifferentiableCentreline`. In the default pouch-only configuration, only one opening is used; centreline and differentiable occupancy losses for full vessel complexes are disabled.

Meshes are normalized before optimization. With hard normalization enabled, the canonical radius is computed from the maximum vertex norm and multiplied by 1.10. When `keep_size=False`, as used by the current batch fitting entry point, the canonical and target are independently normalized by their own radii. The optional `center_opening_at_origin` mode first translates each mesh such that the selected opening centroid is at the origin.

### Graph Harmonic Parameterization

The deformation basis is computed on the canonical mesh graph. The implementation builds three Laplacian operators: a cotangent Laplacian, a normalized Laplacian, and a symmetric graph Laplacian. These are combined as

\[
L_{\mathrm{mix}} = \alpha L_{\mathrm{cot}} + \beta L_{\mathrm{norm}} + \gamma L_{\mathrm{sym}},
\]

with default weights \((\alpha,\beta,\gamma)=(1.0,0.1,0.1)\). The \(K\) smallest-magnitude eigenvectors of \(L_{\mathrm{mix}}\) are retained, with the default \(K=12^2=144\). Let \(\Phi\in\mathbb{R}^{|V_c|\times K}\) denote these eigenvectors and \(A\in\mathbb{R}^{K\times 3}\) the learnable GHD coefficients. The canonical vertex offsets are reconstructed as

\[
\Delta V = \Phi A.
\]

The final deformed mesh is then obtained by applying a global similarity transform:

\[
\hat{V} = \left(V_c + \Delta V\right) R^\top s + T,
\]

where \(R\) is represented as an axis-angle vector, \(s\) is a scalar scale parameter, and \(T\in\mathbb{R}^3\) is a translation. The same deformation and transform are applied to the reconstructed ostium opening patch by indexing \(\Delta V\) through the stored opening-to-mesh vertex map. This keeps the opening geometry differentiably tied to the full mesh deformation.

To avoid recomputing the spectral basis for every target, the code can load or write a canonical eigen checkpoint containing `GBH_eigval` and `GBH_eigvec`.

### Objective

For each target case, fitting is posed as a direct differentiable registration problem in the GHD parameter space. The optimizer minimizes a weighted sum of surface correspondence, ostium alignment, and deformation regularization terms:

\[
\mathcal{L}
= \lambda_{p0}\mathcal{L}_{p0}
+ \lambda_{n1}\mathcal{L}_{n1}
+ \lambda_{\mathrm{lap}}\mathcal{L}_{\mathrm{lap}}
+ \lambda_{\mathrm{edge}}\mathcal{L}_{\mathrm{edge}}
+ \lambda_{\mathrm{cons}}\mathcal{L}_{\mathrm{cons}}
+ \lambda_{\mathrm{rigid}}\mathcal{L}_{\mathrm{rigid}}
+ \lambda_{\mathrm{op,p}}\mathcal{L}_{\mathrm{op,p}}
+ \lambda_{\mathrm{op,n}}\mathcal{L}_{\mathrm{op,n}}
+ \lambda_{\mathrm{area}}\mathcal{L}_{\mathrm{area}}
+ \lambda_{\mathrm{vol}}\mathcal{L}_{\mathrm{vol}}
+ \lambda_{\mathrm{op,smooth}}\mathcal{L}_{\mathrm{op,smooth}}
+ \lambda_{\mathrm{op,overlap}}\mathcal{L}_{\mathrm{op,overlap}}
+ \lambda_{\mathrm{occ}}\mathcal{L}_{\mathrm{occ}}.
\]

Here \(\hat{M}=(\hat{V},F_c)\) denotes the warped canonical mesh, \(M_t\) the target mesh, \(\hat{O}\) the warped canonical ostium patch, and \(O_t\) the selected target ostium patch. The default `pouch_only` configuration sets `num_op=1`, matches the target ostium by the minimum opening-surface Chamfer distance, and disables the full-vessel differentiable occupancy and centreline losses.

#### Surface Matching Terms

The primary shape-matching term is a symmetric Chamfer distance between points sampled from the predicted and target surfaces. Let \(P(\hat{M})\) and \(P(M_t)\) be point sets sampled uniformly from mesh faces. The position loss is

\[
\mathcal{L}_{p0}
= \frac{1}{|P(\hat{M})|}\sum_{x\in P(\hat{M})}\min_{y\in P(M_t)}\|x-y\|_2^2
+ \frac{1}{|P(M_t)|}\sum_{y\in P(M_t)}\min_{x\in P(\hat{M})}\|y-x\|_2^2.
\]

The companion normal term \(\mathcal{L}_{n1}\) uses the normals returned by the same sampled surface points and penalizes disagreement between nearest-neighbor normals. Together, these terms pull the warped canonical surface onto the target while discouraging fits that match point locations with inconsistent local orientation. The default implementation samples `sample_num = 2.5e5` surface points, giving this term high statistical coverage of the pouch geometry.

#### Mesh Regularization Terms

The optimization is intentionally constrained: the spectral basis already limits the deformation to smooth modes, but the loss also penalizes local artifacts that can still appear under strong surface matching.

The cotangent Laplacian smoothing term \(\mathcal{L}_{\mathrm{lap}}\) is computed with PyTorch3D's cotangent mesh smoothing. It penalizes high-frequency surface variation in the deformed mesh and acts as a local smoothness prior. The edge-length loss \(\mathcal{L}_{\mathrm{edge}}\) compares deformed edge lengths against the average canonical edge length, preventing the optimizer from matching the target through local edge stretching or compression. The mesh normal consistency term \(\mathcal{L}_{\mathrm{cons}}\) penalizes discontinuities between adjacent face normals and is weighted strongly in the default setup to suppress folds and flipped triangles.

The ARAP-style rigidity loss \(\mathcal{L}_{\mathrm{rigid}}\) is computed over the canonical mesh neighborhood graph. For each vertex, the method gathers its cotangent-weighted one-ring neighborhood in the canonical mesh and in the warped mesh. It solves a local orthogonal Procrustes alignment and penalizes the residual between the rotated canonical neighborhood and the deformed neighborhood:

\[
\mathcal{L}_{\mathrm{rigid}}
= \frac{1}{Z}\sum_i\sum_{j\in\mathcal{N}(i)}
w_{ij}\left\|R_i(v_j-v_i)-(\hat{v}_j-\hat{v}_i)\right\|_2,
\]

where \(R_i\) is the best local rotation and \(w_{ij}\) are cotangent-derived edge weights. This term favors locally rigid deformations, allowing the global pouch shape to move while discouraging implausible local shearing.

#### Ostium Alignment Terms

The ostium is handled as a first-class geometric object rather than an incidental part of the surface Chamfer loss. The OPA checkpoint stores a triangulated opening patch and a mapping from opening vertices to canonical mesh vertices. During every forward pass, the same GHD displacement field is applied to both \(\hat{M}\) and \(\hat{O}\), so opening losses backpropagate directly into the spectral coefficients.

The opening position loss \(\mathcal{L}_{\mathrm{op,p}}\) is a Chamfer distance between points sampled on \(\hat{O}\) and \(O_t\). This gives a dense ostium-level alignment signal even when the opening contributes only a small fraction of the total mesh area. The opening normal term \(\mathcal{L}_{\mathrm{op,n}}\) avoids stochastic face-normal instability by estimating a robust ring normal from the ordered opening boundary vertices:

\[
\mathcal{L}_{\mathrm{op,n}}
= 1 - \left|\langle n(\hat{O}), n(O_t)\rangle\right|.
\]

The absolute value makes the term robust to sign ambiguity in the triangulated opening normal.

The opening area barrier prevents ostium collapse. Let \(A(\hat{O})\) be the warped opening area and \(A(O_c)\) the canonical reference opening area. With minimum ratio \(\rho\), default `opening_min_ratio = 0.5`, the penalty is

\[
\mathcal{L}_{\mathrm{area}}
= \frac{\max(0, \rho A(O_c)-A(\hat{O}))}{A(O_c)+\epsilon}.
\]

Thus the optimizer is not forced to match an exact area; it is only penalized when the opening shrinks below a safe fraction of the reference ostium.

The opening overlap loss adds a stricter spatial overlap criterion. The code samples point sets on \(\hat{O}\) and \(O_t\), computes symmetric nearest-neighbor distances, and converts them into a soft overlap score with a Gaussian kernel:

\[
S(A,B)
= \frac{1}{2}\left[
\frac{1}{|A|}\sum_{a\in A}\exp\left(-d(a,B)^2/\sigma^2\right)
+ \frac{1}{|B|}\sum_{b\in B}\exp\left(-d(b,A)^2/\sigma^2\right)
\right].
\]

The bandwidth \(\sigma\) is set from the target opening radius using `opening_overlap_sigma_ratio`. The score is normalized by a target self-overlap estimate and multiplied by the smaller-to-larger opening area ratio. The final loss is

\[
\mathcal{L}_{\mathrm{op,overlap}} = 1 - \mathrm{clip}(S_{\mathrm{norm}}\cdot r_{\mathrm{area}}, 0, 1).
\]

This term is small only when the predicted and target openings occupy the same spatial region and have compatible areas.

The opening boundary smoothness term regularizes the ostium rim. It contains a cyclic second-difference penalty on the ordered boundary ring,

\[
\frac{1}{N}\sum_i \|\hat{o}_{i-1}-2\hat{o}_i+\hat{o}_{i+1}\|_2^2,
\]

plus an umbrella-Laplacian penalty over a small \(k\)-ring neighborhood around the opening boundary. This suppresses jagged neck artifacts while preserving the global pouch deformation.

#### Volume and Optional Occupancy Terms

The volume term compares the absolute enclosed volumes of predicted and target meshes:

\[
\mathcal{L}_{\mathrm{vol}}
= \frac{|\mathrm{Vol}(\hat{M})-\mathrm{Vol}(M_t)|}
{\mathrm{Vol}(M_t)+\epsilon}.
\]

It provides a coarse global constraint that complements the surface Chamfer loss, especially when local surface samples can be matched by thin or collapsed configurations.

The optional grid occupancy term \(\mathcal{L}_{\mathrm{occ}}\) is disabled by default in the current pouch-only settings because `loss_grid_occupancy = 0`. When enabled, the method builds a shared 3D grid over the union of canonical and target bounding boxes, evaluates a soft winding-number occupancy field for the target once, and compares it against the warped mesh occupancy using MSE or Dice loss. This term encourages agreement of inside/outside structure rather than only surface proximity.

#### Default Weights

In the current `ghd_fitting.py` defaults for pouch-only fitting, the active base weights are:

```text
loss_p0                       = 20.0
loss_n1                       = 16.0
loss_laplacian                = 0.1
loss_edge                     = 0.1
loss_consistency              = 350.0
loss_rigid                    = 100.0
loss_openings_p               = 10.0
loss_openings_n               = 0.1
loss_opening_area             = 1.0
loss_volume                   = 0.05
loss_opening_boundary_smooth  = 0.5
loss_opening_overlap          = 1.0
loss_grid_occupancy           = 0.0
```

Zero-weight losses are pruned before optimization, so the grid occupancy term is not constructed unless explicitly enabled. The large normal-consistency and rigidity weights reflect that the GHD fit is not only a closest-surface registration problem: it must produce a reusable, anatomically plausible coefficient vector for downstream generative modeling.

For non-pouch-only fitting, the code can additionally include differentiable occupancy (`loss_do`) and centreline Chamfer (`loss_diff_centreline`). Occupancy supervision is generated by sampling query points in a bounding box around canonical and target meshes and labeling them via the target winding number. The current pouch-only path deliberately disables these terms to focus the fitting objective on aneurysm pouch geometry and ostium fidelity.

### Optimization

The fitting variables are the spectral coefficient matrix \(A\), scale \(s\), rotation vector \(R\), and translation \(T\). They are initialized as zero deformation, identity scale, zero rotation, and zero translation, and optimized with AdamW. The default learning rate is \(10^{-3}\), with a StepLR decay every 2500 iterations by factor 0.75. Each epoch performs:

1. reconstruct the deformed mesh and its opening patch from the current GHD parameters;
2. compute the currently active loss dictionary;
3. multiply every loss by its per-epoch weight;
4. backpropagate the summed objective;
5. update \(A,s,R,T\);
6. log losses, write visualizations, and periodically save intermediate checkpoints.

The `strategy_v1_linear` loss weighter warms all active terms from zero to their configured value during the first 10% of the scheduled epochs. The optional two-stage schedule first prioritizes coarse global matching, then increases opening-specific penalties for refinement. Adaptive fitting can also score the target difficulty from opening centroid, area, and normal mismatch, then increase epochs and opening-related weights for medium or hard cases.

### Outputs

For each target case, the final result is written to:

```text
<save_root>/<case_id>/<meta>/ghb_fitting_checkpoint.pkl
```

The checkpoint contains the optimized global transform and GHD coefficients:

```text
R, s, T, GHD_coefficient
```

The run directory also contains `run_config.json`, `loss_log.jsonl`, TensorBoard logs, intermediate checkpoints, and OBJ visualizations of the warped mesh. Reconstruction replays the same formula \(V_c+\Phi A\), followed by the stored similarity transform and, if needed, denormalization back to the aligned mesh scale.

## Code Trace

The description above is based on the following files in `/path/to/reference-vessel-mesh-editing/code/AneuG-Own-edit`:

- `ghd_fitting.py`: command-line entry point, default loss weights, pouch-only mode, multi-GPU case dispatch.
- `ghd/fitting/fitter.py`: registration initialization, adaptive/two-stage schedule, optimization loop, checkpointing.
- `ghd/base/graph_harmonic_deformation.py`: mixed Laplacian construction, eigenbasis loading/computation, GHD forward pass.
- `ghd/fitting/registration.py`: mesh loading, ostium checkpoint loading, opening patch reconstruction and normalization.
- `ghd/losses/meshloss.py`: Chamfer, normal, Laplacian, edge, consistency, and ARAP-style rigidity losses.
- `ghd/losses/meshloss_pouch.py`: pouch-only ostium losses, opening area/overlap/smoothness, volume and optional grid occupancy terms.
- `models/ghd_reconstruct.py`: replay of fitted checkpoints into PyTorch3D meshes.
