# Notices and Provenance

This preliminary repository is provided for paper-review inspection only. It is
not a final public release. The SynVA-A1 project as a whole does not grant an
open-source license unless a separate license is supplied; bundled
subcomponents may carry their own upstream licenses.

## Included source components

- `code/inference` and `code/AneuG-Own-edit`: reference GHD fitting, ostium/OPA,
  Stage-1 inference, and stitching infrastructure originating from the local
  `vessel-mesh-editing-master` source tree.
- `code/synva_method`: SynVA-A1 model code, ablation methods, training wrappers,
  generation wrappers, evaluation tools, and method documentation.
- `code/AneuG-Own-edit/baselines`: baseline code retained from the copied
  AneuG/reference tree for provenance and compatibility.

The source-tree names above are retained for attribution and technical
traceability. Local machine paths, user names, SSH keys, datasets, canonical
meshes, model weights, and generated outputs are intentionally not included.

## Not included

- Model checkpoints and trained weights (`*.pt`, `*.pth`, `*.pkl`).
- Meshes, canonical assets, NumPy data dumps, and generated outputs.

## License status

No standalone project-level license file was found in the local source trees
used to assemble this preliminary repository. Until the final release adds the
appropriate project license or written permissions, treat the SynVA-A1 project
code as review-only material: no permission is granted to redistribute,
sublicense, or reuse that code outside paper review without separate
authorization.

Bundled subcomponents may carry their own licenses. The Michelangelo baseline
under `code/AneuG-Own-edit/baselines/Michelangelo` declares GPL-3.0 in its
README and includes the GPL-3.0 text in `LICENSE`. If that baseline is retained
in a public release, the release must continue to satisfy the corresponding GPL
obligations and preserve upstream attribution.

External dependencies such as PyTorch, PyTorch3D, trimesh, pymeshlab, NumPy,
SciPy, scikit-learn, and related packages are not vendored in this repository.
Their use is governed by their respective upstream licenses.
