"""Create a canonical eigenvector checkpoint for GHD fitting and VAE training.

This script computes eigenvectors on the NORMALIZED canonical mesh (the same
normalization that GHD fitting applies internally) and saves them as a pickle
checkpoint.  Both ``ghd_fitting.py`` and the Stage-1 VAE scripts
(``first_stage_ostium_conditional.py``, ``infer_stage1_ostium_conditional.py``)
must use the **same** eigenvector checkpoint so that the GHD coefficients
produced during fitting are meaningful when reconstructed by the VAE.

Usage
-----
    python create_canonical_eigen_checkpoint.py \
        --canonical checkpoints-new/canonical_model/part_aligned.obj \
        --output    checkpoints-new/canonical_model/canonical_model_144_normed.pkl \
        --num-basis 144

Then pass the output file to GHD fitting:

    python ghd_fitting.py ... --canonical_eigen_chk checkpoints-new/canonical_model/canonical_model_144_normed.pkl

The VAE training / inference scripts already default to this path.
"""

import argparse
import pickle

import torch
from pytorch3d.structures import Meshes

from ghd.base.graph_harmonic_deformation import Graph_Harmonic_Deform
from utils.utils import safe_load_mesh


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create canonical eigenvector checkpoint for GHD pipeline."
    )
    parser.add_argument(
        "--canonical",
        type=str,
        default="checkpoints-new/canonical_model/part_aligned.obj",
        help="Path to the raw canonical mesh OBJ file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints-new/canonical_model/canonical_model_144_normed.pkl",
        help="Output path for the eigenvector checkpoint pickle.",
    )
    parser.add_argument(
        "--num-basis",
        type=int,
        default=144,
        help="Number of eigenvector bases (default: 144 = 12^2).",
    )
    parser.add_argument(
        "--mix-lap-weights",
        type=float,
        nargs=3,
        default=[1.0, 0.1, 0.1],
        help="Weights for [cotangent, normal, standard] Laplacian mixing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading canonical mesh: {args.canonical}")
    canonical_raw = safe_load_mesh(args.canonical)
    v_raw = canonical_raw.verts_packed()

    # Normalize exactly as GHD fitting does: max_vertex_norm * 1.10
    norm_canonical = torch.max(torch.norm(v_raw, dim=-1)).item() * 1.10
    v_normed = v_raw / norm_canonical
    canonical_normed = Meshes(verts=[v_normed], faces=canonical_raw.faces_list())

    print(f"Canonical mesh: {v_raw.shape[0]} vertices, norm_canonical={norm_canonical:.6f}")
    print(f"Normalized vertex range: [{v_normed.min():.4f}, {v_normed.max():.4f}]")
    print(f"Computing {args.num_basis} eigenvectors (mix_lap_weights={args.mix_lap_weights}) ...")

    ghd = Graph_Harmonic_Deform(
        canonical_normed,
        num_Basis=args.num_basis,
        mix_lap_weight=args.mix_lap_weights,
    )

    eigval = ghd.GBH_eigval.detach().cpu()
    eigvec = ghd.GBH_eigvec.detach().cpu()

    # Sanity check: eigenvectors should be orthonormal
    ortho_err = (eigvec.T @ eigvec - torch.eye(args.num_basis)).abs().max().item()
    print(f"Eigenvector orthonormality check: max off-diagonal = {ortho_err:.6e}")

    chk = {
        "GBH_eigval": eigval,
        "GBH_eigvec": eigvec,
    }
    with open(args.output, "wb") as f:
        pickle.dump(chk, f)

    print(f"Saved eigenvector checkpoint to: {args.output}")
    print(f"  eigenvalue range: [{eigval.min():.6f}, {eigval.max():.6f}]")
    print(f"  eigenvector shape: {eigvec.shape}")
    print(f"\nUse this file with:")
    print(f"  ghd_fitting.py --canonical_eigen_chk {args.output}")
    print(f"  (VAE scripts already default to this path)")


if __name__ == "__main__":
    main()
