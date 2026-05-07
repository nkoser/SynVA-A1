#!/usr/bin/env python3
"""Create canonical GHD eigen checkpoint (e.g. canonical_typeB_144.pkl)."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

import torch
from pytorch3d.io import load_objs_as_meshes

# Ensure project root is import priority, otherwise running this script from
# `utils/` can shadow the `utils` package with `utils/utils.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from ghd.base.graph_harmonic_deformation import Graph_Harmonic_Deform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and save GBH_eigval/GBH_eigvec checkpoint for a canonical mesh."
    )
    parser.add_argument(
        "--mesh-path",
        type=Path,
        required=True,
        help="Path to canonical OBJ mesh.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        required=True,
        help="Output pickle path (e.g. canonical_typeB_144.pkl).",
    )
    parser.add_argument(
        "--num-basis",
        type=int,
        default=12 ** 2,
        help="Number of harmonic basis vectors (default: 144).",
    )
    parser.add_argument(
        "--mix-lap-weights",
        type=float,
        nargs=3,
        default=[1.0, 0.1, 0.1],
        metavar=("COT", "NOR", "STD"),
        help="Laplacian mixing weights (default: 1.0 0.1 0.1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for computation (default: cpu).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mesh_path = args.mesh_path.resolve()
    out_path = args.out_path.resolve()

    if not mesh_path.exists():
        raise SystemExit(f"Mesh not found: {mesh_path}")
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists, use --overwrite: {out_path}")

    device = torch.device(args.device)
    mesh = load_objs_as_meshes([str(mesh_path)], device=device)
    ghd = Graph_Harmonic_Deform(
        base_shape=mesh,
        num_Basis=args.num_basis,
        mix_lap_weight=list(args.mix_lap_weights),
        eigen_chk=None,
    )
    chk = {
        "GBH_eigval": ghd.GBH_eigval.detach().cpu(),
        "GBH_eigvec": ghd.GBH_eigvec.detach().cpu(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(chk, f)

    print(f"Saved: {out_path}")
    print(f"GBH_eigval shape: {tuple(chk['GBH_eigval'].shape)}")
    print(f"GBH_eigvec shape: {tuple(chk['GBH_eigvec'].shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
