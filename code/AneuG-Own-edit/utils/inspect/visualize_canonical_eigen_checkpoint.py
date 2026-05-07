#!/usr/bin/env python3
"""Visualize canonical GHD eigen checkpoint (GBH_eigval / GBH_eigvec)."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize canonical eigen checkpoint.")
    parser.add_argument("--mesh-path", type=Path, required=True, help="Canonical OBJ mesh path.")
    parser.add_argument("--eigen-chk", type=Path, required=True, help="Checkpoint with GBH_eigval/GBH_eigvec.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for plots.")
    parser.add_argument("--num-modes", type=int, default=12, help="How many eigenmodes to plot on mesh.")
    return parser.parse_args()


def _set_equal_axes(ax, verts: np.ndarray) -> None:
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def load_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.eigen_chk, "rb") as f:
        chk = pickle.load(f)
    eigval = chk["GBH_eigval"]
    eigvec = chk["GBH_eigvec"]

    if torch.is_tensor(eigval):
        eigval = eigval.detach().cpu().numpy()
    if torch.is_tensor(eigvec):
        eigvec = eigvec.detach().cpu().numpy()

    eigval = np.asarray(eigval).reshape(-1)
    eigvec = np.asarray(eigvec)

    verts, faces = load_mesh(args.mesh_path)
    if eigvec.shape[0] != verts.shape[0]:
        raise ValueError(f"Mismatch: eigvec rows={eigvec.shape[0]} vs mesh verts={verts.shape[0]}")

    # 1) Eigenvalue spectrum
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1, len(eigval) + 1), eigval, marker="o", linewidth=1)
    ax.set_title("GBH Eigenvalue Spectrum")
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Eigenvalue")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    spectrum_path = args.out_dir / "eigenvalue_spectrum.png"
    fig.savefig(spectrum_path, dpi=180)
    plt.close(fig)

    # 2) First K eigenvectors on mesh
    k = min(args.num_modes, eigvec.shape[1])
    cols = 4
    rows = int(np.ceil(k / cols))
    fig = plt.figure(figsize=(4 * cols, 3.6 * rows))
    for i in range(k):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        mode = eigvec[:, i]
        vmax = np.percentile(np.abs(mode), 99)
        vmax = max(vmax, 1e-8)
        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            triangles=faces,
            cmap="coolwarm",
            linewidth=0.0,
            antialiased=False,
            shade=False,
            vmin=-vmax,
            vmax=vmax,
            alpha=1.0,
            edgecolor="none",
            facecolors=plt.cm.coolwarm(np.clip((mode / vmax + 1) * 0.5, 0, 1)),
        )
        ax.set_title(f"Mode {i+1}")
        ax.set_axis_off()
        _set_equal_axes(ax, verts)
    fig.suptitle("First GBH Eigenvectors on Canonical Mesh", y=0.99)
    fig.tight_layout()
    modes_path = args.out_dir / "eigenvectors_first_modes.png"
    fig.savefig(modes_path, dpi=180)
    plt.close(fig)

    # 3) Basic stats
    txt = (
        f"mesh_vertices: {verts.shape[0]}\n"
        f"mesh_faces: {faces.shape[0]}\n"
        f"eigvec_shape: {eigvec.shape}\n"
        f"eigval_shape: {eigval.shape}\n"
        f"eigval_min: {eigval.min():.6e}\n"
        f"eigval_max: {eigval.max():.6e}\n"
        f"eigval_non_decreasing: {bool(np.all(np.diff(eigval) >= -1e-8))}\n"
        f"finite_eigvec: {bool(np.isfinite(eigvec).all())}\n"
        f"finite_eigval: {bool(np.isfinite(eigval).all())}\n"
    )
    stats_path = args.out_dir / "eigen_stats.txt"
    stats_path.write_text(txt, encoding="utf-8")

    print(f"Saved: {spectrum_path}")
    print(f"Saved: {modes_path}")
    print(f"Saved: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
