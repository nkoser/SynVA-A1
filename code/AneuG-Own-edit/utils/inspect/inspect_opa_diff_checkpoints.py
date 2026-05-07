#!/usr/bin/env python3
"""Inspect and visualize OPA + differentiable centreline checkpoints for one case."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect one alignment case: part_aligned.obj + opa_checkpoint.pkl + diff_centreline_checkpoint.pkl"
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
        help="Case folder containing part_aligned.obj and checkpoint pkl files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <case-dir>/inspection).",
    )
    parser.add_argument(
        "--max-mesh-faces",
        type=int,
        default=20000,
        help="Maximum triangle count used for plotting.",
    )
    return parser.parse_args()


def _as_numpy(x: Any) -> np.ndarray:
    return np.asarray(x)


def _load_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def _set_equal_axes(ax: Any, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _plot_mesh(ax: Any, verts: np.ndarray, faces: np.ndarray, max_faces: int) -> None:
    faces_plot = faces
    if len(faces) > max_faces:
        face_idx = np.random.choice(len(faces), size=max_faces, replace=False)
        faces_plot = faces[face_idx]
    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=faces_plot,
        color="lightgray",
        alpha=0.18,
        linewidth=0.0,
    )


def summarize_opa(opa: dict[str, Any]) -> dict[str, Any]:
    op_v_coords = [_as_numpy(x) for x in opa.get("op_v_coords", [])]
    op_rec_v = [_as_numpy(x) for x in opa.get("op_rec_v", [])]
    op_rec_f = [_as_numpy(x) for x in opa.get("op_rec_f", [])]
    return {
        "num_openings": len(op_v_coords),
        "registered_opening_points": [int(x.shape[0]) for x in op_v_coords],
        "reconstructed_opening_vertices": [int(x.shape[0]) for x in op_rec_v],
        "reconstructed_opening_faces": [int(x.shape[0]) for x in op_rec_f],
        "keys": sorted(list(opa.keys())),
    }


def summarize_diff(diff: dict[str, Any]) -> dict[str, Any]:
    seeds = diff.get("diff_cep_registration", [])
    wave_loops = diff.get("wave_loops", [])
    waves_per_seed = [len(seed_loops) for seed_loops in wave_loops]
    loop_sizes = []
    for seed_loops in wave_loops:
        loop_sizes.extend([len(loop) for loop in seed_loops])
    return {
        "num_centreline_seeds": int(len(seeds)),
        "num_seed_waves": int(len(wave_loops)),
        "waves_per_seed": waves_per_seed,
        "total_loops": int(len(loop_sizes)),
        "loop_size_min": int(min(loop_sizes)) if loop_sizes else 0,
        "loop_size_max": int(max(loop_sizes)) if loop_sizes else 0,
        "keys": sorted(list(diff.keys())),
    }


def plot_opa(
    verts: np.ndarray,
    faces: np.ndarray,
    opa: dict[str, Any],
    out_path: Path,
    max_faces: int,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _plot_mesh(ax, verts, faces, max_faces=max_faces)

    op_v_coords = [_as_numpy(x) for x in opa.get("op_v_coords", [])]
    op_n_mean = [_as_numpy(x) for x in opa.get("op_n_mean", [])]
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown"]
    for i, coords in enumerate(op_v_coords):
        if coords.size == 0:
            continue
        c = colors[i % len(colors)]
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=12, color=c, label=f"opening {i}")
        if i < len(op_n_mean):
            center = coords.mean(axis=0)
            normal = op_n_mean[i].reshape(-1)
            ax.quiver(
                center[0],
                center[1],
                center[2],
                normal[0],
                normal[1],
                normal[2],
                length=0.15 * np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)),
                color=c,
                linewidth=2.0,
            )

    ax.set_title("OPA: opening points + mean normal")
    _set_equal_axes(ax, verts)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _extract_centreline_points(verts: np.ndarray, diff: dict[str, Any]) -> np.ndarray:
    wave_loops = diff.get("wave_loops", [])
    c_points = []
    for seed_loops in wave_loops:
        for loop in seed_loops:
            ids = np.asarray(loop, dtype=np.int64)
            ids = ids[(ids >= 0) & (ids < len(verts))]
            if len(ids) == 0:
                continue
            c_points.append(verts[ids].mean(axis=0))
    if not c_points:
        return np.empty((0, 3))
    return np.stack(c_points, axis=0)


def plot_diff_centreline(
    verts: np.ndarray,
    faces: np.ndarray,
    diff: dict[str, Any],
    out_path: Path,
    max_faces: int,
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    _plot_mesh(ax, verts, faces, max_faces=max_faces)

    seeds = np.asarray(diff.get("diff_cep_registration", []), dtype=np.int64)
    seeds = seeds[(seeds >= 0) & (seeds < len(verts))]
    if len(seeds) > 0:
        seed_pts = verts[seeds]
        ax.scatter(seed_pts[:, 0], seed_pts[:, 1], seed_pts[:, 2], s=70, color="gold", edgecolors="k", label="seed vertices")

    cl_points = _extract_centreline_points(verts, diff)
    if len(cl_points) > 0:
        ax.scatter(
            cl_points[:, 0],
            cl_points[:, 1],
            cl_points[:, 2],
            s=8,
            color="tab:cyan",
            alpha=0.8,
            label="wave-loop centers",
        )

    ax.set_title("Differentiable centreline: seeds + wave-loop centers")
    _set_equal_axes(ax, verts)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def inspect_case(case_dir: Path, out_dir: Path, max_mesh_faces: int = 20000) -> dict[str, Any]:
    case_dir = case_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = case_dir / "part_aligned.obj"
    opa_path = case_dir / "opa_checkpoint.pkl"
    diff_path = case_dir / "diff_centreline_checkpoint.pkl"

    missing_required = [p.name for p in [mesh_path, opa_path] if not p.exists()]
    if missing_required:
        raise FileNotFoundError(f"Missing required files in {case_dir}: {missing_required}")

    verts, faces = _load_mesh(mesh_path)
    with open(opa_path, "rb") as f:
        opa = pickle.load(f)

    diff = None
    if diff_path.exists():
        with open(diff_path, "rb") as f:
            diff = pickle.load(f)

    summary = {
        "case_dir": str(case_dir),
        "mesh": {"vertices": int(verts.shape[0]), "faces": int(faces.shape[0])},
        "opa_checkpoint": summarize_opa(opa),
        "diff_centreline_checkpoint": summarize_diff(diff) if diff is not None else {"present": False},
        "missing_optional_files": [p.name for p in [diff_path] if not p.exists()],
    }

    plot_opa(verts, faces, opa, out_dir / "01_opa_openings.png", max_faces=max_mesh_faces)
    if diff is not None:
        plot_diff_centreline(verts, faces, diff, out_dir / "02_diff_centreline.png", max_faces=max_mesh_faces)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir if args.out_dir is not None else args.case_dir / "inspection"
    summary = inspect_case(case_dir=args.case_dir, out_dir=out_dir, max_mesh_faces=args.max_mesh_faces)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved outputs to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
