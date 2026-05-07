#!/usr/bin/env python3
"""Render VAE target/reconstruction/sample meshes with one OPA ostium overlay."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
import numpy as np
from pytorch3d.io import load_obj

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--opa-path", type=Path, required=True)
    parser.add_argument("--target-obj", type=Path, required=True)
    parser.add_argument("--recon-obj", type=Path, required=True)
    parser.add_argument("--sample-obj", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_obj_np(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts, faces, _ = load_obj(str(path))
    return verts.detach().cpu().numpy(), faces.verts_idx.detach().cpu().numpy()


def set_axes(ax, points: np.ndarray, zoom_ring: np.ndarray | None = None) -> None:
    if zoom_ring is None:
        min_xyz = points.min(axis=0)
        max_xyz = points.max(axis=0)
        center = (min_xyz + max_xyz) * 0.5
        radius = max(float(np.max(max_xyz - min_xyz)) * 0.56, 1e-3)
        elev, azim = 16, -72
    else:
        min_xyz = zoom_ring.min(axis=0)
        max_xyz = zoom_ring.max(axis=0)
        center = (min_xyz + max_xyz) * 0.5
        radius = max(float(np.max(max_xyz - min_xyz)) * 0.95, 0.08)
        elev, azim = 70, -88

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def draw_mesh(ax, verts: np.ndarray, faces: np.ndarray, alpha: float) -> None:
    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=faces,
        color="#8eb3d3",
        edgecolor=(0.05, 0.09, 0.13, 0.08),
        linewidth=0.04,
        alpha=alpha,
        shade=True,
        antialiased=True,
    )


def draw_opa(ax, ring: np.ndarray, rec_v: np.ndarray, rec_f: np.ndarray) -> None:
    if rec_f.size:
        ax.plot_trisurf(
            rec_v[:, 0],
            rec_v[:, 1],
            rec_v[:, 2],
            triangles=rec_f,
            color="#f2a541",
            edgecolor="#9b5a00",
            linewidth=0.08,
            alpha=0.38,
            shade=True,
        )
    loop = np.vstack([ring, ring[:1]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="#c4263a", linewidth=2.6, alpha=0.98)
    ax.scatter(ring[:, 0], ring[:, 1], ring[:, 2], color="#c4263a", s=18, depthshade=False)


def nearest_ring_distance(verts: np.ndarray, ring: np.ndarray) -> tuple[float, float]:
    diffs = ring[:, None, :] - verts[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1).min(axis=1)
    return float(dist.mean()), float(dist.max())


def render(args: argparse.Namespace) -> None:
    opa = load_pickle(args.opa_path)
    ring = np.asarray(opa["op_v_coords"][0], dtype=np.float64)
    rec_v = np.asarray(opa["op_rec_v"][0], dtype=np.float64)
    rec_f = np.asarray(opa["op_rec_f"][0], dtype=np.int64)

    items = [
        ("Input target", *load_obj_np(args.target_obj)),
        ("Reconstruction", *load_obj_np(args.recon_obj)),
        ("Latent sample", *load_obj_np(args.sample_obj)),
    ]
    all_points = np.concatenate([*(verts for _, verts, _ in items), ring, rec_v], axis=0)

    fig = plt.figure(figsize=(16.0, 8.2), constrained_layout=True)
    summary = {
        "case": args.case,
        "opa_path": str(args.opa_path),
        "opa_source": str(opa.get("source", "")),
        "ring_points": int(ring.shape[0]),
        "panels": [],
    }

    for col, (title, verts, faces) in enumerate(items, start=1):
        mean_dist, max_dist = nearest_ring_distance(verts, ring)
        summary["panels"].append(
            {
                "title": title,
                "vertices": int(verts.shape[0]),
                "faces": int(faces.shape[0]),
                "ring_to_mesh_mean_dist": mean_dist,
                "ring_to_mesh_max_dist": max_dist,
            }
        )

        ax_full = fig.add_subplot(2, 3, col, projection="3d")
        draw_mesh(ax_full, verts, faces, alpha=0.30)
        draw_opa(ax_full, ring, rec_v, rec_f)
        ax_full.set_title(f"{title}\nfull mesh", fontsize=10, pad=4)
        set_axes(ax_full, all_points)

        ax_zoom = fig.add_subplot(2, 3, col + 3, projection="3d")
        draw_mesh(ax_zoom, verts, faces, alpha=0.20)
        draw_opa(ax_zoom, ring, rec_v, rec_f)
        ax_zoom.set_title(
            f"ostium zoom | mean ring-mesh dist {mean_dist:.4f}",
            fontsize=10,
            pad=4,
        )
        set_axes(ax_zoom, verts, zoom_ring=ring)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"{args.case}: VAE reconstruction and latent sample with OPA ostium", fontsize=15)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)

    summary["output"] = str(args.output)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    render(parse_args())


if __name__ == "__main__":
    main()
