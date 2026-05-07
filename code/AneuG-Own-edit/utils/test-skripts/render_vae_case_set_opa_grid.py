#!/usr/bin/env python3
"""Render target/reconstruction/sample grids for several VAE cases with OPA overlays."""

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
    parser.add_argument("--split-name", required=True, help="Label shown in the figure title.")
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--ghd-root", type=Path, default=Path("checkpoint-v2/ghd_fitting_split_real"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_obj_np(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts, faces, _ = load_obj(str(path))
    return verts.detach().cpu().numpy(), faces.verts_idx.detach().cpu().numpy()


def draw_mesh(ax, verts: np.ndarray, faces: np.ndarray, alpha: float) -> None:
    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=faces,
        color="#8eb3d3",
        edgecolor=(0.05, 0.09, 0.13, 0.08),
        linewidth=0.035,
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
            linewidth=0.06,
            alpha=0.35,
            shade=True,
        )
    loop = np.vstack([ring, ring[:1]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="#c4263a", linewidth=2.0, alpha=0.98)
    ax.scatter(ring[:, 0], ring[:, 1], ring[:, 2], color="#c4263a", s=9, depthshade=False)


def set_axes(ax, points: np.ndarray) -> None:
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    center = (min_xyz + max_xyz) * 0.5
    radius = max(float(np.max(max_xyz - min_xyz)) * 0.56, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.view_init(elev=16, azim=-72)
    ax.set_axis_off()


def nearest_ring_distance(verts: np.ndarray, ring: np.ndarray) -> tuple[float, float]:
    diffs = ring[:, None, :] - verts[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1).min(axis=1)
    return float(dist.mean()), float(dist.max())


def case_paths(base_dir: Path, case: str) -> dict[str, Path]:
    case_file = case.replace("/", "__")
    case_dir = base_dir / case
    return {
        "target": case_dir / "reconstruct" / f"{case_file}_target_raw.obj",
        "recon": case_dir / "reconstruct" / f"{case_file}_recon_raw.obj",
        "sample": case_dir / "sample" / f"{case_file}_sample_000_raw.obj",
        "recon_meta": case_dir / "reconstruct" / "metadata.json",
        "sample_meta": case_dir / "sample" / "metadata.json",
    }


def render(args: argparse.Namespace) -> None:
    fig = plt.figure(figsize=(12.6, 3.05 * len(args.cases)), constrained_layout=True)
    summary = {
        "split": args.split_name,
        "cases": [],
        "output": str(args.output),
    }
    columns = [("target", "Input target"), ("recon", "Reconstruction"), ("sample", "Latent sample")]

    for row, case in enumerate(args.cases):
        paths = case_paths(args.base_dir, case)
        missing = [name for name, path in paths.items() if name.endswith("meta") is False and not path.exists()]
        if missing:
            raise FileNotFoundError(f"{case} missing outputs: {missing}")

        opa = load_pickle(args.ghd_root / case / "opa_checkpoint.pkl")
        ring = np.asarray(opa["op_v_coords"][0], dtype=np.float64)
        rec_v = np.asarray(opa["op_rec_v"][0], dtype=np.float64)
        rec_f = np.asarray(opa["op_rec_f"][0], dtype=np.int64)
        meshes = {name: load_obj_np(paths[name]) for name, _ in columns}
        points = np.concatenate([*(verts for verts, _ in meshes.values()), ring, rec_v], axis=0)
        recon_meta = load_json(paths["recon_meta"]) if paths["recon_meta"].exists() else {}
        sample_meta = load_json(paths["sample_meta"]) if paths["sample_meta"].exists() else {}

        record = {
            "case": case,
            "opa_source": str(opa.get("source", "")),
            "ring_points": int(ring.shape[0]),
            "target_mse": recon_meta.get("target_mse"),
            "ghd_mse": recon_meta.get("ghd_mse"),
            "scale_mse": recon_meta.get("scale_mse"),
            "predicted_scale": recon_meta.get("predicted_scale"),
            "target_scale": recon_meta.get("target_scale"),
            "sample_predicted_scale_mean": sample_meta.get("predicted_scale_mean"),
            "panels": [],
        }

        for col, (mesh_key, title) in enumerate(columns):
            verts, faces = meshes[mesh_key]
            mean_dist, max_dist = nearest_ring_distance(verts, ring)
            record["panels"].append(
                {
                    "panel": mesh_key,
                    "ring_to_mesh_mean_dist": mean_dist,
                    "ring_to_mesh_max_dist": max_dist,
                }
            )
            ax = fig.add_subplot(len(args.cases), 3, row * 3 + col + 1, projection="3d")
            draw_mesh(ax, verts, faces, alpha=0.28)
            draw_opa(ax, ring, rec_v, rec_f)
            if col == 0:
                display_case = case if len(case) <= 30 else f"{case[:27]}..."
                title = f"{display_case}\n{title}"
            elif mesh_key == "recon":
                title = f"{title}\nMSE {recon_meta.get('target_mse', float('nan')):.3f} | d {mean_dist:.3f}"
            else:
                title = f"{title}\nd {mean_dist:.3f}"
            ax.set_title(title, fontsize=8.5, pad=3)
            set_axes(ax, points)

        summary["cases"].append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(
        f"{args.split_name}: {len(args.cases)} case(s) with OPA ostium | target, reconstruction, latent sample",
        fontsize=13,
    )
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    render(parse_args())


if __name__ == "__main__":
    main()
