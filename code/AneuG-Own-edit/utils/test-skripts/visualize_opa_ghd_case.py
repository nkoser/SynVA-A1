#!/usr/bin/env python3
"""Visualize one GHD fitting checkpoint together with its OPA checkpoint."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
import numpy as np
import torch
from pytorch3d.io import load_obj, save_obj
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils.inspect.ghd_checkpoint_replay_current_fitter import (  # noqa: E402
    build_current_fitter_replay,
)
from utils.inspect.vae_inspect_stage1_ostium_conditional import (  # noqa: E402
    order_ring_indices_by_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="C0066")
    parser.add_argument("--ghd-root", type=Path, default=Path("checkpoint-v2/ghd_fitting_split_real"))
    parser.add_argument("--alignment-root", type=Path, default=Path("alignment_vc"))
    parser.add_argument(
        "--canonical-eigen-chk",
        type=Path,
        default=None,
        help="Optional override. By default the script reads vanilla/run_config.json, then ghd-root/canonical_model_144_normed.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vae_optimization_results/vae_opt_20260428_full/opa_ghd_debug"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_obj_np(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts, faces, _ = load_obj(str(path))
    return verts.detach().cpu().numpy(), faces.verts_idx.detach().cpu().numpy()


def mesh_to_numpy(mesh) -> tuple[np.ndarray, np.ndarray]:
    return (
        mesh.verts_packed().detach().cpu().numpy(),
        mesh.faces_packed().detach().cpu().numpy(),
    )


def resolve_canonical_eigen_chk(args: argparse.Namespace, run_root: Path) -> Path:
    if args.canonical_eigen_chk is not None:
        path = args.canonical_eigen_chk
    else:
        path = None
        run_config = run_root / "run_config.json"
        if run_config.exists():
            with run_config.open("r", encoding="utf-8") as handle:
                cfg = json.load(handle)
            cfg_args = cfg.get("args", {})
            if cfg_args.get("canonical_eigen_chk"):
                path = Path(cfg_args["canonical_eigen_chk"])
        if path is None:
            path = args.ghd_root / "canonical_model_144_normed.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Canonical eigen checkpoint not found: {path}")
    return path


def ring_span(points: np.ndarray) -> float:
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def nearest_stats(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    dist = cKDTree(np.asarray(target, dtype=np.float64)).query(np.asarray(source, dtype=np.float64), k=1)[0]
    return {
        "mean": float(dist.mean()),
        "max": float(dist.max()),
        "median": float(np.median(dist)),
    }


def set_equal_axes(ax, vertices: np.ndarray) -> None:
    min_xyz = vertices.min(axis=0)
    max_xyz = vertices.max(axis=0)
    center = (min_xyz + max_xyz) * 0.5
    radius = max(float(np.max(max_xyz - min_xyz)) * 0.55, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_axis_off()
    ax.view_init(elev=16, azim=-72)


def draw_mesh(ax, verts: np.ndarray, faces: np.ndarray, alpha: float = 0.28) -> None:
    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=faces,
        color="#8eb3d3",
        edgecolor=(0.06, 0.12, 0.18, 0.08),
        linewidth=0.06,
        alpha=alpha,
        shade=True,
        antialiased=True,
    )


def draw_ring(ax, points: np.ndarray, color: str, label: str, linewidth: float = 2.4) -> None:
    if points.shape[0] < 3:
        return
    loop = np.vstack([points, points[:1]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=color, linewidth=linewidth, alpha=0.9, label=label)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=12, depthshade=False)


def render_debug(
    output_path: Path,
    case: str,
    ghd_vertices: np.ndarray,
    ghd_faces: np.ndarray,
    warped_vertices: np.ndarray,
    warped_faces: np.ndarray,
    opa_coords: np.ndarray,
    raw_index_points: np.ndarray,
    ordered_index_points: np.ndarray,
    rec_vertices: np.ndarray,
    rec_faces: np.ndarray,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    fig.suptitle(f"C0066 OPA checkpoint vs GHD fitting checkpoint" if case == "C0066" else f"{case} OPA vs GHD", fontsize=16)

    panels = [
        ("GHD reconstructed from ghb_fitting_checkpoint.pkl\nraw OPA op_v_indices order", ghd_vertices, ghd_faces, raw_index_points, "#c4263a"),
        ("GHD reconstructed from ghb_fitting_checkpoint.pkl\nsame OPA indices ordered for readability", ghd_vertices, ghd_faces, ordered_index_points, "#c4263a"),
        ("warped_epoch_02999.obj\nOPA op_v_coords + op_rec_f surface", warped_vertices, warped_faces, opa_coords, "#c4263a"),
    ]
    all_vertices = np.concatenate([ghd_vertices, warped_vertices, opa_coords, raw_index_points, ordered_index_points], axis=0)

    for i, (title, verts, faces, ring, color) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        draw_mesh(ax, verts, faces)
        if i == 3 and rec_faces.size > 0:
            ax.plot_trisurf(
                rec_vertices[:, 0],
                rec_vertices[:, 1],
                rec_vertices[:, 2],
                triangles=rec_faces,
                color="#f2a541",
                alpha=0.34,
                linewidth=0.1,
                edgecolor="#b26b00",
            )
        draw_ring(ax, ring, color, "OPA")
        ax.set_title(title, fontsize=10)
        set_equal_axes(ax, all_vertices)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    case_root = args.ghd_root / args.case
    run_root = case_root / "vanilla"
    ghd_checkpoint = run_root / "ghb_fitting_checkpoint.pkl"
    opa_checkpoint = case_root / "opa_checkpoint.pkl"
    warped_obj = run_root / "viz" / "warped_epoch_02999.obj"
    for path in (ghd_checkpoint, opa_checkpoint, warped_obj):
        if not path.exists():
            raise FileNotFoundError(path)

    canonical_eigen_chk = resolve_canonical_eigen_chk(args, run_root)
    ghd_chk = load_pickle(ghd_checkpoint)
    opa_chk = load_pickle(opa_checkpoint)

    mesh, norm_canonical = build_current_fitter_replay(
        alignment_root=args.alignment_root,
        case_name=args.case,
        checkpoint=ghd_chk,
        device=device,
        canonical_eigen_chk=canonical_eigen_chk,
    )
    ghd_vertices, ghd_faces = mesh_to_numpy(mesh)
    warped_vertices, warped_faces = load_obj_np(warped_obj)

    opa_coords = np.asarray(opa_chk["op_v_coords"][0], dtype=np.float64)
    opa_indices = np.asarray(opa_chk["op_v_indices"][0], dtype=np.int64)
    if np.any(opa_indices < 0) or np.any(opa_indices >= ghd_vertices.shape[0]):
        raise ValueError(
            f"OPA indices out of range: min={opa_indices.min()}, max={opa_indices.max()}, verts={ghd_vertices.shape[0]}"
        )
    raw_index_points = ghd_vertices[opa_indices]
    ordered_indices = order_ring_indices_by_points(
        opa_indices.astype(np.int64),
        torch.from_numpy(raw_index_points.astype(np.float32)).to(device),
    )
    ordered_index_points = ghd_vertices[ordered_indices]
    rec_vertices = np.asarray(opa_chk.get("op_rec_v", [opa_coords])[0], dtype=np.float64)
    rec_faces = np.asarray(opa_chk.get("op_rec_f", [np.empty((0, 3), dtype=np.int64)])[0], dtype=np.int64)

    output_dir = args.output_dir / args.case
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{args.case}_opa_vs_ghd_checkpoint.png"
    render_debug(
        output_path=png_path,
        case=args.case,
        ghd_vertices=ghd_vertices,
        ghd_faces=ghd_faces,
        warped_vertices=warped_vertices,
        warped_faces=warped_faces,
        opa_coords=opa_coords,
        raw_index_points=raw_index_points,
        ordered_index_points=ordered_index_points,
        rec_vertices=rec_vertices,
        rec_faces=rec_faces,
        dpi=args.dpi,
    )

    save_obj(output_dir / f"{args.case}_ghd_reconstructed_from_checkpoint.obj", torch.from_numpy(ghd_vertices).float(), torch.from_numpy(ghd_faces).long())
    if rec_faces.size > 0:
        save_obj(output_dir / f"{args.case}_opa_rec_surface.obj", torch.from_numpy(rec_vertices).float(), torch.from_numpy(rec_faces).long())

    direct = np.linalg.norm(raw_index_points - opa_coords, axis=1)
    warped_direct = np.linalg.norm(warped_vertices[opa_indices] - opa_coords, axis=1)
    summary = {
        "case": args.case,
        "ghd_checkpoint": str(ghd_checkpoint),
        "opa_checkpoint": str(opa_checkpoint),
        "warped_obj": str(warped_obj),
        "alignment_root": str(args.alignment_root),
        "canonical_eigen_chk": str(canonical_eigen_chk),
        "norm_canonical_used_by_current_fitter": norm_canonical,
        "opa_source": str(opa_chk.get("source", "unknown")),
        "ghd_vertices": int(ghd_vertices.shape[0]),
        "ghd_faces": int(ghd_faces.shape[0]),
        "opa_points": int(opa_coords.shape[0]),
        "opa_index_min": int(opa_indices.min()),
        "opa_index_max": int(opa_indices.max()),
        "opa_span": ring_span(opa_coords),
        "opa_bbox_min": opa_coords.min(axis=0).round(8).tolist(),
        "opa_bbox_max": opa_coords.max(axis=0).round(8).tolist(),
        "ghd_reconstruct_index_to_opa_mean": float(direct.mean()),
        "ghd_reconstruct_index_to_opa_max": float(direct.max()),
        "warped_obj_index_to_opa_mean": float(warped_direct.mean()),
        "warped_obj_index_to_opa_max": float(warped_direct.max()),
        "ghd_reconstruct_vs_warped_rmse": float(np.sqrt(np.mean((ghd_vertices - warped_vertices) ** 2))),
        "opa_to_ghd_nearest": nearest_stats(opa_coords, ghd_vertices),
        "opa_to_warped_nearest": nearest_stats(opa_coords, warped_vertices),
        "outputs": {
            "png": str(png_path),
            "ghd_reconstructed_obj": str(output_dir / f"{args.case}_ghd_reconstructed_from_checkpoint.obj"),
            "opa_rec_surface_obj": str(output_dir / f"{args.case}_opa_rec_surface.obj") if rec_faces.size > 0 else "",
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {png_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
