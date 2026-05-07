#!/usr/bin/env python3
"""Render random rebuilt 20-point OPA checkpoints on their GHD meshes."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch
from pytorch3d.io import load_obj

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils.inspect.vae_inspect_stage1_ostium_conditional import (  # noqa: E402
    build_ghd_reconstruct,
    mesh_to_numpy,
    reconstruct_fitted_mesh_from_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ghd-root", type=Path, default=Path("checkpoint-v2/ghd_fitting_split_real"))
    parser.add_argument("--canonical-root", type=Path, default=Path("alignment_vc/canonical_model"))
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional explicit cases to render. If omitted, sample --count random cases.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vae_optimization_results/vae_opt_20260428_full/opa_boundary20_random_grid"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_obj_np(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts, faces, _ = load_obj(str(path))
    return verts.detach().cpu().numpy(), faces.verts_idx.detach().cpu().numpy()


def find_cases(ghd_root: Path) -> list[str]:
    return sorted(path.parent.name for path in ghd_root.glob("*/opa_checkpoint.pkl"))


def get_case_mesh(
    case: str,
    ghd_root: Path,
    device: torch.device,
    ghd_reconstruct,
) -> tuple[np.ndarray, np.ndarray, str]:
    warped = ghd_root / case / "vanilla" / "viz" / "warped_epoch_02999.obj"
    if warped.exists():
        verts, faces = load_obj_np(warped)
        return verts, faces, "warped_epoch_02999.obj"

    chk_path = ghd_root / case / "vanilla" / "ghb_fitting_checkpoint.pkl"
    chk = load_pickle(chk_path)
    mesh = reconstruct_fitted_mesh_from_checkpoint(ghd_reconstruct, chk, device)
    verts, faces = mesh_to_numpy(mesh)
    return verts, faces, "ghb_fitting_checkpoint.pkl replay"


def set_equal_axes(ax, vertices: np.ndarray) -> None:
    min_xyz = vertices.min(axis=0)
    max_xyz = vertices.max(axis=0)
    center = (min_xyz + max_xyz) * 0.5
    radius = max(float(np.max(max_xyz - min_xyz)) * 0.56, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_axis_off()
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.view_init(elev=16, azim=-72)


def draw_ring(ax, points: np.ndarray) -> None:
    loop = np.vstack([points, points[:1]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color="#c4263a", linewidth=2.3, alpha=0.95)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="#c4263a", s=16, depthshade=False)


def render_grid(args: argparse.Namespace, selected: list[str]) -> dict:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ghd_reconstruct = build_ghd_reconstruct(args.canonical_root, device)
    fig = plt.figure(figsize=(4.8 * len(selected), 5.4), constrained_layout=True)
    records = []
    for i, case in enumerate(selected, start=1):
        opa_path = args.ghd_root / case / "opa_checkpoint.pkl"
        opa = load_pickle(opa_path)
        verts, faces, mesh_source = get_case_mesh(case, args.ghd_root, device, ghd_reconstruct)
        ring = np.asarray(opa["op_v_coords"][0], dtype=np.float64)
        rec_v = np.asarray(opa["op_rec_v"][0], dtype=np.float64)
        rec_f = np.asarray(opa["op_rec_f"][0], dtype=np.int64)

        ax = fig.add_subplot(1, len(selected), i, projection="3d")
        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            triangles=faces,
            color="#8eb3d3",
            edgecolor=(0.06, 0.12, 0.18, 0.08),
            linewidth=0.05,
            alpha=0.28,
            shade=True,
            antialiased=True,
        )
        if rec_f.size:
            ax.plot_trisurf(
                rec_v[:, 0],
                rec_v[:, 1],
                rec_v[:, 2],
                triangles=rec_f,
                color="#f2a541",
                edgecolor="#b26b00",
                linewidth=0.08,
                alpha=0.36,
            )
        draw_ring(ax, ring)
        ax.set_title(f"{case}\n{mesh_source}", fontsize=8)
        set_equal_axes(ax, np.concatenate([verts, ring, rec_v], axis=0))
        records.append(
            {
                "case": case,
                "source": str(opa.get("source", "")),
                "mesh_source": mesh_source,
                "opa_points": int(ring.shape[0]),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / f"random_{len(selected)}_seed_{args.seed}.png"
    fig.suptitle("Random rebuilt 20-point GHD OPA checkpoints", fontsize=14)
    fig.savefig(png_path, dpi=args.dpi)
    plt.close(fig)
    summary = {
        "seed": int(args.seed),
        "count": int(len(selected)),
        "png": str(png_path),
        "cases": records,
    }
    summary_path = args.output_dir / f"random_{len(selected)}_seed_{args.seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    cases = find_cases(args.ghd_root)
    if args.cases:
        known = set(cases)
        missing = [case for case in args.cases if case not in known]
        if missing:
            raise ValueError(f"Cases without opa_checkpoint.pkl: {missing}")
        selected = args.cases
    else:
        rng = random.Random(args.seed)
        selected = rng.sample(cases, min(args.count, len(cases)))
    render_grid(args, selected)


if __name__ == "__main__":
    main()
