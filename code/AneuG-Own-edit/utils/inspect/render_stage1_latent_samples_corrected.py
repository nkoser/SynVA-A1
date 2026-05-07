#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pytorch3d.structures import Meshes
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.inspect.ghd_checkpoint_replay_current_fitter import build_current_fitter_replay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render train/test latent samples with corrected checkpoint-based visualization."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ghd-chk-root", type=Path, required=True)
    parser.add_argument("--condition-root", type=Path, default=None)
    parser.add_argument("--alignment-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--canonical-eigen-chk", type=Path, required=True)
    parser.add_argument("--train-cases", type=str, nargs="+", required=True)
    parser.add_argument("--test-cases", type=str, nargs="+", required=True)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--taubin-iter", type=int, default=0)
    return parser.parse_args()


def load_obj_mesh(path: Path) -> trimesh.Trimesh:
    return trimesh.load_mesh(path, process=False)


def mesh_to_arrays(mesh: trimesh.Trimesh | Meshes):
    if isinstance(mesh, Meshes):
        verts = mesh.verts_packed().detach().cpu().numpy()
        faces = mesh.faces_packed().detach().cpu().numpy()
        return verts, faces
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def draw_mesh(
    ax,
    mesh: trimesh.Trimesh | Meshes,
    color: str = "#8eb3d3",
    alpha: float = 0.34,
    share_bounds: tuple[np.ndarray, float] | None = None,
) -> None:
    verts, faces = mesh_to_arrays(mesh)
    tris = verts[faces]
    coll = Poly3DCollection(
        tris,
        facecolor=color,
        edgecolor=(0.06, 0.12, 0.18, 0.08),
        linewidths=0.08,
        alpha=alpha,
    )
    ax.add_collection3d(coll)
    if share_bounds is None:
        min_xyz = verts.min(axis=0)
        max_xyz = verts.max(axis=0)
        center = (min_xyz + max_xyz) * 0.5
        radius = max(float((max_xyz - min_xyz).max()) * 0.5, 1e-3)
    else:
        center, radius = share_bounds
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=18, azim=35)
    ax.set_axis_off()


def run_infer(
    python_exe: str,
    infer_script: Path,
    checkpoint: Path,
    ghd_chk_root: Path,
    condition_root: Path,
    alignment_root: Path,
    canonical_root: Path,
    canonical_eigen_chk: Path,
    case: str,
    mode: str,
    output_dir: Path,
    num_samples: int,
    seed: int,
    taubin_iter: int,
) -> None:
    cmd = [
        python_exe,
        str(infer_script),
        "--checkpoint", str(checkpoint),
        "--ghd-chk-root", str(ghd_chk_root),
        "--condition-root", str(condition_root),
        "--alignment-root", str(alignment_root),
        "--canonical-root", str(canonical_root),
        "--canonical-eigen-chk", str(canonical_eigen_chk),
        "--case", case,
        "--mode", mode,
        "--taubin-iter", str(taubin_iter),
        "--output-dir", str(output_dir),
    ]
    if mode == "reconstruct":
        cmd.extend(["--posterior-noise-scale", "0"])
    else:
        cmd.extend(["--num-samples", str(num_samples), "--seed", str(seed)])
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def render_triptych(
    case: str,
    input_mesh: Meshes,
    recon_mesh: trimesh.Trimesh,
    sample_mesh: trimesh.Trimesh,
    output_path: Path,
    dpi: int,
) -> None:
    input_verts, _ = mesh_to_arrays(input_mesh)
    recon_verts, _ = mesh_to_arrays(recon_mesh)
    sample_verts, _ = mesh_to_arrays(sample_mesh)
    all_verts = np.concatenate([input_verts, recon_verts, sample_verts], axis=0)
    min_xyz = all_verts.min(axis=0)
    max_xyz = all_verts.max(axis=0)
    center = (min_xyz + max_xyz) * 0.5
    radius = max(float((max_xyz - min_xyz).max()) * 0.5, 1e-3)
    shared = (center, radius)

    fig = plt.figure(figsize=(12.0, 4.2), constrained_layout=True)
    fig.suptitle(case, fontsize=14)
    entries = [
        ("Input from ghb_fitting_checkpoint.pkl", input_mesh),
        ("VAE reconstruct", recon_mesh),
        ("Prior sample", sample_mesh),
    ]
    for idx, (title, mesh) in enumerate(entries, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        draw_mesh(ax, mesh, share_bounds=shared)
        ax.set_title(title, fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_contact_sheet(
    case: str,
    sample_meshes: list[trimesh.Trimesh],
    output_path: Path,
    dpi: int,
) -> None:
    ncols = 4
    nrows = int(np.ceil(len(sample_meshes) / ncols))
    fig = plt.figure(figsize=(3.3 * ncols, 3.0 * nrows), constrained_layout=True)
    fig.suptitle(case, fontsize=14)
    for idx, mesh in enumerate(sample_meshes, start=1):
        ax = fig.add_subplot(nrows, ncols, idx, projection="3d")
        draw_mesh(ax, mesh)
        ax.set_title(f"Sample {idx - 1}", fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    condition_root = args.condition_root if args.condition_root is not None else args.ghd_chk_root
    out_root = args.output_dir.resolve()
    infer_script = ROOT / "infer_stage1_ostium_conditional.py"
    python_exe = sys.executable
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    compare_dir = out_root / "checkpoint_recon_sample_correct"
    samples_dir = out_root / "latent_samples_only_correct"
    compare_summary = {"train": [], "test": []}
    samples_summary = {"train": [], "test": []}

    grouped_cases = [("train", args.train_cases), ("test", args.test_cases)]
    for split_name, cases in grouped_cases:
        for case in cases:
            recon_dir = out_root / "_tmp_infer" / split_name / case / "reconstruct"
            sample_dir = out_root / "_tmp_infer" / split_name / case / "sample"
            run_infer(
                python_exe=python_exe,
                infer_script=infer_script,
                checkpoint=args.checkpoint,
                ghd_chk_root=args.ghd_chk_root,
                condition_root=condition_root,
                alignment_root=args.alignment_root,
                canonical_root=args.canonical_root,
                canonical_eigen_chk=args.canonical_eigen_chk,
                case=case,
                mode="reconstruct",
                output_dir=recon_dir,
                num_samples=args.num_samples,
                seed=args.seed,
                taubin_iter=args.taubin_iter,
            )
            run_infer(
                python_exe=python_exe,
                infer_script=infer_script,
                checkpoint=args.checkpoint,
                ghd_chk_root=args.ghd_chk_root,
                condition_root=condition_root,
                alignment_root=args.alignment_root,
                canonical_root=args.canonical_root,
                canonical_eigen_chk=args.canonical_eigen_chk,
                case=case,
                mode="sample",
                output_dir=sample_dir,
                num_samples=args.num_samples,
                seed=args.seed,
                taubin_iter=args.taubin_iter,
            )

            input_mesh, _ = build_current_fitter_replay(
                alignment_root=args.alignment_root,
                case_name=case,
                checkpoint=pickle_load(args.ghd_chk_root / case / "vanilla" / "ghb_fitting_checkpoint.pkl"),
                device=device,
                canonical_eigen_chk=args.canonical_eigen_chk,
            )
            recon_mesh = load_obj_mesh(recon_dir / f"{case}_recon_raw.obj")
            sample_mesh = load_obj_mesh(sample_dir / f"{case}_sample_000_raw.obj")
            triptych_path = compare_dir / split_name / f"{case}_checkpoint_recon_sample_correct.png"
            render_triptych(case, input_mesh, recon_mesh, sample_mesh, triptych_path, args.dpi)
            compare_summary[split_name].append({"case": case, "image": str(triptych_path)})

            sample_meshes = [
                load_obj_mesh(sample_dir / f"{case}_sample_{idx:03d}_raw.obj")
                for idx in range(args.num_samples)
            ]
            sheet_path = samples_dir / split_name / case / f"{case}_latent_samples_only_large.png"
            render_contact_sheet(case, sample_meshes, sheet_path, args.dpi)
            samples_summary[split_name].append(
                {"case": case, "image": str(sheet_path), "num_samples": args.num_samples}
            )

    compare_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    (compare_dir / "summary.json").write_text(json.dumps(compare_summary, indent=2), encoding="utf-8")
    (samples_dir / "summary.json").write_text(json.dumps(samples_summary, indent=2), encoding="utf-8")
    print(compare_dir / "summary.json")
    print(samples_dir / "summary.json")


def pickle_load(path: Path):
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    main()
