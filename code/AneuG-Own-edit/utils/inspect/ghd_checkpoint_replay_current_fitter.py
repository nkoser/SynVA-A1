#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ghd.base.graph_harmonic_deformation import Graph_Harmonic_Deform_opening_alignment_dynamic
from ghd.fitting.fitter import initailize_registration


def parse_args() -> argparse.Namespace:
    checkpoints_root = ROOT / "checkpoint-v2"
    parser = argparse.ArgumentParser(description="Replay GHD checkpoints exactly as the current ghd_fitting code would.")
    parser.add_argument("--ghd-chk-root", type=Path, default=checkpoints_root / "ghd_fitting")
    parser.add_argument("--alignment-root", type=Path, default=ROOT / "alignment")
    parser.add_argument("--canonical-name", type=str, default="canonical_model")
    parser.add_argument("--canonical-eigen-chk", type=Path, default=None)
    parser.add_argument("--run-subdir", type=str, default="vanilla")
    parser.add_argument("--ghd-chk-name", type=str, default="ghb_fitting_checkpoint.pkl")
    parser.add_argument("--cases", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_obj_mesh(path: Path, device: torch.device) -> Meshes:
    verts, faces, _ = load_obj(str(path))
    return Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)])


def find_warped_mesh(run_root: Path) -> Path:
    preferred = run_root / "viz" / "warped_epoch_02999.obj"
    if preferred.exists():
        return preferred
    candidates = sorted((run_root / "viz").glob("warped_epoch_*.obj"))
    if not candidates:
        raise FileNotFoundError(f"No warped mesh found for {run_root}")
    return candidates[-1]


def mesh_radius(verts: np.ndarray) -> float:
    centered = verts - verts.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(centered, axis=1).max())


def direct_rmse(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.shape != b.shape:
        return None
    return float(np.sqrt(np.mean((a - b) ** 2)))


def build_replay_args(alignment_root: Path, case_name: str, device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        device=str(device),
        root_template=str(alignment_root),
        root_target=str(alignment_root),
        name_canonical="canonical_model",
        name_target=case_name,
        num_op=1,
        num_cep=3,
        num_waves=5,
        step_size=2,
        op_bold=1,
        pouch_only=1,
        center_opening_at_origin=0,
        center_opening_index=0,
        num_Basis=12 ** 2,
        mix_lap_weights=[1.0, 0.1, 0.1],
    )


def build_current_fitter_replay(
    alignment_root: Path,
    case_name: str,
    checkpoint: dict,
    device: torch.device,
    canonical_eigen_chk: Path | None,
) -> tuple[Meshes, float]:
    args = build_replay_args(alignment_root=alignment_root, case_name=case_name, device=device)
    canonical, _ = initailize_registration(args, hard_normalize=True, keep_size=False)
    eigen_chk = None
    if canonical_eigen_chk is not None and canonical_eigen_chk.exists():
        eigen_chk = str(canonical_eigen_chk)
    canonical_fitter = Graph_Harmonic_Deform_opening_alignment_dynamic(args, canonical, eigen_chk=eigen_chk)
    canonical_fitter = canonical_fitter.to(device)

    with torch.no_grad():
        canonical_fitter.R.data = checkpoint["R"].reshape(1, 3).to(device=device, dtype=torch.float32)
        canonical_fitter.s.data = checkpoint["s"].abs().reshape(1, 1).to(device=device, dtype=torch.float32)
        canonical_fitter.T.data = checkpoint["T"].reshape(1, 3).to(device=device, dtype=torch.float32)
        mesh, _ = canonical_fitter.forward_with_opening_alignment(
            checkpoint["GHD_coefficient"].reshape(-1, 3).to(device=device, dtype=torch.float32)
        )
    norm_canonical = float(torch.max(torch.norm(getattr(canonical, "mesh_target_p3d").verts_packed(), dim=-1)).detach().cpu().item())
    return mesh, norm_canonical


def render_mesh_grid(
    output_path: Path,
    title: str,
    entries: list[tuple[str, np.ndarray | None, np.ndarray | None]],
    dpi: int,
) -> None:
    finite_entries = [(label, verts, faces) for label, verts, faces in entries if verts is not None and np.isfinite(verts).all()]
    if finite_entries:
        all_verts = np.concatenate([verts for _, verts, _ in finite_entries], axis=0)
        center = 0.5 * (all_verts.min(axis=0) + all_verts.max(axis=0))
        radius = max(float((all_verts.max(axis=0) - all_verts.min(axis=0)).max()) * 0.5, 1e-3)
    else:
        center = np.zeros(3, dtype=np.float32)
        radius = 1.0

    fig = plt.figure(figsize=(4.2 * len(entries), 4.8), constrained_layout=True)
    fig.suptitle(title, fontsize=13)
    for idx, (label, verts, faces) in enumerate(entries, start=1):
        ax = fig.add_subplot(1, len(entries), idx, projection="3d")
        if verts is not None and faces is not None and np.isfinite(verts).all():
            ax.plot_trisurf(
                verts[:, 0],
                verts[:, 1],
                verts[:, 2],
                triangles=faces,
                color="#8eb3d3",
                edgecolor=(0.06, 0.12, 0.18, 0.08),
                linewidth=0.08,
                alpha=0.34,
                shade=True,
            )
        else:
            ax.text2D(0.5, 0.5, "Unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=12)
        ax.set_title(label, fontsize=10)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.view_init(elev=17, azim=35)
        ax.set_box_aspect([1.0, 1.0, 1.0])
        ax.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_cases = []
    for case in args.cases:
        case_root = args.ghd_chk_root.expanduser() / case
        run_root = case_root / args.run_subdir
        short_case = case.split("/", 1)[-1]
        checkpoint_path = run_root / args.ghd_chk_name
        with open(checkpoint_path, "rb") as f:
            chk = pickle.load(f)

        replay_mesh, norm_canonical = build_current_fitter_replay(
            alignment_root=args.alignment_root.expanduser(),
            case_name=short_case,
            checkpoint=chk,
            device=device,
            canonical_eigen_chk=args.canonical_eigen_chk.expanduser() if args.canonical_eigen_chk is not None else None,
        )
        warped_mesh_path = find_warped_mesh(run_root)
        target_mesh_path = run_root / "viz" / "target.obj"
        warped_mesh = load_obj_mesh(warped_mesh_path, device)
        target_mesh = load_obj_mesh(target_mesh_path, device)

        replay_verts = replay_mesh.verts_packed().detach().cpu().numpy()
        replay_faces = replay_mesh.faces_packed().detach().cpu().numpy()
        warped_verts = warped_mesh.verts_packed().detach().cpu().numpy()
        warped_faces = warped_mesh.faces_packed().detach().cpu().numpy()
        target_verts = target_mesh.verts_packed().detach().cpu().numpy()
        target_faces = target_mesh.faces_packed().detach().cpu().numpy()

        image_path = output_dir / f"{case.replace('/', '__')}_current_fitter_replay.png"
        render_mesh_grid(
            image_path,
            f"{case} [current_fitter_replay]",
            [
                ("Saved target.obj", target_verts, target_faces),
                ("Saved warped_epoch", warped_verts, warped_faces),
                ("Replay from checkpoint", replay_verts, replay_faces),
            ],
            dpi=args.dpi,
        )

        summary_cases.append(
            {
                "case": case,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "target_mesh_path": str(target_mesh_path.resolve()),
                "warped_mesh_path": str(warped_mesh_path.resolve()),
                "canonical_eigen_chk": str(args.canonical_eigen_chk.expanduser().resolve()) if args.canonical_eigen_chk is not None else None,
                "norm_canonical_used_by_current_fitter": norm_canonical,
                "replay_vs_warped_rmse": direct_rmse(replay_verts, warped_verts),
                "replay_radius": mesh_radius(replay_verts),
                "warped_radius": mesh_radius(warped_verts),
                "target_radius": mesh_radius(target_verts),
                "image_path": str(image_path),
            }
        )

    summary = {
        "mode": "current_fitter_replay",
        "cases": summary_cases,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote replay diagnostics to {summary_path}")


if __name__ == "__main__":
    main()
