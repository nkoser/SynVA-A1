#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from first_stage_ostium_conditional import compute_fitting_norm_canonical
from models.ghd_reconstruct import GHD_Reconstruct
from utils.utils import safe_load_mesh


@dataclass
class SimilarityFit:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    rmse: float


def parse_args() -> argparse.Namespace:
    checkpoints_root = ROOT / "checkpoint-v2"
    parser = argparse.ArgumentParser(description="Diagnose canonical/replay consistency for Stage-1 and GHD fitting.")
    parser.add_argument("--checkpoints-root", type=Path, default=checkpoints_root)
    parser.add_argument("--ghd-chk-root", type=Path, default=checkpoints_root / "ghd_fitting")
    parser.add_argument("--alignment-root", type=Path, default=ROOT / "alignment")
    parser.add_argument("--canonical-root-new", type=Path, default=checkpoints_root / "canonical_model")
    parser.add_argument("--canonical-root-old", type=Path, default=checkpoints_root / "canonical_model_old")
    parser.add_argument(
        "--eigen-chk-override",
        type=Path,
        default=None,
        help="Optional override for canonical_model_144_normed.pkl when canonical-root-new does not contain one.",
    )
    parser.add_argument("--cases", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_obj_mesh(path: Path, device: torch.device) -> Meshes:
    verts, faces, _ = load_obj(str(path))
    return Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)])


def find_warped_mesh(case_root: Path) -> Path:
    preferred = case_root / "vanilla" / "viz" / "warped_epoch_02999.obj"
    if preferred.exists():
        return preferred
    candidates = sorted((case_root / "vanilla" / "viz").glob("warped_epoch_*.obj"))
    if not candidates:
        raise FileNotFoundError(f"No warped mesh found for {case_root}")
    return candidates[-1]


def build_reconstructor(canonical_root: Path, device: torch.device, eigen_chk_override: Path | None = None) -> GHD_Reconstruct:
    canonical_meshes_raw = safe_load_mesh(str(canonical_root / "part_aligned.obj"))
    norm_canonical = compute_fitting_norm_canonical(canonical_meshes_raw)
    canonical_meshes = canonical_meshes_raw.update_padded(canonical_meshes_raw.verts_padded() / norm_canonical)
    eigen_chk = eigen_chk_override if eigen_chk_override is not None else (canonical_root / "canonical_model_144_normed.pkl")
    if not eigen_chk.exists():
        raise FileNotFoundError(f"Missing normalized canonical basis: {eigen_chk}")
    return GHD_Reconstruct(
        canonical_meshes,
        str(eigen_chk),
        num_Basis=12 ** 2,
        device=device,
        skip_normalize=True,
        norm_canonical_override=norm_canonical,
    )


def reconstruct_checkpoint_mesh(
    ghd_reconstruct: GHD_Reconstruct,
    ghd_chk: dict,
    device: torch.device,
    denormalize: bool,
) -> Meshes:
    ghd_coeff = ghd_chk["GHD_coefficient"].reshape(-1, 3).to(device=device, dtype=torch.float32)
    R = ghd_chk["R"].reshape(1, 3).to(device=device, dtype=torch.float32)
    s = ghd_chk["s"].abs().reshape(1, 1).to(device=device, dtype=torch.float32)
    T = ghd_chk["T"].reshape(1, 3).to(device=device, dtype=torch.float32)
    canonical_ghd = ghd_reconstruct.canonical_ghd
    with torch.no_grad():
        R_prev = canonical_ghd.R.detach().clone()
        s_prev = canonical_ghd.s.detach().clone()
        T_prev = canonical_ghd.T.detach().clone()
        try:
            canonical_ghd.R.data = R
            canonical_ghd.s.data = s
            canonical_ghd.T.data = T
            mesh = canonical_ghd.forward(ghd_coeff)
            if denormalize:
                mesh = mesh.update_padded(mesh.verts_padded() * ghd_reconstruct.norm_canonical)
        finally:
            canonical_ghd.R.data = R_prev
            canonical_ghd.s.data = s_prev
            canonical_ghd.T.data = T_prev
    return mesh


def verts_numpy(mesh: Meshes) -> np.ndarray:
    return mesh.verts_packed().detach().cpu().numpy()


def faces_numpy(mesh: Meshes) -> np.ndarray:
    return mesh.faces_packed().detach().cpu().numpy()


def mesh_radius(verts: np.ndarray) -> float:
    centered = verts - verts.mean(axis=0, keepdims=True)
    return float(np.linalg.norm(centered, axis=1).max())


def direct_rmse(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.shape != b.shape:
        return None
    return float(np.sqrt(np.mean((a - b) ** 2)))


def bidir_nn_rmse(a: np.ndarray, b: np.ndarray) -> float:
    tree_a = cKDTree(np.asarray(a, dtype=np.float64))
    tree_b = cKDTree(np.asarray(b, dtype=np.float64))
    dist_ab = tree_b.query(a, k=1)[0]
    dist_ba = tree_a.query(b, k=1)[0]
    return float(np.sqrt(0.5 * (np.mean(dist_ab ** 2) + np.mean(dist_ba ** 2))))


def umeyama_fit(source: np.ndarray, target: np.ndarray, with_scale: bool) -> SimilarityFit:
    if source.shape != target.shape:
        raise ValueError(f"Shape mismatch for alignment: {source.shape} vs {target.shape}")
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst
    cov = (dst_centered.T @ src_centered) / src.shape[0]
    U, singular_values, Vt = np.linalg.svd(cov)
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    rotation = U @ S @ Vt
    if with_scale:
        src_var = float(np.mean(np.sum(src_centered ** 2, axis=1)))
        scale = float(np.sum(singular_values * np.diag(S)) / max(src_var, 1e-12))
    else:
        scale = 1.0
    translation = mu_dst - scale * (rotation @ mu_src)
    aligned = apply_similarity(source, scale, rotation, translation)
    rmse = float(np.sqrt(np.mean((aligned - target) ** 2)))
    return SimilarityFit(scale=scale, rotation=rotation.astype(np.float32), translation=translation.astype(np.float32), rmse=rmse)


def apply_similarity(verts: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (scale * (verts @ rotation.T) + translation).astype(np.float32)


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

    fig = plt.figure(figsize=(3.9 * len(entries), 4.6), constrained_layout=True)
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


def compare_canonical_meshes(
    canonical_root_old: Path,
    canonical_root_new: Path,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    old_mesh = load_obj_mesh(canonical_root_old / "part_aligned.obj", device)
    new_mesh = load_obj_mesh(canonical_root_new / "part_aligned.obj", device)
    old_verts = verts_numpy(old_mesh)
    new_verts = verts_numpy(new_mesh)
    old_faces = faces_numpy(old_mesh)
    rigid = umeyama_fit(old_verts, new_verts, with_scale=False)
    similarity = umeyama_fit(old_verts, new_verts, with_scale=True)
    old_aligned = apply_similarity(old_verts, rigid.scale, rigid.rotation, rigid.translation)
    old_similarity = apply_similarity(old_verts, similarity.scale, similarity.rotation, similarity.translation)
    summary = {
        "old_root": str(canonical_root_old.resolve()),
        "new_root": str(canonical_root_new.resolve()),
        "vertex_count_old": int(old_verts.shape[0]),
        "vertex_count_new": int(new_verts.shape[0]),
        "direct_rmse": direct_rmse(old_verts, new_verts),
        "centroid_old": old_verts.mean(axis=0).tolist(),
        "centroid_new": new_verts.mean(axis=0).tolist(),
        "centroid_offset_norm": float(np.linalg.norm(old_verts.mean(axis=0) - new_verts.mean(axis=0))),
        "radius_old": mesh_radius(old_verts),
        "radius_new": mesh_radius(new_verts),
        "radius_ratio_new_over_old": float(mesh_radius(new_verts) / max(mesh_radius(old_verts), 1e-12)),
        "rigid_rmse": rigid.rmse,
        "similarity_rmse": similarity.rmse,
        "similarity_scale_old_to_new": similarity.scale,
    }
    return summary, new_verts, old_aligned, old_similarity


def compare_case(
    case: str,
    ghd_chk_root: Path,
    alignment_root: Path,
    canonical_root_old: Path,
    canonical_root_new: Path,
    eigen_chk_override: Path | None,
    device: torch.device,
) -> tuple[dict, list[tuple[str, list[tuple[str, np.ndarray | None, np.ndarray | None]]]]]:
    case_root = ghd_chk_root / case
    alignment_case_root = alignment_root / case.split("/", 1)[-1]
    alignment_mesh = load_obj_mesh(alignment_case_root / "part_aligned.obj", device)
    alignment_verts = verts_numpy(alignment_mesh)
    alignment_faces = faces_numpy(alignment_mesh)
    target_mesh = load_obj_mesh(case_root / "vanilla" / "viz" / "target.obj", device)
    target_verts = verts_numpy(target_mesh)
    target_faces = faces_numpy(target_mesh)
    warped_mesh = load_obj_mesh(find_warped_mesh(case_root), device)
    warped_verts = verts_numpy(warped_mesh)
    warped_faces = faces_numpy(warped_mesh)

    with open(case_root / "vanilla" / "ghb_fitting_checkpoint.pkl", "rb") as f:
        chk = pickle.load(f)

    reconstructor_specs = [
        ("old", canonical_root_old, None),
        ("new", canonical_root_new, eigen_chk_override),
    ]

    summary = {
        "case": case,
        "alignment_mesh_path": str((alignment_case_root / "part_aligned.obj").resolve()),
        "target_mesh_path": str((case_root / "vanilla" / "viz" / "target.obj").resolve()),
        "warped_mesh_path": str(find_warped_mesh(case_root).resolve()),
        "checkpoint_path": str((case_root / "vanilla" / "ghb_fitting_checkpoint.pkl").resolve()),
        "alignment_vertex_count": int(alignment_verts.shape[0]),
        "target_vertex_count": int(target_verts.shape[0]),
        "warped_vertex_count": int(warped_verts.shape[0]),
        "alignment_radius": mesh_radius(alignment_verts),
        "target_radius": mesh_radius(target_verts),
        "warped_radius": mesh_radius(warped_verts),
        "target_vs_alignment_nn_rmse": bidir_nn_rmse(target_verts, alignment_verts),
        "target_vs_alignment_radius_ratio": float(mesh_radius(target_verts) / max(mesh_radius(alignment_verts), 1e-12)),
        "warped_vs_target_raw_rmse": direct_rmse(warped_verts, target_verts),
    }

    fitting_panels: list[tuple[str, np.ndarray | None, np.ndarray | None]] = [
        ("Saved target.obj", target_verts, target_faces),
        ("Saved warped_epoch", warped_verts, warped_faces),
    ]
    raw_panels: list[tuple[str, np.ndarray | None, np.ndarray | None]] = [
        ("Alignment part_aligned", alignment_verts, alignment_faces),
    ]

    for label, canonical_root, basis_override in reconstructor_specs:
        key = f"replay_{label}_canonical"
        try:
            reconstructor = build_reconstructor(canonical_root, device, eigen_chk_override=basis_override)
            replay_norm_mesh = reconstruct_checkpoint_mesh(reconstructor, chk, device, denormalize=False)
            replay_raw_mesh = reconstruct_checkpoint_mesh(reconstructor, chk, device, denormalize=True)
        except Exception as exc:  # noqa: BLE001
            summary[f"{key}_error"] = str(exc)
            fitting_panels.append((f"{label} replay\nUnavailable", None, None))
            raw_panels.append((f"{label} replay raw\nUnavailable", None, None))
            continue

        replay_norm_verts = verts_numpy(replay_norm_mesh)
        replay_norm_faces = faces_numpy(replay_norm_mesh)
        replay_raw_verts = verts_numpy(replay_raw_mesh)
        replay_raw_faces = faces_numpy(replay_raw_mesh)

        rigid_warped = umeyama_fit(replay_norm_verts, warped_verts, with_scale=False)
        sim_warped = umeyama_fit(replay_norm_verts, warped_verts, with_scale=True)
        rigid_align = umeyama_fit(replay_raw_verts, alignment_verts, with_scale=False) if replay_raw_verts.shape == alignment_verts.shape else None
        sim_align = umeyama_fit(replay_raw_verts, alignment_verts, with_scale=True) if replay_raw_verts.shape == alignment_verts.shape else None

        summary[key] = {
            "fitting_space": {
                "raw_rmse_vs_warped": direct_rmse(replay_norm_verts, warped_verts),
                "nn_rmse_vs_target_obj": bidir_nn_rmse(replay_norm_verts, target_verts),
                "radius": mesh_radius(replay_norm_verts),
                "radius_ratio_replay_over_warped": float(mesh_radius(replay_norm_verts) / max(mesh_radius(warped_verts), 1e-12)),
                "rigid_rmse_vs_warped": rigid_warped.rmse,
                "similarity_rmse_vs_warped": sim_warped.rmse,
                "similarity_scale_replay_to_warped": sim_warped.scale,
            },
            "raw_alignment_space": {
                "nn_rmse_vs_alignment_part": bidir_nn_rmse(replay_raw_verts, alignment_verts),
                "radius": mesh_radius(replay_raw_verts),
                "radius_ratio_replay_over_alignment": float(mesh_radius(replay_raw_verts) / max(mesh_radius(alignment_verts), 1e-12)),
                "rigid_rmse_vs_alignment": rigid_align.rmse if rigid_align is not None else None,
                "similarity_rmse_vs_alignment": sim_align.rmse if sim_align is not None else None,
                "similarity_scale_replay_to_alignment": sim_align.scale if sim_align is not None else None,
            },
            "norm_canonical": float(reconstructor.norm_canonical),
            "basis_path": str((basis_override if basis_override is not None else (canonical_root / "canonical_model_144_normed.pkl")).resolve()),
        }

        fitting_panels.extend(
            [
                (f"{label} replay norm", replay_norm_verts, replay_norm_faces),
                (f"{label} rigid->warped {rigid_warped.rmse:.5f}", apply_similarity(replay_norm_verts, rigid_warped.scale, rigid_warped.rotation, rigid_warped.translation), replay_norm_faces),
                (f"{label} sim->warped {sim_warped.rmse:.5f}\nscale {sim_warped.scale:.4f}", apply_similarity(replay_norm_verts, sim_warped.scale, sim_warped.rotation, sim_warped.translation), replay_norm_faces),
            ]
        )
        raw_panels.extend(
            [
                (f"{label} replay raw", replay_raw_verts, replay_raw_faces),
            ]
        )

    return summary, [("fitting_space", fitting_panels), ("raw_alignment_space", raw_panels)]


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_summary, _, canonical_old_rigid, canonical_old_similarity = compare_canonical_meshes(
        args.canonical_root_old.expanduser(),
        args.canonical_root_new.expanduser(),
        device,
    )
    old_mesh = load_obj_mesh(args.canonical_root_old.expanduser() / "part_aligned.obj", device)
    new_mesh = load_obj_mesh(args.canonical_root_new.expanduser() / "part_aligned.obj", device)
    render_mesh_grid(
        output_dir / "canonical_comparison.png",
        "Canonical Comparison",
        [
            ("Old canonical raw", verts_numpy(old_mesh), faces_numpy(old_mesh)),
            ("New canonical raw", verts_numpy(new_mesh), faces_numpy(new_mesh)),
            ("Old -> new rigid aligned", canonical_old_rigid, faces_numpy(old_mesh)),
            ("Old -> new similarity aligned", canonical_old_similarity, faces_numpy(old_mesh)),
        ],
        dpi=args.dpi,
    )

    case_summaries = []
    for case in args.cases:
        summary, panel_groups = compare_case(
            case=case,
            ghd_chk_root=args.ghd_chk_root.expanduser(),
            alignment_root=args.alignment_root.expanduser(),
            canonical_root_old=args.canonical_root_old.expanduser(),
            canonical_root_new=args.canonical_root_new.expanduser(),
            eigen_chk_override=args.eigen_chk_override.expanduser() if args.eigen_chk_override is not None else None,
            device=device,
        )
        for group_name, panels in panel_groups:
            image_path = output_dir / f"{case.replace('/', '__')}_{group_name}.png"
            render_mesh_grid(image_path, f"{case} [{group_name}]", panels, dpi=args.dpi)
            summary[f"{group_name}_image_path"] = str(image_path)
        case_summaries.append(summary)

    summary = {
        "canonical_summary": canonical_summary,
        "cases": case_summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote diagnostics to {summary_path}")


if __name__ == "__main__":
    main()
