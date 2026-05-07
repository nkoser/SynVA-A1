#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from first_stage_ostium_conditional import (  # noqa: E402
    collect_available_cases,
    ensure_canonical_diff_checkpoint,
    load_or_create_split,
    load_split_from_folders,
    prepare_ghd_condition_opa_checkpoints,
    resolve_case_identifier,
)
from models.ghd_reconstruct import GHD_Reconstruct  # noqa: E402
from models.vae_datasets import OstiumGHDDataset  # noqa: E402
from models.vae_models import ConditionalGHDVAE  # noqa: E402
from pytorch3d.io import load_obj  # noqa: E402
from pytorch3d.structures import Meshes  # noqa: E402
from utils.utils import safe_load_mesh  # noqa: E402


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    checkpoints_root = PROJECT_ROOT / "checkpoint-v2"
    parser = argparse.ArgumentParser(
        description="Inspect one Stage-1 ostium VAE sample: input mesh + ostium + reconstructions for multiple epochs."
    )
    parser.add_argument("--checkpoints-root", type=Path, default=checkpoints_root)
    parser.add_argument("--ghd-chk-root", type=Path, default=checkpoints_root / "ghd_fitting")
    parser.add_argument("--canonical-root", type=Path, default=checkpoints_root / "canonical_model")
    parser.add_argument(
        "--condition-root",
        type=Path,
        default=None,
        help="Root with per-case condition opa_checkpoint.pkl. Defaults to --ghd-chk-root.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=checkpoints_root / "first_stage_ostium_conditional" / "ostium_pouch_new_inf",
        help="Directory containing models_epoch_<N>.pth.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=[200, 6500, 23000],
        help="Checkpoint epochs to compare.",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Case id (e.g. train/C0001). If omitted, first test case is used.",
    )
    parser.add_argument(
        "--all-cases",
        type=int,
        default=0,
        help="1: render all selected cases (filtered by --splits), 0: render a single case.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "test"],
        help="Split prefixes used when --all-cases=1 (e.g. train test).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional cap for number of rendered cases (for debugging).",
    )
    parser.add_argument("--ring-points", type=int, default=20)
    parser.add_argument("--split-train-ratio", type=float, default=0.8)
    parser.add_argument("--split-val-ratio", type=float, default=0.1)
    parser.add_argument("--split-test-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--force-resplit", type=int, default=0)
    parser.add_argument(
        "--train-subset-limit",
        type=int,
        default=None,
        help="Optional cap matching the overfit train subset used during training.",
    )
    parser.add_argument(
        "--ostium-source",
        type=str,
        choices=["opa_checkpoint", "opening_debug", "condition_mapped", "canonical_idx"],
        default="opa_checkpoint",
        help=(
            "opa_checkpoint: use op_v_indices from <condition-root>/<case>/opa_checkpoint.pkl, "
            "opening_debug: use fitting opening mesh from viz/opening_debug, "
            "condition_mapped: use VAE condition ring mapped to input frame, "
            "canonical_idx: use canonical opening indices on current mesh"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=checkpoints_root / "vae_inspect",
        help="Output directory for rendered image.",
    )
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--prepare-condition-from-ghd", type=int, default=0)
    parser.add_argument("--force-prepare-condition-from-ghd", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def infer_model_hparams(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int, int, str]:
    hidden_dim = int(state_dict["fc1.weight"].shape[0])
    input_dim = int(state_dict["fc1.weight"].shape[1])
    latent_dim = int(state_dict["fc21.weight"].shape[0])
    if "cond_encoder.2.weight" in state_dict:
        cond_embed_dim = int(state_dict["cond_encoder.2.weight"].shape[0])
    else:
        cond_embed_dim = int(state_dict["fc3.weight"].shape[1] - latent_dim)
    if "res1.bn1.running_mean" in state_dict:
        norm_type = "batch"
    elif "res1.bn1.weight" in state_dict:
        norm_type = "layer"
    else:
        norm_type = "none"
    return input_dim, hidden_dim, latent_dim, cond_embed_dim, norm_type


def load_generators(
    epochs: list[int],
    model_dir: Path,
    dataset: OstiumGHDDataset,
    device: torch.device,
) -> list[tuple[int, ConditionalGHDVAE]]:
    generators: list[tuple[int, ConditionalGHDVAE]] = []
    for epoch in epochs:
        checkpoint_path = model_dir / f"models_epoch_{epoch}.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator_state = checkpoint["generator"]
        input_dim, hidden_dim, latent_dim, cond_embed_dim, norm_type = infer_model_hparams(generator_state)
        generator = ConditionalGHDVAE(
            input_dim,
            hidden_dim,
            latent_dim,
            cond_dim=dataset.get_cond_dim(),
            cond_embed_dim=cond_embed_dim,
            norm_type=norm_type,
        ).to(device)
        generator.load_state_dict(generator_state)
        generator.eval()
        generators.append((epoch, generator))
    return generators


def select_cases(all_cases: list[str], args: argparse.Namespace) -> list[str]:
    if int(args.all_cases) != 1:
        return [choose_case(all_cases, args.case)]

    split_prefixes = tuple(f"{split_name.strip('/')}/" for split_name in args.splits)
    selected = [case for case in all_cases if case.startswith(split_prefixes)]
    selected = sorted(selected)
    if args.max_cases is not None and args.max_cases > 0:
        selected = selected[: int(args.max_cases)]
    return selected


def build_ghd_reconstruct(canonical_root: Path, device: torch.device) -> GHD_Reconstruct:
    canonical_mesh_raw = safe_load_mesh(str(canonical_root / "part_aligned.obj"))
    ensure_canonical_diff_checkpoint(canonical_root, canonical_mesh_raw)

    verts_raw = canonical_mesh_raw.verts_packed()
    norm_canonical = torch.max(torch.norm(verts_raw, dim=-1)).item() * 1.10
    verts_normed = verts_raw / norm_canonical
    canonical_mesh_normed = Meshes(verts=[verts_normed], faces=canonical_mesh_raw.faces_list())

    return GHD_Reconstruct(
        canonical_mesh_normed,
        str(canonical_root / "canonical_model_144_normed.pkl"),
        num_Basis=12**2,
        device=device,
        skip_normalize=True,
        norm_canonical_override=norm_canonical,
    )


def choose_case(cases: list[str], requested_case: str | None) -> str:
    if requested_case is not None:
        return resolve_case_identifier(requested_case, cases)
    test_cases = [case for case in cases if case.startswith("test/")]
    if test_cases:
        return sorted(test_cases)[0]
    return sorted(cases)[0]


def maybe_apply_training_stats(
    dataset: OstiumGHDDataset,
    model_dir: Path,
    available_cases: list[str],
    args: argparse.Namespace,
    checkpoints_root: Path,
    ghd_chk_root: Path,
) -> None:
    checkpoint_with_stats = None
    for epoch in sorted(args.epochs, reverse=True):
        checkpoint_path = model_dir / f"models_epoch_{epoch}.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if all(key in checkpoint for key in ("target_mean", "target_std", "cond_mean", "cond_std")):
                checkpoint_with_stats = checkpoint
                break

    if checkpoint_with_stats is not None:
        dataset.target_mean = checkpoint_with_stats["target_mean"].float()
        dataset.target_std = checkpoint_with_stats["target_std"].float()
        dataset.cond_mean = checkpoint_with_stats["cond_mean"].float()
        dataset.cond_std = checkpoint_with_stats["cond_std"].float()
        print("Loaded normalization statistics from checkpoint.")
        return

    split_cases = load_split_from_folders(
        ghd_chk_root=ghd_chk_root,
        available_cases=available_cases,
        split_val_ratio=float(args.split_val_ratio),
        split_seed=int(args.split_seed),
    )
    if split_cases is None:
        if args.split_file is None:
            split_file = checkpoints_root / "dataset_splits" / f"ostium_conditional_split_seed{args.split_seed}.json"
        else:
            split_file = args.split_file.expanduser()
        split_cases = load_or_create_split(split_file, available_cases, args)

    if args.train_subset_limit is not None:
        split_cases = {
            "train": split_cases["train"][: int(args.train_subset_limit)],
            "val": split_cases["val"],
            "test": split_cases["test"],
        }

    case_to_index = {case: idx for idx, case in enumerate(dataset.updated_cases)}
    train_indices = [case_to_index[case] for case in split_cases["train"] if case in case_to_index]
    if not train_indices:
        raise RuntimeError("Could not recover any train cases to rebuild normalization statistics.")

    if dataset.withscale:
        train_targets = torch.stack(
            [torch.cat([dataset.ghd[idx], dataset.scale[idx]]) for idx in train_indices], dim=0
        )
    else:
        train_targets = torch.stack([dataset.ghd[idx] for idx in train_indices], dim=0)
    train_conditions = torch.stack([dataset.ostium_condition[idx] for idx in train_indices], dim=0)
    dataset.target_mean = train_targets.mean(dim=0, keepdim=True)
    dataset.target_std = train_targets.std(dim=0, keepdim=True) + 0.01
    dataset.cond_mean = train_conditions.mean(dim=0, keepdim=True)
    dataset.cond_std = train_conditions.std(dim=0, keepdim=True) + 0.01
    print(
        "Rebuilt normalization statistics from train split "
        f"(train_cases={len(train_indices)}, train_subset_limit={args.train_subset_limit})."
    )


def find_latest_warped_mesh(case_root: Path) -> Path:
    viz_dir = case_root / "vanilla" / "viz"
    preferred = viz_dir / "warped_epoch_02999.obj"
    if preferred.exists():
        return preferred
    candidates = sorted(viz_dir.glob("warped_epoch_*.obj"))
    if not candidates:
        raise FileNotFoundError(f"No warped mesh found in {viz_dir}")
    return candidates[-1]


def find_latest_opening_debug_mesh(case_root: Path) -> Path | None:
    opening_dir = case_root / "vanilla" / "viz" / "opening_debug"
    if not opening_dir.exists():
        return None
    preferred = opening_dir / "opening_debug_epoch_02999.obj"
    if preferred.exists():
        return preferred
    candidates = sorted(opening_dir.glob("opening_debug_epoch_*.obj"))
    if not candidates:
        return None
    return candidates[-1]


def load_obj_as_mesh(path: Path, device: torch.device) -> Meshes:
    verts, faces, _ = load_obj(str(path))
    return Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)])


def load_case_condition_ring(alignment_root: Path, case_name: str) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    opa_path = alignment_root / case_name / "opa_checkpoint.pkl"
    if not opa_path.exists():
        return None, None, "missing"
    with open(opa_path, "rb") as f:
        chk = pickle.load(f)
    source = str(chk.get("source", "unknown"))
    ring_coords = np.asarray(chk["op_v_coords"][0], dtype=np.float32)
    ring_idx = np.asarray(chk["op_v_indices"][0], dtype=np.int64)
    if ring_coords.ndim != 2 or ring_coords.shape[1] != 3 or ring_coords.shape[0] < 3:
        return None, None, source
    if ring_idx.ndim != 1 or ring_idx.shape[0] < 3:
        ring_idx = None
    return ring_coords, ring_idx, source


def load_case_opa_indices(condition_root: Path, case_name: str) -> tuple[np.ndarray, str]:
    opa_path = condition_root / case_name / "opa_checkpoint.pkl"
    if not opa_path.exists():
        raise FileNotFoundError(f"Missing OPA checkpoint for ostium rendering: {opa_path}")
    with open(opa_path, "rb") as f:
        chk = pickle.load(f)
    if "op_v_indices" not in chk or not chk["op_v_indices"]:
        raise KeyError(f"OPA checkpoint has no op_v_indices: {opa_path}")
    ring_idx = np.asarray(chk["op_v_indices"][0], dtype=np.int64)
    if ring_idx.ndim != 1 or ring_idx.shape[0] < 3:
        raise ValueError(f"Invalid op_v_indices in {opa_path}: shape={ring_idx.shape}")
    return ring_idx, str(chk.get("source", "unknown"))


def load_alignment_opening_mesh(
    alignment_root: Path,
    case_name: str,
) -> tuple[torch.Tensor, torch.Tensor, str] | None:
    opening_dir = alignment_root / case_name / "inspection_opening_planes"
    for filename in ("target_opening_matched.obj", "target_opening_00.obj"):
        path = opening_dir / filename
        if not path.exists():
            continue
        verts, faces, _ = load_obj(str(path))
        if verts.shape[0] >= 3:
            return verts.float(), faces.verts_idx.long(), filename
    return None


def extract_obj_object_mesh(path: Path, object_name: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    current = None
    global_vertex_index = 0
    verts: list[list[float]] = []
    global_to_local: dict[int, int] = {}
    faces: list[list[int]] = []

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("o "):
            current = line.split(maxsplit=1)[1].strip()
            continue
        if line.startswith("v "):
            global_vertex_index += 1
            if current == object_name:
                tokens = line.split()
                if len(tokens) >= 4:
                    global_to_local[global_vertex_index] = len(verts)
                    verts.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            continue
        if current == object_name and line.startswith("f "):
            tokens = line.split()[1:]
            indices: list[int] = []
            for token in tokens:
                raw_idx = token.split("/")[0]
                if not raw_idx:
                    continue
                idx = int(raw_idx)
                if idx < 0:
                    idx = global_vertex_index + 1 + idx
                if idx in global_to_local:
                    indices.append(global_to_local[idx])
            if len(indices) >= 3:
                first = indices[0]
                for j in range(1, len(indices) - 1):
                    faces.append([first, indices[j], indices[j + 1]])

    if not verts:
        return None
    verts_t = torch.tensor(verts, dtype=torch.float32)
    if faces:
        faces_t = torch.tensor(faces, dtype=torch.int64)
    else:
        faces_t = torch.empty((0, 3), dtype=torch.int64)
    return verts_t, faces_t


def boundary_loop_from_faces(num_verts: int, faces: torch.Tensor) -> list[int] | None:
    if faces.numel() == 0:
        return None

    edge_count: dict[tuple[int, int], int] = {}
    for tri in faces.tolist():
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        tri_edges = ((a, b), (b, c), (c, a))
        for u, v in tri_edges:
            key = (u, v) if u < v else (v, u)
            edge_count[key] = edge_count.get(key, 0) + 1

    boundary_edges = [edge for edge, cnt in edge_count.items() if cnt == 1]
    if not boundary_edges:
        return None

    adj: dict[int, list[int]] = {}
    for u, v in boundary_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    visited = set()
    loops: list[list[int]] = []
    for start in sorted(adj.keys()):
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        prev = None
        cur = start
        guard = 0
        while guard < max(num_verts * 3, 16):
            neighbors = adj.get(cur, [])
            if not neighbors:
                break
            nxt = None
            for cand in neighbors:
                if cand != prev:
                    nxt = cand
                    break
            if nxt is None:
                break
            if nxt == start:
                break
            loop.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
            guard += 1
        if len(loop) >= 3:
            loops.append(loop)

    if not loops:
        return None
    loops.sort(key=len, reverse=True)
    return loops[0]


def order_ring_points(points: torch.Tensor) -> torch.Tensor:
    if points.shape[0] <= 3:
        return points
    center = points.mean(dim=0, keepdim=True)
    centered = points - center
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis_u = vh[0]
    basis_v = vh[1]
    uv_x = centered @ basis_u
    uv_y = centered @ basis_v
    angles = torch.atan2(uv_y, uv_x)
    order = torch.argsort(angles)
    return points[order]


def order_ring_indices_by_points(indices: np.ndarray, points: torch.Tensor) -> np.ndarray:
    if points.shape[0] != indices.shape[0]:
        raise ValueError(f"Index/point count mismatch: {indices.shape[0]} vs {points.shape[0]}")
    if points.shape[0] <= 3:
        return indices
    center = points.mean(dim=0, keepdim=True)
    centered = points - center
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    basis_u = vh[0]
    basis_v = vh[1]
    uv_x = centered @ basis_u
    uv_y = centered @ basis_v
    angles = torch.atan2(uv_y, uv_x)
    order = torch.argsort(angles).detach().cpu().numpy().astype(np.int64)
    return indices[order]


def project_ring_to_mesh(ring_points: np.ndarray, mesh_vertices: np.ndarray) -> np.ndarray:
    points_t = torch.from_numpy(ring_points.astype(np.float32))
    verts_t = torch.from_numpy(mesh_vertices.astype(np.float32))
    nearest_idx = torch.cdist(points_t.unsqueeze(0), verts_t.unsqueeze(0)).squeeze(0).argmin(dim=1).cpu().numpy()

    cleaned_idx: list[int] = []
    for idx in nearest_idx.tolist():
        if not cleaned_idx or cleaned_idx[-1] != idx:
            cleaned_idx.append(idx)
    if len(cleaned_idx) >= 2 and cleaned_idx[0] == cleaned_idx[-1]:
        cleaned_idx = cleaned_idx[:-1]

    if len(cleaned_idx) < 3:
        unique_idx = []
        seen = set()
        for idx in nearest_idx.tolist():
            if idx in seen:
                continue
            seen.add(idx)
            unique_idx.append(idx)
        cleaned_idx = unique_idx

    if not cleaned_idx:
        return np.empty((0, 3), dtype=np.float32)
    return mesh_vertices[np.asarray(cleaned_idx, dtype=np.int64)]


def reconstruct_fitted_mesh_from_checkpoint(
    ghd_reconstruct: GHD_Reconstruct,
    ghd_chk: dict,
    device: torch.device,
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
            mesh = mesh.update_padded(mesh.verts_padded() * ghd_reconstruct.norm_canonical)
        finally:
            canonical_ghd.R.data = R_prev
            canonical_ghd.s.data = s_prev
            canonical_ghd.T.data = T_prev
    return mesh


def fit_similarity_transform(source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_center = source.mean(dim=0, keepdim=True)
    target_center = target.mean(dim=0, keepdim=True)
    source_centered = source - source_center
    target_centered = target - target_center

    source_scale = torch.sqrt((source_centered**2).sum())
    target_scale = torch.sqrt((target_centered**2).sum())
    source_norm = source_centered / (source_scale + 1e-12)
    target_norm = target_centered / (target_scale + 1e-12)

    covariance = source_norm.transpose(0, 1) @ target_norm
    U, _, Vh = torch.linalg.svd(covariance)
    rotation = Vh.transpose(0, 1) @ U.transpose(0, 1)
    if torch.det(rotation) < 0:
        Vh_fix = Vh.clone()
        Vh_fix[-1, :] *= -1
        rotation = Vh_fix.transpose(0, 1) @ U.transpose(0, 1)

    scale = target_scale / (source_scale + 1e-12)
    translation = target_center.squeeze(0) - scale * (source_center.squeeze(0) @ rotation)
    return scale, rotation, translation


def apply_similarity(points: torch.Tensor, scale: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    return scale * (points @ rotation) + translation


def nearest_distances(points: torch.Tensor, mesh_vertices: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(points.unsqueeze(0), mesh_vertices.unsqueeze(0)).squeeze(0)
    return dist.min(dim=1).values


def mesh_to_numpy(mesh: Meshes) -> tuple[np.ndarray, np.ndarray]:
    return (
        mesh.verts_padded()[0].detach().cpu().numpy(),
        mesh.faces_padded()[0].detach().cpu().numpy(),
    )


def apply_similarity_mesh(mesh: Meshes, scale: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> Meshes:
    verts = apply_similarity(mesh.verts_padded()[0], scale, rotation, translation)
    return Meshes(verts=[verts], faces=[mesh.faces_padded()[0]])


def render_panels(
    output_path: Path,
    case_name: str,
    input_title: str,
    ostium_indices: np.ndarray | None,
    ostium_points: np.ndarray,
    input_mesh_np: tuple[np.ndarray, np.ndarray],
    recon_panels: list[tuple[int, float, tuple[np.ndarray, np.ndarray]]],
    dpi: int,
) -> None:
    panels = [(input_title, None, input_mesh_np)]
    for epoch, rmse, mesh_np in recon_panels:
        panels.append((f"VAE Epoch {epoch}", rmse, mesh_np))

    all_vertices = np.concatenate([panel[2][0] for panel in panels], axis=0)
    min_xyz = all_vertices.min(axis=0)
    max_xyz = all_vertices.max(axis=0)
    center = (min_xyz + max_xyz) * 0.5
    extent = float(np.max(max_xyz - min_xyz))
    radius = max(0.5 * extent, 1e-3)

    fig = plt.figure(figsize=(5.0 * len(panels), 5.8), constrained_layout=True)
    fig.suptitle(f"Stage-1 Ostium VAE Reconstruction | {case_name}", fontsize=14)

    for i, (title, rmse, mesh_np) in enumerate(panels, start=1):
        verts, faces = mesh_np
        ax = fig.add_subplot(1, len(panels), i, projection="3d")
        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            triangles=faces,
            color="#8eb3d3",
            edgecolor=(0.06, 0.12, 0.18, 0.09),
            linewidth=0.08,
            antialiased=True,
            alpha=0.28,
            shade=True,
        )

        if ostium_indices is not None and ostium_indices.size >= 3:
            valid_idx = ostium_indices[(ostium_indices >= 0) & (ostium_indices < verts.shape[0])]
            ostium_on_mesh = verts[valid_idx] if valid_idx.size >= 3 else np.empty((0, 3), dtype=np.float32)
        else:
            ostium_on_mesh = project_ring_to_mesh(ostium_points, verts)
        if ostium_on_mesh.shape[0] > 0:
            loop = np.vstack([ostium_on_mesh, ostium_on_mesh[:1]])
            ax.plot(
                loop[:, 0],
                loop[:, 1],
                loop[:, 2],
                color="#e63946",
                linewidth=2.2,
                alpha=0.98,
            )
            ax.scatter(
                ostium_on_mesh[:, 0],
                ostium_on_mesh[:, 1],
                ostium_on_mesh[:, 2],
                s=10,
                c="#e63946",
                depthshade=False,
                alpha=0.95,
            )

        panel_title = title if rmse is None else f"{title}\nVertex RMSE: {rmse:.5f}"
        ax.set_title(panel_title, fontsize=11)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.view_init(elev=17, azim=35)
        ax.set_box_aspect([1.0, 1.0, 1.0])
        ax.set_axis_off()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def process_case(
    case_name: str,
    args: argparse.Namespace,
    device: torch.device,
    ghd_chk_root: Path,
    dataset: OstiumGHDDataset,
    case_to_idx: dict[str, int],
    ghd_reconstruct: GHD_Reconstruct,
    generators: list[tuple[int, ConditionalGHDVAE]],
    ghd_mean: torch.Tensor,
    ghd_std: torch.Tensor,
    ghd_dim: int,
    condition_root: Path,
    alignment_root: Path,
    output_dir: Path,
    output_name_override: str | None = None,
) -> dict:
    if case_name not in case_to_idx:
        raise ValueError(f"Case {case_name} is not available in assembled dataset.")
    idx = case_to_idx[case_name]
    case_root = ghd_chk_root / case_name

    warped_mesh_path = find_latest_warped_mesh(case_root)
    input_mesh = load_obj_as_mesh(warped_mesh_path, device)
    input_vertices = input_mesh.verts_padded()[0]

    batch = dataset[idx]
    target_norm = batch["target"].unsqueeze(0).to(device)
    condition_norm = batch["condition"].unsqueeze(0).to(device)
    condition_ring_denorm = dataset.load_case_ring(case_name).view(-1, 3).to(device)
    target_ghd = target_norm[:, :ghd_dim]
    target_scale_norm = target_norm[:, ghd_dim:]
    target_scale = dataset.denorm_scale(target_scale_norm).to(device)

    target_mesh_canonical = ghd_reconstruct.ghd_forward_as_Meshes(
        target_ghd,
        denormalize_shape=False,
        mean=ghd_mean,
        std=ghd_std,
        scale=target_scale,
    )
    target_vertices_canonical = target_mesh_canonical.verts_padded()[0]
    render_scale, render_rotation, render_translation = fit_similarity_transform(
        target_vertices_canonical, input_vertices
    )

    ghd_chk_path = case_root / "vanilla" / "ghb_fitting_checkpoint.pkl"
    with open(ghd_chk_path, "rb") as f:
        ghd_chk = pickle.load(f)
    mesh_input_denorm = reconstruct_fitted_mesh_from_checkpoint(ghd_reconstruct, ghd_chk, device)
    input_denorm_vertices = mesh_input_denorm.verts_padded()[0]

    sim_scale, sim_rotation, sim_translation = fit_similarity_transform(input_denorm_vertices, input_vertices)
    condition_ring_mapped = apply_similarity(condition_ring_denorm, sim_scale, sim_rotation, sim_translation)

    cond_ring_coords, cond_ring_idx, cond_source = load_case_condition_ring(condition_root, case_name)
    alignment_opening_mesh = load_alignment_opening_mesh(alignment_root, case_name)

    opening_debug_points = None
    opening_debug_ring = None
    opening_debug_path = find_latest_opening_debug_mesh(case_root)
    if opening_debug_path is not None:
        opening_mesh = extract_obj_object_mesh(opening_debug_path, "opening_0_warped_mesh")
        if opening_mesh is not None:
            opening_debug_points, opening_debug_faces = opening_mesh
            opening_debug_points = opening_debug_points.to(device)
            opening_debug_faces = opening_debug_faces.to(device)
            loop_idx = boundary_loop_from_faces(opening_debug_points.shape[0], opening_debug_faces)
            if loop_idx is not None and len(loop_idx) >= 3:
                opening_debug_ring = opening_debug_points[torch.tensor(loop_idx, dtype=torch.int64, device=device)]
            else:
                opening_debug_ring = order_ring_points(opening_debug_points)

    ostium_indices = None
    if args.ostium_source == "opa_checkpoint":
        opa_idx, opa_source = load_case_opa_indices(condition_root, case_name)
        if np.any(opa_idx < 0) or np.any(opa_idx >= input_vertices.shape[0]):
            raise ValueError(
                f"OPA op_v_indices out of bounds for {case_name}: "
                f"min={int(opa_idx.min())}, max={int(opa_idx.max())}, verts={input_vertices.shape[0]}"
            )
        input_opa_points = input_vertices[torch.tensor(opa_idx, dtype=torch.long, device=device)]
        ostium_indices = order_ring_indices_by_points(opa_idx.astype(np.int64), input_opa_points)
        ostium_points = input_vertices[torch.tensor(ostium_indices, dtype=torch.long, device=device)]
        ostium_source_used = f"opa_checkpoint/op_v_indices_ordered ({opa_source})"
    elif args.ostium_source == "opening_debug":
        if cond_ring_idx is not None and cond_source.startswith("ghd_fitting_opening_debug"):
            valid_idx = cond_ring_idx[(cond_ring_idx >= 0) & (cond_ring_idx < input_vertices.shape[0])]
            if valid_idx.shape[0] >= 3:
                ostium_indices = valid_idx.astype(np.int64)
                ostium_points = input_vertices[torch.tensor(valid_idx, dtype=torch.long, device=device)]
                ostium_source_used = f"condition_idx_from_{cond_source}"
            elif opening_debug_ring is not None and opening_debug_ring.shape[0] > 0:
                ostium_points = opening_debug_ring
                ostium_source_used = f"opening_debug ({opening_debug_path.name})"
            else:
                ostium_points = order_ring_points(condition_ring_mapped)
                ostium_source_used = "condition_mapped (fallback)"
        elif opening_debug_ring is not None and opening_debug_ring.shape[0] > 0:
            ostium_points = opening_debug_ring
            ostium_source_used = f"opening_debug ({opening_debug_path.name})"
        elif alignment_opening_mesh is not None:
            opening_verts, opening_faces, opening_name = alignment_opening_mesh
            opening_verts = opening_verts.to(device)
            opening_faces = opening_faces.to(device)
            opening_verts_mapped = apply_similarity(
                opening_verts,
                render_scale,
                render_rotation,
                render_translation,
            )
            loop_idx = boundary_loop_from_faces(opening_verts_mapped.shape[0], opening_faces)
            if loop_idx is not None and len(loop_idx) >= 3:
                ostium_points = opening_verts_mapped[torch.tensor(loop_idx, dtype=torch.int64, device=device)]
            else:
                ostium_points = order_ring_points(opening_verts_mapped)
            ostium_source_used = f"alignment_opening_planes/{opening_name}"
        elif cond_ring_coords is not None:
            ostium_points = order_ring_points(condition_ring_mapped)
            ostium_source_used = f"condition_mapped_from_{cond_source}"
        else:
            ostium_points = order_ring_points(condition_ring_mapped)
            ostium_source_used = "condition_mapped (fallback)"
    elif args.ostium_source == "condition_mapped":
        ostium_points = order_ring_points(condition_ring_mapped)
        ostium_source_used = "condition_mapped"
    else:
        canonical_idx = dataset.get_canonical_opening_idx(device=device)
        ostium_indices = canonical_idx.detach().cpu().numpy().astype(np.int64)
        ostium_points = input_vertices[canonical_idx]
        ostium_source_used = "canonical_idx"

    recon_panels = []
    for epoch, generator in generators:
        with torch.no_grad():
            mu, _ = generator.encode(target_norm)
            recon_target = generator.decode(mu, condition_norm)
            recon_ghd = recon_target[:, :ghd_dim]
            recon_scale_norm = recon_target[:, ghd_dim:]
            recon_scale = dataset.denorm_scale(recon_scale_norm).to(device)
            recon_mesh = ghd_reconstruct.ghd_forward_as_Meshes(
                recon_ghd,
                denormalize_shape=False,
                mean=ghd_mean,
                std=ghd_std,
                scale=recon_scale,
            )
            recon_mesh_aligned = apply_similarity_mesh(
                recon_mesh, render_scale, render_rotation, render_translation
            )
        rmse = torch.sqrt(torch.mean((recon_mesh_aligned.verts_padded()[0] - input_vertices) ** 2)).item()
        recon_panels.append((epoch, float(rmse), mesh_to_numpy(recon_mesh_aligned)))

    input_mesh_np = mesh_to_numpy(input_mesh)
    if output_name_override is not None:
        output_name = output_name_override
    else:
        case_tag = case_name.replace("/", "__")
        epochs_tag = "_".join(str(ep) for ep, _ in generators)
        output_name = f"{case_tag}_vae_recon_{epochs_tag}.png"
    output_path = output_dir / output_name

    render_panels(
        output_path=output_path,
        case_name=case_name,
        input_title=f"Input ({warped_mesh_path.name})",
        ostium_indices=ostium_indices,
        ostium_points=ostium_points.detach().cpu().numpy(),
        input_mesh_np=input_mesh_np,
        recon_panels=recon_panels,
        dpi=args.dpi,
    )

    cond_to_denorm = nearest_distances(condition_ring_denorm, input_denorm_vertices)
    cond_to_input = nearest_distances(condition_ring_mapped, input_vertices)

    opening_diag = None
    warning = False
    if opening_debug_points is not None and opening_debug_points.shape[0] > 0:
        open_to_input = nearest_distances(opening_debug_points, input_vertices)
        cond_to_open = nearest_distances(condition_ring_mapped, opening_debug_points)
        centroid_shift = torch.norm(
            condition_ring_mapped.mean(dim=0) - opening_debug_points.mean(dim=0)
        ).item()
        opening_diag = {
            "opening_to_input_mean": float(open_to_input.mean()),
            "opening_to_input_max": float(open_to_input.max()),
            "mapped_condition_to_opening_mean": float(cond_to_open.mean()),
            "mapped_condition_to_opening_max": float(cond_to_open.max()),
            "centroid_shift": float(centroid_shift),
        }
        warning = float(cond_to_open.mean()) > 0.15
        if cond_source.startswith("ghd_fitting_opening_debug") and cond_ring_idx is not None and cond_ring_idx.shape[0] >= 3:
            warning = False

    return {
        "case": case_name,
        "output_path": str(output_path),
        "ostium_source_used": ostium_source_used,
        "condition_diag": {
            "to_denorm_mesh_mean": float(cond_to_denorm.mean()),
            "to_denorm_mesh_max": float(cond_to_denorm.max()),
            "mapped_to_input_mean": float(cond_to_input.mean()),
            "mapped_to_input_max": float(cond_to_input.max()),
        },
        "opening_diag": opening_diag,
        "warning": bool(warning),
        "epoch_rmse": {str(epoch): float(rmse) for epoch, rmse, _ in recon_panels},
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ghd_chk_root = args.ghd_chk_root.expanduser().resolve()
    canonical_root = args.canonical_root.expanduser().resolve()
    condition_root = (
        args.condition_root.expanduser().resolve() if args.condition_root is not None else ghd_chk_root
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ghd_reconstruct = build_ghd_reconstruct(canonical_root, device)

    if int(args.prepare_condition_from_ghd) == 1:
        prep_summary = prepare_ghd_condition_opa_checkpoints(
            ghd_chk_root=ghd_chk_root,
            canonical_opa_chk=canonical_root / "opa_checkpoint.pkl",
            ghd_reconstruct=ghd_reconstruct,
            ghd_run="vanilla",
            ghd_chk_name="ghb_fitting_checkpoint.pkl",
            output_root=condition_root,
            force=bool(args.force_prepare_condition_from_ghd),
            condition_filename="opa_checkpoint.pkl",
            device=device,
        )
        if prep_summary["failed"]:
            raise RuntimeError(
                f"Failed preparing condition checkpoints for {len(prep_summary['failed'])} cases. "
                f"First failures: {prep_summary['failed'][:5]}"
            )

    cases = collect_available_cases(
        ghd_chk_root=ghd_chk_root,
        condition_root=condition_root,
        ghd_run="vanilla",
        ghd_chk_name="ghb_fitting_checkpoint.pkl",
        condition_filename="opa_checkpoint.pkl",
    )
    if not cases:
        raise RuntimeError(f"No valid cases found under {ghd_chk_root}")

    dataset = OstiumGHDDataset(
        str(ghd_chk_root),
        str(condition_root),
        str(canonical_root / "opa_checkpoint.pkl"),
        cases,
        ghd_run="vanilla",
        ghd_chk_name="ghb_fitting_checkpoint.pkl",
        withscale=True,
        normalize=True,
        ring_points=args.ring_points,
    )
    maybe_apply_training_stats(
        dataset=dataset,
        model_dir=args.model_dir.expanduser().resolve(),
        available_cases=cases,
        args=args,
        checkpoints_root=args.checkpoints_root.expanduser().resolve(),
        ghd_chk_root=ghd_chk_root,
    )
    case_to_idx = {name: i for i, name in enumerate(dataset.updated_cases)}
    selected_cases = select_cases(dataset.updated_cases, args)
    if not selected_cases:
        raise RuntimeError("No cases selected. Check --splits/--all-cases/--case settings.")

    generators = load_generators(args.epochs, args.model_dir.expanduser().resolve(), dataset, device)
    ghd_mean, ghd_std = dataset.get_mean_std()
    ghd_mean = ghd_mean.to(device)
    ghd_std = ghd_std.to(device)
    ghd_dim = dataset.get_ghd_dim()

    if len(selected_cases) > 1 and args.output_name is not None:
        print("Ignoring --output-name because multiple cases are selected.")

    successes: list[dict] = []
    failures: list[dict] = []
    for index, case_name in enumerate(selected_cases, start=1):
        try:
            info = process_case(
                case_name=case_name,
                args=args,
                device=device,
                ghd_chk_root=ghd_chk_root,
                dataset=dataset,
                case_to_idx=case_to_idx,
                ghd_reconstruct=ghd_reconstruct,
                generators=generators,
                ghd_mean=ghd_mean,
                ghd_std=ghd_std,
                ghd_dim=ghd_dim,
                condition_root=condition_root,
                alignment_root=canonical_root.parent,
                output_dir=output_dir,
                output_name_override=args.output_name if len(selected_cases) == 1 else None,
            )
            successes.append(info)
            output_rel = Path(info["output_path"]).resolve().relative_to(output_dir)
            last_epoch = str(generators[-1][0])
            rmse_last = info["epoch_rmse"][last_epoch]
            warn_tag = " [warning]" if info["warning"] else ""
            print(
                f"[{index:04d}/{len(selected_cases):04d}] wrote {output_rel} | "
                f"ostium={info['ostium_source_used']} | rmse@{last_epoch}={rmse_last:.6f}{warn_tag}"
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"case": case_name, "error": repr(exc)})
            print(f"[{index:04d}/{len(selected_cases):04d}] failed {case_name}: {exc}")

    summary = {
        "num_selected": len(selected_cases),
        "num_success": len(successes),
        "num_failed": len(failures),
        "ostium_source": args.ostium_source,
        "epochs": [int(ep) for ep in args.epochs],
        "results": successes,
        "failures": failures,
    }
    summary_name = "summary_single_case.json" if len(selected_cases) == 1 else "summary_all_cases.json"
    summary_path = output_dir / summary_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Completed: {len(successes)}/{len(selected_cases)} cases succeeded")
    if failures:
        failures_txt = output_dir / "failures.txt"
        failures_txt.write_text(
            "\n".join([f"{item['case']}: {item['error']}" for item in failures]) + "\n",
            encoding="utf-8",
        )
        print(f"Failures written to: {failures_txt}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
