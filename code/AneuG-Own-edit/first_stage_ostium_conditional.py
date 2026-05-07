import argparse
from datetime import timedelta
import json
import os
import pickle
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
import wandb

from ghd.losses.mesh_loss import Rigid_Loss
from models.ghd_reconstruct import GHD_Reconstruct
from models.mesh_plugins import MeshPlugins, MeshRegulizer
from models.utils import load_models, save_models
from models.vae_datasets import OstiumGHDDataset
from models.vae_models import ConditionalGHDVAE, KL_divergence
from pytorch3d.io import load_obj
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.transforms import axis_angle_to_matrix
from utils.utils import safe_load_mesh


KNOWN_DATA_SPLITS = ("train", "val", "test")


def parse_args():
    project_root = Path(__file__).resolve().parent
    default_checkpoints_root = project_root / "checkpoint-v2"
    parser = argparse.ArgumentParser(description="Train Stage 1 ostium-conditional pouch VAE.")

    path_group = parser.add_argument_group("Paths")
    path_group.add_argument(
        "--checkpoints-root",
        type=Path,
        default=default_checkpoints_root,
        help="Root directory containing checkpoint folders. Default: <project_root>/checkpoint-v2",
    )
    path_group.add_argument(
        "--ghd-chk-root",
        type=Path,
        default=default_checkpoints_root / "ghd_fitting",
        help="Directory containing per-case GHD fitting outputs. Default: <checkpoints_root>/ghd_fitting",
    )
    path_group.add_argument(
        "--alignment-root",
        type=Path,
        default=default_checkpoints_root / "alignment",
        help="Directory containing per-case alignment outputs. Default: <checkpoints_root>/alignment",
    )
    path_group.add_argument(
        "--canonical-root",
        type=Path,
        default=default_checkpoints_root / "canonical_model",
        help="Directory containing canonical assets. Default: <checkpoints_root>/canonical_model",
    )
    path_group.add_argument(
        "--condition-root",
        type=Path,
        default=None,
        help=(
            "Root containing per-case condition opa_checkpoint.pkl. "
            "Default: <ghd-chk-root> (generated from fitted GHD outputs)."
        ),
    )
    path_group.add_argument(
        "--condition-source",
        type=str,
        choices=["condition", "alignment"],
        default="condition",
        help=(
            "Which ostium checkpoints to use as dataset conditioning input. "
            "'condition' uses --condition-root (matches inference conditioning by default), "
            "'alignment' uses --alignment-root."
        ),
    )
    path_group.add_argument(
        "--prepare-condition-from-ghd",
        type=int,
        default=1,
        help="1: generate missing per-case opa_checkpoint.pkl from ghd_fitting_output, 0: skip.",
    )
    path_group.add_argument(
        "--force-prepare-condition-from-ghd",
        type=int,
        default=0,
        help="1: overwrite existing generated condition checkpoints, 0: keep existing files.",
    )

    preview_group = parser.add_argument_group("Preview Inference")
    preview_group.add_argument("--run-checkpoint-inference", type=int, default=0, help="1: run preview inference after each saved checkpoint, 0: skip.")
    preview_group.add_argument("--preview-case", type=str, default="C0002", help="Alignment case used for checkpoint preview inference.")
    preview_group.add_argument("--preview-num-samples", type=int, default=1, help="Number of preview samples exported after each saved checkpoint.")
    preview_group.add_argument(
        "--preview-seed",
        type=int,
        default=None,
        help="Random seed for checkpoint preview inference. If omitted, sampling is non-deterministic.",
    )

    model_group = parser.add_argument_group("Model and Training")
    model_group.add_argument("--epochs", type=int, default=30000, help="Number of training epochs.")
    model_group.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
    model_group.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension for ConditionalGHDVAE.")
    model_group.add_argument("--latent-dim", type=int, default=108, help="Latent dimension for ConditionalGHDVAE.")
    model_group.add_argument("--cond-embed-dim", type=int, default=64, help="Condition embedding dimension.")
    model_group.add_argument(
        "--norm-type",
        type=str,
        default="batch",
        choices=["batch", "layer", "none"],
        help="Normalization used inside residual MLP blocks.",
    )
    model_group.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    model_group.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    model_group.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor when num-workers > 0.")
    model_group.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Run mode.")
    model_group.add_argument("--use-norm", type=int, default=1, help="1: include normal reconstruction loss, 0: disable.")
    model_group.add_argument("--use-reg", type=int, default=0, help="1: enable regularization losses, 0: disable.")
    model_group.add_argument("--withscale", type=int, default=1, help="1: train with scale prediction, 0: shape only.")
    model_group.add_argument("--mea", type=int, default=1, help="1: MEA regularization sampling, 0: standard Gaussian sampling.")
    model_group.add_argument("--overreg", type=int, default=0, help="1: over-regularization mode in mesh regularizer.")
    model_group.add_argument("--reload-epoch", type=int, default=None, help="Reload checkpoint epoch (None to start fresh).")
    model_group.add_argument("--huber-delta", type=float, default=2.0, help="Delta for scale Huber loss.")
    model_group.add_argument("--ring-points", type=int, default=20, help="Number of ostium ring points used as condition.")
    model_group.add_argument("--run-shapiro-test", type=int, default=0, help="1: run Shapiro-Wilk diagnostics for MEA energies (slower startup).")
    model_group.add_argument(
        "--stage1-objective",
        type=str,
        default="mesh_vae",
        choices=["mesh_vae", "legacy"],
        help="mesh_vae: direct warped-mesh reconstruction with KL warmup, legacy: previous mixed objective.",
    )
    model_group.add_argument(
        "--loss-profile",
        type=str,
        default="default",
        choices=["default", "overfit"],
        help="default: original mixed objective, overfit: reconstruction-heavy objective for memorization/debugging.",
    )
    model_group.add_argument(
        "--posterior-noise-scale",
        type=float,
        default=0.1,
        help="Scale for posterior sampling noise during reconstruction. Set to 0 for deterministic decode from mu.",
    )
    model_group.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm. Set <= 0 to disable.")
    model_group.add_argument("--target-clamp", type=float, default=8.0, help="Clamp decoded normalized target values to +/- this range.")
    model_group.add_argument("--scale-clamp", type=float, default=6.0, help="Clamp decoded normalized scale values to +/- this range.")

    loss_group = parser.add_argument_group("Loss Weights")
    loss_group.add_argument("--w-kl-max", type=float, default=0.00020, help="Maximum KL weight after warmup.")
    loss_group.add_argument("--kl-warmup-epochs", type=int, default=1000, help="KL warmup epochs.")
    loss_group.add_argument("--kl-free-bits", type=float, default=0.01, help="Free-bits floor per latent dimension.")
    loss_group.add_argument("--w-target", type=float, default=1.0, help="Weight for target-space GHD reconstruction loss.")
    loss_group.add_argument("--w-reg", type=float, default=0.0, help="Weight for laplacian + consistency (MEA distribution matching).")
    loss_group.add_argument("--w-rigid", type=float, default=0.0, help="Weight for rigid loss (MEA distribution matching).")
    loss_group.add_argument("--w-trumpet", type=float, default=0.0, help="Weight for trumpet loss (0 for pouch-only).")
    loss_group.add_argument("--w-smooth", type=float, default=0.0, help="Weight for DIRECT laplacian smoothing (0 recommended — use spectral reg instead).")
    loss_group.add_argument("--w-normal", type=float, default=0.0, help="Weight for DIRECT normal consistency (0 recommended — use spectral reg instead).")
    loss_group.add_argument("--w-vert", type=float, default=250.0, help="Weight for direct warped-mesh reconstruction loss.")
    loss_group.add_argument("--w-norm", type=float, default=0.0, help="Weight for vertex-normal reconstruction loss.")
    loss_group.add_argument("--w-consistency", type=float, default=0.0, help="Direct mesh normal-consistency loss")
    loss_group.add_argument("--w-spectral", type=float, default=0.0, help="Weight for spectral regularization (penalizes high-frequency GHD coefficients).")
    loss_group.add_argument("--w-cond", type=float, default=0.0, help="Weight for explicit ostium condition loss (pose-invariant ring Procrustes loss).")
    loss_group.add_argument("--w-scale", type=float, default=1.0, help="Weight for scale regression loss.")
    loss_group.add_argument("--reg-every", type=int, default=1, help="Compute regularization every N steps (reuse cached loss otherwise). Higher = faster.")

    split_group = parser.add_argument_group("Dataset Split")
    split_group.add_argument("--split-train-ratio", type=float, default=0.8, help="Train split ratio.")
    split_group.add_argument("--split-val-ratio", type=float, default=0.1, help="Validation split ratio.")
    split_group.add_argument("--split-test-ratio", type=float, default=0.1, help="Test split ratio.")
    split_group.add_argument("--split-seed", type=int, default=42, help="Random seed for case split creation.")
    split_group.add_argument(
        "--split-file",
        type=Path,
        default=default_checkpoints_root / "dataset_splits" / "ostium_conditional_split_seed42.json",
        help=(
            "Optional JSON file with train/val/test case lists. "
            "Ignored when --ghd-chk-root already contains train/val/test subfolders."
        ),
    )
    split_group.add_argument("--force-resplit", type=int, default=0, help="1: ignore existing split file and create a new split.")
    split_group.add_argument(
        "--train-subset-limit",
        type=int,
        default=None,
        help="Optional cap on the number of train cases after the split, useful for hard-overfit debugging.",
    )

    early_stop_group = parser.add_argument_group("Early Stopping")
    early_stop_group.add_argument("--early-stopping", type=int, default=0, help="1: enable early stopping on validation loss.")
    early_stop_group.add_argument("--early-stopping-patience", type=int, default=500, help="Validation checks without improvement before stopping.")
    early_stop_group.add_argument("--early-stopping-min-delta", type=float, default=0.0, help="Minimum validation loss improvement to reset patience.")
    early_stop_group.add_argument("--val-every", type=int, default=50, help="Run validation every N epochs.")

    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument("--log-wandb", type=int, default=1, help="1: enable Weights & Biases logging, 0: disable.")
    logging_group.add_argument("--meta", type=str, default="ostium_pouch_new_inf", help="Run name / checkpoint subfolder name.")
    logging_group.add_argument("--wandb-project", type=str, default="AneuG", help="Weights & Biases project name.")
    logging_group.add_argument("--log-every", type=int, default=10, help="Log metrics every N epochs.")
    logging_group.add_argument(
        "--log-gradients",
        type=int,
        default=1,
        help="1: log per-loss gradient norms/ratios to W&B, 0: disable.",
    )
    logging_group.add_argument(
        "--grad-log-every",
        type=int,
        default=25,
        help="Compute per-loss gradient probes every N train steps.",
    )
    logging_group.add_argument(
        "--grad-probe-max-params",
        type=int,
        default=8,
        help="Maximum number of decoder/fallback params used for gradient probes.",
    )

    return parser.parse_args()


def apply_loss_profile(args):
    if args.loss_profile != "overfit":
        return

    args.stage1_objective = "mesh_vae"
    args.w_kl_max = 0.0
    args.kl_warmup_epochs = 1
    args.kl_free_bits = 0.0
    args.w_target = 0.25
    args.w_reg = 0.0
    args.w_rigid = 0.0
    args.w_trumpet = 0.0
    args.w_smooth = 0.0
    args.w_normal = 0.0
    args.w_consistency = 0.0
    args.w_spectral = 0.0
    args.w_cond = 0.0
    args.w_scale = 5.0
    args.w_vert = 250.0
    args.w_norm = 0.0
    args.use_reg = 0
    args.posterior_noise_scale = 0.0
    args.norm_type = "none"


def _select_gradient_probe_params(module: nn.Module, max_params: int = 8):
    named_params = [(name, param) for name, param in module.named_parameters() if param.requires_grad]
    if not named_params:
        return [], []

    decoder_params = [(name, param) for name, param in named_params if "decoder" in name.lower()]
    selected_pool = decoder_params if decoder_params else named_params
    max_params = max(1, int(max_params))
    selected = selected_pool[-max_params:]
    selected_names = [name for name, _ in selected]
    selected_params = [param for _, param in selected]
    return selected_params, selected_names


def _term_grad_norm(term: torch.Tensor, params):
    if (not torch.is_tensor(term)) or (not term.requires_grad) or (len(params) == 0):
        return 0.0

    grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    sq_sum = None
    for grad in grads:
        if grad is None:
            continue
        grad_sq = grad.detach().float().pow(2).sum()
        sq_sum = grad_sq if sq_sum is None else (sq_sum + grad_sq)

    if sq_sum is None:
        return 0.0
    return float(torch.sqrt(sq_sum + 1e-12).item())


def _kl_divergence_with_free_bits(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> torch.Tensor:
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=float(free_bits))
    return kl_per_dim.sum(dim=1).mean()


def _apply_case_transform(
    verts: torch.Tensor,
    rotation_axis_angle: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    rotation = axis_angle_to_matrix(rotation_axis_angle)
    return verts @ rotation.transpose(-1, -2) + translation.unsqueeze(1)


def compute_fitting_norm_canonical(canonical_meshes_raw) -> float:
    v_raw = canonical_meshes_raw.verts_packed()
    return torch.max(torch.norm(v_raw, dim=-1)).item() * 1.10 * 2.50


def _tensor_is_finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def _module_nonfinite_grad_names(module: nn.Module, limit: int = 8):
    names = []
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            names.append(name)
            if len(names) >= limit:
                break
    return names


def _module_nonfinite_param_names(module: nn.Module, limit: int = 8):
    names = []
    for name, param in module.named_parameters():
        if not torch.isfinite(param).all():
            names.append(name)
            if len(names) >= limit:
                break
    return names


def ensure_canonical_diff_checkpoint(canonical_root: Path, canonical_meshes) -> Path:
    diff_chk_path = canonical_root / "diff_centreline_checkpoint.pkl"
    if diff_chk_path.exists():
        return diff_chk_path

    centroid_path = canonical_root / "07_other" / "centroid_ostium.npy"
    if not centroid_path.exists():
        raise FileNotFoundError(f"Missing canonical ostium centroid file: {centroid_path}")

    centroid = torch.tensor(np.asarray(np.load(centroid_path)).reshape(3), dtype=torch.float32)
    verts = canonical_meshes.verts_packed().detach().cpu()
    seed_idx = torch.argmin(torch.norm(verts - centroid.unsqueeze(0), dim=-1)).item()

    with open(diff_chk_path, "wb") as f:
        pickle.dump({"diff_cep_registration": [int(seed_idx)]}, f)
    print(f"Generated canonical diff checkpoint at {diff_chk_path}")
    return diff_chk_path


def _safe_unit_np(vec: np.ndarray, eps=1e-12) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _reconstruct_fitted_mesh_from_checkpoint(ghd_reconstruct: GHD_Reconstruct, ghd_chk: dict, device: torch.device):
    """
    Reconstruct the final fitted mesh exactly as in ghd_fitting:
    deformed canonical + affine (R,s,T), then denormalize by norm_canonical.
    """
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


def _find_latest_opening_debug_obj(case_dir: Path, ghd_run: str) -> Path | None:
    opening_debug_dir = case_dir / ghd_run / "viz" / "opening_debug"
    if not opening_debug_dir.exists():
        return None
    preferred = opening_debug_dir / "opening_debug_epoch_02999.obj"
    if preferred.exists():
        return preferred
    candidates = sorted(opening_debug_dir.glob("opening_debug_epoch_*.obj"))
    if not candidates:
        return None
    return candidates[-1]


def _find_latest_warped_obj(case_dir: Path, ghd_run: str) -> Path | None:
    viz_dir = case_dir / ghd_run / "viz"
    preferred = viz_dir / "warped_epoch_02999.obj"
    if preferred.exists():
        return preferred
    candidates = sorted(viz_dir.glob("warped_epoch_*.obj"))
    if not candidates:
        return None
    return candidates[-1]


def _extract_obj_object_mesh(obj_path: Path, object_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    current = None
    global_vertex_index = 0
    vertices = []
    global_to_local = {}
    faces = []

    for raw in obj_path.read_text(encoding="utf-8", errors="ignore").splitlines():
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
                    global_to_local[global_vertex_index] = len(vertices)
                    vertices.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            continue
        if current == object_name and line.startswith("f "):
            tokens = line.split()[1:]
            face_idx = []
            for token in tokens:
                raw_idx = token.split("/")[0]
                if not raw_idx:
                    continue
                idx = int(raw_idx)
                if idx < 0:
                    idx = global_vertex_index + 1 + idx
                if idx in global_to_local:
                    face_idx.append(global_to_local[idx])
            if len(face_idx) >= 3:
                root = face_idx[0]
                for j in range(1, len(face_idx) - 1):
                    faces.append([root, face_idx[j], face_idx[j + 1]])

    if not vertices:
        return None
    vertices_np = np.asarray(vertices, dtype=np.float32)
    faces_np = np.asarray(faces, dtype=np.int64) if faces else np.empty((0, 3), dtype=np.int64)
    return vertices_np, faces_np


def _extract_boundary_loop_from_faces(num_verts: int, faces: np.ndarray) -> np.ndarray | None:
    if faces.size == 0:
        return None

    edge_count = {}
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge_count[key] = edge_count.get(key, 0) + 1

    boundary_edges = [edge for edge, cnt in edge_count.items() if cnt == 1]
    if not boundary_edges:
        return None

    adjacency = {}
    for u, v in boundary_edges:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)

    visited = set()
    loops = []
    for start in sorted(adjacency.keys()):
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        prev = None
        cur = start
        guard = 0
        while guard < max(num_verts * 3, 16):
            neighbors = adjacency.get(cur, [])
            if not neighbors:
                break
            nxt = None
            for cand in neighbors:
                if cand != prev:
                    nxt = cand
                    break
            if nxt is None or nxt == start:
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
    return np.asarray(loops[0], dtype=np.int64)


def prepare_ghd_condition_opa_checkpoints(ghd_chk_root: Path,
                                           canonical_opa_chk: Path,
                                           ghd_reconstruct: GHD_Reconstruct,
                                          ghd_run="vanilla",
                                          ghd_chk_name="ghb_fitting_checkpoint.pkl",
                                          output_root: Path | None = None,
                                          force=False,
                                          condition_filename="opa_checkpoint.pkl",
                                          device=None):
    """
    Build per-case OPA checkpoints from fitted GHD outputs so condition and target are consistent.
    Prefer ostium geometry from ghd_fitting opening_debug output (opening_0_warped_mesh).
    Fallback to canonical opening indices if debug geometry is unavailable.
    Output layout: <output_root>/<case>/<condition_filename>
    """
    if output_root is None:
        output_root = ghd_chk_root
    output_root = Path(output_root)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(canonical_opa_chk, "rb") as f:
        canonical_opa = pickle.load(f)
    opening_idx = np.asarray(canonical_opa["op_v_indices"][0], dtype=np.int64)
    if opening_idx.ndim != 1 or opening_idx.size < 3:
        raise ValueError(f"Invalid canonical opening indices in {canonical_opa_chk}")

    op_rec_f = canonical_opa.get("op_rec_f", [None])[0]
    op_rec_f_map = canonical_opa.get("op_rec_f_map", [None])[0]
    op_rec_v_indices_map = canonical_opa.get("op_rec_v_indices_map", [opening_idx.tolist()])[0]

    created = 0
    skipped = 0
    failed = []

    for case, case_dir in iter_case_dirs(ghd_chk_root):
        ghd_checkpoint = case_dir / ghd_run / ghd_chk_name
        if not ghd_checkpoint.exists():
            continue
        out_case_dir = output_root / case
        out_case_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_case_dir / condition_filename
        if out_path.exists() and (not force):
            skipped += 1
            continue

        try:
            with open(ghd_checkpoint, "rb") as f:
                ghd_chk = pickle.load(f)

            mesh = _reconstruct_fitted_mesh_from_checkpoint(ghd_reconstruct, ghd_chk, device=device)
            verts = mesh.verts_padded()[0].detach().cpu().numpy().astype(np.float32)
            normals = mesh.verts_normals_padded()[0].detach().cpu().numpy().astype(np.float32)

            warped_obj_path = _find_latest_warped_obj(case_dir, ghd_run)
            if warped_obj_path is not None:
                warped_verts, warped_faces, _ = load_obj(str(warped_obj_path))
                verts_ref = warped_verts.detach().cpu().numpy().astype(np.float32)
                mesh_ref = mesh.update_padded(warped_verts.unsqueeze(0).to(mesh.device))
                normals_ref = mesh_ref.verts_normals_padded()[0].detach().cpu().numpy().astype(np.float32)
            else:
                verts_ref = verts
                normals_ref = normals

            debug_obj_path = _find_latest_opening_debug_obj(case_dir, ghd_run)
            debug_opening = None
            if debug_obj_path is not None:
                debug_opening = _extract_obj_object_mesh(debug_obj_path, "opening_0_warped_mesh")

            source = "ghd_fitting_output_reconstructed"
            if debug_opening is not None and debug_opening[0].shape[0] >= 3:
                debug_vertices, debug_faces = debug_opening
                loop_idx = _extract_boundary_loop_from_faces(debug_vertices.shape[0], debug_faces)
                if loop_idx is not None and loop_idx.size >= 3:
                    ring_points = debug_vertices[loop_idx]
                else:
                    ring_points = debug_vertices

                debug_opening_t = torch.from_numpy(ring_points)
                verts_t = torch.from_numpy(verts_ref)
                nearest_idx = torch.cdist(debug_opening_t.unsqueeze(0), verts_t.unsqueeze(0)).squeeze(0).argmin(dim=1)
                op_v_indices_np = nearest_idx.detach().cpu().numpy().astype(np.int64)
                if op_v_indices_np.size > 0:
                    cleaned = [int(op_v_indices_np[0])]
                    for val in op_v_indices_np[1:]:
                        val_i = int(val)
                        if val_i != cleaned[-1]:
                            cleaned.append(val_i)
                    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
                        cleaned = cleaned[:-1]
                    unique_ordered = []
                    seen = set()
                    for val in cleaned:
                        if val in seen:
                            continue
                        seen.add(val)
                        unique_ordered.append(val)
                    cleaned = unique_ordered
                    op_v_indices_np = np.asarray(cleaned, dtype=np.int64)
                if op_v_indices_np.size >= 3:
                    source = "ghd_fitting_opening_debug"
                else:
                    op_v_indices_np = opening_idx
            else:
                op_v_indices_np = opening_idx

            op_v_coords = verts_ref[op_v_indices_np]
            op_v_normal = normals_ref[op_v_indices_np]
            op_n_mean = _safe_unit_np(op_v_normal.mean(axis=0))

            if source == "ghd_fitting_opening_debug":
                op_rec_f_case = np.empty((0, 3), dtype=np.int64)
                op_rec_f_map_case = np.empty((0, 3), dtype=np.int64)
                op_rec_v_indices_map_case = op_v_indices_np.tolist()
            else:
                op_rec_f_case = np.asarray(op_rec_f, dtype=np.int64) if op_rec_f is not None else np.empty((0, 3), dtype=np.int64)
                op_rec_f_map_case = np.asarray(op_rec_f_map, dtype=np.int64) if op_rec_f_map is not None else np.empty((0, 3), dtype=np.int64)
                op_rec_v_indices_map_case = list(op_rec_v_indices_map)

            chk = {
                "label": case,
                "source": source,
                "op_v_indices": [op_v_indices_np.tolist()],
                "op_v_coords": [op_v_coords],
                "op_v_normal": [op_v_normal],
                "op_n_mean": [op_n_mean],
                "op_rec_v": [op_v_coords.copy()],
                "op_rec_f": [op_rec_f_case],
                "op_rec_f_map": [op_rec_f_map_case],
                "op_rec_v_indices_map": [op_rec_v_indices_map_case],
            }
            with open(out_path, "wb") as f:
                pickle.dump(chk, f)
            created += 1
        except Exception as e:
            failed.append((case, repr(e)))

    return {
        "created": int(created),
        "skipped": int(skipped),
        "failed": failed,
        "output_root": str(output_root),
        "condition_filename": str(condition_filename),
    }


def extract_opening_condition(meshes, opening_idx):
    return meshes.verts_padded()[:, opening_idx, :].reshape(meshes.verts_padded().shape[0], -1)


def _reshape_ring(flat_ring):
    if flat_ring.dim() != 2 or (flat_ring.shape[1] % 3) != 0:
        raise ValueError(f"Expected ring tensor with shape [B, 3*K], got {tuple(flat_ring.shape)}")
    return flat_ring.reshape(flat_ring.shape[0], -1, 3)


def _ring_procrustes_mse(pred_ring_flat, target_ring_flat, eps=1e-8):
    """
    Pose-invariant opening-ring loss.
    Removes translation + isotropic scale and aligns with optimal rotation (Kabsch),
    then computes vertex-wise MSE on the aligned rings.
    """
    pred = _reshape_ring(pred_ring_flat)
    target = _reshape_ring(target_ring_flat)

    # Remove translation.
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)

    # Remove isotropic scale.
    pred_scale = torch.sqrt((pred ** 2).sum(dim=(1, 2), keepdim=True) / max(pred.shape[1], 1) + eps)
    target_scale = torch.sqrt((target ** 2).sum(dim=(1, 2), keepdim=True) / max(target.shape[1], 1) + eps)
    pred = pred / pred_scale
    target = target / target_scale

    # Optimal rotation: pred @ R ~= target
    cov = torch.matmul(pred.transpose(1, 2), target)  # [B, 3, 3]
    U, _, Vh = torch.linalg.svd(cov, full_matrices=False)
    R = torch.matmul(Vh.transpose(-2, -1), U.transpose(-2, -1))

    # Enforce proper rotation (det=+1), avoid reflections.
    det = torch.det(R)
    sign_fix = torch.ones((R.shape[0], 3), dtype=R.dtype, device=R.device)
    sign_fix[:, -1] = torch.where(det < 0, -1.0, 1.0)
    S = torch.diag_embed(sign_fix)
    R = torch.matmul(torch.matmul(Vh.transpose(-2, -1), S), U.transpose(-2, -1))

    pred_aligned = torch.matmul(pred, R)
    return F.mse_loss(pred_aligned, target)


def run_checkpoint_preview_inference(project_root: Path,
                                     checkpoint_path: Path,
                                     preview_case: str,
                                     output_dir: Path,
                                     num_samples: int,
                                     seed=None,
                                     ring_points=None,
                                     hidden_dim=None,
                                     latent_dim=None,
                                     cond_embed_dim=None,
                                     condition_root=None,
                                     checkpoints_root=None,
                                     ghd_chk_root=None,
                                     alignment_root=None,
                                     canonical_root=None):
    infer_script = project_root / "infer_stage1_ostium_conditional.py"
    command = [
        sys.executable,
        str(infer_script),
        "--checkpoint",
        str(checkpoint_path),
        "--case",
        preview_case,
        "--num-samples",
        str(num_samples),
        "--output-dir",
        str(output_dir),
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    if ring_points is not None:
        command.extend(["--ring-points", str(ring_points)])
    if hidden_dim is not None:
        command.extend(["--hidden-dim", str(hidden_dim)])
    if latent_dim is not None:
        command.extend(["--latent-dim", str(latent_dim)])
    if cond_embed_dim is not None:
        command.extend(["--cond-embed-dim", str(cond_embed_dim)])
    if condition_root is not None:
        command.extend(["--condition-root", str(condition_root)])
    if checkpoints_root is not None:
        command.extend(["--checkpoints-root", str(checkpoints_root)])
    if ghd_chk_root is not None:
        command.extend(["--ghd-chk-root", str(ghd_chk_root)])
    if alignment_root is not None:
        command.extend(["--alignment-root", str(alignment_root)])
    if canonical_root is not None:
        command.extend(["--canonical-root", str(canonical_root)])
    subprocess.run(command, cwd=project_root, check=True)


def detect_explicit_split_dirs(ghd_chk_root: Path):
    return [split for split in KNOWN_DATA_SPLITS if (ghd_chk_root / split).is_dir()]


def iter_case_dirs(ghd_chk_root: Path):
    split_dirs = detect_explicit_split_dirs(ghd_chk_root)
    if split_dirs:
        for split_name in split_dirs:
            split_root = ghd_chk_root / split_name
            for case_dir in sorted([p for p in split_root.iterdir() if p.is_dir()]):
                yield f"{split_name}/{case_dir.name}", case_dir
    else:
        for case_dir in sorted([p for p in ghd_chk_root.iterdir() if p.is_dir()]):
            yield case_dir.name, case_dir


def resolve_case_identifier(requested_case: str, available_cases):
    if requested_case in available_cases:
        return requested_case

    suffix = f"/{requested_case}"
    suffix_matches = [case for case in available_cases if case.endswith(suffix)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        raise ValueError(
            f"Case name '{requested_case}' is ambiguous across split folders. "
            f"Use full case id like train/<case>. Matches: {suffix_matches[:10]}"
        )

    raise ValueError(
        f"Case '{requested_case}' not found. "
        f"First available: {available_cases[:10]}"
    )


def collect_available_cases(ghd_chk_root: Path,
                            condition_root: Path,
                            ghd_run: str,
                            ghd_chk_name: str,
                            condition_filename="opa_checkpoint.pkl"):
    cases = []
    for case, case_dir in iter_case_dirs(ghd_chk_root):
        ghd_checkpoint = case_dir / ghd_run / ghd_chk_name
        opa_checkpoint = condition_root / case / condition_filename
        if ghd_checkpoint.exists() and opa_checkpoint.exists():
            cases.append(case)
    return cases


def split_train_cases_for_validation(train_cases, val_ratio: float, seed: int):
    train_cases = list(train_cases)
    if len(train_cases) == 0:
        return [], []
    if val_ratio <= 0 or len(train_cases) < 2:
        return train_cases, []

    rng = np.random.default_rng(seed)
    shuffled = np.array(train_cases)[rng.permutation(len(train_cases))].tolist()
    n_val = int(len(shuffled) * val_ratio)
    if n_val <= 0:
        n_val = 1
    n_val = min(n_val, len(shuffled) - 1)
    val_cases = shuffled[:n_val]
    train_only_cases = shuffled[n_val:]
    return train_only_cases, val_cases


def load_split_from_folders(ghd_chk_root: Path,
                            available_cases,
                            split_val_ratio: float,
                            split_seed: int):
    split_dirs = detect_explicit_split_dirs(ghd_chk_root)
    if not split_dirs:
        return None

    by_split = {split_name: [] for split_name in split_dirs}
    for case in available_cases:
        split_name, _, _ = case.partition("/")
        if split_name in by_split:
            by_split[split_name].append(case)

    train_pool = by_split.get("train", [])
    if len(train_pool) == 0:
        raise RuntimeError(
            f"Detected split folders {split_dirs} under {ghd_chk_root}, but no valid train cases were found."
        )

    if ("val" in by_split) and (len(by_split["val"]) > 0):
        train_cases = train_pool
        val_cases = by_split["val"]
    else:
        train_cases, val_cases = split_train_cases_for_validation(
            train_pool,
            val_ratio=split_val_ratio,
            seed=split_seed,
        )

    split_cases = {
        "train": train_cases,
        "val": val_cases,
        "test": by_split.get("test", []),
    }
    return split_cases


def create_case_split(cases, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.6f}")
    if len(cases) < 3:
        raise ValueError(f"Need at least 3 cases for train/val/test split, got {len(cases)}")

    rng = np.random.default_rng(seed)
    shuffled = np.array(cases)[rng.permutation(len(cases))].tolist()

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"Invalid split sizes for {n_total} cases: train={n_train}, val={n_val}, test={n_test}. "
            "Adjust ratios or provide more data."
        )

    train_cases = shuffled[:n_train]
    val_cases = shuffled[n_train:n_train + n_val]
    test_cases = shuffled[n_train + n_val:]
    return {"train": train_cases, "val": val_cases, "test": test_cases}


def load_or_create_split(split_file: Path, available_cases, args):
    if split_file.exists() and not bool(args.force_resplit):
        with open(split_file, "r", encoding="utf-8") as f:
            split = json.load(f)
    else:
        split = create_case_split(
            available_cases,
            train_ratio=args.split_train_ratio,
            val_ratio=args.split_val_ratio,
            test_ratio=args.split_test_ratio,
            seed=args.split_seed,
        )
        split_file.parent.mkdir(parents=True, exist_ok=True)
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "train": split["train"],
                    "val": split["val"],
                    "test": split["test"],
                    "meta": {
                        "split_seed": int(args.split_seed),
                        "train_ratio": float(args.split_train_ratio),
                        "val_ratio": float(args.split_val_ratio),
                        "test_ratio": float(args.split_test_ratio),
                    },
                },
                f,
                indent=2,
            )

    split = {
        "train": split["train"],
        "val": split["val"],
        "test": split["test"],
    }
    available_set = set(available_cases)
    all_split_cases = split["train"] + split["val"] + split["test"]
    missing = [case for case in all_split_cases if case not in available_set]
    if missing:
        raise ValueError(
            f"Split file contains {len(missing)} cases not available in current data. "
            f"Use --force-resplit 1 to regenerate. First missing: {missing[:5]}"
        )
    return split


def init_distributed_training():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return {
            "enabled": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "backend": None,
        }

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA GPUs.")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Some workstation drivers hang during DDP init with NCCL P2P/IPC enabled.
    # Keep user overrides if they explicitly set these env vars.
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=20),
    )
    return {
        "enabled": True,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "backend": "nccl",
    }


def distributed_barrier(distributed: bool, local_rank: int, backend: str | None):
    if not distributed:
        return
    if backend == "nccl":
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def _format_seconds(total_seconds: float) -> str:
    total_seconds = max(0.0, float(total_seconds))
    secs = int(total_seconds + 0.5)
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


if __name__ == "__main__":
    args = parse_args()
    apply_loss_profile(args)
    dist_info = init_distributed_training()
    distributed = dist_info["enabled"]
    rank = dist_info["rank"]
    local_rank = dist_info["local_rank"]
    world_size = dist_info["world_size"]
    dist_backend = dist_info["backend"]
    is_main_process = rank == 0

    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if is_main_process:
        print(
            (
                f"Initialized training process: distributed={distributed}, world_size={world_size}, device={device}, "
                f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE')}, "
                f"NCCL_IB_DISABLE={os.environ.get('NCCL_IB_DISABLE')}"
            ),
            flush=True,
        )
        if args.loss_profile != "default":
            print(
                f"Active loss profile: {args.loss_profile} | "
                f"posterior_noise_scale={args.posterior_noise_scale}",
                flush=True,
            )
        print(f"Stage-1 objective: {args.stage1_objective}", flush=True)

    torch.backends.cudnn.benchmark = True
    project_root = Path(__file__).resolve().parent

    # data conf
    checkpoints_root = args.checkpoints_root.expanduser()
    ghd_chk_root = args.ghd_chk_root.expanduser()
    canonical_root = args.canonical_root.expanduser()
    condition_root = args.condition_root.expanduser() if args.condition_root is not None else ghd_chk_root
    ghd_run = "vanilla"
    ghd_chk_name = "ghb_fitting_checkpoint.pkl"
    condition_filename = "opa_checkpoint.pkl"
    canonical_mesh_path = canonical_root / "part_aligned.obj"
    canonical_opa_chk = canonical_root / "opa_checkpoint.pkl"
    eigen_chk = canonical_root / "canonical_model_144_normed.pkl"

    canonical_meshes_raw = safe_load_mesh(str(canonical_mesh_path))
    canonical_diff_chk = canonical_root / "diff_centreline_checkpoint.pkl"
    if is_main_process:
        canonical_diff_chk = ensure_canonical_diff_checkpoint(canonical_root, canonical_meshes_raw)
    distributed_barrier(distributed, local_rank, dist_backend)

    # Build the NORMALIZED canonical mesh (matching GHD fitting's normalization)
    from pytorch3d.structures import Meshes as P3dMeshes
    v_raw = canonical_meshes_raw.verts_packed()
    norm_canonical_val = compute_fitting_norm_canonical(canonical_meshes_raw)
    v_normed = v_raw / norm_canonical_val
    canonical_meshes = P3dMeshes(verts=[v_normed], faces=canonical_meshes_raw.faces_list())

    # Build reconstructor early so we can generate condition OPA checkpoints from fitted GHD meshes.
    ghd_reconstruct = GHD_Reconstruct(
        canonical_meshes, str(eigen_chk), num_Basis=12**2, device=device,
        skip_normalize=True, norm_canonical_override=norm_canonical_val,
    )

    if int(args.prepare_condition_from_ghd) == 1:
        prep_failed = 0
        if is_main_process:
            prep_summary = prepare_ghd_condition_opa_checkpoints(
                ghd_chk_root=ghd_chk_root,
                canonical_opa_chk=canonical_opa_chk,
                ghd_reconstruct=ghd_reconstruct,
                ghd_run=ghd_run,
                ghd_chk_name=ghd_chk_name,
                output_root=condition_root,
                force=bool(args.force_prepare_condition_from_ghd),
                condition_filename=condition_filename,
                device=device,
            )
            print(
                "Condition OPA preparation: "
                f"created={prep_summary['created']}, skipped={prep_summary['skipped']}, "
                f"failed={len(prep_summary['failed'])}, root={prep_summary['output_root']}"
            )
            prep_failed = len(prep_summary["failed"])
            if prep_summary["failed"]:
                first_fail = prep_summary["failed"][:5]
                print(
                    f"[error] Failed to create condition checkpoints for {prep_failed} cases. "
                    f"First failures: {first_fail}"
                )

        if distributed:
            prep_failed_t = torch.tensor(float(prep_failed), device=device)
            dist.broadcast(prep_failed_t, src=0)
            prep_failed = int(prep_failed_t.item())
            distributed_barrier(distributed, local_rank, dist_backend)
        if prep_failed > 0:
            raise RuntimeError("Condition checkpoint preparation failed. See rank-0 logs for details.")

    available_cases = collect_available_cases(
        ghd_chk_root, condition_root, ghd_run, ghd_chk_name, condition_filename=condition_filename
    )
    split_cases = load_split_from_folders(
        ghd_chk_root=ghd_chk_root,
        available_cases=available_cases,
        split_val_ratio=float(args.split_val_ratio),
        split_seed=int(args.split_seed),
    )
    if split_cases is None:
        if len(available_cases) < 3:
            raise RuntimeError(
                f"Not enough valid cases for split with condition root {condition_root}. "
                f"Found {len(available_cases)}."
            )

        if args.split_file is None:
            split_file = checkpoints_root / "dataset_splits" / f"ostium_conditional_split_seed{args.split_seed}.json"
        else:
            split_file = args.split_file.expanduser()
        split_cases = load_or_create_split(split_file, available_cases, args)
        if is_main_process:
            print(
                f"Using split file: {split_file} | "
                f"train={len(split_cases['train'])}, val={len(split_cases['val'])}, test={len(split_cases['test'])}"
            )
    else:
        split_dirs = detect_explicit_split_dirs(ghd_chk_root)
        if is_main_process:
            print(
                f"Detected folder-based split ({split_dirs}) in {ghd_chk_root} | "
                f"train={len(split_cases['train'])}, val={len(split_cases['val'])}, test={len(split_cases['test'])}"
            )

    train_subset_limit = args.train_subset_limit
    if train_subset_limit is not None:
        train_subset_limit = int(train_subset_limit)
        if train_subset_limit <= 0:
            raise ValueError(f"train_subset_limit must be positive, got {train_subset_limit}")
        original_train_count = len(split_cases["train"])
        split_cases = {
            "train": split_cases["train"][:train_subset_limit],
            "val": split_cases["val"],
            "test": split_cases["test"],
        }
        if is_main_process:
            print(
                f"Applying train subset limit: {len(split_cases['train'])}/{original_train_count} train cases kept",
                flush=True,
            )

    cases = split_cases["train"] + split_cases["val"] + split_cases["test"]
    if len(split_cases["train"]) == 0:
        raise RuntimeError("Train split is empty after applying available-case filtering.")

    preview_case = args.preview_case
    if int(args.run_checkpoint_inference) == 1:
        preview_case = resolve_case_identifier(args.preview_case, cases)

    # mesh plugins / regularizer
    mesh_plugin = MeshPlugins(
        canonical_meshes,
        str(canonical_diff_chk),
        max_loops=[15],
        loop_start=[5],
        loop_range=4,
        trimmed_mesh_path=None,
        neck_chk_path=None,
    )
    rigidloss = Rigid_Loss(canonical_meshes.to(device))
    mesh_regulizer = MeshRegulizer(
        mesh_plugin=mesh_plugin,
        loop_start=[5],
        loop_range=4,
        device=device,
        rigidloss=rigidloss,
        run_shapiro_test=bool(args.run_shapiro_test),
    )

    # model conf
    epochs = args.epochs
    hidden_dim = args.hidden_dim
    latent_dim = args.latent_dim
    cond_embed_dim = args.cond_embed_dim
    global_batch_size = int(args.batch_size)
    if distributed:
        if global_batch_size < world_size:
            raise ValueError(
                f"Global batch size ({global_batch_size}) must be >= world size ({world_size})."
            )
        if (global_batch_size % world_size) != 0:
            raise ValueError(
                f"Global batch size ({global_batch_size}) must be divisible by world size ({world_size}) "
                "to preserve training dynamics."
            )
        batch_size = global_batch_size // world_size
    else:
        batch_size = global_batch_size
    num_workers = int(args.num_workers)
    if num_workers < 0:
        cpu_count = os.cpu_count() or 8
        num_workers = max(0, min(8, cpu_count // max(1, world_size) // 2))
    prefetch_factor = max(2, int(args.prefetch_factor))
    mode = args.mode
    use_norm = bool(args.use_norm)
    use_reg = bool(args.use_reg)
    withscale = bool(args.withscale)
    MEA = bool(args.mea)
    overreg = bool(args.overreg)
    norm_type = str(args.norm_type)
    if batch_size <= 1 and norm_type == "batch":
        if is_main_process:
            print("Per-device batch size is 1; switching norm_type from batch to layer.", flush=True)
        norm_type = "layer"
    reload_epoch = args.reload_epoch
    huber_loss = nn.HuberLoss(delta=args.huber_delta)
    stage1_objective = str(args.stage1_objective)
    learning_rate = float(args.lr)
    max_grad_norm = float(args.max_grad_norm)
    target_clamp = float(args.target_clamp)
    scale_clamp = float(args.scale_clamp)

    # loss weights — match unconditional VAE that works, only difference: no trumpet (pouch has no vessels)
    w_kl_max = args.w_kl_max
    w_kl = 0.0              # will be warmed up to w_kl_max
    kl_warmup_epochs = args.kl_warmup_epochs  # linearly ramp KL from 0 to w_kl_max
    kl_free_bits = float(args.kl_free_bits)
    w_target = args.w_target
    w_reg = args.w_reg              # laplacian + consistency MEA (distribution matching)
    w_rigid = args.w_rigid          # rigid MEA (distribution matching)
    w_trumpet = args.w_trumpet      # trumpet loss (0 for pouch, unconditional uses 2000)
    w_smooth = args.w_smooth        # DIRECT laplacian smoothing (0 recommended, use spectral reg)
    w_normal = args.w_normal        # DIRECT normal consistency (0 recommended, use spectral reg)
    w_vert = args.w_vert            # direct vertex-position reconstruction
    w_norm = args.w_norm            # direct normal reconstruction
    w_consistency = args.w_consistency  # direct mesh normal consistency like ghd_fitting
    w_spectral = args.w_spectral    # spectral regularization — penalizes high-frequency GHD modes
    w_cond = args.w_cond
    w_scale = args.w_scale
    posterior_noise_scale = float(args.posterior_noise_scale)

    # wandb conf
    log_wandb = bool(args.log_wandb)
    if distributed and log_wandb and (not is_main_process):
        log_wandb = False
    log_gradients = bool(args.log_gradients)
    grad_log_every = max(1, int(args.grad_log_every))
    grad_probe_max_params = max(1, int(args.grad_probe_max_params))
    meta = args.meta
    log_every = max(1, int(args.log_every))
    log_path = checkpoints_root / "first_stage_ostium_conditional" / meta
    os.makedirs(log_path, exist_ok=True)
    if log_wandb:
        wandb_config = {}
        for key, value in vars(args).items():
            if isinstance(value, Path):
                wandb_config[key] = str(value)
            else:
                wandb_config[key] = value
        wandb_config.update(
            {
                "latent_dim": latent_dim,
                "batch_size": global_batch_size,
                "per_device_batch_size": batch_size,
                "world_size": world_size,
                "cond_dim": "ostium_ring",
                "cond_embed_dim": cond_embed_dim,
                "stage1_objective": stage1_objective,
                "w_kl_max": w_kl_max,
                "kl_warmup_epochs": kl_warmup_epochs,
                "kl_free_bits": kl_free_bits,
                "w_target": w_target,
                "w_reg": w_reg,
                "w_rigid": w_rigid,
                "w_trumpet": w_trumpet,
                "w_smooth": w_smooth,
                "w_normal": w_normal,
                "w_vert": w_vert,
                "w_norm": w_norm,
                "w_consistency": w_consistency,
                "w_spectral": w_spectral,
                "w_cond": w_cond,
                "w_scale": w_scale,
                "MEA": MEA,
            }
        )
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key, relogin=False)
        wandb.init(
            project=args.wandb_project,
            name=meta,
            config=wandb_config,
            dir=str(log_path),
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    # dataset
    dataset_condition_root = condition_root if args.condition_source == "condition" else alignment_root
    if is_main_process:
        print(
            f"Dataset conditioning source: {args.condition_source} ({dataset_condition_root})",
            flush=True,
        )
    dataset = OstiumGHDDataset(
        str(ghd_chk_root),
        str(dataset_condition_root),
        str(canonical_opa_chk),
        cases,
        ghd_run=ghd_run,
        ghd_chk_name=ghd_chk_name,
        withscale=withscale,
        normalize=True,
        ring_points=args.ring_points,
    )

    case_to_index = {case: idx for idx, case in enumerate(dataset.updated_cases)}
    missing_after_dataset = [case for case in cases if case not in case_to_index]
    if missing_after_dataset:
        raise RuntimeError(
            f"{len(missing_after_dataset)} split cases are missing after dataset assembly. "
            f"First missing: {missing_after_dataset[:5]}"
        )
    train_indices = [case_to_index[case] for case in split_cases["train"]]
    val_indices = [case_to_index[case] for case in split_cases["val"]]
    test_indices = [case_to_index[case] for case in split_cases["test"]]

    # Normalize all splits using TRAIN statistics only.
    if withscale:
        train_targets = torch.stack(
            [torch.cat([dataset.ghd[idx], dataset.scale[idx]]) for idx in train_indices], dim=0
        )
    else:
        train_targets = torch.stack([dataset.ghd[idx] for idx in train_indices], dim=0)
    train_conditions = torch.stack([dataset.ostium_condition[idx] for idx in train_indices], dim=0)
    dataset.target_mean = train_targets.mean(dim=0, keepdim=True)
    dataset.target_std = train_targets.std(dim=0, keepdim=True, unbiased=False) + 0.01
    dataset.cond_mean = train_conditions.mean(dim=0, keepdim=True)
    dataset.cond_std = train_conditions.std(dim=0, keepdim=True, unbiased=False) + 0.01
    checkpoint_extra_state = {
        "target_mean": dataset.target_mean.detach().cpu(),
        "target_std": dataset.target_std.detach().cpu(),
        "cond_mean": dataset.cond_mean.detach().cpu(),
        "cond_std": dataset.cond_std.detach().cpu(),
        "train_cases": list(split_cases["train"]),
        "val_cases": list(split_cases["val"]),
        "test_cases": list(split_cases["test"]),
        "loss_profile": args.loss_profile,
        "stage1_objective": stage1_objective,
        "posterior_noise_scale": float(args.posterior_noise_scale),
        "kl_free_bits": kl_free_bits,
        "w_target": w_target,
        "train_subset_limit": args.train_subset_limit,
        "split_seed": int(args.split_seed),
        "split_val_ratio": float(args.split_val_ratio),
        "condition_source": args.condition_source,
        "dataset_condition_root": str(dataset_condition_root),
    }

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_subset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )

    common_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": (num_workers > 0),
    }
    if distributed and (num_workers > 0):
        common_loader_kwargs["multiprocessing_context"] = "spawn"
    if num_workers > 0:
        common_loader_kwargs["prefetch_factor"] = prefetch_factor

    train_drop_last = (train_sampler is None)
    if (train_sampler is None) and (len(train_subset) < batch_size):
        train_drop_last = False
        if is_main_process:
            print(
                f"Train subset ({len(train_subset)}) is smaller than batch size ({batch_size}); "
                "disabling drop_last to keep at least one train batch.",
                flush=True,
            )

    dataloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=train_drop_last,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common_loader_kwargs,
    )
    if is_main_process:
        print(
            f"Loaded dataset sizes -> train: {len(train_subset)}, val: {len(val_subset)}, test: {len(test_subset)} | "
            f"global_batch={global_batch_size}, per_device_batch={batch_size}, world_size={world_size}, "
            f"num_workers={num_workers}, prefetch_factor={prefetch_factor if num_workers > 0 else 'n/a'}"
        )
    if len(dataloader) == 0:
        raise RuntimeError(
            "Train DataLoader has zero batches. Reduce --batch-size or increase training cases "
            "(or remove a too-small --train-subset-limit)."
        )
    if is_main_process:
        print("Building model and DDP wrapper...", flush=True)

    mean, std = dataset.get_mean_std()
    ghd_dim = dataset.get_ghd_dim()
    opening_idx = dataset.get_canonical_opening_idx(device=device)
    num_bases = ghd_dim // 3  # 144

    # Spectral regularization weights: eigenvalues are low→high frequency
    # Higher eigenvalue = more oscillatory basis = stronger penalty
    eigvals = ghd_reconstruct.canonical_ghd.GBH_eigval.squeeze().to(device)  # [144]
    spectral_weights = (eigvals - eigvals[0]) / (eigvals[-1] - eigvals[0])   # normalize to [0, 1]
    spectral_weights = spectral_weights ** 2  # quadratic: mostly penalizes highest frequencies

    # model
    generator_module = ConditionalGHDVAE(
        dataset.get_target_dim(),
        hidden_dim,
        latent_dim,
        cond_dim=dataset.get_cond_dim(),
        cond_embed_dim=cond_embed_dim,
        norm_type=norm_type,
    ).to(device)

    grad_probe_params, grad_probe_names = ([], [])
    if log_gradients:
        grad_probe_params, grad_probe_names = _select_gradient_probe_params(
            generator_module, max_params=grad_probe_max_params
        )
        if is_main_process:
            if grad_probe_names:
                print(
                    f"Gradient probes enabled: {len(grad_probe_names)} params, every {grad_log_every} steps | "
                    f"sample: {grad_probe_names[0]}"
                )
            else:
                print("Gradient probes requested but no trainable parameters found; disabling gradient logging.")
                log_gradients = False

    optimizer_G = torch.optim.Adam(generator_module.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=2000, gamma=0.5)

    if reload_epoch is not None or mode == "eval":
        generator_module, optimizer_G, epoch_ = load_models(generator_module, optimizer_G, str(log_path), reload_epoch)
        for param_group in optimizer_G.param_groups:
            param_group["lr"] = learning_rate
        if is_main_process:
            print(f"Reloaded model from epoch: {reload_epoch} | overriding optimizer lr to {learning_rate}")
    else:
        epoch_ = 0

    if distributed:
        print(f"[rank {rank}] Wrapping model with DDP on device {device}...", flush=True)
        generator = DDP(
            generator_module,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        print(f"[rank {rank}] DDP model ready.", flush=True)
    else:
        generator = generator_module

    if is_main_process:
        print("Starting training loop...", flush=True)

    # Regularization step counter — expensive reg computed every N steps
    reg_every = args.reg_every
    global_step = 0
    use_mesh_vae_objective = (stage1_objective == "mesh_vae")
    # Enable expensive branches only if their corresponding loss has non-zero weight.
    need_mesh_reg = (not use_mesh_vae_objective) and ((w_reg > 0) or (w_rigid > 0) or (w_trumpet > 0))
    need_cond_loss = (w_cond > 0)
    need_cond_reg = need_cond_loss
    need_smooth_reg = (not use_mesh_vae_objective) and (w_smooth > 0)
    need_normal_reg = (not use_mesh_vae_objective) and (w_normal > 0)
    need_vert_recon = (w_vert > 0)
    need_norm_recon = (not use_mesh_vae_objective) and use_norm and (w_norm > 0)
    need_consistency_fit = (not use_mesh_vae_objective) and (w_consistency > 0)
    need_spectral_reg = (not use_mesh_vae_objective) and (w_spectral > 0)
    need_surface_recon = need_vert_recon or need_norm_recon
    need_recon_meshes_every_step = need_smooth_reg or need_normal_reg or need_consistency_fit or need_cond_loss
    need_fake_meshes = need_mesh_reg or need_cond_reg or need_smooth_reg or need_normal_reg
    need_real_meshes = need_mesh_reg
    need_fake_decode = need_fake_meshes or need_spectral_reg
    reg_active = use_reg and need_fake_decode

    # Cache reg losses (detached) for steps where reg is skipped
    _zero = torch.tensor(0.0, device=device)
    cached_reg = dict(
        loss_rigid=_zero, loss_lap=_zero, loss_consistency=_zero,
        trumpet_loss=_zero, loss_smooth_fake=_zero, loss_normal_fake=_zero,
        loss_cond_fake=_zero, loss_spectral_fake=_zero,
    )
    best_total_loss = float("inf")
    best_epoch = -1
    best_val_loss = float("inf")
    best_val_epoch = -1
    epochs_without_improvement = 0
    early_stopping = bool(args.early_stopping)
    early_stopping_patience = int(args.early_stopping_patience)
    early_stopping_min_delta = float(args.early_stopping_min_delta)
    val_every = max(1, int(args.val_every))
    best_ckpt_path = log_path / "best_model.pth"
    stopped_early = False
    last_epoch = epoch_
    train_wall_start = time.perf_counter()
    ema_epoch_time_sec = None
    eta_smoothing = 0.3

    def evaluate_loader(eval_loader, kl_weight):
        if len(eval_loader) == 0:
            return None
        generator.eval()
        total_loss_sum = 0.0
        mse_loss_sum = 0.0
        vert_loss_sum = 0.0
        norm_loss_sum = 0.0
        scale_loss_sum = 0.0
        kl_loss_sum = 0.0
        consistency_fit_sum = 0.0
        spectral_loss_sum = 0.0
        cond_loss_sum = 0.0
        batches = 0
        with torch.no_grad():
            for batch in eval_loader:
                target_eval = batch["target"].to(device, non_blocking=True)
                cond_eval = batch["condition"].to(device, non_blocking=True)
                ghd_eval = target_eval[:, :ghd_dim]
                scale_eval = target_eval[:, ghd_dim:] if withscale else None
                alignment_rotation_eval = batch["alignment_rotation"].to(device, non_blocking=True)
                alignment_translation_eval = batch["alignment_translation"].to(device, non_blocking=True)

                target_recon_eval, mu_eval, logvar_eval = generator(
                    target_eval, cond_eval, noise_scale=0.0
                )
                if not (_tensor_is_finite(target_recon_eval) and _tensor_is_finite(mu_eval) and _tensor_is_finite(logvar_eval)):
                    return {
                        "total_loss": float("nan"),
                        "mse_loss": float("nan"),
                    }
                target_recon_eval = torch.clamp(target_recon_eval, min=-target_clamp, max=target_clamp)
                ghd_recon_eval = target_recon_eval[:, :ghd_dim]
                scale_recon_eval = target_recon_eval[:, ghd_dim:] if withscale else None
                if withscale and scale_recon_eval is not None:
                    scale_recon_eval = torch.clamp(scale_recon_eval, min=-scale_clamp, max=scale_clamp)

                if need_cond_loss:
                    if withscale:
                        recon_scale_cond_eval = dataset.denorm_scale(scale_recon_eval).to(device)
                        recon_meshes_cond_eval = ghd_reconstruct.ghd_forward_as_Meshes(
                            ghd_recon_eval, mean=mean, std=std, scale=recon_scale_cond_eval
                        )
                    else:
                        recon_meshes_cond_eval = ghd_reconstruct.ghd_forward_as_Meshes(
                            ghd_recon_eval, mean=mean, std=std
                        )
                    cond_eval_pred = extract_opening_condition(recon_meshes_cond_eval, opening_idx)
                    cond_eval_target = dataset.denormalize_condition(cond_eval)
                    cond_eval_loss = _ring_procrustes_mse(cond_eval_pred, cond_eval_target)
                else:
                    cond_eval_loss = _zero

                if use_mesh_vae_objective and need_vert_recon:
                    if withscale:
                        target_scale_eval = dataset.denorm_scale(scale_eval).to(device)
                        target_mesh_eval = ghd_reconstruct.ghd_forward_as_Meshes(
                            ghd_eval, mean=mean, std=std, scale=target_scale_eval
                        )
                        target_vertices_eval = _apply_case_transform(
                            target_mesh_eval.verts_padded(),
                            alignment_rotation_eval,
                            alignment_translation_eval,
                        )
                        recon_scale_eval = dataset.denorm_scale(scale_recon_eval).to(device)
                        recon_mesh_eval = ghd_reconstruct.ghd_forward_as_Meshes(
                            ghd_recon_eval, mean=mean, std=std, scale=recon_scale_eval
                        )
                        warped_recon_eval = _apply_case_transform(
                            recon_mesh_eval.verts_padded(),
                            alignment_rotation_eval,
                            alignment_translation_eval,
                        )
                        vert_eval = F.mse_loss(warped_recon_eval, target_vertices_eval)
                    else:
                        target_mesh_eval = ghd_reconstruct.ghd_forward_as_Meshes(
                            ghd_eval, mean=mean, std=std
                        )
                        target_vertices_eval = _apply_case_transform(
                            target_mesh_eval.verts_padded(),
                            alignment_rotation_eval,
                            alignment_translation_eval,
                        )
                        recon_mesh_eval = ghd_reconstruct.ghd_forward_as_Meshes(
                            ghd_recon_eval, mean=mean, std=std
                        )
                        warped_recon_eval = _apply_case_transform(
                            recon_mesh_eval.verts_padded(),
                            alignment_rotation_eval,
                            alignment_translation_eval,
                        )
                        vert_eval = F.mse_loss(warped_recon_eval, target_vertices_eval)
                    norm_eval = _zero
                    mse_eval = F.mse_loss(ghd_eval, ghd_recon_eval)
                    kl_eval = _kl_divergence_with_free_bits(mu_eval, logvar_eval, kl_free_bits)
                    scale_eval_loss = huber_loss(scale_recon_eval, scale_eval) if withscale else _zero
                    consistency_eval = _zero
                    spectral_eval = _zero
                    total_eval = (
                        kl_weight * kl_eval
                        + w_target * mse_eval
                        + w_scale * scale_eval_loss
                        + w_vert * vert_eval
                        + w_cond * cond_eval_loss
                    )
                else:
                    if need_surface_recon:
                        data_real_eval = ghd_reconstruct.forward(ghd_eval, mean, std, need_norm_recon)
                        data_recon_eval = ghd_reconstruct.forward(ghd_recon_eval, mean, std, need_norm_recon)
                    else:
                        data_real_eval = None
                        data_recon_eval = None

                    if need_vert_recon:
                        if withscale:
                            real_scale_eval = dataset.denorm_scale(scale_eval).to(device)
                            recon_scale_eval = dataset.denorm_scale(scale_recon_eval).to(device)
                            real_mesh_eval = ghd_reconstruct.ghd_forward_as_Meshes(
                                ghd_eval, mean=mean, std=std, scale=real_scale_eval
                            )
                            recon_mesh_eval = ghd_reconstruct.ghd_forward_as_Meshes(
                                ghd_recon_eval, mean=mean, std=std, scale=recon_scale_eval
                            )
                            vert_eval = F.mse_loss(
                                recon_mesh_eval.verts_padded(), real_mesh_eval.verts_padded()
                            )
                        else:
                            vert_eval = F.mse_loss(data_recon_eval.pos, data_real_eval.pos)
                    else:
                        vert_eval = _zero
                    norm_eval = F.mse_loss(data_recon_eval.x[:, 3:], data_real_eval.x[:, 3:]) if need_norm_recon else _zero
                    mse_eval = F.mse_loss(ghd_eval, ghd_recon_eval)
                    kl_eval = KL_divergence(mu_eval, logvar_eval)
                    scale_eval_loss = huber_loss(scale_recon_eval, scale_eval) if withscale else _zero

                    if need_consistency_fit:
                        recon_meshes_eval = ghd_reconstruct.ghd_forward_as_Meshes(ghd_recon_eval, mean=mean, std=std)
                        consistency_eval = mesh_normal_consistency(recon_meshes_eval)
                    else:
                        consistency_eval = _zero

                    if need_spectral_reg:
                        coeffs_eval = ghd_recon_eval.reshape(-1, num_bases, 3)
                        energy_eval = (coeffs_eval ** 2).sum(dim=-1)
                        spectral_eval = (energy_eval * spectral_weights.unsqueeze(0)).mean()
                    else:
                        spectral_eval = _zero

                    total_eval = (
                        kl_weight * kl_eval
                        + mse_eval
                        + w_scale * scale_eval_loss
                        + w_vert * vert_eval
                        + w_norm * norm_eval
                        + w_consistency * consistency_eval
                        + w_spectral * spectral_eval
                        + w_cond * cond_eval_loss
                    )

                total_loss_sum += float(total_eval.item())
                mse_weight = w_target if use_mesh_vae_objective else 1.0
                mse_loss_sum += float((mse_weight * mse_eval).item())
                vert_loss_sum += float((w_vert * vert_eval).item())
                norm_loss_sum += float((w_norm * norm_eval).item())
                scale_loss_sum += float((w_scale * scale_eval_loss).item())
                kl_loss_sum += float((kl_weight * kl_eval).item())
                consistency_fit_sum += float((w_consistency * consistency_eval).item())
                spectral_loss_sum += float((w_spectral * spectral_eval).item())
                cond_loss_sum += float((w_cond * cond_eval_loss).item())
                batches += 1
        generator.train()

        inv_batches = 1.0 / max(1, batches)
        metrics = {
            "total_loss": total_loss_sum * inv_batches,
            "mse_loss": mse_loss_sum * inv_batches,
        }
        if w_vert > 0:
            metrics["vert_loss"] = vert_loss_sum * inv_batches
            if use_mesh_vae_objective:
                metrics["mesh_loss"] = vert_loss_sum * inv_batches
        if w_kl_max > 0:
            metrics["kl_loss"] = kl_loss_sum * inv_batches
        if withscale and (w_scale > 0):
            metrics["scale_loss"] = scale_loss_sum * inv_batches
        if need_norm_recon:
            metrics["norm_loss"] = norm_loss_sum * inv_batches
        if w_consistency > 0:
            metrics["loss_consistency_fit"] = consistency_fit_sum * inv_batches
        if w_spectral > 0:
            metrics["loss_spectral"] = spectral_loss_sum * inv_batches
        if w_cond > 0:
            metrics["loss_cond"] = cond_loss_sum * inv_batches
        return metrics

    for epoch in range(epoch_, epochs + 1):
        last_epoch = epoch
        if is_main_process and ((epoch == epoch_) or (epoch % log_every == 0)):
            print(f"Epoch {epoch} in progress...", flush=True)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        # KL warmup: linearly ramp from 0 to w_kl_max
        if epoch < kl_warmup_epochs:
            w_kl = w_kl_max * (epoch / kl_warmup_epochs)
        else:
            w_kl = w_kl_max
        epoch_start = time.perf_counter()
        epoch_batches = 0
        epoch_sums = {
            "total_loss": 0.0,
            "kl_loss_w": 0.0,
            "mse_loss_w": 0.0,
            "scale_loss_w": 0.0,
            "vert_loss_w": 0.0,
            "mesh_loss_w": 0.0,
            "norm_loss_w": 0.0,
            "loss_lap_mea_w": 0.0,
            "loss_cons_mea_w": 0.0,
            "loss_rigid_mea_w": 0.0,
            "loss_smooth_w": 0.0,
            "loss_normal_w": 0.0,
            "loss_consistency_fit_w": 0.0,
            "loss_spectral_w": 0.0,
            "loss_cond_w": 0.0,
            "loss_trumpet_w": 0.0,
            "kl_loss_raw": 0.0,
            "mse_loss_raw": 0.0,
            "scale_loss_raw": 0.0,
            "vert_loss_raw": 0.0,
            "mesh_loss_raw": 0.0,
            "norm_loss_raw": 0.0,
        }
        epoch_grad_sums = {
            "total": 0.0,
            "mse": 0.0,
            "kl": 0.0,
            "scale": 0.0,
            "vert": 0.0,
            "norm": 0.0,
            "reg_lap_mea": 0.0,
            "reg_cons_mea": 0.0,
            "reg_rigid_mea": 0.0,
            "trumpet": 0.0,
            "smooth": 0.0,
            "normal": 0.0,
            "consistency_fit": 0.0,
            "spectral": 0.0,
            "cond": 0.0,
        }
        epoch_grad_samples = 0
        nonfinite_steps = 0

        for batch in dataloader:
            target = batch["target"].to(device, non_blocking=True)
            cond = batch["condition"].to(device, non_blocking=True)
            alignment_rotation = batch["alignment_rotation"].to(device, non_blocking=True)
            alignment_translation = batch["alignment_translation"].to(device, non_blocking=True)

            ghd = target[:, :ghd_dim]
            scale = target[:, ghd_dim:] if withscale else None

            do_reg = reg_active and (global_step % reg_every == 0)

            optimizer_G.zero_grad(set_to_none=True)

            warmup_progress = 1.0
            if kl_warmup_epochs > 0:
                warmup_progress = min(1.0, float(epoch + 1) / float(kl_warmup_epochs))
            train_noise_scale = posterior_noise_scale * warmup_progress

            # --- Reconstruction path (every step) ---
            if need_surface_recon and (not use_mesh_vae_objective):
                data_real = ghd_reconstruct.forward(ghd, mean, std, need_norm_recon)
            else:
                data_real = None

            target_recon, mu, logvar = generator(target, cond, noise_scale=train_noise_scale)
            if not (_tensor_is_finite(target_recon) and _tensor_is_finite(mu) and _tensor_is_finite(logvar)):
                nonfinite_steps += 1
                optimizer_G.zero_grad(set_to_none=True)
                if is_main_process and (nonfinite_steps <= 5):
                    print(
                        f"[warn] Skipping step at epoch {epoch}, global_step={global_step}: "
                        "non-finite generator outputs detected.",
                        flush=True,
                    )
                global_step += 1
                continue
            target_recon = torch.clamp(target_recon, min=-target_clamp, max=target_clamp)
            ghd_recon = target_recon[:, :ghd_dim]
            scale_recon = target_recon[:, ghd_dim:] if withscale else None
            if withscale and scale_recon is not None:
                scale_recon = torch.clamp(scale_recon, min=-scale_clamp, max=scale_clamp)

            if need_surface_recon and (not use_mesh_vae_objective):
                data_recon = ghd_reconstruct.forward(ghd_recon, mean, std, need_norm_recon)
            else:
                data_recon = None

            if use_mesh_vae_objective and need_vert_recon:
                if withscale:
                    target_scale = dataset.denorm_scale(scale).to(device)
                    target_mesh_vert = ghd_reconstruct.ghd_forward_as_Meshes(
                        ghd, mean=mean, std=std, scale=target_scale
                    )
                    target_vertices = _apply_case_transform(
                        target_mesh_vert.verts_padded(),
                        alignment_rotation,
                        alignment_translation,
                    )
                    recon_scale = dataset.denorm_scale(scale_recon).to(device)
                    recon_mesh_vert = ghd_reconstruct.ghd_forward_as_Meshes(
                        ghd_recon, mean=mean, std=std, scale=recon_scale
                    )
                    warped_recon = _apply_case_transform(
                        recon_mesh_vert.verts_padded(),
                        alignment_rotation,
                        alignment_translation,
                    )
                    vert_loss = F.mse_loss(warped_recon, target_vertices)
                else:
                    target_mesh_vert = ghd_reconstruct.ghd_forward_as_Meshes(
                        ghd, mean=mean, std=std
                    )
                    target_vertices = _apply_case_transform(
                        target_mesh_vert.verts_padded(),
                        alignment_rotation,
                        alignment_translation,
                    )
                    recon_mesh_vert = ghd_reconstruct.ghd_forward_as_Meshes(
                        ghd_recon, mean=mean, std=std
                    )
                    warped_recon = _apply_case_transform(
                        recon_mesh_vert.verts_padded(),
                        alignment_rotation,
                        alignment_translation,
                    )
                    vert_loss = F.mse_loss(warped_recon, target_vertices)
            elif need_vert_recon:
                if withscale:
                    real_scale = dataset.denorm_scale(scale).to(device)
                    recon_scale = dataset.denorm_scale(scale_recon).to(device)
                    real_mesh_vert = ghd_reconstruct.ghd_forward_as_Meshes(
                        ghd, mean=mean, std=std, scale=real_scale
                    )
                    recon_mesh_vert = ghd_reconstruct.ghd_forward_as_Meshes(
                        ghd_recon, mean=mean, std=std, scale=recon_scale
                    )
                    vert_loss = F.mse_loss(
                        recon_mesh_vert.verts_padded(), real_mesh_vert.verts_padded()
                    )
                else:
                    vert_loss = F.mse_loss(data_recon.pos, data_real.pos)
            else:
                vert_loss = _zero
            norm_loss = F.mse_loss(data_recon.x[:, 3:], data_real.x[:, 3:]) if need_norm_recon else _zero
            mse_loss = F.mse_loss(ghd, ghd_recon)
            if use_mesh_vae_objective:
                kl_loss = _kl_divergence_with_free_bits(mu, logvar, kl_free_bits)
            else:
                kl_loss = KL_divergence(mu, logvar)
            scale_loss = huber_loss(scale_recon, scale) if withscale else _zero

            # Direct smoothness on RECONSTRUCTED meshes (every step — cheap, directly improves quality)
            recon_meshes = None
            if need_recon_meshes_every_step:
                recon_meshes = ghd_reconstruct.ghd_forward_as_Meshes(ghd_recon, mean=mean, std=std)
                loss_smooth_recon = mesh_laplacian_smoothing(recon_meshes, method="uniform") if need_smooth_reg else _zero
                normal_consistency_recon = mesh_normal_consistency(recon_meshes) if (need_normal_reg or need_consistency_fit) else _zero
                loss_normal_recon = normal_consistency_recon if need_normal_reg else _zero
                loss_consistency_fit_recon = normal_consistency_recon if need_consistency_fit else _zero
            else:
                loss_smooth_recon = _zero
                loss_normal_recon = _zero
                loss_consistency_fit_recon = _zero

            # Spectral regularization: penalize high-frequency GHD coefficients
            # Applied every step — cheap (pure tensor ops, no mesh needed)
            if need_spectral_reg:
                # Recon: encourage encoder to favor low-frequency modes
                coeffs_recon = ghd_recon.reshape(-1, num_bases, 3)
                energy_recon = (coeffs_recon ** 2).sum(dim=-1)  # [B, 144]
                loss_spectral_recon = (energy_recon * spectral_weights.unsqueeze(0)).mean()
            else:
                loss_spectral_recon = _zero

            if need_cond_loss:
                if withscale:
                    recon_scale_cond = dataset.denorm_scale(scale_recon).to(device)
                    recon_meshes_for_cond = ghd_reconstruct.ghd_forward_as_Meshes(
                        ghd_recon, mean=mean, std=std, scale=recon_scale_cond
                    )
                else:
                    recon_meshes_for_cond = ghd_reconstruct.ghd_forward_as_Meshes(
                        ghd_recon, mean=mean, std=std
                    )
                cond_recon = extract_opening_condition(recon_meshes_for_cond, opening_idx)
                cond_target = dataset.denormalize_condition(cond)
                loss_cond_recon = _ring_procrustes_mse(cond_recon, cond_target)
            else:
                loss_cond_recon = _zero

            # --- Regularization path (every reg_every steps — expensive) ---
            if do_reg:
                real_meshes = ghd_reconstruct.ghd_forward_as_Meshes(ghd, mean=mean, std=std) if need_real_meshes else None

                B = target.shape[0]
                ghd_fake = None
                fake_meshes = None
                cond_sample = None
                if need_fake_decode:
                    if not MEA:
                        z_reg = torch.randn(B, latent_dim, device=device)
                    else:
                        z_reg = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                    cond_sample = cond[torch.randperm(B, device=device)]
                    target_fake = generator(None, cond_sample, z=z_reg, decode_only=True)
                    ghd_fake = target_fake[:, :ghd_dim]

                    if need_fake_meshes:
                        fake_meshes = ghd_reconstruct.ghd_forward_as_Meshes(ghd_fake, mean=mean, std=std)

                if need_cond_reg:
                    cond_fake = extract_opening_condition(fake_meshes, opening_idx)
                    cond_sample_target = dataset.denormalize_condition(cond_sample)
                    loss_cond_fake = _ring_procrustes_mse(cond_fake, cond_sample_target)
                else:
                    loss_cond_fake = _zero

                if need_mesh_reg:
                    loss_rigid, loss_lap, loss_consistency, trumpet_loss = mesh_regulizer.KL_regulization(
                        fake_meshes, real_meshes, MEA=MEA, overreg=overreg,
                    )
                else:
                    loss_rigid, loss_lap, loss_consistency, trumpet_loss = _zero, _zero, _zero, _zero

                loss_smooth_fake = mesh_laplacian_smoothing(fake_meshes, method="uniform") if need_smooth_reg else _zero
                loss_normal_fake = mesh_normal_consistency(fake_meshes) if need_normal_reg else _zero

                # Spectral reg on generated coefficients
                if need_spectral_reg:
                    coeffs_fake = ghd_fake.reshape(-1, num_bases, 3)
                    energy_fake = (coeffs_fake ** 2).sum(dim=-1)
                    loss_spectral_fake = (energy_fake * spectral_weights.unsqueeze(0)).mean()
                else:
                    loss_spectral_fake = _zero

                # Cache detached values for non-reg steps
                cached_reg = dict(
                    loss_rigid=loss_rigid.detach(), loss_lap=loss_lap.detach(),
                    loss_consistency=loss_consistency.detach(), trumpet_loss=trumpet_loss.detach(),
                    loss_smooth_fake=loss_smooth_fake.detach(), loss_normal_fake=loss_normal_fake.detach(),
                    loss_cond_fake=loss_cond_fake.detach() if need_cond_reg else _zero,
                    loss_spectral_fake=loss_spectral_fake.detach(),
                )
            else:
                # Reuse cached values (no gradients — only reconstruction trains this step)
                loss_rigid = cached_reg["loss_rigid"]
                loss_lap = cached_reg["loss_lap"]
                loss_consistency = cached_reg["loss_consistency"]
                trumpet_loss = cached_reg["trumpet_loss"]
                loss_smooth_fake = cached_reg["loss_smooth_fake"]
                loss_normal_fake = cached_reg["loss_normal_fake"]
                loss_cond_fake = cached_reg["loss_cond_fake"]
                loss_spectral_fake = cached_reg["loss_spectral_fake"]

            loss_smooth = 0.5 * (loss_smooth_recon + loss_smooth_fake)
            loss_normal = 0.5 * (loss_normal_recon + loss_normal_fake)
            loss_cond = 0.5 * (loss_cond_recon + loss_cond_fake) if do_reg else loss_cond_recon
            loss_spectral = 0.5 * (loss_spectral_recon + loss_spectral_fake)

            if use_mesh_vae_objective:
                loss = (
                    w_kl * kl_loss
                    + w_target * mse_loss
                    + w_scale * scale_loss
                    + w_vert * vert_loss
                    + w_cond * loss_cond
                )
            else:
                loss = (
                    w_kl * kl_loss
                    + mse_loss
                    + w_scale * scale_loss
                    + w_vert * vert_loss
                    + w_norm * norm_loss
                    + w_reg * loss_lap
                    + w_reg * loss_consistency
                    + w_rigid * loss_rigid
                    + w_trumpet * trumpet_loss
                    + w_cond * loss_cond
                    + w_smooth * loss_smooth
                    + w_normal * loss_normal
                    + w_consistency * loss_consistency_fit_recon
                    + w_spectral * loss_spectral
                )

            if not _tensor_is_finite(loss):
                nonfinite_steps += 1
                optimizer_G.zero_grad(set_to_none=True)
                if is_main_process and (nonfinite_steps <= 5):
                    print(
                        f"[warn] Skipping step at epoch {epoch}, global_step={global_step}: "
                        "non-finite total loss detected.",
                        flush=True,
                    )
                global_step += 1
                continue

            if log_gradients and grad_probe_params and (global_step % grad_log_every == 0):
                if use_mesh_vae_objective:
                    grad_terms = {
                        "mse": w_target * mse_loss,
                        "kl": w_kl * kl_loss,
                        "scale": w_scale * scale_loss,
                        "vert": w_vert * vert_loss,
                        "norm": _zero,
                        "reg_lap_mea": _zero,
                        "reg_cons_mea": _zero,
                        "reg_rigid_mea": _zero,
                        "trumpet": _zero,
                        "smooth": _zero,
                        "normal": _zero,
                        "consistency_fit": _zero,
                        "spectral": _zero,
                        "cond": w_cond * loss_cond,
                    }
                else:
                    grad_terms = {
                        "mse": mse_loss,
                        "kl": w_kl * kl_loss,
                        "scale": w_scale * scale_loss,
                        "vert": w_vert * vert_loss,
                        "norm": w_norm * norm_loss,
                        "reg_lap_mea": w_reg * loss_lap,
                        "reg_cons_mea": w_reg * loss_consistency,
                        "reg_rigid_mea": w_rigid * loss_rigid,
                        "trumpet": w_trumpet * trumpet_loss,
                        "smooth": w_smooth * loss_smooth,
                        "normal": w_normal * loss_normal,
                        "consistency_fit": w_consistency * loss_consistency_fit_recon,
                        "spectral": w_spectral * loss_spectral,
                        "cond": w_cond * loss_cond,
                    }
                epoch_grad_sums["total"] += _term_grad_norm(loss, grad_probe_params)
                for key, term in grad_terms.items():
                    epoch_grad_sums[key] += _term_grad_norm(term, grad_probe_params)
                epoch_grad_samples += 1

            loss.backward()
            bad_grad_names = _module_nonfinite_grad_names(generator_module)
            if bad_grad_names:
                nonfinite_steps += 1
                optimizer_G.zero_grad(set_to_none=True)
                if is_main_process and (nonfinite_steps <= 5):
                    print(
                        f"[warn] Skipping optimizer step at epoch {epoch}, global_step={global_step}: "
                        f"non-finite gradients in {bad_grad_names}",
                        flush=True,
                    )
                global_step += 1
                continue
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=max_grad_norm)
            optimizer_G.step()
            bad_param_names = _module_nonfinite_param_names(generator_module)
            if bad_param_names:
                raise RuntimeError(
                    f"Non-finite model parameters detected after optimizer step at epoch {epoch}, "
                    f"global_step={global_step}. First bad params: {bad_param_names}"
                )
            global_step += 1
            epoch_batches += 1

            # Epoch-wise averages for cleaner and more stable W&B curves.
            epoch_sums["total_loss"] += float(loss.item())
            epoch_sums["kl_loss_w"] += float((w_kl * kl_loss).item())
            epoch_sums["mse_loss_w"] += float(((w_target if use_mesh_vae_objective else 1.0) * mse_loss).item())
            epoch_sums["scale_loss_w"] += float((w_scale * scale_loss).item())
            epoch_sums["vert_loss_w"] += float((w_vert * vert_loss).item())
            if use_mesh_vae_objective:
                epoch_sums["mesh_loss_w"] += float((w_vert * vert_loss).item())
            epoch_sums["norm_loss_w"] += float((w_norm * norm_loss).item())
            epoch_sums["loss_lap_mea_w"] += float((w_reg * loss_lap).item())
            epoch_sums["loss_cons_mea_w"] += float((w_reg * loss_consistency).item())
            epoch_sums["loss_rigid_mea_w"] += float((w_rigid * loss_rigid).item())
            epoch_sums["loss_smooth_w"] += float((w_smooth * loss_smooth).item())
            epoch_sums["loss_normal_w"] += float((w_normal * loss_normal).item())
            epoch_sums["loss_consistency_fit_w"] += float((w_consistency * loss_consistency_fit_recon).item())
            epoch_sums["loss_spectral_w"] += float((w_spectral * loss_spectral).item())
            epoch_sums["loss_cond_w"] += float((w_cond * loss_cond).item())
            epoch_sums["loss_trumpet_w"] += float((w_trumpet * trumpet_loss).item())
            epoch_sums["kl_loss_raw"] += float(kl_loss.item())
            epoch_sums["mse_loss_raw"] += float(mse_loss.item())
            epoch_sums["scale_loss_raw"] += float(scale_loss.item())
            if need_vert_recon:
                epoch_sums["vert_loss_raw"] += float(vert_loss.item())
                if use_mesh_vae_objective:
                    epoch_sums["mesh_loss_raw"] += float(vert_loss.item())
            if need_norm_recon:
                epoch_sums["norm_loss_raw"] += float(norm_loss.item())

        epoch_time_sec = time.perf_counter() - epoch_start
        if distributed:
            epoch_time_t = torch.tensor(epoch_time_sec, dtype=torch.float64, device=device)
            dist.all_reduce(epoch_time_t, op=dist.ReduceOp.MAX)
            epoch_time_sec = float(epoch_time_t.item())

            reduce_keys = list(epoch_sums.keys())
            packed = torch.tensor(
                [epoch_sums[k] for k in reduce_keys] + [float(epoch_batches)],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            for idx, key in enumerate(reduce_keys):
                epoch_sums[key] = float(packed[idx].item())
            epoch_batches = int(packed[-1].item())

            grad_reduce_keys = list(epoch_grad_sums.keys())
            grad_packed = torch.tensor(
                [epoch_grad_sums[k] for k in grad_reduce_keys] + [float(epoch_grad_samples)],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(grad_packed, op=dist.ReduceOp.SUM)
            for idx, key in enumerate(grad_reduce_keys):
                epoch_grad_sums[key] = float(grad_packed[idx].item())
            epoch_grad_samples = int(grad_packed[-1].item())

        run_validation = ((epoch > epoch_) and (epoch % val_every == 0)) or (epoch == epochs)
        if run_validation:
            distributed_barrier(distributed, local_rank, dist_backend)
        val_metrics = evaluate_loader(val_loader, w_kl) if (run_validation and is_main_process) else None
        if run_validation:
            distributed_barrier(distributed, local_rank, dist_backend)

        if (epoch_batches > 0) and is_main_process:
            inv_batches = 1.0 / float(epoch_batches)
            samples_per_sec = float(epoch_batches * global_batch_size) / max(epoch_time_sec, 1e-8)
            epochs_done = (epoch - epoch_ + 1)
            avg_epoch_time_sec = (time.perf_counter() - train_wall_start) / max(1, epochs_done)
            if ema_epoch_time_sec is None:
                ema_epoch_time_sec = epoch_time_sec
            else:
                ema_epoch_time_sec = (1.0 - eta_smoothing) * ema_epoch_time_sec + eta_smoothing * epoch_time_sec
            epochs_remaining = max(0, epochs - epoch)
            eta_remaining_sec = epochs_remaining * ema_epoch_time_sec
            log_dict = {
                "epoch": epoch,
                "mse_loss": epoch_sums["mse_loss_w"] * inv_batches,
                "total_loss": epoch_sums["total_loss"] * inv_batches,
                "mse_loss_raw": epoch_sums["mse_loss_raw"] * inv_batches,
                "train_noise_scale": train_noise_scale,
                "lr": scheduler_G.optimizer.param_groups[0]["lr"],
                "epoch_time_sec": epoch_time_sec,
                "avg_epoch_time_sec": avg_epoch_time_sec,
                "samples_per_sec": samples_per_sec,
                "global_step": global_step,
                "epochs_remaining": epochs_remaining,
                "eta_remaining_sec": eta_remaining_sec,
                "eta_remaining_h": eta_remaining_sec / 3600.0,
                "nonfinite_steps": nonfinite_steps,
            }

            if w_kl_max > 0:
                log_dict["w_kl"] = w_kl
                log_dict["kl_loss"] = epoch_sums["kl_loss_w"] * inv_batches
                log_dict["kl_loss_raw"] = epoch_sums["kl_loss_raw"] * inv_batches
            if withscale and (w_scale > 0):
                log_dict["scale_loss"] = epoch_sums["scale_loss_w"] * inv_batches
                log_dict["scale_loss_raw"] = epoch_sums["scale_loss_raw"] * inv_batches
            if w_vert > 0:
                log_dict["vert_loss"] = epoch_sums["vert_loss_w"] * inv_batches
                log_dict["vert_loss_raw"] = epoch_sums["vert_loss_raw"] * inv_batches
                if use_mesh_vae_objective:
                    log_dict["mesh_loss"] = epoch_sums["mesh_loss_w"] * inv_batches
                    log_dict["mesh_loss_raw"] = epoch_sums["mesh_loss_raw"] * inv_batches
            if need_norm_recon:
                log_dict["norm_loss"] = epoch_sums["norm_loss_w"] * inv_batches
                log_dict["norm_loss_raw"] = epoch_sums["norm_loss_raw"] * inv_batches
            if use_reg and (w_reg > 0):
                log_dict["loss_lap_mea"] = epoch_sums["loss_lap_mea_w"] * inv_batches
                log_dict["loss_cons_mea"] = epoch_sums["loss_cons_mea_w"] * inv_batches
            if use_reg and (w_rigid > 0):
                log_dict["loss_rigid_mea"] = epoch_sums["loss_rigid_mea_w"] * inv_batches
            if use_reg and (w_trumpet > 0):
                log_dict["loss_trumpet"] = epoch_sums["loss_trumpet_w"] * inv_batches
            if w_smooth > 0:
                log_dict["loss_smooth"] = epoch_sums["loss_smooth_w"] * inv_batches
            if w_normal > 0:
                log_dict["loss_normal"] = epoch_sums["loss_normal_w"] * inv_batches
            if w_consistency > 0:
                log_dict["loss_consistency_fit"] = epoch_sums["loss_consistency_fit_w"] * inv_batches
            if w_spectral > 0:
                log_dict["loss_spectral"] = epoch_sums["loss_spectral_w"] * inv_batches
            if w_cond > 0:
                log_dict["loss_cond"] = epoch_sums["loss_cond_w"] * inv_batches

            if log_gradients and (epoch_grad_samples > 0):
                inv_grad_samples = 1.0 / float(epoch_grad_samples)
                grad_total = epoch_grad_sums["total"] * inv_grad_samples
                log_dict["grad_norm_total"] = grad_total
                grad_component_keys = [
                    "mse", "kl", "scale", "vert", "norm",
                    "reg_lap_mea", "reg_cons_mea", "reg_rigid_mea", "trumpet",
                    "smooth", "normal", "consistency_fit", "spectral", "cond",
                ]
                grad_component_sum = 0.0
                grad_component_means = {}
                for key in grad_component_keys:
                    mean_grad = epoch_grad_sums[key] * inv_grad_samples
                    grad_component_means[key] = mean_grad
                    grad_component_sum += mean_grad
                    log_dict[f"grad_norm_{key}"] = mean_grad
                grad_component_sum = max(grad_component_sum, 1e-12)
                for key, mean_grad in grad_component_means.items():
                    log_dict[f"grad_ratio_{key}"] = mean_grad / grad_component_sum

            if val_metrics is not None:
                for key, value in val_metrics.items():
                    log_dict[f"val_{key}"] = value

                if early_stopping:
                    current_val_loss = val_metrics["total_loss"]
                    if current_val_loss < (best_val_loss - early_stopping_min_delta):
                        best_val_loss = current_val_loss
                        best_val_epoch = epoch
                        epochs_without_improvement = 0
                        bad_param_names = _module_nonfinite_param_names(generator_module)
                        if bad_param_names:
                            print(
                                f"[warn] Refusing to save best_model at epoch {epoch} because parameters are non-finite: {bad_param_names}",
                                flush=True,
                            )
                        else:
                            torch.save(
                                {
                                    "epoch": epoch,
                                    "generator": generator_module.state_dict(),
                                    "optimizer_G": optimizer_G.state_dict(),
                                    "best_val_loss": best_val_loss,
                                    "w_kl": w_kl,
                                    **checkpoint_extra_state,
                                },
                                best_ckpt_path,
                            )
                        if log_wandb:
                            wandb.summary["best_val_loss"] = best_val_loss
                            wandb.summary["best_val_epoch"] = best_val_epoch
                    else:
                        epochs_without_improvement += 1
                        log_dict["early_stop_patience_counter"] = epochs_without_improvement

            if log_dict["total_loss"] < best_total_loss:
                best_total_loss = log_dict["total_loss"]
                best_epoch = epoch
                if log_wandb:
                    wandb.summary["best_total_loss"] = best_total_loss
                    wandb.summary["best_epoch"] = best_epoch

            eta_msg = _format_seconds(log_dict["eta_remaining_sec"])
            print(
                f"Epoch {epoch}/{epochs} done | epoch_time={log_dict['epoch_time_sec']:.2f}s | "
                f"avg_epoch={log_dict['avg_epoch_time_sec']:.2f}s | ETA={eta_msg}",
                flush=True,
            )

            if epoch % log_every == 0:
                print(log_dict)

            if log_wandb:
                wandb_log_dict = {
                    "epoch": epoch,
                    "train/loss/total": log_dict["total_loss"],
                    "train/loss/weighted/mse": log_dict["mse_loss"],
                    "train/loss/raw/mse": log_dict["mse_loss_raw"],
                    "train/opt/lr": log_dict["lr"],
                    "train/perf/epoch_time_sec": log_dict["epoch_time_sec"],
                    "train/perf/avg_epoch_time_sec": log_dict["avg_epoch_time_sec"],
                    "train/perf/samples_per_sec": log_dict["samples_per_sec"],
                    "train/perf/eta_remaining_sec": log_dict["eta_remaining_sec"],
                    "train/perf/eta_remaining_h": log_dict["eta_remaining_h"],
                    "train/progress/global_step": log_dict["global_step"],
                    "train/progress/epochs_remaining": log_dict["epochs_remaining"],
                    "train/debug/nonfinite_steps": log_dict["nonfinite_steps"],
                }
                if "vert_loss" in log_dict:
                    wandb_log_dict["train/loss/weighted/vert"] = log_dict["vert_loss"]
                    wandb_log_dict["train/loss/raw/vert"] = log_dict["vert_loss_raw"]

                if "w_kl" in log_dict:
                    wandb_log_dict["train/weights/w_kl"] = log_dict["w_kl"]
                    wandb_log_dict["train/loss/weighted/kl"] = log_dict["kl_loss"]
                    wandb_log_dict["train/loss/raw/kl"] = log_dict["kl_loss_raw"]
                if "scale_loss" in log_dict:
                    wandb_log_dict["train/loss/weighted/scale"] = log_dict["scale_loss"]
                    wandb_log_dict["train/loss/raw/scale"] = log_dict["scale_loss_raw"]
                if "norm_loss" in log_dict:
                    wandb_log_dict["train/loss/weighted/norm"] = log_dict["norm_loss"]
                    wandb_log_dict["train/loss/raw/norm"] = log_dict["norm_loss_raw"]
                if "loss_lap_mea" in log_dict:
                    wandb_log_dict["train/loss/weighted/reg_lap_mea"] = log_dict["loss_lap_mea"]
                if "loss_cons_mea" in log_dict:
                    wandb_log_dict["train/loss/weighted/reg_cons_mea"] = log_dict["loss_cons_mea"]
                if "loss_rigid_mea" in log_dict:
                    wandb_log_dict["train/loss/weighted/reg_rigid_mea"] = log_dict["loss_rigid_mea"]
                if "loss_trumpet" in log_dict:
                    wandb_log_dict["train/loss/weighted/trumpet"] = log_dict["loss_trumpet"]
                if "loss_smooth" in log_dict:
                    wandb_log_dict["train/loss/weighted/smooth"] = log_dict["loss_smooth"]
                if "loss_normal" in log_dict:
                    wandb_log_dict["train/loss/weighted/normal"] = log_dict["loss_normal"]
                if "loss_consistency_fit" in log_dict:
                    wandb_log_dict["train/loss/weighted/consistency_fit"] = log_dict["loss_consistency_fit"]
                if "loss_spectral" in log_dict:
                    wandb_log_dict["train/loss/weighted/spectral"] = log_dict["loss_spectral"]
                if "loss_cond" in log_dict:
                    wandb_log_dict["train/loss/weighted/cond"] = log_dict["loss_cond"]

                if "grad_norm_total" in log_dict:
                    wandb_log_dict["train/grad_norm/total"] = log_dict["grad_norm_total"]
                for grad_key in [
                    "mse", "kl", "scale", "vert", "norm",
                    "reg_lap_mea", "reg_cons_mea", "reg_rigid_mea", "trumpet",
                    "smooth", "normal", "consistency_fit", "spectral", "cond",
                ]:
                    gn_key = f"grad_norm_{grad_key}"
                    gr_key = f"grad_ratio_{grad_key}"
                    if gn_key in log_dict:
                        wandb_log_dict[f"train/grad_norm/{grad_key}"] = log_dict[gn_key]
                    if gr_key in log_dict:
                        wandb_log_dict[f"train/grad_ratio/{grad_key}"] = log_dict[gr_key]

                if "val_total_loss" in log_dict:
                    wandb_log_dict["val/loss/total"] = log_dict["val_total_loss"]
                if "val_kl_loss" in log_dict:
                    wandb_log_dict["val/loss/weighted/kl"] = log_dict["val_kl_loss"]
                if "val_mse_loss" in log_dict:
                    wandb_log_dict["val/loss/weighted/mse"] = log_dict["val_mse_loss"]
                if "val_vert_loss" in log_dict:
                    wandb_log_dict["val/loss/weighted/vert"] = log_dict["val_vert_loss"]
                if "val_scale_loss" in log_dict:
                    wandb_log_dict["val/loss/weighted/scale"] = log_dict["val_scale_loss"]
                if "val_norm_loss" in log_dict:
                    wandb_log_dict["val/loss/weighted/norm"] = log_dict["val_norm_loss"]
                if "val_loss_consistency_fit" in log_dict:
                    wandb_log_dict["val/loss/weighted/consistency_fit"] = log_dict["val_loss_consistency_fit"]
                if "val_loss_spectral" in log_dict:
                    wandb_log_dict["val/loss/weighted/spectral"] = log_dict["val_loss_spectral"]
                if "val_loss_cond" in log_dict:
                    wandb_log_dict["val/loss/weighted/cond"] = log_dict["val_loss_cond"]
                if "early_stop_patience_counter" in log_dict:
                    wandb_log_dict["train/early_stop/patience_counter"] = log_dict["early_stop_patience_counter"]

                wandb.log(wandb_log_dict, step=epoch)

        if is_main_process and ((epoch % 500 == 0) or (epoch == epochs - 1) or epoch in [50, 100, 200, 300, 400]):
            if reload_epoch is None or epoch != reload_epoch:
                checkpoint_path = log_path / f"models_epoch_{epoch}.pth"
                checkpoint_saved = False
                bad_param_names = _module_nonfinite_param_names(generator_module)
                if bad_param_names:
                    print(
                        f"[warn] Refusing to save epoch {epoch} checkpoint because parameters are non-finite: {bad_param_names}",
                        flush=True,
                    )
                else:
                    save_models(
                        generator_module,
                        optimizer_G,
                        epoch,
                        str(log_path),
                        cond_keys=["ostium_ring"],
                        cond_loss_style="ring_mse",
                        extra_state=checkpoint_extra_state,
                    )
                    checkpoint_saved = True
                if checkpoint_saved and (int(args.run_checkpoint_inference) == 1):
                    preview_output_dir = log_path / "preview_inference" / f"epoch_{epoch:05d}"
                    try:
                        run_checkpoint_preview_inference(
                            project_root=project_root,
                            checkpoint_path=checkpoint_path,
                            preview_case=preview_case,
                            output_dir=preview_output_dir,
                            num_samples=args.preview_num_samples,
                            seed=args.preview_seed,
                            ring_points=args.ring_points,
                            hidden_dim=hidden_dim,
                            latent_dim=latent_dim,
                            cond_embed_dim=cond_embed_dim,
                            condition_root=condition_root,
                            checkpoints_root=checkpoints_root,
                            ghd_chk_root=ghd_chk_root,
                            alignment_root=condition_root,
                            canonical_root=canonical_root,
                        )
                    except subprocess.CalledProcessError as e:
                        print(
                            f"[warn] Preview inference failed at epoch {epoch} with return code {e.returncode}. "
                            "Continuing training."
                        )

        should_stop = False
        if is_main_process and early_stopping and (best_val_epoch >= 0) and (epochs_without_improvement >= early_stopping_patience):
            print(
                f"Early stopping at epoch {epoch}: no val improvement for {epochs_without_improvement} "
                f"validation checks (patience={early_stopping_patience})."
            )
            stopped_early = True
            should_stop = True

        if distributed:
            should_stop_t = torch.tensor(1.0 if should_stop else 0.0, device=device)
            dist.broadcast(should_stop_t, src=0)
            should_stop = bool(should_stop_t.item())

        if should_stop:
            break

        scheduler_G.step()

    if is_main_process and (len(test_subset) > 0):
        eval_w_kl = w_kl
        if best_ckpt_path.exists():
            best_chk = torch.load(best_ckpt_path, map_location=device)
            generator_module.load_state_dict(best_chk["generator"])
            eval_w_kl = float(best_chk.get("w_kl", w_kl))
            print(f"Loaded best validation model from {best_ckpt_path} (epoch={best_chk.get('epoch')}).")

        test_metrics = evaluate_loader(test_loader, eval_w_kl)
        if test_metrics is not None:
            test_log = {f"test_{k}": v for k, v in test_metrics.items()}
            test_log["epoch"] = last_epoch
            print(test_log)
            if log_wandb:
                wandb_test_log = {"epoch": last_epoch}
                for key, value in test_metrics.items():
                    wandb_test_log[f"test/loss/{key}"] = value
                wandb.log(wandb_test_log, step=last_epoch)
                if "total_loss" in test_metrics:
                    wandb.summary["test_total_loss"] = test_metrics["total_loss"]
                if stopped_early:
                    wandb.summary["stopped_early"] = 1

    if log_wandb:
        wandb.finish()

    if distributed:
        distributed_barrier(distributed, local_rank, dist_backend)
        dist.destroy_process_group()
