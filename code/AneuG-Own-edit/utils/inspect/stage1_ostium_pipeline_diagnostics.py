#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from first_stage_ostium_conditional import (
    collect_available_cases,
    compute_fitting_norm_canonical,
    _reconstruct_fitted_mesh_from_checkpoint,
)
from infer_stage1_ostium_conditional import ensure_canonical_diff_checkpoint, maybe_apply_training_stats
from models.ghd_reconstruct import GHD_Reconstruct
from models.vae_datasets import OstiumGHDDataset
from models.vae_models import ConditionalGHDVAE
from utils.utils import safe_load_mesh


class GHDAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        # Residual reconstruction makes identity overfit easier to test.
        return x + self.decoder(z)


def parse_args() -> argparse.Namespace:
    checkpoints_root = ROOT / "checkpoint-v2"
    parser = argparse.ArgumentParser(description="Stage-1 ostium pipeline diagnostics with side-by-side renders.")
    parser.add_argument("--checkpoints-root", type=Path, default=checkpoints_root)
    parser.add_argument("--ghd-chk-root", type=Path, default=checkpoints_root / "ghd_fitting")
    parser.add_argument("--canonical-root", type=Path, default=checkpoints_root / "canonical_model")
    parser.add_argument("--condition-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ring-points", type=int, default=20)
    parser.add_argument("--split-train-ratio", type=float, default=0.8)
    parser.add_argument("--split-val-ratio", type=float, default=0.1)
    parser.add_argument("--split-test-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--force-resplit", type=int, default=0)
    parser.add_argument("--train-subset-limit", type=int, default=None)
    parser.add_argument("--posterior-noise-scale", type=float, default=0.0)
    parser.add_argument("--ae-epochs", type=int, default=1500)
    parser.add_argument("--ae-hidden-dim", type=int, default=2048)
    parser.add_argument("--ae-latent-dim", type=int, default=512)
    parser.add_argument("--ae-lr", type=float, default=1e-3)
    parser.add_argument("--ae-log-every", type=int, default=100)
    parser.add_argument(
        "--comparison-mode",
        type=str,
        default="legacy_visual",
        choices=["legacy_visual", "strict_replay_debug"],
        help=(
            "legacy_visual: render checkpoints in the historical Stage-1 visual space for comparability, "
            "strict_replay_debug: use the fitting-scale replay path for debugging checkpoint vs warped mismatches."
        ),
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="full",
        choices=["simple", "full"],
        help="simple: render only input vs current VAE output, full: include all diagnostic references.",
    )
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


def denormalize_target(dataset: OstiumGHDDataset, target_norm: torch.Tensor) -> torch.Tensor:
    return target_norm * dataset.target_std.to(target_norm.device) + dataset.target_mean.to(target_norm.device)


def build_reconstructor(canonical_root: Path, device: torch.device) -> GHD_Reconstruct:
    return build_reconstructor_for_mode(canonical_root, device, comparison_mode="strict_replay_debug")


def compute_legacy_norm_canonical(canonical_meshes_raw) -> float:
    v_raw = canonical_meshes_raw.verts_packed()
    return torch.max(torch.norm(v_raw, dim=-1)).item() * 1.10


def build_reconstructor_for_mode(
    canonical_root: Path,
    device: torch.device,
    comparison_mode: str,
) -> GHD_Reconstruct:
    canonical_meshes_raw = safe_load_mesh(str(canonical_root / "part_aligned.obj"))
    ensure_canonical_diff_checkpoint(canonical_root, canonical_meshes_raw)
    from pytorch3d.structures import Meshes as P3dMeshes

    v_raw = canonical_meshes_raw.verts_packed()
    if comparison_mode == "strict_replay_debug":
        norm_canonical_val = compute_fitting_norm_canonical(canonical_meshes_raw)
    else:
        norm_canonical_val = compute_legacy_norm_canonical(canonical_meshes_raw)
    v_normed = v_raw / norm_canonical_val
    canonical_meshes = P3dMeshes(verts=[v_normed], faces=canonical_meshes_raw.faces_list())
    return GHD_Reconstruct(
        canonical_meshes,
        str(canonical_root / "canonical_model_144_normed.pkl"),
        num_Basis=12 ** 2,
        device=device,
        skip_normalize=True,
        norm_canonical_override=norm_canonical_val,
    )


def apply_case_transform(verts: torch.Tensor, r_axis: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    rotation = axis_angle_to_matrix(r_axis)
    return verts @ rotation.transpose(-1, -2) + t.unsqueeze(1)


def mesh_from_target_vector(ghd_reconstruct: GHD_Reconstruct, target_raw: torch.Tensor) -> Meshes:
    ghd = target_raw[:, :-1]
    scale = target_raw[:, -1:]
    return ghd_reconstruct.ghd_forward_as_Meshes(ghd, denormalize_shape=False, scale=scale)


def render_case_diagnostic(
    output_path: Path,
    case_name: str,
    input_mesh: Meshes,
    replay_mesh: Meshes,
    vae_mesh: Meshes,
    cond_ae_mesh: Meshes,
    ae_mesh: Meshes,
    replay_rmse: float,
    vae_rmse: float,
    cond_ae_rmse: float,
    ae_rmse: float,
    comparison_mode: str,
    render_mode: str,
    dpi: int,
) -> None:
    if render_mode == "simple":
        entries = [
            ("Input from ghb_fitting_checkpoint.pkl", input_mesh),
            (f"Current Stage-1 VAE\nRMSE {vae_rmse:.5f}", vae_mesh),
        ]
    else:
        entries = [
            ("Input from ghb_fitting_checkpoint.pkl", input_mesh),
            (f"Replay from ghb_fitting_checkpoint.pkl\nRMSE {replay_rmse:.5f}", replay_mesh),
            (f"Current Stage-1 VAE\nRMSE {vae_rmse:.5f}", vae_mesh),
            (f"One-loss Conditional AE\nRMSE {cond_ae_rmse:.5f}", cond_ae_mesh),
            (f"One-loss GHD AE\nRMSE {ae_rmse:.5f}", ae_mesh),
        ]
    finite_meshes = []
    for _, mesh in entries:
        verts = mesh.verts_packed()
        if torch.isfinite(verts).all():
            finite_meshes.append(verts)
    if finite_meshes:
        all_verts = torch.cat(finite_meshes, dim=0).detach().cpu().numpy()
        min_xyz = all_verts.min(axis=0)
        max_xyz = all_verts.max(axis=0)
        center = (min_xyz + max_xyz) * 0.5
        extent = float((max_xyz - min_xyz).max())
        radius = max(0.5 * extent, 1e-3)
    else:
        center = torch.zeros(3).numpy()
        radius = 1.0

    if render_mode == "simple":
        fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    else:
        fig = plt.figure(figsize=(15.5, 4.6), constrained_layout=True)
    fig.suptitle(case_name, fontsize=14)
    for idx, (title, mesh) in enumerate(entries, start=1):
        verts = mesh.verts_packed().detach().cpu().numpy()
        faces = mesh.faces_packed().detach().cpu().numpy()
        ax = fig.add_subplot(1, len(entries), idx, projection="3d")
        if np.isfinite(verts).all():
            ax.plot_trisurf(
                verts[:, 0], verts[:, 1], verts[:, 2],
                triangles=faces,
                color="#8eb3d3",
                edgecolor=(0.06, 0.12, 0.18, 0.08),
                linewidth=0.08,
                alpha=0.32,
                shade=True,
            )
        else:
            ax.text2D(0.5, 0.5, "Invalid mesh\nNaN/Inf", transform=ax.transAxes, ha="center", va="center", fontsize=12)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.view_init(elev=17, azim=35)
        ax.set_box_aspect([1.0, 1.0, 1.0])
        ax.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    if (not torch.isfinite(a).all()) or (not torch.isfinite(b).all()):
        return float("nan")
    return float(torch.sqrt(torch.mean((a - b) ** 2)).item())


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_root = args.checkpoints_root.expanduser()
    ghd_chk_root = args.ghd_chk_root.expanduser()
    canonical_root = args.canonical_root.expanduser()
    condition_root = args.condition_root.expanduser() if args.condition_root is not None else ghd_chk_root
    available_cases = collect_available_cases(
        ghd_chk_root=ghd_chk_root,
        condition_root=condition_root,
        ghd_run="vanilla",
        ghd_chk_name="ghb_fitting_checkpoint.pkl",
        condition_filename="opa_checkpoint.pkl",
    )

    ghd_reconstruct = build_reconstructor_for_mode(canonical_root, device, args.comparison_mode)
    dataset = OstiumGHDDataset(
        ghd_chk_root=str(ghd_chk_root),
        alignment_root=str(condition_root),
        canonical_opa_chk_path=str(canonical_root / "opa_checkpoint.pkl"),
        cases=available_cases,
        ghd_run="vanilla",
        ghd_chk_name="ghb_fitting_checkpoint.pkl",
        withscale=True,
        normalize=True,
        ring_points=args.ring_points,
    )

    vae_checkpoint = torch.load(args.checkpoint.expanduser(), map_location=device)
    maybe_apply_training_stats(
        dataset,
        vae_checkpoint,
        available_cases,
        args,
        checkpoints_root,
        ghd_chk_root,
        condition_root,
    )
    dataset.target_mean = dataset.target_mean.cpu()
    dataset.target_std = dataset.target_std.cpu()
    dataset.cond_mean = dataset.cond_mean.cpu()
    dataset.cond_std = dataset.cond_std.cpu()

    vae_state = vae_checkpoint["generator"]
    input_dim, hidden_dim, latent_dim, cond_embed_dim, norm_type = infer_model_hparams(vae_state)
    vae = ConditionalGHDVAE(
        input_dim,
        hidden_dim,
        latent_dim,
        cond_dim=dataset.get_cond_dim(),
        cond_embed_dim=cond_embed_dim,
        norm_type=norm_type,
    ).to(device)
    vae.load_state_dict(vae_state)
    vae.eval()

    case_to_index = {case: idx for idx, case in enumerate(dataset.updated_cases)}
    selected_cases = []
    for case in args.cases:
        if case not in case_to_index:
            raise KeyError(f"Case not found in dataset: {case}")
        selected_cases.append(case)

    targets_norm = []
    conds_norm = []
    input_verts = []
    input_faces = None
    r_axes = []
    translations = []
    replay_meshes: dict[str, Meshes] = {}
    simplified_replay_meshes: dict[str, Meshes] = {}
    input_meshes: dict[str, Meshes] = {}

    for case in selected_cases:
        case_idx = case_to_index[case]
        item = dataset[case_idx]
        targets_norm.append(item["target"].to(device))
        conds_norm.append(item["condition"].to(device))

        case_root = ghd_chk_root / case

        with open(case_root / "vanilla" / "ghb_fitting_checkpoint.pkl", "rb") as f:
            chk = pickle.load(f)
        r_axes.append(chk["R"].reshape(3).float().to(device))
        translations.append(chk["T"].reshape(3).float().to(device))

        target_raw = torch.cat([dataset.ghd[case_idx], dataset.scale[case_idx]], dim=0).unsqueeze(0).to(device)
        simplified_replay_mesh = mesh_from_target_vector(ghd_reconstruct, target_raw)
        simplified_replay_verts = apply_case_transform(
            simplified_replay_mesh.verts_padded(),
            r_axes[-1].unsqueeze(0),
            translations[-1].unsqueeze(0),
        )
        simplified_replay_meshes[case] = Meshes(
            verts=[simplified_replay_verts[0]],
            faces=[simplified_replay_mesh.faces_packed()],
        )
        input_meshes[case] = simplified_replay_meshes[case]
        input_verts.append(simplified_replay_verts[0])
        if input_faces is None:
            input_faces = simplified_replay_mesh.faces_packed()
        replay_mesh = _reconstruct_fitted_mesh_from_checkpoint(ghd_reconstruct, chk, device)
        replay_meshes[case] = Meshes(
            verts=[replay_mesh.verts_padded()[0]],
            faces=[replay_mesh.faces_packed()],
        )

    targets_norm_batch = torch.stack(targets_norm, dim=0)
    conds_norm_batch = torch.stack(conds_norm, dim=0)
    input_verts_batch = torch.stack(input_verts, dim=0)
    r_axes_batch = torch.stack(r_axes, dim=0)
    translations_batch = torch.stack(translations, dim=0)

    with torch.no_grad():
        vae_pred_norm, _, _ = vae(targets_norm_batch, conds_norm_batch, noise_scale=float(args.posterior_noise_scale))
        vae_pred_raw = denormalize_target(dataset, vae_pred_norm)
        vae_mesh = mesh_from_target_vector(ghd_reconstruct, vae_pred_raw)
        vae_verts = apply_case_transform(vae_mesh.verts_padded(), r_axes_batch, translations_batch)

    cond_ae = ConditionalGHDVAE(
        input_dim,
        hidden_dim,
        latent_dim,
        cond_dim=dataset.get_cond_dim(),
        cond_embed_dim=cond_embed_dim,
        norm_type=norm_type,
    ).to(device)
    optimizer_cond_ae = torch.optim.Adam(cond_ae.parameters(), lr=args.ae_lr)
    cond_ae_history = []
    for epoch in range(1, args.ae_epochs + 1):
        optimizer_cond_ae.zero_grad(set_to_none=True)
        cond_ae_pred_norm, _, _ = cond_ae(targets_norm_batch, conds_norm_batch, noise_scale=0.0)
        cond_ae_pred_raw = denormalize_target(dataset, cond_ae_pred_norm)
        cond_ae_mesh_batch = mesh_from_target_vector(ghd_reconstruct, cond_ae_pred_raw)
        cond_ae_verts = apply_case_transform(cond_ae_mesh_batch.verts_padded(), r_axes_batch, translations_batch)
        cond_ae_loss = F.mse_loss(cond_ae_verts, input_verts_batch)
        cond_ae_loss.backward()
        optimizer_cond_ae.step()
        if epoch == 1 or epoch % args.ae_log_every == 0 or epoch == args.ae_epochs:
            cond_ae_history.append(
                {"epoch": epoch, "rmse": float(torch.sqrt(cond_ae_loss).item()), "loss": float(cond_ae_loss.item())}
            )
            print({"cond_ae": cond_ae_history[-1]}, flush=True)

    with torch.no_grad():
        cond_ae_pred_norm, _, _ = cond_ae(targets_norm_batch, conds_norm_batch, noise_scale=0.0)
        cond_ae_pred_raw = denormalize_target(dataset, cond_ae_pred_norm)
        cond_ae_mesh_batch = mesh_from_target_vector(ghd_reconstruct, cond_ae_pred_raw)
        cond_ae_verts = apply_case_transform(cond_ae_mesh_batch.verts_padded(), r_axes_batch, translations_batch)

    ae = GHDAutoencoder(input_dim=targets_norm_batch.shape[1], hidden_dim=args.ae_hidden_dim, latent_dim=args.ae_latent_dim).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=args.ae_lr)
    ae_history = []
    for epoch in range(1, args.ae_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        ae_pred_norm = ae(targets_norm_batch)
        ae_pred_raw = denormalize_target(dataset, ae_pred_norm)
        ae_mesh_batch = mesh_from_target_vector(ghd_reconstruct, ae_pred_raw)
        ae_verts = apply_case_transform(ae_mesh_batch.verts_padded(), r_axes_batch, translations_batch)
        loss = F.mse_loss(ae_verts, input_verts_batch)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % args.ae_log_every == 0 or epoch == args.ae_epochs:
            ae_history.append({"epoch": epoch, "rmse": float(torch.sqrt(loss).item()), "loss": float(loss.item())})
            print(ae_history[-1], flush=True)

    with torch.no_grad():
        ae_pred_norm = ae(targets_norm_batch)
        ae_pred_raw = denormalize_target(dataset, ae_pred_norm)
        ae_mesh_batch = mesh_from_target_vector(ghd_reconstruct, ae_pred_raw)
        ae_verts = apply_case_transform(ae_mesh_batch.verts_padded(), r_axes_batch, translations_batch)

    summary_cases = []
    for idx, case in enumerate(selected_cases):
        input_mesh = input_meshes[case]
        replay_mesh = replay_meshes[case]
        vae_case_mesh = Meshes(verts=[vae_verts[idx]], faces=[input_faces])
        cond_ae_case_mesh = Meshes(verts=[cond_ae_verts[idx]], faces=[input_faces])
        ae_case_mesh = Meshes(verts=[ae_verts[idx]], faces=[input_faces])
        replay_rmse = rmse(replay_mesh.verts_packed(), input_mesh.verts_packed())
        vae_rmse = rmse(vae_verts[idx], input_mesh.verts_packed())
        cond_ae_rmse = rmse(cond_ae_verts[idx], input_mesh.verts_packed())
        ae_rmse = rmse(ae_verts[idx], input_mesh.verts_packed())

        image_path = output_dir / f"{case.replace('/', '__')}_stage1_pipeline_diagnostic.png"
        render_case_diagnostic(
            image_path,
            case,
            input_mesh,
            replay_mesh,
            vae_case_mesh,
            cond_ae_case_mesh,
            ae_case_mesh,
            replay_rmse,
            vae_rmse,
            cond_ae_rmse,
            ae_rmse,
            args.comparison_mode,
            args.render_mode,
            dpi=args.dpi,
        )

        summary_cases.append(
            {
                "case": case,
                "input_source": "ghb_fitting_checkpoint.pkl",
                "replay_rmse": replay_rmse,
                "vae_rmse": vae_rmse,
                "cond_ae_rmse": cond_ae_rmse,
                "ae_rmse": ae_rmse,
                "image_path": str(image_path),
            }
        )

    summary = {
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "comparison_mode": args.comparison_mode,
        "render_mode": args.render_mode,
        "cases": summary_cases,
        "cond_ae_history": cond_ae_history,
        "ae_history": ae_history,
        "ae_hidden_dim": args.ae_hidden_dim,
        "ae_latent_dim": args.ae_latent_dim,
        "ae_epochs": args.ae_epochs,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote diagnostics to {summary_path}")


if __name__ == "__main__":
    main()
