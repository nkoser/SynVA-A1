"""Shared mesh-aware loss for methods/* trainers.

Provides:
  - CoeffToMesh(canonical_obj, eigen_chk, num_basis, device)
      Converts normalized GHD coefficients [B,432] -> mesh vertices [B,V,3]
      and (optionally) per-vertex normals.
  - mesh_recon_losses(coeff2mesh, ghd_real, ghd_recon, mean, std, want_normals=True)
      Returns dict {ghd_mse, vert_mse, normal_mse}.
  - apply_loss_mix(losses, w_mse=1.0, w_vert=100.0, w_normal=10.0)
      Weighted sum matching the works VAE recipe.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix

from ghd.base.graph_harmonic_deformation import Graph_Harmonic_Deform


class CoeffToMesh:
    """Decode normalized GHD coefficients to mesh vertices/normals.

    Mirrors first_stage_unconditional_works.py:CoeffToMesh — single source of
    truth used by both the works VAE and methods/* trainers.

    Norm scale is `verts.norm.max() * canonical_norm_factor` (default 1.10),
    matching the GHD fitter (ghd/fitting/fitter.py:903).
    """

    def __init__(self, canonical_mesh_path: str, eigen_chk: str,
                 num_basis: int = 144, device: str = "cuda",
                 canonical_norm_factor: float = 1.10):
        self.device = torch.device(device)
        canonical = load_objs_as_meshes([canonical_mesh_path])
        verts = canonical.verts_packed()
        scale = float(torch.norm(verts, dim=-1).max().item() * canonical_norm_factor)
        canonical = canonical.update_padded(canonical.verts_padded() / scale)
        self.canonical = canonical.to(self.device)
        self.norm_scale = scale
        ghd_module = Graph_Harmonic_Deform(
            base_shape=self.canonical, num_Basis=num_basis, eigen_chk=eigen_chk,
        )
        self.eigvec = ghd_module.GBH_eigvec.to(self.device)  # [N, num_basis]

    def __call__(self, ghd_normalized: torch.Tensor,
                 mean: torch.Tensor, std: torch.Tensor,
                 want_normals: bool = True,
                 use_scale_dim: bool = False,
                 case_rotation: torch.Tensor | None = None,
                 case_translation: torch.Tensor | None = None):
        B = ghd_normalized.shape[0]
        scale = None
        mean = mean.to(self.device)
        std = std.to(self.device)
        ghd_normalized = ghd_normalized.to(self.device)
        if use_scale_dim and ghd_normalized.shape[-1] > 432 and mean.shape[-1] > 432:
            raw = ghd_normalized * std[:, :ghd_normalized.shape[-1]] + mean[:, :ghd_normalized.shape[-1]]
            x = raw[:, :432]
            scale = raw[:, 432:433].abs()
        else:
            x = ghd_normalized[:, :432] * std[:, :432] + mean[:, :432]
        x = x.view(B, -1, 3)
        offset = torch.einsum("nm,bmc->bnc", self.eigvec, x)
        verts = self.canonical.verts_padded().repeat(B, 1, 1) + offset
        if scale is not None:
            verts = verts * scale.unsqueeze(-1)
        if case_rotation is not None and case_translation is not None:
            rotation = axis_angle_to_matrix(case_rotation.to(self.device))
            verts = verts @ rotation.transpose(-1, -2) + case_translation.to(self.device).unsqueeze(1)
        if not want_normals:
            return verts, None
        m = Meshes(
            verts=verts,
            faces=self.canonical.faces_padded().repeat(B, 1, 1),
        )
        return verts, m.verts_normals_padded()


def mesh_recon_losses(coeff2mesh: CoeffToMesh,
                      ghd_real: torch.Tensor,
                      ghd_recon: torch.Tensor,
                      mean: torch.Tensor,
                      std: torch.Tensor,
                      want_normals: bool = True,
                      use_scale_dim: bool = False,
                      case_rotation: torch.Tensor | None = None,
                      case_translation: torch.Tensor | None = None):
    """Returns (ghd_mse, vert_mse, normal_mse) — normal_mse is 0 if disabled."""
    v_real, n_real = coeff2mesh(
        ghd_real, mean, std, want_normals=want_normals,
        use_scale_dim=use_scale_dim,
        case_rotation=case_rotation,
        case_translation=case_translation,
    )
    v_rec,  n_rec  = coeff2mesh(
        ghd_recon, mean, std, want_normals=want_normals,
        use_scale_dim=use_scale_dim,
        case_rotation=case_rotation,
        case_translation=case_translation,
    )
    ghd_mse  = F.mse_loss(ghd_recon, ghd_real)
    vert_mse = F.mse_loss(v_rec, v_real)
    if want_normals:
        normal_mse = F.mse_loss(n_rec, n_real)
    else:
        normal_mse = torch.zeros((), device=ghd_real.device)
    return ghd_mse, vert_mse, normal_mse


def opening_ring_losses(pred_verts: torch.Tensor,
                        target_ring: torch.Tensor,
                        opening_idx: torch.Tensor):
    """Ordered ring MSE plus symmetric chamfer for the generated opening.

    pred_verts is [B,V,3] in mesh/GHD-local frame; target_ring is [B,R,3]
    in the same frame.  opening_idx may have a different count than R, so the
    ordered MSE uses a simple index linspace while chamfer uses all points.
    """
    idx = opening_idx.to(pred_verts.device).long()
    idx = idx[(idx >= 0) & (idx < pred_verts.shape[1])]
    if idx.numel() < 3:
        z = torch.zeros((), device=pred_verts.device, dtype=pred_verts.dtype)
        return z, z
    pred_ring_full = pred_verts[:, idx, :]
    R = target_ring.shape[1]
    if pred_ring_full.shape[1] != R:
        sel = torch.linspace(0, pred_ring_full.shape[1] - 1, R,
                             device=pred_verts.device).round().long()
        pred_ring_ordered = pred_ring_full[:, sel, :]
    else:
        pred_ring_ordered = pred_ring_full
    ring_mse = F.mse_loss(pred_ring_ordered, target_ring)
    dist = torch.cdist(pred_ring_full, target_ring)
    chamfer = 0.5 * (dist.min(dim=2).values.mean() + dist.min(dim=1).values.mean())
    return ring_mse, chamfer


def apply_loss_mix(ghd_mse, vert_mse, normal_mse,
                   w_mse: float = 1.0, w_vert: float = 100.0, w_normal: float = 10.0):
    """Paper recipe: 1*MSE(GHD) + 100*MSE(vert) + 10*MSE(normal)."""
    return w_mse * ghd_mse + w_vert * vert_mse + w_normal * normal_mse


def add_mesh_loss_args(p):
    """Add CLI args for mesh-aware loss weights + canonical paths."""
    p.add_argument("--canonical_mesh_obj",
                   default="/path/to/SynVA-A1/checkpoints/canonical_average/part_aligned.obj")
    p.add_argument("--eigen_chk",
                   default="/path/to/SynVA-A1/checkpoints/canonical_average/eigen_chk_144.pkl")
    p.add_argument("--num_basis", type=int, default=144)
    p.add_argument("--w_mse",    type=float, default=1.0)
    p.add_argument("--w_vert",   type=float, default=100.0)
    p.add_argument("--w_normal", type=float, default=10.0)
    p.add_argument("--w_ring", type=float, default=0.0,
                   help="Weight for ordered generated-opening-to-conditioned-ring MSE.")
    p.add_argument("--w_ring_chamfer", type=float, default=0.0,
                   help="Weight for symmetric generated-opening-to-conditioned-ring chamfer.")
    p.add_argument("--no_normals", action="store_true",
                   help="Disable normal-MSE term (faster, slight quality loss).")
