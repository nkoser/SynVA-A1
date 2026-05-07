from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
sys.path.insert(0, ROOT)

from first_stage_vessel_aware import collate_fn, KL_divergence_terms
from methods.W_cond_ghd_vae.model import ConditionalGHDVAE
from methods._common.data import add_common_args, build_conditioner, copy_norm, load_cases, make_dataset, set_seed
from methods._common.mesh_loss import (
    CoeffToMesh,
    mesh_recon_losses,
    opening_ring_losses,
)
from methods._common.stage3_surrogate_loss import stage3_surrogate_losses
from methods.E_collision.collision_loss import load_opening_idx


def parse_args():
    p = argparse.ArgumentParser("W_cond_ghd_vae")
    add_common_args(p)
    p.add_argument("--hidden_dim", type=int, default=384)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--cond_embed_dim", type=int, default=128)
    p.add_argument("--norm_type", choices=["batch", "layer", "none"], default="batch")
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--w_mse", type=float, default=1.0)
    p.add_argument("--w_kl", type=float, default=1.0)
    p.add_argument("--w_scale", type=float, default=0.0)
    p.add_argument("--canonical_mesh_obj", default="/path/to/SynVA-A1/checkpoints/canonical_average/part_aligned.obj")
    p.add_argument("--eigen_chk", default="/path/to/SynVA-A1/checkpoints/canonical_average/eigen_chk_144.pkl")
    p.add_argument("--num_basis", type=int, default=144)
    p.add_argument("--w_vert", type=float, default=0.0)
    p.add_argument("--w_normal", type=float, default=0.0)
    p.add_argument("--w_ring", type=float, default=0.0)
    p.add_argument("--w_ring_chamfer", type=float, default=0.0)
    p.add_argument(
        "--w_pouch_offset",
        type=float,
        default=0.0,
        help="Match predicted pouch center offset from its opening center to the fitted target offset.",
    )
    p.add_argument(
        "--w_pouch_axis",
        type=float,
        default=0.0,
        help="Match predicted pouch center direction from its opening center to the fitted target direction.",
    )
    p.add_argument("--w_stage3_label2", type=float, default=0.0)
    p.add_argument("--w_stage3_center", type=float, default=0.0)
    p.add_argument("--w_stage3_nearest", type=float, default=0.0)
    p.add_argument("--w_stage3_side", type=float, default=0.0)
    p.add_argument("--w_stage3_opening", type=float, default=0.0)
    p.add_argument("--w_shape_extent", type=float, default=0.0,
                   help="Match sac bounding-box extents between predicted and target meshes.")
    p.add_argument("--w_shape_area", type=float, default=0.0,
                   help="Match sac surface-area proxy between predicted and target meshes.")
    p.add_argument("--w_shape_volume", type=float, default=0.0,
                   help="Match sac covariance-volume proxy between predicted and target meshes.")
    p.add_argument("--w_shape_moment", type=float, default=0.0,
                   help="Match sac coordinate moments/stds between predicted and target meshes.")
    p.add_argument("--w_surface_chamfer", type=float, default=0.0,
                   help="Direct sampled surface Chamfer between predicted and fitted target pouch meshes.")
    p.add_argument("--w_surface_normal", type=float, default=0.0,
                   help="Normal term from sampled surface Chamfer between predicted and target pouch meshes.")
    p.add_argument("--surface_samples", type=int, default=2048,
                   help="Number of points sampled per pouch for the direct surface Chamfer loss.")
    p.add_argument("--w_prior_mse", type=float, default=0.0,
                   help="Supervise the actual inference path decode(z~N(0,I), cond) against the target coefficients.")
    p.add_argument("--w_prior_batch_mean", type=float, default=0.0,
                   help="Match batch mean of prior-decoded coefficients to target batch mean.")
    p.add_argument("--w_prior_batch_std", type=float, default=0.0,
                   help="Match batch standard deviation of prior-decoded coefficients to target batch standard deviation.")
    p.add_argument("--prior_noise_scale", type=float, default=1.0,
                   help="Scale for z~N(0,I) used by prior-path training losses.")
    p.add_argument("--no_normals", action="store_true")
    p.add_argument("--early_stop_metric",
                   choices=["val_mse", "val_total", "val_vert_mse", "val_ring_chamfer"],
                   default="val_mse")
    p.add_argument("--kl_cap", type=float, default=30.0)
    p.add_argument("--free_bits", type=float, default=0.5)
    p.add_argument("--kl_warmup", type=int, default=400)
    p.add_argument("--condition_dropout", type=float, default=0.10)
    p.add_argument(
        "--condition_mode",
        choices=["ring", "vessel"],
        default="ring",
        help="ring: original W-style flat ordered ring; vessel: PointNet vessel/ostium/ring conditioner.",
    )
    p.add_argument("--use_morphology_condition", action="store_true",
                   help="Append normalized scalar aneurysm morphology parameters to the condition vector.")
    p.add_argument("--val_freq", type=int, default=25)
    p.add_argument("--log_freq", type=int, default=25)
    p.add_argument("--save_freq", type=int, default=500)
    return p.parse_args()


def kl_weight(args, epoch):
    return float(args.w_kl) * min(1.0, max(0.0, epoch / max(1, args.kl_warmup)))


def condition_from_batch(batch, device, drop_prob=0.0, conditioner=None, mode="ring",
                         use_morphology=False):
    if mode == "vessel":
        if conditioner is None:
            raise ValueError("condition_mode='vessel' requires a conditioner")
        cond = conditioner(
            batch["vessel_pts"].to(device),
            batch["ostium_params"].to(device),
            ostium_pts=batch.get("ostium_pts", None).to(device) if batch.get("ostium_pts", None) is not None else None,
            ostium_ring=batch.get("ostium_ring", None).to(device) if batch.get("ostium_ring", None) is not None else None,
        )
    else:
        cond = batch["ostium_ring"].to(device).reshape(batch["ostium_ring"].shape[0], -1)
    if use_morphology:
        if "morphology" not in batch:
            raise ValueError("--use_morphology_condition requires dataset batches with 'morphology'")
        cond = torch.cat([cond, batch["morphology"].to(device)], dim=-1)
    if drop_prob > 0:
        keep = (torch.rand(cond.shape[0], 1, device=device) >= drop_prob).float()
        cond = cond * keep
    return cond


def pouch_placement_losses(pred_verts, target_verts, opening_idx):
    idx = opening_idx.to(pred_verts.device).long()
    idx = idx[(idx >= 0) & (idx < pred_verts.shape[1])]
    if idx.numel() < 3:
        z = torch.zeros((), device=pred_verts.device, dtype=pred_verts.dtype)
        return z, z
    sac_mask = torch.ones(pred_verts.shape[1], dtype=torch.bool, device=pred_verts.device)
    sac_mask[idx] = False
    if int(sac_mask.sum().item()) < 3:
        z = torch.zeros((), device=pred_verts.device, dtype=pred_verts.dtype)
        return z, z

    pred_open = pred_verts[:, idx, :].mean(dim=1)
    target_open = target_verts[:, idx, :].mean(dim=1)
    pred_center = pred_verts[:, sac_mask, :].mean(dim=1)
    target_center = target_verts[:, sac_mask, :].mean(dim=1)
    pred_offset = pred_center - pred_open
    target_offset = target_center - target_open
    offset_loss = F.mse_loss(pred_offset, target_offset)

    pred_dir = F.normalize(pred_offset, dim=-1, eps=1e-8)
    target_dir = F.normalize(target_offset, dim=-1, eps=1e-8)
    axis_loss = (1.0 - (pred_dir * target_dir).sum(dim=-1)).mean()
    return offset_loss, axis_loss


def _relative_mse(pred, target, eps=1e-6):
    return (((pred - target) / target.detach().abs().clamp_min(eps)) ** 2).mean()


def batch_moment_losses(pred, target):
    if pred.shape[0] < 2:
        z = torch.zeros((), device=pred.device, dtype=pred.dtype)
        return z, z
    mean_loss = F.mse_loss(pred.mean(dim=0), target.detach().mean(dim=0))
    pred_std = pred.std(dim=0, unbiased=False)
    target_std = target.detach().std(dim=0, unbiased=False)
    std_loss = F.mse_loss(pred_std, target_std)
    return mean_loss, std_loss


def _sac_mask(num_verts, opening_idx, device):
    idx = opening_idx.to(device).long()
    idx = idx[(idx >= 0) & (idx < num_verts)]
    mask = torch.ones(num_verts, dtype=torch.bool, device=device)
    if idx.numel() > 0:
        mask[idx] = False
    return mask


def _sac_surface_area(verts, faces, sac_mask):
    if faces is None:
        return None
    faces = faces.to(verts.device).long()
    face_mask = sac_mask[faces].all(dim=-1)
    faces = faces[face_mask]
    if faces.numel() == 0:
        return None
    tri = verts[:, faces, :]
    cross = torch.linalg.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0], dim=-1)
    return 0.5 * cross.norm(dim=-1).sum(dim=1)


def sac_shape_losses(pred_verts, target_verts, opening_idx, faces=None):
    mask = _sac_mask(pred_verts.shape[1], opening_idx, pred_verts.device)
    if int(mask.sum().item()) < 4:
        z = torch.zeros((), device=pred_verts.device, dtype=pred_verts.dtype)
        return z, z, z, z

    pred = pred_verts[:, mask, :]
    target = target_verts[:, mask, :]

    pred_extent = pred.max(dim=1).values - pred.min(dim=1).values
    target_extent = target.max(dim=1).values - target.min(dim=1).values
    extent_loss = _relative_mse(pred_extent, target_extent)

    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    pred_std = pred_centered.std(dim=1).clamp_min(1e-6)
    target_std = target_centered.std(dim=1).clamp_min(1e-6)
    moment_loss = _relative_mse(pred_std, target_std)

    pred_volume = pred_std.prod(dim=-1)
    target_volume = target_std.prod(dim=-1)
    volume_loss = _relative_mse(pred_volume, target_volume)

    pred_area = _sac_surface_area(pred_verts, faces, mask)
    target_area = _sac_surface_area(target_verts, faces, mask)
    if pred_area is None or target_area is None:
        area_loss = torch.zeros((), device=pred_verts.device, dtype=pred_verts.dtype)
    else:
        area_loss = _relative_mse(pred_area, target_area)
    return extent_loss, area_loss, volume_loss, moment_loss


def _sac_faces(faces, opening_idx, num_verts, device):
    faces = faces.to(device).long()
    sac_mask = _sac_mask(num_verts, opening_idx, device)
    keep = sac_mask[faces].all(dim=-1)
    faces = faces[keep]
    if faces.numel() == 0:
        return None
    return faces


def pouch_surface_chamfer_losses(pred_verts, target_verts, faces, opening_idx, samples, want_normals=False):
    faces = _sac_faces(faces, opening_idx, pred_verts.shape[1], pred_verts.device)
    if faces is None or int(samples) <= 0:
        z = torch.zeros((), device=pred_verts.device, dtype=pred_verts.dtype)
        return z, z

    B = pred_verts.shape[0]
    batched_faces = faces.unsqueeze(0).expand(B, -1, -1)
    pred_mesh = Meshes(verts=pred_verts, faces=batched_faces)
    target_mesh = Meshes(verts=target_verts.detach(), faces=batched_faces)

    if want_normals:
        pred_pts, pred_normals = sample_points_from_meshes(
            pred_mesh, num_samples=int(samples), return_normals=True
        )
        target_pts, target_normals = sample_points_from_meshes(
            target_mesh, num_samples=int(samples), return_normals=True
        )
        dist, normal = chamfer_distance(
            pred_pts, target_pts,
            x_normals=pred_normals,
            y_normals=target_normals,
        )
        return dist, normal

    pred_pts = sample_points_from_meshes(pred_mesh, num_samples=int(samples), return_normals=False)
    target_pts = sample_points_from_meshes(target_mesh, num_samples=int(samples), return_normals=False)
    dist, _ = chamfer_distance(pred_pts, target_pts)
    z = torch.zeros((), device=pred_verts.device, dtype=pred_verts.dtype)
    return dist, z


def run_epoch(model, loader, optimizer, args, device, epoch, train,
              coeff2mesh=None, opening_idx=None,
              ghd_mean=None, ghd_std=None, ring_mean=None, ring_std=None,
              conditioner=None):
    model.train(train)
    totals, mses, kls, scales, verts, normals, rings, ring_chamfers = [], [], [], [], [], [], [], []
    pouch_offsets, pouch_axes, s3_label2s, s3_centers, s3_nearests, s3_sides, s3_opens = [], [], [], [], [], [], []
    shape_extents, shape_areas, shape_volumes, shape_moments = [], [], [], []
    surface_chamfers, surface_normals = [], []
    prior_mses, prior_batch_means, prior_batch_stds = [], [], []
    last_kl_w = kl_weight(args, epoch)
    use_mesh = coeff2mesh is not None and (
        args.w_vert > 0 or args.w_normal > 0 or
        args.w_ring > 0 or args.w_ring_chamfer > 0 or
        args.w_pouch_offset > 0 or args.w_pouch_axis > 0 or
        args.w_stage3_label2 > 0 or args.w_stage3_center > 0 or
        args.w_stage3_nearest > 0 or args.w_stage3_side > 0 or
        args.w_stage3_opening > 0 or args.w_shape_extent > 0 or
        args.w_shape_area > 0 or args.w_shape_volume > 0 or
        args.w_shape_moment > 0 or args.w_surface_chamfer > 0 or
        args.w_surface_normal > 0
    )
    use_normals = use_mesh and (not args.no_normals) and args.w_normal > 0
    for batch in loader:
        x = batch["ghd"].to(device)
        cond = condition_from_batch(
            batch,
            device,
            args.condition_dropout if train else 0.0,
            conditioner=conditioner,
            mode=args.condition_mode,
            use_morphology=bool(args.use_morphology_condition),
        )
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            recon, mu, logvar = model(x, cond, noise_scale=1.0 if train else 0.0)
            mse = F.mse_loss(recon, x)
            kl_raw, kl_train = KL_divergence_terms(mu, logvar, free_bits=args.free_bits)
            kl_capped = torch.clamp(kl_train, max=args.kl_cap)
            loss = args.w_mse * mse + last_kl_w * kl_capped
            prior_mse = torch.zeros((), device=device)
            prior_batch_mean = torch.zeros((), device=device)
            prior_batch_std = torch.zeros((), device=device)
            if args.w_prior_mse > 0 or args.w_prior_batch_mean > 0 or args.w_prior_batch_std > 0:
                z_prior = torch.randn(x.shape[0], model.latent_dim, device=device, dtype=x.dtype)
                z_prior = z_prior * float(args.prior_noise_scale)
                prior = model.decode(z_prior, cond)
                prior_mse = F.huber_loss(prior, x, delta=2.0)
                prior_batch_mean, prior_batch_std = batch_moment_losses(prior, x)
                loss = (
                    loss
                    + args.w_prior_mse * prior_mse
                    + args.w_prior_batch_mean * prior_batch_mean
                    + args.w_prior_batch_std * prior_batch_std
                )
            scale_loss = torch.zeros((), device=device)
            if args.w_scale > 0 and x.shape[-1] > 432 and recon.shape[-1] > 432:
                scale_loss = F.huber_loss(recon[:, 432:433], x[:, 432:433], delta=2.0)
                loss = loss + args.w_scale * scale_loss
            vert_mse = torch.zeros((), device=device)
            normal_mse = torch.zeros((), device=device)
            ring_mse = torch.zeros((), device=device)
            ring_chamfer = torch.zeros((), device=device)
            pouch_offset = torch.zeros((), device=device)
            pouch_axis = torch.zeros((), device=device)
            s3_label2 = torch.zeros((), device=device)
            s3_center = torch.zeros((), device=device)
            s3_nearest = torch.zeros((), device=device)
            s3_side = torch.zeros((), device=device)
            s3_open = torch.zeros((), device=device)
            shape_extent = torch.zeros((), device=device)
            shape_area = torch.zeros((), device=device)
            shape_volume = torch.zeros((), device=device)
            shape_moment = torch.zeros((), device=device)
            surface_chamfer = torch.zeros((), device=device)
            surface_normal = torch.zeros((), device=device)
            if use_mesh:
                _, vert_mse, normal_mse = mesh_recon_losses(
                    coeff2mesh, x, recon, ghd_mean, ghd_std, want_normals=use_normals,
                    use_scale_dim=bool(getattr(args, "withscale", False)),
                    case_rotation=batch.get("alignment_rotation", None).to(device) if batch.get("alignment_rotation", None) is not None else None,
                    case_translation=batch.get("alignment_translation", None).to(device) if batch.get("alignment_translation", None) is not None else None,
                )
                loss = loss + args.w_vert * vert_mse + args.w_normal * normal_mse
                needs_opening_mesh = opening_idx is not None and (
                    args.w_ring > 0 or args.w_ring_chamfer > 0 or
                    args.w_pouch_offset > 0 or args.w_pouch_axis > 0 or
                    args.w_stage3_label2 > 0 or args.w_stage3_center > 0 or
                    args.w_stage3_nearest > 0 or args.w_stage3_side > 0 or
                    args.w_stage3_opening > 0 or args.w_shape_extent > 0 or
                    args.w_shape_area > 0 or args.w_shape_volume > 0 or
                    args.w_shape_moment > 0 or args.w_surface_chamfer > 0 or
                    args.w_surface_normal > 0
                )
                if needs_opening_mesh:
                    if args.w_ring > 0 or args.w_ring_chamfer > 0:
                        pred_verts, _ = coeff2mesh(recon, ghd_mean, ghd_std, want_normals=False)
                        target_ring = batch["ostium_ring"].to(device).reshape(x.shape[0], -1)
                        target_ring = (target_ring * ring_std + ring_mean).view(x.shape[0], args.ring_points, 3)
                        ring_mse, ring_chamfer = opening_ring_losses(pred_verts, target_ring, opening_idx)
                        loss = loss + args.w_ring * ring_mse + args.w_ring_chamfer * ring_chamfer
                    if args.w_pouch_offset > 0 or args.w_pouch_axis > 0:
                        pred_verts, _ = coeff2mesh(
                            recon, ghd_mean, ghd_std, want_normals=False,
                            use_scale_dim=bool(getattr(args, "withscale", False)),
                        )
                        target_verts, _ = coeff2mesh(
                            x, ghd_mean, ghd_std, want_normals=False,
                            use_scale_dim=bool(getattr(args, "withscale", False)),
                        )
                        pouch_offset, pouch_axis = pouch_placement_losses(pred_verts, target_verts, opening_idx)
                        loss = loss + args.w_pouch_offset * pouch_offset + args.w_pouch_axis * pouch_axis
                    if (
                        args.w_stage3_label2 > 0 or args.w_stage3_center > 0 or
                        args.w_stage3_nearest > 0 or args.w_stage3_side > 0 or
                        args.w_stage3_opening > 0
                    ):
                        pred_verts, _ = coeff2mesh(
                            recon, ghd_mean, ghd_std, want_normals=False,
                            use_scale_dim=bool(getattr(args, "withscale", False)),
                        )
                        target_verts, _ = coeff2mesh(
                            x, ghd_mean, ghd_std, want_normals=False,
                            use_scale_dim=bool(getattr(args, "withscale", False)),
                        )
                        s3 = stage3_surrogate_losses(
                            pred_verts,
                            target_verts,
                            opening_idx,
                            batch["label2_pts"].to(device),
                            batch["target_ring_world"].to(device),
                            batch["target_ostium_center"].to(device),
                            batch["target_ostium_normal"].to(device),
                            batch["alignment_rotation"].to(device),
                            batch["alignment_translation"].to(device),
                        )
                        s3_label2 = s3["label2"]
                        s3_center = s3["center"]
                        s3_nearest = s3["nearest"]
                        s3_side = s3["side"]
                        s3_open = s3["opening_center"]
                        loss = (
                            loss
                            + args.w_stage3_label2 * s3_label2
                            + args.w_stage3_center * s3_center
                            + args.w_stage3_nearest * s3_nearest
                            + args.w_stage3_side * s3_side
                            + args.w_stage3_opening * s3_open
                        )
                    if (
                        args.w_shape_extent > 0 or args.w_shape_area > 0 or
                        args.w_shape_volume > 0 or args.w_shape_moment > 0
                    ):
                        pred_verts, _ = coeff2mesh(
                            recon, ghd_mean, ghd_std, want_normals=False,
                            use_scale_dim=bool(getattr(args, "withscale", False)),
                        )
                        target_verts, _ = coeff2mesh(
                            x, ghd_mean, ghd_std, want_normals=False,
                            use_scale_dim=bool(getattr(args, "withscale", False)),
                        )
                        faces = coeff2mesh.canonical.faces_padded()[0]
                        shape_extent, shape_area, shape_volume, shape_moment = sac_shape_losses(
                            pred_verts, target_verts, opening_idx, faces=faces
                        )
                        loss = (
                            loss
                            + args.w_shape_extent * shape_extent
                            + args.w_shape_area * shape_area
                            + args.w_shape_volume * shape_volume
                            + args.w_shape_moment * shape_moment
                        )
                    if args.w_surface_chamfer > 0 or args.w_surface_normal > 0:
                        pred_verts, _ = coeff2mesh(
                            recon, ghd_mean, ghd_std, want_normals=False,
                            use_scale_dim=bool(getattr(args, "withscale", False)),
                        )
                        target_verts, _ = coeff2mesh(
                            x, ghd_mean, ghd_std, want_normals=False,
                            use_scale_dim=bool(getattr(args, "withscale", False)),
                        )
                        faces = coeff2mesh.canonical.faces_padded()[0]
                        surface_chamfer, surface_normal = pouch_surface_chamfer_losses(
                            pred_verts,
                            target_verts,
                            faces,
                            opening_idx,
                            samples=args.surface_samples,
                            want_normals=(args.w_surface_normal > 0),
                        )
                        loss = loss + args.w_surface_chamfer * surface_chamfer + args.w_surface_normal * surface_normal
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        totals.append(float(loss.detach().cpu()))
        mses.append(float(mse.detach().cpu()))
        kls.append(float(kl_raw.detach().cpu()))
        scales.append(float(scale_loss.detach().cpu()))
        prior_mses.append(float(prior_mse.detach().cpu()))
        prior_batch_means.append(float(prior_batch_mean.detach().cpu()))
        prior_batch_stds.append(float(prior_batch_std.detach().cpu()))
        verts.append(float(vert_mse.detach().cpu()))
        normals.append(float(normal_mse.detach().cpu()))
        rings.append(float(ring_mse.detach().cpu()))
        ring_chamfers.append(float(ring_chamfer.detach().cpu()))
        pouch_offsets.append(float(pouch_offset.detach().cpu()))
        pouch_axes.append(float(pouch_axis.detach().cpu()))
        s3_label2s.append(float(s3_label2.detach().cpu()))
        s3_centers.append(float(s3_center.detach().cpu()))
        s3_nearests.append(float(s3_nearest.detach().cpu()))
        s3_sides.append(float(s3_side.detach().cpu()))
        s3_opens.append(float(s3_open.detach().cpu()))
        shape_extents.append(float(shape_extent.detach().cpu()))
        shape_areas.append(float(shape_area.detach().cpu()))
        shape_volumes.append(float(shape_volume.detach().cpu()))
        shape_moments.append(float(shape_moment.detach().cpu()))
        surface_chamfers.append(float(surface_chamfer.detach().cpu()))
        surface_normals.append(float(surface_normal.detach().cpu()))
    prefix = "train" if train else "val"
    return {
        f"{prefix}_total": float(np.mean(totals)),
        f"{prefix}_mse": float(np.mean(mses)),
        f"{prefix}_kl_raw": float(np.mean(kls)),
        f"{prefix}_scale": float(np.mean(scales)),
        f"{prefix}_prior_mse": float(np.mean(prior_mses)),
        f"{prefix}_prior_batch_mean": float(np.mean(prior_batch_means)),
        f"{prefix}_prior_batch_std": float(np.mean(prior_batch_stds)),
        f"{prefix}_vert_mse": float(np.mean(verts)),
        f"{prefix}_normal_mse": float(np.mean(normals)),
        f"{prefix}_ring_mse": float(np.mean(rings)),
        f"{prefix}_ring_chamfer": float(np.mean(ring_chamfers)),
        f"{prefix}_pouch_offset": float(np.mean(pouch_offsets)),
        f"{prefix}_pouch_axis": float(np.mean(pouch_axes)),
        f"{prefix}_stage3_label2": float(np.mean(s3_label2s)),
        f"{prefix}_stage3_center": float(np.mean(s3_centers)),
        f"{prefix}_stage3_nearest": float(np.mean(s3_nearests)),
        f"{prefix}_stage3_side": float(np.mean(s3_sides)),
        f"{prefix}_stage3_opening": float(np.mean(s3_opens)),
        f"{prefix}_shape_extent": float(np.mean(shape_extents)),
        f"{prefix}_shape_area": float(np.mean(shape_areas)),
        f"{prefix}_shape_volume": float(np.mean(shape_volumes)),
        f"{prefix}_shape_moment": float(np.mean(shape_moments)),
        f"{prefix}_surface_chamfer": float(np.mean(surface_chamfers)),
        f"{prefix}_surface_normal": float(np.mean(surface_normals)),
        "kl_w": last_kl_w,
    }


def save_checkpoint(path, epoch, args, model, optimizer, train_ds, val_ds, conditioner=None):
    saved_args = dict(vars(args))
    saved_args["input_dim"] = int(train_ds.get_dim())
    saved_args["morphology_dim"] = int(train_ds.get_morphology_dim()) if getattr(args, "use_morphology_condition", False) else 0
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "conditioner_state_dict": conditioner.state_dict() if conditioner is not None else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "saved_args": saved_args,
        "ghd_mean": train_ds.ghd_mean,
        "ghd_std": train_ds.ghd_std,
        "ostium_mean": train_ds.ostium_mean,
        "ostium_std": train_ds.ostium_std,
        "ostium_ring_mean": train_ds.ostium_ring_mean,
        "ostium_ring_std": train_ds.ostium_ring_std,
        "vessel_center": train_ds.vessel_center,
        "vessel_scale": train_ds.vessel_scale,
        "morphology_mean": getattr(train_ds, "morphology_mean", None),
        "morphology_std": getattr(train_ds, "morphology_std", None),
        "morphology_feature_names": getattr(train_ds, "morphology_feature_names", None),
        "case_names": train_ds.case_names,
        "val_case_names": val_ds.case_names,
    }, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_cases = load_cases(args.train_cases_file)
    val_cases = load_cases(args.val_cases_file)
    train_ds = make_dataset(train_cases, args)
    val_ds = make_dataset(val_cases, args)
    copy_norm(train_ds, val_ds)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    input_dim = train_ds.get_dim()
    if args.condition_mode == "vessel":
        conditioner = build_conditioner(args, device).train()
        cond_dim = int(args.vessel_cond_dim)
    else:
        conditioner = None
        cond_dim = args.ring_points * 3
    if args.use_morphology_condition:
        morph_dim = int(train_ds.get_morphology_dim())
        if morph_dim <= 0:
            raise ValueError("--use_morphology_condition requires --morphology_root with valid morphology files")
        cond_dim += morph_dim
        print(f"morphology condition enabled: dim={morph_dim} keys={getattr(train_ds, 'morphology_feature_names', [])}")
    model = ConditionalGHDVAE(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        cond_dim=cond_dim,
        cond_embed_dim=args.cond_embed_dim,
        norm_type=args.norm_type,
    ).to(device)
    params = list(model.parameters())
    if conditioner is not None:
        params += list(conditioner.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    coeff2mesh = None
    opening_idx = None
    ghd_mean = ghd_std = ring_mean = ring_std = None
    if (
        args.w_vert > 0 or args.w_normal > 0 or args.w_ring > 0 or args.w_ring_chamfer > 0 or
        args.w_pouch_offset > 0 or args.w_pouch_axis > 0 or
        args.w_stage3_label2 > 0 or args.w_stage3_center > 0 or
        args.w_stage3_nearest > 0 or args.w_stage3_side > 0 or
        args.w_stage3_opening > 0 or args.w_shape_extent > 0 or
        args.w_shape_area > 0 or args.w_shape_volume > 0 or
        args.w_shape_moment > 0 or args.w_surface_chamfer > 0 or
        args.w_surface_normal > 0
    ):
        coeff2mesh = CoeffToMesh(
            canonical_mesh_path=args.canonical_mesh_obj,
            eigen_chk=args.eigen_chk,
            num_basis=args.num_basis,
            device=device,
            canonical_norm_factor=args.canonical_norm_factor,
        )
        opening_idx = load_opening_idx(args.canonical_mesh_obj, args.canonical_opa_checkpoint)
        opening_idx = opening_idx.to(device) if opening_idx is not None else None
        ghd_mean = train_ds.ghd_mean.to(device)
        ghd_std = train_ds.ghd_std.to(device)
        ring_mean = train_ds.ostium_ring_mean.to(device)
        ring_std = train_ds.ostium_ring_std.to(device)
        print(
            f"mesh-aware W+: w_vert={args.w_vert} w_normal={args.w_normal} "
            f"w_ring={args.w_ring} w_ring_chamfer={args.w_ring_chamfer} "
            f"w_surface_chamfer={args.w_surface_chamfer} "
            f"w_surface_normal={args.w_surface_normal} "
            f"surface_samples={args.surface_samples} "
            f"opening_idx={0 if opening_idx is None else int(opening_idx.numel())}",
            flush=True,
        )

    run_dir = os.path.join(args.save_root, args.meta)
    os.makedirs(run_dir, exist_ok=False)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    best = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        if conditioner is not None:
            conditioner.train(True)
        tr = run_epoch(model, train_dl, optimizer, args, device, epoch, train=True,
                       coeff2mesh=coeff2mesh, opening_idx=opening_idx,
                       ghd_mean=ghd_mean, ghd_std=ghd_std,
                       ring_mean=ring_mean, ring_std=ring_std,
                       conditioner=conditioner)
        va = None
        if epoch % args.val_freq == 0 or epoch == args.epochs:
            if conditioner is not None:
                conditioner.train(False)
            va = run_epoch(model, val_dl, optimizer, args, device, epoch, train=False,
                           coeff2mesh=coeff2mesh, opening_idx=opening_idx,
                           ghd_mean=ghd_mean, ghd_std=ghd_std,
                           ring_mean=ring_mean, ring_std=ring_std,
                           conditioner=conditioner)
            selected = va[args.early_stop_metric]
            if best is None or selected < best:
                best = selected
                best_epoch = epoch
                save_checkpoint(os.path.join(run_dir, "models_best_val.pth"), epoch, args, model, optimizer, train_ds, val_ds, conditioner)
        if epoch % args.log_freq == 0 or va is not None:
            row = {"epoch": epoch, **tr}
            if va:
                row.update(va)
            print(json.dumps(row), flush=True)
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            save_checkpoint(os.path.join(run_dir, f"models_epoch_{epoch}.pth"), epoch, args, model, optimizer, train_ds, val_ds, conditioner)
    print(f"done best_epoch={best_epoch} best_{args.early_stop_metric}={best}")


if __name__ == "__main__":
    main()
