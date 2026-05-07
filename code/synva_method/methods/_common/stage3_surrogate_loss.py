from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix


def _select_ordered_ring(verts: torch.Tensor, opening_idx: torch.Tensor, ring_points: int) -> torch.Tensor:
    idx = opening_idx.to(verts.device).long()
    idx = idx[(idx >= 0) & (idx < verts.shape[1])]
    if idx.numel() < 3:
        raise ValueError("opening_idx must contain at least three valid vertices")
    ring = verts[:, idx, :]
    if ring.shape[1] != ring_points:
        sel = torch.linspace(0, ring.shape[1] - 1, ring_points, device=verts.device).round().long()
        ring = ring[:, sel, :]
    return ring


def _sac_mask(num_verts: int, opening_idx: torch.Tensor, device: torch.device) -> torch.Tensor:
    idx = opening_idx.to(device).long()
    idx = idx[(idx >= 0) & (idx < num_verts)]
    mask = torch.ones(num_verts, dtype=torch.bool, device=device)
    mask[idx] = False
    return mask


def _umeyama_align(source: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """Return scale, rotation, translation mapping source -> target.

    Uses row-vector convention: aligned = source @ R * scale + t.
    """
    src_mean = source.mean(dim=1, keepdim=True)
    tgt_mean = target.mean(dim=1, keepdim=True)
    src = source - src_mean
    tgt = target - tgt_mean
    cov = torch.matmul(src.transpose(1, 2), tgt) / max(1, source.shape[1])
    U, S, Vh = torch.linalg.svd(cov, full_matrices=False)
    R = torch.matmul(Vh.transpose(-2, -1), U.transpose(-2, -1))
    det = torch.det(R)
    fix = torch.ones((R.shape[0], 3), dtype=R.dtype, device=R.device)
    fix[:, -1] = torch.where(det < 0, -1.0, 1.0)
    Sfix = torch.diag_embed(fix)
    R = torch.matmul(torch.matmul(Vh.transpose(-2, -1), Sfix), U.transpose(-2, -1))
    var = (src ** 2).sum(dim=(1, 2)) / max(1, source.shape[1])
    scale = ((S * fix).sum(dim=1) / (var + eps)).clamp(min=0.05, max=20.0)
    t = tgt_mean.squeeze(1) - scale[:, None] * torch.matmul(src_mean.squeeze(1).unsqueeze(1), R).squeeze(1)
    return scale, R, t


def _apply_similarity(points: torch.Tensor, scale: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor):
    return points @ rotation * scale[:, None, None] + translation[:, None, :]


def _softmin_distance(points: torch.Tensor, center: torch.Tensor, tau: float = 0.01) -> torch.Tensor:
    dist = torch.linalg.norm(points - center[:, None, :], dim=-1)
    weights = torch.softmax(-dist / float(tau), dim=1)
    return (weights * dist).sum(dim=1)


def _local_to_world(
    verts_local_scaled: torch.Tensor,
    case_rotation: torch.Tensor,
    case_translation: torch.Tensor,
    ostium_center: torch.Tensor,
) -> torch.Tensor:
    rotation = axis_angle_to_matrix(case_rotation.to(verts_local_scaled.device))
    verts = verts_local_scaled @ rotation.transpose(-1, -2) + case_translation.to(verts_local_scaled.device).unsqueeze(1)
    return verts + ostium_center.to(verts.device).unsqueeze(1)


def stage3_surrogate_losses(
    pred_verts_local_scaled: torch.Tensor,
    target_verts_local_scaled: torch.Tensor,
    opening_idx: torch.Tensor,
    label2_pts: torch.Tensor,
    target_ring_world: torch.Tensor,
    ostium_center: torch.Tensor,
    ostium_normal: torch.Tensor,
    case_rotation: torch.Tensor,
    case_translation: torch.Tensor,
):
    """Differentiable approximation of the official Step3 placement metrics.

    The generated mesh is first moved to the case frame, then ring-fit to the
    target OPA ring.  Losses are computed after that fit, mirroring the geometry
    that Step3 later evaluates.
    """
    device = pred_verts_local_scaled.device
    ring_points = int(target_ring_world.shape[1])
    label2_pts = label2_pts.to(device)
    target_ring_world = target_ring_world.to(device)
    ostium_center = ostium_center.to(device)
    ostium_normal = F.normalize(ostium_normal.to(device), dim=-1, eps=1e-8)

    pred_world = _local_to_world(pred_verts_local_scaled, case_rotation, case_translation, ostium_center)
    target_world = _local_to_world(target_verts_local_scaled, case_rotation, case_translation, ostium_center)

    pred_ring = _select_ordered_ring(pred_world, opening_idx, ring_points)
    target_gt_ring = _select_ordered_ring(target_world, opening_idx, ring_points)
    s_pred, r_pred, t_pred = _umeyama_align(pred_ring, target_ring_world)
    s_gt, r_gt, t_gt = _umeyama_align(target_gt_ring, target_ring_world)
    pred_aligned = _apply_similarity(pred_world, s_pred, r_pred, t_pred)
    target_aligned = _apply_similarity(target_world, s_gt, r_gt, t_gt)

    mask = _sac_mask(pred_aligned.shape[1], opening_idx, device)
    pred_sac = pred_aligned[:, mask, :]
    target_sac = target_aligned[:, mask, :]

    d = torch.cdist(label2_pts, pred_sac)
    label2_to_pouch = d.min(dim=2).values.mean()

    pred_center = pred_sac.mean(dim=1)
    target_center = target_sac.mean(dim=1)
    center = F.mse_loss(pred_center, target_center)

    pred_nearest = _softmin_distance(pred_sac, ostium_center)
    target_nearest = _softmin_distance(target_sac, ostium_center).detach()
    nearest = F.smooth_l1_loss(pred_nearest, target_nearest, beta=0.01)

    target_rel = target_center - ostium_center
    pred_rel = pred_center - ostium_center
    target_sign = torch.sign((target_rel * ostium_normal).sum(dim=-1, keepdim=True)).clamp(min=-1.0, max=1.0)
    target_sign = torch.where(target_sign == 0, torch.ones_like(target_sign), target_sign)
    signed_pred = ((pred_rel * ostium_normal).sum(dim=-1, keepdim=True) * target_sign).squeeze(-1)
    signed_target = ((target_rel * ostium_normal).sum(dim=-1, keepdim=True) * target_sign).squeeze(-1).detach()
    side = F.relu(0.25 * signed_target - signed_pred).mean()

    opening_center = F.mse_loss(pred_aligned[:, mask.logical_not(), :].mean(dim=1), target_ring_world.mean(dim=1))

    return {
        "label2": label2_to_pouch,
        "center": center,
        "nearest": nearest,
        "side": side,
        "opening_center": opening_center,
    }
