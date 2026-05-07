"""Vessel-Collision Loss for aneurysm generation (Experiment E).

All inputs live in the SAME `ghd_local` frame:
  - pred_verts  : decoded sac vertices, [B, V, 3]
  - vessel_pts  : surrounding vessel surface points, [B, N, 3]
                  (already in ghd_local because condition_space="ghd_local")
  - opening_idx : indices of the ostium ring on the canonical mesh, [N_ring]

Variant (a) — hinge-clearance:
    For each "sac" vertex (excluding ostium ring + a tubular neighbourhood
    near the ostium centroid), penalise its distance to the nearest vessel
    point being smaller than `clearance`.

        L = mean( relu(clearance - d_min)^2 )

This is the simplest, most robust starting point.  Variant (b) signed-
penetration via the ostium plane can be added later without touching the
existing trained pipeline.
"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_min_dist(a: torch.Tensor, b: torch.Tensor,
                      chunk: int = 4096) -> torch.Tensor:
    """For each point in a [B, M, 3] return min L2 dist to any point in b [B, N, 3].

    Chunked over M to keep memory bounded.  Returns [B, M].
    """
    B, M, _ = a.shape
    out = a.new_empty(B, M)
    for s in range(0, M, chunk):
        e = min(M, s + chunk)
        # [B, m, 1, 3] - [B, 1, N, 3] -> [B, m, N]
        d2 = ((a[:, s:e, :].unsqueeze(2) - b.unsqueeze(1)) ** 2).sum(-1)
        out[:, s:e] = d2.clamp_min(1e-12).sqrt().min(dim=-1).values
    return out


def build_sac_mask(num_verts: int, opening_idx: torch.Tensor,
                   ring_neighbour_pad: int = 0) -> torch.Tensor:
    """Boolean mask [V] selecting "sac" verts (i.e. not the ostium ring).

    `ring_neighbour_pad` is left at 0 here; if a wider exclusion is desired,
    the caller can union opening_idx with neighbouring vertex indices on the
    canonical mesh before calling this function.
    """
    mask = torch.ones(num_verts, dtype=torch.bool, device=opening_idx.device)
    mask[opening_idx.long()] = False
    return mask


class VesselCollisionLoss(nn.Module):
    """Hinge-clearance loss between sac vertices and surrounding vessel pts."""

    def __init__(self,
                 clearance: float = 0.04,
                 sac_mask: Optional[torch.Tensor] = None,
                 chunk: int = 4096,
                 reduction: str = "mean"):
        """
        Args:
            clearance: target minimum distance in `ghd_local` units.
                       Typical aneurysm radii in this frame are ~0.10-0.30,
                       so a clearance of 0.02-0.06 is a reasonable start.
            sac_mask:  optional bool mask [V] of sac vertices.  If None, all
                       verts are used.
            chunk:     chunk size for pairwise distance computation.
            reduction: "mean" or "sum".
        """
        super().__init__()
        self.clearance = float(clearance)
        self.chunk = int(chunk)
        self.reduction = reduction
        if sac_mask is not None:
            self.register_buffer("sac_mask", sac_mask.bool(), persistent=False)
        else:
            self.sac_mask = None

    def forward(self, pred_verts: torch.Tensor,
                vessel_pts: torch.Tensor,
                return_stats: bool = False
                ) -> Tuple[torch.Tensor, dict]:
        """
        pred_verts: [B, V, 3]  decoded sac mesh in ghd_local frame
        vessel_pts: [B, N, 3]  surrounding vessel pts in ghd_local frame
        """
        if self.sac_mask is not None:
            verts = pred_verts[:, self.sac_mask, :]
        else:
            verts = pred_verts
        d = pairwise_min_dist(verts, vessel_pts, chunk=self.chunk)  # [B, M]
        gap = (self.clearance - d).clamp_min(0.0)                   # >=0
        loss = (gap ** 2)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(self.reduction)
        if return_stats:
            with torch.no_grad():
                df = d.flatten()
                if df.numel() > 0:
                    k = max(1, int(0.05 * df.numel()))
                    p05 = df.kthvalue(k).values.item()
                else:
                    p05 = 0.0
                stats = {
                    "min_dist_mean": d.mean().item(),
                    "min_dist_p05": p05,
                    "violators_frac": (d < self.clearance).float().mean().item(),
                }
            return loss, stats
        return loss, {}


def add_collision_args(p):
    p.add_argument("--w_collision", type=float, default=50.0,
                   help="Weight on vessel-collision hinge loss.")
    p.add_argument("--collision_clearance", type=float, default=0.04,
                   help="Target min distance (ghd_local units) sac->vessel.")
    p.add_argument("--collision_phase_in", type=int, default=200,
                   help="Epoch at which collision loss starts (0=immediate).")
    p.add_argument("--collision_ramp", type=int, default=200,
                   help="Linear ramp length in epochs after phase-in.")
    p.add_argument("--collision_chunk", type=int, default=4096,
                   help="Chunk size for pairwise distance.")
    p.add_argument("--exclude_ring_from_collision", action="store_true",
                   default=True,
                   help="Exclude ostium-ring verts from the sac mask.")


def collision_weight(epoch: int, args) -> float:
    """Linear phase-in/ramp scheduler."""
    if epoch < args.collision_phase_in:
        return 0.0
    if args.collision_ramp <= 0:
        return float(args.w_collision)
    t = (epoch - args.collision_phase_in) / float(args.collision_ramp)
    return float(args.w_collision) * min(1.0, max(0.0, t))


def load_opening_idx(canonical_mesh_path: str,
                     opa_checkpoint: str = "/path/to/SynVA-A1/checkpoints/canonical_average/opa_checkpoint_1op.pkl"
                     ) -> Optional[torch.Tensor]:
    """Try to load opening-ring vertex indices used during GHD fitting.

    Returns None if not found; caller will then use all sac verts.
    """
    import os, pickle
    if not os.path.isfile(opa_checkpoint):
        return None
    try:
        with open(opa_checkpoint, "rb") as f:
            data = pickle.load(f)
    except Exception:
        return None
    # Heuristic: walk dict for ostium ring vertex indices.
    # AneuG's opa_checkpoint_1op.pkl stores them under "op_v_indices" as a
    # list-of-lists (one entry per opening).  Older naming variants also tried.
    keys = ("opening_idx", "opening_indices", "ostium_idx", "ring_idx",
            "op_v_indices")
    if isinstance(data, dict):
        for k in keys:
            if k in data:
                v = data[k]
                # Unwrap [[idx_array]] -> idx_array
                if isinstance(v, list) and len(v) >= 1:
                    v = v[0]
                if torch.is_tensor(v):
                    return v.long().flatten()
                try:
                    return torch.as_tensor(v, dtype=torch.long).flatten()
                except Exception:
                    continue
    return None
