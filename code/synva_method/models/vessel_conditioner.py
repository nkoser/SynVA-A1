"""
Vessel & Ostium Conditioning for Aneurysm Generation.

Provides:
  - OstiumFeatureExtractor:  offline per-case feature extraction from prepared_meshes_3
  - VesselPointEncoder:      PointNet-style encoder for local vessel surface points
  - OstiumConditioner:       combines ostium plane params + vessel context into condition vector
"""
import os
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
from typing import Tuple, Optional, Dict, List


# ---------------------------------------------------------------------------
# 1.  Offline Feature Extraction (runs once per case, results cached)
# ---------------------------------------------------------------------------

def _as_mesh(mesh):
    if isinstance(mesh, trimesh.Scene):
        return trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def _boundary_edges(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ])
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1]


def _ordered_boundary_loops(faces: np.ndarray) -> List[np.ndarray]:
    edges = _boundary_edges(faces)
    if edges.size == 0:
        return []
    adj = defaultdict(set)
    unused = set()
    for a, b in edges.astype(np.int64):
        a_i, b_i = int(a), int(b)
        adj[a_i].add(b_i)
        adj[b_i].add(a_i)
        unused.add(tuple(sorted((a_i, b_i))))

    loops = []
    while unused:
        start, current = next(iter(unused))
        previous = start
        unused.discard(tuple(sorted((start, current))))
        loop = [start, current]
        while True:
            candidates = [
                nb for nb in sorted(adj.get(current, ()))
                if nb != previous and tuple(sorted((current, nb))) in unused
            ]
            if not candidates:
                break
            nxt = candidates[0]
            unused.discard(tuple(sorted((current, nxt))))
            if nxt == start:
                break
            loop.append(nxt)
            previous, current = current, nxt
        if len(loop) >= 3:
            loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def _order_points_by_angle(points: np.ndarray, center: np.ndarray, normal: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    normal = np.asarray(normal, dtype=np.float64).reshape(3)
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref, normal))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(normal, ref)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(normal, u)
    rel = points - center.reshape(1, 3)
    angles = np.arctan2(rel @ v, rel @ u)
    return points[np.argsort(angles)]


def _vessel_boundary_ring(vessel_mesh: trimesh.Trimesh, centroid: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vessel_mesh.vertices, dtype=np.float64)
    loops = _ordered_boundary_loops(np.asarray(vessel_mesh.faces, dtype=np.int64))
    if not loops:
        raise ValueError("vessel_submesh.obj has no open boundary loop")
    centroid = np.asarray(centroid, dtype=np.float64).reshape(3)
    loop = min(loops, key=lambda idx: float(np.linalg.norm(vertices[idx].mean(axis=0) - centroid)))
    return vertices[loop].astype(np.float32)

class OstiumFeatureExtractor:
    """
    Given a case directory in prepared_meshes_3, extract:
      - ostium_centroid  [3]
      - ostium_normal    [3]
      - ostium_radius    [1]   mean dist of ostium verts from centroid
      - ostium_ecc       [1]   eccentricity (ratio of PCA eigenvalues)
      - vessel_local_pts [N, 3] vessel surface samples near the ostium
    """

    def __init__(
        self,
        data_root: str,
        num_vessel_pts: int = 256,
        radius_factor: float = 3.0,
        ostium_source: str = "vessel_boundary",
    ):
        self.data_root = data_root
        self.num_vessel_pts = num_vessel_pts
        self.radius_factor = radius_factor
        if ostium_source not in ("vessel_boundary", "label2", "label1"):
            raise ValueError("ostium_source must be 'vessel_boundary', 'label2', or 'label1'")
        self.ostium_source = ostium_source

    def extract_case(self, case_name: str) -> Dict[str, np.ndarray]:
        case_dir = os.path.join(self.data_root, case_name)
        centroid = np.load(os.path.join(case_dir, '07_other', 'centroid_ostium.npy')).astype(np.float32)
        normal   = np.load(os.path.join(case_dir, '07_other', 'normal_vector.npy')).astype(np.float32)

        vessel_mesh = _as_mesh(trimesh.load(
            os.path.join(case_dir, '05_submeshes', 'vessel_submesh.obj'), process=False
        ))

        if self.ostium_source == "vessel_boundary":
            ostium_ring = _vessel_boundary_ring(vessel_mesh, centroid)
            ostium_verts = ostium_ring
        else:
            label_value = 2 if self.ostium_source == "label2" else 1
            full_mesh = _as_mesh(trimesh.load(os.path.join(case_dir, '01_mesh', 'mesh.obj'), process=False))
            labels = np.load(os.path.join(case_dir, '02_labels', 'labels.npy'))
            ostium_verts = np.array(full_mesh.vertices[labels == label_value], dtype=np.float32)
            if len(ostium_verts) >= 3:
                ostium_ring = _order_points_by_angle(ostium_verts, centroid, normal).astype(np.float32)
            else:
                ostium_ring = ostium_verts

        # Ostium radius & eccentricity
        if len(ostium_verts) >= 3:
            dists = np.linalg.norm(ostium_verts - centroid, axis=1)
            ostium_radius = float(dists.mean())
            # PCA for eccentricity
            centered = ostium_verts - centroid
            try:
                _, svals, _ = np.linalg.svd(centered, full_matrices=False)
                ecc = float(svals[1] / (svals[0] + 1e-8))
            except np.linalg.LinAlgError:
                ecc = 1.0
        else:
            ostium_radius = 0.1
            ecc = 1.0

        # Vessel submesh: sample points near the ostium
        vessel_verts = np.array(vessel_mesh.vertices, dtype=np.float32)
        vessel_dists = np.linalg.norm(vessel_verts - centroid, axis=1)
        cutoff = self.radius_factor * ostium_radius
        near_mask = vessel_dists < cutoff
        local_verts = vessel_verts[near_mask]

        # Sub-/over-sample to fixed size
        if len(local_verts) == 0:
            local_verts = vessel_verts  # fallback: use all
        if len(local_verts) >= self.num_vessel_pts:
            idx = np.random.choice(len(local_verts), self.num_vessel_pts, replace=False)
        else:
            idx = np.random.choice(len(local_verts), self.num_vessel_pts, replace=True)
        vessel_local_pts = local_verts[idx]

        return {
            'ostium_centroid':   centroid,                                    # [3]
            'ostium_normal':     normal / (np.linalg.norm(normal) + 1e-8),   # [3]
            'ostium_radius':     np.array([ostium_radius], dtype=np.float32),# [1]
            'ostium_ecc':        np.array([ecc], dtype=np.float32),          # [1]
            'ostium_verts':      ostium_verts,                                # [N_ostium, 3]
            'ostium_ring_verts': ostium_ring,                                  # ordered [N_ring, 3]
            'vessel_local_pts':  vessel_local_pts,                            # [N, 3]
        }

    def extract_all(self, case_names: List[str], verbose: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
        all_features = {}
        skipped = 0
        for i, case in enumerate(case_names):
            try:
                all_features[case] = self.extract_case(case)
            except Exception as e:
                if verbose:
                    print(f"[OstiumFeatureExtractor] skip {case}: {e}")
                skipped += 1
        if verbose:
            print(
                f"Extracted ostium features for {len(all_features)}/{len(case_names)} cases "
                f"(skipped {skipped}, source={self.ostium_source})"
            )
        return all_features


# ---------------------------------------------------------------------------
# 2.  PointNet-style Vessel Encoder (for local vessel points)
# ---------------------------------------------------------------------------

class VesselPointEncoder(nn.Module):
    """
    Simple PointNet: per-point MLP → max-pool → global feature.
    Input:  [B, N, 3]
    Output: [B, feat_dim]
    """

    def __init__(self, feat_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feat_dim),
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """pts: [B, N, 3] → [B, feat_dim]"""
        h = self.mlp(pts)          # [B, N, feat_dim]
        return h.max(dim=1)[0]     # [B, feat_dim]


# ---------------------------------------------------------------------------
# 3.  Ostium Conditioner — produces the full condition vector
# ---------------------------------------------------------------------------

class OstiumConditioner(nn.Module):
    """
    Combines:
      - ostium plane params (centroid, normal, radius, eccentricity) → MLP → [B, ostium_feat_dim]
      - vessel local points → VesselPointEncoder → [B, vessel_feat_dim]
      - (optional) ostium ring point cloud → VesselPointEncoder → [B, ring_feat_dim]
    Fuses into a single condition vector  [B, cond_out_dim].
    """

    def __init__(
        self,
        vessel_feat_dim: int = 64,
        ostium_plane_dim: int = 8,
        ostium_feat_dim: int = 16,
        cond_out_dim: int = 32,
        use_ring_pts: bool = False,
        ring_feat_dim: int = 32,
        use_ordered_ring: bool = False,
        ring_points: int = 20,
        ordered_ring_feat_dim: int = 64,
    ):
        super().__init__()
        self.vessel_encoder = VesselPointEncoder(feat_dim=vessel_feat_dim)

        # ostium plane: centroid(3) + normal(3) + radius(1) + ecc(1) = 8
        self.ostium_plane_encoder = nn.Sequential(
            nn.Linear(ostium_plane_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, ostium_feat_dim),
        )

        self.use_ring_pts = bool(use_ring_pts)
        self.use_ordered_ring = bool(use_ordered_ring)
        self.ring_points = int(ring_points)
        if self.use_ring_pts:
            self.ring_encoder = VesselPointEncoder(feat_dim=ring_feat_dim)
            fuse_in = vessel_feat_dim + ostium_feat_dim + ring_feat_dim
        else:
            self.ring_encoder = None
            fuse_in = vessel_feat_dim + ostium_feat_dim

        if self.use_ordered_ring:
            self.ordered_ring_encoder = nn.Sequential(
                nn.Linear(self.ring_points * 3, ordered_ring_feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(ordered_ring_feat_dim, ordered_ring_feat_dim),
                nn.ReLU(inplace=True),
            )
            fuse_in += ordered_ring_feat_dim
        else:
            self.ordered_ring_encoder = None

        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, cond_out_dim),
            nn.ReLU(inplace=True),
        )
        self.cond_out_dim = cond_out_dim

    def forward(
        self,
        vessel_pts: torch.Tensor,
        ostium_params: torch.Tensor,
        ostium_pts: torch.Tensor = None,
        ostium_ring: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        vessel_pts:    [B, N, 3]
        ostium_params: [B, 8]
        ostium_pts:    [B, K, 3]  (only used if use_ring_pts=True)
        ostium_ring:   [B, ring_points, 3] ordered ring condition
        returns:       [B, cond_out_dim]
        """
        v_feat = self.vessel_encoder(vessel_pts)           # [B, vessel_feat_dim]
        o_feat = self.ostium_plane_encoder(ostium_params)  # [B, ostium_feat_dim]
        feats = [v_feat, o_feat]
        if self.use_ring_pts:
            if ostium_pts is None:
                raise ValueError("OstiumConditioner(use_ring_pts=True) requires ostium_pts")
            r_feat = self.ring_encoder(ostium_pts)         # [B, ring_feat_dim]
            feats.append(r_feat)
        if self.use_ordered_ring:
            if ostium_ring is None:
                raise ValueError("OstiumConditioner(use_ordered_ring=True) requires ostium_ring")
            if ostium_ring.shape[1] != self.ring_points:
                raise ValueError(
                    f"Expected ostium_ring with {self.ring_points} points, got {ostium_ring.shape[1]}"
                )
            ordered_feat = self.ordered_ring_encoder(ostium_ring.reshape(ostium_ring.shape[0], -1))
            feats.append(ordered_feat)
        return self.fuse(torch.cat(feats, dim=-1))
