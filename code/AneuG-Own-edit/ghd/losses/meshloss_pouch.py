import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points, sample_points_from_meshes
from pytorch3d.structures import Meshes

from ghd.base.mesh_geometry3 import Winding_Occupancy
from ghd.losses.meshloss_oa import Mesh_loss_opening_alignment


def _mesh_surface_area(mesh: Meshes) -> torch.Tensor:
    verts = mesh.verts_packed()
    faces = mesh.faces_packed().long()
    tri = verts[faces]  # [F, 3, 3]
    cross = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    return 0.5 * torch.norm(cross, dim=1).sum()


def _mesh_abs_volume(mesh: Meshes) -> torch.Tensor:
    """Absolute enclosed volume from triangle mesh via signed tetrahedra sum."""
    verts = mesh.verts_packed()
    faces = mesh.faces_packed().long()
    tri = verts[faces]  # [F, 3, 3]
    signed_6v = torch.sum(tri[:, 0] * torch.cross(tri[:, 1], tri[:, 2], dim=1), dim=1)
    signed_v = signed_6v.sum() / 6.0
    return torch.abs(signed_v)


def _mesh_mean_unit_normal(mesh: Meshes) -> torch.Tensor:
    """Area-weighted mean normal (unit vector) from triangle faces."""
    verts = mesh.verts_packed()
    faces = mesh.faces_packed().long()
    tri = verts[faces]  # [F, 3, 3]
    face_normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    mean_n = face_normals.sum(dim=0)
    return mean_n / (torch.norm(mean_n) + 1e-12)


def _ring_mean_unit_normal(ring_verts: torch.Tensor) -> torch.Tensor:
    """Unit normal from ordered ring vertices using a polygon/Newell-style accumulation."""
    if ring_verts.shape[0] < 3:
        return torch.tensor([0.0, 0.0, 1.0], device=ring_verts.device, dtype=ring_verts.dtype)
    center = ring_verts.mean(dim=0, keepdim=True)
    v0 = ring_verts - center
    v1 = torch.roll(v0, shifts=-1, dims=0)
    n = torch.cross(v0, v1, dim=1).sum(dim=0)
    return n / (torch.norm(n) + 1e-12)


def _pointset_overlap_score(points_a: torch.Tensor, points_b: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """Symmetric soft-overlap score in [0, 1] based on nearest-neighbor distances."""
    dist2_ab = knn_points(points_a, points_b, K=1).dists[..., 0]
    dist2_ba = knn_points(points_b, points_a, K=1).dists[..., 0]
    score_ab = torch.exp(-dist2_ab / (sigma2 + 1e-12)).mean()
    score_ba = torch.exp(-dist2_ba / (sigma2 + 1e-12)).mean()
    return 0.5 * (score_ab + score_ba)


def _build_uniform_grid_points_between_meshes(
    mesh_a: Meshes,
    mesh_b: Meshes,
    dpi: int,
    expand_ratio: float = 0.125,
    max_points: int = 12000,
) -> torch.Tensor:
    bbox_a = mesh_a.get_bounding_boxes().squeeze(0)
    bbox_b = mesh_b.get_bounding_boxes().squeeze(0)
    box_min = torch.min(torch.stack([bbox_a[:, 0], bbox_b[:, 0]], dim=1), dim=1).values
    box_max = torch.max(torch.stack([bbox_a[:, 1], bbox_b[:, 1]], dim=1), dim=1).values
    extent = box_max - box_min
    box_min = box_min - expand_ratio * extent
    box_max = box_max + expand_ratio * extent

    dpi = int(max(4, dpi))
    min_extent = torch.clamp(extent.min(), min=1e-8)
    steps = ((extent / min_extent) * float(dpi)).round().long() + 1
    steps = torch.clamp(steps, min=2)

    x = torch.linspace(box_min[0], box_max[0], int(steps[0].item()), device=mesh_a.device)
    y = torch.linspace(box_min[1], box_max[1], int(steps[1].item()), device=mesh_a.device)
    z = torch.linspace(box_min[2], box_max[2], int(steps[2].item()), device=mesh_a.device)
    gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
    points = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)

    if int(max_points) > 0 and points.shape[0] > int(max_points):
        keep_ids = torch.linspace(
            0,
            points.shape[0] - 1,
            steps=int(max_points),
            device=points.device,
        ).long()
        points = points[keep_ids]
    return points


def _soft_winding_occupancy(
    mesh: Meshes,
    points: torch.Tensor,
    chunk_size: int = 1024,
    use_abs_winding: bool = True,
) -> torch.Tensor:
    chunk_size = int(max(1, chunk_size))
    out = []
    for i in range(0, points.shape[0], chunk_size):
        q = points[i : i + chunk_size]
        winding = Winding_Occupancy(mesh, q)
        if use_abs_winding:
            winding = winding.abs()
        occ = torch.sigmoid((winding - 0.5) * 100.0)
        out.append(occ)
    return torch.cat(out, dim=0) if out else torch.empty((0,), device=points.device)


def _build_vertex_adjacency(num_verts: int, faces: torch.Tensor):
    adj = [set() for _ in range(num_verts)]
    for tri in faces.tolist():
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        adj[i].add(j); adj[i].add(k)
        adj[j].add(i); adj[j].add(k)
        adj[k].add(i); adj[k].add(j)
    return adj


def _k_ring_from_seed(seed_idx, adjacency, depth: int):
    visited = set(int(x) for x in seed_idx)
    frontier = set(visited)
    for _ in range(max(0, depth)):
        nxt = set()
        for v in frontier:
            nxt.update(adjacency[v])
        nxt -= visited
        if not nxt:
            break
        visited |= nxt
        frontier = nxt
    return sorted(visited)


class Mesh_loss_pouch_only(Mesh_loss_opening_alignment):
    """Pouch-only loss:
    - uses mesh + opening alignment losses
    - adds opening-area barrier to prevent ostium collapse
    - adds opening-overlap loss (0 = perfect overlap, 1 = no overlap)
    - adds optional 3D-grid occupancy loss (inside/outside consistency)
    """

    def __init__(self, args, oa_class_canonical, oa_class_target):
        super().__init__(args, oa_class_canonical, oa_class_target)
        self.opening_min_ratio = float(getattr(args, "opening_min_ratio", 0.5))
        self.opening_overlap_sigma_ratio = float(getattr(args, "opening_overlap_sigma_ratio", 0.1))
        self.opening_area_eps = 1e-8
        self.volume_eps = 1e-8
        self.grid_occupancy_dpi = int(getattr(args, "grid_occupancy_dpi", 18))
        self.grid_occupancy_max_points = int(getattr(args, "grid_occupancy_max_points", 12000))
        self.grid_occupancy_samples_per_step = int(getattr(args, "grid_occupancy_samples_per_step", 2048))
        self.grid_occupancy_chunk_size = int(getattr(args, "grid_occupancy_chunk_size", 1024))
        self.grid_occupancy_loss_type = str(getattr(args, "grid_occupancy_loss_type", "mse")).lower()
        self._grid_occ_points = None
        self._grid_occ_target = None
        # In pouch_only, the dynamic opening is warped_openings[0] (args.num_op=1).
        # Target checkpoint opening order can vary, so match target opening by minimum Chamfer distance.
        self.pouch_target_opening_idx = 0
        canonical_openings = oa_class_canonical.return_opening_Meshes_static(register_normal=False)
        best_idx = 0
        best_loss = None
        pcd_c = sample_points_from_meshes(
            canonical_openings[0].to(self.device), self.op_sample_num, return_normals=False
        )
        for idx, tgt_open in enumerate(self.target_openings):
            pcd_t = sample_points_from_meshes(tgt_open.to(self.device), self.op_sample_num, return_normals=False)
            loss_p, _ = chamfer_distance(pcd_c, pcd_t)
            loss_val = float(loss_p.detach().cpu().item())
            if (best_loss is None) or (loss_val < best_loss):
                best_loss = loss_val
                best_idx = idx
        self.pouch_target_opening_idx = best_idx
        print(
            f"[pouch_only] matched target opening index = {self.pouch_target_opening_idx} "
            f"(min opening-surface Chamfer = {best_loss:.6f})"
        )

        self.reference_opening_areas = []
        for op in oa_class_canonical.return_opening_Meshes_static(register_normal=False):
            self.reference_opening_areas.append(_mesh_surface_area(op.to(self.device)).detach())
        self.target_volume = _mesh_abs_volume(self.target_mesh).detach()
        # Keep direct handle for selected target opening used in original opening losses.
        self.target_opening_selected = self.target_openings[self.pouch_target_opening_idx].to(self.device)
        # Boundary smoothness setup (ordered opening ring from checkpoint mapping).
        self.opening_boundary_smooth_width = int(getattr(args, "opening_boundary_smooth_width", 3))
        self.opening_boundary_indices = torch.tensor(
            getattr(oa_class_canonical, "op_rec_v_indices_map")[0], dtype=torch.long, device=self.device
        )
        faces_full = getattr(oa_class_canonical, "mesh_target_p3d").faces_packed().long().detach().cpu()
        num_verts_full = int(getattr(oa_class_canonical, "mesh_target_p3d").verts_packed().shape[0])
        self._full_adjacency = _build_vertex_adjacency(num_verts_full, faces_full)
        smooth_region = _k_ring_from_seed(
            self.opening_boundary_indices.detach().cpu().tolist(),
            self._full_adjacency,
            depth=max(0, self.opening_boundary_smooth_width - 1),
        )
        self.opening_smooth_region_indices = torch.tensor(smooth_region, dtype=torch.long, device=self.device)
        # Precompute local neighbor ids within smooth region for umbrella Laplacian smoothing.
        smooth_region_set = set(int(x) for x in smooth_region)
        self._smooth_local_neighbors = {}
        for vid in smooth_region:
            loc_nbr = [n for n in self._full_adjacency[int(vid)] if n in smooth_region_set]
            self._smooth_local_neighbors[int(vid)] = loc_nbr

    def _ensure_grid_occupancy_cache(self):
        if self._grid_occ_points is not None and self._grid_occ_target is not None:
            return
        self._grid_occ_points = _build_uniform_grid_points_between_meshes(
            self.mesh_std,
            self.target_mesh,
            dpi=self.grid_occupancy_dpi,
            max_points=self.grid_occupancy_max_points,
        ).to(self.device)
        self._grid_occ_target = _soft_winding_occupancy(
            self.target_mesh,
            self._grid_occ_points,
            chunk_size=self.grid_occupancy_chunk_size,
            use_abs_winding=True,
        ).detach()
        print(
            "[pouch_only] grid occupancy cache: points={} dpi={} max_points={}".format(
                int(self._grid_occ_points.shape[0]),
                int(self.grid_occupancy_dpi),
                int(self.grid_occupancy_max_points),
            )
        )

    def forward_pouch_only(self, warped_mesh, warped_openings, loss_weighting: dict, B=1):
        # Base geometric losses
        loss_dict = self.forward(meshes_scr=warped_mesh, trg=self.target_mesh, loss_list=loss_weighting, B=B)
        # Opening losses (single opening): Chamfer position + Chamfer normal term.
        if ("loss_openings_p" in loss_weighting) or ("loss_openings_n" in loss_weighting):
            loss_p_list = []
            loss_n_list = []
            if len(warped_openings) > 0 and self.target_opening_selected is not None:
                pcd_wo = sample_points_from_meshes(
                    warped_openings[0], self.op_sample_num, return_normals=False
                )
                pcd_to = sample_points_from_meshes(
                    self.target_opening_selected, self.op_sample_num, return_normals=False
                )
                loss_p, _ = chamfer_distance(pcd_wo, pcd_to)

                # Robust normal alignment from ordered opening ring vertices.
                # This avoids unstable face-normal sampling on triangulated opening patches.
                n_w = _ring_mean_unit_normal(warped_openings[0].verts_packed())
                n_t = _ring_mean_unit_normal(self.target_opening_selected.verts_packed())
                cos_sim = torch.sum(n_w * n_t)
                loss_n = 1.0 - torch.abs(cos_sim)

                loss_p_list.append(loss_p if not torch.isnan(loss_p) else torch.tensor(0.0, device=self.device))
                loss_n_list.append(loss_n if not torch.isnan(loss_n) else torch.tensor(0.0, device=self.device))
            else:
                loss_p_list.append(torch.tensor(0.0, device=self.device))
                loss_n_list.append(torch.tensor(0.0, device=self.device))
            if "loss_openings_p" in loss_weighting:
                loss_dict["loss_openings_p"] = loss_p_list
            if "loss_openings_n" in loss_weighting:
                loss_dict["loss_openings_n"] = loss_n_list
        if "loss_opening_area" in loss_weighting:
            area_losses = []
            for idx, op_mesh in enumerate(warped_openings):
                area_now = _mesh_surface_area(op_mesh)
                area_ref = self.reference_opening_areas[idx]
                min_area = self.opening_min_ratio * area_ref
                # penalize only if opening area shrinks too much
                penalty = torch.relu(min_area - area_now) / (area_ref + self.opening_area_eps)
                area_losses.append(penalty)
            if len(area_losses) > 0:
                loss_dict["loss_opening_area"] = torch.stack(area_losses).mean()
            else:
                loss_dict["loss_opening_area"] = torch.zeros(1, device=self.device).mean()
        if (
            "loss_opening_overlap" in loss_weighting
            and float(loss_weighting.get("loss_opening_overlap", 0.0)) > 0.0
        ):
            if len(warped_openings) > 0 and self.target_opening_selected is not None:
                warped_opening = warped_openings[0]
                target_opening = self.target_opening_selected

                pcd_wo = sample_points_from_meshes(
                    warped_opening, self.op_sample_num, return_normals=False
                )
                pcd_to = sample_points_from_meshes(
                    target_opening, self.op_sample_num, return_normals=False
                )

                area_w = _mesh_surface_area(warped_opening)
                area_t = _mesh_surface_area(target_opening)
                area_ratio = torch.minimum(area_w, area_t) / (
                    torch.maximum(area_w, area_t) + self.opening_area_eps
                )

                ref_radius = torch.sqrt(area_t / torch.pi + self.opening_area_eps)
                sigma = torch.clamp(self.opening_overlap_sigma_ratio * ref_radius, min=1e-6)
                sigma2 = sigma * sigma
                overlap_spatial = _pointset_overlap_score(pcd_wo, pcd_to, sigma2)
                # Normalize by "self-overlap" of the target so perfect overlap trends to 0 loss
                # even with stochastic surface sampling.
                pcd_to_ref = sample_points_from_meshes(
                    target_opening, self.op_sample_num, return_normals=False
                )
                overlap_ref = _pointset_overlap_score(pcd_to, pcd_to_ref, sigma2).detach()
                overlap_spatial_norm = overlap_spatial / (overlap_ref + self.opening_area_eps)
                overlap_score = torch.clamp(overlap_spatial_norm * area_ratio, min=0.0, max=1.0)

                loss_dict["loss_opening_overlap"] = 1.0 - overlap_score
            else:
                # Maximal mismatch if opening is unavailable.
                loss_dict["loss_opening_overlap"] = torch.ones(1, device=self.device).mean()
        if (
            "loss_grid_occupancy" in loss_weighting
            and float(loss_weighting.get("loss_grid_occupancy", 0.0)) > 0.0
        ):
            self._ensure_grid_occupancy_cache()
            points = self._grid_occ_points
            target_occ = self._grid_occ_target
            if (
                self.grid_occupancy_samples_per_step > 0
                and points.shape[0] > self.grid_occupancy_samples_per_step
            ):
                sel = torch.randperm(points.shape[0], device=points.device)[: self.grid_occupancy_samples_per_step]
                points_eval = points[sel]
                target_eval = target_occ[sel]
            else:
                points_eval = points
                target_eval = target_occ

            pred_occ = _soft_winding_occupancy(
                warped_mesh,
                points_eval,
                chunk_size=self.grid_occupancy_chunk_size,
                use_abs_winding=True,
            )
            if self.grid_occupancy_loss_type == "dice":
                loss_grid_occ = self.dice_loss.forward(pred_occ.unsqueeze(0), target_eval.unsqueeze(0))
            else:
                loss_grid_occ = self.mse_loss(pred_occ, target_eval)
            loss_dict["loss_grid_occupancy"] = (
                loss_grid_occ if not torch.isnan(loss_grid_occ) else torch.tensor(0.0, device=self.device)
            )
        if "loss_volume" in loss_weighting:
            warped_volume = _mesh_abs_volume(warped_mesh)
            loss_dict["loss_volume"] = torch.abs(warped_volume - self.target_volume) / (
                self.target_volume + self.volume_eps
            )
        if "loss_opening_boundary_smooth" in loss_weighting:
            verts_full = warped_mesh.verts_packed()
            # Ring curvature penalty on ordered opening boundary (cyclic).
            ring = verts_full[self.opening_boundary_indices]  # [N, 3]
            ring_prev = torch.roll(ring, shifts=1, dims=0)
            ring_next = torch.roll(ring, shifts=-1, dims=0)
            ring_curv = ((ring_prev - 2.0 * ring + ring_next) ** 2).sum(dim=1).mean()

            # Slightly wider smoothness (3-4 triangle rows) via umbrella Laplacian in k-ring region.
            wide_terms = []
            for vid in self.opening_smooth_region_indices.detach().cpu().tolist():
                nbrs = self._smooth_local_neighbors.get(int(vid), [])
                if not nbrs:
                    continue
                v = verts_full[int(vid)]
                v_n = verts_full[torch.tensor(nbrs, dtype=torch.long, device=self.device)].mean(dim=0)
                wide_terms.append(((v - v_n) ** 2).sum())
            if wide_terms:
                wide_smooth = torch.stack(wide_terms).mean()
                loss_dict["loss_opening_boundary_smooth"] = ring_curv + 0.5 * wide_smooth
            else:
                loss_dict["loss_opening_boundary_smooth"] = ring_curv
        return loss_dict
