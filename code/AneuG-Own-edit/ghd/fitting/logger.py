import os
from typing import Dict, Any, Optional

import torch
from pytorch3d.io import save_obj
from pytorch3d.structures import Meshes


def log_dict_printer(log_dict: Dict[str, Any]) -> None:
    """Pretty-print loss dictionary for quick inspection."""
    if not log_dict:
        print("[logger] empty log dict")
        return
    entries = []
    for key, value in log_dict.items():
        if isinstance(value, (float, int)):
            entries.append(f"{key}: {value:.6f}")
        else:
            entries.append(f"{key}: {value}")
    print(" | ".join(entries))


def _save_mesh(mesh: Meshes, file_path: str) -> None:
    verts = mesh.verts_packed().detach().cpu()
    faces = mesh.faces_packed().detach().cpu()
    save_obj(file_path, verts, faces)


def _mesh_mean_unit_normal(mesh: Meshes) -> torch.Tensor:
    verts = mesh.verts_packed()
    faces = mesh.faces_packed().long()
    tri = verts[faces]
    n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1).sum(dim=0)
    return n / (torch.norm(n) + 1e-12)


def _orthonormal_basis_from_dir(d: torch.Tensor):
    d = d / (torch.norm(d) + 1e-12)
    ref = torch.tensor([1.0, 0.0, 0.0], dtype=d.dtype, device=d.device)
    if torch.abs(torch.dot(d, ref)) > 0.9:
        ref = torch.tensor([0.0, 1.0, 0.0], dtype=d.dtype, device=d.device)
    u = torch.cross(d, ref, dim=0)
    u = u / (torch.norm(u) + 1e-12)
    v = torch.cross(d, u, dim=0)
    v = v / (torch.norm(v) + 1e-12)
    return u, v


def _write_octa_marker_mesh(f, p: torch.Tensor, scale: float, base_idx: int) -> int:
    """Write a small octahedron marker. Returns updated vertex index."""
    px, py, pz = float(p[0]), float(p[1]), float(p[2])
    verts = [
        (px + scale, py, pz), (px - scale, py, pz),
        (px, py + scale, pz), (px, py - scale, pz),
        (px, py, pz + scale), (px, py, pz - scale),
    ]
    faces = [
        (1, 3, 5), (3, 2, 5), (2, 4, 5), (4, 1, 5),
        (3, 1, 6), (2, 3, 6), (4, 2, 6), (1, 4, 6),
    ]
    for v in verts:
        f.write(f"v {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n")
    for a, b, c in faces:
        f.write(f"f {base_idx+a-1} {base_idx+b-1} {base_idx+c-1}\n")
    return base_idx + 6


def _write_arrow_mesh(
    f,
    origin: torch.Tensor,
    direction: torch.Tensor,
    length: float,
    radius: float,
    base_idx: int,
) -> int:
    """Write a shaft+tip arrow mesh (octagonal prism + cone)."""
    d = direction / (torch.norm(direction) + 1e-12)
    u, v = _orthonormal_basis_from_dir(d)
    shaft_len = float(max(length * 0.70, 1e-8))
    tip_len = float(max(length - shaft_len, 1e-8))
    shaft_r = float(max(radius * 0.45, 1e-7))
    head_r = float(max(radius * 1.30, shaft_r * 1.6))

    start = origin
    neck = origin + d * shaft_len
    tip = neck + d * tip_len

    ring_n = 8
    verts = []
    for i in range(ring_n):
        theta = (2.0 * torch.pi * i) / ring_n
        ct = torch.cos(torch.tensor(theta, dtype=d.dtype, device=d.device))
        st = torch.sin(torch.tensor(theta, dtype=d.dtype, device=d.device))
        radial = u * ct + v * st
        verts.append(start + radial * shaft_r)  # shaft start ring
    for i in range(ring_n):
        theta = (2.0 * torch.pi * i) / ring_n
        ct = torch.cos(torch.tensor(theta, dtype=d.dtype, device=d.device))
        st = torch.sin(torch.tensor(theta, dtype=d.dtype, device=d.device))
        radial = u * ct + v * st
        verts.append(neck + radial * head_r)  # arrow head base ring
    verts.append(tip)  # final tip vertex

    for p in verts:
        f.write(f"v {float(p[0]):.10f} {float(p[1]):.10f} {float(p[2]):.10f}\n")

    def gidx(local_idx: int) -> int:
        return base_idx + local_idx

    # Shaft side quads (triangulated)
    for i in range(ring_n):
        j = (i + 1) % ring_n
        f.write(f"f {gidx(i)} {gidx(j)} {gidx(ring_n + j)}\n")
        f.write(f"f {gidx(i)} {gidx(ring_n + j)} {gidx(ring_n + i)}\n")

    # Shaft base cap fan
    for i in range(1, ring_n - 1):
        f.write(f"f {gidx(0)} {gidx(i)} {gidx(i + 1)}\n")

    # Arrow head cone sides
    tip_idx = 2 * ring_n
    for i in range(ring_n):
        j = (i + 1) % ring_n
        f.write(f"f {gidx(ring_n + i)} {gidx(ring_n + j)} {gidx(tip_idx)}\n")

    return base_idx + len(verts)


def _write_line_arrow_obj(
    f,
    origin: torch.Tensor,
    direction: torch.Tensor,
    length: float,
    head_length: float,
    head_width: float,
    base_idx: int,
    color,
) -> int:
    """Write a colored line-arrow in OBJ (no faces), using vertex RGB extension.

    Many viewers (e.g., MeshLab) support vertex colors as: v x y z r g b.
    """
    d = direction / (torch.norm(direction) + 1e-12)
    u, v = _orthonormal_basis_from_dir(d)

    tail = origin
    tip = origin + d * float(length)
    back = tip - d * float(head_length)

    h1 = back + u * float(head_width)
    h2 = back - u * float(head_width)
    h3 = back + v * float(head_width)
    h4 = back - v * float(head_width)

    r, g, b = float(color[0]), float(color[1]), float(color[2])
    verts = [tail, tip, h1, h2, h3, h4]
    for p in verts:
        f.write(
            f"v {float(p[0]):.10f} {float(p[1]):.10f} {float(p[2]):.10f} "
            f"{r:.6f} {g:.6f} {b:.6f}\n"
        )

    tail_idx = base_idx
    tip_idx = base_idx + 1
    h1_idx = base_idx + 2
    h2_idx = base_idx + 3
    h3_idx = base_idx + 4
    h4_idx = base_idx + 5

    f.write(f"l {tail_idx} {tip_idx}\n")
    f.write(f"l {tip_idx} {h1_idx}\n")
    f.write(f"l {tip_idx} {h2_idx}\n")
    f.write(f"l {tip_idx} {h3_idx}\n")
    f.write(f"l {tip_idx} {h4_idx}\n")

    return base_idx + 6


def _write_opening_debug_mtl(mtl_path: str) -> None:
    with open(mtl_path, "w", encoding="utf-8") as m:
        m.write("newmtl warped_mesh\nKd 0.35 0.35 0.35\nKa 0.08 0.08 0.08\nKs 0.02 0.02 0.02\nNs 8\n\n")
        m.write("newmtl target_mesh\nKd 0.65 0.65 0.65\nKa 0.10 0.10 0.10\nKs 0.02 0.02 0.02\nNs 8\n\n")
        m.write("newmtl warped_points\nKd 0.20 0.60 1.00\nKa 0.04 0.12 0.20\nKs 0.05 0.05 0.05\nNs 8\n\n")
        m.write("newmtl target_points\nKd 1.00 0.65 0.20\nKa 0.20 0.12 0.04\nKs 0.05 0.05 0.05\nNs 8\n\n")
        m.write("newmtl warped_normal\nKd 1.00 0.25 0.25\nKa 0.20 0.05 0.05\nKs 0.10 0.10 0.10\nNs 16\n\n")
        m.write("newmtl target_normal\nKd 0.25 0.90 0.35\nKa 0.05 0.18 0.07\nKs 0.10 0.10 0.10\nNs 16\n\n")


def _save_opening_debug_obj(
    file_path: str,
    warped_openings,
    target_openings,
    normal_scale: float,
    point_marker_scale: float,
) -> None:
    """Deterministic opening debug export:
    - opening meshes/vertices for opening_p inspection
    - centroid mean-normal arrows for opening_n inspection
    """
    mtl_path = os.path.splitext(file_path)[0] + ".mtl"
    _write_opening_debug_mtl(mtl_path)
    mtl_name = os.path.basename(mtl_path)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_name}\n")
        f.write("s off\n")
        v_idx = 1  # obj 1-based index
        pair_num = min(len(warped_openings), len(target_openings))
        for op_idx in range(pair_num):
            w_mesh = warped_openings[op_idx].to(torch.device("cpu"))
            t_mesh = target_openings[op_idx].to(torch.device("cpu"))
            wv = w_mesh.verts_packed()
            wf = w_mesh.faces_packed().long()
            tv = t_mesh.verts_packed()
            tf = t_mesh.faces_packed().long()

            # Opening meshes themselves (useful for opening_p inspection)
            f.write(f"o opening_{op_idx}_warped_mesh\n")
            f.write("usemtl warped_mesh\n")
            for p in wv:
                f.write(f"v {float(p[0]):.10f} {float(p[1]):.10f} {float(p[2]):.10f}\n")
            for tri in wf:
                f.write(f"f {int(tri[0])+v_idx} {int(tri[1])+v_idx} {int(tri[2])+v_idx}\n")
            v_idx += wv.shape[0]

            f.write(f"o opening_{op_idx}_target_mesh\n")
            f.write("usemtl target_mesh\n")
            for p in tv:
                f.write(f"v {float(p[0]):.10f} {float(p[1]):.10f} {float(p[2]):.10f}\n")
            for tri in tf:
                f.write(f"f {int(tri[0])+v_idx} {int(tri[1])+v_idx} {int(tri[2])+v_idx}\n")
            v_idx += tv.shape[0]

            # Vertex markers (optional).
            if point_marker_scale > 0:
                f.write(f"o opening_{op_idx}_warped_vertices_markers\n")
                f.write("usemtl warped_points\n")
                for p in wv:
                    v_idx = _write_octa_marker_mesh(f, p, point_marker_scale, v_idx)
                f.write(f"o opening_{op_idx}_target_vertices_markers\n")
                f.write("usemtl target_points\n")
                for p in tv:
                    v_idx = _write_octa_marker_mesh(f, p, point_marker_scale, v_idx)

            # Mean normal arrows (what opening_n effectively uses in pouch-only now).
            wc = wv.mean(dim=0)
            tc = tv.mean(dim=0)
            wn = _mesh_mean_unit_normal(w_mesh)
            tn = _mesh_mean_unit_normal(t_mesh)
            if normal_scale > 0:
                f.write(f"o opening_{op_idx}_warped_mean_normal\n")
                v_idx = _write_line_arrow_obj(
                    f,
                    origin=wc,
                    direction=wn,
                    length=float(normal_scale),
                    head_length=float(normal_scale * 0.28),
                    head_width=float(normal_scale * 0.16),
                    base_idx=v_idx,
                    color=(1.0, 0.15, 0.15),
                )

                f.write(f"o opening_{op_idx}_target_mean_normal\n")
                v_idx = _write_line_arrow_obj(
                    f,
                    origin=tc,
                    direction=tn,
                    length=float(normal_scale),
                    head_length=float(normal_scale * 0.28),
                    head_width=float(normal_scale * 0.16),
                    base_idx=v_idx,
                    color=(0.2, 1.0, 0.2),
                )


def _save_opening_normals_obj(
    file_path: str,
    warped_openings,
    target_openings,
    normal_scale: float,
) -> None:
    """Export only opening centroid normal arrows as colored OBJ lines."""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("s off\n")
        v_idx = 1
        pair_num = min(len(warped_openings), len(target_openings))
        for op_idx in range(pair_num):
            w_mesh = warped_openings[op_idx].to(torch.device("cpu"))
            t_mesh = target_openings[op_idx].to(torch.device("cpu"))
            wv = w_mesh.verts_packed().detach().cpu()
            tv = t_mesh.verts_packed().detach().cpu()

            wc = wv.mean(dim=0)
            tc = tv.mean(dim=0)
            wn = _mesh_mean_unit_normal(w_mesh)
            tn = _mesh_mean_unit_normal(t_mesh)

            # Keep arrows visible independent of world scale.
            span_w = float(torch.max(wv.max(dim=0).values - wv.min(dim=0).values))
            span_t = float(torch.max(tv.max(dim=0).values - tv.min(dim=0).values))
            ref_span = max(span_w, span_t, 1e-6)
            arrow_len = max(float(normal_scale), 0.08 * ref_span)

            f.write(f"o opening_{op_idx}_warped_mean_normal\n")
            v_idx = _write_line_arrow_obj(
                f,
                origin=wc,
                direction=wn,
                length=arrow_len,
                head_length=float(arrow_len * 0.28),
                head_width=float(arrow_len * 0.16),
                base_idx=v_idx,
                color=(1.0, 0.15, 0.15),
            )

            f.write(f"o opening_{op_idx}_target_mean_normal\n")
            v_idx = _write_line_arrow_obj(
                f,
                origin=tc,
                direction=tn,
                length=arrow_len,
                head_length=float(arrow_len * 0.28),
                head_width=float(arrow_len * 0.16),
                base_idx=v_idx,
                color=(0.2, 1.0, 0.2),
            )


def _save_opening_debug_png(
    file_path: str,
    warped_openings,
    target_openings,
    normal_scale: float,
) -> None:
    """Render a lightweight PNG preview with colored normal arrows."""
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except Exception as e:
        raise RuntimeError(f"matplotlib import failed: {e}")

    def _ordered_ring(points_xyz: np.ndarray) -> np.ndarray:
        """Get a stable boundary ring via 2D convex hull on best-fit plane."""
        if points_xyz.shape[0] < 4:
            return points_xyz
        center = points_xyz.mean(axis=0, keepdims=True)
        p0 = points_xyz - center
        _, _, vh = np.linalg.svd(p0, full_matrices=False)
        u = vh[0]
        v = vh[1]
        uv = np.stack([p0 @ u, p0 @ v], axis=1)

        # Monotonic-chain convex hull in 2D.
        pts = [(float(uv[i, 0]), float(uv[i, 1]), int(i)) for i in range(uv.shape[0])]
        pts.sort(key=lambda x: (x[0], x[1]))

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
                upper.pop()
            upper.append(p)

        hull = lower[:-1] + upper[:-1]
        if len(hull) < 3:
            # Fallback: angle sort if hull degenerates.
            ang = np.arctan2(uv[:, 1], uv[:, 0])
            order = np.argsort(ang)
            return points_xyz[order]

        hull_idx = [p[2] for p in hull]
        return points_xyz[hull_idx]

    pair_num = min(len(warped_openings), len(target_openings))
    if pair_num <= 0:
        return

    fig = plt.figure(figsize=(7.2, 7.2), dpi=200)
    ax = fig.add_subplot(111, projection="3d")

    all_pts = []
    for op_idx in range(pair_num):
        w_mesh = warped_openings[op_idx].to(torch.device("cpu"))
        t_mesh = target_openings[op_idx].to(torch.device("cpu"))
        wv = w_mesh.verts_packed().detach().cpu().numpy()
        tv = t_mesh.verts_packed().detach().cpu().numpy()
        wv_ring = _ordered_ring(wv)
        tv_ring = _ordered_ring(tv)

        all_pts.append(wv)
        all_pts.append(tv)

        # Draw clean opening rings (no triangulation artifacts).
        wv_loop = np.vstack([wv_ring, wv_ring[0]])
        tv_loop = np.vstack([tv_ring, tv_ring[0]])
        ax.plot(wv_loop[:, 0], wv_loop[:, 1], wv_loop[:, 2], color=(0.28, 0.28, 0.28), linewidth=2.0)
        ax.plot(tv_loop[:, 0], tv_loop[:, 1], tv_loop[:, 2], color=(0.62, 0.62, 0.62), linewidth=1.8)

        # Light fan fill from centroid for visual context (stable with sorted ring).
        wc_fill = wv_ring.mean(axis=0)
        tc_fill = tv_ring.mean(axis=0)
        w_polys = [[wc_fill, wv_ring[i], wv_ring[(i + 1) % len(wv_ring)]] for i in range(len(wv_ring))]
        t_polys = [[tc_fill, tv_ring[i], tv_ring[(i + 1) % len(tv_ring)]] for i in range(len(tv_ring))]
        ax.add_collection3d(Poly3DCollection(w_polys, facecolor=(0.40, 0.40, 0.40, 0.10), edgecolor='none'))
        ax.add_collection3d(Poly3DCollection(t_polys, facecolor=(0.78, 0.78, 0.78, 0.10), edgecolor='none'))

        wc = wv_ring.mean(axis=0)
        tc = tv_ring.mean(axis=0)
        wn = _mesh_mean_unit_normal(w_mesh).detach().cpu().numpy()
        tn = _mesh_mean_unit_normal(t_mesh).detach().cpu().numpy()

        ref_span = float(max(np.max(wv_ring.max(axis=0) - wv_ring.min(axis=0)), np.max(tv_ring.max(axis=0) - tv_ring.min(axis=0)), 1e-6))
        qlen = float(max(normal_scale, 0.08 * ref_span))
        ax.scatter([wc[0]], [wc[1]], [wc[2]], c=[(1.0, 0.15, 0.15)], s=18)
        ax.scatter([tc[0]], [tc[1]], [tc[2]], c=[(0.2, 1.0, 0.2)], s=18)
        ax.quiver(wc[0], wc[1], wc[2], wn[0], wn[1], wn[2],
                  color=(1.0, 0.15, 0.15), length=qlen, normalize=True, linewidths=2.2)
        ax.quiver(tc[0], tc[1], tc[2], tn[0], tn[1], tn[2],
                  color=(0.2, 1.0, 0.2), length=qlen, normalize=True, linewidths=2.2)

    all_pts_arr = np.concatenate(all_pts, axis=0)
    pmin = all_pts_arr.min(axis=0)
    pmax = all_pts_arr.max(axis=0)
    center = 0.5 * (pmin + pmax)
    span = float(np.max(pmax - pmin))
    radius = 0.55 * max(span, normal_scale * 4.0, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect([1.0, 1.0, 1.0])

    ax.set_title("Opening + normals (red=warped, green=target)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig(file_path)
    plt.close(fig)


def viz_fitting_static(epoch: int, log_path: str, warped_mesh: Meshes,
                       target_mesh: Optional[Meshes], args: Any, force: bool = False,
                       warped_openings=None, target_openings=None) -> None:
    """Periodically dump warped/target meshes for qualitative inspection."""
    freq = getattr(args, "viz_freq", None)
    if freq is None or freq <= 0:
        return
    if (not force) and (epoch % freq != 0):
        return
    viz_dir = os.path.join(log_path, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    warped_path = os.path.join(viz_dir, f"warped_epoch_{epoch:05d}.obj")
    _save_mesh(warped_mesh, warped_path)
    if target_mesh is not None:
        target_path = os.path.join(viz_dir, "target.obj")
        if not os.path.exists(target_path):
            _save_mesh(target_mesh, target_path)
    if getattr(args, "debug_opening_losses", 0) == 1 and warped_openings is not None and target_openings is not None:
        try:
            normal_scale = float(getattr(args, "debug_opening_normal_scale", 0.01))
            point_marker_scale = float(getattr(args, "debug_opening_point_marker_scale", normal_scale * 0.35))
            opening_debug_dir = os.path.join(viz_dir, "opening_debug")
            os.makedirs(opening_debug_dir, exist_ok=True)
            debug_path = os.path.join(opening_debug_dir, f"opening_debug_epoch_{epoch:05d}.obj")
            debug_png_path = os.path.join(opening_debug_dir, f"opening_debug_epoch_{epoch:05d}.png")
            normals_only_obj_path = os.path.join(opening_debug_dir, f"opening_normals_epoch_{epoch:05d}.obj")
            _save_opening_debug_obj(
                debug_path,
                warped_openings,
                target_openings,
                normal_scale=normal_scale,
                point_marker_scale=point_marker_scale,
            )
            _save_opening_normals_obj(
                normals_only_obj_path,
                warped_openings,
                target_openings,
                normal_scale=normal_scale,
            )
            _save_opening_debug_png(
                debug_png_path,
                warped_openings,
                target_openings,
                normal_scale=normal_scale,
            )
        except Exception as e:
            print(f"[logger] opening debug export failed at epoch {epoch}: {e}")


def viz_fitting_debug(epoch: int, log_path: str, warped_mesh: Meshes,
                      target_mesh: Optional[Meshes], args: Any, prefix: str = "debug") -> None:
    """Optional extra dumps when debugging."""
    if not getattr(args, "debug_viz", False):
        return
    debug_dir = os.path.join(log_path, f"{prefix}_viz")
    os.makedirs(debug_dir, exist_ok=True)
    warped_path = os.path.join(debug_dir, f"warped_epoch_{epoch:05d}.obj")
    _save_mesh(warped_mesh, warped_path)
    if target_mesh is not None:
        target_path = os.path.join(debug_dir, f"target_epoch_{epoch:05d}.obj")
        _save_mesh(target_mesh, target_path)
