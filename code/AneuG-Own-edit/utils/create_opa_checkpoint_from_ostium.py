#!/usr/bin/env python3
"""Create one-opening opa_checkpoint.pkl from aneurysm-only ostium data.

Source per case (under zero root):
- 04_subpointclouds/subpointcloud_label_2.ply OR subpointcloud_label_2.ply
- 07_other/centroid_ostium.npy
- 07_other/normal_vector.npy

Target per case (under alignment root):
- part_aligned.obj
- opa_checkpoint.pkl (generated)
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import shapely.geometry
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one-opening opa_checkpoint.pkl from ostium annotations.")
    parser.add_argument(
        "--zero-root",
        type=Path,
        default=Path("checkpoints-new/zero-aneurysmen"),
        help="Root with aneurysm-only source folders.",
    )
    parser.add_argument(
        "--alignment-root",
        type=Path,
        default=Path("checkpoints-new/alignment"),
        help="Root with alignment folders containing part_aligned.obj.",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Optional single case name in alignment space (e.g. C0093). If omitted, process all.",
    )
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=["aneux_"],
        help="Prefix to strip from zero-root directory names to derive alignment case names.",
    )
    parser.add_argument(
        "--ostium-label",
        type=int,
        default=2,
        help="Subpointcloud label number used for ostium (default: 2).",
    )
    parser.add_argument(
        "--flip-inside-normal",
        action="store_true",
        default=False,
        help="Flip provided ostium normal so it points outward for OPA convention (default: enabled).",
    )
    parser.add_argument(
        "--keep-inside-normal",
        action="store_true",
        help="Disable normal flipping and keep normal as stored in normal_vector.npy.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing opa_checkpoint.pkl.",
    )
    parser.add_argument(
        "--smooth-iters",
        type=int,
        default=0,
        help="Circular smoothing iterations on projected opening ring (default: 0).",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.0,
        help="Smoothing strength in [0,1) for projected opening ring (default: 0.0).",
    )
    parser.add_argument(
        "--target-opening-triangles",
        type=int,
        default=50,
        help="Approximate triangle count for opening plane (default: 50). Set <=0 to disable downsampling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without writing files.",
    )
    parser.add_argument(
        "--export-opening-planes",
        type=int,
        default=0,
        help="1: run utils/inspect/export_opening_planes.py for each processed target case.",
    )
    parser.add_argument(
        "--canonical-name",
        type=str,
        default="canonical_model",
        help="Canonical case name under alignment-root for opening-plane export.",
    )
    parser.add_argument(
        "--canonical-index",
        type=int,
        default=0,
        help="Canonical opening index forwarded to export_opening_planes.py.",
    )
    parser.add_argument(
        "--normal-scale",
        type=float,
        default=0.01,
        help="Normal visualization scale forwarded to export_opening_planes.py.",
    )
    return parser.parse_args()


def normalize_case_name(case_name: str, prefixes: Iterable[str]) -> str:
    for prefix in prefixes:
        if prefix and case_name.startswith(prefix):
            return case_name[len(prefix) :]
    return case_name


def load_mesh_vertices_and_normals(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    verts = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)
    return verts, normals


def nearest_vertex_indices(points: np.ndarray, verts: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        _, indices = cKDTree(verts).query(points, k=1)
        return np.asarray(indices, dtype=np.int64)
    except Exception:
        # Fallback without scipy
        distances = np.linalg.norm(points[:, None, :] - verts[None, :, :], axis=2)
        return np.argmin(distances, axis=1).astype(np.int64)


def unique_preserve_order(indices: np.ndarray) -> np.ndarray:
    seen = set()
    out = []
    for idx in indices.tolist():
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return np.asarray(out, dtype=np.int64)


def build_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normal / (np.linalg.norm(normal) + 1e-12)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(n, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(n, ref)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return u, v


def fit_plane_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    x = points - center.reshape(1, 3)
    cov = x.T @ x / max(1, x.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    e0, e1, e2 = eigvecs[:, order[0]], eigvecs[:, order[1]], eigvecs[:, order[2]]
    normal = e2 / (np.linalg.norm(e2) + 1e-12)
    u = e0 / (np.linalg.norm(e0) + 1e-12)
    v = e1 / (np.linalg.norm(e1) + 1e-12)
    return center, normal, u, v


def project_to_plane(points: np.ndarray, center: np.ndarray, normal: np.ndarray) -> np.ndarray:
    rel = points - center.reshape(1, 3)
    dist = rel @ normal
    return points - dist.reshape(-1, 1) * normal.reshape(1, 3)


def to_plane_2d(points: np.ndarray, center: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    rel = points - center.reshape(1, 3)
    return np.stack([rel @ u, rel @ v], axis=1)


def signed_distance_to_plane(points: np.ndarray, center: np.ndarray, normal: np.ndarray) -> np.ndarray:
    rel = points - center.reshape(1, 3)
    return rel @ normal.reshape(3, 1)


def sort_boundary_points(points2d: np.ndarray) -> np.ndarray:
    center2d = points2d.mean(axis=0)
    rel = points2d - center2d.reshape(1, 2)
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    return np.argsort(angles)


def smooth_ring(points2d: np.ndarray, iters: int = 2, alpha: float = 0.25) -> np.ndarray:
    if points2d.shape[0] < 3 or iters <= 0 or alpha <= 0:
        return points2d.copy()
    x = points2d.copy()
    a = float(max(0.0, min(0.49, alpha)))
    for _ in range(iters):
        prev_ = np.roll(x, 1, axis=0)
        next_ = np.roll(x, -1, axis=0)
        x = (1.0 - 2.0 * a) * x + a * prev_ + a * next_
    return x


def downsample_ring_by_arclength(points2d: np.ndarray, target_count: int) -> np.ndarray:
    """Return ordered index subset sampled approximately uniformly along ring arclength."""
    n = points2d.shape[0]
    if target_count <= 0 or n <= target_count:
        return np.arange(n, dtype=np.int64)
    nxt = np.roll(points2d, -1, axis=0)
    seg = np.linalg.norm(nxt - points2d, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-12:
        return np.linspace(0, n - 1, num=target_count, dtype=np.int64)

    targets = np.linspace(0.0, total, num=target_count, endpoint=False)
    idx = np.searchsorted(cum, targets, side="right") - 1
    idx = np.clip(idx, 0, n - 1)
    # ensure uniqueness and keep order
    out = []
    seen = set()
    for i in idx.tolist():
        if i not in seen:
            seen.add(i)
            out.append(i)
    # if collisions occurred, fill from remaining points to hit target_count
    if len(out) < target_count:
        for i in range(n):
            if i not in seen:
                out.append(i)
                seen.add(i)
                if len(out) == target_count:
                    break
    return np.asarray(out, dtype=np.int64)


def fan_triangulation(num_vertices: int) -> np.ndarray:
    if num_vertices < 3:
        return np.empty((0, 3), dtype=np.int64)
    faces = []
    for i in range(1, num_vertices - 1):
        faces.append([0, i, i + 1])
    return np.asarray(faces, dtype=np.int64)


def triangulate_polygon_2d(points2d_ordered: np.ndarray) -> np.ndarray:
    """Triangulate using boundary vertices only (no synthetic center point)."""
    n = points2d_ordered.shape[0]
    if n < 3:
        return np.empty((0, 3), dtype=np.int64)
    poly = shapely.geometry.Polygon(points2d_ordered)
    if (not poly.is_valid) or poly.area <= 1e-12:
        return fan_triangulation(n)

    # Try Delaunay-on-boundary-vertices and keep triangles whose centroid lies in polygon.
    try:
        from scipy.spatial import Delaunay

        tri = Delaunay(points2d_ordered)
        faces = np.asarray(tri.simplices, dtype=np.int64)
        keep = []
        for f in faces:
            c = points2d_ordered[f].mean(axis=0)
            if poly.buffer(1e-12).contains(shapely.geometry.Point(float(c[0]), float(c[1]))):
                keep.append(f.tolist())
        faces_kept = np.asarray(keep, dtype=np.int64) if keep else np.empty((0, 3), dtype=np.int64)
        if faces_kept.shape[0] > 0:
            return faces_kept
    except Exception:
        pass

    # Fallback: robust default.
    return fan_triangulation(n)


def resolve_ostium_ply_path(zero_case_dir: Path, ostium_label: int) -> Path:
    cand = [
        zero_case_dir / "04_subpointclouds" / f"subpointcloud_label_{ostium_label}.ply",
        zero_case_dir / f"subpointcloud_label_{ostium_label}.ply",
    ]
    for p in cand:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find ostium ply in {zero_case_dir} for label {ostium_label}")


def create_opa_checkpoint_for_case(
    zero_case_dir: Path,
    alignment_case_dir: Path,
    ostium_label: int = 2,
    flip_inside_normal: bool = True,
    smooth_iters: int = 2,
    smooth_alpha: float = 0.25,
    target_opening_triangles: int = 20,
) -> dict:
    mesh_path = alignment_case_dir / "part_aligned.obj"
    ostium_ply_path = resolve_ostium_ply_path(zero_case_dir, ostium_label)
    centroid_path = zero_case_dir / "07_other" / "centroid_ostium.npy"
    normal_path = zero_case_dir / "07_other" / "normal_vector.npy"

    required = [mesh_path, ostium_ply_path, centroid_path, normal_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    verts, vnormals = load_mesh_vertices_and_normals(mesh_path)
    ostium_pc = trimesh.load(ostium_ply_path, process=False)
    ostium_points = np.asarray(ostium_pc.vertices)
    centroid = np.asarray(np.load(centroid_path)).reshape(3)
    normal = np.asarray(np.load(normal_path)).reshape(3)
    if flip_inside_normal:
        normal = -normal
    normal = normal / (np.linalg.norm(normal) + 1e-12)

    mapped_indices_raw = nearest_vertex_indices(ostium_points, verts)
    mapped_indices = unique_preserve_order(mapped_indices_raw)
    mapped_coords = verts[mapped_indices]
    mapped_normals = vnormals[mapped_indices]

    # Use best-fit plane (robust to non-planar ostium ring input).
    plane_center, pca_normal, u, v = fit_plane_pca(mapped_coords)
    # Keep source normal convention but align it with PCA orientation to avoid sign flips.
    if np.dot(pca_normal, normal) < 0:
        pca_normal = -pca_normal
    normal = pca_normal / (np.linalg.norm(pca_normal) + 1e-12)

    mapped_coords_proj = project_to_plane(mapped_coords, plane_center, normal)
    mapped_coords_2d = to_plane_2d(mapped_coords_proj, plane_center, u, v)
    # Sort boundary points to form a stable opening polygon.
    order = sort_boundary_points(mapped_coords_2d)
    mapped_indices = mapped_indices[order]
    mapped_coords = mapped_coords[order]
    mapped_coords_proj = mapped_coords_proj[order]
    mapped_coords_2d = mapped_coords_2d[order]
    mapped_normals = mapped_normals[order]

    # Optional coarse ring resolution (triangles ~= vertices-2 for simple polygon).
    if target_opening_triangles > 0:
        target_vertices = max(3, int(target_opening_triangles) + 2)
        keep = downsample_ring_by_arclength(mapped_coords_2d, target_vertices)
        mapped_indices = mapped_indices[keep]
        mapped_coords = mapped_coords[keep]
        mapped_coords_proj = mapped_coords_proj[keep]
        mapped_coords_2d = mapped_coords_2d[keep]
        mapped_normals = mapped_normals[keep]

    # Smooth boundary jitter in-plane, but preserve each vertex's signed out-of-plane
    # height so the reconstructed opening is not forced to be flat.
    mapped_coords_2d_smooth = smooth_ring(mapped_coords_2d, iters=smooth_iters, alpha=smooth_alpha)
    mapped_signed_heights = signed_distance_to_plane(mapped_coords, plane_center, normal)
    rec_vertices = (
        plane_center.reshape(1, 3)
        + mapped_coords_2d_smooth[:, 0:1] * u.reshape(1, 3)
        + mapped_coords_2d_smooth[:, 1:2] * v.reshape(1, 3)
        + mapped_signed_heights * normal.reshape(1, 3)
    )

    rec_faces = triangulate_polygon_2d(mapped_coords_2d_smooth)
    rec_v_indices_map = mapped_indices.copy()
    rec_f_map = rec_v_indices_map[rec_faces] if rec_faces.shape[0] > 0 else np.empty((0, 3), dtype=np.int64)

    return {
        "op_v_indices": [mapped_indices.tolist()],
        "op_v_coords": [mapped_coords],
        "op_v_normal": [mapped_normals],
        "op_n_mean": [normal],
        "op_rec_v": [rec_vertices],
        "op_rec_f": [rec_faces],
        "op_rec_v_indices_map": [rec_v_indices_map],
        "op_rec_f_map": [rec_f_map],
    }


def collect_case_pairs(zero_root: Path, alignment_root: Path, strip_prefixes: Iterable[str]) -> list[tuple[str, Path, Path]]:
    pairs = []
    for zero_case_dir in sorted([p for p in zero_root.iterdir() if p.is_dir()]):
        case_name = normalize_case_name(zero_case_dir.name, strip_prefixes)
        alignment_case_dir = alignment_root / case_name
        if alignment_case_dir.is_dir():
            pairs.append((case_name, zero_case_dir, alignment_case_dir))
    return pairs


def run_export_opening_planes(
    alignment_root: Path,
    canonical_name: str,
    target_name: str,
    canonical_index: int,
    normal_scale: float,
) -> int:
    script_path = Path(__file__).resolve().parent / "inspect" / "export_opening_planes.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--root",
        str(alignment_root),
        "--canonical",
        str(canonical_name),
        "--target",
        str(target_name),
        "--canonical-index",
        str(int(canonical_index)),
        "--normal-scale",
        str(float(normal_scale)),
    ]
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    args = parse_args()
    zero_root = args.zero_root
    alignment_root = args.alignment_root

    if not zero_root.is_dir():
        raise SystemExit(f"Zero root not found: {zero_root}")
    if not alignment_root.is_dir():
        raise SystemExit(f"Alignment root not found: {alignment_root}")

    flip_inside_normal = args.flip_inside_normal and not args.keep_inside_normal
    pairs = collect_case_pairs(zero_root, alignment_root, args.strip_prefix)
    if args.case is not None:
        pairs = [p for p in pairs if p[0] == args.case]
        if not pairs:
            raise SystemExit(f"Case '{args.case}' not found in mapped zero/alignment pairs.")

    made = 0
    skipped_exists = 0
    failed = 0
    export_targets: list[str] = []
    for case_name, zero_case_dir, alignment_case_dir in pairs:
        out_path = alignment_case_dir / "opa_checkpoint.pkl"
        if out_path.exists() and not args.overwrite:
            skipped_exists += 1
            if int(args.export_opening_planes) == 1:
                export_targets.append(case_name)
            continue
        try:
            chk = create_opa_checkpoint_for_case(
                zero_case_dir=zero_case_dir,
                alignment_case_dir=alignment_case_dir,
                ostium_label=args.ostium_label,
                flip_inside_normal=flip_inside_normal,
                smooth_iters=args.smooth_iters,
                smooth_alpha=args.smooth_alpha,
                target_opening_triangles=args.target_opening_triangles,
            )
            print(f"{case_name}: {zero_case_dir} -> {out_path}")
            if not args.dry_run:
                with open(out_path, "wb") as f:
                    pickle.dump(chk, f)
            made += 1
            if int(args.export_opening_planes) == 1:
                export_targets.append(case_name)
        except Exception as exc:  # Keep batch processing robust.
            failed += 1
            print(f"{case_name}: FAILED ({exc})")

    export_total = 0
    export_failed = 0
    if int(args.export_opening_planes) == 1 and not args.dry_run:
        canonical_chk = alignment_root / args.canonical_name / "opa_checkpoint.pkl"
        if not canonical_chk.exists():
            raise SystemExit(
                f"Missing canonical checkpoint for export: {canonical_chk}. "
                "Generate canonical OPA first or set --export-opening-planes 0."
            )
        seen = set()
        unique_targets = []
        for t in export_targets:
            if t in seen:
                continue
            seen.add(t)
            unique_targets.append(t)
        for target_name in unique_targets:
            export_total += 1
            rc = run_export_opening_planes(
                alignment_root=alignment_root,
                canonical_name=args.canonical_name,
                target_name=target_name,
                canonical_index=args.canonical_index,
                normal_scale=args.normal_scale,
            )
            if rc != 0:
                export_failed += 1
                print(f"{target_name}: export_opening_planes FAILED (rc={rc})")

    print("\nSummary")
    print(f"Pairs considered: {len(pairs)}")
    print(f"Written: {made}")
    print(f"Skipped (exists): {skipped_exists}")
    print(f"Failed: {failed}")
    if int(args.export_opening_planes) == 1:
        print(f"export_opening_planes runs: {export_total}, failed: {export_failed}")
    print(f"Normal mode: {'flipped (outside)' if flip_inside_normal else 'kept (inside)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
