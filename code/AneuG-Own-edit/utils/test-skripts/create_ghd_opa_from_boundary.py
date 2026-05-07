#!/usr/bin/env python3
"""Create a GHD-condition OPA checkpoint from the boundary loop of a fitted mesh."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from matplotlib.path import Path as MplPath
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from scipy.spatial import Delaunay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="C0066")
    parser.add_argument("--ghd-root", type=Path, default=Path("checkpoint-v2/ghd_fitting_split_real"))
    parser.add_argument("--run", default="vanilla")
    parser.add_argument("--mesh-name", default="warped_epoch_02999.obj")
    parser.add_argument("--output-name", default="opa_checkpoint.pkl")
    parser.add_argument("--force", type=int, default=0)
    return parser.parse_args()


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts, faces, _ = load_obj(str(path))
    return verts.detach().cpu().numpy().astype(np.float32), faces.verts_idx.detach().cpu().numpy().astype(np.int64)


def boundary_components(faces: np.ndarray) -> tuple[list[list[int]], dict[int, list[int]]]:
    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            a_i, b_i = int(a), int(b)
            if a_i > b_i:
                a_i, b_i = b_i, a_i
            edge_count[(a_i, b_i)] += 1

    adj: dict[int, list[int]] = defaultdict(list)
    for (a, b), count in edge_count.items():
        if count == 1:
            adj[a].append(b)
            adj[b].append(a)

    seen: set[int] = set()
    comps: list[list[int]] = []
    for start in adj:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        comp = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nxt in adj[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(comp)
    return comps, adj


def order_boundary_loop(component: list[int], adj: dict[int, list[int]]) -> np.ndarray:
    if not component:
        raise ValueError("Empty boundary component")
    degree_counts = {len(adj[idx]) for idx in component}
    if degree_counts != {2}:
        raise ValueError(f"Boundary component is not a closed 2-regular loop: degrees={sorted(degree_counts)}")

    start = min(component)
    prev = None
    cur = start
    ordered = []
    component_set = set(component)
    while True:
        ordered.append(cur)
        candidates = [idx for idx in adj[cur] if idx != prev and idx in component_set]
        if not candidates:
            break
        nxt = candidates[0]
        if nxt == start:
            break
        prev, cur = cur, nxt
        if len(ordered) > len(component) + 1:
            raise RuntimeError("Boundary traversal did not close")
    if len(ordered) != len(component):
        raise RuntimeError(f"Boundary traversal visited {len(ordered)} of {len(component)} vertices")
    return np.asarray(ordered, dtype=np.int64)


def plane_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    x = points - center
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    u = vh[0] / (np.linalg.norm(vh[0]) + 1e-12)
    v = vh[1] / (np.linalg.norm(vh[1]) + 1e-12)
    n = np.cross(u, v)
    n = n / (np.linalg.norm(n) + 1e-12)
    return center, u, v


def triangulate_boundary(points: np.ndarray) -> np.ndarray:
    center, u, v = plane_basis(points)
    rel = points - center
    points_2d = np.stack([rel @ u, rel @ v], axis=1)
    delaunay = Delaunay(points_2d)
    polygon = MplPath(points_2d)
    centroids = points_2d[delaunay.simplices].mean(axis=1)
    inside = polygon.contains_points(centroids, radius=1e-9)
    faces = delaunay.simplices[inside].astype(np.int64)
    if faces.size == 0:
        raise RuntimeError("Boundary triangulation produced no faces")
    return faces


def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    mesh = Meshes(verts=[torch.from_numpy(verts).float()], faces=[torch.from_numpy(faces).long()])
    return mesh.verts_normals_packed().detach().cpu().numpy().astype(np.float32)


def safe_unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def main() -> None:
    args = parse_args()
    case_root = args.ghd_root / args.case
    mesh_path = case_root / args.run / "viz" / args.mesh_name
    out_path = case_root / args.output_name
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)
    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} exists. Use --force 1 to replace it after backup.")

    verts, faces = load_mesh(mesh_path)
    components, adj = boundary_components(faces)
    if len(components) != 1:
        sizes = [len(comp) for comp in components]
        raise ValueError(f"Expected exactly one boundary loop for {args.case}, found {len(components)}: {sizes}")
    boundary_idx = order_boundary_loop(components[0], adj)
    boundary_points = verts[boundary_idx]
    rec_faces = triangulate_boundary(boundary_points)
    rec_faces_map = boundary_idx[rec_faces]

    normals = vertex_normals(verts, faces)
    op_v_normal = normals[boundary_idx]
    op_n_mean = safe_unit(op_v_normal.mean(axis=0))
    if float(np.linalg.norm(op_n_mean)) < 1e-6:
        _, u, v = plane_basis(boundary_points)
        op_n_mean = safe_unit(np.cross(u, v))

    chk = {
        "label": args.case,
        "source": "ghd_warped_boundary_loop",
        "mesh_path": str(mesh_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "op_v_indices": [boundary_idx.tolist()],
        "op_v_coords": [boundary_points.astype(np.float32)],
        "op_v_normal": [op_v_normal.astype(np.float32)],
        "op_n_mean": [op_n_mean.astype(np.float32)],
        "op_rec_v": [boundary_points.astype(np.float32)],
        "op_rec_f": [rec_faces.astype(np.int64)],
        "op_rec_f_map": [rec_faces_map.astype(np.int64)],
        "op_rec_v_indices_map": [boundary_idx.astype(np.int64)],
    }

    backup_path = None
    if out_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = out_path.with_name(f"{out_path.stem}.backup_before_boundary_fix_{stamp}{out_path.suffix}")
        shutil.copy2(out_path, backup_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(chk, handle)

    bbox = boundary_points.max(axis=0) - boundary_points.min(axis=0)
    summary = {
        "case": args.case,
        "mesh_path": str(mesh_path),
        "output_path": str(out_path),
        "backup_path": str(backup_path) if backup_path else None,
        "boundary_vertices": int(boundary_idx.size),
        "boundary_faces": int(rec_faces.shape[0]),
        "span": float(np.linalg.norm(bbox)),
        "bbox_min": boundary_points.min(axis=0).round(8).tolist(),
        "bbox_max": boundary_points.max(axis=0).round(8).tolist(),
        "source": chk["source"],
    }
    summary_path = out_path.with_name(f"{out_path.stem}.boundary_fix_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
