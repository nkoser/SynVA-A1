#!/usr/bin/env python3
"""Export OPA opening-plane meshes for canonical and target to OBJ for inspection."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export opening planes from OPA checkpoints.")
    p.add_argument("--root", type=Path, required=True, help="Alignment root containing case folders.")
    p.add_argument(
        "--canonical",
        type=str,
        default=None,
        help="Canonical case folder name (required unless --target-only is set).",
    )
    p.add_argument("--target", type=str, required=True, help="Target case folder name.")
    p.add_argument(
        "--target-only",
        action="store_true",
        help=(
            "Skip canonical matching and export only target opening artifacts "
            "(target_opening_plane.obj and target_part_plus_opening_plane.obj)."
        ),
    )
    p.add_argument(
        "--target-index",
        type=int,
        default=0,
        help="Target opening index used in --target-only mode (default: 0).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/<target>/inspection_opening_planes",
    )
    p.add_argument(
        "--canonical-index",
        type=int,
        default=0,
        help="Canonical opening index to use for pouch comparison (default: 0).",
    )
    p.add_argument(
        "--normal-scale",
        type=float,
        default=0.01,
        help="Arrow length used when exporting normal vectors as line segments.",
    )
    return p.parse_args()


def _load_opa(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        chk = pickle.load(f)
    if "op_rec_v" not in chk or "op_rec_f" not in chk:
        raise KeyError(f"{path} missing op_rec_v/op_rec_f")
    return chk


def _write_obj(path: Path, verts: np.ndarray, faces: np.ndarray, object_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"o {object_name}\n")
        for v in verts:
            f.write(f"v {float(v[0]):.10f} {float(v[1]):.10f} {float(v[2]):.10f}\n")
        for tri in faces:
            f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


def _load_obj_vertices_faces(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    verts = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p = line.strip().split()
                if len(p) >= 4:
                    verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                p = line.strip().split()[1:]
                if len(p) < 3:
                    continue
                idx = []
                for tok in p[:3]:
                    # Support "f v", "f v/vt", "f v/vt/vn", "f v//vn"
                    vtok = tok.split("/")[0]
                    if not vtok:
                        idx = []
                        break
                    i = int(vtok)
                    if i < 0:
                        i = len(verts) + i + 1
                    idx.append(i - 1)  # OBJ is 1-based
                if len(idx) == 3:
                    faces.append(idx)
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _write_combined_part_and_opening(
    path: Path,
    part_verts: np.ndarray,
    part_faces: np.ndarray,
    opening_verts: np.ndarray,
    opening_faces: np.ndarray,
    part_name: str,
    opening_name: str,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"o {part_name}\n")
        for v in part_verts:
            f.write(f"v {float(v[0]):.10f} {float(v[1]):.10f} {float(v[2]):.10f}\n")
        for tri in part_faces:
            f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")

        off = part_verts.shape[0]
        f.write(f"\no {opening_name}\n")
        for v in opening_verts:
            f.write(f"v {float(v[0]):.10f} {float(v[1]):.10f} {float(v[2]):.10f}\n")
        for tri in opening_faces:
            f.write(f"f {int(tri[0]) + 1 + off} {int(tri[1]) + 1 + off} {int(tri[2]) + 1 + off}\n")


def _write_cross_markers(
    f,
    points: np.ndarray,
    scale: float,
    base_index: int,
) -> int:
    """Write 3-axis cross markers centered at points as line segments. Returns new base index."""
    idx = base_index
    for p in points:
        px, py, pz = float(p[0]), float(p[1]), float(p[2])
        verts = [
            (px - scale, py, pz), (px + scale, py, pz),
            (px, py - scale, pz), (px, py + scale, pz),
            (px, py, pz - scale), (px, py, pz + scale),
        ]
        for v in verts:
            f.write(f"v {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n")
        f.write(f"l {idx+1} {idx+2}\n")
        f.write(f"l {idx+3} {idx+4}\n")
        f.write(f"l {idx+5} {idx+6}\n")
        idx += 6
    return idx


def _face_centers_and_normals(verts: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tri = verts[faces]  # [F, 3, 3]
    centers = tri.mean(axis=1)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    n_norm = np.where(n_norm > 1e-12, n_norm, 1.0)
    normals = normals / n_norm
    return centers, normals


def _write_normals_obj(
    path: Path,
    verts: np.ndarray,
    faces: np.ndarray,
    object_name: str,
    normal_scale: float,
) -> None:
    """Write mesh plus normal arrows (line segments) at face centers."""
    centers, normals = _face_centers_and_normals(verts, faces)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"o {object_name}\n")
        for v in verts:
            f.write(f"v {float(v[0]):.10f} {float(v[1]):.10f} {float(v[2]):.10f}\n")
        for tri in faces:
            f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")

        base = verts.shape[0]
        f.write(f"\no {object_name}_loss_normals\n")
        for c, n in zip(centers, normals):
            p0 = c
            p1 = c + n * normal_scale
            f.write(f"v {float(p0[0]):.10f} {float(p0[1]):.10f} {float(p0[2]):.10f}\n")
            f.write(f"v {float(p1[0]):.10f} {float(p1[1]):.10f} {float(p1[2]):.10f}\n")
        for i in range(centers.shape[0]):
            i0 = base + 2 * i + 1
            i1 = base + 2 * i + 2
            f.write(f"l {i0} {i1}\n")


def _chamfer_vertices(a: np.ndarray, b: np.ndarray) -> float:
    # Small opening meshes -> direct pairwise distance is fine.
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(d.min(axis=1).mean() + d.min(axis=0).mean())


def _extract_openings(chk: dict[str, Any]) -> List[Tuple[np.ndarray, np.ndarray]]:
    verts_list = [np.asarray(x, dtype=np.float64) for x in chk["op_rec_v"]]
    faces_list = [np.asarray(x, dtype=np.int64) for x in chk["op_rec_f"]]
    if len(verts_list) != len(faces_list):
        raise ValueError("Mismatch between number of opening vertices and faces")
    return list(zip(verts_list, faces_list))


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    target_dir = root / args.target
    out_dir = args.out_dir.resolve() if args.out_dir is not None else (root / args.target / "inspection_opening_planes")
    out_dir.mkdir(parents=True, exist_ok=True)

    target_chk = _load_opa(target_dir / "opa_checkpoint.pkl")
    target_openings = _extract_openings(target_chk)

    if args.target_only:
        if not target_openings:
            raise ValueError("Target has no opening planes")
        if args.target_index < 0 or args.target_index >= len(target_openings):
            raise IndexError(f"target-index {args.target_index} out of range [0, {len(target_openings) - 1}]")

        t_idx = args.target_index
        t_verts, t_faces = target_openings[t_idx]
        _write_obj(out_dir / "target_opening_plane.obj", t_verts, t_faces, f"target_opening_{t_idx}")

        target_part_path = target_dir / "part_aligned.obj"
        if target_part_path.exists():
            tp_v, tp_f = _load_obj_vertices_faces(target_part_path)
            _write_combined_part_and_opening(
                out_dir / "target_part_plus_opening_plane.obj",
                tp_v,
                tp_f,
                t_verts,
                t_faces,
                f"target_part_{args.target}",
                f"target_opening_{t_idx}",
            )

        summary = {
            "mode": "target_only",
            "root": str(root),
            "target": args.target,
            "target_openings": len(target_openings),
            "target_selected_index": t_idx,
            "output_dir": str(out_dir),
        }
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(json.dumps(summary, indent=2))
        print(f"\nSaved opening plane exports to: {out_dir}")
        return 0

    if not args.canonical:
        raise SystemExit("--canonical is required unless --target-only is set.")
    canonical_dir = root / args.canonical
    canonical_chk = _load_opa(canonical_dir / "opa_checkpoint.pkl")

    canonical_openings = _extract_openings(canonical_chk)

    if not canonical_openings:
        raise ValueError("Canonical has no opening planes")
    if not target_openings:
        raise ValueError("Target has no opening planes")
    if args.canonical_index < 0 or args.canonical_index >= len(canonical_openings):
        raise IndexError(f"canonical-index {args.canonical_index} out of range [0, {len(canonical_openings) - 1}]")

    c_idx = args.canonical_index
    c_verts, c_faces = canonical_openings[c_idx]

    # Match target opening by chamfer over opening-plane vertices.
    target_scores = []
    for i, (t_verts, _) in enumerate(target_openings):
        target_scores.append((i, _chamfer_vertices(c_verts, t_verts)))
    target_scores.sort(key=lambda x: x[1])
    t_idx = target_scores[0][0]
    t_verts, t_faces = target_openings[t_idx]

    _write_obj(out_dir / "canonical_opening_selected.obj", c_verts, c_faces, f"canonical_opening_{c_idx}")
    _write_obj(out_dir / "target_opening_matched.obj", t_verts, t_faces, f"target_opening_{t_idx}")
    _write_normals_obj(
        out_dir / "canonical_opening_selected_with_loss_normals.obj",
        c_verts,
        c_faces,
        f"canonical_opening_{c_idx}",
        args.normal_scale,
    )
    _write_normals_obj(
        out_dir / "target_opening_matched_with_loss_normals.obj",
        t_verts,
        t_faces,
        f"target_opening_{t_idx}",
        args.normal_scale,
    )

    # Also export all target openings for manual inspection.
    for i, (v, f) in enumerate(target_openings):
        _write_obj(out_dir / f"target_opening_{i:02d}.obj", v, f, f"target_opening_{i}")

    # Combined files: part_aligned + selected plane in one OBJ.
    canonical_part_path = canonical_dir / "part_aligned.obj"
    target_part_path = target_dir / "part_aligned.obj"
    if canonical_part_path.exists():
        cp_v, cp_f = _load_obj_vertices_faces(canonical_part_path)
        _write_combined_part_and_opening(
            out_dir / "canonical_part_plus_selected_opening_plane.obj",
            cp_v,
            cp_f,
            c_verts,
            c_faces,
            f"canonical_part_{args.canonical}",
            f"canonical_opening_{c_idx}",
        )
    if target_part_path.exists():
        tp_v, tp_f = _load_obj_vertices_faces(target_part_path)
        _write_combined_part_and_opening(
            out_dir / "target_part_plus_matched_opening_plane.obj",
            tp_v,
            tp_f,
            t_verts,
            t_faces,
            f"target_part_{args.target}",
            f"target_opening_{t_idx}",
        )

    # Mapping debug for canonical: compare opening plane vertices to mapped part vertices.
    mapping_debug = {"present": False}
    if canonical_part_path.exists():
        cp_v, cp_f = _load_obj_vertices_faces(canonical_part_path)
        op_map_all = canonical_chk.get("op_rec_v_indices_map", [])
        if c_idx < len(op_map_all):
            op_map = np.asarray(op_map_all[c_idx], dtype=np.int64).reshape(-1)
            valid_mask = (op_map >= 0) & (op_map < cp_v.shape[0])
            valid_ids = op_map[valid_mask]
            # Align with opening vertices by shared prefix length after validity filtering.
            c_use = c_verts[: valid_ids.shape[0]]
            m_use = cp_v[valid_ids]
            dists = np.linalg.norm(m_use - c_use, axis=1)
            sort_idx = np.argsort(-dists)  # descending error
            top_k = min(10, dists.shape[0])
            worst_local = sort_idx[:top_k]

            mapping_debug = {
                "present": True,
                "opening_index": int(c_idx),
                "num_mapped_vertices": int(valid_ids.shape[0]),
                "dist_min": float(dists.min()) if dists.size else 0.0,
                "dist_mean": float(dists.mean()) if dists.size else 0.0,
                "dist_max": float(dists.max()) if dists.size else 0.0,
                "worst_vertices": [
                    {
                        "rank": int(i + 1),
                        "local_id": int(worst_local[i]),
                        "mapped_part_vertex_id": int(valid_ids[worst_local[i]]),
                        "distance": float(dists[worst_local[i]]),
                    }
                    for i in range(top_k)
                ],
            }

            # Export one combined debug OBJ with:
            # 1) canonical part mesh
            # 2) selected opening plane
            # 3) line segments from mapped part vertex -> opening plane vertex
            # 4) cross markers on top-k worst mapped part vertices
            dbg_path = out_dir / "canonical_mapping_debug_part_plane_links.obj"
            with open(dbg_path, "w", encoding="utf-8") as f:
                f.write(f"o canonical_part_{args.canonical}\n")
                for v in cp_v:
                    f.write(f"v {float(v[0]):.10f} {float(v[1]):.10f} {float(v[2]):.10f}\n")
                for tri in cp_f:
                    f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")
                part_count = cp_v.shape[0]

                f.write(f"\no canonical_opening_{c_idx}\n")
                for v in c_verts:
                    f.write(f"v {float(v[0]):.10f} {float(v[1]):.10f} {float(v[2]):.10f}\n")
                for tri in c_faces:
                    f.write(
                        f"f {int(tri[0]) + 1 + part_count} "
                        f"{int(tri[1]) + 1 + part_count} "
                        f"{int(tri[2]) + 1 + part_count}\n"
                    )
                open_count = c_verts.shape[0]

                f.write("\no mapping_links\n")
                link_base = part_count + open_count
                for i in range(valid_ids.shape[0]):
                    p0 = m_use[i]
                    p1 = c_use[i]
                    f.write(f"v {float(p0[0]):.10f} {float(p0[1]):.10f} {float(p0[2]):.10f}\n")
                    f.write(f"v {float(p1[0]):.10f} {float(p1[1]):.10f} {float(p1[2]):.10f}\n")
                for i in range(valid_ids.shape[0]):
                    f.write(f"l {link_base + 2*i + 1} {link_base + 2*i + 2}\n")

                f.write("\no mapping_worst_markers\n")
                marker_points = m_use[worst_local] if top_k > 0 else np.zeros((0, 3), dtype=np.float64)
                mesh_extent = np.max(cp_v, axis=0) - np.min(cp_v, axis=0)
                marker_scale = 0.004 * float(np.linalg.norm(mesh_extent))
                _write_cross_markers(
                    f,
                    marker_points,
                    marker_scale,
                    base_index=link_base + 2 * valid_ids.shape[0],
                )

    # Overlay file for quick visual comparison.
    overlay_path = out_dir / "overlay_canonical_selected_vs_target_matched.obj"
    with open(overlay_path, "w", encoding="utf-8") as out:
        out.write(f"o canonical_opening_{c_idx}\n")
        for v in c_verts:
            out.write(f"v {float(v[0]):.10f} {float(v[1]):.10f} {float(v[2]):.10f}\n")
        for tri in c_faces:
            out.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")
        off = c_verts.shape[0]
        out.write(f"\no target_opening_{t_idx}\n")
        for v in t_verts:
            out.write(f"v {float(v[0]):.10f} {float(v[1]):.10f} {float(v[2]):.10f}\n")
        for tri in t_faces:
            out.write(f"f {int(tri[0]) + 1 + off} {int(tri[1]) + 1 + off} {int(tri[2]) + 1 + off}\n")

    summary = {
        "root": str(root),
        "canonical": args.canonical,
        "target": args.target,
        "canonical_openings": len(canonical_openings),
        "target_openings": len(target_openings),
        "canonical_selected_index": c_idx,
        "target_matched_index": t_idx,
        "target_match_scores": [{"index": i, "chamfer_vertices": s} for i, s in target_scores],
        "normal_scale": float(args.normal_scale),
        "canonical_mapping_debug": mapping_debug,
        "output_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved opening plane exports to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
