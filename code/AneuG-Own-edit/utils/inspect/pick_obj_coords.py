#!/usr/bin/env python3
"""Interactive OBJ coordinate picker.

Usage:
  python utils/inspect/pick_obj_coords.py --obj path/to/mesh.obj

Controls:
  - Press `P` (or right click) to pick a point on the mesh.
  - Press `Q` to quit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick coordinates on an OBJ mesh.")
    parser.add_argument("--obj", type=Path, required=True, help="Path to .obj mesh")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional text file to append picks",
    )
    parser.add_argument("--point-size", type=float, default=14.0, help="Picked point marker size")
    return parser.parse_args()


def load_mesh(obj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(obj_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return verts, faces


def to_pv_faces(faces: np.ndarray) -> np.ndarray:
    # PyVista expects [3, i, j, k, 3, i, j, k, ...]
    n = faces.shape[0]
    return np.hstack([np.full((n, 1), 3, dtype=np.int64), faces]).reshape(-1)


def main() -> int:
    args = parse_args()
    obj_path = args.obj.resolve()
    if not obj_path.exists():
        raise SystemExit(f"OBJ not found: {obj_path}")

    verts, faces = load_mesh(obj_path)
    poly = pv.PolyData(verts, to_pv_faces(faces))

    picks: list[tuple[int, np.ndarray, float, np.ndarray]] = []
    kdtree = None
    try:
        from scipy.spatial import cKDTree

        kdtree = cKDTree(verts)
    except Exception:
        pass

    pl = pv.Plotter()
    pl.add_mesh(poly, color="lightgray", opacity=1.0, show_edges=False)
    pl.add_text("Pick with P / Right-click. Quit with Q.", position="upper_left", font_size=10)

    def _on_pick(point: np.ndarray, *_args) -> None:
        if point is None:
            return
        p = np.asarray(point).reshape(3)
        if kdtree is not None:
            dist, vid = kdtree.query(p, k=1)
        else:
            d = np.linalg.norm(verts - p.reshape(1, 3), axis=1)
            vid = int(np.argmin(d))
            dist = float(d[vid])
        v = verts[int(vid)]
        picks.append((int(vid), p, float(dist), v))
        msg = (
            f"[pick {len(picks)}] click=({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f}) | "
            f"nearest_vid={int(vid)} nearest_v=({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}) | dist={float(dist):.6e}"
        )
        print(msg)
        pl.add_points(p.reshape(1, 3), color="red", point_size=args.point_size, render_points_as_spheres=True)

    pl.enable_point_picking(
        callback=_on_pick,
        use_picker=True,
        show_message=True,
        show_point=True,
        left_clicking=False,
        color="red",
        point_size=args.point_size,
    )
    pl.show()

    if len(picks) == 0:
        print("No picks recorded.")
        return 0

    if args.out is not None:
        out_path = args.out.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(f"# Picks for {obj_path}\n")
            for i, (vid, p, dist, v) in enumerate(picks, start=1):
                f.write(
                    f"pick={i} click=({p[0]:.6f},{p[1]:.6f},{p[2]:.6f}) "
                    f"nearest_vid={vid} nearest_v=({v[0]:.6f},{v[1]:.6f},{v[2]:.6f}) dist={dist:.6e}\n"
                )
        print(f"Saved picks to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
