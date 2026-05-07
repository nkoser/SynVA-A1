#!/usr/bin/env python3
"""Overlay exactly two OBJ meshes and export a clean side-view render.

Usage:
  python utils/inspect/overlay_objs.py \
    --obj path/wireframe.obj --obj path/surface.obj \
    --out-pdf overlay_side_view.pdf

Fixed render style:
  mesh[0] -> saturated blue wireframe
  mesh[1] -> gray slightly transparent surface

No interactive UI window is opened; this script is terminal/export oriented.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay 2 OBJ meshes and export a side-view PDF/PNG.")
    parser.add_argument(
        "--obj",
        type=Path,
        action="append",
        required=True,
        help="Path to OBJ mesh. Use exactly twice: --obj first.obj --obj second.obj",
    )
    parser.add_argument(
        "--second-opacity",
        type=float,
        default=0.9,
        help="Opacity in [0, 1] for mesh[1] (gray surface).",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Center each mesh at origin before plotting (debug helper).",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="white",
        help="PyVista background color (e.g. white, black).",
    )
    parser.add_argument(
        "--out-obj",
        type=Path,
        default=None,
        help="Optional path to save all overlaid meshes as one merged OBJ.",
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=Path("overlay_side_view.pdf"),
        help="Path to save scalable side-view PDF (default: overlay_side_view.pdf).",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="Optional path to save a PNG screenshot of the same render.",
    )
    return parser.parse_args()


def load_mesh(obj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np
    import trimesh

    mesh = trimesh.load(obj_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if verts.size == 0 or faces.size == 0:
        raise ValueError(f"Empty mesh: {obj_path}")
    return verts, faces


def to_pv_faces(faces: np.ndarray) -> np.ndarray:
    import numpy as np

    n = faces.shape[0]
    return np.hstack([np.full((n, 1), 3, dtype=np.int64), faces]).reshape(-1)


def resolve_output_path(out_arg: Path, default_name: str, expected_suffix: str, arg_name: str) -> Path:
    out_path = out_arg.expanduser()
    if out_path.exists() and out_path.is_dir():
        return (out_path / default_name).resolve()
    if out_path.suffix == "":
        return out_path.with_suffix(expected_suffix).resolve()
    if out_path.suffix.lower() != expected_suffix:
        raise SystemExit(f"{arg_name} must end with {expected_suffix} (got: {out_path})")
    return out_path.resolve()


def resolve_out_obj_path(out_arg: Path) -> Path:
    return resolve_output_path(
        out_arg=out_arg, default_name="merged_overlay.obj", expected_suffix=".obj", arg_name="--out-obj"
    )


def resolve_out_pdf_path(out_arg: Path) -> Path:
    return resolve_output_path(
        out_arg=out_arg,
        default_name="overlay_side_view.pdf",
        expected_suffix=".pdf",
        arg_name="--out-pdf",
    )


def resolve_out_png_path(out_arg: Path) -> Path:
    return resolve_output_path(
        out_arg=out_arg, default_name="overlay_side_view.png", expected_suffix=".png", arg_name="--out-png"
    )


def main() -> int:
    args = parse_args()
    if len(args.obj) != 2:
        raise SystemExit(
            "Please provide exactly two meshes via --obj. "
            "mesh[0]=blue wireframe, mesh[1]=gray transparent surface."
        )
    if not (0.0 <= args.second_opacity <= 1.0):
        raise SystemExit("--second-opacity must be in [0, 1].")

    obj_paths = [p.resolve() for p in args.obj]
    for p in obj_paths:
        if not p.exists():
            raise SystemExit(f"OBJ not found: {p}")

    import pyvista as pv
    import trimesh

    out_pdf = resolve_out_pdf_path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    pl = pv.Plotter(off_screen=True)
    pl.set_background(args.background)
    merged_meshes = []

    for idx, obj_path in enumerate(obj_paths):
        verts, faces = load_mesh(obj_path)
        centroid = verts.mean(axis=0)
        if args.center:
            verts = verts - centroid.reshape(1, 3)
        poly = pv.PolyData(verts, to_pv_faces(faces))
        merged_meshes.append(trimesh.Trimesh(vertices=verts, faces=faces, process=False))

        if idx == 0:
            continue
            pl.add_mesh(
                poly,
                color="#0057ff",
                style="wireframe",
                line_width=2.0,
                opacity=1.0,
                lighting=False,
            )
            mesh_style = "blue-wireframe"
            mesh_opacity = 1.0
        else:
            pl.add_mesh(
                poly,
                color="#9c9c9c",
                opacity=args.second_opacity,
                smooth_shading=True,
                show_edges=False,
            )
            mesh_style = "gray-surface"
            mesh_opacity = args.second_opacity

        bounds = poly.bounds
        print(
            f"[{idx}] {obj_path}\n"
            f"    verts={verts.shape[0]} faces={faces.shape[0]}\n"
            f"    style={mesh_style}\n"
            f"    render_opacity={mesh_opacity:.3f}\n"
            f"    centroid=({centroid[0]:.6f}, {centroid[1]:.6f}, {centroid[2]:.6f})\n"
            f"    bounds=({bounds[0]:.6f}, {bounds[1]:.6f}) x "
            f"({bounds[2]:.6f}, {bounds[3]:.6f}) x ({bounds[4]:.6f}, {bounds[5]:.6f})"
        )

    pl.view_yz()
    pl.reset_camera()
    pl.render()

    if not hasattr(pl, "save_graphic"):
        raise SystemExit("Installed PyVista version does not support PDF export via save_graphic().")
    pl.save_graphic(str(out_pdf), raster=False, painter=True)
    print(f"\nSaved scalable overlay PDF (vector): {out_pdf}")

    if args.out_png is not None:
        out_png = resolve_out_png_path(args.out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        pl.screenshot(str(out_png))
        print(f"Saved overlay PNG: {out_png}")

    if args.out_obj is not None:
        out_obj = resolve_out_obj_path(args.out_obj)
        out_obj.parent.mkdir(parents=True, exist_ok=True)
        merged = trimesh.util.concatenate(merged_meshes)
        merged.export(str(out_obj), file_type="obj")
        print(f"\nSaved merged OBJ: {out_obj}")

    pl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
