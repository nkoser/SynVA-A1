#!/usr/bin/env python3
"""Align canonical source annotations to alignment/canonical_model coordinates.

This script computes a rigid transform from:
  <canonical-src>/<source-mesh-name>
to:
  <alignment-case-dir>/<target-mesh-name>

Then it applies that transform to canonical inputs needed by
`create_opa_checkpoint_from_ostium.py`, writing a zero-style case folder
(`aneurysm_aligned.obj`, `04_subpointclouds/*.ply`, `07_other/*.npy`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Align canonical source data into alignment canonical frame for normal OPA generation."
    )
    p.add_argument("--canonical-src", type=Path, default=Path("checkpoints-new/canonical_model"))
    p.add_argument("--alignment-case-dir", type=Path, default=Path("checkpoints-new/alignment/canonical_model"))
    p.add_argument(
        "--output-zero-case",
        type=Path,
        default=Path("checkpoints-new/zero-aneurysmen/aneux_canonical_model"),
        help="Output folder in zero-aneurysmen style.",
    )
    p.add_argument("--source-mesh-name", type=str, default="part_aligned.obj")
    p.add_argument("--target-mesh-name", type=str, default="part_aligned.obj")
    p.add_argument(
        "--output-mesh-name",
        type=str,
        default="aneurysm_aligned.obj",
        help="Primary mesh name written into output-zero-case.",
    )
    p.add_argument("--subpointcloud-dir", type=str, default="04_subpointclouds")
    p.add_argument("--other-dir", type=str, default="07_other")
    p.add_argument(
        "--point-npy",
        action="append",
        default=["centroid_ostium.npy"],
        help="Point-valued .npy file in 07_other to transform with rotation+translation.",
    )
    p.add_argument(
        "--vector-npy",
        action="append",
        default=["normal_vector.npy"],
        help="Vector-valued .npy file in 07_other to transform with rotation only.",
    )
    p.add_argument(
        "--max-residual",
        type=float,
        default=1e-4,
        help="Maximum allowed source->target residual (in world units) when --strict is used.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if computed rigid transform residual exceeds --max-residual.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    p.add_argument("--dry-run", action="store_true", help="Only print transform stats and planned outputs.")
    return p.parse_args()


def _load_geometry(path: Path):
    geom = trimesh.load(path, process=False)
    if isinstance(geom, trimesh.Scene) and len(geom.geometry) == 1:
        geom = next(iter(geom.geometry.values()))
    return geom


def _load_mesh_vertices(path: Path) -> np.ndarray:
    mesh = trimesh.load_mesh(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return np.asarray(mesh.vertices, dtype=np.float64)


def _rigid_kabsch(src_pts: np.ndarray, dst_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_cent = src_pts.mean(axis=0)
    dst_cent = dst_pts.mean(axis=0)
    src_zm = src_pts - src_cent.reshape(1, 3)
    dst_zm = dst_pts - dst_cent.reshape(1, 3)

    h = src_zm.T @ dst_zm
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = dst_cent - (r @ src_cent)
    return r, t


def _transform_points(points: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return pts @ r.T + t.reshape(1, 3)


def _transform_vectors(vectors: np.ndarray, r: np.ndarray) -> np.ndarray:
    vec = np.asarray(vectors, dtype=np.float64).reshape(-1, 3)
    vec_t = vec @ r.T
    nrm = np.linalg.norm(vec_t, axis=1, keepdims=True)
    nrm = np.where(nrm > 1e-12, nrm, 1.0)
    return vec_t / nrm


def _transform_and_export_geometry(src_path: Path, dst_path: Path, r: np.ndarray, t: np.ndarray) -> None:
    geom = _load_geometry(src_path)
    if isinstance(geom, trimesh.Scene):
        for g in geom.geometry.values():
            if not hasattr(g, "vertices"):
                continue
            g.vertices = _transform_points(np.asarray(g.vertices, dtype=np.float64), r, t)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        geom.export(dst_path)
        return

    if not hasattr(geom, "vertices"):
        raise TypeError(f"Unsupported geometry type for {src_path}: {type(geom).__name__}")

    geom.vertices = _transform_points(np.asarray(geom.vertices, dtype=np.float64), r, t)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    geom.export(dst_path)


def _save_npy(dst: Path, arr: np.ndarray) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, arr)


def main() -> int:
    args = parse_args()
    canonical_src = args.canonical_src.resolve()
    alignment_case_dir = args.alignment_case_dir.resolve()
    output_zero_case = args.output_zero_case.resolve()

    src_mesh_path = canonical_src / args.source_mesh_name
    tgt_mesh_path = alignment_case_dir / args.target_mesh_name
    required = [src_mesh_path, tgt_mesh_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    src_v = _load_mesh_vertices(src_mesh_path)
    tgt_v = _load_mesh_vertices(tgt_mesh_path)
    if src_v.shape != tgt_v.shape:
        raise SystemExit(
            f"Canonical source/target vertex count mismatch: {src_v.shape[0]} vs {tgt_v.shape[0]}. "
            "This script requires vertex-wise correspondence."
        )

    r, t = _rigid_kabsch(src_v, tgt_v)
    src_v_aligned = _transform_points(src_v, r, t)
    residual = np.linalg.norm(src_v_aligned - tgt_v, axis=1)

    print("Rigid transform (source canonical -> alignment canonical)")
    print("Rotation:")
    print(np.array2string(r, precision=10, suppress_small=True))
    print(f"Translation: {np.array2string(t, precision=10, suppress_small=True)}")
    print(
        "Residual | mean={:.6e} max={:.6e} p95={:.6e}".format(
            float(residual.mean()),
            float(residual.max()),
            float(np.quantile(residual, 0.95)),
        )
    )

    if args.strict and float(residual.max()) > float(args.max_residual):
        raise SystemExit(
            f"Residual max {float(residual.max()):.6e} exceeds --max-residual {float(args.max_residual):.6e}"
        )

    sub_src_dir = canonical_src / args.subpointcloud_dir
    subpointcloud_files = sorted(sub_src_dir.glob("*.ply")) if sub_src_dir.is_dir() else []
    if not subpointcloud_files:
        raise SystemExit(f"No subpointcloud .ply files found under: {sub_src_dir}")

    point_names = sorted(set(args.point_npy))
    vector_names = sorted(set(args.vector_npy))
    other_src_dir = canonical_src / args.other_dir
    point_files = [other_src_dir / name for name in point_names]
    vector_files = [other_src_dir / name for name in vector_names]
    missing_anno = [str(p) for p in point_files + vector_files if not p.exists()]
    if missing_anno:
        raise SystemExit(f"Missing annotation files in {other_src_dir}: {missing_anno}")

    mesh_outputs = sorted(set([args.output_mesh_name, args.target_mesh_name]))
    planned = {
        "output_zero_case": str(output_zero_case),
        "mesh_outputs": mesh_outputs,
        "subpointcloud_outputs": [str(output_zero_case / args.subpointcloud_dir / p.name) for p in subpointcloud_files],
        "point_npy_outputs": [str(output_zero_case / args.other_dir / p.name) for p in point_files],
        "vector_npy_outputs": [str(output_zero_case / args.other_dir / p.name) for p in vector_files],
    }

    if args.dry_run:
        print("\nDry run: planned outputs")
        print(json.dumps(planned, indent=2))
        return 0

    output_zero_case.mkdir(parents=True, exist_ok=True)

    for mesh_name in mesh_outputs:
        dst_mesh = output_zero_case / mesh_name
        if dst_mesh.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {dst_mesh} (use --overwrite)")
        _transform_and_export_geometry(src_mesh_path, dst_mesh, r, t)

    for src_ply in subpointcloud_files:
        dst_ply = output_zero_case / args.subpointcloud_dir / src_ply.name
        if dst_ply.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {dst_ply} (use --overwrite)")
        _transform_and_export_geometry(src_ply, dst_ply, r, t)

    for src_npy in point_files:
        dst_npy = output_zero_case / args.other_dir / src_npy.name
        if dst_npy.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {dst_npy} (use --overwrite)")
        arr = np.asarray(np.load(src_npy), dtype=np.float64)
        if arr.shape[-1] != 3:
            raise SystemExit(f"Point npy must end with dim=3: {src_npy} shape={arr.shape}")
        arr_t = _transform_points(arr.reshape(-1, 3), r, t).reshape(arr.shape)
        _save_npy(dst_npy, arr_t)

    for src_npy in vector_files:
        dst_npy = output_zero_case / args.other_dir / src_npy.name
        if dst_npy.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing file: {dst_npy} (use --overwrite)")
        arr = np.asarray(np.load(src_npy), dtype=np.float64)
        if arr.shape[-1] != 3:
            raise SystemExit(f"Vector npy must end with dim=3: {src_npy} shape={arr.shape}")
        arr_t = _transform_vectors(arr.reshape(-1, 3), r).reshape(arr.shape)
        _save_npy(dst_npy, arr_t)

    summary = {
        "canonical_src": str(canonical_src),
        "alignment_case_dir": str(alignment_case_dir),
        "output_zero_case": str(output_zero_case),
        "source_mesh": str(src_mesh_path),
        "target_mesh": str(tgt_mesh_path),
        "rotation": r.tolist(),
        "translation": t.tolist(),
        "residual_mean": float(residual.mean()),
        "residual_max": float(residual.max()),
        "residual_p95": float(np.quantile(residual, 0.95)),
        "transformed_subpointclouds": [p.name for p in subpointcloud_files],
        "transformed_point_npy": point_names,
        "transformed_vector_npy": vector_names,
    }
    summary_path = output_zero_case / "canonical_alignment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote aligned canonical zero-case: {output_zero_case}")
    print(f"Summary: {summary_path}")
    print("\nNow you can run normal OPA generation with:")
    print(
        "python utils/create_opa_checkpoint_from_ostium.py "
        "--zero-root checkpoints-new/zero-aneurysmen "
        "--alignment-root checkpoints-new/alignment "
        "--case canonical_model --overwrite --flip-inside-normal"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
