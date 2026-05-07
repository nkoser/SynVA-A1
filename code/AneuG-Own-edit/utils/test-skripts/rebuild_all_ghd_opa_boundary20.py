#!/usr/bin/env python3
"""Rebuild all GHD-condition OPA checkpoints from fitted mesh boundary loops."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from create_ghd_opa_from_boundary import (
    boundary_components,
    load_mesh,
    order_boundary_loop,
    safe_unit,
    triangulate_boundary,
    vertex_normals,
)
from resample_ghd_opa_checkpoint import resample_indices_by_arclength


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ghd-root", type=Path, default=Path("checkpoint-v2/ghd_fitting_split_real"))
    parser.add_argument("--run", default="vanilla")
    parser.add_argument("--mesh-name", default="warped_epoch_02999.obj")
    parser.add_argument("--points", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vae_optimization_results/vae_opt_20260428_full/opa_rebuild_boundary20_all"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", type=int, default=0)
    return parser.parse_args()


def iter_cases(ghd_root: Path) -> list[str]:
    return sorted(path.name for path in ghd_root.iterdir() if path.is_dir() and not path.name.startswith("_"))


def build_checkpoint(case: str, mesh_path: Path, points: int) -> tuple[dict, dict]:
    verts, faces = load_mesh(mesh_path)
    components, adj = boundary_components(faces)
    if len(components) != 1:
        raise ValueError(f"Expected exactly one boundary component, found {len(components)}")
    boundary_idx = order_boundary_loop(components[0], adj)
    boundary_points = verts[boundary_idx]
    local_selection = resample_indices_by_arclength(boundary_points, points)

    op_indices = boundary_idx[local_selection]
    op_coords = boundary_points[local_selection]
    normals = vertex_normals(verts, faces)
    op_normals = normals[op_indices]
    op_n_mean = safe_unit(op_normals.mean(axis=0))
    rec_faces = triangulate_boundary(op_coords)
    rec_faces_map = op_indices[rec_faces]

    bbox = op_coords.max(axis=0) - op_coords.min(axis=0)
    chk = {
        "label": case,
        "source": f"ghd_warped_boundary_loop_resampled_{points}",
        "mesh_path": str(mesh_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "boundary_original_points": int(boundary_idx.size),
        "op_v_indices": [op_indices.astype(np.int64).tolist()],
        "op_v_coords": [op_coords.astype(np.float32)],
        "op_v_normal": [op_normals.astype(np.float32)],
        "op_n_mean": [op_n_mean.astype(np.float32)],
        "op_rec_v": [op_coords.astype(np.float32)],
        "op_rec_f": [rec_faces.astype(np.int64)],
        "op_rec_f_map": [rec_faces_map.astype(np.int64)],
        "op_rec_v_indices_map": [op_indices.astype(np.int64)],
    }
    row = {
        "case": case,
        "status": "ok",
        "mesh_path": str(mesh_path),
        "boundary_original_points": int(boundary_idx.size),
        "opa_points": int(op_coords.shape[0]),
        "span": float(np.linalg.norm(bbox)),
        "bbox_min": json.dumps(op_coords.min(axis=0).round(8).tolist()),
        "bbox_max": json.dumps(op_coords.max(axis=0).round(8).tolist()),
        "source": chk["source"],
        "error": "",
    }
    return chk, row


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = iter_cases(args.ghd_root)
    if args.limit > 0:
        cases = cases[: args.limit]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = args.ghd_root / f"_opa_checkpoint_backups_boundary20_{stamp}"
    rows = []
    ok = 0
    failed = 0
    for i, case in enumerate(cases, start=1):
        case_root = args.ghd_root / case
        out_path = case_root / "opa_checkpoint.pkl"
        mesh_path = case_root / args.run / "viz" / args.mesh_name
        if not mesh_path.exists():
            rows.append(
                {
                    "case": case,
                    "status": "failed",
                    "mesh_path": str(mesh_path),
                    "boundary_original_points": "",
                    "opa_points": "",
                    "span": "",
                    "bbox_min": "",
                    "bbox_max": "",
                    "source": "",
                    "error": "missing warped mesh",
                }
            )
            failed += 1
            continue
        if out_path.exists() and not args.force:
            raise FileExistsError(f"{out_path} exists. Use --force 1 to rebuild after backup.")

        try:
            chk, row = build_checkpoint(case, mesh_path, args.points)
            if out_path.exists():
                backup_path = backup_root / case / "opa_checkpoint.pkl"
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(out_path, backup_path)
                row["backup_path"] = str(backup_path)
            else:
                row["backup_path"] = ""
            with out_path.open("wb") as handle:
                pickle.dump(chk, handle)
            rows.append(row)
            ok += 1
        except Exception as exc:
            rows.append(
                {
                    "case": case,
                    "status": "failed",
                    "mesh_path": str(mesh_path),
                    "boundary_original_points": "",
                    "opa_points": "",
                    "span": "",
                    "bbox_min": "",
                    "bbox_max": "",
                    "source": "",
                    "backup_path": "",
                    "error": repr(exc),
                }
            )
            failed += 1

        if i % 50 == 0 or i == len(cases):
            print(f"[{i}/{len(cases)}] rebuilt={ok} failed={failed}", flush=True)

    csv_path = args.output_dir / "rebuild_summary.csv"
    fieldnames = [
        "case",
        "status",
        "mesh_path",
        "boundary_original_points",
        "opa_points",
        "span",
        "bbox_min",
        "bbox_max",
        "source",
        "backup_path",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "ghd_root": str(args.ghd_root),
        "points": int(args.points),
        "cases_total": int(len(cases)),
        "rebuilt": int(ok),
        "failed": int(failed),
        "backup_root": str(backup_root),
        "csv": str(csv_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
