#!/usr/bin/env python3
"""Rebuild missing-mesh GHD OPA checkpoints from ghb_fitting_checkpoint.pkl replay."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from create_ghd_opa_from_boundary import boundary_components, order_boundary_loop, safe_unit, triangulate_boundary, vertex_normals
from resample_ghd_opa_checkpoint import resample_indices_by_arclength
from utils.inspect.vae_inspect_stage1_ostium_conditional import (
    build_ghd_reconstruct,
    mesh_to_numpy,
    reconstruct_fitted_mesh_from_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-csv", type=Path, required=True)
    parser.add_argument("--ghd-root", type=Path, default=Path("checkpoint-v2/ghd_fitting_split_real"))
    parser.add_argument("--canonical-root", type=Path, default=Path("alignment_vc/canonical_model"))
    parser.add_argument("--run", default="vanilla")
    parser.add_argument("--points", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vae_optimization_results/vae_opt_20260428_full/opa_rebuild_boundary20_missing_reconstructed"),
    )
    parser.add_argument("--force", type=int, default=0)
    return parser.parse_args()


def load_failed_cases(path: Path) -> list[str]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    return [row["case"] for row in rows if row.get("status") == "failed"]


def build_checkpoint_from_mesh(case: str, verts: np.ndarray, faces: np.ndarray, points: int) -> tuple[dict, dict]:
    components, adj = boundary_components(faces)
    if len(components) != 1:
        raise ValueError(f"Expected exactly one boundary component, found {len(components)}")
    boundary_idx = order_boundary_loop(components[0], adj)
    boundary_points = verts[boundary_idx]
    local_selection = resample_indices_by_arclength(boundary_points, points)
    op_indices = boundary_idx[local_selection]
    op_coords = boundary_points[local_selection]
    normals = vertex_normals(verts.astype(np.float32), faces.astype(np.int64))
    op_normals = normals[op_indices]
    rec_faces = triangulate_boundary(op_coords)
    rec_faces_map = op_indices[rec_faces]
    bbox = op_coords.max(axis=0) - op_coords.min(axis=0)
    chk = {
        "label": case,
        "source": f"ghd_checkpoint_reconstructed_boundary_loop_resampled_{points}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "boundary_original_points": int(boundary_idx.size),
        "op_v_indices": [op_indices.astype(np.int64).tolist()],
        "op_v_coords": [op_coords.astype(np.float32)],
        "op_v_normal": [op_normals.astype(np.float32)],
        "op_n_mean": [safe_unit(op_normals.mean(axis=0))],
        "op_rec_v": [op_coords.astype(np.float32)],
        "op_rec_f": [rec_faces.astype(np.int64)],
        "op_rec_f_map": [rec_faces_map.astype(np.int64)],
        "op_rec_v_indices_map": [op_indices.astype(np.int64)],
    }
    row = {
        "case": case,
        "status": "ok",
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
    if not args.force:
        raise RuntimeError("Use --force 1 to overwrite checkpoints after backup.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_failed_cases(args.failed_csv)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ghd_reconstruct = build_ghd_reconstruct(args.canonical_root, device)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = args.ghd_root / f"_opa_checkpoint_backups_boundary20_reconstructed_{stamp}"
    rows = []
    ok = 0
    failed = 0
    for i, case in enumerate(cases, start=1):
        checkpoint_path = args.ghd_root / case / args.run / "ghb_fitting_checkpoint.pkl"
        out_path = args.ghd_root / case / "opa_checkpoint.pkl"
        try:
            with checkpoint_path.open("rb") as handle:
                ghd_chk = pickle.load(handle)
            mesh = reconstruct_fitted_mesh_from_checkpoint(ghd_reconstruct, ghd_chk, device)
            verts, faces = mesh_to_numpy(mesh)
            chk, row = build_checkpoint_from_mesh(case, verts.astype(np.float32), faces.astype(np.int64), args.points)
            row["ghd_checkpoint"] = str(checkpoint_path)
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
                    "boundary_original_points": "",
                    "opa_points": "",
                    "span": "",
                    "bbox_min": "",
                    "bbox_max": "",
                    "source": "",
                    "error": repr(exc),
                    "ghd_checkpoint": str(checkpoint_path),
                    "backup_path": "",
                }
            )
            failed += 1
        if i % 20 == 0 or i == len(cases):
            print(f"[{i}/{len(cases)}] rebuilt={ok} failed={failed}", flush=True)

    csv_path = args.output_dir / "rebuild_missing_summary.csv"
    fieldnames = [
        "case",
        "status",
        "boundary_original_points",
        "opa_points",
        "span",
        "bbox_min",
        "bbox_max",
        "source",
        "error",
        "ghd_checkpoint",
        "backup_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "failed_csv": str(args.failed_csv),
        "cases_total": int(len(cases)),
        "rebuilt": int(ok),
        "failed": int(failed),
        "points": int(args.points),
        "backup_root": str(backup_root),
        "csv": str(csv_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
