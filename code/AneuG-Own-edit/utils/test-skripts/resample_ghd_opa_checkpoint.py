#!/usr/bin/env python3
"""Resample a one-opening GHD OPA checkpoint to a fixed number of mesh vertices."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from create_ghd_opa_from_boundary import triangulate_boundary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="C0066")
    parser.add_argument("--ghd-root", type=Path, default=Path("checkpoint-v2/ghd_fitting_split_real"))
    parser.add_argument("--points", type=int, default=20)
    parser.add_argument("--force", type=int, default=0)
    return parser.parse_args()


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def ring_perimeter_positions(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    diffs = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(diffs, axis=1)
    if np.all(lengths < 1e-8):
        raise ValueError("Degenerate ring")
    cum = np.concatenate(([0.0], np.cumsum(lengths)))
    return lengths, cum, float(cum[-1])


def resample_indices_by_arclength(points: np.ndarray, count: int) -> np.ndarray:
    if count < 3:
        raise ValueError("--points must be >= 3")
    if points.shape[0] < count:
        raise ValueError(f"Cannot select {count} points from ring with only {points.shape[0]} vertices")

    _, cum, total = ring_perimeter_positions(points)
    targets = np.linspace(0.0, total, num=count, endpoint=False)
    vertex_positions = cum[:-1]
    selected = []
    used = set()
    for target in targets:
        circular_dist = np.abs(vertex_positions - target)
        circular_dist = np.minimum(circular_dist, total - circular_dist)
        order = np.argsort(circular_dist)
        chosen = None
        for idx in order:
            idx_i = int(idx)
            if idx_i not in used:
                chosen = idx_i
                break
        if chosen is None:
            raise RuntimeError("Could not choose unique resampled indices")
        used.add(chosen)
        selected.append(chosen)
    return np.asarray(selected, dtype=np.int64)


def safe_unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def main() -> None:
    args = parse_args()
    path = args.ghd_root / args.case / "opa_checkpoint.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    if not args.force:
        raise RuntimeError("Use --force 1 to overwrite the checkpoint after creating a backup.")

    chk = load_pickle(path)
    old_indices = np.asarray(chk["op_v_indices"][0], dtype=np.int64)
    old_coords = np.asarray(chk["op_v_coords"][0], dtype=np.float32)
    old_normals = np.asarray(chk["op_v_normal"][0], dtype=np.float32)
    if old_indices.ndim != 1 or old_coords.ndim != 2 or old_coords.shape[1] != 3:
        raise ValueError(f"Invalid OPA checkpoint shape in {path}")
    if old_indices.shape[0] != old_coords.shape[0]:
        raise ValueError("op_v_indices and op_v_coords have different lengths")

    local_selection = resample_indices_by_arclength(old_coords, args.points)
    new_indices = old_indices[local_selection]
    new_coords = old_coords[local_selection]
    new_normals = old_normals[local_selection]
    rec_faces = triangulate_boundary(new_coords)
    rec_faces_map = new_indices[rec_faces]

    new_chk = dict(chk)
    new_chk.update(
        {
            "source": f"{chk.get('source', 'unknown')}_resampled_{args.points}",
            "resampled_from_points": int(old_coords.shape[0]),
            "resampled_to_points": int(args.points),
            "resampled_at": datetime.now().isoformat(timespec="seconds"),
            "op_v_indices": [new_indices.astype(np.int64).tolist()],
            "op_v_coords": [new_coords.astype(np.float32)],
            "op_v_normal": [new_normals.astype(np.float32)],
            "op_n_mean": [safe_unit(new_normals.mean(axis=0))],
            "op_rec_v": [new_coords.astype(np.float32)],
            "op_rec_f": [rec_faces.astype(np.int64)],
            "op_rec_f_map": [rec_faces_map.astype(np.int64)],
            "op_rec_v_indices_map": [new_indices.astype(np.int64)],
        }
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}.backup_before_resample_{args.points}_{stamp}{path.suffix}")
    shutil.copy2(path, backup)
    with path.open("wb") as handle:
        pickle.dump(new_chk, handle)

    bbox = new_coords.max(axis=0) - new_coords.min(axis=0)
    summary = {
        "case": args.case,
        "path": str(path),
        "backup_path": str(backup),
        "old_points": int(old_coords.shape[0]),
        "new_points": int(new_coords.shape[0]),
        "span": float(np.linalg.norm(bbox)),
        "bbox_min": new_coords.min(axis=0).round(8).tolist(),
        "bbox_max": new_coords.max(axis=0).round(8).tolist(),
        "source": new_chk["source"],
    }
    summary_path = path.with_name(f"{path.stem}.resample_{args.points}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
