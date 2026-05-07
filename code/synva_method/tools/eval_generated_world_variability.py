#!/usr/bin/env python
"""Evaluate variability of generated aneurysm world meshes.

The script expects roots that contain either:

  cases/test/<case>/outputs/final/*_generated_aneurysm_world.obj

or a multisample sweep layout:

  sample_00_seed1/cases/test/<case>/outputs/final/*_generated_aneurysm_world.obj
  sample_01_seed2/cases/test/<case>/outputs/final/*_generated_aneurysm_world.obj

It reports within-case pairwise diversity across samples. A single-sample root
is accepted but cannot produce within-case diversity metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "runs",
        nargs="+",
        help="name=root, e.g. W_morph=/path/to/output/reference_stitching/W_stage3surrogate_morph",
    )
    p.add_argument("--out_json", default=None)
    return p.parse_args()


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts = []
    faces = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = []
                for part in line.strip().split()[1:]:
                    idx.append(int(part.split("/")[0]) - 1)
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) > 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def surface_area(verts: np.ndarray, faces: np.ndarray) -> float:
    if faces.size == 0:
        return float("nan")
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return float(area.sum())


def extent_norm(verts: np.ndarray) -> float:
    if verts.size == 0:
        return float("nan")
    return float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))


def discover(root: Path) -> dict[str, dict[str, Path]]:
    sample_dirs = sorted(p for p in root.glob("sample_*") if p.is_dir())
    if not sample_dirs:
        sample_dirs = [root]
    out: dict[str, dict[str, Path]] = {}
    for sample_dir in sample_dirs:
        sample = sample_dir.name if sample_dir.name.startswith("sample_") else "sample_00"
        for path in sample_dir.glob("cases/*/*/outputs/final/*_generated_aneurysm_world.obj"):
            case = path.parents[2].name
            out.setdefault(case, {})[sample] = path
    return out


def summarize_variant(root: Path) -> dict[str, object]:
    index = discover(root)
    per_case = []
    all_areas = []
    all_extents = []
    all_centroids = []
    for case, samples in sorted(index.items()):
        meshes = {}
        stats = {}
        for sample, path in sorted(samples.items()):
            verts, faces = load_obj(path)
            meshes[sample] = (verts, faces)
            area = surface_area(verts, faces)
            ext = extent_norm(verts)
            centroid = verts.mean(axis=0)
            stats[sample] = {"area": area, "extent": ext, "centroid": centroid}
            all_areas.append(area)
            all_extents.append(ext)
            all_centroids.append(centroid)

        rms_pairs = []
        centroid_pairs = []
        rel_area_pairs = []
        extent_pairs = []
        for a, b in combinations(sorted(meshes), 2):
            va, _ = meshes[a]
            vb, _ = meshes[b]
            if va.shape == vb.shape:
                rms_pairs.append(float(np.sqrt(np.mean((va - vb) ** 2))))
            ca = stats[a]["centroid"]
            cb = stats[b]["centroid"]
            centroid_pairs.append(float(np.linalg.norm(ca - cb)))
            aa = stats[a]["area"]
            ab = stats[b]["area"]
            rel_area_pairs.append(float(abs(aa - ab) / max((aa + ab) * 0.5, 1e-12)))
            extent_pairs.append(float(abs(stats[a]["extent"] - stats[b]["extent"])))

        per_case.append(
            {
                "case": case,
                "n_samples": len(samples),
                "n_pairs": len(centroid_pairs),
                "vertex_rms_mean": float(np.mean(rms_pairs)) if rms_pairs else float("nan"),
                "vertex_rms_max": float(np.max(rms_pairs)) if rms_pairs else float("nan"),
                "centroid_pair_mean": float(np.mean(centroid_pairs)) if centroid_pairs else float("nan"),
                "relative_area_pair_mean": float(np.mean(rel_area_pairs)) if rel_area_pairs else float("nan"),
                "extent_pair_mean": float(np.mean(extent_pairs)) if extent_pairs else float("nan"),
            }
        )

    def mean_finite(key: str) -> float:
        vals = np.asarray([row[key] for row in per_case], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        return float(vals.mean()) if vals.size else float("nan")

    areas = np.asarray(all_areas, dtype=np.float64)
    extents = np.asarray(all_extents, dtype=np.float64)
    centroids = np.asarray(all_centroids, dtype=np.float64) if all_centroids else np.zeros((0, 3))
    return {
        "root": str(root),
        "n_cases": len(index),
        "n_meshes": int(len(all_areas)),
        "mean_samples_per_case": float(np.mean([len(v) for v in index.values()])) if index else 0.0,
        "within_case_vertex_rms": mean_finite("vertex_rms_mean"),
        "within_case_vertex_rms_max": mean_finite("vertex_rms_max"),
        "within_case_centroid_distance": mean_finite("centroid_pair_mean"),
        "within_case_relative_area_difference": mean_finite("relative_area_pair_mean"),
        "within_case_extent_difference": mean_finite("extent_pair_mean"),
        "area_mean": float(np.nanmean(areas)) if areas.size else float("nan"),
        "area_cv": float(np.nanstd(areas) / max(abs(np.nanmean(areas)), 1e-12)) if areas.size else float("nan"),
        "extent_mean": float(np.nanmean(extents)) if extents.size else float("nan"),
        "extent_cv": float(np.nanstd(extents) / max(abs(np.nanmean(extents)), 1e-12)) if extents.size else float("nan"),
        "centroid_global_std": float(np.sqrt(np.nanmean(np.var(centroids, axis=0)))) if len(centroids) else float("nan"),
        "per_case": per_case,
    }


def fmt(x: float) -> str:
    return "nan" if not math.isfinite(float(x)) else f"{float(x):.6f}"


def main() -> int:
    args = parse_args()
    results = {}
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"Expected name=root, got {spec}")
        name, root = spec.split("=", 1)
        results[name] = summarize_variant(Path(root))

    for name, row in results.items():
        print(
            f"{name:>28s} cases={row['n_cases']} meshes={row['n_meshes']} "
            f"samples/case={fmt(row['mean_samples_per_case'])} "
            f"vertex_rms={fmt(row['within_case_vertex_rms'])} "
            f"centroid={fmt(row['within_case_centroid_distance'])} "
            f"rel_area={fmt(row['within_case_relative_area_difference'])} "
            f"area_mean={fmt(row['area_mean'])} extent_mean={fmt(row['extent_mean'])}"
        )
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(results, indent=2, allow_nan=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
