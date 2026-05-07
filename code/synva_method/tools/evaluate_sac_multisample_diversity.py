#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sac_roots",
        nargs="+",
        required=True,
        help="One or more run roots containing sac_samples/<variant>/cases/test/<case>/outputs/stage1_sample/*.obj",
    )
    p.add_argument(
        "--real_json",
        default="outputs/real_vs_generated_sac_diversity_20260503.json",
        help="Existing JSON containing real_test_sacs statistics. Used for ratio table.",
    )
    p.add_argument("--out_json", required=True)
    p.add_argument("--out_csv", required=True)
    return p.parse_args()


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx: list[int] = []
                for token in line.split()[1:]:
                    idx.append(int(token.split("/")[0]) - 1)
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) > 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
    if not vertices or not faces:
        raise ValueError(f"Could not read vertices/faces from {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def mesh_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def stat(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {k: float("nan") for k in ["mean", "median", "p10", "p90", "min", "max"]}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def cv(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or abs(float(arr.mean())) < 1e-12:
        return float("nan")
    return float(arr.std(ddof=0) / arr.mean())


def variant_case_samples(run_root: Path) -> dict[str, dict[str, list[Path]]]:
    out: dict[str, dict[str, list[Path]]] = {}
    sac_root = run_root / "sac_samples"
    if not sac_root.exists():
        raise FileNotFoundError(f"Missing sac_samples directory: {sac_root}")
    for variant_dir in sorted(p for p in sac_root.iterdir() if p.is_dir()):
        variant = variant_dir.name
        case_root = variant_dir / "cases" / "test"
        if not case_root.exists():
            continue
        for case_dir in sorted(p for p in case_root.iterdir() if p.is_dir()):
            sample_dir = case_dir / "outputs" / "stage1_sample"
            samples = sorted(sample_dir.glob("*_sample_*_raw.obj"))
            if samples:
                out.setdefault(variant, {})[case_dir.name] = samples
    return out


def evaluate_variant(case_to_samples: dict[str, list[Path]]) -> dict[str, object]:
    pair_rms: list[float] = []
    pair_rms_max_per_case: list[float] = []
    pair_centroid: list[float] = []
    pair_area_rel: list[float] = []
    area_cvs: list[float] = []
    extent_cvs: list[float] = []
    areas_all: list[float] = []
    extents_all: list[float] = []
    n_samples = 0

    for case, sample_paths in sorted(case_to_samples.items()):
        if len(sample_paths) < 2:
            continue
        verts: list[np.ndarray] = []
        areas: list[float] = []
        extents: list[float] = []
        centroids: list[np.ndarray] = []
        faces_ref: np.ndarray | None = None
        for path in sample_paths:
            v, f = read_obj(path)
            if faces_ref is None:
                faces_ref = f
            if verts and v.shape != verts[0].shape:
                raise ValueError(f"Vertex count mismatch in case {case}: {path}")
            verts.append(v)
            areas.append(mesh_area(v, f))
            extents.append(float(np.linalg.norm(v.max(axis=0) - v.min(axis=0))))
            centroids.append(v.mean(axis=0))

        case_pair_rms: list[float] = []
        for i, j in combinations(range(len(verts)), 2):
            rms = float(np.sqrt(np.mean(np.sum((verts[i] - verts[j]) ** 2, axis=1))))
            case_pair_rms.append(rms)
            pair_centroid.append(float(np.linalg.norm(centroids[i] - centroids[j])))
            denom = 0.5 * (areas[i] + areas[j])
            pair_area_rel.append(float(abs(areas[i] - areas[j]) / denom) if denom > 1e-12 else float("nan"))
        pair_rms.extend(case_pair_rms)
        pair_rms_max_per_case.append(float(max(case_pair_rms)))
        area_cvs.append(cv(areas))
        extent_cvs.append(cv(extents))
        areas_all.extend(areas)
        extents_all.extend(extents)
        n_samples += len(sample_paths)

    return {
        "n_cases": len(case_to_samples),
        "n_samples": n_samples,
        "within_case_pair_vertex_rms": stat(pair_rms),
        "within_case_pair_vertex_rms_max_per_case": stat(pair_rms_max_per_case),
        "within_case_pair_centroid": stat(pair_centroid),
        "within_case_pair_area_relative": stat(pair_area_rel),
        "within_case_area_cv": stat(area_cvs),
        "within_case_extent_norm_cv": stat(extent_cvs),
        "sample_area": stat(areas_all),
        "sample_extent_norm": stat(extents_all),
    }


def ratios(generated: dict[str, object], real: dict[str, object]) -> dict[str, dict[str, float]]:
    real_vertex = float(real["cross_case_pair_vertex_rms"]["mean"])
    real_centroid = float(real["cross_case_pair_centroid"]["mean"])
    real_area_rel = float(real["cross_case_pair_area_relative"]["mean"])
    real_area = float(real["area"]["mean"])
    real_extent = float(real["extent_norm"]["mean"])
    out: dict[str, dict[str, float]] = {}
    for variant, metrics_obj in generated.items():
        metrics = metrics_obj  # type: ignore[assignment]
        out[variant] = {
            "within_vertex_rms_vs_real_crosscase": float(metrics["within_case_pair_vertex_rms"]["mean"]) / real_vertex,
            "within_centroid_vs_real_crosscase": float(metrics["within_case_pair_centroid"]["mean"]) / real_centroid,
            "within_area_rel_vs_real_crosscase": float(metrics["within_case_pair_area_relative"]["mean"]) / real_area_rel,
            "generated_mean_area_vs_real_mean_area": float(metrics["sample_area"]["mean"]) / real_area,
            "generated_mean_extent_vs_real_mean_extent": float(metrics["sample_extent_norm"]["mean"]) / real_extent,
            "within_vertex_rms_percent_of_real_extent": 100.0
            * float(metrics["within_case_pair_vertex_rms"]["mean"])
            / real_extent,
        }
    return out


def write_csv(path: Path, generated: dict[str, object], comparison: dict[str, dict[str, float]]) -> None:
    rows: list[dict[str, object]] = []
    for variant, metrics_obj in sorted(generated.items()):
        metrics = metrics_obj  # type: ignore[assignment]
        comp = comparison[variant]
        rows.append(
            {
                "variant": variant,
                "n_cases": metrics["n_cases"],
                "n_samples": metrics["n_samples"],
                "vertex_rms_diversity": metrics["within_case_pair_vertex_rms"]["mean"],
                "max_pair_diversity": metrics["within_case_pair_vertex_rms_max_per_case"]["mean"],
                "centroid_diversity": metrics["within_case_pair_centroid"]["mean"],
                "relative_area_diversity": metrics["within_case_pair_area_relative"]["mean"],
                "area_cv": metrics["within_case_area_cv"]["mean"],
                "extent_cv": metrics["within_case_extent_norm_cv"]["mean"],
                "mean_area": metrics["sample_area"]["mean"],
                "mean_extent_norm": metrics["sample_extent_norm"]["mean"],
                "vertex_rms_vs_real_crosscase": comp["within_vertex_rms_vs_real_crosscase"],
                "centroid_vs_real_crosscase": comp["within_centroid_vs_real_crosscase"],
                "area_rel_vs_real_crosscase": comp["within_area_rel_vs_real_crosscase"],
                "mean_area_vs_real": comp["generated_mean_area_vs_real_mean_area"],
                "mean_extent_vs_real": comp["generated_mean_extent_vs_real_mean_extent"],
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    generated: dict[str, object] = {}
    for root_text in args.sac_roots:
        for variant, case_to_samples in variant_case_samples(Path(root_text)).items():
            if variant in generated:
                raise ValueError(f"Duplicate variant {variant}; pass unique variants or evaluate separately")
            generated[variant] = evaluate_variant(case_to_samples)

    real_json = json.loads(Path(args.real_json).read_text(encoding="utf-8"))
    real = real_json["real_test_sacs"]
    comparison = ratios(generated, real)
    result = {
        "real_test_sacs": real,
        "generated_multisample": generated,
        "comparison_ratios": comparison,
    }

    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(out_csv, generated, comparison)
    print(json.dumps({"out_json": str(out_json), "out_csv": str(out_csv), "variants": sorted(generated)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
