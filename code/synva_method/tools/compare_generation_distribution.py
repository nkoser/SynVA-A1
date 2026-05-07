#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist, squareform


DEFAULT_FEATURES = [
    "A_A",
    "V_A",
    "A_CH",
    "V_CH",
    "D_max",
    "H_max",
    "W_max",
    "H_ortho",
    "W_ortho",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare generated aneurysm distributions against GT in morphology-feature space. "
            "This is an unpaired distribution test, complementary to paired GT Chamfer."
        )
    )
    p.add_argument(
        "--metric-csvs",
        nargs="+",
        required=True,
        help="name=per_case_metrics.csv entries from compare_reference_stitching_to_gt.py",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    p.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.95,
        help="Quantile of GT leave-one-out NN distances used for precision/coverage threshold.",
    )
    p.add_argument(
        "--mmd-scales",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0, 4.0],
        help="RBF bandwidth multipliers around the GT median pairwise distance.",
    )
    return p.parse_args()


def finite_float(value: str | float | int | None) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def load_metric_csv(spec: str) -> list[dict[str, str]]:
    if "=" not in spec:
        raise SystemExit(f"Expected name=csv, got {spec!r}")
    name, path_s = spec.split("=", 1)
    path = Path(path_s)
    if not path.exists():
        raise SystemExit(f"Missing metric CSV for {name}: {path}")
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("variant") and row.get("variant") != name:
                continue
            row = dict(row)
            row["_run_name"] = name
            rows.append(row)
    if not rows:
        raise SystemExit(f"No rows for run {name!r} in {path} (checked the 'variant' column)")
    return rows


def feature_vector(row: dict[str, str], prefix: str, features: list[str]) -> np.ndarray:
    return np.asarray([finite_float(row.get(f"{prefix}_{key}")) for key in features], dtype=np.float64)


def build_matrices(rows_by_run: dict[str, list[dict[str, str]]], features: list[str]):
    cases_by_run = {
        name: {row["case"]: row for row in rows if row.get("case")}
        for name, rows in rows_by_run.items()
    }
    common_cases = sorted(set.intersection(*(set(cases) for cases in cases_by_run.values())))
    if not common_cases:
        raise SystemExit("No common cases across runs")

    gt_by_case = {}
    for case in common_cases:
        for cases in cases_by_run.values():
            row = cases[case]
            vec = feature_vector(row, "gt", features)
            if np.all(np.isfinite(vec)):
                gt_by_case[case] = vec
                break

    usable_cases = []
    for case in common_cases:
        if case not in gt_by_case:
            continue
        ok = True
        for cases in cases_by_run.values():
            vec = feature_vector(cases[case], "pred", features)
            if not np.all(np.isfinite(vec)):
                ok = False
                break
        if ok:
            usable_cases.append(case)

    gt = np.vstack([gt_by_case[case] for case in usable_cases])
    pred_by_run = {
        name: np.vstack([feature_vector(cases[case], "pred", features) for case in usable_cases])
        for name, cases in cases_by_run.items()
    }
    return usable_cases, gt, pred_by_run


def standardize_by_gt(gt: np.ndarray, pred_by_run: dict[str, np.ndarray]):
    mean = gt.mean(axis=0, keepdims=True)
    std = gt.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return (gt - mean) / std, {name: (x - mean) / std for name, x in pred_by_run.items()}, mean, std


def mmd_rbf_unbiased(x: np.ndarray, y: np.ndarray, bandwidths: list[float]) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    dxx = cdist(x, x, "sqeuclidean")
    dyy = cdist(y, y, "sqeuclidean")
    dxy = cdist(x, y, "sqeuclidean")
    np.fill_diagonal(dxx, np.nan)
    np.fill_diagonal(dyy, np.nan)
    vals = []
    for bw in bandwidths:
        if not math.isfinite(bw) or bw <= 0:
            continue
        gamma = 1.0 / (2.0 * bw * bw)
        kxx = np.exp(-gamma * dxx)
        kyy = np.exp(-gamma * dyy)
        kxy = np.exp(-gamma * dxy)
        vals.append(np.nanmean(kxx) + np.nanmean(kyy) - 2.0 * np.mean(kxy))
    return float(np.mean(vals)) if vals else float("nan")


def frechet_distance(x: np.ndarray, y: np.ndarray) -> float:
    mu_x = x.mean(axis=0)
    mu_y = y.mean(axis=0)
    cov_x = np.cov(x, rowvar=False)
    cov_y = np.cov(y, rowvar=False)
    covmean = sqrtm(cov_x @ cov_y)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    out = np.sum((mu_x - mu_y) ** 2) + np.trace(cov_x + cov_y - 2.0 * covmean)
    return float(max(out, 0.0))


def distribution_metrics(gt: np.ndarray, pred: np.ndarray, threshold_q: float, bandwidths: list[float]):
    gt_tree = cKDTree(gt)
    pred_tree = cKDTree(pred)

    gt_loo = gt_tree.query(gt, k=2)[0][:, 1]
    threshold = float(np.quantile(gt_loo, threshold_q))

    gen_to_gt = gt_tree.query(pred, k=1)[0]
    gt_to_gen = pred_tree.query(gt, k=1)[0]
    paired = np.linalg.norm(pred - gt, axis=1)

    return {
        "n": int(len(gt)),
        "threshold_q": float(threshold_q),
        "gt_loo_threshold": threshold,
        "paired_z_l2_mean": float(np.mean(paired)),
        "paired_z_l2_median": float(np.median(paired)),
        "gen_to_gt_nn_mean": float(np.mean(gen_to_gt)),
        "gen_to_gt_nn_median": float(np.median(gen_to_gt)),
        "gt_to_gen_nn_mean": float(np.mean(gt_to_gen)),
        "gt_to_gen_nn_median": float(np.median(gt_to_gen)),
        "precision_at_gt_q": float(np.mean(gen_to_gt <= threshold)),
        "coverage_at_gt_q": float(np.mean(gt_to_gen <= threshold)),
        "mmd_rbf": mmd_rbf_unbiased(gt, pred, bandwidths),
        "frechet": frechet_distance(gt, pred),
    }


def per_feature_summary(gt_raw: np.ndarray, pred_raw: np.ndarray, features: list[str]):
    rows = []
    for idx, key in enumerate(features):
        gt = gt_raw[:, idx]
        pred = pred_raw[:, idx]
        rows.append(
            {
                "feature": key,
                "gt_mean": float(np.mean(gt)),
                "gt_std": float(np.std(gt)),
                "pred_mean": float(np.mean(pred)),
                "pred_std": float(np.std(pred)),
                "mean_delta": float(np.mean(pred) - np.mean(gt)),
                "std_ratio": float(np.std(pred) / np.std(gt)) if np.std(gt) > 1e-12 else float("nan"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows_by_run = {}
    for spec in args.metric_csvs:
        name = spec.split("=", 1)[0]
        rows_by_run[name] = load_metric_csv(spec)

    cases, gt_raw, pred_raw_by_run = build_matrices(rows_by_run, args.features)
    gt_z, pred_z_by_run, mean, std = standardize_by_gt(gt_raw, pred_raw_by_run)

    gt_pair = pdist(gt_z)
    median_pair = float(np.median(gt_pair[gt_pair > 0])) if np.any(gt_pair > 0) else 1.0
    bandwidths = [median_pair * scale for scale in args.mmd_scales]

    summary = []
    feature_rows = []
    for name, pred_z in pred_z_by_run.items():
        row = {"run": name}
        row.update(distribution_metrics(gt_z, pred_z, args.threshold_quantile, bandwidths))
        summary.append(row)
        for item in per_feature_summary(gt_raw, pred_raw_by_run[name], args.features):
            feature_rows.append({"run": name, **item})

    payload = {
        "features": args.features,
        "n_cases": len(cases),
        "cases": cases,
        "standardization": {"mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist()},
        "mmd_bandwidths": bandwidths,
        "summary": summary,
    }

    write_csv(args.out_dir / "distribution_summary.csv", summary)
    write_csv(args.out_dir / "feature_summary.csv", feature_rows)
    with (args.out_dir / "distribution_summary.json").open("w") as f:
        json.dump(payload, f, indent=2, allow_nan=True)

    print(f"Done. runs={len(summary)} cases={len(cases)}")
    print(args.out_dir)
    print(json.dumps(summary, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
