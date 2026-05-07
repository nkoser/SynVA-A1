#!/usr/bin/env python3
"""Validate whether GHD-fitting OPA checkpoints describe compact ostium rings."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from pytorch3d.io import load_obj
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ghd-root", type=Path, default=Path("checkpoint-v2/ghd_fitting_split_real"))
    parser.add_argument("--alignment-root", type=Path, default=Path("alignment_vc"))
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Cases to validate. Defaults to every case with a GHD OPA checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("vae_optimization_results/vae_opt_20260428_full/opa_validation"),
    )
    parser.add_argument(
        "--span-ratio-warning",
        type=float,
        default=2.5,
        help="Warn if GHD OPA span is this many times larger than alignment OPA span.",
    )
    return parser.parse_args()


def load_opa(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def ring_span(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def mesh_index_distance(mesh_path: Path, coords: np.ndarray, indices: np.ndarray) -> tuple[float, float, float, float]:
    verts, _, _ = load_obj(str(mesh_path))
    vertices = verts.detach().cpu().numpy().astype(np.float64)
    valid = indices[(indices >= 0) & (indices < vertices.shape[0])]
    if valid.shape[0] != indices.shape[0]:
        raise ValueError(f"OPA indices out of bounds for {mesh_path}: valid={valid.shape[0]} total={indices.shape[0]}")
    indexed = vertices[indices]
    direct = np.linalg.norm(indexed - coords, axis=1)
    nearest = cKDTree(vertices).query(coords, k=1)[0]
    return float(direct.mean()), float(direct.max()), float(nearest.mean()), float(nearest.max())


def find_warped_mesh(case_root: Path) -> Path | None:
    preferred = case_root / "vanilla" / "viz" / "warped_epoch_02999.obj"
    if preferred.exists():
        return preferred
    candidates = sorted((case_root / "vanilla" / "viz").glob("warped_epoch_*.obj"))
    return candidates[-1] if candidates else None


def validate_case(case: str, args: argparse.Namespace) -> dict[str, object]:
    ghd_opa_path = args.ghd_root / case / "opa_checkpoint.pkl"
    align_opa_path = args.alignment_root / case / "opa_checkpoint.pkl"
    warped_mesh_path = find_warped_mesh(args.ghd_root / case)
    if not ghd_opa_path.exists():
        raise FileNotFoundError(ghd_opa_path)

    ghd = load_opa(ghd_opa_path)
    ghd_coords = np.asarray(ghd["op_v_coords"][0], dtype=np.float64)
    ghd_indices = np.asarray(ghd["op_v_indices"][0], dtype=np.int64)
    if warped_mesh_path is None:
        direct_mean = direct_max = nearest_mean = nearest_max = np.nan
    else:
        direct_mean, direct_max, nearest_mean, nearest_max = mesh_index_distance(
            warped_mesh_path,
            ghd_coords,
            ghd_indices,
        )

    align_span = np.nan
    align_points = 0
    span_ratio = np.nan
    if align_opa_path.exists():
        align = load_opa(align_opa_path)
        align_coords = np.asarray(align["op_v_coords"][0], dtype=np.float64)
        align_span = ring_span(align_coords)
        align_points = int(align_coords.shape[0])

    ghd_span = ring_span(ghd_coords)
    if np.isfinite(align_span) and align_span > 1e-12:
        span_ratio = ghd_span / align_span

    warning = bool(np.isfinite(span_ratio) and span_ratio >= float(args.span_ratio_warning))
    return {
        "case": case,
        "source": str(ghd.get("source", "unknown")),
        "ghd_points": int(ghd_coords.shape[0]),
        "alignment_points": align_points,
        "ghd_span": ghd_span,
        "alignment_span": float(align_span) if np.isfinite(align_span) else "",
        "span_ratio_ghd_to_alignment": float(span_ratio) if np.isfinite(span_ratio) else "",
        "mesh_index_direct_mean": direct_mean,
        "mesh_index_direct_max": direct_max,
        "coords_to_mesh_nearest_mean": nearest_mean,
        "coords_to_mesh_nearest_max": nearest_max,
        "warning_large_span_ratio": warning,
        "ghd_opa": str(ghd_opa_path),
        "alignment_opa": str(align_opa_path) if align_opa_path.exists() else "",
        "warped_mesh": str(warped_mesh_path) if warped_mesh_path is not None else "",
    }


def main() -> None:
    args = parse_args()
    if args.cases:
        cases = args.cases
    else:
        cases = sorted(path.parent.name for path in args.ghd_root.glob("*/opa_checkpoint.pkl"))

    rows = []
    failed = []
    for case in cases:
        try:
            rows.append(validate_case(case, args))
        except Exception as exc:
            failed.append({"case": case, "error": str(exc)})
    if not rows:
        raise RuntimeError(f"No cases validated. Failures: {failed[:5]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "opa_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    source_counts: dict[str, int] = {}
    warnings = 0
    for row in rows:
        source_counts[str(row["source"])] = source_counts.get(str(row["source"]), 0) + 1
        warnings += int(bool(row["warning_large_span_ratio"]))

    md_path = args.output_dir / "summary.md"
    top = sorted(
        rows,
        key=lambda row: float(row["span_ratio_ghd_to_alignment"] or 0.0),
        reverse=True,
    )[:20]
    lines = [
        "# GHD OPA Validation",
        "",
        f"Cases checked: {len(rows)}",
        f"Large span-ratio warnings: {warnings}",
        f"Failed cases: {len(failed)}",
        f"Source counts: {source_counts}",
        "",
        "Interpretation: direct/nearest distances near zero mean the OPA checkpoint is internally consistent with the GHD mesh. "
        "A large GHD/alignment span ratio means the stored OPA ring is much broader than the compact alignment ostium.",
        "",
        "## Largest GHD/Alignment Span Ratios",
        "",
        "| case | source | ghd span | alignment span | ratio | direct max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in top:
        lines.append(
            f"| {row['case']} | {row['source']} | {float(row['ghd_span']):.6f} | "
            f"{float(row['alignment_span'] or 0.0):.6f} | "
            f"{float(row['span_ratio_ghd_to_alignment'] or 0.0):.3f} | "
            f"{float(row['mesh_index_direct_max']):.6g} |"
        )
    lines.append("")
    lines.append(f"Full CSV: `{csv_path}`")
    if failed:
        failed_path = args.output_dir / "failures.csv"
        with failed_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["case", "error"])
            writer.writeheader()
            writer.writerows(failed)
        lines.append(f"Failures CSV: `{failed_path}`")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
