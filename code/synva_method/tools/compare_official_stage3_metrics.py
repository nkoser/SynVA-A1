#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = [
    "ring_to_target_mean_distance",
    "opening_center_to_ostium_distance",
    "ring_to_label2_mean_distance",
    "label2_to_pouch_mean_distance",
    "nearest_vertex_to_ostium_distance",
    "pouch_center_to_ostium_distance",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="name=output_root, e.g. W=outputs/official_pipeline_theirW_test")
    p.add_argument("--metrics", nargs="*", default=METRICS)
    return p.parse_args()


def load_run(root: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for path in sorted(root.glob("cases/*/*/outputs/step3_compose_summary.json")):
        case = path.parents[1].name
        try:
            summary = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        out[case] = summary
    return out


def fmt(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{x:.6f}"


def main() -> int:
    args = parse_args()
    runs: dict[str, dict[str, dict[str, float]]] = {}
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"Expected name=path, got: {spec}")
        name, path = spec.split("=", 1)
        runs[name] = load_run(Path(path))
        print(f"[load] {name}: {len(runs[name])} cases from {path}")

    names = list(runs)
    common = sorted(set.intersection(*(set(runs[n]) for n in names)))
    print(f"\npaired_cases {len(common)}")
    if not common:
        return 1

    for metric in args.metrics:
        print(f"\n{metric}")
        values = {}
        for name in names:
            arr = np.asarray([runs[name][case].get(metric, np.nan) for case in common], dtype=float)
            arr = arr[np.isfinite(arr)]
            values[name] = arr
            print(f"  {name:>14s} mean={fmt(float(arr.mean()))} median={fmt(float(np.median(arr)))} n={len(arr)}")
        base = names[0]
        for name in names[1:]:
            pairs = [
                (runs[name][case].get(metric, np.nan), runs[base][case].get(metric, np.nan))
                for case in common
            ]
            deltas = np.asarray([a - b for a, b in pairs if np.isfinite(a) and np.isfinite(b)], dtype=float)
            wins = int((deltas < 0).sum())
            print(
                f"  {name:>14s} vs {base:<14s} wins={wins}/{len(deltas)} "
                f"mean_delta={fmt(float(deltas.mean()))} median_delta={fmt(float(np.median(deltas)))}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
