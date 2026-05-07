#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


VARIANTS: dict[str, dict[str, str | None]] = {
    "W_ref": {
        "method": None,
        "checkpoint": None,
    },
    "W_stage3surrogate": {
        "method": "W",
        "checkpoint": "checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3surrogate/W_vessel_stage3surrogate_seed1_20260502_222924/models_best_val.pth",
    },
    "W_stage3nearest": {
        "method": "W",
        "checkpoint": "checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3nearest/W_vessel_stage3nearest_seed1_20260502_230124/models_best_val.pth",
    },
    "W_stage3surrogate_morph": {
        "method": "W",
        "checkpoint": "checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3surrogate_morph/W_vessel_stage3surrogate_morph_seed1_20260503_154326/models_best_val.pth",
    },
    "W_stage3surrogate_morph_ostium": {
        "method": "W",
        "checkpoint": "checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3surrogate_morph_ostium/W_vessel_stage3surrogate_morph_ostium_seed1_20260503_181654/models_best_val.pth",
    },
    "W_stage3surrogate_morph_ostium_shape": {
        "method": "W",
        "checkpoint": "checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3surrogate_morph_ostium_shape/W_vessel_stage3surrogate_morph_ostium_shape_seed1_20260503_190305/models_best_val.pth",
    },
    "W_stage3surrogate_morph_ostium_surface": {
        "method": "W",
        "checkpoint": "checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3surrogate_morph_ostium_surface/W_vessel_stage3surrogate_morph_ostium_surface_seed1_20260503_194606/models_best_val.pth",
    },
    "W_stage3surrogate_morph_priorcalib": {
        "method": "W",
        "checkpoint": "checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3surrogate_morph_priorcalib/W_vessel_stage3surrogate_morph_priorcalib_seed1_20260503_203915/models_best_val.pth",
    },
    "W_stage3surrogate_priorcalib": {
        "method": "W",
        "checkpoint": "checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3surrogate_priorcalib/W_vessel_stage3surrogate_priorcalib_seed1_20260504_010917/models_best_val.pth",
    },
}


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default=f"/path/to/aneug_reference_multisample_{stamp}")
    p.add_argument(
        "--cases_file",
        default="checkpoints/aneug_ghds/splits/aneug_ghds_realcsv_opa20_seed42_20260502_013432/cases_test.json",
    )
    p.add_argument("--variants", nargs="+", default=["W_ref", "W_stage3surrogate"])
    p.add_argument("--num_samples_per_case", type=int, default=8)
    p.add_argument("--seed_start", type=int, default=1)
    p.add_argument("--max_cases", type=int, default=0)
    p.add_argument("--continue_on_error", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("[run] " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT, text=True)
    print(f"[done rc={proc.returncode}] {log_path}", flush=True)
    return int(proc.returncode)


def main() -> int:
    args = parse_args()
    cases_file = Path(args.cases_file)
    if not cases_file.is_absolute():
        cases_file = ROOT / cases_file
    cases = json.loads(cases_file.read_text(encoding="utf-8"))
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    case_list = out_root / "cases_eval.json"
    case_list.write_text(json.dumps(cases, indent=2), encoding="utf-8")

    for name in args.variants:
        if name not in VARIANTS:
            raise ValueError(f"Unknown variant {name}; choose from {sorted(VARIANTS)}")

    manifest = {
        "out_root": str(out_root),
        "cases_file": str(cases_file),
        "num_cases": len(cases),
        "variants": args.variants,
        "num_samples_per_case": int(args.num_samples_per_case),
        "seed_start": int(args.seed_start),
        "note": "Each sample index is generated as an independent reference-all run with seed=seed_start+sample_index, because reference Step 3 stitches sample_000 only.",
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    results: list[dict[str, object]] = []
    failed = False
    for variant_name in args.variants:
        variant = VARIANTS[variant_name]
        for sample_idx in range(int(args.num_samples_per_case)):
            seed = int(args.seed_start) + sample_idx
            sample_name = f"sample_{sample_idx:02d}_seed{seed}"
            sample_out = out_root / "reference_stitching" / variant_name / sample_name
            cmd = [
                sys.executable,
                str(ROOT / "tools" / "run_strict_reference_stage3_batch.py"),
                "--cases_file", str(case_list),
                "--out_root", str(sample_out),
                "--seed", str(seed),
                "--num_samples", "1",
                "--skip_reconstruct",
                "--continue_on_error",
                "--overwrite",
            ]
            if int(args.max_cases) > 0:
                cmd += ["--max_cases", str(int(args.max_cases))]
            if variant["method"] is not None:
                cmd += [
                    "--external_method_type", str(variant["method"]),
                    "--external_method_checkpoint", str((ROOT / str(variant["checkpoint"])).resolve()),
                ]
            rc = run(cmd, out_root / "logs" / f"{variant_name}__{sample_name}.log")
            record = {
                "variant": variant_name,
                "sample_idx": sample_idx,
                "seed": seed,
                "returncode": rc,
                "out": str(sample_out),
            }
            results.append(record)
            (out_root / "run_summary.json").write_text(
                json.dumps({"failed": failed or rc != 0, "results": results}, indent=2),
                encoding="utf-8",
            )
            if rc != 0:
                failed = True
                if not args.continue_on_error:
                    print(json.dumps({"failed": True, "out_root": str(out_root)}, indent=2))
                    return 1

    print(json.dumps({"failed": failed, "out_root": str(out_root), "summary": str(out_root / "run_summary.json")}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
