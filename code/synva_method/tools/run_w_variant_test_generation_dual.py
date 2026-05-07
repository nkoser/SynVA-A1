#!/usr/bin/env python
"""Generate full test-set meshes for the good W variants in two pipelines.

Outputs are written below --out_root without overwriting an existing run unless
--overwrite is passed.  The two pipelines are:

  reference_stitching:
    vessel-mesh-editing-master's run_inference_pipeline.py all path, with
    our W checkpoint plugged in only through its external-method hook.

  normal_stitching:
    AneuG's direct method sampler/decode wrapper + the standard step3 call.

This script is intentionally only an orchestrator; the actual generation and
stitching stays in the existing wrappers.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


W_VARIANTS: list[dict[str, str]] = [
    {
        "name": "W_ref",
        "method": "baseline",
        "checkpoint": "",
        "note": "Their original Stage-1 CVAE checkpoint, no external AneuG model.",
    },
    {
        "name": "W_stage3surrogate",
        "method": "W",
        "checkpoint": "checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3surrogate/W_vessel_stage3surrogate_seed1_20260502_222924/models_best_val.pth",
        "note": "Our W model with stage3 surrogate loss.",
    },
    {
        "name": "W_stage3nearest",
        "method": "W",
        "checkpoint": "checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3nearest/W_vessel_stage3nearest_seed1_20260502_230124/models_best_val.pth",
        "note": "Our W model with nearest-stage3 objective; best recent strict run.",
    },
    {
        "name": "W_stage3surrogate_morph",
        "method": "W",
        "checkpoint": "checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3surrogate_morph/W_vessel_stage3surrogate_morph_seed1_20260503_154326/models_best_val.pth",
        "note": "Our W stage3-surrogate model with additional target morphology condition.",
    },
    {
        "name": "W_stage3surrogate_morph_ostium",
        "method": "W",
        "checkpoint": "checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3surrogate_morph_ostium/W_vessel_stage3surrogate_morph_ostium_seed1_20260503_181654/models_best_val.pth",
        "note": "Our W stage3-surrogate model with the selected ostium/pouch morphology condition.",
    },
    {
        "name": "W_stage3surrogate_morph_ostium_shape",
        "method": "W",
        "checkpoint": "checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3surrogate_morph_ostium_shape/W_vessel_stage3surrogate_morph_ostium_shape_seed1_20260503_190305/models_best_val.pth",
        "note": "Our W stage3-surrogate model with selected morphology condition and explicit sac-shape losses.",
    },
    {
        "name": "W_stage3surrogate_morph_ostium_surface",
        "method": "W",
        "checkpoint": "checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3surrogate_morph_ostium_surface/W_vessel_stage3surrogate_morph_ostium_surface_seed1_20260503_194606/models_best_val.pth",
        "note": "Our W stage3-surrogate model with selected morphology condition and direct sampled pouch-surface Chamfer loss.",
    },
    {
        "name": "W_stage3surrogate_priorcalib",
        "method": "W",
        "checkpoint": "checkpoints/methods_aneug_ghds_refstage1/W_vessel_stage3surrogate_priorcalib/W_vessel_stage3surrogate_priorcalib_seed1_20260504_010917/models_best_val.pth",
        "note": "No-morphology ablation with the same stage3-surrogate and prior-path calibration as the final morph-priorcalib model.",
    },
]


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default=f"/path/to/SynVA-A1_outputs/w_variant_test_generation_{stamp}")
    p.add_argument("--cases_file", default="checkpoints/aneug_ghds/splits/aneug_ghds_realcsv_opa20_seed42_20260502_013432/cases_test.json")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max_cases", type=int, default=0, help="0 = all test cases.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--continue_on_error", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip_reference", action="store_true")
    p.add_argument("--skip_normal", action="store_true")
    return p.parse_args()


def run(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[run] " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    print(f"[done rc={proc.returncode}] log={log_path}", flush=True)
    return int(proc.returncode)


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    if out_root.exists() and not args.overwrite:
        raise FileExistsError(f"{out_root} exists; choose a new --out_root or pass --overwrite.")
    out_root.mkdir(parents=True, exist_ok=True)

    cases_file = str((ROOT / args.cases_file).resolve() if not os.path.isabs(args.cases_file) else Path(args.cases_file))
    cases = json.loads(Path(cases_file).read_text(encoding="utf-8"))
    manifest = {
        "out_root": str(out_root),
        "cases_file": cases_file,
        "num_test_cases": len(cases) if int(args.max_cases) <= 0 else min(len(cases), int(args.max_cases)),
        "seed": int(args.seed),
        "temperature": float(args.temperature),
        "variants": W_VARIANTS,
        "pipelines": {
            "reference_stitching": "tools/run_strict_reference_stage3_batch.py (reference all pipeline)",
            "normal_stitching": "tools/generate_and_stitch_reference.py",
        },
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLCONFIGDIR"] = "/tmp/matplotlib"
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        env["LD_LIBRARY_PATH"] = str(Path(conda_prefix) / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")

    results: list[dict[str, object]] = []
    failed = False
    for variant in W_VARIANTS:
        name = variant["name"]
        method = variant["method"]
        checkpoint = variant["checkpoint"]

        if not args.skip_reference:
            ref_out = out_root / "reference_stitching" / name
            cmd = [
                sys.executable,
                str(ROOT / "tools" / "run_strict_reference_stage3_batch.py"),
                "--cases_file", cases_file,
                "--out_root", str(ref_out),
                "--seed", str(int(args.seed)),
                "--num_samples", "1",
                "--continue_on_error",
                "--overwrite",
            ]
            if int(args.max_cases) > 0:
                cmd += ["--max_cases", str(int(args.max_cases))]
            if method != "baseline":
                cmd += [
                    "--external_method_type", method,
                    "--external_method_checkpoint", str((ROOT / checkpoint).resolve()),
                ]
            rc = run(cmd, out_root / "logs" / f"reference__{name}.log", env)
            rec = {"pipeline": "reference_stitching", "variant": name, "returncode": rc, "out": str(ref_out)}
            results.append(rec)
            if rc != 0:
                failed = True
                if not args.continue_on_error:
                    break

        if not args.skip_normal and not (failed and not args.continue_on_error):
            normal_out = out_root / "normal_stitching" / name
            if method == "baseline":
                # The direct AneuG sampler wrapper is only for external A/B/C/D/E/W
                # models. The baseline/reference W is already covered above.
                rec = {
                    "pipeline": "normal_stitching",
                    "variant": name,
                    "returncode": None,
                    "out": str(normal_out),
                    "skipped": "baseline uses the reference Stage-1 checkpoint; no direct AneuG normal sampler.",
                }
                print(f"[skip] normal_stitching {name}: {rec['skipped']}", flush=True)
                results.append(rec)
            else:
                cmd = [
                    sys.executable,
                    str(ROOT / "tools" / "generate_and_stitch_reference.py"),
                    "--cases_file", cases_file,
                    "--ckpt", str((ROOT / checkpoint).resolve()),
                    "--method", method,
                    "--out_root", str(normal_out),
                    "--device", str(args.device),
                    "--seed", str(int(args.seed)),
                    "--temperature", str(float(args.temperature)),
                    "--num_candidates", "1",
                    "--select_by", "first",
                    "--decode_backend", "reference",
                    "--continue_on_error",
                    "--overwrite",
                ]
                if int(args.max_cases) > 0:
                    cmd += ["--max_cases", str(int(args.max_cases))]
                rc = run(cmd, out_root / "logs" / f"normal__{name}.log", env)
                rec = {"pipeline": "normal_stitching", "variant": name, "returncode": rc, "out": str(normal_out)}
                results.append(rec)
                if rc != 0:
                    failed = True
                    if not args.continue_on_error:
                        break

        (out_root / "run_summary.json").write_text(
            json.dumps({"failed": failed, "results": results}, indent=2),
            encoding="utf-8",
        )

    (out_root / "run_summary.json").write_text(
        json.dumps({"failed": failed, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"failed": failed, "out_root": str(out_root), "summary": str(out_root / "run_summary.json")}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
