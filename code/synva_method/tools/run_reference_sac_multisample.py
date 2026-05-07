#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
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
    p.add_argument("--out_root", default=f"/path/to/aneug_reference_sac_multisample_{stamp}")
    p.add_argument(
        "--cases_file",
        default="checkpoints/aneug_ghds/splits/aneug_ghds_realcsv_opa20_seed42_20260502_013432/cases_test.json",
    )
    p.add_argument("--variants", nargs="+", default=["W_ref", "W_stage3surrogate"])
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max_cases", type=int, default=0)
    p.add_argument("--prepared_root", default="/path/to/prepared_meshes_3")
    p.add_argument("--alignment_root", default="/path/to/aneug-ghds/data/alignment")
    p.add_argument("--ghd_root", default="/path/to/aneug-ghds/data/ghd_fitting")
    p.add_argument("--canonical_root", default="/path/to/aneug-ghds/data/alignment/canonical_model")
    p.add_argument("--aneug_ref_root", default="/path/to/reference-vessel-mesh-editing/code/AneuG-Own-edit")
    p.add_argument("--reference_pipeline", default="/path/to/reference-vessel-mesh-editing/code/inference/run_inference_pipeline.py")
    p.add_argument(
        "--stage1_checkpoint",
        default="/path/to/reference-vessel-mesh-editing/code/AneuG-Own-edit/checkpoints-new/first_stage_ostium_conditional/models_epoch_2000.pth",
    )
    p.add_argument("--continue_on_error", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve_prepared_case(root: Path, case: str) -> Path:
    candidates = [
        root / case,
        root / f"aneux_{case}",
        root / case.replace("cmha_", "cmch_"),
        root / f"aneux_{case.replace('cmha_', 'cmch_')}",
    ]
    for candidate in candidates:
        if (candidate / "05_submeshes" / "vessel_submesh.obj").exists():
            return candidate
    raise FileNotFoundError(f"No prepared case for {case}")


def copy_case_layout(args: argparse.Namespace, case: str, case_root: Path) -> None:
    if args.overwrite and case_root.exists():
        shutil.rmtree(case_root)
    prepared = resolve_prepared_case(Path(args.prepared_root), case)
    paths = {
        prepared / "04_subpointclouds" / "subpointcloud_label_2.ply": case_root / "04_subpointclouds" / "subpointcloud_label_2.ply",
        prepared / "05_submeshes" / "vessel_submesh.obj": case_root / "05_submeshes" / "vessel_submesh.obj",
        prepared / "07_other" / "centroid_ostium.npy": case_root / "07_other" / "centroid_ostium.npy",
        prepared / "07_other" / "normal_vector.npy": case_root / "07_other" / "normal_vector.npy",
    }
    for src, dst in paths.items():
        if not src.exists():
            raise FileNotFoundError(str(src))
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or args.overwrite:
            shutil.copy2(src, dst)


def run(cmd: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[run]", " ".join(cmd), flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("[run] " + " ".join(cmd) + "\n")
        handle.flush()
        subprocess.run(cmd, check=True, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)


def main() -> int:
    args = parse_args()
    cases_file = Path(args.cases_file)
    if not cases_file.is_absolute():
        cases_file = ROOT / cases_file
    cases = json.loads(cases_file.read_text(encoding="utf-8"))
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    for name in args.variants:
        if name not in VARIANTS:
            raise ValueError(f"Unknown variant {name}; choose from {sorted(VARIANTS)}")

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "cases_eval.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    (out_root / "manifest.json").write_text(
        json.dumps(
            {
                "out_root": str(out_root),
                "cases_file": str(cases_file),
                "num_cases": len(cases),
                "variants": args.variants,
                "num_samples": int(args.num_samples),
                "seed": int(args.seed),
                "note": "Sac-only multi-sample export: reference step1 + step2 only, no step3 stitching.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(args.aneug_ref_root)) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLCONFIGDIR"] = "/tmp/matplotlib"
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        env["LD_LIBRARY_PATH"] = str(Path(conda_prefix) / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")

    ok: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []
    for variant_name in args.variants:
        variant = VARIANTS[variant_name]
        variant_root = out_root / "sac_samples" / variant_name
        for idx, case in enumerate(cases, start=1):
            print(f"\n[{variant_name} {idx}/{len(cases)}] {case}", flush=True)
            log_path = out_root / "logs" / variant_name / f"{case.replace('/', '__')}.log"
            try:
                case_root = variant_root / "cases" / "test" / case
                copy_case_layout(args, case, case_root)
                common = [
                    sys.executable,
                    str(Path(args.reference_pipeline)),
                    "--case-name", case,
                    "--case-split", "test",
                    "--cases-root", str(variant_root / "cases"),
                    "--aneug-root", str(Path(args.aneug_ref_root)),
                    "--stage1-checkpoint", str(Path(args.stage1_checkpoint)),
                    "--stage1-ghd-root", str(Path(args.ghd_root)),
                    "--stage1-alignment-root", str(Path(args.alignment_root)),
                    "--stage1-canonical-root", str(Path(args.canonical_root)),
                    "--ring-points", "20",
                    "--overwrite",
                ]
                run(common[:2] + ["step1"] + common[2:], env, log_path)
                step2 = common[:2] + [
                    "step2",
                    "--skip-reconstruct",
                    "--num-samples", str(int(args.num_samples)),
                    "--seed", str(int(args.seed)),
                ] + common[2:]
                if variant["method"] is not None:
                    step2 += [
                        "--external-method-type", str(variant["method"]),
                        "--external-method-checkpoint", str((ROOT / str(variant["checkpoint"])).resolve()),
                        "--external-aneug-root", str(ROOT),
                    ]
                run(step2, env, log_path)
                ok.append({"variant": variant_name, "case": case})
            except Exception as exc:
                print(f"[failed] {variant_name} {case}: {exc}", flush=True)
                failed.append({"variant": variant_name, "case": case, "error": str(exc)})
                if not args.continue_on_error:
                    break
            (out_root / "run_summary.json").write_text(
                json.dumps({"ok_count": len(ok), "failed_count": len(failed), "ok": ok, "failed": failed}, indent=2),
                encoding="utf-8",
            )
        if failed and not args.continue_on_error:
            break

    print(json.dumps({"ok_count": len(ok), "failed_count": len(failed), "summary": str(out_root / "run_summary.json")}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
