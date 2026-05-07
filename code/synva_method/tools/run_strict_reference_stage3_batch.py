#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cases_file", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--prepared_root", default="/path/to/prepared_meshes_3")
    p.add_argument("--alignment_root", default="/path/to/aneug-ghds/data/alignment")
    p.add_argument("--ghd_root", default="/path/to/aneug-ghds/data/ghd_fitting")
    p.add_argument("--canonical_root", default="/path/to/aneug-ghds/data/alignment/canonical_model")
    p.add_argument("--aneug_ref_root", default="/path/to/SynVA-A1/code/AneuG-Own-edit")
    p.add_argument("--reference_pipeline", default="/path/to/SynVA-A1/code/inference/run_inference_pipeline.py")
    p.add_argument("--stage1_checkpoint", default="/path/to/SynVA-A1/code/AneuG-Own-edit/checkpoints-new/first_stage_ostium_conditional/models_epoch_2000.pth")
    p.add_argument("--external_method_type", choices=["A", "B", "C", "D", "E", "W", "baseline"], default=None)
    p.add_argument("--external_method_checkpoint", default=None)
    p.add_argument("--external_aneug_root", default="/path/to/SynVA-A1/code/synva_method")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--max_cases", type=int, default=0)
    p.add_argument("--smooth_ostium_iterations", type=int, default=10)
    p.add_argument("--smooth_ostium_hops", type=int, default=2)
    p.add_argument("--stitch_bridge_steps", type=int, default=1)
    p.add_argument("--stitch_smooth_intersection", action="store_true")
    p.add_argument("--skip_reconstruct", action="store_true")
    p.add_argument("--continue_on_error", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--workers", type=int, default=1,
                   help="Run multiple cases concurrently. Each case writes its own log under out_root/logs.")
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


def copy_case_layout(args: argparse.Namespace, case: str) -> Path:
    out_root = Path(args.out_root)
    case_root = out_root / "cases" / "test" / case
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
    return case_root


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def build_case_cmd(args: argparse.Namespace, out_root: Path, pipeline: Path, case: str) -> list[str]:
    cmd = [
        sys.executable,
        str(pipeline),
        "all",
        "--case-name", case,
        "--case-split", "test",
        "--cases-root", str(out_root / "cases"),
        "--aneug-root", str(Path(args.aneug_ref_root)),
        "--stage1-checkpoint", str(Path(args.stage1_checkpoint)),
        "--stage1-ghd-root", str(Path(args.ghd_root)),
        "--stage1-alignment-root", str(Path(args.alignment_root)),
        "--stage1-canonical-root", str(Path(args.canonical_root)),
        "--num-samples", str(int(args.num_samples)),
        "--seed", str(int(args.seed)),
        "--ring-points", "20",
        "--resample-aneurysm-to-vessel-resolution",
        "--stitch",
        "--stitch-method", "bridge",
        "--stitch-bridge-steps", str(int(args.stitch_bridge_steps)),
        "--smooth-ostium-transition",
        "--smooth-ostium-iterations", str(int(args.smooth_ostium_iterations)),
        "--smooth-ostium-hops", str(int(args.smooth_ostium_hops)),
        "--overwrite",
    ]
    if args.skip_reconstruct:
        cmd.append("--skip-reconstruct")
    if args.external_method_type:
        if not args.external_method_checkpoint:
            raise ValueError("--external_method_checkpoint is required")
        if "--skip-reconstruct" not in cmd:
            cmd.append("--skip-reconstruct")
        cmd += [
            "--external-method-type", args.external_method_type,
            "--external-method-checkpoint", str(Path(args.external_method_checkpoint)),
            "--external-aneug-root", str(Path(args.external_aneug_root)),
        ]
    if args.stitch_smooth_intersection:
        cmd.append("--stitch-smooth-intersection")
    return cmd


def process_case(args: argparse.Namespace, case: str, idx: int, total: int,
                 out_root: Path, pipeline: Path, env: dict[str, str]) -> dict[str, str | int]:
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{idx:03d}_{case}.log"
    try:
        copy_case_layout(args, case)
        cmd = build_case_cmd(args, out_root, pipeline, case)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"[{idx}/{total}] {case}\n")
            log.write("[run] " + " ".join(cmd) + "\n")
            log.flush()
            subprocess.run(cmd, check=True, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        return {"case": case, "idx": idx, "status": "ok", "log": str(log_path)}
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[failed] {case}: {exc}\n")
        return {"case": case, "idx": idx, "status": "failed", "error": str(exc), "log": str(log_path)}


def main() -> int:
    args = parse_args()
    cases = json.loads(Path(args.cases_file).read_text())
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(args.aneug_ref_root)) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLCONFIGDIR"] = "/tmp/matplotlib"
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        env["LD_LIBRARY_PATH"] = str(Path(conda_prefix) / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")

    pipeline = Path(args.reference_pipeline)
    ok: list[str] = []
    failed: list[dict[str, str]] = []

    if int(args.workers) > 1:
        workers = max(1, int(args.workers))
        print(f"[parallel] workers={workers} cases={len(cases)} out={out_root}", flush=True)
        results: list[dict[str, str | int]] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(process_case, args, case, idx, len(cases), out_root, pipeline, env.copy())
                for idx, case in enumerate(cases, start=1)
            ]
            for fut in as_completed(futures):
                result = fut.result()
                results.append(result)
                case = str(result["case"])
                if result["status"] == "ok":
                    ok.append(case)
                    print(f"[ok {len(ok)}/{len(cases)}] {case} log={result['log']}", flush=True)
                else:
                    failed.append({"case": case, "error": str(result.get("error", "")), "log": str(result["log"])})
                    print(f"[failed {len(failed)}] {case}: {result.get('error', '')} log={result['log']}", flush=True)
                    if not args.continue_on_error:
                        for pending in futures:
                            pending.cancel()
                        break
                (out_root / "summary.json").write_text(
                    json.dumps(
                        {
                            "ok_count": len(ok),
                            "failed_count": len(failed),
                            "ok": sorted(ok),
                            "failed": failed,
                            "workers": workers,
                            "results": sorted(results, key=lambda r: int(r["idx"])),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        print(json.dumps({"ok_count": len(ok), "failed_count": len(failed), "summary": str(out_root / "summary.json")}, indent=2))
        return 0 if not failed else 1

    for idx, case in enumerate(cases, start=1):
        print(f"\n[{idx}/{len(cases)}] {case}", flush=True)
        try:
            copy_case_layout(args, case)
            # Deliberately run the reference pipeline end-to-end.  The only
            # permitted model substitution is through its own external-method
            # hook, so Step 1 OPA creation and Step 3 stitching stay identical
            # to vessel-mesh-editing-master.
            cmd = build_case_cmd(args, out_root, pipeline, case)
            run(cmd, env)
            ok.append(case)
        except Exception as exc:
            print(f"[failed] {case}: {exc}", flush=True)
            failed.append({"case": case, "error": str(exc)})
            if not args.continue_on_error:
                break
        (out_root / "summary.json").write_text(
            json.dumps({"ok_count": len(ok), "failed_count": len(failed), "ok": ok, "failed": failed}, indent=2),
            encoding="utf-8",
        )
    print(json.dumps({"ok_count": len(ok), "failed_count": len(failed), "summary": str(out_root / "summary.json")}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
