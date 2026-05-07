#!/usr/bin/env python
"""Batch-attach generated aneurysm meshes onto their healthy vessel for the test set.

Assumes per-case OBJs are already written by `methods/visualize_val_samples.py`
into `<samples_dir>/<case>/{gt,A,B,C,D,E,baseline}.obj` in ghd_local space.

For every case x method, runs `tools/attach_aneurysm_to_healthy.py` and stores
the result at `<samples_dir>/<case>/<method>_attached.obj`. Skips if the source
sample OBJ is missing or the attached file already exists (use --overwrite).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
ATTACH = os.path.join(ROOT, "tools", "attach_aneurysm_to_healthy.py")
TAGS_DEFAULT = ["gt", "A", "B", "C", "D", "E", "baseline"]
ATTACH_REPORT_KEYS = [
    "attach_mode",
    "fuse_requested",
    "fuse_used",
    "fuse_smoother",
    "fuse_band_rings",
    "fuse_sigma_rings",
    "fuse_smooth_iters",
    "fuse_smooth_lam",
    "fuse_smooth_nu",
    "open_bridge_requested",
    "open_bridge_used",
    "open_bridge_info",
    "rim_presmooth_requested",
    "rim_presmooth_info",
    "input_boundary_edges",
    "input_aneurysm_boundary_edges",
    "selected_ostium_loop_edges",
    "selected_aneurysm_rim_edges",
    "expected_remaining_boundary_edges",
    "combined_boundary_edges",
    "seam_boundary_delta",
    "seam_closed",
    "combined_watertight",
    "bridge_faces",
    "remesh_info",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cases_file", required=True)
    p.add_argument("--samples_dir", required=True,
                   help="output dir of methods/visualize_val_samples.py")
    p.add_argument("--tags", nargs="+", default=TAGS_DEFAULT)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="0 = all cases")
    p.add_argument("--jagged_amp", type=float, default=0.16)
    p.add_argument("--cut_slab", type=float, default=0.06)
    p.add_argument("--radius_scale", type=float, default=1.10)
    p.add_argument("--log_dir", default=None,
                   help="if given, write per-attach stderr/stdout there")
    p.add_argument("--report_json", default=None)
    p.add_argument("--use_vessel_submesh", action="store_true",
                   help="use prepared vessel_submesh.obj (real ostium boundary) "
                        "instead of cutting a procedural hole into the closed mesh.")
    p.add_argument("--no_remesh_loops", action="store_true",
                   help="disable boundary loop densification (default ON in attach).")
    p.add_argument("--fuse_rims", action="store_true",
                   help="topologically merge aneurysm rim into hole (no bridge band).")
    p.add_argument("--fuse_band_rings", type=int, default=6)
    p.add_argument("--fuse_sigma_rings", type=float, default=2.5)
    p.add_argument("--fuse_smoother", choices=["taubin", "laplacian"], default="taubin")
    p.add_argument("--fuse_smooth_iters", type=int, default=10)
    p.add_argument("--fuse_smooth_lam", type=float, default=0.5)
    p.add_argument("--fuse_smooth_nu", type=float, default=0.53)
    p.add_argument("--fuse_bridge_steps", type=int, default=0)
    p.add_argument("--fuse_bridge_alpha", type=float, default=0.6)
    p.add_argument("--open_bridge", action="store_true",
                   help="use open-boundary bridge stitch (Weg B): N intermediate "
                        "rings between vessel ostium and aneurysm rim, no snap.")
    p.add_argument("--open_bridge_steps", type=int, default=4,
                   help="number of intermediate rings inserted in --open_bridge mode.")
    p.add_argument("--rim_presmooth", action="store_true",
                   help="local Taubin smoothing of the first N rings inside "
                        "the aneurysm rim before stitching (rim itself fixed).")
    p.add_argument("--rim_presmooth_rings", type=int, default=4)
    p.add_argument("--rim_presmooth_iters", type=int, default=8)
    p.add_argument("--rim_presmooth_lam", type=float, default=0.5)
    p.add_argument("--rim_presmooth_nu", type=float, default=0.53)
    p.add_argument("--bridge_smooth_iters", type=int, default=4,
                   help="Taubin iters on intermediate bridge rings only. 0 disables.")
    p.add_argument("--bridge_smooth_lam", type=float, default=0.5)
    p.add_argument("--bridge_smooth_nu", type=float, default=0.53)
    return p.parse_args()


def _read_attach_report_metrics(log_path):
    if not log_path or not os.path.isfile(log_path):
        return {}
    try:
        text = open(log_path, "r", encoding="utf-8", errors="replace").read()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        report = json.loads(text[start:end + 1])
    except Exception:
        return {}
    return {k: report[k] for k in ATTACH_REPORT_KEYS if k in report}


def main():
    args = parse_args()
    cases = json.load(open(args.cases_file))
    if args.limit > 0:
        cases = cases[: args.limit]
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    summary = {"ok": [], "skip": [], "fail": []}
    t0 = time.time()
    for ci, case in enumerate(cases):
        case_dir = os.path.join(args.samples_dir, case)
        if not os.path.isdir(case_dir):
            print(f"[{ci+1}/{len(cases)}] {case}: no samples dir, skip", flush=True)
            summary["skip"].append({"case": case, "reason": "no_samples_dir"})
            continue
        for tag in args.tags:
            src = os.path.join(case_dir, f"{tag}.obj")
            dst = os.path.join(case_dir, f"{tag}_attached.obj")
            if not os.path.isfile(src):
                summary["skip"].append({"case": case, "tag": tag, "reason": "no_src"})
                continue
            if os.path.isfile(dst) and not args.overwrite:
                summary["skip"].append({"case": case, "tag": tag, "reason": "exists"})
                continue
            cmd = [
                "python", ATTACH,
                "--case", case,
                "--aneurysm_mesh", src,
                "--aneurysm_space", "ghd_local",
                "--jagged_amp", str(args.jagged_amp),
                "--cut_slab", str(args.cut_slab),
                "--radius_scale", str(args.radius_scale),
                "--out_mesh", dst,
            ]
            if args.use_vessel_submesh:
                cmd.append("--use_vessel_submesh")
            if args.no_remesh_loops:
                cmd.append("--no-remesh_loops")
            if args.fuse_rims:
                cmd += [
                    "--fuse_rims",
                    "--fuse_band_rings", str(args.fuse_band_rings),
                    "--fuse_sigma_rings", str(args.fuse_sigma_rings),
                    "--fuse_smoother", args.fuse_smoother,
                    "--fuse_smooth_iters", str(args.fuse_smooth_iters),
                    "--fuse_smooth_lam", str(args.fuse_smooth_lam),
                    "--fuse_smooth_nu", str(args.fuse_smooth_nu),
                    "--fuse_bridge_steps", str(args.fuse_bridge_steps),
                    "--fuse_bridge_alpha", str(args.fuse_bridge_alpha),
                ]
            if args.open_bridge:
                cmd += [
                    "--open_bridge",
                    "--open_bridge_steps", str(args.open_bridge_steps),
                    "--bridge_smooth_iters", str(args.bridge_smooth_iters),
                    "--bridge_smooth_lam", str(args.bridge_smooth_lam),
                    "--bridge_smooth_nu", str(args.bridge_smooth_nu),
                ]
                if args.rim_presmooth:
                    cmd += [
                        "--rim_presmooth",
                        "--rim_presmooth_rings", str(args.rim_presmooth_rings),
                        "--rim_presmooth_iters", str(args.rim_presmooth_iters),
                        "--rim_presmooth_lam", str(args.rim_presmooth_lam),
                        "--rim_presmooth_nu", str(args.rim_presmooth_nu),
                    ]
            log_path = None
            if args.log_dir:
                log_path = os.path.join(args.log_dir, f"{case}__{tag}.log")
            try:
                with open(log_path, "w") if log_path else open(os.devnull, "w") as fh:
                    r = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                       timeout=300)
                if r.returncode == 0 and os.path.isfile(dst):
                    ok_entry = {"case": case, "tag": tag}
                    ok_entry.update(_read_attach_report_metrics(log_path))
                    summary["ok"].append(ok_entry)
                    status = "ok"
                else:
                    summary["fail"].append({"case": case, "tag": tag,
                                              "rc": r.returncode, "log": log_path})
                    status = f"fail rc={r.returncode}"
            except subprocess.TimeoutExpired:
                summary["fail"].append({"case": case, "tag": tag,
                                          "reason": "timeout", "log": log_path})
                status = "timeout"
            print(f"[{ci+1}/{len(cases)}] {case} {tag}: {status}", flush=True)
        elapsed = time.time() - t0
        rate = (ci + 1) / elapsed if elapsed > 0 else 0.0
        eta = (len(cases) - ci - 1) / rate if rate > 0 else 0.0
        print(f"   ... elapsed {elapsed/60:.1f} min  eta {eta/60:.1f} min  "
              f"ok={len(summary['ok'])} skip={len(summary['skip'])} fail={len(summary['fail'])}",
              flush=True)

    print("\n=== DONE ===")
    print(f"  ok   : {len(summary['ok'])}")
    print(f"  skip : {len(summary['skip'])}")
    print(f"  fail : {len(summary['fail'])}")
    if args.report_json:
        os.makedirs(os.path.dirname(args.report_json), exist_ok=True)
        json.dump(summary, open(args.report_json, "w"), indent=2)
        print(f"  report -> {args.report_json}")


if __name__ == "__main__":
    main()
