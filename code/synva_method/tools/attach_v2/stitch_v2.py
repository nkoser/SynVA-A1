"""
attach_v2: Stitch a generated open aneurysm OBJ onto the patient vessel
submesh, using the EXACT pipeline from the reference repo at
/path/to/reference-vessel-mesh-editing/code/inference/run_inference_pipeline.py

Pipeline (per case x tag):
  1. Load vessel_submesh.obj as-is (open ostium boundary already present).
  2. Load aneurysm OBJ in raw/prepared coords.
  3. Compute opening_indices from open boundary of aneurysm.
  4. Isotropic-remesh aneurysm to vessel median edge length (pymeshlab).
  5. Recover opening indices on the remeshed aneurysm.
  6. infer_vessel_labels_from_ostium  (label vessel ostium vertices = 2).
  7. stitch_meshes_bridge(bridge_steps=1, loop_source="auto").
  8. smooth_ostium_transition_band(iterations=8, hops=4).
  9. fix_normals + save.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import trimesh

REF_PIPELINE_PATH = "/path/to/reference-vessel-mesh-editing/code/inference/run_inference_pipeline.py"


def _import_ref():
    spec = importlib.util.spec_from_file_location("ref_pipeline", REF_PIPELINE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ref_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


REF = _import_ref()


def _load_ostium(case_dir: Path):
    centroid = np.load(case_dir / "07_other" / "centroid_ostium.npy").astype(np.float64).reshape(3)
    normal = np.load(case_dir / "07_other" / "normal_vector.npy").astype(np.float64).reshape(3)
    nrm = np.linalg.norm(normal)
    if nrm < 1e-12:
        raise RuntimeError("zero ostium normal")
    normal = normal / nrm
    ply_path = case_dir / "04_subpointclouds" / "subpointcloud_label_2.ply"
    ostium_points = REF.load_pointcloud_vertices(ply_path)
    return centroid, normal, ostium_points


def attach_one(
    aneurysm_obj: Path,
    vessel_submesh: Path,
    case_dir: Path,
    out_obj: Path,
    bridge_steps: int = 1,
    smooth_iters: int = 8,
    smooth_hops: int = 4,
    remesh_iters: int = 10,
    target_edge_scale: float = 1.0,
):
    out_obj.parent.mkdir(parents=True, exist_ok=True)
    centroid, normal, ostium_points = _load_ostium(case_dir)

    vessel = REF.load_mesh(vessel_submesh)
    pouch = REF.load_mesh(aneurysm_obj)

    # Step 1: opening indices from raw aneurysm (open boundary)
    opening_indices = REF.recover_opening_indices_from_ostium(pouch, ostium_points)
    if opening_indices.size < 3:
        raise RuntimeError("could not recover aneurysm opening indices")

    # Step 2: isotropic-remesh aneurysm to vessel median edge length
    vstats = REF.mesh_edge_stats(vessel)
    if vstats["median"] is None:
        raise RuntimeError("vessel mesh has no edges")
    target_edge_length = float(vstats["median"]) * float(target_edge_scale)
    pouch, _ = REF.remesh_to_target_edge_length(
        pouch, target_edge_length=target_edge_length, iterations=int(remesh_iters)
    )
    opening_indices = REF.recover_opening_indices_from_ostium(pouch, ostium_points)
    if opening_indices.size < 3:
        raise RuntimeError("could not recover opening indices after remesh")

    # Step 3: vessel labels (2 = ostium)
    vessel_labels = REF.infer_vessel_labels_from_ostium(vessel, ostium_points, ostium_label=2)

    # Step 4: bridge stitch
    stitched, stitched_labels, _matches, _meta = REF.stitch_meshes_bridge(
        vessel_submesh=vessel,
        labels_vessel_submesh=vessel_labels,
        transformed_mesh=pouch,
        ostium_points=ostium_points,
        opening_indices=opening_indices,
        normal=normal,
        bridge_steps=int(bridge_steps),
        ostium_label=2,
        transformed_label=1,
        merge_digits=12,
        loop_source="auto",
        smooth_intersection_enabled=False,
    )

    # Step 5: ostium transition band smoothing (N-hop neighbor average)
    if smooth_iters > 0 and smooth_hops > 0:
        stitched, _ = REF.smooth_ostium_transition_band(
            mesh=stitched,
            labels=stitched_labels,
            ostium_points=ostium_points,
            iterations=int(smooth_iters),
            hops=int(smooth_hops),
            ostium_label=2,
        )

    stitched.fix_normals()
    REF.save_mesh(stitched, out_obj)
    return {
        "vertices": int(len(stitched.vertices)),
        "faces": int(len(stitched.faces)),
        "winding_consistent": bool(stitched.is_winding_consistent),
        "watertight": bool(stitched.is_watertight),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases_file", required=True,
                   help="JSON list of case ids (or .txt one per line)")
    p.add_argument("--samples_dir", required=True,
                   help="dir with <case>/<tag>.obj inputs (raw-space aneurysms)")
    p.add_argument("--prepared_root", default="/path/to/prepared_meshes_3")
    p.add_argument("--tags", nargs="+", default=["gt", "A", "C", "D", "E", "baseline"])
    p.add_argument("--out_dir", default=None,
                   help="output dir; defaults to samples_dir (writes <tag>_attached_v2.obj)")
    p.add_argument("--out_suffix", default="_attached_v2")
    p.add_argument("--bridge_steps", type=int, default=1)
    p.add_argument("--smooth_iters", type=int, default=8)
    p.add_argument("--smooth_hops", type=int, default=4)
    p.add_argument("--remesh_iters", type=int, default=10)
    p.add_argument("--target_edge_scale", type=float, default=1.0)
    p.add_argument("--report_json", default=None)
    p.add_argument("--max_cases", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    cases_file = Path(args.cases_file)
    text = cases_file.read_text()
    try:
        cases = json.loads(text)
        if isinstance(cases, dict):
            cases = list(cases.keys())
    except Exception:
        cases = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    samples_dir = Path(args.samples_dir)
    out_root = Path(args.out_dir) if args.out_dir else samples_dir
    out_root.mkdir(parents=True, exist_ok=True)

    report = {"args": vars(args), "results": {}}
    n_ok = n_fail = n_skip = 0
    for case in cases:
        case_dir = Path(args.prepared_root) / case
        vessel_submesh = case_dir / "05_submeshes" / "vessel_submesh.obj"
        if not vessel_submesh.exists():
            print(f"[skip] {case}: no vessel_submesh.obj", flush=True)
            n_skip += 1
            continue
        case_in = samples_dir / case
        case_out = out_root / case
        case_out.mkdir(parents=True, exist_ok=True)
        report["results"][case] = {}
        for tag in args.tags:
            in_obj = case_in / f"{tag}.obj"
            out_obj = case_out / f"{tag}{args.out_suffix}.obj"
            if not in_obj.exists():
                report["results"][case][tag] = {"status": "skip_no_input"}
                n_skip += 1
                continue
            if out_obj.exists() and not args.overwrite:
                report["results"][case][tag] = {"status": "skip_exists"}
                n_skip += 1
                continue
            try:
                stats = attach_one(
                    aneurysm_obj=in_obj,
                    vessel_submesh=vessel_submesh,
                    case_dir=case_dir,
                    out_obj=out_obj,
                    bridge_steps=args.bridge_steps,
                    smooth_iters=args.smooth_iters,
                    smooth_hops=args.smooth_hops,
                    remesh_iters=args.remesh_iters,
                    target_edge_scale=args.target_edge_scale,
                )
                report["results"][case][tag] = {"status": "ok", **stats}
                n_ok += 1
                print(f"[ok]   {case}/{tag}  V={stats['vertices']} F={stats['faces']} wc={stats['winding_consistent']}",
                      flush=True)
            except Exception as e:
                tb = traceback.format_exc(limit=4)
                report["results"][case][tag] = {"status": "fail", "error": str(e), "tb": tb}
                n_fail += 1
                print(f"[FAIL] {case}/{tag}: {e}", flush=True)

    summary = {"ok": n_ok, "fail": n_fail, "skip": n_skip}
    report["summary"] = summary
    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(json.dumps(report, indent=2))
    print(f"\n=== DONE  ok={n_ok}  fail={n_fail}  skip={n_skip} ===", flush=True)


if __name__ == "__main__":
    main()
