#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree


sys.path.insert(0, "/data")
from compute_morphology import (  # noqa: E402
    _safe_div,
    compute_a_o,
    compute_ch,
    compute_morphological_parameters,
    compute_ei,
    compute_h_max,
    compute_h_ortho,
    compute_n_avg,
    compute_nsi,
)


SCALAR_MORPH_KEYS = [
    "A_A",
    "V_A",
    "A_O1",
    "A_O2",
    "D_max",
    "H_max",
    "W_max",
    "H_ortho",
    "W_ortho",
    "N_max",
    "N_avg",
    "AR_1",
    "AR_2",
    "V_CH",
    "A_CH",
    "EI",
    "NSI",
    "UI",
]


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def finite_float(value):
    try:
        value = float(np.asarray(value).reshape(-1)[0])
    except Exception:
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def rel_diff(pred, gt):
    pred = finite_float(pred)
    gt = finite_float(gt)
    if not math.isfinite(pred) or not math.isfinite(gt) or gt == 0.0:
        return float("nan")
    return (pred - gt) / gt


def abs_diff(pred, gt):
    pred = finite_float(pred)
    gt = finite_float(gt)
    if not math.isfinite(pred) or not math.isfinite(gt):
        return float("nan")
    return abs(pred - gt)


def convex_hull_volume(vertices):
    vertices = np.asarray(vertices)
    if vertices.ndim != 2 or vertices.shape[0] < 4:
        return float("nan")
    if np.unique(vertices, axis=0).shape[0] < 4:
        return float("nan")
    try:
        return finite_float(trimesh.convex.convex_hull(vertices).volume)
    except Exception:
        return float("nan")


def pairwise_max_distance(vertices):
    vertices = np.asarray(vertices)
    if len(vertices) < 2:
        return 0.0
    return float(np.max(pdist(vertices)))


def ostium_max_distance(vertices, labels):
    ostium_vertices = np.asarray(vertices)[np.asarray(labels) == 2]
    return pairwise_max_distance(ostium_vertices)


def normalize_case_key(name: str) -> str:
    key = name
    if key.startswith("aneux_"):
        key = key[len("aneux_") :]
    if key.startswith("cmha_"):
        key = "cmch_" + key[len("cmha_") :]
    return key


def build_gt_index(gt_root: Path):
    index = {}
    for folder in sorted(p for p in gt_root.iterdir() if p.is_dir()):
        required = [
            folder / "05_submeshes" / "aneurysm_submesh.obj",
            folder / "06_submesh_labels" / "labels_aneurysm.npy",
            folder / "07_other" / "centroid_ostium.npy",
            folder / "07_other" / "normal_vector.npy",
        ]
        if not all(p.exists() for p in required):
            continue
        candidates = {
            folder.name,
            normalize_case_key(folder.name),
        }
        if folder.name.startswith("cmch_"):
            candidates.add("cmha_" + folder.name[len("cmch_") :])
        if folder.name.startswith("aneux_"):
            raw = folder.name[len("aneux_") :]
            candidates.add(raw)
            if raw.startswith("cmch_"):
                candidates.add("cmha_" + raw[len("cmch_") :])
        for key in candidates:
            index[key] = folder
    return index


def load_mesh(path: Path):
    mesh = trimesh.load(path, process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"not a Trimesh: {path}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"empty mesh: {path}")
    return mesh


def chamfer_loss_sampled(mesh_gt, mesh_pred, samples, device, seed, backend):
    np.random.seed(seed)
    vertices_pred_np = mesh_pred.sample(samples)
    np.random.seed(seed + 1)
    vertices_gt_np = mesh_gt.sample(samples)

    if backend == "scipy":
        dist1 = cKDTree(vertices_pred_np).query(vertices_gt_np, k=1, workers=-1)[0]
        dist2 = cKDTree(vertices_gt_np).query(vertices_pred_np, k=1, workers=-1)[0]
        return float(dist1.mean() + dist2.mean())

    vertices_pred = torch.tensor(vertices_pred_np, device=device, dtype=torch.float32)
    vertices_gt = torch.tensor(vertices_gt_np, device=device, dtype=torch.float32)
    dist1 = torch.cdist(vertices_gt, vertices_pred, p=2).min(dim=1)[0]
    dist2 = torch.cdist(vertices_pred, vertices_gt, p=2).min(dim=1)[0]
    chamfer_loss = dist1.mean() + dist2.mean()
    return float(chamfer_loss.detach().cpu().item())


def max_width_bruteforce(mesh, vertices, centroid_ostium, axis_vector):
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    centroid = np.asarray(centroid_ostium, dtype=np.float64).reshape(3)
    axis = np.asarray(axis_vector, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(axis)
    if norm < 1e-12 or len(vertices) == 0 or len(triangles) == 0:
        return float("nan")
    axis = axis / norm

    eps = 1e-5
    best = 0.0
    tri_v0 = triangles[:, 0]
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]

    for vertex in vertices:
        closest_point_on_axis = centroid + np.dot(vertex - centroid, axis) * axis
        direction = vertex - closest_point_on_axis
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-6:
            continue
        direction = direction / direction_norm
        ray_origin = vertex - direction * eps
        ray_direction = -direction

        h = np.cross(np.broadcast_to(ray_direction, edge2.shape), edge2)
        a = np.einsum("ij,ij->i", edge1, h)
        valid = np.abs(a) > 1e-12
        if not np.any(valid):
            continue

        f = np.zeros_like(a)
        f[valid] = 1.0 / a[valid]
        s = ray_origin - tri_v0
        u = f * np.einsum("ij,ij->i", s, h)
        valid &= (u >= -1e-9) & (u <= 1.0 + 1e-9)
        if not np.any(valid):
            continue

        q = np.cross(s, edge1)
        v = f * np.einsum("j,ij->i", ray_direction, q)
        valid &= (v >= -1e-9) & ((u + v) <= 1.0 + 1e-9)
        if not np.any(valid):
            continue

        t = f * np.einsum("ij,ij->i", edge2, q)
        valid &= t > eps
        if np.any(valid):
            best = max(best, float(np.max(t[valid])))

    return best


def compute_morphology_with_bruteforce_width(mesh, labels, centroid_ostium, normal_vector_ostium):
    vertices = np.asarray(mesh.vertices)
    labels = np.asarray(labels)
    params = {}
    params["C_O"] = np.asarray(centroid_ostium)
    params["A_A"] = finite_float(mesh.area)
    params["V_A"] = convex_hull_volume(vertices)
    a_o1, a_o2 = compute_a_o(mesh, labels, centroid_ostium, normal_vector_ostium)
    params["A_O1"] = finite_float(a_o1)
    params["A_O2"] = finite_float(a_o2)
    params["D_max"] = finite_float(pairwise_max_distance(vertices))
    h_max, highest_vertex = compute_h_max(vertices, centroid_ostium)
    params["H_max"] = finite_float(h_max)
    params["W_max"] = finite_float(max_width_bruteforce(mesh, vertices, centroid_ostium, highest_vertex - np.asarray(centroid_ostium).reshape(3)))
    params["H_ortho"] = finite_float(compute_h_ortho(vertices, centroid_ostium, normal_vector_ostium))
    params["W_ortho"] = finite_float(max_width_bruteforce(mesh, vertices, centroid_ostium, normal_vector_ostium))
    params["N_max"] = finite_float(ostium_max_distance(vertices, labels))
    params["N_avg"] = finite_float(compute_n_avg(vertices, labels, centroid_ostium))
    params["AR_1"] = finite_float(_safe_div(params["H_ortho"], params["N_max"]))
    params["AR_2"] = finite_float(_safe_div(params["H_ortho"], params["N_avg"]))
    v_ch, a_ch = compute_ch(vertices)
    params["V_CH"] = finite_float(v_ch)
    params["A_CH"] = finite_float(a_ch)
    params["EI"] = finite_float(compute_ei(params["V_CH"], params["A_CH"]))
    params["NSI"] = finite_float(compute_nsi(params["V_A"], params["A_A"]))
    ui_ratio = _safe_div(params["V_A"], params["V_CH"])
    params["UI"] = float("nan") if not math.isfinite(finite_float(ui_ratio)) else 1.0 - finite_float(ui_ratio)
    return params


def extract_generated_labeled_submesh(case_dir: Path, case_name: str):
    final_dir = case_dir / "outputs" / "final"
    stitched_path = final_dir / f"{case_name}_vessel_with_generated_aneurysm_stitched.obj"
    labels_path = final_dir / f"{case_name}_vessel_with_generated_aneurysm_stitched_labels.npy"
    stitched = load_mesh(stitched_path)
    labels = np.load(labels_path)
    if len(labels) != len(stitched.vertices):
        raise ValueError(f"stitched labels length mismatch for {case_name}: {len(labels)} != {len(stitched.vertices)}")

    keep_vertices = labels > 0
    keep_faces = keep_vertices[np.asarray(stitched.faces)].all(axis=1)
    old_to_new = -np.ones(len(labels), dtype=np.int64)
    old_ids = np.flatnonzero(keep_vertices)
    old_to_new[old_ids] = np.arange(len(old_ids), dtype=np.int64)
    faces = old_to_new[np.asarray(stitched.faces)[keep_faces]]
    mesh = trimesh.Trimesh(vertices=np.asarray(stitched.vertices)[old_ids], faces=faces, process=False)
    sublabels = labels[old_ids]
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"generated labeled aneurysm submesh is empty for {case_name}")
    return mesh, sublabels


def gt_paths(gt_case_dir: Path):
    return {
        "mesh": gt_case_dir / "05_submeshes" / "aneurysm_submesh.obj",
        "labels": gt_case_dir / "06_submesh_labels" / "labels_aneurysm.npy",
        "centroid": gt_case_dir / "07_other" / "centroid_ostium.npy",
        "normal": gt_case_dir / "07_other" / "normal_vector.npy",
    }


def compute_case(row_base, case_dir, pred_world_path, gt_case_dir, samples, device, chamfer_backend, morphology_backend):
    case_name = row_base["case"]
    paths = gt_paths(gt_case_dir)

    pred_world = load_mesh(pred_world_path)
    gt_mesh = load_mesh(paths["mesh"])
    gt_labels = np.load(paths["labels"])
    gt_centroid = np.load(paths["centroid"])
    gt_normal = np.load(paths["normal"])

    seed = stable_seed(row_base["variant"], case_name)
    row = dict(row_base)
    row["gt_case"] = gt_case_dir.name
    row["pred_world_vertices"] = len(pred_world.vertices)
    row["pred_world_faces"] = len(pred_world.faces)
    row["gt_vertices"] = len(gt_mesh.vertices)
    row["gt_faces"] = len(gt_mesh.faces)
    row["chamfer_10k"] = chamfer_loss_sampled(gt_mesh, pred_world, samples, device, seed, chamfer_backend)

    if morphology_backend == "exact":
        gt_params = compute_morphological_parameters(gt_mesh, gt_labels, gt_centroid, gt_normal)
    else:
        gt_params = compute_morphology_with_bruteforce_width(gt_mesh, gt_labels, gt_centroid, gt_normal)
    pred_labeled_mesh, pred_labels = extract_generated_labeled_submesh(case_dir, case_name)
    pred_centroid = np.load(case_dir / "07_other" / "centroid_ostium.npy")
    pred_normal = np.load(case_dir / "07_other" / "normal_vector.npy")
    if morphology_backend == "exact":
        pred_params = compute_morphological_parameters(pred_labeled_mesh, pred_labels, pred_centroid, pred_normal)
    else:
        pred_params = compute_morphology_with_bruteforce_width(pred_labeled_mesh, pred_labels, pred_centroid, pred_normal)

    pred_c = np.asarray(pred_params["C_O"], dtype=np.float64).reshape(-1)[:3]
    gt_c = np.asarray(gt_params["C_O"], dtype=np.float64).reshape(-1)[:3]
    row["C_O_dist"] = float(np.linalg.norm(pred_c - gt_c)) if pred_c.size == 3 and gt_c.size == 3 else float("nan")

    for key in SCALAR_MORPH_KEYS:
        pred_value = finite_float(pred_params.get(key))
        gt_value = finite_float(gt_params.get(key))
        row[f"pred_{key}"] = pred_value
        row[f"gt_{key}"] = gt_value
        row[f"absdiff_{key}"] = abs_diff(pred_value, gt_value)
        row[f"reldiff_{key}"] = rel_diff(pred_value, gt_value)
    return row


def discover_pred_cases(reference_root: Path):
    for variant_dir in sorted(p for p in reference_root.iterdir() if p.is_dir()):
        cases_root = variant_dir / "cases" / "test"
        if not cases_root.exists():
            continue
        for pred_world_path in sorted(cases_root.glob("*/outputs/final/*_generated_aneurysm_world.obj")):
            case_dir = pred_world_path.parents[2]
            case_name = case_dir.name
            yield variant_dir.name, case_name, case_dir, pred_world_path


def compute_case_task(task):
    (
        row_base,
        case_dir,
        pred_world_path,
        gt_case_dir,
        samples,
        device,
        chamfer_backend,
        morphology_backend,
    ) = task
    row = compute_case(
        row_base,
        case_dir,
        pred_world_path,
        gt_case_dir,
        samples,
        device,
        chamfer_backend,
        morphology_backend,
    )
    return row


def write_csv(path: Path, rows):
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


def summarize(rows):
    summary = []
    variants = sorted({r["variant"] for r in rows})
    for variant in variants:
        subset = [r for r in rows if r["variant"] == variant]
        item = {"variant": variant, "n": len(subset)}
        for key in ["chamfer_10k", "C_O_dist"] + [f"absdiff_{k}" for k in SCALAR_MORPH_KEYS]:
            values = np.asarray([finite_float(r.get(key)) for r in subset], dtype=np.float64)
            values = values[np.isfinite(values)]
            item[f"mean_{key}"] = float(values.mean()) if values.size else float("nan")
            item[f"median_{key}"] = float(np.median(values)) if values.size else float("nan")
        summary.append(item)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-root", type=Path, default=Path("/path/to/aneug_w_variants_exact_ref_all_test100_20260503_122120/reference_stitching"))
    parser.add_argument("--gt-root", type=Path, default=Path("/path/to/prepared_meshes_2"))
    parser.add_argument("--out-dir", type=Path, default=Path("/path/to/SynVA-A1/analysis_results/reference_stitching_gt_compare_20260503"))
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chamfer-backend", choices=["torch", "scipy"], default="torch")
    parser.add_argument("--morphology-backend", choices=["exact", "robust"], default="robust")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt_index = build_gt_index(args.gt_root)
    rows = []
    errors = []

    pred_cases = list(discover_pred_cases(args.reference_root))
    if args.limit > 0:
        pred_cases = pred_cases[: args.limit]
    print(f"Discovered {len(pred_cases)} generated meshes under {args.reference_root}", flush=True)
    print(f"Indexed {len(gt_index)} GT case aliases under {args.gt_root}", flush=True)
    print(
        f"Using device={args.device}, samples={args.samples}, "
        f"chamfer_backend={args.chamfer_backend}, morphology_backend={args.morphology_backend}",
        flush=True,
    )

    tasks = []
    for i, (variant, case_name, case_dir, pred_world_path) in enumerate(pred_cases, start=1):
        row_base = {"variant": variant, "case": case_name, "pred_world_path": str(pred_world_path)}
        gt_case_dir = gt_index.get(case_name) or gt_index.get(normalize_case_key(case_name))
        if gt_case_dir is None:
            errors.append({**row_base, "error": "no_gt_match"})
            print(f"[{i}/{len(pred_cases)}] {variant}/{case_name}: no GT match", flush=True)
            continue
        tasks.append(
            (
                i,
                (
                    row_base,
                    case_dir,
                    pred_world_path,
                    gt_case_dir,
                    args.samples,
                    args.device,
                    args.chamfer_backend,
                    args.morphology_backend,
                ),
            )
        )

    if args.workers <= 1:
        iterator = tasks
        for i, task in iterator:
            row_base = task[0]
            variant = row_base["variant"]
            case_name = row_base["case"]
            gt_case_dir = task[3]
            try:
                row = compute_case_task(task)
                rows.append(row)
                print(f"[{i}/{len(pred_cases)}] {variant}/{case_name}: chamfer={row['chamfer_10k']:.6g}", flush=True)
            except Exception as exc:
                errors.append({**row_base, "gt_case": gt_case_dir.name, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{i}/{len(pred_cases)}] {variant}/{case_name}: ERROR {type(exc).__name__}: {exc}", flush=True)
    else:
        mp_ctx = mp.get_context("spawn")
        print(f"Parallel workers={args.workers}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp_ctx) as ex:
            futures = {ex.submit(compute_case_task, task): (i, task) for i, task in tasks}
            completed = 0
            for fut in as_completed(futures):
                i, task = futures[fut]
                row_base = task[0]
                variant = row_base["variant"]
                case_name = row_base["case"]
                gt_case_dir = task[3]
                completed += 1
                try:
                    row = fut.result()
                    rows.append(row)
                    print(
                        f"[{completed}/{len(tasks)} done; case {i}/{len(pred_cases)}] "
                        f"{variant}/{case_name}: chamfer={row['chamfer_10k']:.6g}",
                        flush=True,
                    )
                except Exception as exc:
                    errors.append({**row_base, "gt_case": gt_case_dir.name, "error": f"{type(exc).__name__}: {exc}"})
                    print(
                        f"[{completed}/{len(tasks)} done; case {i}/{len(pred_cases)}] "
                        f"{variant}/{case_name}: ERROR {type(exc).__name__}: {exc}",
                        flush=True,
                    )

    summary_rows = summarize(rows)
    write_csv(args.out_dir / "per_case_metrics.csv", rows)
    write_csv(args.out_dir / "summary_by_variant.csv", summary_rows)
    write_csv(args.out_dir / "errors.csv", errors)
    with (args.out_dir / "per_case_metrics.json").open("w") as f:
        json.dump(rows, f, indent=2, allow_nan=True)
    with (args.out_dir / "summary_by_variant.json").open("w") as f:
        json.dump(summary_rows, f, indent=2, allow_nan=True)
    with (args.out_dir / "errors.json").open("w") as f:
        json.dump(errors, f, indent=2, allow_nan=True)

    print(f"Done. rows={len(rows)}, errors={len(errors)}", flush=True)
    print(args.out_dir, flush=True)


if __name__ == "__main__":
    main()
