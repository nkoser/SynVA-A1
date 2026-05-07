#!/usr/bin/env python
"""Post-process reference stitched meshes with constrained ostium seam fairing.

This deliberately runs after the vessel-mesh-editing-master Step 3 output:
the reference stitching creates the topology, labels the bridge/ostium band as
2, and this script only fair-smooths a narrow graph neighborhood around that
label. The original files are copied into a new output root and the original
stitched OBJ is preserved as *_pre_fair.obj.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import deque
from pathlib import Path

import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_root", required=True, help="Reference run root containing cases/test/<case>/outputs/final.")
    p.add_argument("--out_root", required=True, help="New run root for faired outputs. Existing files are not overwritten unless --overwrite.")
    p.add_argument("--cases_file", default=None, help="Optional JSON list or text file with case ids.")
    p.add_argument("--case_split", default="test")
    p.add_argument("--max_cases", type=int, default=0)
    p.add_argument("--ostium_label", type=int, default=2)
    p.add_argument("--hops", type=int, default=5, help="Graph radius around ostium/bridge label to fair.")
    p.add_argument("--iterations", type=int, default=24)
    p.add_argument("--method", choices=["taubin", "harmonic"], default="taubin")
    p.add_argument("--relax", type=float, default=0.65, help="Relaxation for --method harmonic.")
    p.add_argument("--blend", type=float, default=1.0, help="Blend fair result with original vertices in the selected band.")
    p.add_argument("--lamb", type=float, default=0.42, help="Positive Taubin step.")
    p.add_argument("--nu", type=float, default=0.45, help="Negative Taubin step magnitude.")
    p.add_argument("--anchor_power", type=float, default=2.0, help="Higher values keep the outer band closer to the reference mesh.")
    p.add_argument("--anchor_min", type=float, default=0.02, help="Minimum pullback to the reference mesh inside the seam.")
    p.add_argument("--anchor_max", type=float, default=0.65, help="Maximum pullback at the outer band.")
    p.add_argument("--label0_mobility", type=float, default=0.55, help="Mobility for vessel-side vertices.")
    p.add_argument("--label1_mobility", type=float, default=0.75, help="Mobility for pouch-side vertices.")
    p.add_argument("--label2_mobility", type=float, default=1.0, help="Mobility for bridge/ostium vertices.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def read_cases(path: Path | None, input_root: Path, case_split: str) -> list[str]:
    if path is not None:
        text = path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                data = data.get("ok", list(data.keys()))
            return [str(x) for x in data]
        except Exception:
            return [line.strip() for line in text.splitlines() if line.strip()]
    cases_dir = input_root / "cases" / case_split
    return sorted(p.name for p in cases_dir.iterdir() if p.is_dir())


def build_adjacency(n_vertices: int, faces: np.ndarray) -> list[np.ndarray]:
    neigh: list[set[int]] = [set() for _ in range(n_vertices)]
    for tri in np.asarray(faces, dtype=np.int64):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        neigh[a].update((b, c))
        neigh[b].update((a, c))
        neigh[c].update((a, b))
    return [np.asarray(sorted(s), dtype=np.int64) for s in neigh]


def graph_distance_from_seed(adj: list[np.ndarray], seed: np.ndarray, max_hops: int) -> np.ndarray:
    dist = np.full(len(adj), max_hops + 1, dtype=np.int32)
    q: deque[int] = deque()
    for idx in np.unique(np.asarray(seed, dtype=np.int64)):
        if 0 <= int(idx) < len(adj):
            dist[int(idx)] = 0
            q.append(int(idx))
    while q:
        cur = q.popleft()
        if dist[cur] >= max_hops:
            continue
        for nxt in adj[cur]:
            nxt_i = int(nxt)
            if dist[nxt_i] > dist[cur] + 1:
                dist[nxt_i] = dist[cur] + 1
                q.append(nxt_i)
    return dist


def taubin_fair_band(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    ostium_label: int,
    hops: int,
    iterations: int,
    lamb: float,
    nu: float,
    anchor_power: float,
    anchor_min: float,
    anchor_max: float,
    label_mobility: dict[int, float],
) -> tuple[np.ndarray, dict[str, object]]:
    adj = build_adjacency(len(vertices), faces)
    seed = np.flatnonzero(labels == int(ostium_label))
    if seed.size == 0:
        return vertices.copy(), {"enabled": True, "message": "No ostium/bridge label vertices found."}

    dist = graph_distance_from_seed(adj, seed, max(0, int(hops)))
    active = np.flatnonzero(dist <= int(hops))
    if active.size == 0:
        return vertices.copy(), {"enabled": True, "message": "Empty fairing band."}

    original = np.asarray(vertices, dtype=np.float64).copy()
    current = original.copy()

    denom = max(float(hops), 1.0)
    t = np.clip(dist.astype(np.float64) / denom, 0.0, 1.0)
    mobility = np.zeros(len(vertices), dtype=np.float64)
    for label in np.unique(labels):
        mobility[labels == label] = float(label_mobility.get(int(label), 0.0))
    mobility *= np.clip(1.0 - (t ** max(float(anchor_power), 1e-6)), 0.0, 1.0)
    mobility[active] = np.maximum(mobility[active], 0.05)
    mobility[dist > int(hops)] = 0.0

    anchor = float(anchor_min) + (float(anchor_max) - float(anchor_min)) * (t ** max(float(anchor_power), 1e-6))
    anchor = np.clip(anchor, 0.0, 1.0)
    anchor[dist > int(hops)] = 1.0

    active_list = [int(i) for i in active]

    def step(src: np.ndarray, factor: float) -> np.ndarray:
        dst = src.copy()
        for idx in active_list:
            ni = adj[idx]
            if ni.size == 0:
                continue
            lap = src[ni].mean(axis=0) - src[idx]
            proposed = src[idx] + float(factor) * mobility[idx] * lap
            dst[idx] = (1.0 - anchor[idx]) * proposed + anchor[idx] * original[idx]
        return dst

    for _ in range(max(0, int(iterations))):
        current = step(current, float(lamb))
        current = step(current, -float(nu))

    displacement = np.linalg.norm(current - original, axis=1)
    label_counts = {
        str(int(label)): int(np.count_nonzero((labels == label) & (dist <= int(hops))))
        for label in np.unique(labels)
    }
    return current, {
        "enabled": True,
        "method": "constrained_windowed_taubin",
        "seed_vertices": int(seed.size),
        "selected_vertices": int(active.size),
        "selected_label_counts": label_counts,
        "hops": int(hops),
        "iterations": int(iterations),
        "lamb": float(lamb),
        "nu": float(nu),
        "anchor_min": float(anchor_min),
        "anchor_max": float(anchor_max),
        "anchor_power": float(anchor_power),
        "max_displacement": float(displacement[active].max()) if active.size else 0.0,
        "mean_displacement": float(displacement[active].mean()) if active.size else 0.0,
        "p95_displacement": float(np.percentile(displacement[active], 95)) if active.size else 0.0,
    }


def harmonic_fair_band(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    ostium_label: int,
    hops: int,
    iterations: int,
    relax: float,
    blend: float,
    anchor_power: float,
    anchor_min: float,
    anchor_max: float,
    label_mobility: dict[int, float],
) -> tuple[np.ndarray, dict[str, object]]:
    adj = build_adjacency(len(vertices), faces)
    seed = np.flatnonzero(labels == int(ostium_label))
    if seed.size == 0:
        return vertices.copy(), {"enabled": True, "message": "No ostium/bridge label vertices found."}

    dist = graph_distance_from_seed(adj, seed, max(0, int(hops)))
    active = np.flatnonzero(dist <= int(hops))
    if active.size == 0:
        return vertices.copy(), {"enabled": True, "message": "Empty fairing band."}

    original = np.asarray(vertices, dtype=np.float64).copy()
    current = original.copy()

    denom = max(float(hops), 1.0)
    t = np.clip(dist.astype(np.float64) / denom, 0.0, 1.0)
    mobility = np.zeros(len(vertices), dtype=np.float64)
    for label in np.unique(labels):
        mobility[labels == label] = float(label_mobility.get(int(label), 0.0))
    mobility *= np.clip(1.0 - (t ** max(float(anchor_power), 1e-6)), 0.0, 1.0)
    mobility[active] = np.maximum(mobility[active], 0.05)
    mobility[dist > int(hops)] = 0.0

    fidelity = float(anchor_min) + (float(anchor_max) - float(anchor_min)) * (t ** max(float(anchor_power), 1e-6))
    fidelity = np.clip(fidelity, 0.0, 1.0)
    fidelity[dist > int(hops)] = 1.0

    active_list = [int(i) for i in active]
    relax = float(np.clip(relax, 0.0, 1.0))
    for _ in range(max(0, int(iterations))):
        prev = current.copy()
        for idx in active_list:
            ni = adj[idx]
            if ni.size == 0:
                continue
            harmonic = prev[ni].mean(axis=0)
            target = (1.0 - fidelity[idx]) * harmonic + fidelity[idx] * original[idx]
            current[idx] = (1.0 - relax * mobility[idx]) * prev[idx] + (relax * mobility[idx]) * target

    blend = float(np.clip(blend, 0.0, 1.0))
    result = original.copy()
    result[active] = (1.0 - blend) * original[active] + blend * current[active]

    displacement = np.linalg.norm(result - original, axis=1)
    label_counts = {
        str(int(label)): int(np.count_nonzero((labels == label) & (dist <= int(hops))))
        for label in np.unique(labels)
    }
    return result, {
        "enabled": True,
        "method": "constrained_harmonic_band",
        "seed_vertices": int(seed.size),
        "selected_vertices": int(active.size),
        "selected_label_counts": label_counts,
        "hops": int(hops),
        "iterations": int(iterations),
        "relax": float(relax),
        "blend": float(blend),
        "anchor_min": float(anchor_min),
        "anchor_max": float(anchor_max),
        "anchor_power": float(anchor_power),
        "max_displacement": float(displacement[active].max()) if active.size else 0.0,
        "mean_displacement": float(displacement[active].mean()) if active.size else 0.0,
        "p95_displacement": float(np.percentile(displacement[active], 95)) if active.size else 0.0,
    }


def copy_final_dir(src_final: Path, dst_final: Path, overwrite: bool) -> None:
    dst_final.mkdir(parents=True, exist_ok=True)
    for src in src_final.iterdir():
        if not src.is_file():
            continue
        dst = dst_final / src.name
        if dst.exists() and not overwrite:
            continue
        shutil.copy2(src, dst)


def process_case(args: argparse.Namespace, case: str) -> dict[str, object]:
    input_root = Path(args.input_root)
    out_root = Path(args.out_root)
    src_final = input_root / "cases" / args.case_split / case / "outputs" / "final"
    dst_final = out_root / "cases" / args.case_split / case / "outputs" / "final"
    stitched = src_final / f"{case}_vessel_with_generated_aneurysm_stitched.obj"
    labels_path = src_final / f"{case}_vessel_with_generated_aneurysm_stitched_labels.npy"
    if not stitched.exists():
        raise FileNotFoundError(str(stitched))
    if not labels_path.exists():
        raise FileNotFoundError(str(labels_path))

    copy_final_dir(src_final, dst_final, bool(args.overwrite))
    pre_fair = dst_final / f"{case}_vessel_with_generated_aneurysm_stitched_pre_fair.obj"
    if not pre_fair.exists() or args.overwrite:
        shutil.copy2(stitched, pre_fair)

    mesh = trimesh.load_mesh(stitched, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    labels = np.load(labels_path).astype(np.int64)
    if labels.shape[0] != len(mesh.vertices):
        raise ValueError(f"{case}: labels length {labels.shape[0]} != vertices {len(mesh.vertices)}")

    common_kwargs = dict(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        labels=labels,
        ostium_label=int(args.ostium_label),
        hops=int(args.hops),
        iterations=int(args.iterations),
        anchor_power=float(args.anchor_power),
        anchor_min=float(args.anchor_min),
        anchor_max=float(args.anchor_max),
        label_mobility={
            0: float(args.label0_mobility),
            1: float(args.label1_mobility),
            2: float(args.label2_mobility),
        },
    )
    if args.method == "harmonic":
        new_vertices, fairing = harmonic_fair_band(
            **common_kwargs,
            relax=float(args.relax),
            blend=float(args.blend),
        )
    else:
        new_vertices, fairing = taubin_fair_band(
            **common_kwargs,
            lamb=float(args.lamb),
            nu=float(args.nu),
        )

    out_mesh = mesh.copy()
    out_mesh.vertices = new_vertices
    out_mesh.fix_normals()
    out_path = dst_final / stitched.name
    out_mesh.export(out_path)

    result = {
        "case": case,
        "input_mesh": str(stitched),
        "output_mesh": str(out_path),
        "pre_fair_mesh": str(pre_fair),
        "vertices": int(len(out_mesh.vertices)),
        "faces": int(len(out_mesh.faces)),
        "watertight": bool(out_mesh.is_watertight),
        "winding_consistent": bool(out_mesh.is_winding_consistent),
        "fairing": fairing,
    }
    (dst_final / f"{case}_seam_fairing_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    out_root = Path(args.out_root)
    if out_root.exists() and not args.overwrite:
        raise FileExistsError(f"{out_root} already exists; pass --overwrite or choose a new --out_root.")
    out_root.mkdir(parents=True, exist_ok=True)

    cases = read_cases(Path(args.cases_file) if args.cases_file else None, input_root, args.case_split)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    ok: list[dict[str, object]] = []
    failed: list[dict[str, str]] = []
    for idx, case in enumerate(cases, start=1):
        print(f"[{idx}/{len(cases)}] fair seam {case}", flush=True)
        try:
            result = process_case(args, case)
            ok.append(result)
            f = result["fairing"]
            print(
                "  selected={selected_vertices} mean_disp={mean_displacement:.6f} "
                "p95_disp={p95_displacement:.6f} max_disp={max_displacement:.6f}".format(**f),
                flush=True,
            )
        except Exception as exc:
            failed.append({"case": case, "error": str(exc)})
            print(f"  [failed] {exc}", flush=True)

    summary = {
        "args": vars(args),
        "ok_count": len(ok),
        "failed_count": len(failed),
        "ok": ok,
        "failed": failed,
    }
    (out_root / "seam_fairing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"ok_count": len(ok), "failed_count": len(failed), "summary": str(out_root / "seam_fairing_summary.json")}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
