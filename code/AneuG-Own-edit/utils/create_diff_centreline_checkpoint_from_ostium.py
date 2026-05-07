#!/usr/bin/env python3
"""Create diff_centreline_checkpoint.pkl from aneurysm-only ostium metadata.

Per case sources (zero-root):
- 07_other/centroid_ostium.npy

Per case target (alignment-root):
- part_aligned.obj
- diff_centreline_checkpoint.pkl (generated)
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable

import igraph as ig
import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate diff_centreline_checkpoint.pkl from ostium centroid seed."
    )
    parser.add_argument(
        "--zero-root",
        type=Path,
        default=Path("checkpoints-new/zero-aneurysmen"),
        help="Root with aneurysm-only source folders.",
    )
    parser.add_argument(
        "--alignment-root",
        type=Path,
        default=Path("checkpoints-new/alignment"),
        help="Root with alignment folders containing part_aligned.obj.",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Optional single case in alignment naming (e.g. ANSYS_UNIGE_09).",
    )
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=["aneux_"],
        help="Prefix stripped from zero-root folder names.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=2,
        help="Wave discretization step (matches Registration default: 2).",
    )
    parser.add_argument(
        "--add-com-seed",
        action="store_true",
        help="Use two seeds: ostium centroid and mesh center-of-mass.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing diff_centreline_checkpoint.pkl.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without writing files.",
    )
    return parser.parse_args()


def normalize_case_name(case_name: str, prefixes: Iterable[str]) -> str:
    for prefix in prefixes:
        if prefix and case_name.startswith(prefix):
            return case_name[len(prefix) :]
    return case_name


def unique_preserve_order(values: Iterable[int]) -> list[int]:
    seen = set()
    out: list[int] = []
    for v in values:
        iv = int(v)
        if iv not in seen:
            seen.add(iv)
            out.append(iv)
    return out


def nearest_vertex_index(point: np.ndarray, verts: np.ndarray) -> int:
    try:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(verts).query(point.reshape(1, 3), k=1)
        return int(np.asarray(idx).reshape(-1)[0])
    except Exception:
        dist = np.linalg.norm(verts - point.reshape(1, 3), axis=1)
        return int(np.argmin(dist))


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def cast_wave_loops(
    verts: np.ndarray, edges_unique: np.ndarray, seed_indices: list[int], step_size: int = 2
) -> list[list[np.ndarray]]:
    """Reproduce the loop structure used in Registration._cast_waves.

    Returns:
    - wave_loops: list over seeds (waves), each item is list of vertex-index loops.
    """
    if len(seed_indices) < 1:
        raise ValueError("seed_indices must contain at least one index.")
    if step_size < 1:
        raise ValueError("step_size must be >= 1.")

    n_verts = int(verts.shape[0])
    graph = ig.Graph(n=n_verts, edges=edges_unique.tolist(), directed=False)
    components = graph.connected_components()

    # Keep only component(s) that contain at least one seed.
    # For aneurysm meshes there is usually one connected component.
    wave_loops: list[list[np.ndarray]] = []
    for comp in components:
        comp_arr = np.asarray(comp, dtype=np.int64)
        in_comp = np.isin(np.asarray(seed_indices), comp_arr)
        if not np.any(in_comp):
            continue

        subgraph = graph.subgraph(comp_arr.tolist())
        comp_index = {v: i for i, v in enumerate(comp_arr.tolist())}
        sub_seeds = np.asarray([comp_index[s] for s in np.asarray(seed_indices)[in_comp]], dtype=np.int64)

        # distances shape: [n_seeds, n_sub_vertices]
        dist = np.asarray(subgraph.distances(source=sub_seeds, target=None, mode="all"))
        if step_size > 1:
            finite = dist[np.isfinite(dist)]
            max_d = finite.max() if finite.size > 0 else 0
            dist = np.digitize(dist, bins=np.arange(0, max_d, step_size))

        for w in range(dist.shape[0]):
            this_wave = dist[w, :]
            finite = this_wave[np.isfinite(this_wave)]
            if finite.size == 0:
                wave_loops.append([])
                continue

            max_level = int(finite.max())
            loop_list: list[np.ndarray] = []
            for level in range(0, max_level + 1):
                level_mask = this_wave == level
                ix = np.where(level_mask)[0]
                if ix.size == 0:
                    continue
                level_subgraph = subgraph.subgraph(ix.tolist())
                for cc2 in level_subgraph.connected_components():
                    # map back to global vertex indices
                    loop_global = comp_arr[ix[np.asarray(cc2, dtype=np.int64)]]
                    loop_list.append(loop_global.astype(np.int64))
            wave_loops.append(loop_list)

    if len(wave_loops) == 0:
        raise RuntimeError("No wave loops created. Check seed/component mapping.")
    return wave_loops


def create_diff_checkpoint_for_case(
    zero_case_dir: Path, alignment_case_dir: Path, step_size: int = 2, add_com_seed: bool = False
) -> dict:
    centroid_path = zero_case_dir / "07_other" / "centroid_ostium.npy"
    mesh_path = alignment_case_dir / "part_aligned.obj"
    required = [centroid_path, mesh_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    mesh = load_mesh(mesh_path)
    verts = np.asarray(mesh.vertices)
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)

    centroid_ostium = np.asarray(np.load(centroid_path)).reshape(3)
    seed_ostium = nearest_vertex_index(centroid_ostium, verts)
    seeds = [seed_ostium]

    if add_com_seed:
        seed_com = nearest_vertex_index(verts.mean(axis=0), verts)
        seeds.append(seed_com)
    seeds = unique_preserve_order(seeds)

    wave_loops = cast_wave_loops(verts=verts, edges_unique=edges, seed_indices=seeds, step_size=step_size)
    return {
        "diff_cep_registration": seeds,
        "wave_loops": wave_loops,
    }


def collect_case_pairs(zero_root: Path, alignment_root: Path, strip_prefixes: Iterable[str]) -> list[tuple[str, Path, Path]]:
    pairs = []
    for zero_case_dir in sorted([p for p in zero_root.iterdir() if p.is_dir()]):
        case_name = normalize_case_name(zero_case_dir.name, strip_prefixes)
        alignment_case_dir = alignment_root / case_name
        if alignment_case_dir.is_dir():
            pairs.append((case_name, zero_case_dir, alignment_case_dir))
    return pairs


def main() -> int:
    args = parse_args()
    if not args.zero_root.is_dir():
        raise SystemExit(f"Zero root not found: {args.zero_root}")
    if not args.alignment_root.is_dir():
        raise SystemExit(f"Alignment root not found: {args.alignment_root}")

    pairs = collect_case_pairs(args.zero_root, args.alignment_root, args.strip_prefix)
    if args.case is not None:
        pairs = [p for p in pairs if p[0] == args.case]
        if not pairs:
            raise SystemExit(f"Case '{args.case}' not found in mapped zero/alignment pairs.")

    written = 0
    skipped_exists = 0
    failed = 0
    for case_name, zero_case_dir, alignment_case_dir in pairs:
        out_path = alignment_case_dir / "diff_centreline_checkpoint.pkl"
        if out_path.exists() and not args.overwrite:
            skipped_exists += 1
            continue
        try:
            chk = create_diff_checkpoint_for_case(
                zero_case_dir=zero_case_dir,
                alignment_case_dir=alignment_case_dir,
                step_size=args.step_size,
                add_com_seed=args.add_com_seed,
            )
            print(f"{case_name}: {zero_case_dir} -> {out_path}")
            if not args.dry_run:
                with open(out_path, "wb") as f:
                    pickle.dump(chk, f)
            written += 1
        except Exception as exc:
            failed += 1
            print(f"{case_name}: FAILED ({exc})")

    print("\nSummary")
    print(f"Pairs considered: {len(pairs)}")
    print(f"Written: {written}")
    print(f"Skipped (exists): {skipped_exists}")
    print(f"Failed: {failed}")
    print(f"Seeds mode: {'ostium + COM' if args.add_com_seed else 'ostium only'}")
    print(f"Step size: {args.step_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
