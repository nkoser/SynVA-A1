#!/usr/bin/env python3
"""Create a complete alignment bundle from zero-aneurysmen + canonical source.

Per generated case under alignment-root:
- part_aligned.obj
- opa_checkpoint.pkl
- diff_centreline_checkpoint.pkl (optional)

Optionally runs export_opening_planes for all target cases.
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import trimesh

try:
    from utils.create_opa_checkpoint_from_ostium import create_opa_checkpoint_for_case
    from utils.create_diff_centreline_checkpoint_from_ostium import create_diff_checkpoint_for_case
except ModuleNotFoundError:
    from create_opa_checkpoint_from_ostium import create_opa_checkpoint_for_case
    from create_diff_centreline_checkpoint_from_ostium import create_diff_checkpoint_for_case


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create full alignment bundle with OPA/diff checkpoints.")
    p.add_argument("--zero-root", type=Path, default=Path("checkpoints-new/zero-aneurysmen"))
    p.add_argument(
        "--canonical-src",
        type=Path,
        default=Path("checkpoints-new/canonical_model"),
        help="Canonical source dir containing part_aligned.obj + 04_subpointclouds + 07_other.",
    )
    p.add_argument("--alignment-root", type=Path, default=Path("checkpoints-new/alignment"))
    p.add_argument("--canonical-name", type=str, default="canonical_model")
    p.add_argument("--case", type=str, default=None, help="Optional single case in alignment naming.")
    p.add_argument("--strip-prefix", action="append", default=["aneux_"])
    p.add_argument("--source-mesh-name", type=str, default="aneurysm_aligned.obj")
    p.add_argument("--target-mesh-name", type=str, default="part_aligned.obj")
    p.add_argument("--ostium-label", type=int, default=2)
    p.add_argument("--smooth-iters", type=int, default=2)
    p.add_argument("--smooth-alpha", type=float, default=0.25)
    p.add_argument("--target-opening-triangles", type=int, default=20)
    p.add_argument("--flip-inside-normal", action="store_true", default=False)
    p.add_argument("--keep-inside-normal", action="store_true")
    p.add_argument("--step-size", type=int, default=2)
    p.add_argument("--add-com-seed", action="store_true")
    p.add_argument(
        "--center-mode",
        type=str,
        choices=["none", "ostium", "opening"],
        default="ostium",
        help="How to recenter each case after checkpoint creation.",
    )
    p.add_argument("--include-diff", type=int, default=1, help="1: write diff_centreline_checkpoint.pkl")
    p.add_argument(
        "--export-opening-planes-all",
        type=int,
        default=0,
        help="1: run utils/inspect/export_opening_planes.py for all target cases.",
    )
    p.add_argument("--canonical-index", type=int, default=0, help="Passed to export_opening_planes.")
    p.add_argument("--normal-scale", type=float, default=0.01, help="Passed to export_opening_planes.")
    p.add_argument(
        "--overwrite",
        type=int,
        default=1,
        help="1: overwrite existing outputs (default: 1). Set to 0 to keep existing files.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def normalize_case_name(case_name: str, prefixes: Iterable[str]) -> str:
    for prefix in prefixes:
        if prefix and case_name.startswith(prefix):
            return case_name[len(prefix) :]
    return case_name


def _load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def _save_mesh(mesh: trimesh.Trimesh, mesh_path: Path) -> None:
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(mesh_path)


def _translate_case_mesh_and_opa(case_dir: Path, t: np.ndarray, mesh_name: str) -> None:
    t = np.asarray(t, dtype=np.float64).reshape(1, 3)
    mesh_path = case_dir / mesh_name
    opa_path = case_dir / "opa_checkpoint.pkl"

    mesh = _load_mesh(mesh_path)
    mesh.vertices = np.asarray(mesh.vertices) + t
    _save_mesh(mesh, mesh_path)

    with open(opa_path, "rb") as f:
        chk = pickle.load(f)

    for key in ["op_v_coords", "op_rec_v"]:
        if key in chk:
            chk[key] = [np.asarray(x, dtype=np.float64) + t for x in chk[key]]

    with open(opa_path, "wb") as f:
        pickle.dump(chk, f)


def _centering_translation(
    center_mode: str,
    zero_case_dir: Path,
    case_dir: Path,
) -> np.ndarray:
    if center_mode == "none":
        return np.zeros((1, 3), dtype=np.float64)
    if center_mode == "ostium":
        c = np.asarray(np.load(zero_case_dir / "07_other" / "centroid_ostium.npy"), dtype=np.float64).reshape(1, 3)
        return -c
    if center_mode == "opening":
        with open(case_dir / "opa_checkpoint.pkl", "rb") as f:
            chk = pickle.load(f)
        op0 = np.asarray(chk["op_rec_v"][0], dtype=np.float64)
        return -op0.mean(axis=0, keepdims=True)
    raise ValueError(f"Unsupported center_mode: {center_mode}")


def _copy_mesh(src: Path, dst: Path, overwrite: bool) -> bool:
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _build_case(
    case_name: str,
    zero_case_dir: Path,
    alignment_case_dir: Path,
    args: argparse.Namespace,
    is_canonical: bool = False,
) -> tuple[bool, str]:
    src_mesh = zero_case_dir / ("part_aligned.obj" if is_canonical else args.source_mesh_name)
    dst_mesh = alignment_case_dir / args.target_mesh_name
    if not src_mesh.exists():
        return False, f"missing source mesh: {src_mesh}"

    if not args.dry_run:
        _copy_mesh(src_mesh, dst_mesh, overwrite=args.overwrite)

    flip_inside_normal = args.flip_inside_normal and not args.keep_inside_normal
    if not args.dry_run:
        chk_opa = create_opa_checkpoint_for_case(
            zero_case_dir=zero_case_dir,
            alignment_case_dir=alignment_case_dir,
            ostium_label=args.ostium_label,
            flip_inside_normal=flip_inside_normal,
            smooth_iters=args.smooth_iters,
            smooth_alpha=args.smooth_alpha,
            target_opening_triangles=args.target_opening_triangles,
        )
        with open(alignment_case_dir / "opa_checkpoint.pkl", "wb") as f:
            pickle.dump(chk_opa, f)

        if int(args.include_diff) == 1:
            chk_diff = create_diff_checkpoint_for_case(
                zero_case_dir=zero_case_dir,
                alignment_case_dir=alignment_case_dir,
                step_size=args.step_size,
                add_com_seed=args.add_com_seed,
            )
            with open(alignment_case_dir / "diff_centreline_checkpoint.pkl", "wb") as f:
                pickle.dump(chk_diff, f)

        t = _centering_translation(args.center_mode, zero_case_dir, alignment_case_dir)
        if np.linalg.norm(t) > 0:
            _translate_case_mesh_and_opa(alignment_case_dir, t, args.target_mesh_name)

    return True, "ok"


def _iter_zero_cases(zero_root: Path, strip_prefixes: Iterable[str]):
    for zero_case_dir in sorted([p for p in zero_root.iterdir() if p.is_dir()]):
        case_name = normalize_case_name(zero_case_dir.name, strip_prefixes)
        yield case_name, zero_case_dir


def _run_export_opening_planes_for_all(
    alignment_root: Path,
    canonical_name: str,
    canonical_index: int,
    normal_scale: float,
) -> tuple[int, int]:
    script = Path(__file__).resolve().parent / "inspect" / "export_opening_planes.py"
    total = 0
    failed = 0
    for case_dir in sorted([p for p in alignment_root.iterdir() if p.is_dir()]):
        case_name = case_dir.name
        if case_name == canonical_name:
            continue
        if not (case_dir / "opa_checkpoint.pkl").exists():
            continue
        cmd = [
            sys.executable,
            str(script),
            "--root",
            str(alignment_root),
            "--canonical",
            canonical_name,
            "--target",
            case_name,
            "--canonical-index",
            str(canonical_index),
            "--normal-scale",
            str(normal_scale),
        ]
        total += 1
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            failed += 1
    return total, failed


def main() -> int:
    args = parse_args()
    args.zero_root = args.zero_root.resolve()
    args.canonical_src = args.canonical_src.resolve()
    args.alignment_root = args.alignment_root.resolve()

    if not args.zero_root.is_dir():
        raise SystemExit(f"Zero root not found: {args.zero_root}")
    if not args.canonical_src.is_dir():
        raise SystemExit(f"Canonical source not found: {args.canonical_src}")
    args.alignment_root.mkdir(parents=True, exist_ok=True)

    built = 0
    failed = 0
    skipped = 0

    # Canonical case is always required for downstream comparisons/fitting.
    canonical_case_dir = args.alignment_root / args.canonical_name
    ok, msg = _build_case(
        case_name=args.canonical_name,
        zero_case_dir=args.canonical_src,
        alignment_case_dir=canonical_case_dir,
        args=args,
        is_canonical=True,
    )
    print(f"{args.canonical_name}: {msg}")
    if ok:
        built += 1
    else:
        failed += 1

    # Zero cases
    for case_name, zero_case_dir in _iter_zero_cases(args.zero_root, args.strip_prefix):
        if args.case is not None and case_name != args.case:
            continue
        if case_name == args.canonical_name:
            skipped += 1
            continue
        alignment_case_dir = args.alignment_root / case_name
        try:
            if not args.dry_run:
                alignment_case_dir.mkdir(parents=True, exist_ok=True)
            ok, msg = _build_case(
                case_name=case_name,
                zero_case_dir=zero_case_dir,
                alignment_case_dir=alignment_case_dir,
                args=args,
                is_canonical=False,
            )
            print(f"{case_name}: {msg}")
            if ok:
                built += 1
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            print(f"{case_name}: FAILED ({exc})")

    export_total = 0
    export_failed = 0
    if int(args.export_opening_planes_all) == 1 and not args.dry_run:
        export_total, export_failed = _run_export_opening_planes_for_all(
            alignment_root=args.alignment_root,
            canonical_name=args.canonical_name,
            canonical_index=int(args.canonical_index),
            normal_scale=float(args.normal_scale),
        )

    print("\nSummary")
    print(f"Built: {built}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Center mode: {args.center_mode}")
    if int(args.export_opening_planes_all) == 1:
        print(f"export_opening_planes runs: {export_total}, failed: {export_failed}")
    return 0 if failed == 0 and export_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
