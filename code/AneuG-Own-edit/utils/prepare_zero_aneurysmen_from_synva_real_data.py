#!/usr/bin/env python3
"""Build zero-aneurysmen structure from synva_real_data cases.

Per source case:
- 05_submeshes/aneurysm_submesh.obj -> aneurysm_aligned.obj
- 04_subpointclouds/*.ply -> 04_subpointclouds/*.ply
- 07_other/*.npy -> 07_other/*.npy
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy synva_real_data case assets into zero-aneurysmen layout "
            "(aneurysm_aligned.obj + 04_subpointclouds + 07_other)."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("checkpoints-v2/synva_real_data"),
        help="Directory containing source case folders in synva_real_data layout.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("checkpoints-v2/zero-aneurysmen"),
        help="Directory where zero-aneurysmen case folders are created.",
    )
    parser.add_argument(
        "--source-mesh-relpath",
        type=Path,
        default=Path("05_submeshes/aneurysm_submesh.obj"),
        help="Relative path (inside each source case) to aneurysm mesh.",
    )
    parser.add_argument(
        "--target-mesh-name",
        default="aneurysm_aligned.obj",
        help="Mesh filename written in each target case folder.",
    )
    parser.add_argument(
        "--subpointcloud-rel-dir",
        type=Path,
        default=Path("04_subpointclouds"),
        help="Relative source dir containing subpointcloud files.",
    )
    parser.add_argument(
        "--subpointcloud-pattern",
        default="*.ply",
        help="Glob pattern for files copied from subpointcloud dir.",
    )
    parser.add_argument(
        "--other-rel-dir",
        type=Path,
        default=Path("07_other"),
        help="Relative source dir containing additional per-case metadata files.",
    )
    parser.add_argument(
        "--other-pattern",
        action="append",
        default=["*.npy"],
        help="Glob pattern for files copied from other dir. Can be passed multiple times.",
    )
    parser.add_argument(
        "--require-other-file",
        action="append",
        default=["centroid_ostium.npy", "normal_vector.npy"],
        help="Filename that must exist in other dir. Can be passed multiple times.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="Optional case name filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing target files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without copying files.",
    )
    return parser.parse_args()


def _collect_unique_files(root: Path, patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    seen = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file() and path not in seen:
                out.append(path)
                seen.add(path)
    return out


def _copy_case_files(ops: list[tuple[Path, Path]], overwrite: bool, dry_run: bool) -> tuple[int, int]:
    copied = 0
    skipped_exists = 0
    for src, dst in ops:
        if dst.exists() and not overwrite:
            skipped_exists += 1
            continue
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied += 1
    return copied, skipped_exists


def main() -> int:
    args = parse_args()
    source_root = args.source_root
    target_root = args.target_root

    if not source_root.exists() or not source_root.is_dir():
        raise SystemExit(f"Source root not found or not a directory: {source_root}")

    selected_cases = set(args.case) if args.case else None

    case_dirs = sorted(p for p in source_root.iterdir() if p.is_dir())
    if selected_cases is not None:
        case_dirs = [p for p in case_dirs if p.name in selected_cases]

    scanned = 0
    processed = 0
    skipped_missing = 0
    file_copied = 0
    file_skipped_exists = 0

    for case_dir in case_dirs:
        scanned += 1
        src_mesh = case_dir / args.source_mesh_relpath
        src_sub_dir = case_dir / args.subpointcloud_rel_dir
        src_other_dir = case_dir / args.other_rel_dir

        missing_reasons: list[str] = []
        if not src_mesh.is_file():
            missing_reasons.append(f"missing mesh: {src_mesh}")
        if not src_sub_dir.is_dir():
            missing_reasons.append(f"missing subpointcloud dir: {src_sub_dir}")
        if not src_other_dir.is_dir():
            missing_reasons.append(f"missing other dir: {src_other_dir}")

        if src_other_dir.is_dir():
            for required_name in args.require_other_file:
                required_path = src_other_dir / required_name
                if not required_path.is_file():
                    missing_reasons.append(f"missing required other file: {required_path}")

        sub_files = _collect_unique_files(src_sub_dir, [args.subpointcloud_pattern]) if src_sub_dir.is_dir() else []
        if not sub_files:
            missing_reasons.append(
                f"no subpointcloud files matched '{args.subpointcloud_pattern}' in {src_sub_dir}"
            )
        other_files = _collect_unique_files(src_other_dir, args.other_pattern) if src_other_dir.is_dir() else []
        if not other_files:
            missing_reasons.append(f"no other files matched {args.other_pattern} in {src_other_dir}")

        if missing_reasons:
            skipped_missing += 1
            print(f"SKIP {case_dir.name}: {'; '.join(missing_reasons)}")
            continue

        target_case = target_root / case_dir.name
        ops: list[tuple[Path, Path]] = []
        ops.append((src_mesh, target_case / args.target_mesh_name))
        for path in sub_files:
            ops.append((path, target_case / "04_subpointclouds" / path.name))
        for path in other_files:
            ops.append((path, target_case / "07_other" / path.name))

        copied, skipped_exists = _copy_case_files(ops, overwrite=args.overwrite, dry_run=args.dry_run)
        processed += 1
        file_copied += copied
        file_skipped_exists += skipped_exists
        print(
            f"{case_dir.name}: files_total={len(ops)}, copied={copied}, "
            f"skipped_exists={skipped_exists}"
        )

    print("\nSummary")
    print(f"Cases scanned: {scanned}")
    print(f"Cases processed: {processed}")
    print(f"Cases skipped (missing data): {skipped_missing}")
    print(f"Files copied: {file_copied}")
    print(f"Files skipped (exists): {file_skipped_exists}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
