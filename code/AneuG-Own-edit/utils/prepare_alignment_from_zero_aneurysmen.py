#!/usr/bin/env python3
"""Build alignment structure from zero-aneurysmen data."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy <source_root>/<case>/aneurysm_aligned.obj to "
            "<target_root>/<case>/part_aligned.obj."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("checkpoints-v2/zero-aneurysmen"),
        help="Directory containing per-case source folders.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("checkpoints-v2/alignment"),
        help="Directory where per-case alignment folders are created.",
    )
    parser.add_argument(
        "--source-filename",
        default="aneurysm_aligned.obj",
        help="Source filename expected inside each case directory.",
    )
    parser.add_argument(
        "--target-filename",
        default="part_aligned.obj",
        help="Output filename written in each alignment case directory.",
    )
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=["aneux_"],
        help=(
            "Prefix to strip from source case directory names when creating "
            "alignment case names. Can be provided multiple times."
        ),
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


def normalize_case_name(case_name: str, prefixes: list[str]) -> str:
    for prefix in prefixes:
        if prefix and case_name.startswith(prefix):
            return case_name[len(prefix) :]
    return case_name


def main() -> int:
    args = parse_args()
    source_root = args.source_root
    target_root = args.target_root

    if not source_root.exists() or not source_root.is_dir():
        raise SystemExit(f"Source root not found or not a directory: {source_root}")

    copied = 0
    skipped_missing = 0
    skipped_exists = 0

    case_dirs = sorted(p for p in source_root.iterdir() if p.is_dir())
    for case_dir in case_dirs:
        source_obj = case_dir / args.source_filename
        if not source_obj.exists():
            skipped_missing += 1
            continue

        case_name = normalize_case_name(case_dir.name, args.strip_prefix)
        target_case_dir = target_root / case_name
        target_obj = target_case_dir / args.target_filename

        if target_obj.exists() and not args.overwrite:
            skipped_exists += 1
            continue

        print(f"{source_obj} -> {target_obj}")
        if not args.dry_run:
            target_case_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_obj, target_obj)
        copied += 1

    print("\nSummary")
    print(f"Cases scanned: {len(case_dirs)}")
    print(f"Copied: {copied}")
    print(f"Skipped (missing source): {skipped_missing}")
    print(f"Skipped (target exists): {skipped_exists}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
