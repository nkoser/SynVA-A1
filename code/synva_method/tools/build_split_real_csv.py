#!/usr/bin/env python3
"""
Build train/val/test split files from /path/to/data_split_real.csv.

- Test set = csv `test` rows.
- Train+val pool = csv `train` rows.
  - Random 80/20 split (seed=42 by default), -> cases_train.json / cases_val.json.
- Cases without a fitted GHD checkpoint (epoch >= 3999) are silently dropped.
- Cases without aligned data on disk are silently dropped.

Outputs:
  checkpoints/vessel_aware_cvae/splits_real_csv_<timestamp>/
    cases_train.json   cases_val.json   cases_test.json   cases_all.json   summary.json
"""
import argparse, csv, json, os, random
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/path/to/data_split_real.csv")
    p.add_argument("--ghd_root", default="checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999")
    p.add_argument("--ghd_run", default="prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3")
    p.add_argument("--aligned_root", default="/path/to/ghd_prepared_meshes_3_aneurysm_1op_new")
    p.add_argument("--min_epoch", type=int, default=3999,
                   help="Require fitting_preview_epoch_<min_epoch>.png to exist.")
    p.add_argument("--val_frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_root", default="checkpoints/vessel_aware_cvae")
    p.add_argument("--tag", default="real_csv")
    p.add_argument(
        "--alias_cmha_to_cmch",
        action="store_true",
        help="Map CSV ids with prefix cmha_ to existing on-disk cmch_ case ids.",
    )
    return p.parse_args()


def load_csv(path):
    cases = {"train": [], "test": []}
    with open(path, newline="") as f:
        rdr = csv.DictReader(f, delimiter=";")
        for row in rdr:
            uid = row["uid"].strip()
            split = row["split"].strip().lower()
            if split in cases:
                cases[split].append(uid)
    return cases


def case_is_fit(uid, ghd_root, ghd_run, min_epoch):
    case_dir = (ROOT / ghd_root / uid / ghd_run)
    if not case_dir.exists():
        # try resolving symlinks (entry might be a symlink to old root)
        link = ROOT / ghd_root / uid
        if not link.exists():
            return False
        case_dir = (link.resolve() / ghd_run)
        if not case_dir.exists():
            return False
    preview = case_dir / f"fitting_preview_epoch_{min_epoch:06d}.png"
    return preview.exists()


def case_has_aligned(uid, aligned_root):
    return (Path(aligned_root) / uid / "part_aligned.obj").exists()


def resolve_case_uid(uid, args):
    if not args.alias_cmha_to_cmch or not uid.startswith("cmha_"):
        return uid

    candidate = "cmch_" + uid[len("cmha_"):]
    has_aligned = case_has_aligned(candidate, args.aligned_root)
    has_fit_dir = (ROOT / args.ghd_root / candidate).exists()
    if has_aligned or has_fit_dir:
        return candidate
    return uid


def main():
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / args.out_root / f"splits_{args.tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_cases = load_csv(args.csv)
    raw_train = csv_cases["train"]
    raw_test = csv_cases["test"]

    skipped = {"no_fit": [], "no_aligned": []}
    aliases = {}

    def _filter(cases):
        kept = []
        for uid in cases:
            resolved_uid = resolve_case_uid(uid, args)
            if resolved_uid != uid:
                aliases[uid] = resolved_uid
            if not case_has_aligned(resolved_uid, args.aligned_root):
                skipped["no_aligned"].append(uid)
                continue
            if not case_is_fit(resolved_uid, args.ghd_root, args.ghd_run, args.min_epoch):
                skipped["no_fit"].append(uid)
                continue
            kept.append(resolved_uid)
        return sorted(kept)

    train_pool = _filter(raw_train)
    test_kept = _filter(raw_test)

    rng = random.Random(args.seed)
    pool = list(train_pool)
    rng.shuffle(pool)
    n_val = int(round(len(pool) * args.val_frac))
    val = sorted(pool[:n_val])
    train = sorted(pool[n_val:])
    all_train_val = sorted(set(train) | set(val))

    def _dump(name, data):
        (out_dir / name).write_text(json.dumps(data, indent=2))

    _dump("cases_train.json", train)
    _dump("cases_val.json", val)
    _dump("cases_test.json", test_kept)
    _dump("cases_all.json", all_train_val)

    summary = {
        "csv": args.csv,
        "ghd_root": args.ghd_root,
        "ghd_run": args.ghd_run,
        "aligned_root": args.aligned_root,
        "min_epoch": args.min_epoch,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "alias_cmha_to_cmch": args.alias_cmha_to_cmch,
        "aliases": aliases,
        "raw_train": len(raw_train),
        "raw_test": len(raw_test),
        "num_train": len(train),
        "num_val": len(val),
        "num_test": len(test_kept),
        "skipped_counts": {k: len(v) for k, v in skipped.items()},
        "skipped": skipped,
    }
    _dump("summary.json", summary)

    print(f"Wrote splits to: {out_dir}")
    print(f"  train: {len(train)}   val: {len(val)}   test: {len(test_kept)}")
    print(f"  skipped no_fit: {len(skipped['no_fit'])}, no_aligned: {len(skipped['no_aligned'])}")


if __name__ == "__main__":
    main()
