#!/usr/bin/env python
"""Build a train/val/test split for /path/to/aneug-ghds GHD fitting outputs."""
from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser("build_aneug_ghds_split")
    p.add_argument("--data_root", default="/path/to/aneug-ghds/data")
    p.add_argument("--ghd_run", default="vanilla")
    p.add_argument("--ghd_chk_name", default="ghb_fitting_checkpoint.pkl")
    p.add_argument("--opa_name", default="opa_checkpoint.pkl")
    p.add_argument("--csv", default="/path/to/data_split_real.csv",
                   help="Optional uid;split CSV. CSV train rows become train/val pool; CSV test rows stay test.")
    p.add_argument("--out_root", default="checkpoints/aneug_ghds/splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_fraction", type=float, default=0.20)
    p.add_argument("--test_fraction", type=float, default=0.15,
                   help="Only used when --csv is empty or missing.")
    p.add_argument("--path_only", action="store_true")
    return p.parse_args()


def load_csv_split(path: str):
    if not path:
        return None
    csv_path = Path(path)
    if not csv_path.is_file():
        return None
    out = {"train": [], "test": []}
    with csv_path.open(newline="") as f:
        rdr = csv.DictReader(f, delimiter=";")
        for row in rdr:
            uid = row.get("uid", "").strip()
            split = row.get("split", "").strip().lower()
            if uid and split in out:
                out[split].append(uid)
    return out


def write_json(path: Path, values):
    with path.open("w") as f:
        json.dump(list(values), f, indent=2)


def case_candidates(uid: str):
    cands = [uid]
    if uid.startswith("aneux_"):
        cands.append(uid[len("aneux_"):])
    else:
        cands.append("aneux_" + uid)
    for name in list(cands):
        if name.startswith("cmha_"):
            cands.append("cmch_" + name[len("cmha_"):])
        if name.startswith("cmch_"):
            cands.append("cmha_" + name[len("cmch_"):])
    return list(dict.fromkeys(cands))


def usable_case(case: str, ghd_root: Path, align_root: Path, args) -> bool:
    return (
        (ghd_root / case / args.ghd_run / args.ghd_chk_name).is_file()
        and (ghd_root / case / args.opa_name).is_file()
        and (align_root / case / "part_aligned.obj").is_file()
    )


def resolve_uid(uid: str, ghd_root: Path, align_root: Path, args):
    for case in case_candidates(uid):
        if usable_case(case, ghd_root, align_root, args):
            return case
    return None


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    ghd_root = data_root / "ghd_fitting"
    align_root = data_root / "alignment"
    if not ghd_root.is_dir():
        raise FileNotFoundError(ghd_root)
    if not align_root.is_dir():
        raise FileNotFoundError(align_root)

    all_usable = []
    skipped = []
    for case_dir in sorted(p for p in ghd_root.iterdir() if p.is_dir()):
        case = case_dir.name
        if usable_case(case, ghd_root, align_root, args):
            all_usable.append(case)
        else:
            skipped.append(case)

    if len(all_usable) < 3:
        raise RuntimeError(f"Need at least 3 usable cases, found {len(all_usable)}")

    rng = random.Random(args.seed)
    csv_split = load_csv_split(args.csv)
    unresolved = {"train": [], "test": []}
    aliases = {}
    if csv_split is not None:
        train_pool = []
        for uid in csv_split["train"]:
            case = resolve_uid(uid, ghd_root, align_root, args)
            if case is None:
                unresolved["train"].append(uid)
                continue
            aliases[uid] = case
            train_pool.append(case)
        test = []
        for uid in csv_split["test"]:
            case = resolve_uid(uid, ghd_root, align_root, args)
            if case is None:
                unresolved["test"].append(uid)
                continue
            aliases[uid] = case
            test.append(case)
        train_pool = sorted(set(train_pool))
        test = sorted(set(test))
        pool = train_pool[:]
        rng.shuffle(pool)
        n_val = max(1, int(round(len(pool) * args.val_fraction)))
        val = sorted(pool[:n_val])
        train = sorted(pool[n_val:])
    else:
        shuffled = all_usable[:]
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_test = max(1, int(round(n_total * args.test_fraction)))
        n_val = max(1, int(round(n_total * args.val_fraction)))
        n_train = n_total - n_val - n_test
        if n_train <= 0:
            raise RuntimeError("Split fractions leave no training cases")
        test = sorted(shuffled[:n_test])
        val = sorted(shuffled[n_test:n_test + n_val])
        train = sorted(shuffled[n_test + n_val:])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "realcsv" if csv_split is not None else "random"
    out_dir = Path(args.out_root) / f"aneug_ghds_{tag}_opa20_seed{args.seed}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    write_json(out_dir / "cases_train.json", train)
    write_json(out_dir / "cases_val.json", val)
    write_json(out_dir / "cases_test.json", test)
    with (out_dir / "cases_all.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "split"])
        for split, names in (("train", train), ("val", val), ("test", test)):
            for name in names:
                writer.writerow([name, split])

    summary = {
        "data_root": str(data_root),
        "ghd_root": str(ghd_root),
        "alignment_root": str(align_root),
        "ghd_run": args.ghd_run,
        "ghd_chk_name": args.ghd_chk_name,
        "opa_name": args.opa_name,
        "csv": args.csv if csv_split is not None else None,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "num_usable_total": len(all_usable),
        "num_train": len(train),
        "num_val": len(val),
        "num_test": len(test),
        "num_skipped": len(skipped),
        "skipped_examples": skipped[:20],
        "aliases_examples": dict(list(aliases.items())[:20]),
        "unresolved_counts": {k: len(v) for k, v in unresolved.items()},
        "unresolved_examples": {k: v[:20] for k, v in unresolved.items()},
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(out_dir)
    if not args.path_only:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
