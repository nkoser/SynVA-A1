#!/usr/bin/env python3
"""Run inference pipeline across all cases in inference/cases/<split>/<case>."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CASES_ROOT = SCRIPT_DIR / "cases"
DEFAULT_PIPELINE = SCRIPT_DIR / "run_inference_pipeline.py"
DEFAULT_SUMMARY = SCRIPT_DIR / "shared" / "last_all_cases_run_summary.json"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run inference pipeline for all cases in selected splits.")
    parser.add_argument("step", nargs="?", default="all", choices=["all", "step1", "step2", "step3"])
    parser.add_argument("--cases-root", type=Path, default=DEFAULT_CASES_ROOT)
    parser.add_argument("--pipeline-script", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--splits", nargs="+", default=["train", "test"], help="Splits to run, e.g. train test.")
    parser.add_argument("--max-cases", type=int, default=0, help="Optional limit for quick tests (0 = no limit).")
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of cases to run in parallel. Default: number of CPU cores.",
    )
    parser.add_argument("--continue-on-error", action="store_true", help="Continue with next case if one case fails.")
    parser.add_argument("--overwrite", action="store_true", help="Forward --overwrite to case pipeline.")
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_known_args()


def discover_cases(cases_root: Path, splits: list[str]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for split in splits:
        split_dir = cases_root / split
        if not split_dir.exists():
            continue
        names = sorted([path.name for path in split_dir.iterdir() if path.is_dir()])
        result.extend((split, name) for name in names)
    return result


def run_case(
    pipeline_script: Path,
    cases_root: Path,
    split: str,
    case_name: str,
    step: str,
    overwrite: bool,
    passthrough: list[str],
) -> tuple[int, str, str, str]:
    cmd = [
        sys.executable,
        str(pipeline_script),
        step,
        "--cases-root",
        str(cases_root),
        "--case-split",
        split,
        "--case-name",
        case_name,
        "--ring-points",
        "20",
        "--resample-aneurysm-to-vessel-resolution",
        "--stitch",
        "--stitch-method", 
        "bridge",
        "--smooth-ostium-transition"

    ]
    if overwrite:
        cmd.append("--overwrite")
    cmd.extend(passthrough)

    print(f"\n=== [{split}] {case_name} ===")
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return completed.returncode, " ".join(cmd), completed.stdout or "", completed.stderr or ""


def main() -> int:
    args, passthrough = parse_args()
    args.cases_root = args.cases_root.expanduser().resolve()
    args.pipeline_script = args.pipeline_script.expanduser().resolve()
    args.summary_path = args.summary_path.expanduser().resolve()
    args.jobs = max(1, int(args.jobs))

    cases = discover_cases(args.cases_root, args.splits)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    if not cases:
        raise FileNotFoundError(f"No case folders found under {args.cases_root} for splits {args.splits}.")

    summary: dict[str, object] = {
        "step": args.step,
        "cases_root": str(args.cases_root),
        "pipeline_script": str(args.pipeline_script),
        "splits": args.splits,
        "overwrite": bool(args.overwrite),
        "jobs": int(args.jobs),
        "passthrough_args": passthrough,
        "total_cases": len(cases),
        "ok": [],
        "failed": [],
        "aborted_after_failure": False,
    }

    if args.jobs == 1:
        for split, case_name in cases:
            code, cmd, stdout_text, stderr_text = run_case(
                pipeline_script=args.pipeline_script,
                cases_root=args.cases_root,
                split=split,
                case_name=case_name,
                step=args.step,
                overwrite=bool(args.overwrite),
                passthrough=passthrough,
            )
            if stdout_text:
                print(stdout_text, end="" if stdout_text.endswith("\n") else "\n")
            if stderr_text:
                print(stderr_text, file=sys.stderr, end="" if stderr_text.endswith("\n") else "\n")
            item = {"split": split, "case_name": case_name, "command": cmd, "exit_code": int(code)}
            if code == 0:
                summary["ok"].append(item)
            else:
                summary["failed"].append(item)
                if not args.continue_on_error:
                    summary["aborted_after_failure"] = True
                    break
    else:
        print(f"Running {len(cases)} cases with --jobs={args.jobs}")
        indexed_cases = list(enumerate(cases))
        submitted_idx = 0
        stop_submitting = False
        pending: dict[concurrent.futures.Future[tuple[int, str, str, str]], tuple[int, str, str]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            while submitted_idx < len(indexed_cases) and len(pending) < args.jobs:
                idx, (split, case_name) = indexed_cases[submitted_idx]
                future = executor.submit(
                    run_case,
                    args.pipeline_script,
                    args.cases_root,
                    split,
                    case_name,
                    args.step,
                    bool(args.overwrite),
                    passthrough,
                )
                pending[future] = (idx, split, case_name)
                submitted_idx += 1

            while pending:
                done, _ = concurrent.futures.wait(pending.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    idx, split, case_name = pending.pop(future)
                    code, cmd, stdout_text, stderr_text = future.result()
                    if stdout_text:
                        print(stdout_text, end="" if stdout_text.endswith("\n") else "\n")
                    if stderr_text:
                        print(stderr_text, file=sys.stderr, end="" if stderr_text.endswith("\n") else "\n")
                    item = {"index": int(idx), "split": split, "case_name": case_name, "command": cmd, "exit_code": int(code)}
                    if code == 0:
                        summary["ok"].append(item)
                    else:
                        summary["failed"].append(item)
                        if not args.continue_on_error:
                            stop_submitting = True
                            summary["aborted_after_failure"] = True
                            for pending_future in list(pending):
                                pending_future.cancel()
                            pending = {k: v for k, v in pending.items() if not k.cancelled()}

                while not stop_submitting and submitted_idx < len(indexed_cases) and len(pending) < args.jobs:
                    idx, (split, case_name) = indexed_cases[submitted_idx]
                    future = executor.submit(
                        run_case,
                        args.pipeline_script,
                        args.cases_root,
                        split,
                        case_name,
                        args.step,
                        bool(args.overwrite),
                        passthrough,
                    )
                    pending[future] = (idx, split, case_name)
                    submitted_idx += 1

                if stop_submitting and not pending:
                    break

    summary["ok"] = sorted(summary["ok"], key=lambda x: (x.get("index", 0), x["split"], x["case_name"]))
    summary["failed"] = sorted(summary["failed"], key=lambda x: (x.get("index", 0), x["split"], x["case_name"]))
    for item in summary["ok"]:
        item.pop("index", None)
    for item in summary["failed"]:
        item.pop("index", None)
    summary["ok_count"] = len(summary["ok"])
    summary["failed_count"] = len(summary["failed"])
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Batch summary ===")
    print(json.dumps({"ok_count": summary["ok_count"], "failed_count": summary["failed_count"], "summary_path": str(args.summary_path)}, indent=2))
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
