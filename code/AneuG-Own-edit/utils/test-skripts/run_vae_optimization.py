#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path(sys.executable)


@dataclass(frozen=True)
class RunConfig:
    name: str
    overrides: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reversible VAE loss sweeps and collect test/inference outputs.")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "vae_optimization_results")
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap for debugging the sweep.")
    parser.add_argument("--skip-existing", type=int, default=1)
    parser.add_argument("--test-cases", type=str, nargs="*", default=None)
    parser.add_argument("--num-visual-cases", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--train-subset-limit", type=int, default=8)
    parser.add_argument("--run-inference", type=int, default=1)
    parser.add_argument("--render-inspect", type=int, default=1)
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated physical GPU ids for training workers.")
    parser.add_argument("--parallel-train", type=int, default=2, help="Number of training runs to execute concurrently.")
    return parser.parse_args()


def base_args(args: argparse.Namespace, meta: str) -> list[str]:
    return [
        "first_stage_ostium_conditional.py",
        "--stage1-objective", "mesh_vae",
        "--ghd-chk-root", "checkpoint-v2/ghd_fitting_split_real",
        "--condition-root", "checkpoint-v2/ghd_fitting_split_real",
        "--alignment-root", "alignment_vc",
        "--canonical-root", "alignment_vc/canonical_model",
        "--prepare-condition-from-ghd", "1",
        "--force-prepare-condition-from-ghd", "0",
        "--split-file", "checkpoint-v2/dataset_splits/data_split_real.json",
        "--force-resplit", "0",
        "--epochs", str(args.epochs),
        "--batch-size", "4",
        "--train-subset-limit", str(args.train_subset_limit),
        "--hidden-dim", "2048",
        "--latent-dim", "512",
        "--cond-embed-dim", "256",
        "--norm-type", "layer",
        "--lr", "1e-4",
        "--posterior-noise-scale", "0.1",
        "--max-grad-norm", "1.0",
        "--target-clamp", "8.0",
        "--scale-clamp", "6.0",
        "--w-vert", "250",
        "--w-target", "1.0",
        "--w-scale", "1.0",
        "--w-kl-max", "0.0002",
        "--kl-warmup-epochs", "300",
        "--kl-free-bits", "0.01",
        "--use-reg", "0",
        "--w-reg", "0",
        "--w-rigid", "0",
        "--w-trumpet", "0",
        "--w-smooth", "0",
        "--w-normal", "0",
        "--w-consistency", "0",
        "--w-spectral", "0",
        "--w-cond", "0",
        "--num-workers", "0",
        "--log-wandb", "0",
        "--log-every", str(args.log_every),
        "--val-every", str(args.val_every),
        "--run-checkpoint-inference", "0",
        "--meta", meta,
    ]


def sweep_configs() -> list[RunConfig]:
    return [
        RunConfig("baseline_user", {}),
        RunConfig("kl_zero_deterministic", {"--w-kl-max": "0", "--kl-free-bits": "0", "--posterior-noise-scale": "0"}),
        RunConfig("kl_low_freebits0", {"--w-kl-max": "0.00005", "--kl-free-bits": "0"}),
        RunConfig("kl_mid_freebits001", {"--w-kl-max": "0.0001", "--kl-free-bits": "0.001"}),
        RunConfig("kl_high", {"--w-kl-max": "0.0005", "--kl-free-bits": "0.01"}),
        RunConfig("cond_10", {"--w-cond": "10"}),
        RunConfig("cond_50", {"--w-cond": "50"}),
        RunConfig("cond_100", {"--w-cond": "100"}),
        RunConfig("vert_100_target025", {"--w-vert": "100", "--w-target": "0.25"}),
        RunConfig("vert_500_target025", {"--w-vert": "500", "--w-target": "0.25"}),
        RunConfig("scale_3", {"--w-scale": "3.0"}),
        RunConfig("small_1024_256_cond50", {"--hidden-dim": "1024", "--latent-dim": "256", "--cond-embed-dim": "128", "--w-cond": "50"}),
        RunConfig("small_512_128_cond50", {"--hidden-dim": "512", "--latent-dim": "128", "--cond-embed-dim": "96", "--w-cond": "50"}),
        RunConfig("small_512_128_kl_low", {"--hidden-dim": "512", "--latent-dim": "128", "--cond-embed-dim": "96", "--w-kl-max": "0.00005", "--kl-free-bits": "0.001"}),
    ]


def apply_overrides(command: list[str], overrides: dict[str, Any]) -> list[str]:
    command = list(command)
    for key, value in overrides.items():
        if key in command:
            index = command.index(key)
            command[index + 1] = str(value)
        else:
            command.extend([key, str(value)])
    return command


def run_command(command: list[str], cwd: Path, log_path: Path, env_overrides: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# started: {started}\n")
        log.write("$ " + " ".join(command) + "\n\n")
        if env_overrides:
            log.write("# env_overrides: " + json.dumps(env_overrides, sort_keys=True) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def start_logged_process(
    command: list[str],
    cwd: Path,
    log_path: Path,
    env_overrides: dict[str, str] | None = None,
) -> tuple[subprocess.Popen, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    log = log_path.open("w", encoding="utf-8")
    log.write(f"# started: {datetime.now().isoformat(timespec='seconds')}\n")
    log.write("$ " + " ".join(command) + "\n\n")
    if env_overrides:
        log.write("# env_overrides: " + json.dumps(env_overrides, sort_keys=True) + "\n\n")
    log.flush()
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return process, log


def parse_log_metrics(log_path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    last_epoch_log: dict[str, Any] | None = None
    test_log: dict[str, Any] | None = None
    if not log_path.exists():
        return metrics
    for raw in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            data = ast.literal_eval(line)
        except Exception:
            continue
        if isinstance(data, dict) and "test_total_loss" in data:
            test_log = data
        elif isinstance(data, dict) and "total_loss" in data and "epoch" in data:
            last_epoch_log = data
    if last_epoch_log is not None:
        metrics.update({f"train_{k}": v for k, v in last_epoch_log.items() if isinstance(v, (int, float))})
    if test_log is not None:
        metrics.update(test_log)
    return metrics


def load_split_cases(split_file: Path, count: int, explicit_cases: list[str] | None) -> list[str]:
    if explicit_cases:
        return explicit_cases
    split = json.loads(split_file.read_text(encoding="utf-8"))
    return list(split["test"][:count])


def run_inference_for_run(
    args: argparse.Namespace,
    run_dir: Path,
    meta: str,
    cases: list[str],
    checkpoint: Path,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"cases": {}, "reconstruct_target_mse_mean": None, "sample_scale_mean": None}
    target_mses: list[float] = []
    sample_scales: list[float] = []
    for case in cases:
        case_key = case.replace("/", "__")
        case_summary: dict[str, Any] = {}
        for mode in ("reconstruct", "sample"):
            out_dir = run_dir / "inference" / case_key / mode
            command = [
                str(args.python),
                "infer_stage1_ostium_conditional.py",
                "--checkpoint", str(checkpoint),
                "--case", case,
                "--output-dir", str(out_dir),
                "--mode", mode,
                "--num-samples", str(args.num_samples),
                "--seed", str(args.seed),
                "--ghd-chk-root", "checkpoint-v2/ghd_fitting_split_real",
                "--condition-root", "checkpoint-v2/ghd_fitting_split_real",
                "--alignment-root", "alignment_vc",
                "--canonical-root", "alignment_vc/canonical_model",
                "--prepare-condition-from-ghd", "0",
                "--split-file", "checkpoint-v2/dataset_splits/data_split_real.json",
                "--train-subset-limit", str(args.train_subset_limit),
                "--ring-points", "20",
                "--posterior-noise-scale", "0",
            ]
            code = run_command(command, PROJECT_ROOT, run_dir / "logs" / f"infer_{case_key}_{mode}.log")
            case_summary[f"{mode}_returncode"] = code
            meta_path = out_dir / "metadata.json"
            if meta_path.exists():
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                case_summary[mode] = metadata
                if mode == "reconstruct" and isinstance(metadata.get("target_mse"), (int, float)):
                    target_mses.append(float(metadata["target_mse"]))
                if mode == "sample" and isinstance(metadata.get("predicted_scale_mean"), (int, float)):
                    sample_scales.append(float(metadata["predicted_scale_mean"]))
        summary["cases"][case] = case_summary

    if target_mses:
        summary["reconstruct_target_mse_mean"] = sum(target_mses) / len(target_mses)
    if sample_scales:
        summary["sample_scale_mean"] = sum(sample_scales) / len(sample_scales)
    return summary


def render_inspection(
    args: argparse.Namespace,
    run_dir: Path,
    cases: list[str],
    checkpoint_epoch: int,
    model_dir: Path,
) -> dict[str, Any]:
    output_dir = run_dir / "inspect"
    command = [
        str(args.python),
        "utils/inspect/vae_inspect_stage1_ostium_conditional.py",
        "--ghd-chk-root", "checkpoint-v2/ghd_fitting_split_real",
        "--condition-root", "checkpoint-v2/ghd_fitting_split_real",
        "--canonical-root", "alignment_vc/canonical_model",
        "--model-dir", str(model_dir),
        "--epochs", str(checkpoint_epoch),
        "--all-cases", "0",
        "--ring-points", "20",
        "--ostium-source", "opa_checkpoint",
        "--split-file", "checkpoint-v2/dataset_splits/data_split_real.json",
        "--train-subset-limit", str(args.train_subset_limit),
        "--prepare-condition-from-ghd", "0",
        "--output-dir", str(output_dir),
    ]
    summaries = []
    for case in cases:
        case_command = command + ["--case", case]
        code = run_command(case_command, PROJECT_ROOT, run_dir / "logs" / f"inspect_{case.replace('/', '__')}.log")
        summary_path = output_dir / "summary_single_case.json"
        if summary_path.exists():
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            data["returncode"] = code
            summaries.append(data)
            case_summary_path = output_dir / f"summary_{case.replace('/', '__')}.json"
            shutil.copyfile(summary_path, case_summary_path)
    rmse_values = []
    warnings = 0
    for data in summaries:
        for result in data.get("results", []):
            warnings += int(bool(result.get("warning")))
            rmse = result.get("epoch_rmse", {}).get(str(checkpoint_epoch))
            if isinstance(rmse, (int, float)):
                rmse_values.append(float(rmse))
    return {
        "num_cases": len(summaries),
        "warning_count": warnings,
        "rmse_mean": sum(rmse_values) / len(rmse_values) if rmse_values else None,
        "summaries": summaries,
    }


def normalize(values: list[float | None]) -> list[float | None]:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return [None for _ in values]
    lo = min(finite)
    hi = max(finite)
    if abs(hi - lo) < 1e-12:
        return [0.0 if v is not None else None for v in values]
    return [((v - lo) / (hi - lo)) if v is not None and math.isfinite(v) else None for v in values]


def add_scores(rows: list[dict[str, Any]]) -> None:
    keys = ["test_total_loss", "test_vert_loss", "inspect_rmse_mean", "infer_reconstruct_target_mse_mean"]
    normalized = {key: normalize([as_float(row.get(key)) for row in rows]) for key in keys}
    for idx, row in enumerate(rows):
        weighted_sum = 0.0
        weight_sum = 0.0
        for key, weight in [
            ("test_total_loss", 0.35),
            ("test_vert_loss", 0.25),
            ("inspect_rmse_mean", 0.25),
            ("infer_reconstruct_target_mse_mean", 0.15),
        ]:
            val = normalized[key][idx]
            if val is not None:
                weighted_sum += weight * val
                weight_sum += weight
        row["balanced_score"] = weighted_sum / max(1e-12, weight_sum) if weight_sum > 0 else float("inf")


def checkpoint_epoch(checkpoint: Path) -> int | None:
    stem = checkpoint.stem
    prefix = "models_epoch_"
    if stem.startswith(prefix):
        try:
            return int(stem[len(prefix):])
        except ValueError:
            return None
    return None


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def write_comparison(result_dir: Path, rows: list[dict[str, Any]]) -> None:
    add_scores(rows)
    rows.sort(key=lambda item: (float(item.get("balanced_score", float("inf"))), float(item.get("test_total_loss", float("inf")))))
    csv_path = result_dir / "comparison.csv"
    fieldnames = [
        "rank",
        "run",
        "meta",
        "returncode",
        "balanced_score",
        "test_total_loss",
        "test_mse_loss",
        "test_vert_loss",
        "test_scale_loss",
        "test_kl_loss",
        "inspect_rmse_mean",
        "inspect_warning_count",
        "infer_reconstruct_target_mse_mean",
        "checkpoint",
        "config",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
            writer.writerow({name: row.get(name) for name in fieldnames})

    best = rows[0] if rows else None
    lines = [
        "# VAE Optimization Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Runs completed: {len(rows)}",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                "## Best Run",
                "",
                f"- Run: `{best['run']}`",
                f"- Meta: `{best['meta']}`",
                f"- Balanced score: `{best.get('balanced_score')}`",
                f"- Test total loss: `{best.get('test_total_loss')}`",
                f"- Test vertex loss: `{best.get('test_vert_loss')}`",
                f"- Inspect RMSE mean: `{best.get('inspect_rmse_mean')}`",
                f"- Checkpoint: `{best.get('checkpoint')}`",
                "",
            ]
        )
    lines.extend(["## Ranked Runs", ""])
    for row in rows:
        lines.append(
            f"{row['rank']}. `{row['run']}` score={row.get('balanced_score'):.6g} "
            f"test={row.get('test_total_loss')} vert={row.get('test_vert_loss')} "
            f"inspect_rmse={row.get('inspect_rmse_mean')}"
        )
    lines.append("")
    lines.append(f"Full table: `{csv_path}`")
    (result_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_run(
    args: argparse.Namespace,
    result_dir: Path,
    cfg: RunConfig,
    index: int,
    timestamp: str,
) -> dict[str, Any]:
    meta = f"vae_opt_{timestamp}_{index:02d}_{cfg.name}"
    run_dir = result_dir / f"{index:02d}_{cfg.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = PROJECT_ROOT / "checkpoint-v2" / "first_stage_ostium_conditional" / meta
    checkpoint = model_dir / f"models_epoch_{args.epochs}.pth"
    command = [str(args.python)] + apply_overrides(base_args(args, meta), cfg.overrides)
    (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps({"name": cfg.name, "meta": meta, "overrides": cfg.overrides, "command": command}, indent=2),
        encoding="utf-8",
    )
    return {
        "index": index,
        "cfg": cfg,
        "meta": meta,
        "run_dir": run_dir,
        "model_dir": model_dir,
        "checkpoint": checkpoint,
        "command": command,
        "train_log": run_dir / "logs" / "train.log",
    }


def finalize_run(
    args: argparse.Namespace,
    run_info: dict[str, Any],
    returncode: int,
    visual_cases: list[str],
) -> dict[str, Any]:
    cfg: RunConfig = run_info["cfg"]
    run_dir: Path = run_info["run_dir"]
    model_dir: Path = run_info["model_dir"]
    checkpoint: Path = run_info["checkpoint"]
    train_log: Path = run_info["train_log"]

    metrics = parse_log_metrics(train_log)
    if not checkpoint.exists():
        candidates = sorted(model_dir.glob("models_epoch_*.pth"))
        checkpoint = candidates[-1] if candidates else checkpoint
    actual_epoch = checkpoint_epoch(checkpoint) or args.epochs

    inference_summary = {}
    inference_summary_path = run_dir / "inference_summary.json"
    if returncode == 0 and checkpoint.exists() and int(args.run_inference) == 1:
        if inference_summary_path.exists():
            inference_summary = json.loads(inference_summary_path.read_text(encoding="utf-8"))
        else:
            inference_summary = run_inference_for_run(args, run_dir, run_info["meta"], visual_cases, checkpoint)
            inference_summary_path.write_text(json.dumps(inference_summary, indent=2), encoding="utf-8")

    inspect_summary = {}
    inspect_summary_path = run_dir / "inspect_summary.json"
    if returncode == 0 and checkpoint.exists() and int(args.render_inspect) == 1:
        if inspect_summary_path.exists():
            inspect_summary = json.loads(inspect_summary_path.read_text(encoding="utf-8"))
        else:
            inspect_summary = render_inspection(args, run_dir, visual_cases, actual_epoch, model_dir)
            inspect_summary_path.write_text(json.dumps(inspect_summary, indent=2), encoding="utf-8")

    return {
        "run": cfg.name,
        "meta": run_info["meta"],
        "returncode": returncode,
        "checkpoint": str(checkpoint),
        "config": str(run_dir / "config.json"),
        **metrics,
        "inspect_rmse_mean": inspect_summary.get("rmse_mean"),
        "inspect_warning_count": inspect_summary.get("warning_count"),
        "infer_reconstruct_target_mse_mean": inference_summary.get("reconstruct_target_mse_mean"),
    }


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = args.results_root.expanduser().resolve() / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)

    configs = sweep_configs()
    if args.max_runs is not None:
        configs = configs[: int(args.max_runs)]

    visual_cases = load_split_cases(
        PROJECT_ROOT / "checkpoint-v2/dataset_splits/data_split_real.json",
        int(args.num_visual_cases),
        args.test_cases,
    )

    git_diff_path = result_dir / "pre_run_git_diff.patch"
    subprocess.run(["git", "diff", "--", "."], cwd=PROJECT_ROOT.parent, stdout=git_diff_path.open("w", encoding="utf-8"), check=False)

    manifest = {
        "timestamp": timestamp,
        "python": str(args.python),
        "epochs": args.epochs,
        "train_subset_limit": args.train_subset_limit,
        "visual_cases": visual_cases,
        "gpus": args.gpus,
        "parallel_train": int(args.parallel_train),
        "configs": [{"name": cfg.name, "overrides": cfg.overrides} for cfg in configs],
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    gpus = [gpu.strip() for gpu in str(args.gpus).split(",") if gpu.strip()]
    if not gpus:
        gpus = ["0"]
    parallel_train = max(1, min(int(args.parallel_train), len(gpus)))
    pending = [(index, cfg) for index, cfg in enumerate(configs, start=1)]
    active: list[dict[str, Any]] = []

    def maybe_finalize_skipped(run_info: dict[str, Any]) -> None:
        row = finalize_run(args, run_info, 0, visual_cases)
        rows.append(row)
        write_comparison(result_dir, rows)

    try:
        while pending or active:
            while pending and len(active) < parallel_train:
                index, cfg = pending.pop(0)
                run_info = prepare_run(args, result_dir, cfg, index, timestamp)
                checkpoint: Path = run_info["checkpoint"]
                if int(args.skip_existing) == 1 and checkpoint.exists():
                    print(f"[skip] {cfg.name}: checkpoint already exists at {checkpoint}")
                    maybe_finalize_skipped(run_info)
                    continue

                used_gpus = {item["gpu"] for item in active}
                gpu = next((candidate for candidate in gpus if candidate not in used_gpus), gpus[len(active) % len(gpus)])
                print(f"[run {index}/{len(configs)}] training {cfg.name} on CUDA_VISIBLE_DEVICES={gpu}")
                process, log_handle = start_logged_process(
                    run_info["command"],
                    PROJECT_ROOT,
                    run_info["train_log"],
                    env_overrides={"CUDA_VISIBLE_DEVICES": gpu},
                )
                run_info.update(
                    {
                        "process": process,
                        "log_handle": log_handle,
                        "gpu": gpu,
                        "start_time": time.perf_counter(),
                    }
                )
                active.append(run_info)

            if not active:
                continue

            time.sleep(5.0)
            for run_info in list(active):
                process: subprocess.Popen = run_info["process"]
                returncode = process.poll()
                if returncode is None:
                    continue
                run_info["log_handle"].close()
                active.remove(run_info)
                elapsed = time.perf_counter() - float(run_info["start_time"])
                cfg: RunConfig = run_info["cfg"]
                print(
                    f"[run {run_info['index']}/{len(configs)}] {cfg.name} finished "
                    f"rc={returncode} gpu={run_info['gpu']} elapsed={elapsed:.1f}s"
                )
                row = finalize_run(args, run_info, int(returncode), visual_cases)
                rows.append(row)
                write_comparison(result_dir, rows)
    except KeyboardInterrupt:
        for run_info in active:
            process: subprocess.Popen = run_info["process"]
            if process.poll() is None:
                process.terminate()
            run_info["log_handle"].close()
        raise

    write_comparison(result_dir, rows)
    print(f"Results written to: {result_dir}")
    print(f"Summary: {result_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
