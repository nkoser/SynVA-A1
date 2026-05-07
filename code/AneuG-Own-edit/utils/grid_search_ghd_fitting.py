import argparse
import csv
import itertools
import json
import os
import queue
import subprocess
import sys
import threading
import time
from html import escape

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


def parse_args():
    parser = argparse.ArgumentParser("Grid search runner for ghd_fitting.py")

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='Single device (e.g. "cuda:0") or GPU pool list (e.g. "0,1,2,3") or "all".',
    )
    parser.add_argument("--root_template", type=str, default=os.path.join(REPO_ROOT, "checkpoints-new/alignment"))
    parser.add_argument("--root_target", type=str, default=os.path.join(REPO_ROOT, "checkpoints-new/alignment"))
    parser.add_argument("--name_canonical", type=str, default="canonical_model")
    parser.add_argument("--pouch_only", type=int, default=1)
    parser.add_argument("--opening_min_ratio", type=float, default=0.3)
    parser.add_argument("--loss_volume", type=float, default=0.0)
    parser.add_argument("--loss_opening_boundary_smooth", type=float, default=0.1)
    parser.add_argument(
        "--loss_ablation",
        type=int,
        default=0,
        help="1: run loss on/off ablation across loss groups (pouch-only recommended).",
    )
    parser.add_argument(
        "--loss_ablation_groups",
        type=str,
        default="p0n1,laplacian,edge,consistency,rigid,openings,volume,opening_boundary_smooth",
        help="Comma-separated loss groups to ablate when --loss_ablation=1.",
    )
    parser.add_argument("--opening_boundary_smooth_width", type=int, default=3)
    parser.add_argument("--center_opening_at_origin", type=int, default=1)
    parser.add_argument("--center_opening_index", type=int, default=0)
    parser.add_argument("--debug_opening_losses", type=int, default=0)
    parser.add_argument("--debug_opening_normal_scale", type=float, default=0.01)
    parser.add_argument("--debug_opening_point_marker_scale", type=float, default=0.0035)
    parser.add_argument("--save_root", type=str, default=os.path.join(REPO_ROOT, "checkpoints-new/ghd_fitting_output_new_grid"))
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--target_case", type=str, default="C0013")

    parser.add_argument("--lr_values", type=float, nargs="+", default=[0.0025, 0.001, 0.00075])
    parser.add_argument("--loss_rigid_values", type=float, nargs="+", default=[90.0, 70.0, 50.0])
    parser.add_argument("--loss_openings_p_values", type=float, nargs="+", default=[7.5, 5.0, 2.5])
    parser.add_argument(
        "--loss_p0n1_scale_values",
        type=float,
        nargs="+",
        default=[2.0],
        help="Shared sweep values for loss_p0/loss_n1 scale (loss_p0=1.0*x, loss_n1=0.8*x).",
    )

    parser.add_argument("--meta_prefix", type=str, default="grid")
    parser.add_argument("--output_dir", type=str, default=os.path.join(REPO_ROOT, "checkpoints-new/ghd_grid_search/c0013_all_settings"))
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--run_contact_sheet", type=int, default=1,
                        help="1: run utils/inspect/make_grid_mesh_contact_sheet.py after grid search")
    parser.add_argument("--contact_out_name", type=str, default="grid_mesh_comparison.png")
    parser.add_argument(
        "--contact_rerun_every",
        type=int,
        default=1000,
        help="Create additional contact sheets every N epochs using warped_epoch_XXXXX.obj (default: 1000).",
    )
    parser.add_argument(
        "--contact_rerun_out_prefix",
        type=str,
        default="grid_mesh_comparison_rerun_epoch_",
        help="Filename prefix for periodic contact sheets.",
    )
    parser.add_argument("--contact_dpi", type=int, default=220)
    parser.add_argument("--contact_elev", type=float, default=20.0)
    parser.add_argument("--contact_azim", type=float, default=35.0)
    parser.add_argument("--cpu_threads", type=int, default=5,
                        help="Pass through to ghd_fitting.py --cpu_threads (0 keeps default).")
    parser.add_argument("--cpu_interop_threads", type=int, default=3,
                        help="Pass through to ghd_fitting.py --cpu_interop_threads (0 keeps default).")

    return parser.parse_args()


def _unique_keep_order(seq):
    seen = set()
    out = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _cuda_device_count():
    try:
        import torch
    except ModuleNotFoundError:
        return 0
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def _parse_device_pool(device_arg: str):
    if not device_arg:
        return []
    device_str = device_arg.strip().lower()
    if device_str in {"all", "cuda:all"}:
        count = _cuda_device_count()
        return list(range(count))
    if "," not in device_arg:
        return []
    parts = []
    for token in device_arg.split(","):
        tok = token.strip()
        if not tok:
            continue
        if tok.lower().startswith("cuda:"):
            tok = tok.split("cuda:")[-1]
        if tok.isdigit():
            parts.append(int(tok))
        else:
            raise ValueError(f"Unrecognized device token '{token}' in --device")
    return _unique_keep_order(parts)


def _normalize_single_device(device_arg: str):
    if device_arg and device_arg.isdigit():
        return f"cuda:{device_arg}"
    return device_arg


def _sanitize_case_for_dir(name: str) -> str:
    if not name:
        return "unknown_case"
    cleaned = "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in name)
    return cleaned or "unknown_case"


def _loss_group_codes(groups):
    code_map = {
        "p0n1": "p0n1",
        "laplacian": "lap",
        "edge": "edge",
        "consistency": "cons",
        "rigid": "rig",
        "openings": "op",
        "volume": "vol",
        "opening_boundary_smooth": "obs",
    }
    return [code_map.get(g, g) for g in groups]


def _build_ghd_command(
    args,
    ghd_fitting_script,
    lr,
    rigid,
    openings_p,
    p0n1_scale,
    meta,
    device_str,
):
    cmd = [
        args.python_bin,
        ghd_fitting_script,
        "--device",
        device_str,
        "--root_template",
        args.root_template,
        "--root_target",
        args.root_target,
        "--name_canonical",
        args.name_canonical,
        "--pouch_only",
        str(args.pouch_only),
        "--opening_min_ratio",
        str(args.opening_min_ratio),
        "--loss_volume",
        str(args.loss_volume),
        "--loss_opening_boundary_smooth",
        str(args.loss_opening_boundary_smooth),
        "--opening_boundary_smooth_width",
        str(args.opening_boundary_smooth_width),
        "--center_opening_at_origin",
        str(args.center_opening_at_origin),
        "--center_opening_index",
        str(args.center_opening_index),
        "--debug_opening_losses",
        str(args.debug_opening_losses),
        "--debug_opening_normal_scale",
        str(args.debug_opening_normal_scale),
        "--debug_opening_point_marker_scale",
        str(args.debug_opening_point_marker_scale),
        "--save_root",
        args.save_root,
        "--epochs",
        str(args.epochs),
        "--target_label",
        args.target_case,
        "--lr",
        str(lr),
        "--loss_rigid",
        str(rigid),
        "--loss_openings_p",
        str(openings_p),
        "--loss_p0n1_scale",
        str(p0n1_scale),
        "--cpu_threads",
        str(args.cpu_threads),
        "--cpu_interop_threads",
        str(args.cpu_interop_threads),
        "--meta",
        meta,
    ]
    return cmd


def _run_combo(
    idx,
    total,
    combo,
    args,
    labels,
    ghd_fitting_script,
    run_dir,
    device_str,
    device_id=None,
):
    if isinstance(combo, dict):
        lr = combo["lr"]
        rigid = combo["loss_rigid"]
        openings_p = combo["loss_openings_p"]
        p0n1_scale = combo["loss_p0n1_scale"]
        loss_disable = combo.get("loss_disable")
        meta_extra = combo.get("meta_extra")
    else:
        lr, rigid, openings_p, p0n1_scale = combo
        loss_disable = None
        meta_extra = None
    run_id = f"run_{idx:02d}"
    meta = (
        f"{args.meta_prefix}"
        f"_lr{make_slug(lr)}"
        f"_rigid{make_slug(rigid)}"
        f"_op{make_slug(openings_p)}"
        f"_p0n1{make_slug(p0n1_scale)}"
    )
    if meta_extra:
        meta = f"{meta}_{meta_extra}"

    stdout_log = os.path.join(run_dir, f"{run_id}_stdout.log")
    stderr_log = os.path.join(run_dir, f"{run_id}_stderr.log")

    cmd = _build_ghd_command(
        args,
        ghd_fitting_script,
        lr,
        rigid,
        openings_p,
        p0n1_scale,
        meta,
        device_str,
    )
    if loss_disable:
        print(
            "[loss_ablation] warning: legacy loss_disable='{}' is ignored (flag removed in ghd_fitting.py).".format(
                loss_disable
            )
        )

    prefix = f"[GPU {device_id}] " if device_id is not None else ""
    print(
        f"{prefix}[{idx}/{total}] Starting {run_id}: "
        f"lr={lr:g}, loss_rigid={rigid:g}, loss_openings_p={openings_p:g}, "
        f"loss_p0n1_scale={p0n1_scale:g}, meta={meta}"
    )

    start = time.time()
    return_code = -1
    error = None
    try:
        with open(stdout_log, "w", encoding="utf-8") as so, open(stderr_log, "w", encoding="utf-8") as se:
            completed = subprocess.run(cmd, stdout=so, stderr=se)
        return_code = int(completed.returncode)
    except Exception as exc:
        error = repr(exc)
    duration = time.time() - start

    metrics = aggregate_combo_metrics(args.save_root, labels, meta)

    rec = {
        "run_id": run_id,
        "meta": meta,
        "lr": float(lr),
        "loss_rigid": float(rigid),
        "loss_openings_p": float(openings_p),
        "loss_p0n1_scale": float(p0n1_scale),
        "duration_sec": float(duration),
        "return_code": return_code,
        "cases_expected": int(len(labels)),
        "cases_with_logs": int(metrics["cases_with_logs"]),
        "avg_final_total_loss": metrics["avg_final_total_loss"],
        "median_final_total_loss": metrics["median_final_total_loss"],
        "min_final_total_loss": metrics["min_final_total_loss"],
        "max_final_total_loss": metrics["max_final_total_loss"],
        "stdout_log": stdout_log,
        "stderr_log": stderr_log,
        "per_case": metrics["per_case"],
        "final_obj_path": None,
        "final_obj_epoch": None,
    }
    if loss_disable:
        rec["loss_disable_requested"] = loss_disable
    if error:
        rec["error"] = error

    if args.target_case:
        viz_dir = os.path.join(args.save_root, args.target_case, meta, "viz")
        final_obj_path, final_obj_epoch = find_latest_warped_obj(viz_dir)
        rec["final_obj_path"] = final_obj_path
        rec["final_obj_epoch"] = final_obj_epoch

    print(
        f"{prefix}[{idx}/{total}] Finished {run_id} in {duration:.1f}s, rc={return_code}, "
        f"cases={rec['cases_with_logs']}/{rec['cases_expected']}, avg_final_total_loss={rec['avg_final_total_loss']}"
    )
    return rec


def make_slug(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def discover_valid_labels(root_target: str, name_canonical: str, pouch_only: bool):
    labels = []
    if not os.path.isdir(root_target):
        return labels

    for label in sorted(os.listdir(root_target)):
        case_dir = os.path.join(root_target, label)
        if not os.path.isdir(case_dir):
            continue
        if label == name_canonical:
            continue
        has_opa = os.path.exists(os.path.join(case_dir, "opa_checkpoint.pkl"))
        has_diff = os.path.exists(os.path.join(case_dir, "diff_centreline_checkpoint.pkl"))
        if has_opa if pouch_only else (has_opa and has_diff):
            labels.append(label)
    return labels


def read_last_total_loss(loss_log_path: str):
    if not os.path.exists(loss_log_path):
        return None, None

    last_total_loss = None
    last_epoch = None
    with open(loss_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                last_total_loss = float(rec.get("total_loss"))
                last_epoch = int(rec.get("epoch"))
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
    return last_total_loss, last_epoch


def find_latest_warped_obj(viz_dir: str):
    if not os.path.isdir(viz_dir):
        return None, None
    prefix = "warped_epoch_"
    suffix = ".obj"
    best_epoch = None
    best_file = None
    for name in os.listdir(viz_dir):
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        mid = name[len(prefix):-len(suffix)]
        try:
            epoch = int(mid)
        except ValueError:
            continue
        if best_epoch is None or epoch > best_epoch:
            best_epoch = epoch
            best_file = os.path.join(viz_dir, name)
    return best_file, best_epoch


def aggregate_combo_metrics(save_root: str, labels, meta: str):
    per_case = []
    for label in labels:
        loss_log_path = os.path.join(save_root, label, meta, "loss_log.jsonl")
        total_loss, epoch = read_last_total_loss(loss_log_path)
        if total_loss is None:
            continue
        per_case.append({"label": label, "final_total_loss": total_loss, "last_epoch": epoch})

    if not per_case:
        return {
            "cases_with_logs": 0,
            "avg_final_total_loss": None,
            "median_final_total_loss": None,
            "min_final_total_loss": None,
            "max_final_total_loss": None,
            "per_case": per_case,
        }

    sorted_vals = sorted(item["final_total_loss"] for item in per_case)
    n = len(sorted_vals)
    median = sorted_vals[n // 2] if n % 2 == 1 else 0.5 * (sorted_vals[n // 2 - 1] + sorted_vals[n // 2])

    return {
        "cases_with_logs": len(per_case),
        "avg_final_total_loss": sum(sorted_vals) / len(sorted_vals),
        "median_final_total_loss": median,
        "min_final_total_loss": min(sorted_vals),
        "max_final_total_loss": max(sorted_vals),
        "per_case": per_case,
    }


def write_combo_case_csv(path: str, combo_records):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "meta",
            "lr",
            "loss_rigid",
            "loss_openings_p",
            "loss_p0n1_scale",
            "label",
            "final_total_loss",
            "last_epoch",
        ])
        for rec in combo_records:
            for case in rec["per_case"]:
                writer.writerow([
                    rec["run_id"],
                    rec["meta"],
                    rec["lr"],
                    rec["loss_rigid"],
                    rec["loss_openings_p"],
                    rec["loss_p0n1_scale"],
                    case["label"],
                    case["final_total_loss"],
                    case["last_epoch"],
                ])


def write_summary_csv(path: str, combo_records):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "run_id",
            "meta",
            "lr",
            "loss_rigid",
            "loss_openings_p",
            "loss_p0n1_scale",
            "duration_sec",
            "return_code",
            "cases_expected",
            "cases_with_logs",
            "avg_final_total_loss",
            "median_final_total_loss",
            "min_final_total_loss",
            "max_final_total_loss",
            "stdout_log",
            "stderr_log",
        ])

        sorted_records = sorted(
            combo_records,
            key=lambda x: float("inf") if x["avg_final_total_loss"] is None else x["avg_final_total_loss"],
        )

        for idx, rec in enumerate(sorted_records, start=1):
            writer.writerow([
                idx,
                rec["run_id"],
                rec["meta"],
                rec["lr"],
                rec["loss_rigid"],
                rec["loss_openings_p"],
                rec["loss_p0n1_scale"],
                rec["duration_sec"],
                rec["return_code"],
                rec["cases_expected"],
                rec["cases_with_logs"],
                rec["avg_final_total_loss"],
                rec["median_final_total_loss"],
                rec["min_final_total_loss"],
                rec["max_final_total_loss"],
                rec["stdout_log"],
                rec["stderr_log"],
            ])


def write_markdown_report(path: str, combo_records):
    sorted_records = sorted(
        combo_records,
        key=lambda x: float("inf") if x["avg_final_total_loss"] is None else x["avg_final_total_loss"],
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("# GHD Fitting Grid Search Report\n\n")
        f.write(f"Total runs: {len(combo_records)}\n\n")

        f.write("## Top 10 by avg_final_total_loss\n\n")
        f.write("| Rank | run_id | lr | loss_rigid | loss_openings_p | loss_p0n1_scale | avg_final_total_loss | median_final_total_loss | cases_with_logs | duration_sec | rc |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")

        for idx, rec in enumerate(sorted_records[:10], start=1):
            f.write(
                "| "
                + " | ".join(
                    [
                        str(idx),
                        rec["run_id"],
                        str(rec["lr"]),
                        str(rec["loss_rigid"]),
                        str(rec["loss_openings_p"]),
                        str(rec["loss_p0n1_scale"]),
                        str(rec["avg_final_total_loss"]),
                        str(rec["median_final_total_loss"]),
                        str(rec["cases_with_logs"]),
                        str(round(rec["duration_sec"], 2)),
                        str(rec["return_code"]),
                    ]
                )
                + " |\n"
            )


def write_final_mesh_index(path: str, combo_records):
    sorted_records = sorted(
        combo_records,
        key=lambda x: float("inf") if x["avg_final_total_loss"] is None else x["avg_final_total_loss"],
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Final Warped Mesh Index\n\n")
        f.write("One row per hyperparameter setting, pointing to the latest available warped OBJ in that run.\n\n")
        f.write("| Rank | run_id | lr | loss_rigid | loss_openings_p | loss_p0n1_scale | avg_final_total_loss | final_obj_epoch | final_obj_path |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|---|\n")
        for idx, rec in enumerate(sorted_records, start=1):
            f.write(
                "| "
                + " | ".join(
                    [
                        str(idx),
                        rec["run_id"],
                        str(rec["lr"]),
                        str(rec["loss_rigid"]),
                        str(rec["loss_openings_p"]),
                        str(rec["loss_p0n1_scale"]),
                        str(rec["avg_final_total_loss"]),
                        str(rec.get("final_obj_epoch")),
                        str(rec.get("final_obj_path")),
                    ]
                )
                + " |\n"
            )


def _heatmap_matrix(records, lr, rigid_values, opening_values, p0n1_scale):
    matrix = []
    for rigid in rigid_values:
        row = []
        for opening in opening_values:
            val = None
            for rec in records:
                if (
                    rec["lr"] == lr
                    and rec["loss_rigid"] == rigid
                    and rec["loss_openings_p"] == opening
                    and rec["loss_p0n1_scale"] == p0n1_scale
                ):
                    val = rec["avg_final_total_loss"]
                    break
            row.append(val)
        matrix.append(row)
    return matrix


def _value_to_rgb(value, vmin, vmax):
    if value is None:
        return (240, 240, 240)
    if vmax <= vmin:
        t = 0.5
    else:
        t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    r = int(35 + 220 * t)
    g = int(135 + 80 * (1.0 - t))
    b = int(220 - 190 * t)
    return (r, g, b)


def write_svg_heatmap(output_dir: str, combo_records, lr_values, rigid_values, opening_values, p0n1_scale):
    sorted_lr = sorted(set(lr_values), reverse=True)
    sorted_rigid = sorted(set(rigid_values), reverse=True)
    sorted_opening = sorted(set(opening_values), reverse=True)
    all_vals = [rec["avg_final_total_loss"] for rec in combo_records if rec["avg_final_total_loss"] is not None]

    left_margin = 120
    top_margin = 65
    panel_gap = 40
    cell_w = 105
    cell_h = 44
    panel_w = cell_w * len(sorted_opening)
    panel_h = cell_h * len(sorted_rigid)

    width = left_margin + len(sorted_lr) * panel_w + (len(sorted_lr) - 1) * panel_gap + 35
    height = top_margin + panel_h + 120

    vmin = min(all_vals) if all_vals else 0.0
    vmax = max(all_vals) if all_vals else 1.0

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="white"/>')
    lines.append('<text x="16" y="30" font-size="20" font-family="Arial" font-weight="bold">Grid Search Comparison (avg_final_total_loss, lower is better)</text>')

    if not all_vals:
        lines.append('<text x="16" y="60" font-size="14" font-family="Arial">No valid loss values found to draw heatmap.</text>')
    else:
        for panel_idx, lr in enumerate(sorted_lr):
            panel_x = left_margin + panel_idx * (panel_w + panel_gap)
            panel_y = top_margin

            lines.append(
                f'<text x="{panel_x + panel_w / 2}" y="48" text-anchor="middle" '
                f'font-size="16" font-family="Arial" font-weight="bold">lr={lr:g}</text>'
            )

            matrix = _heatmap_matrix(combo_records, lr, sorted_rigid, sorted_opening, p0n1_scale)

            for i, rigid in enumerate(sorted_rigid):
                y = panel_y + i * cell_h
                if panel_idx == 0:
                    lines.append(
                        f'<text x="{left_margin - 8}" y="{y + cell_h / 2 + 5}" text-anchor="end" '
                        f'font-size="12" font-family="Arial">{rigid:g}</text>'
                    )
                for j, opening in enumerate(sorted_opening):
                    x = panel_x + j * cell_w
                    val = matrix[i][j]
                    r, g, b = _value_to_rgb(val, vmin, vmax)
                    lines.append(
                        f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" '
                        f'fill="rgb({r},{g},{b})" stroke="#333" stroke-width="0.6"/>'
                    )
                    text = "NA" if val is None else f"{val:.3f}"
                    lines.append(
                        f'<text x="{x + (cell_w - 2) / 2}" y="{y + cell_h / 2 + 4}" text-anchor="middle" '
                        f'font-size="11" font-family="Arial">{escape(text)}</text>'
                    )

            for j, opening in enumerate(sorted_opening):
                x = panel_x + j * cell_w + (cell_w - 2) / 2
                lines.append(
                    f'<text x="{x}" y="{panel_y + panel_h + 20}" text-anchor="middle" '
                    f'font-size="12" font-family="Arial">{opening:g}</text>'
                )

        lines.append(
            f'<text x="{left_margin - 45}" y="{top_margin - 12}" text-anchor="start" '
            f'font-size="12" font-family="Arial" font-weight="bold">loss_rigid</text>'
        )
        lines.append(
            f'<text x="{left_margin + (len(sorted_lr) * panel_w + (len(sorted_lr) - 1) * panel_gap)/2}" '
            f'y="{top_margin + panel_h + 45}" text-anchor="middle" '
            f'font-size="12" font-family="Arial" font-weight="bold">loss_openings_p</text>'
        )

        legend_x = left_margin
        legend_y = top_margin + panel_h + 70
        legend_w = 360
        legend_h = 14
        for step in range(100):
            t = step / 99
            val = vmin + t * (vmax - vmin)
            r, g, b = _value_to_rgb(val, vmin, vmax)
            x = legend_x + t * legend_w
            lines.append(
                f'<rect x="{x}" y="{legend_y}" width="{legend_w/100 + 1}" height="{legend_h}" '
                f'fill="rgb({r},{g},{b})" stroke="none"/>'
            )
        lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="{legend_w}" height="{legend_h}" fill="none" stroke="#333" stroke-width="0.6"/>')
        lines.append(f'<text x="{legend_x}" y="{legend_y + 30}" font-size="11" font-family="Arial">{vmin:.4f} (best)</text>')
        lines.append(f'<text x="{legend_x + legend_w}" y="{legend_y + 30}" text-anchor="end" font-size="11" font-family="Arial">{vmax:.4f} (worst)</text>')

    lines.append("</svg>")
    svg_path = os.path.join(output_dir, "grid_heatmaps.svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return svg_path


def plot_heatmaps(output_dir: str, combo_records, lr_values, rigid_values, opening_values, p0n1_scale):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is not installed in the active environment; skipping heatmap generation.")
        return False

    sorted_lr = sorted(set(lr_values), reverse=True)
    sorted_rigid = sorted(set(rigid_values), reverse=True)
    sorted_opening = sorted(set(opening_values), reverse=True)

    fig, axes = plt.subplots(1, len(sorted_lr), figsize=(6 * len(sorted_lr), 5), squeeze=False)
    axes = axes[0]

    all_vals = [rec["avg_final_total_loss"] for rec in combo_records if rec["avg_final_total_loss"] is not None]
    if not all_vals:
        return False
    vmin = min(all_vals)
    vmax = max(all_vals)

    for ax, lr in zip(axes, sorted_lr):
        matrix = _heatmap_matrix(combo_records, lr, sorted_rigid, sorted_opening, p0n1_scale)
        image_vals = [[vmax if v is None else v for v in row] for row in matrix]
        im = ax.imshow(image_vals, aspect="auto", vmin=vmin, vmax=vmax)

        ax.set_title(f"lr={lr:g}")
        ax.set_xlabel("loss_openings_p")
        ax.set_ylabel("loss_rigid")
        ax.set_xticks(range(len(sorted_opening)))
        ax.set_xticklabels([f"{x:g}" for x in sorted_opening])
        ax.set_yticks(range(len(sorted_rigid)))
        ax.set_yticklabels([f"{x:g}" for x in sorted_rigid])

        for i in range(len(sorted_rigid)):
            for j in range(len(sorted_opening)):
                val = matrix[i][j]
                text = "NA" if val is None else f"{val:.3f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.9)
    cbar.set_label("avg_final_total_loss (lower is better)")
    fig.suptitle("Grid Search Comparison")
    fig.tight_layout()

    heatmap_path = os.path.join(output_dir, "grid_heatmaps.png")
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    ghd_fitting_script = os.path.join(REPO_ROOT, "ghd_fitting.py")
    contact_sheet_script = os.path.join(REPO_ROOT, "utils/inspect/make_grid_mesh_contact_sheet.py")

    default_output_dir = os.path.join(REPO_ROOT, "checkpoints-new/ghd_grid_search/c0013_all_settings")
    if args.output_dir == default_output_dir and args.target_case:
        case_slug = _sanitize_case_for_dir(args.target_case)
        args.output_dir = os.path.join(REPO_ROOT, "checkpoints-new/ghd_grid_search", f"{case_slug}_all_settings")

    run_dir = args.output_dir
    os.makedirs(run_dir, exist_ok=True)

    labels = discover_valid_labels(args.root_target, args.name_canonical, bool(args.pouch_only))
    if args.target_case:
        labels = [label for label in labels if label == args.target_case]
    print(f"Discovered {len(labels)} valid target cases.")

    combos = list(
        itertools.product(
            args.lr_values,
            args.loss_rigid_values,
            args.loss_openings_p_values,
            args.loss_p0n1_scale_values,
        )
    )
    if int(getattr(args, "loss_ablation", 0)) == 1:
        groups_raw = [g.strip() for g in args.loss_ablation_groups.split(",") if g.strip()]
        groups = _unique_keep_order(groups_raw)
        if not groups:
            raise RuntimeError("loss_ablation requested but no loss groups were provided.")

        def _first_or_warn(label, values):
            if len(values) > 1:
                print(f"[loss_ablation] {label} has {len(values)} values; using first: {values[0]}")
            return values[0]

        base_lr = _first_or_warn("lr_values", args.lr_values)
        base_rigid = _first_or_warn("loss_rigid_values", args.loss_rigid_values)
        base_openings = _first_or_warn("loss_openings_p_values", args.loss_openings_p_values)
        base_p0n1 = _first_or_warn("loss_p0n1_scale_values", args.loss_p0n1_scale_values)

        combos = []
        total_masks = 2 ** len(groups)
        always_disabled = ["opening_area"]
        for mask in range(1, total_masks):
            active = [g for i, g in enumerate(groups) if (mask >> i) & 1]
            disabled = [g for g in groups if g not in active]
            disabled = _unique_keep_order(disabled + always_disabled)
            loss_disable = ",".join(disabled) if disabled else ""
            meta_extra = "lossmask_" + "_".join(_loss_group_codes(active))
            combos.append(
                {
                    "lr": float(base_lr),
                    "loss_rigid": float(base_rigid),
                    "loss_openings_p": float(base_openings),
                    "loss_p0n1_scale": float(base_p0n1),
                    "loss_disable": loss_disable,
                    "meta_extra": meta_extra,
                }
            )
        print(f"[loss_ablation] {len(groups)} groups -> {len(combos)} combinations.")

    print(f"Running {len(combos)} combinations.")

    combo_records = []
    device_pool = _parse_device_pool(args.device)
    if device_pool:
        device_pool = _unique_keep_order(device_pool)
        count = _cuda_device_count()
        if count <= 0:
            raise RuntimeError("CUDA is not available, but multiple devices were requested.")
        if max(device_pool) >= count:
            raise RuntimeError(
                f"Requested GPU {max(device_pool)} but only {count} CUDA devices are visible. "
                "Use indices relative to CUDA_VISIBLE_DEVICES."
            )

    if device_pool and len(device_pool) > 1:
        total = len(combos)
        print(f"Launching parallel grid search on devices: {device_pool} ({total} runs)")
        work_queue = queue.Queue()
        result_queue = queue.Queue()

        for idx, combo in enumerate(combos, start=1):
            work_queue.put((idx, combo))
        for _ in device_pool:
            work_queue.put(None)

        def _worker(device_id):
            device_str = f"cuda:{device_id}"
            while True:
                item = work_queue.get()
                if item is None:
                    break
                idx, combo = item
                try:
                    rec = _run_combo(
                        idx,
                        total,
                        combo,
                        args,
                        labels,
                        ghd_fitting_script,
                        run_dir,
                        device_str,
                        device_id=device_id,
                    )
                except Exception as exc:
                    if isinstance(combo, dict):
                        lr = combo.get("lr")
                        rigid = combo.get("loss_rigid")
                        openings_p = combo.get("loss_openings_p")
                        p0n1_scale = combo.get("loss_p0n1_scale")
                    else:
                        lr, rigid, openings_p, p0n1_scale = combo
                    rec = {
                        "run_id": f"run_{idx:02d}",
                        "meta": None,
                        "lr": float(lr),
                        "loss_rigid": float(rigid),
                        "loss_openings_p": float(openings_p),
                        "loss_p0n1_scale": float(p0n1_scale),
                        "duration_sec": 0.0,
                        "return_code": -1,
                        "cases_expected": int(len(labels)),
                        "cases_with_logs": 0,
                        "avg_final_total_loss": None,
                        "median_final_total_loss": None,
                        "min_final_total_loss": None,
                        "max_final_total_loss": None,
                        "stdout_log": None,
                        "stderr_log": None,
                        "per_case": [],
                        "final_obj_path": None,
                        "final_obj_epoch": None,
                        "error": repr(exc),
                    }
                    print(f"[GPU {device_id}] Error while processing run_{idx:02d}: {exc}")
                result_queue.put(rec)

        threads = []
        for device_id in device_pool:
            t = threading.Thread(target=_worker, args=(device_id,), daemon=True)
            t.start()
            threads.append(t)

        for _ in range(total):
            combo_records.append(result_queue.get())

        for t in threads:
            t.join()
    else:
        device_str = _normalize_single_device(args.device)
        total = len(combos)
        for idx, combo in enumerate(combos, start=1):
            rec = _run_combo(
                idx,
                total,
                combo,
                args,
                labels,
                ghd_fitting_script,
                run_dir,
                device_str,
            )
            combo_records.append(rec)

    summary_csv = os.path.join(run_dir, "grid_summary.csv")
    per_case_csv = os.path.join(run_dir, "grid_per_case.csv")
    report_md = os.path.join(run_dir, "grid_report.md")
    final_mesh_index_md = os.path.join(run_dir, "grid_final_mesh_index.md")
    raw_json = os.path.join(run_dir, "grid_summary.json")

    write_summary_csv(summary_csv, combo_records)
    write_combo_case_csv(per_case_csv, combo_records)
    write_markdown_report(report_md, combo_records)
    write_final_mesh_index(final_mesh_index_md, combo_records)

    with open(raw_json, "w", encoding="utf-8") as f:
        json.dump(combo_records, f, indent=2)

    unique_p0n1 = sorted(set(float(x) for x in args.loss_p0n1_scale_values))
    svg_heatmap_path = None
    heatmaps_created = False
    if len(unique_p0n1) == 1:
        p0n1_for_heatmap = unique_p0n1[0]
        svg_heatmap_path = write_svg_heatmap(
            run_dir,
            combo_records,
            args.lr_values,
            args.loss_rigid_values,
            args.loss_openings_p_values,
            p0n1_for_heatmap,
        )
        heatmaps_created = plot_heatmaps(
            run_dir,
            combo_records,
            args.lr_values,
            args.loss_rigid_values,
            args.loss_openings_p_values,
            p0n1_for_heatmap,
        )
    else:
        print(
            "\nSkipping heatmaps: multiple --loss_p0n1_scale_values were provided. "
            "Use a single p0n1 scale to generate 2D heatmaps."
        )

    sorted_records = sorted(
        combo_records,
        key=lambda x: float("inf") if x["avg_final_total_loss"] is None else x["avg_final_total_loss"],
    )

    print("\nTop 5 combinations by avg_final_total_loss:")
    for rank, rec in enumerate(sorted_records[:5], start=1):
        print(
            f"#{rank}: lr={rec['lr']:g}, loss_rigid={rec['loss_rigid']:g}, "
            f"loss_openings_p={rec['loss_openings_p']:g}, loss_p0n1_scale={rec['loss_p0n1_scale']:g}, "
            f"avg={rec['avg_final_total_loss']}, "
            f"cases={rec['cases_with_logs']}/{rec['cases_expected']}, rc={rec['return_code']}"
        )

    print("\nArtifacts:")
    print(f"- {summary_csv}")
    print(f"- {per_case_csv}")
    print(f"- {report_md}")
    print(f"- {final_mesh_index_md}")
    print(f"- {raw_json}")
    if svg_heatmap_path:
        print(f"- {svg_heatmap_path}")
    if heatmaps_created:
        print(f"- {os.path.join(run_dir, 'grid_heatmaps.png')}")

    if int(getattr(args, "run_contact_sheet", 1)) == 1:
        if not args.target_case:
            print("\nSkipping contact sheet: --target_case is empty; contact sheet requires one specific case root.")
        elif not os.path.exists(contact_sheet_script):
            print(f"\nSkipping contact sheet: script not found at {contact_sheet_script}")
        else:
            case_root = os.path.join(args.save_root, args.target_case)
            rerun_every = int(getattr(args, "contact_rerun_every", 1000))
            if rerun_every > 0:
                epoch_points = [e for e in range(rerun_every, int(args.epochs) + 1, rerun_every)]
                for epoch_point in epoch_points:
                    rerun_name = f"{args.contact_rerun_out_prefix}{epoch_point:05d}.png"
                    rerun_cmd = [
                        args.python_bin,
                        contact_sheet_script,
                        "--case_root",
                        case_root,
                        "--grid_dir",
                        run_dir,
                        "--out_name",
                        rerun_name,
                        "--epoch",
                        str(epoch_point),
                        "--dpi",
                        str(args.contact_dpi),
                        "--elev",
                        str(args.contact_elev),
                        "--azim",
                        str(args.contact_azim),
                    ]
                    print(f"\nRunning periodic contact sheet for epoch {epoch_point} ...")
                    rerun_completed = subprocess.run(rerun_cmd)
                    if rerun_completed.returncode == 0:
                        print(f"Periodic contact sheet saved: {os.path.join(run_dir, rerun_name)}")
                    else:
                        print(
                            "Periodic contact sheet generation failed with return code "
                            f"{rerun_completed.returncode} (epoch {epoch_point})"
                        )
            contact_cmd = [
                args.python_bin,
                contact_sheet_script,
                "--case_root",
                case_root,
                "--grid_dir",
                run_dir,
                "--out_name",
                args.contact_out_name,
                "--dpi",
                str(args.contact_dpi),
                "--elev",
                str(args.contact_elev),
                "--azim",
                str(args.contact_azim),
            ]
            print("\nRunning contact sheet generation...")
            contact_completed = subprocess.run(contact_cmd)
            if contact_completed.returncode == 0:
                print(f"Contact sheet saved: {os.path.join(run_dir, args.contact_out_name)}")
            else:
                print(f"Contact sheet generation failed with return code {contact_completed.returncode}")


if __name__ == "__main__":
    main()
