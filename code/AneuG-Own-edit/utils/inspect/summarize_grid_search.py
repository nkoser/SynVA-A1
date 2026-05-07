import argparse
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

META_PATTERN = re.compile(
    r"grid_lr(?P<lr>[0-9mp]+)_rigid(?P<rigid>[0-9mp]+)_op(?P<op>[0-9mp]+)(?:_p0n1(?P<p0n1>[0-9mp]+))?"
)


def slug_to_float(text: str) -> float:
    return float(text.replace("m", "-").replace("p", "."))


def parse_meta(meta: str) -> Optional[Tuple[float, float, float, float]]:
    match = META_PATTERN.fullmatch(meta)
    if match is None:
        return None
    lr = slug_to_float(match.group("lr"))
    rigid = slug_to_float(match.group("rigid"))
    op = slug_to_float(match.group("op"))
    p0n1_text = match.group("p0n1")
    p0n1 = slug_to_float(p0n1_text) if p0n1_text is not None else 1.0
    return lr, rigid, op, p0n1


def find_latest_warped_obj(viz_dir: str) -> Optional[str]:
    if not os.path.isdir(viz_dir):
        return None
    best_epoch = -1
    best_path = None
    for name in os.listdir(viz_dir):
        if not (name.startswith("warped_epoch_") and name.endswith(".obj")):
            continue
        epoch_str = name[len("warped_epoch_"):-len(".obj")]
        try:
            epoch = int(epoch_str)
        except ValueError:
            continue
        if epoch > best_epoch:
            best_epoch = epoch
            best_path = os.path.join(viz_dir, name)
    return best_path


def find_warped_obj_for_epoch(viz_dir: str, epoch: int) -> Optional[str]:
    if not os.path.isdir(viz_dir):
        return None
    candidate = os.path.join(viz_dir, f"warped_epoch_{int(epoch):05d}.obj")
    if os.path.exists(candidate):
        return candidate
    best_epoch = -1
    best_path = None
    for name in os.listdir(viz_dir):
        if not (name.startswith("warped_epoch_") and name.endswith(".obj")):
            continue
        epoch_str = name[len("warped_epoch_"):-len(".obj")]
        try:
            epoch_i = int(epoch_str)
        except ValueError:
            continue
        if epoch_i <= int(epoch) and epoch_i > best_epoch:
            best_epoch = epoch_i
            best_path = os.path.join(viz_dir, name)
    return best_path


def load_obj_vertices_faces(obj_path: str) -> Tuple[np.ndarray, np.ndarray]:
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                idx = []
                for token in parts:
                    v_idx = token.split("/")[0]
                    idx.append(int(v_idx) - 1)
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) > 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def mesh_volume_from_obj(obj_path: str) -> Optional[float]:
    verts, faces = load_obj_vertices_faces(obj_path)
    if verts.size == 0 or faces.size == 0:
        return None
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1, v2)
    volume = np.einsum("ij,ij->i", v0, cross).sum() / 6.0
    return float(abs(volume))


def read_last_loss_terms(loss_log_path: str) -> Optional[Dict]:
    if not os.path.exists(loss_log_path):
        return None
    last = None
    with open(loss_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            last = rec
    return last


def draw_mesh(ax, obj_path: str, elev: float, azim: float, title: str):
    verts, faces = load_obj_vertices_faces(obj_path)
    if verts.size > 0 and faces.size > 0:
        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            triangles=faces,
            linewidth=0.04,
            edgecolor=(0.1, 0.1, 0.1, 0.25),
            color="#5DA5DA",
            alpha=0.95,
            antialiased=True,
        )
        center = verts.mean(axis=0)
        span = np.max(np.abs(verts - center), axis=0).max()
        if span > 0:
            ax.set_xlim(center[0] - span, center[0] + span)
            ax.set_ylim(center[1] - span, center[1] + span)
            ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_title(title, fontsize=10, pad=4)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def resolve_target_obj(case_root: str, target_obj: str) -> Optional[str]:
    if target_obj:
        return target_obj if os.path.exists(target_obj) else None
    for meta in sorted(os.listdir(case_root)):
        meta_dir = os.path.join(case_root, meta)
        if not os.path.isdir(meta_dir):
            continue
        candidate = os.path.join(meta_dir, "viz", "target.obj")
        if os.path.exists(candidate):
            return candidate
    return None


def write_top_k_csv(path: str, records: List[Dict], metric: str):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "run_id",
            "meta",
            "metric",
            "lr",
            "loss_rigid",
            "loss_openings_p",
            "loss_p0n1_scale",
            "cases_with_logs",
            "cases_expected",
            "return_code",
        ])
        for idx, rec in enumerate(records, start=1):
            writer.writerow([
                idx,
                rec.get("run_id"),
                rec.get("meta"),
                rec.get(metric),
                rec.get("lr"),
                rec.get("loss_rigid"),
                rec.get("loss_openings_p"),
                rec.get("loss_p0n1_scale"),
                rec.get("cases_with_logs"),
                rec.get("cases_expected"),
                rec.get("return_code"),
            ])


def write_top_k_md(path: str, records: List[Dict], metric: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Top Grid Search Results\n\n")
        f.write(f"Metric: `{metric}` (lower is better)\n\n")
        f.write("| Rank | run_id | metric | lr | loss_rigid | loss_openings_p | loss_p0n1_scale | cases | rc |\n")
        f.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for idx, rec in enumerate(records, start=1):
            cases = f"{rec.get('cases_with_logs')}/{rec.get('cases_expected')}"
            f.write(
                f"| {idx} | {rec.get('run_id')} | {rec.get(metric)} | "
                f"{rec.get('lr')} | {rec.get('loss_rigid')} | {rec.get('loss_openings_p')} | "
                f"{rec.get('loss_p0n1_scale')} | {cases} | {rec.get('return_code')} |\n"
            )


def main():
    parser = argparse.ArgumentParser("Summarize grid search results and optionally render Top-K meshes.")
    parser.add_argument("--grid_dir", type=str, default=os.path.join(REPO_ROOT, "checkpoints-new/ghd_grid_search/c0013_all_settings"))
    parser.add_argument("--case_root", type=str, default=os.path.join(REPO_ROOT, "checkpoints-new/ghd_fitting_output_new/C0013"))
    parser.add_argument("--metric", type=str, default="avg_final_total_loss",
                        choices=["avg_final_total_loss", "median_final_total_loss", "min_final_total_loss", "max_final_total_loss"])
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--require_success", type=int, default=1)
    parser.add_argument("--require_complete", type=int, default=1)
    parser.add_argument("--render_top", type=int, default=1)
    parser.add_argument("--include_target", type=int, default=1)
    parser.add_argument("--target_obj", type=str, default="")
    parser.add_argument("--out_prefix", type=str, default="top_k")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=35.0)
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument("--max_cols", type=int, default=5)
    parser.add_argument("--report_extremes", type=int, default=1,
                        help="1: report largest mesh volume and best opening (ostium) match.")
    args = parser.parse_args()

    summary_path = os.path.join(args.grid_dir, "grid_summary.json")
    if not os.path.exists(summary_path):
        raise RuntimeError(f"grid_summary.json not found at {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    filtered = []
    for rec in records:
        if args.require_success and int(rec.get("return_code", 1)) != 0:
            continue
        if args.require_complete and rec.get("cases_with_logs") != rec.get("cases_expected"):
            continue
        metric_val = rec.get(args.metric)
        if metric_val is None:
            continue
        filtered.append(rec)

    if not filtered:
        raise RuntimeError("No valid records found after filtering.")

    filtered.sort(key=lambda r: math.inf if r.get(args.metric) is None else r.get(args.metric))
    top_k = filtered[: max(1, int(args.top_k))]

    print(f"Top {len(top_k)} runs by {args.metric}:")
    for idx, rec in enumerate(top_k, start=1):
        print(
            f"#{idx}: run_id={rec.get('run_id')} {args.metric}={rec.get(args.metric)} "
            f"lr={rec.get('lr')} rigid={rec.get('loss_rigid')} op={rec.get('loss_openings_p')} "
            f"p0n1={rec.get('loss_p0n1_scale')}"
        )

    csv_path = os.path.join(args.grid_dir, f"{args.out_prefix}_summary.csv")
    md_path = os.path.join(args.grid_dir, f"{args.out_prefix}_summary.md")
    write_top_k_csv(csv_path, top_k, args.metric)
    write_top_k_md(md_path, top_k, args.metric)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")

    if int(args.report_extremes) == 1:
        largest = None
        best_ostium = None
        for rec in filtered:
            meta = rec.get("meta")
            if not meta:
                continue
            viz_dir = os.path.join(args.case_root, meta, "viz")
            if args.epoch >= 0:
                obj_path = find_warped_obj_for_epoch(viz_dir, args.epoch)
            else:
                obj_path = find_latest_warped_obj(viz_dir)
            if obj_path:
                vol = mesh_volume_from_obj(obj_path)
                if vol is not None:
                    if largest is None or vol > largest["volume"]:
                        largest = {
                            "run_id": rec.get("run_id"),
                            "meta": meta,
                            "volume": vol,
                            "obj_path": obj_path,
                            "lr": rec.get("lr"),
                            "loss_rigid": rec.get("loss_rigid"),
                            "loss_openings_p": rec.get("loss_openings_p"),
                            "loss_p0n1_scale": rec.get("loss_p0n1_scale"),
                        }

            loss_log_path = os.path.join(args.case_root, meta, "loss_log.jsonl")
            last_terms = read_last_loss_terms(loss_log_path)
            if last_terms:
                p = last_terms.get("loss_openings_p")
                n = last_terms.get("loss_openings_n")
                if p is None:
                    continue
                try:
                    p = float(p)
                except (TypeError, ValueError):
                    continue
                ostium_score = p
                if n is not None:
                    try:
                        ostium_score += float(n)
                    except (TypeError, ValueError):
                        pass
                if best_ostium is None or ostium_score < best_ostium["ostium_score"]:
                    best_ostium = {
                        "run_id": rec.get("run_id"),
                        "meta": meta,
                        "ostium_score": ostium_score,
                        "loss_openings_p": p,
                        "loss_openings_n": n,
                        "loss_log_path": loss_log_path,
                        "lr": rec.get("lr"),
                        "loss_rigid": rec.get("loss_rigid"),
                        "loss_openings_p_hp": rec.get("loss_openings_p"),
                        "loss_p0n1_scale": rec.get("loss_p0n1_scale"),
                    }

        extremes = {"largest_volume": largest, "best_ostium": best_ostium}
        extremes_path = os.path.join(args.grid_dir, f"{args.out_prefix}_extremes.json")
        with open(extremes_path, "w", encoding="utf-8") as f:
            json.dump(extremes, f, indent=2)
        if largest:
            print(
                "Largest mesh volume:\n"
                f"- run_id={largest['run_id']} volume={largest['volume']:.6f}\n"
                f"- meta={largest['meta']}\n"
                f"- obj={largest['obj_path']}"
            )
        else:
            print("Largest mesh volume: not found.")
        if best_ostium:
            print(
                "Best ostium match (min loss_openings_p [+ loss_openings_n]):\n"
                f"- run_id={best_ostium['run_id']} score={best_ostium['ostium_score']:.6f}\n"
                f"- meta={best_ostium['meta']}\n"
                f"- loss_log={best_ostium['loss_log_path']}"
            )
        else:
            print("Best ostium match: not found.")
        print(f"Wrote: {extremes_path}")

    if int(args.render_top) != 1:
        return
    if plt is None:
        print("matplotlib not installed; skipping render.")
        return

    target_obj_path = resolve_target_obj(args.case_root, args.target_obj)
    if int(args.include_target) == 1 and target_obj_path is None:
        print("Target mesh not found; skipping target preview.")
        args.include_target = 0

    items = []
    if int(args.include_target) == 1 and target_obj_path:
        items.append(("TARGET", target_obj_path, None))

    for rec in top_k:
        meta = rec.get("meta")
        obj_path = rec.get("final_obj_path")
        if not obj_path and meta:
            viz_dir = os.path.join(args.case_root, meta, "viz")
            if args.epoch >= 0:
                obj_path = find_warped_obj_for_epoch(viz_dir, args.epoch)
            else:
                obj_path = find_latest_warped_obj(viz_dir)
        if not obj_path or not os.path.exists(obj_path):
            continue
        parsed = parse_meta(meta) if meta else None
        if parsed:
            lr, rigid, op, p0n1 = parsed
            title = f"r{rigid:g} o{op:g} pn{p0n1:g}\nL={rec.get(args.metric)}"
        else:
            title = f"{rec.get('run_id')}\nL={rec.get(args.metric)}"
        items.append((title, obj_path, rec.get(args.metric)))

    if not items:
        print("No mesh previews found for top-k runs.")
        return

    n = len(items)
    max_cols = max(1, int(args.max_cols))
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(cols * 2.6, rows * 2.6))

    for i, (title, obj_path, _) in enumerate(items):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        draw_mesh(ax, obj_path, args.elev, args.azim, title)

    epoch_tag = f" @ epoch {args.epoch:05d}" if args.epoch >= 0 else " @ latest"
    fig.suptitle(
        f"Top {len(top_k)} by {args.metric}{epoch_tag}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0.02, 0.02, 1.0, 0.93])

    out_path = os.path.join(args.grid_dir, f"{args.out_prefix}_meshes.png")
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
