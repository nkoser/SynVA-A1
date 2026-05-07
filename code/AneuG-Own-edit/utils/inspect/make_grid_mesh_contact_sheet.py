import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    if best_path is not None:
        return best_path
    return None


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
                    # face token format can be v, v/vt, v//vn, v/vt/vn
                    v_idx = token.split("/")[0]
                    idx.append(int(v_idx) - 1)
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) > 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])

    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def load_summary_loss(summary_csv_path: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    if not os.path.exists(summary_csv_path):
        return values

    with open(summary_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta = row.get("meta")
            value = row.get("avg_final_total_loss")
            if not meta or not value or value == "None":
                continue
            try:
                values[meta] = float(value)
            except ValueError:
                continue
    return values


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
    ax.set_title(title, fontsize=11, pad=4)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def main():
    parser = argparse.ArgumentParser("Create visual contact sheet of all grid-search warped meshes.")
    parser.add_argument("--case_root", type=str, default=os.path.join(REPO_ROOT, "checkpoints-new/ghd_fitting_output_new/C0013"))
    parser.add_argument("--grid_dir", type=str, default=os.path.join(REPO_ROOT, "checkpoints-new/ghd_grid_search/c0013_all_settings"))
    parser.add_argument("--target_obj", type=str, default="", help="Optional explicit path to target.obj")
    parser.add_argument("--out_name", type=str, default="grid_mesh_comparison.png")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=35.0)
    parser.add_argument(
        "--epoch",
        type=int,
        default=-1,
        help="If >=0, use warped_epoch_XXXXX.obj for that epoch; otherwise use latest available warped mesh.",
    )
    args = parser.parse_args()

    summary_path = os.path.join(args.grid_dir, "grid_summary.csv")
    loss_by_meta = load_summary_loss(summary_path)

    entries = []
    for meta in sorted(os.listdir(args.case_root)):
        meta_dir = os.path.join(args.case_root, meta)
        if not os.path.isdir(meta_dir):
            continue
        parsed = parse_meta(meta)
        if parsed is None:
            continue
        lr, rigid, op, p0n1 = parsed
        viz_dir = os.path.join(meta_dir, "viz")
        latest_obj = find_warped_obj_for_epoch(viz_dir, args.epoch) if args.epoch >= 0 else find_latest_warped_obj(viz_dir)
        if latest_obj is None:
            continue
        entries.append({
            "meta": meta,
            "lr": lr,
            "rigid": rigid,
            "op": op,
            "p0n1": p0n1,
            "obj": latest_obj,
            "loss": loss_by_meta.get(meta),
        })

    if not entries:
        raise RuntimeError(f"No valid mesh entries found under {args.case_root}")

    target_obj_path = resolve_target_obj(args.case_root, args.target_obj)
    if target_obj_path is None:
        raise RuntimeError("Target mesh not found. Provide --target_obj or ensure viz/target.obj exists in at least one meta folder.")

    lr_values = sorted({e["lr"] for e in entries}, reverse=True)
    p0n1_values = sorted({e["p0n1"] for e in entries}, reverse=True)
    rigid_values = sorted({e["rigid"] for e in entries}, reverse=True)
    op_values = sorted({e["op"] for e in entries}, reverse=True)

    row_keys = [(lr, p0n1) for lr in lr_values for p0n1 in p0n1_values]
    nrows = len(row_keys)
    ncols = 1 + len(rigid_values) * len(op_values)
    fig = plt.figure(figsize=(ncols * 2.6, nrows * 2.55))

    def get_col_idx(rigid: float, op: float) -> int:
        return rigid_values.index(rigid) * len(op_values) + op_values.index(op)

    def get_row_idx(lr: float, p0n1: float) -> int:
        return row_keys.index((lr, p0n1))

    case_name = os.path.basename(os.path.normpath(args.case_root))

    for row, _ in enumerate(row_keys):
        subplot_idx = row * ncols + 1
        ax_target = fig.add_subplot(nrows, ncols, subplot_idx, projection="3d")
        draw_mesh(ax_target, target_obj_path, args.elev, args.azim, "TARGET")

    for entry in entries:
        row = get_row_idx(entry["lr"], entry["p0n1"])
        col = 1 + get_col_idx(entry["rigid"], entry["op"])
        subplot_idx = row * ncols + col + 1
        ax = fig.add_subplot(nrows, ncols, subplot_idx, projection="3d")

        loss_text = "NA" if entry["loss"] is None else f"{entry['loss']:.3f}"
        draw_mesh(
            ax,
            entry["obj"],
            args.elev,
            args.azim,
            f"r{entry['rigid']:g} o{entry['op']:g} pn{entry['p0n1']:g}\nL={loss_text}",
        )

    for row, (lr, p0n1) in enumerate(row_keys):
        x = 0.002
        y = 1.0 - (row + 0.5) / nrows
        fig.text(x, y, f"lr={lr:g} pn={p0n1:g}", fontsize=11, fontweight="bold", va="center")

    epoch_tag = f" @ epoch {args.epoch:05d}" if args.epoch >= 0 else " @ latest"
    fig.suptitle(
        f"{case_name}: Target + Warped Meshes Across Grid Settings{epoch_tag}",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.015,
        "Glossary: r = loss_rigid, o = loss_openings_p, pn = loss_p0n1_scale, L = avg_final_total_loss",
        ha="center",
        va="center",
        fontsize=11,
    )
    plt.tight_layout(rect=[0.02, 0.04, 1.0, 0.94])

    out_path = os.path.join(args.grid_dir, args.out_name)
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Saved mesh comparison image to: {out_path}")


if __name__ == "__main__":
    main()
