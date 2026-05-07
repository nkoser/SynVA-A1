import argparse
import math
import os
from typing import List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ModuleNotFoundError:
    plt = None
    PdfPages = None

LOSS_ORDER = ["p0n1", "lap", "edge", "cons", "rig", "op", "vol", "obs"]
LOSS_LABELS = {
    "p0n1": "P0N1",
    "lap": "Lap",
    "edge": "Edge",
    "cons": "Cons",
    "rig": "Rigid",
    "op": "Open",
    "vol": "Vol",
    "obs": "OpSmooth",
}


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


def draw_mesh(ax, obj_path: str, elev: float, azim: float):
    verts, faces = load_obj_vertices_faces(obj_path)
    if verts.size > 0 and faces.size > 0:
        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            triangles=faces,
            linewidth=0.15,
            edgecolor=(0.0, 0.0, 0.0, 0.12),
            color="#6EA6D7",
            alpha=0.98,
            antialiased=True,
        )
        center = verts.mean(axis=0)
        span = np.max(np.abs(verts - center), axis=0).max()
        if span > 0:
            ax.set_xlim(center[0] - span, center[0] + span)
            ax.set_ylim(center[1] - span, center[1] + span)
            ax.set_zlim(center[2] - span, center[2] + span)
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.view_init(elev=elev, azim=azim)
    ax.set_facecolor("white")
    ax.set_axis_off()


def find_latest_warped_obj(viz_dir: str, epoch: int) -> Optional[str]:
    if not os.path.isdir(viz_dir):
        return None
    if epoch >= 0:
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
        if epoch >= 0 and epoch_i > epoch:
            continue
        if epoch_i > best_epoch:
            best_epoch = epoch_i
            best_path = os.path.join(viz_dir, name)
    return best_path


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


def extract_loss_codes(meta: str) -> List[str]:
    if "lossmask_" not in meta:
        return LOSS_ORDER[:]
    suffix = meta.split("lossmask_", 1)[1]
    tokens = [t for t in suffix.split("_") if t]
    return [code for code in LOSS_ORDER if code in tokens]


def build_loss_labels(meta: str) -> Tuple[str, str]:
    included_codes = extract_loss_codes(meta)
    included = [LOSS_LABELS[c] for c in included_codes]
    excluded = [LOSS_LABELS[c] for c in LOSS_ORDER if c not in included_codes]
    on = format_loss_lines(included)
    off = format_loss_lines(excluded)
    return on, off


def format_loss_lines(labels: List[str], per_line: int = 5) -> str:
    if not labels:
        return "-"
    lines = []
    for i in range(0, len(labels), per_line):
        lines.append(", ".join(labels[i:i + per_line]))
    return "\n".join(lines)


def annotate_losses(ax, on_text: str, off_text: str, fontsize: int = 6):
    label = f"ON: {on_text}\nOFF: {off_text}"
    ax.text2D(
        0.5,
        -0.08,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=fontsize,
        color="#333333",
    )


def main():
    parser = argparse.ArgumentParser("Export all warped meshes into a multi-page PDF contact sheet.")
    parser.add_argument("--case_root", type=str, required=True, help="Case root folder containing meta subfolders.")
    parser.add_argument("--out_pdf", type=str, default="", help="Output PDF path (default: <case_root>/mesh_grid.pdf)")
    parser.add_argument("--epoch", type=int, default=-1, help="If >=0, use warped_epoch_XXXXX.obj for that epoch; else use latest.")
    parser.add_argument("--cols", type=int, default=6, help="Number of columns per page.")
    parser.add_argument("--per_page", type=int, default=30, help="Number of meshes per page (excluding target).")
    parser.add_argument("--include_target", type=int, default=1, help="1: include target mesh on every page.")
    parser.add_argument("--target_obj", type=str, default="", help="Optional explicit path to target.obj")
    parser.add_argument("--cell_size", type=float, default=2.1, help="Base size per cell in inches.")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=35.0)
    args = parser.parse_args()

    if plt is None or PdfPages is None:
        raise RuntimeError("matplotlib is required to render the PDF.")

    case_root = args.case_root
    if not os.path.isdir(case_root):
        raise RuntimeError(f"case_root does not exist: {case_root}")

    target_obj = resolve_target_obj(case_root, args.target_obj)
    if int(args.include_target) == 1 and target_obj is None:
        print("Target mesh not found; disabling target preview.")
        args.include_target = 0

    entries = []
    for meta in sorted(os.listdir(case_root)):
        meta_dir = os.path.join(case_root, meta)
        if not os.path.isdir(meta_dir):
            continue
        viz_dir = os.path.join(meta_dir, "viz")
        obj_path = find_latest_warped_obj(viz_dir, args.epoch)
        if obj_path is None:
            continue
        entries.append((meta, obj_path))

    def _sort_key(item):
        meta, _ = item
        codes = extract_loss_codes(meta)
        k_len = len(codes)
        full_count = len(LOSS_ORDER)
        if k_len == 1:
            rank = 0
        elif k_len == 2:
            rank = 1
        elif k_len == full_count:
            rank = 3
        else:
            rank = 2
        return (rank, k_len, meta)

    entries.sort(key=_sort_key)

    if not entries:
        raise RuntimeError(f"No warped meshes found under {case_root}")

    out_pdf = args.out_pdf or os.path.join(case_root, "mesh_grid.pdf")
    cols = max(1, int(args.cols))
    meshes_per_page = max(1, int(args.per_page))

    total_pages = math.ceil(len(entries) / meshes_per_page)
    with PdfPages(out_pdf) as pdf:
        for page_idx in range(total_pages):
            page_entries = entries[page_idx * meshes_per_page:(page_idx + 1) * meshes_per_page]
            n_meshes = len(page_entries)
            extra = 1 if int(args.include_target) == 1 else 0
            n_cells = n_meshes + extra
            rows = math.ceil(n_cells / cols)

            fig = plt.figure(figsize=(cols * args.cell_size, rows * args.cell_size), facecolor="white")
            cell_idx = 0
            if int(args.include_target) == 1 and target_obj:
                ax = fig.add_subplot(rows, cols, cell_idx + 1, projection="3d")
                draw_mesh(ax, target_obj, args.elev, args.azim)
                ax.text2D(
                    0.5,
                    -0.08,
                    "ORIGINAL",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=7,
                    color="#111111",
                )
                cell_idx += 1

            for meta, obj_path in page_entries:
                ax = fig.add_subplot(rows, cols, cell_idx + 1, projection="3d")
                draw_mesh(ax, obj_path, args.elev, args.azim)
                on_text, off_text = build_loss_labels(meta)
                annotate_losses(ax, on_text, off_text)
                cell_idx += 1

            epoch_tag = f"epoch {args.epoch:05d}" if args.epoch >= 0 else "latest"
            fig.suptitle(
                f"{os.path.basename(case_root)} | page {page_idx + 1}/{total_pages} | {epoch_tag}",
                fontsize=12,
                fontweight="bold",
                y=0.98,
            )
            plt.tight_layout(rect=[0.02, 0.02, 1.0, 0.95])
            pdf.savefig(fig, dpi=args.dpi)
            plt.close(fig)

    print(f"Saved PDF: {out_pdf}")


if __name__ == "__main__":
    main()
