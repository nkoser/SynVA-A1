#!/usr/bin/env python
"""Render a 1xN comparison of attached meshes (GT + methods) for one case."""
import argparse, os
import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _draw(ax, V, F, title, color):
    ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=F,
                    color=color, edgecolor=(0, 0, 0, 0.08),
                    linewidth=0.03, alpha=0.92)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=18, azim=35)
    rng = max(np.ptp(V, axis=0)); mid = V.mean(axis=0); h = rng * 0.55
    ax.set_xlim(mid[0]-h, mid[0]+h); ax.set_ylim(mid[1]-h, mid[1]+h)
    ax.set_zlim(mid[2]-h, mid[2]+h); ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--attached_dir", required=True,
                    help="dir containing {gt,A,C,D}_attached.obj")
    ap.add_argument("--healthy_root", default="/path/to/healthy_vessel")
    ap.add_argument("--healthy_mesh", default=None,
                    help="Explicit path to healthy/cut vessel mesh (overrides healthy_root layout).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.healthy_mesh:
        healthy_path = args.healthy_mesh
    else:
        healthy_path = os.path.join(args.healthy_root,
                                    f"{args.case}_vessel_submesh_closed",
                                    f"{args.case}_vessel_submesh_closed.obj")
    healthy = trimesh.load(healthy_path, process=False)

    tags = ["healthy", "gt", "A", "C", "D"]
    colors = {"healthy": "#cccccc", "gt": "#888888",
              "A": "#1f77b4", "C": "#2ca02c", "D": "#d62728"}
    fig, axes = plt.subplots(1, 5, figsize=(2.8 * 5, 3.2),
                             subplot_kw={"projection": "3d"})
    for i, tag in enumerate(tags):
        if tag == "healthy":
            m = healthy
            title = f"{args.case[:18]}\nhealthy vessel (cut target)"
        else:
            p = os.path.join(args.attached_dir, f"{tag}_attached.obj")
            m = trimesh.load(p, process=False)
            title = f"{tag} attached"
        V = np.asarray(m.vertices); F = np.asarray(m.faces)
        _draw(axes[i], V, F, title, colors[tag])
    plt.suptitle(f"Aneurysm-to-vessel attachment — {args.case}", fontsize=12)
    plt.tight_layout()
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
