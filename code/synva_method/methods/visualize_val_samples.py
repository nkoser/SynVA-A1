#!/usr/bin/env python
"""Visualize per-method val samples (1 sample/method) alongside ground-truth.

For each of the first N val cases, samples 1 GHD per method (A/B/C/D/baseline),
reconstructs the canonical-frame mesh from each GHD, and saves:

  out_dir/<case>/gt.obj
  out_dir/<case>/{A,B,C,D,baseline}.obj
  out_dir/<case>/grid.png   (matplotlib panel: GT + each method)

  out_dir/_overview.png     (n_cases rows × (1+n_methods) cols)
"""
from __future__ import annotations
import argparse, json, os, sys, pickle, io as _io
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)

from pytorch3d.io import save_obj
from methods._common.mesh_loss import CoeffToMesh
from methods.eval_all import _sa, _build_dataset, _copy_stats, METHOD_LOADERS
from first_stage_vessel_aware import collate_fn
from train_vessel_flow_matching import condition_from_batch


class _CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(_io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cases_file", required=True)
    p.add_argument("--A", default=None); p.add_argument("--B", default=None)
    p.add_argument("--C", default=None); p.add_argument("--D", default=None)
    p.add_argument("--E", default=None)
    p.add_argument("--W", default=None)
    p.add_argument("--baseline", default=None)
    p.add_argument("--n_cases", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--flow_steps", type=int, default=64)
    p.add_argument("--flow_sampler", default="heun")
    p.add_argument("--canonical_mesh", default="/path/to/SynVA-A1/checkpoints/canonical_average/part_aligned.obj")
    p.add_argument("--eigen_chk", default="/path/to/SynVA-A1/checkpoints/canonical_average/eigen_chk_144.pkl")
    p.add_argument("--num_basis", type=int, default=144)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def _denorm_to_verts(c2m, ghd_normalized, ds):
    """Decode normalized GHD [B, 432] to verts [B, V, 3] using scaled canonical mesh."""
    mean = ds.ghd_mean.to(ghd_normalized.device)
    std  = ds.ghd_std.to(ghd_normalized.device)
    v, _ = c2m(ghd_normalized, mean, std, want_normals=False)
    return v


def _ghd_to_mesh(ghd_module, coeff_144x3):
    """DEPRECATED — kept for backward-compat callers."""
    return ghd_module.forward(coeff_144x3)


def _plot_mesh(ax, verts, faces, title, color):
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                    triangles=faces, edgecolor=(0, 0, 0, 0.15),
                    linewidth=0.05, color=color, alpha=0.85)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=35)
    rng = max(np.ptp(verts, axis=0))
    mid = verts.mean(axis=0)
    half = rng * 0.55
    ax.set_xlim(mid[0]-half, mid[0]+half)
    ax.set_ylim(mid[1]-half, mid[1]+half)
    ax.set_zlim(mid[2]-half, mid[2]+half)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


COLORS = {"GT": "lightgray", "A": "#d62728", "B": "#1f77b4",
          "C": "#2ca02c", "D": "#ff7f0e", "E": "#17becf",
          "W": "#9467bd", "baseline": "#8c564b"}


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    cases_full = json.load(open(args.cases_file))
    cases = cases_full[: args.n_cases]
    print(f"[viz] {len(cases)}/{len(cases_full)} val cases")

    # Load every requested method.
    methods = []
    for tag in ("A", "B", "C", "D", "E", "W", "baseline"):
        path = getattr(args, tag)
        if not path:
            continue
        print(f"[load] {tag}: {path}")
        ck, sa, cond_net, sample = METHOD_LOADERS[tag](path, device)
        methods.append({"tag": tag, "ckpt": ck, "sa": sa,
                        "cond": cond_net, "sample": sample, "path": path})
    if not methods:
        raise SystemExit("No method ckpts given")

    # GHD module shared by all methods (canonical-frame reconstruction only).
    canonical_norm = float(methods[0]["sa"].get("canonical_norm_factor", 1.10))
    c2m = CoeffToMesh(args.canonical_mesh, args.eigen_chk,
                      num_basis=args.num_basis, device=device,
                      canonical_norm_factor=canonical_norm)
    canonical_faces = c2m.canonical.faces_packed().detach().cpu().numpy()

    sa0 = methods[0]["sa"]
    ds = _build_dataset(cases, sa0)
    _copy_stats(ds, methods[0]["ckpt"])
    if "orig_ghd_mean" in methods[0]["ckpt"]:
        ds.ghd_mean = methods[0]["ckpt"]["orig_ghd_mean"].cpu()
        ds.ghd_std = methods[0]["ckpt"]["orig_ghd_std"].cpu()

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=len(cases), shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))
    gt_norm = batch["ghd"].to(device)              # [N, 432] normalized

    n_cases = gt_norm.shape[0]
    n_meth = len(methods)
    cols = 1 + n_meth
    fig, axes = plt.subplots(n_cases, cols, figsize=(2.4 * cols, 2.4 * n_cases),
                             subplot_kw={"projection": "3d"})
    if n_cases == 1:
        axes = axes[None, :]

    # Reconstruct GT (per-method datasets share the canonical frame; GT coeffs are
    # actual stored ghd, denormalized via the same stats used by methods[0]).

    def _recon(ghd_norm_1xN):
        v = _denorm_to_verts(c2m, ghd_norm_1xN.to(device), ds)[0].detach().cpu().numpy()
        return v, canonical_faces

    for ci, case in enumerate(cases):
        case_dir = os.path.join(args.out_dir, case)
        os.makedirs(case_dir, exist_ok=True)
        gv, gf = _recon(gt_norm[ci:ci+1])
        save_obj(os.path.join(case_dir, "gt.obj"),
                 verts=torch.from_numpy(gv), faces=torch.from_numpy(gf))
        _plot_mesh(axes[ci, 0], gv, gf, f"{case[:18]}\nGT", COLORS["GT"])

        for mi, m in enumerate(methods):
            tag = m["tag"]; sa = m["sa"]
            one = {key: value[ci:ci+1] for key, value in batch.items() if torch.is_tensor(value)}
            cond = condition_from_batch(
                m["cond"],
                one,
                device,
                zero_vessel=bool(sa.get("no_vessel_pts", False)),
                zero_all=bool(sa.get("no_conditioning", False)),
            )
            with torch.no_grad():
                S = m["sample"](cond, 1, args)                 # [1, 1, 432]
            x = S[0, 0]
            sv, sf = _recon(x.unsqueeze(0))
            save_obj(os.path.join(case_dir, f"{tag}.obj"),
                     verts=torch.from_numpy(sv), faces=torch.from_numpy(sf))
            _plot_mesh(axes[ci, 1 + mi], sv, sf,
                       f"{tag} sample", COLORS.get(tag, "tan"))

    plt.tight_layout()
    out_png = os.path.join(args.out_dir, "_overview.png")
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] overview -> {out_png}")
    print(f"[done] per-case OBJs -> {args.out_dir}/<case>/*.obj")


if __name__ == "__main__":
    main()
