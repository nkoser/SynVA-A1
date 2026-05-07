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

from pytorch3d.io import save_obj, load_objs_as_meshes
from ghd.base.graph_harmonic_deformation import Graph_Harmonic_Deform
from methods.eval_all import _sa, _build_dataset, _copy_stats, METHOD_LOADERS
from first_stage_vessel_aware import collate_fn


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


def _denorm_to_coeff(x_normalized, ds):
    """x: [..., 432] in normalized space -> [..., 144, 3] raw GHD coefficients."""
    mean = ds.ghd_mean.to(x_normalized.device).view(-1)
    std = ds.ghd_std.to(x_normalized.device).view(-1)
    raw = x_normalized * std + mean
    return raw.view(*raw.shape[:-1], 144, 3)


def _ghd_to_mesh(ghd_module, coeff_144x3):
    """coeff: [144, 3] tensor on same device as ghd_module."""
    return ghd_module.forward(coeff_144x3)  # pytorch3d Meshes (single)


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
          "C": "#2ca02c", "D": "#ff7f0e", "baseline": "#9467bd"}


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
    for tag in ("A", "B", "C", "D", "baseline"):
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
    canonical = load_objs_as_meshes([args.canonical_mesh]).to(device)
    ghd_module = Graph_Harmonic_Deform(base_shape=canonical,
                                       num_Basis=args.num_basis,
                                       eigen_chk=args.eigen_chk)
    # Use identity rigid pose for *all* reconstructions (canonical frame).
    import torch.nn as nn
    setattr(ghd_module, "R", nn.Parameter(torch.zeros(1, 3, device=device)))
    setattr(ghd_module, "s", nn.Parameter(torch.ones(1, 1, device=device)))
    setattr(ghd_module, "T", nn.Parameter(torch.zeros(1, 3, device=device)))

    # Build a dataset using method[0] sa to get GT GHDs (orig-space coeff).
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
    ostium = batch["ostium_params"].to(device)
    vessel_full = batch["vessel_pts"].to(device)

    n_cases = gt_norm.shape[0]
    n_meth = len(methods)
    cols = 1 + n_meth
    fig, axes = plt.subplots(n_cases, cols, figsize=(2.4 * cols, 2.4 * n_cases),
                             subplot_kw={"projection": "3d"})
    if n_cases == 1:
        axes = axes[None, :]

    # Reconstruct GT (per-method datasets share the canonical frame; GT coeffs are
    # actual stored ghd, denormalized via the same stats used by methods[0]).
    canonical_faces = canonical.faces_packed().detach().cpu().numpy()

    def _recon(coef_144x3):
        m = _ghd_to_mesh(ghd_module, coef_144x3.to(device))
        v = m.verts_packed().detach().cpu().numpy()
        f = m.faces_packed().detach().cpu().numpy()
        return v, f

    for ci, case in enumerate(cases):
        case_dir = os.path.join(args.out_dir, case)
        os.makedirs(case_dir, exist_ok=True)
        gt_coef = _denorm_to_coeff(gt_norm[ci:ci+1], ds)[0]   # [144, 3]
        gv, gf = _recon(gt_coef)
        save_obj(os.path.join(case_dir, "gt.obj"),
                 verts=torch.from_numpy(gv), faces=torch.from_numpy(gf))
        _plot_mesh(axes[ci, 0], gv, gf, f"{case[:18]}\nGT", COLORS["GT"])

        for mi, m in enumerate(methods):
            tag = m["tag"]; sa = m["sa"]
            v_eff = vessel_full.clone()
            o_eff = ostium.clone()
            if sa.get("no_vessel_pts", False) or sa.get("no_conditioning", False):
                v_eff = torch.zeros_like(v_eff)
            if sa.get("no_conditioning", False):
                o_eff = torch.zeros_like(o_eff)
            cond = m["cond"](v_eff[ci:ci+1], o_eff[ci:ci+1])  # [1, cond_dim]
            with torch.no_grad():
                S = m["sample"](cond, 1, args)                 # [1, 1, 432]
            x = S[0, 0]
            # Methods other than baseline produce normalized-space samples.
            # Baseline already in *orig* space when ck has orig_ghd_mean → ds was
            # set to orig stats so it's consistent; either way, denorm via ds.
            coef = _denorm_to_coeff(x.unsqueeze(0), ds)[0]
            sv, sf = _recon(coef)
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
