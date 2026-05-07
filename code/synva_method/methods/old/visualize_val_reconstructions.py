#!/usr/bin/env python
"""Visualize per-method val *reconstructions* alongside ground-truth.

Reconstruction = encode GT, then decode (no random sampling). This isolates
the autoencoder/quantizer floor from the prior/sampler quality.

  A (PCA+FlowMatching)  : project GT -> PCA basis -> lift back     (PCA recon floor)
  B (MoG-prior CVAE)    : encode(x,cond) mu -> decode               (CVAE recon)
  C (FSQ-VAE+AR)        : encode -> quantize -> decode              (FSQ recon)
  D (VQ-VAE+AR)         : encode -> quantize -> decode              (VQ  recon)
  baseline (PCA-CVAE)   : project to PCA -> encode mu -> decode -> lift PCA

Outputs OBJ files per case + an overview PNG.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)

from pytorch3d.io import save_obj, load_objs_as_meshes
from ghd.base.graph_harmonic_deformation import Graph_Harmonic_Deform
from methods.eval_all import _sa, _build_dataset, _copy_stats
from first_stage_vessel_aware import collate_fn
from models.vessel_conditioner import OstiumConditioner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cases_file", required=True)
    p.add_argument("--A", default=None); p.add_argument("--B", default=None)
    p.add_argument("--C", default=None); p.add_argument("--D", default=None)
    p.add_argument("--baseline", default=None)
    p.add_argument("--n_cases", type=int, default=10)
    p.add_argument("--canonical_mesh", default="/path/to/SynVA-A1/checkpoints/canonical_average/part_aligned.obj")
    p.add_argument("--eigen_chk", default="/path/to/SynVA-A1/checkpoints/canonical_average/eigen_chk_144.pkl")
    p.add_argument("--num_basis", type=int, default=144)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


# ---- per-method "reconstruct from GT" closures --------------------------------

def _load_A_recon(path, device):
    """PCA reconstruction: project normalized GT through PCA basis."""
    ck = torch.load(path, map_location=device, weights_only=False)
    sa = _sa(ck)
    pca_mean = ck["pca_mean"].to(device)
    pca_comp = ck["pca_components"].to(device)  # [K, 432]
    cond_net = OstiumConditioner(
        vessel_feat_dim=int(sa.get("vessel_feat_dim", 64)),
        ostium_plane_dim=8, ostium_feat_dim=16,
        cond_out_dim=int(sa.get("vessel_cond_dim", 32))).to(device).eval()
    cond_net.load_state_dict(ck["conditioner"])

    def recon(gt_norm, cond):
        z = (gt_norm - pca_mean) @ pca_comp.T   # project
        return z @ pca_comp + pca_mean          # lift
    return ck, sa, cond_net, recon


def _load_B_recon(path, device):
    from methods.B_mog_prior_cvae.model import VesselAwareCVAEMoGPrior
    ck = torch.load(path, map_location=device, weights_only=False)
    sa = _sa(ck)
    model = VesselAwareCVAEMoGPrior(
        input_dim=432,
        hidden_dim=int(sa.get("hidden_dim", 384)),
        latent_dim=int(sa.get("latent_dim", 64)),
        vessel_cond_dim=int(sa.get("vessel_cond_dim", 32)),
        extra_cond_dim=0,
        dropout=float(sa.get("dropout", 0.02)),
        encoder_blocks=int(sa.get("encoder_blocks", 3)),
        decoder_blocks=int(sa.get("decoder_blocks", 6)),
        mog_components=int(sa.get("mog_components", 8)),
    ).to(device).eval()
    model.load_state_dict(ck["generator"])
    cond_net = OstiumConditioner(
        vessel_feat_dim=int(sa.get("vessel_feat_dim", 64)),
        ostium_plane_dim=8, ostium_feat_dim=16,
        cond_out_dim=int(sa.get("vessel_cond_dim", 32))).to(device).eval()
    cond_net.load_state_dict(ck["conditioner"])

    def recon(gt_norm, cond):
        mu, _ = model.encode(gt_norm, cond)
        return model.decode(mu, cond)
    return ck, sa, cond_net, recon


def _load_C_recon(path, device):
    from methods.C_fsq_ar.sample import load as cload
    fsq, cond_net, ar, ck, sa = cload(path, device)
    sa = _sa(ck)

    def recon(gt_norm, cond):
        z_q, _ = fsq.encode(gt_norm, cond)
        return fsq.decode(z_q, cond)
    return ck, sa, cond_net, recon


def _load_D_recon(path, device):
    from methods.D_vq_transformer.sample import load as dload
    vq, cond_net, ar, ck, sa = dload(path, device)
    sa = _sa(ck)

    def recon(gt_norm, cond):
        out = vq.encode(gt_norm, cond)
        z_q = out[0]
        return vq.decode(z_q, cond)
    return ck, sa, cond_net, recon


def _load_baseline_recon(path, device):
    from models.vessel_aware_cvae_v8_resnet import VesselAwareCVAEV8ResNet
    from models.vessel_aware_cvae import VesselAwareCVAE as VesselAwareCVAEV2
    ck = torch.load(path, map_location=device, weights_only=False)
    sa = _sa(ck)
    pca_basis = ck.get("pca_basis", None)
    if pca_basis is not None:
        input_dim = pca_basis.shape[0]
        pca_basis = pca_basis.to(device)  # [K, 432]
    else:
        input_dim = 432
    common = dict(
        input_dim=input_dim,
        hidden_dim=int(sa.get("hidden_dim", 256)),
        latent_dim=int(sa.get("latent_dim", 64)),
        vessel_cond_dim=int(sa.get("vessel_cond_dim", 32)),
        extra_cond_dim=0,
        dropout=float(sa.get("dropout", 0.02)),
    )
    if sa.get("model_type") == "v8_resnet":
        model = VesselAwareCVAEV8ResNet(
            **common,
            encoder_blocks=int(sa.get("encoder_blocks", 3)),
            decoder_blocks=int(sa.get("decoder_blocks", 6)))
    else:
        model = VesselAwareCVAEV2(**common)
    model.load_state_dict(ck["generator"]); model.to(device).eval()
    cond_net = OstiumConditioner(
        vessel_feat_dim=int(sa.get("vessel_feat_dim", 64)),
        ostium_plane_dim=8, ostium_feat_dim=16,
        cond_out_dim=int(sa.get("vessel_cond_dim", 32))).to(device).eval()
    cond_net.load_state_dict(ck["conditioner"])

    def recon(gt_norm, cond):
        # gt_norm is in *orig* (normalized) 432-D space (ds.ghd_mean/std come from
        # ck's orig_ghd_mean/std). Project to PCA, encode, decode, lift back.
        if pca_basis is None:
            x_in = gt_norm
        else:
            x_in = gt_norm @ pca_basis.T  # 432 -> K
        mu, _ = model.encode(x_in, cond)
        out = model.decode(mu, cond)
        if pca_basis is not None:
            out = out @ pca_basis  # K -> 432
        return out
    return ck, sa, cond_net, recon


METHOD_LOADERS = dict(A=_load_A_recon, B=_load_B_recon, C=_load_C_recon,
                      D=_load_D_recon, baseline=_load_baseline_recon)


# ---- mesh helpers -------------------------------------------------------------

def _denorm_to_coeff(x_normalized, ds):
    mean = ds.ghd_mean.to(x_normalized.device).view(-1)
    std = ds.ghd_std.to(x_normalized.device).view(-1)
    raw = x_normalized * std + mean
    return raw.view(*raw.shape[:-1], 144, 3)


def _ghd_to_mesh(ghd_module, coeff_144x3):
    return ghd_module.forward(coeff_144x3)


def _plot_mesh(ax, verts, faces, title, color):
    ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                    triangles=faces, edgecolor=(0, 0, 0, 0.15),
                    linewidth=0.05, color=color, alpha=0.85)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=18, azim=35)
    rng = max(np.ptp(verts, axis=0))
    mid = verts.mean(axis=0); half = rng * 0.55
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
    print(f"[viz] reconstruction for {len(cases)}/{len(cases_full)} val cases")

    methods = []
    for tag in ("A", "B", "C", "D", "baseline"):
        path = getattr(args, tag)
        if not path: continue
        print(f"[load] {tag}: {path}")
        ck, sa, cond_net, recon = METHOD_LOADERS[tag](path, device)
        methods.append({"tag": tag, "ckpt": ck, "sa": sa,
                        "cond": cond_net, "recon": recon, "path": path})
    if not methods:
        raise SystemExit("No method ckpts given")

    canonical = load_objs_as_meshes([args.canonical_mesh]).to(device)
    ghd_module = Graph_Harmonic_Deform(base_shape=canonical,
                                       num_Basis=args.num_basis,
                                       eigen_chk=args.eigen_chk)
    import torch.nn as nn
    setattr(ghd_module, "R", nn.Parameter(torch.zeros(1, 3, device=device)))
    setattr(ghd_module, "s", nn.Parameter(torch.ones(1, 1, device=device)))
    setattr(ghd_module, "T", nn.Parameter(torch.zeros(1, 3, device=device)))

    # Per-method dataset (each ckpt has its own normalization stats)
    from torch.utils.data import DataLoader
    per_method_data = {}
    for m in methods:
        ck, sa = m["ckpt"], m["sa"]
        ds = _build_dataset(cases, sa)
        _copy_stats(ds, ck)
        if "orig_ghd_mean" in ck:
            ds.ghd_mean = ck["orig_ghd_mean"].cpu()
            ds.ghd_std = ck["orig_ghd_std"].cpu()
        loader = DataLoader(ds, batch_size=len(cases), shuffle=False, collate_fn=collate_fn)
        batch = next(iter(loader))
        per_method_data[m["tag"]] = (ds, batch)

    # Use methods[0] dataset for GT meshes (all ckpts share a canonical frame)
    ds0, batch0 = per_method_data[methods[0]["tag"]]
    gt_norm0 = batch0["ghd"].to(device)

    n_cases = gt_norm0.shape[0]
    n_meth = len(methods)
    cols = 1 + n_meth
    fig, axes = plt.subplots(n_cases, cols,
                             figsize=(2.4 * cols, 2.4 * n_cases),
                             subplot_kw={"projection": "3d"})
    if n_cases == 1: axes = axes[None, :]

    def _recon_mesh(coef_144x3):
        m = _ghd_to_mesh(ghd_module, coef_144x3.to(device))
        v = m.verts_packed().detach().cpu().numpy()
        f = m.faces_packed().detach().cpu().numpy()
        return v, f

    for ci, case in enumerate(cases):
        case_dir = os.path.join(args.out_dir, case)
        os.makedirs(case_dir, exist_ok=True)
        gt_coef = _denorm_to_coeff(gt_norm0[ci:ci+1], ds0)[0]
        gv, gf = _recon_mesh(gt_coef)
        save_obj(os.path.join(case_dir, "gt.obj"),
                 verts=torch.from_numpy(gv), faces=torch.from_numpy(gf))
        _plot_mesh(axes[ci, 0], gv, gf, f"{case[:18]}\nGT", COLORS["GT"])

        for mi, m in enumerate(methods):
            tag, sa = m["tag"], m["sa"]
            ds, batch = per_method_data[tag]
            gt_norm = batch["ghd"].to(device)
            ostium = batch["ostium_params"].to(device)
            vessel = batch["vessel_pts"].to(device)
            if sa.get("no_vessel_pts", False) or sa.get("no_conditioning", False):
                vessel = torch.zeros_like(vessel)
            if sa.get("no_conditioning", False):
                ostium = torch.zeros_like(ostium)
            cond = m["cond"](vessel[ci:ci+1], ostium[ci:ci+1])
            with torch.no_grad():
                xr = m["recon"](gt_norm[ci:ci+1], cond)   # [1, 432] in this method's normalized space
            coef = _denorm_to_coeff(xr, ds)[0]
            sv, sf = _recon_mesh(coef)
            # per-method RMSE in normalized space
            rmse = float(torch.sqrt(((xr - gt_norm[ci:ci+1]) ** 2).mean()))
            save_obj(os.path.join(case_dir, f"{tag}.obj"),
                     verts=torch.from_numpy(sv), faces=torch.from_numpy(sf))
            _plot_mesh(axes[ci, 1 + mi], sv, sf,
                       f"{tag} recon\nrmse={rmse:.3f}", COLORS.get(tag, "tan"))

    plt.tight_layout()
    out_png = os.path.join(args.out_dir, "_overview_recon.png")
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] overview -> {out_png}")


if __name__ == "__main__":
    main()
