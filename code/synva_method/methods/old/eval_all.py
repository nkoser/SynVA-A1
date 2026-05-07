#!/usr/bin/env python
"""Unified best-of-K evaluation for methods A/B/C/D + (optional) baseline CVAE.

For each method, samples K conditional GHDs per val case from p(GHD|ostium)
and reports mean per-case best-of-K RMSE in 432-D normalized GHD space.

Usage:
  python methods/eval_all.py \
    --cases_file checkpoints/vessel_aware_cvae/splits_finish_v5_only3999_full_20260429/cases_val.json \
    --A checkpoints/methods/A_pca_flow_matching/.../models_best_val.pth \
    --B checkpoints/methods/B_mog_prior_cvae/.../models_best_val.pth \
    --C checkpoints/methods/C_fsq_ar/.../best.pt \
    --D checkpoints/methods/D_vq_transformer/.../best.pt \
    --num_samples 32 --device cuda --out_json out.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)

from first_stage_vessel_aware import collate_fn
from models.vae_datasets_vessel import VesselAwareGHDDataset
from models.vessel_conditioner import OstiumConditioner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cases_file", required=True)
    p.add_argument("--A", default=None, help="Method A (PCA+Flow) ckpt")
    p.add_argument("--B", default=None, help="Method B (MoG-Prior CVAE) ckpt")
    p.add_argument("--C", default=None, help="Method C (FSQ-VAE+AR) ckpt")
    p.add_argument("--D", default=None, help="Method D (VQ-VAE+AR) ckpt")
    p.add_argument("--baseline", default=None, help="Baseline v8_resnet (PCA-CVAE) ckpt")
    p.add_argument("--num_samples", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0, help="0 = disabled")
    p.add_argument("--flow_steps", type=int, default=32)
    p.add_argument("--flow_sampler", choices=["euler", "heun"], default="heun")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", default=None)
    return p.parse_args()


def _sa(ck):
    """Return merged saved_args ∪ args dict (saved_args wins on duplicates)."""
    out = {}
    for key in ("args", "saved_args"):  # args first, saved_args overrides
        a = ck.get(key, None)
        if a is None:
            continue
        if not isinstance(a, dict):
            a = vars(a)
        out.update(a)
    return out


def _build_dataset(cases, sa):
    ds = VesselAwareGHDDataset(
        ghd_chk_root=sa.get("ghd_chk_root"),
        ghd_run=sa.get("ghd_run"),
        ghd_chk_name=sa.get("ghd_chk_name", "ghb_fitting_checkpoint.pkl"),
        data_root=sa.get("data_root", "/path/to/prepared_meshes_3"),
        cases=cases,
        num_vessel_pts=int(sa.get("num_vessel_pts", 256)),
        condition_space=sa.get("condition_space", "ghd_local"),
        aligned_data_root=sa.get("aligned_data_root",
                                 "/path/to/ghd_prepared_meshes_3_aneurysm_1op_new"),
        canonical_mesh=sa.get("canonical_mesh",
                              "/path/to/SynVA-A1/checkpoints/canonical_average/part_aligned.obj"),
        canonical_norm_factor=float(sa.get("canonical_norm_factor", 1.10)),
        normalize=True,
    )
    return ds


def _copy_stats(ds, ck):
    for n in ("ghd_mean", "ghd_std", "ostium_mean", "ostium_std",
              "vessel_center", "vessel_scale"):
        if n in ck:
            setattr(ds, n, ck[n].cpu())


# ----- Per-method sampler factories ------------------------------------------

def _load_A(path, device):
    """PCA + Flow Matching."""
    from models.vessel_aware_flow_matching import VesselAwareFlowMatching
    ck = torch.load(path, map_location=device, weights_only=False)
    sa = _sa(ck)
    pca_mean = ck["pca_mean"].to(device)
    pca_comp = ck["pca_components"].to(device)
    pca_scale = ck["pca_score_scale"].to(device)
    pca_dim = int(ck.get("pca_dim_actual", pca_comp.shape[0]))
    cond_net = OstiumConditioner(
        vessel_feat_dim=int(sa.get("vessel_feat_dim", 64)),
        ostium_plane_dim=8, ostium_feat_dim=16,
        cond_out_dim=int(sa.get("vessel_cond_dim", 32))).to(device).eval()
    cond_net.load_state_dict(ck["conditioner"])
    flow = VesselAwareFlowMatching(
        input_dim=pca_dim,
        hidden_dim=int(sa.get("hidden_dim", 256)),
        cond_dim=int(sa.get("vessel_cond_dim", 32)),
        time_dim=int(sa.get("time_dim", 64)),
        blocks=int(sa.get("flow_blocks", 4)),
        dropout=float(sa.get("dropout", 0.02)),
    ).to(device).eval()
    flow.load_state_dict(ck["flow_model"])

    def sample(cond, K, args):
        cond_rep = cond.repeat_interleave(K, dim=0)
        z = flow.sample(cond_rep, num_steps=args.flow_steps,
                        temperature=args.temperature, method=args.flow_sampler)
        x = (z * pca_scale) @ pca_comp + pca_mean  # [B*K, 432] in normalized GHD space
        return x.view(cond.size(0), K, -1).transpose(0, 1)  # [K, B, 432]
    return ck, sa, cond_net, sample


def _load_B(path, device):
    """MoG-Prior CVAE."""
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

    def sample(cond, K, args):
        cond_rep = cond.repeat_interleave(K, dim=0)
        z = model.sample_prior(cond_rep)
        x = model.decode(z, cond_rep)
        return x.view(cond.size(0), K, -1).transpose(0, 1)
    return ck, sa, cond_net, sample


def _load_C(path, device):
    from methods.C_fsq_ar.sample import load as cload, sample_ghd as csample
    fsq, cond_net, ar, ck, _ = cload(path, device)
    sa = _sa(ck)

    def sample(cond, K, args):
        x = csample(fsq, ar, cond, num_samples=K,
                    temperature=args.temperature,
                    top_k=(args.top_k or None))
        return x.view(cond.size(0), K, -1).transpose(0, 1)
    return ck, sa, cond_net, sample


def _load_D(path, device):
    from methods.D_vq_transformer.sample import load as dload, sample_ghd as dsample
    vq, cond_net, ar, ck, _ = dload(path, device)
    sa = _sa(ck)

    def sample(cond, K, args):
        x = dsample(vq, ar, cond, num_samples=K,
                    temperature=args.temperature,
                    top_k=(args.top_k or None))
        return x.view(cond.size(0), K, -1).transpose(0, 1)
    return ck, sa, cond_net, sample


def _load_baseline(path, device):
    """v8_resnet PCA-CVAE (existing baseline)."""
    from models.vessel_aware_cvae_v8_resnet import VesselAwareCVAEV8ResNet
    from models.vessel_aware_cvae import VesselAwareCVAE as VesselAwareCVAEV2
    ck = torch.load(path, map_location=device, weights_only=False)
    sa = _sa(ck)
    pca_basis = ck.get("pca_basis", None)  # [K, 432]
    if pca_basis is not None:
        input_dim = pca_basis.shape[0]
        pca_basis = pca_basis.to(device)
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
    model.load_state_dict(ck["generator"])
    model.to(device).eval()
    cond_net = OstiumConditioner(
        vessel_feat_dim=int(sa.get("vessel_feat_dim", 64)),
        ostium_plane_dim=8, ostium_feat_dim=16,
        cond_out_dim=int(sa.get("vessel_cond_dim", 32))).to(device).eval()
    cond_net.load_state_dict(ck["conditioner"])
    use_cp = bool(sa.get("use_conditional_prior", False)) and hasattr(model, "prior")
    L = int(sa.get("latent_dim", 64))

    def sample(cond, K, args):
        cond_rep = cond.repeat_interleave(K, dim=0)
        if use_cp:
            mu, lv = model.prior(cond_rep)
            std = (0.5 * lv).exp()
        else:
            mu = torch.zeros(cond_rep.size(0), L, device=device)
            std = torch.ones_like(mu)
        z = mu + std * torch.randn_like(mu)
        x = model.decode(z, cond_rep)
        if pca_basis is not None:
            x = x @ pca_basis  # PCA -> 432 (normalized in checkpoint's *orig* space)
        return x.view(cond.size(0), K, -1).transpose(0, 1)
    return ck, sa, cond_net, sample


METHOD_LOADERS = dict(A=_load_A, B=_load_B, C=_load_C, D=_load_D, baseline=_load_baseline)


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    methods = []
    for tag in ("A", "B", "C", "D", "baseline"):
        path = getattr(args, tag)
        if path:
            print(f"[load] {tag}: {path}")
            ck, sa, cond_net, sample = METHOD_LOADERS[tag](path, device)
            methods.append({"tag": tag, "ckpt": ck, "sa": sa,
                            "cond": cond_net, "sample": sample, "path": path})
    if not methods:
        raise SystemExit("No method ckpts given")

    cases = json.load(open(args.cases_file))
    print(f"[data] {len(cases)} cases")

    results = {}
    K = args.num_samples
    for m in methods:
        ck, sa = m["ckpt"], m["sa"]
        ds = _build_dataset(cases, sa)
        _copy_stats(ds, ck)
        # Some baseline ckpts store orig stats separately; PCA-baseline reconstructs in
        # *orig* space — already what we want.
        if "orig_ghd_mean" in ck:
            ds.ghd_mean = ck["orig_ghd_mean"].cpu()
            ds.ghd_std = ck["orig_ghd_std"].cpu()
        loader = DataLoader(ds, batch_size=200, shuffle=False, collate_fn=collate_fn)

        per_case_best = []
        per_case_recon = []  # naive reconstruction floor: K=1, t=0 sample dispersion proxy
        with torch.no_grad():
            for batch in loader:
                gt = batch["ghd"].to(device)            # [B, 432]
                ostium = batch["ostium_params"].to(device)
                vessel = batch["vessel_pts"].to(device)
                # Match training behavior: zero vessel pts if model trained without them.
                if sa.get("no_vessel_pts", False) or sa.get("no_conditioning", False):
                    vessel = torch.zeros_like(vessel)
                if sa.get("no_conditioning", False):
                    ostium = torch.zeros_like(ostium)
                cond = m["cond"](vessel, ostium)
                S = m["sample"](cond, K, args)            # [K, B, 432]
                rmse = torch.sqrt(((S - gt.unsqueeze(0)) ** 2).mean(-1) + 1e-8)  # [K, B]
                per_case_best.append(rmse.min(0).values.cpu())
                per_case_recon.append(rmse.mean(0).cpu())

        b = torch.cat(per_case_best).mean().item()
        s = torch.cat(per_case_recon).mean().item()
        results[m["tag"]] = {"best_of_K": b, "mean_to_gt": s,
                             "K": K, "ckpt": m["path"]}
        print(f"[{m['tag']:>8s}] best_of_K={b:.4f}  mean_s_to_gt={s:.4f}")

    print("\n=== Summary (best_of_K val RMSE; lower is better) ===")
    for tag, r in sorted(results.items(), key=lambda kv: kv[1]["best_of_K"]):
        print(f"  {tag:>10s}  best_of_K={r['best_of_K']:.4f}  mean={r['mean_to_gt']:.4f}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        json.dump({"K": K, "cases_file": args.cases_file, "results": results},
                  open(args.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
