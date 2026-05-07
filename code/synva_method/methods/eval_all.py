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
from methods._common.mesh_loss import CoeffToMesh
from methods.E_collision.collision_loss import (
    pairwise_min_dist, build_sac_mask, load_opening_idx,
)
from train_vessel_flow_matching import build_conditioner, condition_from_batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cases_file", required=True)
    p.add_argument("--A", default=None, help="Method A (PCA+Flow) ckpt")
    p.add_argument("--B", default=None, help="Method B (MoG-Prior CVAE) ckpt")
    p.add_argument("--C", default=None, help="Method C (FSQ-VAE+AR) ckpt")
    p.add_argument("--D", default=None, help="Method D (VQ-VAE+AR) ckpt")
    p.add_argument("--E", default=None, help="Method E (collision-aware D/C) ckpt")
    p.add_argument("--W", default=None, help="Method W (works ConditionalGHDVAE) ckpt")
    p.add_argument("--baseline", default=None, help="Baseline v8_resnet (PCA-CVAE) ckpt")
    p.add_argument("--num_samples", type=int, default=1,
                   help="Samples drawn per case. Primary metrics (vert_mse, ostium_chamfer, "
                        "viol_frac, single_sample_rmse) are reported on the FIRST sample only "
                        "to mirror real inference (no GT). best_of_K is reported as an oracle "
                        "upper bound only when --num_samples > 1.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0, help="0 = disabled")
    p.add_argument("--flow_steps", type=int, default=32)
    p.add_argument("--flow_sampler", choices=["euler", "heun"], default="heun")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", default=None)
    p.add_argument("--collision_clearance", type=float, default=0.10,
                   help="Clearance (ghd_local units) for test-time violation reporting.")
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
        num_ostium_pts=int(sa.get("num_ostium_pts", 64)),
        num_label2_pts=int(sa.get("num_label2_pts", 256)),
        ring_points=int(sa.get("ring_points", 20)),
        canonical_opa_checkpoint=sa.get(
            "canonical_opa_checkpoint",
            "/path/to/SynVA-A1/checkpoints/canonical_average/opa_checkpoint_1op.pkl",
        ),
        ostium_source=sa.get("ostium_source", "label1"),
        condition_space=sa.get("condition_space", "ghd_local"),
        aligned_data_root=sa.get("aligned_data_root",
                                 "/path/to/ghd_prepared_meshes_3_aneurysm_1op_new"),
        canonical_mesh=sa.get("canonical_mesh",
                              "/path/to/SynVA-A1/checkpoints/canonical_average/part_aligned.obj"),
        canonical_norm_factor=float(sa.get("canonical_norm_factor", 1.10)),
        condition_data_mode=sa.get("condition_data_mode", "prepared"),
        morphology_root=sa.get("morphology_root", None),
        morphology_keys=sa.get("morphology_keys", None),
        withscale=bool(sa.get("withscale", False)),
        normalize=True,
    )
    return ds


def _copy_stats(ds, ck):
    for n in ("ghd_mean", "ghd_std", "ostium_mean", "ostium_std",
              "ostium_ring_mean", "ostium_ring_std",
              "vessel_center", "vessel_scale",
              "morphology_mean", "morphology_std", "morphology_feature_names"):
        if n in ck and ck[n] is not None:
            if n == "morphology_feature_names":
                setattr(ds, n, ck[n])
                continue
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
    cond_net = build_conditioner(sa, device).eval()
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
    input_dim = int(sa.get("input_dim", ck.get("ghd_mean", torch.empty(1, 432)).shape[-1]))
    model = VesselAwareCVAEMoGPrior(
        input_dim=input_dim,
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
    cond_net = build_conditioner(sa, device).eval()
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
    cond_net = build_conditioner(sa, device).eval()
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


def _load_W(path, device):
    """vessel-mesh-editing-master ConditionalGHDVAE baseline."""
    from methods.W_cond_ghd_vae.model import ConditionalGHDVAE
    from train_vessel_flow_matching import build_conditioner

    def _infer_external_hparams(state_dict):
        hidden_dim = int(state_dict["fc1.weight"].shape[0])
        input_dim = int(state_dict["fc1.weight"].shape[1])
        if "cond_encoder.2.weight" in state_dict:
            cond_embed_dim = int(state_dict["cond_encoder.2.weight"].shape[0])
            latent_dim = int(state_dict["fc3.weight"].shape[1] - cond_embed_dim)
        else:
            latent_dim = int(state_dict["fc21.weight"].shape[0])
            cond_embed_dim = int(state_dict["fc3.weight"].shape[1] - latent_dim)
        if "res1.bn1.running_mean" in state_dict:
            norm_type = "batch"
        elif "res1.bn1.weight" in state_dict:
            norm_type = "layer"
        else:
            norm_type = "none"
        return input_dim, hidden_dim, latent_dim, cond_embed_dim, norm_type

    class RingConditioner(torch.nn.Module):
        def forward(self, vessel_pts, ostium_params, ostium_pts=None, ostium_ring=None):
            if ostium_ring is None:
                raise ValueError("Method W requires ostium_ring")
            return ostium_ring.reshape(ostium_ring.shape[0], -1)

    ck = torch.load(path, map_location=device, weights_only=False)
    is_external = "generator" in ck and "target_mean" in ck and "cond_mean" in ck
    if is_external:
        state = ck["generator"]
        input_dim, hidden_dim, latent_dim, cond_embed_dim, norm_type = _infer_external_hparams(state)
        ring_points = int(ck["cond_mean"].shape[1] // 3)
        sa = {
            "ghd_chk_root": "/path/to/aneug-ghds/data/ghd_fitting",
            "ghd_run": "vanilla",
            "ghd_chk_name": "ghb_fitting_checkpoint.pkl",
            "data_root": "__alignment_vessel__",
            "aligned_data_root": "/path/to/aneug-ghds/data/alignment",
            "condition_space": "raw",
            "condition_data_mode": "alignment_vessel",
            "canonical_mesh": "/path/to/aneug-ghds/data/alignment/canonical_model/part_aligned.obj",
            "eigen_chk": "/path/to/aneug-ghds/data/alignment/canonical_model/canonical_model_144_normed.pkl",
            "canonical_opa_checkpoint": "/path/to/aneug-ghds/data/alignment/canonical_model/opa_checkpoint.pkl",
            "ostium_source": "opa_checkpoint",
            "ring_points": ring_points,
            "num_vessel_pts": 256,
            "num_ostium_pts": 64,
            "input_dim": input_dim,
            "withscale": input_dim > 432,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "cond_embed_dim": cond_embed_dim,
            "norm_type": norm_type,
        }
        # Make the generic eval dataset use the original checkpoint's stats.
        ck["saved_args"] = sa
        ck["ghd_mean"] = ck["target_mean"][:, :input_dim].detach().cpu()
        ck["ghd_std"] = ck["target_std"][:, :input_dim].detach().cpu()
        ck["ostium_ring_mean"] = ck["cond_mean"].detach().cpu()
        ck["ostium_ring_std"] = ck["cond_std"].detach().cpu()
    else:
        state = ck["model_state_dict"]
        sa = _sa(ck)
        input_dim = int(sa.get("input_dim", ck.get("ghd_mean", torch.empty(1, 432)).shape[-1]))
        hidden_dim = int(sa.get("hidden_dim", 384))
        latent_dim = int(sa.get("latent_dim", 64))
        ring_points = int(sa.get("ring_points", 20))
        cond_embed_dim = int(sa.get("cond_embed_dim", 128))
        norm_type = sa.get("norm_type", "batch")
    condition_mode = str(sa.get("condition_mode", "ring"))
    if condition_mode == "vessel":
        cond_dim = int(sa.get("vessel_cond_dim", 32))
    else:
        cond_dim = ring_points * 3
    if bool(sa.get("use_morphology_condition", False)):
        morph_dim = int(sa.get("morphology_dim", 0))
        if morph_dim <= 0:
            mean = ck.get("morphology_mean", None)
            morph_dim = int(mean.shape[-1]) if mean is not None else 0
        if morph_dim <= 0:
            raise ValueError("W checkpoint uses morphology condition but morphology_dim/statistics are missing")
        cond_dim += morph_dim

    model = ConditionalGHDVAE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        cond_embed_dim=cond_embed_dim,
        norm_type=norm_type,
    ).to(device).eval()
    model.load_state_dict(state)
    if condition_mode == "vessel":
        cond_net = build_conditioner(sa, device).eval()
        cond_state = ck.get("conditioner_state_dict", None)
        if cond_state is None:
            raise ValueError("W checkpoint declares condition_mode='vessel' but has no conditioner_state_dict")
        cond_net.load_state_dict(cond_state)
    else:
        cond_net = RingConditioner().to(device).eval()
    if bool(sa.get("use_morphology_condition", False)):
        setattr(cond_net, "use_morphology_condition", True)

    def sample(cond, K, args):
        cond_rep = cond.repeat_interleave(K, dim=0)
        z = torch.randn(cond_rep.shape[0], latent_dim, device=device) * float(args.temperature)
        x = model.decode(z, cond_rep)
        return x.view(cond.size(0), K, -1).transpose(0, 1)

    return ck, sa, cond_net, sample


# Method E reuses the D-style checkpoint layout (vqvae + ar_prior + conditioner).
_load_E = _load_D

METHOD_LOADERS = dict(A=_load_A, B=_load_B, C=_load_C, D=_load_D, E=_load_E, W=_load_W, baseline=_load_baseline)


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    methods = []
    for tag in ("A", "B", "C", "D", "E", "W", "baseline"):
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

    # Set up shared mesh decoder + collision-eval helpers (uses first method's frame).
    sa0 = methods[0]["sa"]
    canonical_norm = float(sa0.get("canonical_norm_factor", 1.10))
    canonical_mesh_path = sa0.get(
        "canonical_mesh",
        "/path/to/SynVA-A1/checkpoints/canonical_average/part_aligned.obj",
    )
    eigen_chk = sa0.get(
        "eigen_chk",
        "/path/to/SynVA-A1/checkpoints/canonical_average/eigen_chk_144.pkl",
    )
    c2m = CoeffToMesh(canonical_mesh_path, eigen_chk,
                      num_basis=int(sa0.get("num_basis", 144)),
                      device=device,
                      canonical_norm_factor=canonical_norm)
    opening_idx = load_opening_idx(canonical_mesh_path).to(device)
    V = c2m.canonical.verts_packed().shape[0]
    sac_mask = build_sac_mask(V, opening_idx).to(device)
    clearance = float(args.collision_clearance)

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

        per_case_best = []        # ORACLE upper bound (uses GT to pick best of K)
        per_case_recon = []       # mean RMSE across K samples (diversity proxy, GT-aware)
        per_case_single = []      # honest single-sample RMSE on S[0] (no GT peeking)
        per_case_viol = []        # fraction sac verts with d<clearance to vessel pts (S[0])
        per_case_vert = []        # vertex-MSE in mesh frame (S[0])  [10^-3]
        per_case_chamfer = []     # symmetric chamfer pred-ring vs GT-ring (S[0])
        # Vessel center/scale are dataset-global (set in VesselAwareGHDDataset).
        v_center = ds.vessel_center.to(device) if hasattr(ds, "vessel_center") else None
        v_scale  = ds.vessel_scale.to(device) if hasattr(ds, "vessel_scale") else None
        with torch.no_grad():
            for batch in loader:
                gt = batch["ghd"].to(device)            # [B, 432]
                cond = condition_from_batch(
                    m["cond"],
                    batch,
                    device,
                    zero_vessel=bool(sa.get("no_vessel_pts", False)),
                    zero_all=bool(sa.get("no_conditioning", False)),
                )
                S = m["sample"](cond, K, args)            # [K, B, 432]
                rmse = torch.sqrt(((S - gt.unsqueeze(0)) ** 2).mean(-1) + 1e-8)  # [K, B]
                per_case_best.append(rmse.min(0).values.cpu())   # ORACLE
                per_case_recon.append(rmse.mean(0).cpu())        # diversity
                per_case_single.append(rmse[0].cpu())            # honest single-sample

                # Decode FIRST sample + GT in mesh frame; this is the honest
                # inference-time prediction (no GT peeking).
                B = gt.shape[0]
                ghd_mean = ds.ghd_mean.to(device)
                ghd_std  = ds.ghd_std.to(device)
                S_first = S[0]                                                   # [B, 432]
                v_pred, _ = c2m(S_first, ghd_mean, ghd_std, want_normals=False)  # [B, V, 3]
                v_gt,   _ = c2m(gt,     ghd_mean, ghd_std, want_normals=False)
                vert_mse = ((v_pred - v_gt) ** 2).mean(dim=(1, 2))               # [B]
                per_case_vert.append(vert_mse.cpu())

                # Ostium-treffgenauigkeit: symmetric chamfer between the predicted
                # sac's opening ring (decoded vertices at op_v_indices) and the GT
                # ostium ring (FPS-subsampled, in GHD-local mesh frame).
                if "ostium_pts" in batch and v_center is not None and v_scale is not None:
                    pred_ring = v_pred[:, opening_idx, :]                          # [B, R, 3]
                    gt_ring_norm = batch["ostium_pts"].to(device)                  # [B, K, 3] normalised
                    gt_ring = gt_ring_norm * v_scale + v_center                    # mesh frame
                    # symmetric mean chamfer
                    dpw = torch.cdist(pred_ring, gt_ring)                          # [B, R, K]
                    d_p2g = dpw.min(dim=2).values.mean(dim=1)
                    d_g2p = dpw.min(dim=1).values.mean(dim=1)
                    chamfer = 0.5 * (d_p2g + d_g2p)                                # [B]
                    per_case_chamfer.append(chamfer.cpu())

                # Denormalize vessel_pts back to GHD-local mesh frame and measure
                # vessel-collision violation regardless of whether the model used
                # vessel_pts as conditioning (this is a *test-set* property, not a
                # training-config property).
                if v_center is not None and v_scale is not None:
                    raw_vessel = batch["vessel_pts"].to(device)
                    v_pts_mesh = raw_vessel * v_scale + v_center  # [B, N, 3]
                    sac_v = v_pred[:, sac_mask, :]
                    d = pairwise_min_dist(sac_v, v_pts_mesh, chunk=4096)         # [B, M]
                    viol = (d < clearance).float().mean(dim=1)                   # [B]
                    per_case_viol.append(viol.cpu())

        b = torch.cat(per_case_best).mean().item()
        s = torch.cat(per_case_recon).mean().item()
        single = torch.cat(per_case_single).mean().item()
        vert = torch.cat(per_case_vert).mean().item() if per_case_vert else float("nan")
        viol_v = (torch.cat(per_case_viol).mean().item()
                  if per_case_viol else float("nan"))
        chamfer_v = (torch.cat(per_case_chamfer).mean().item()
                     if per_case_chamfer else float("nan"))
        results[m["tag"]] = {"single_sample_rmse": single,
                             "oracle_best_of_K": b, "mean_to_gt": s,
                             "vert_mse": vert, "viol_frac": viol_v,
                             "ostium_chamfer": chamfer_v,
                             "K": K, "ckpt": m["path"],
                             # legacy field name (== single_sample_rmse for K=1 only)
                             "best_of_K": b}
        print(f"[{m['tag']:>8s}] single_sample_rmse={single:.4f}  "
              f"vert_mse={vert:.5f}  viol@{clearance:.2f}={viol_v*100:.2f}%  "
              f"ostium_chamfer={chamfer_v:.4f}  (oracle_best_of_{K}={b:.4f})")

    print("\n=== Summary (single_sample_rmse; lower is better; oracle in parens) ===")
    for tag, r in sorted(results.items(), key=lambda kv: kv[1]["single_sample_rmse"]):
        print(f"  {tag:>10s}  single={r['single_sample_rmse']:.4f}  "
              f"vert_mse={1000*r['vert_mse']:.3f}e-3  "
              f"ostium_chamfer={r['ostium_chamfer']:.4f}  "
              f"(oracle_best_of_{K}={r['oracle_best_of_K']:.4f})")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        json.dump({"K": K, "cases_file": args.cases_file, "results": results},
                  open(args.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
