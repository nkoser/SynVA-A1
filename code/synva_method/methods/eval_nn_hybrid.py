#!/usr/bin/env python
"""NN-Hybrid evaluator: combine kNN-ostium retrievals with trained-CVAE samples.

For each test case we build a pool of candidate GHDs:
  - K_knn nearest-by-ostium training GHDs (the knn_ostium floor, vert_mse 2.36e-3)
  - K_cvae samples from the supplied CVAE conditioned on test ostium+vessel
  - optional alpha-blends of (cvae_sample[k], knn_sample[k]) for a list of alphas

We report best-of-pool RMSE and vert_mse for several pool definitions:
  - knn_only            (sanity, should match trivial knn_ostium)
  - cvae_only           (sanity, should match plain CVAE eval)
  - knn_plus_cvae       (union of both)
  - knn_plus_cvae_blend (union + alpha blends in alpha_grid)

This is purely a test-time hybrid — no retraining.
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
from methods._common.mesh_loss import CoeffToMesh
from methods.eval_all import _sa, _build_dataset, _copy_stats, METHOD_LOADERS
from methods.eval_trivial_baselines import (
    _gather_ghd_ostium_vessel,
    best_of_k_rmse,
    best_of_k_vert_mse,
)
from train_vessel_flow_matching import condition_from_batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_cases_file", required=True)
    p.add_argument("--test_cases_file", required=True)
    p.add_argument("--cvae_ckpt", required=True,
                   help="Trained CVAE checkpoint (loaded via methods.eval_all baseline loader).")
    p.add_argument("--cvae_tag", default="baseline",
                   choices=list(METHOD_LOADERS.keys()),
                   help="Which method-loader to use for the CVAE ckpt.")
    p.add_argument("--K_knn", type=int, default=16)
    p.add_argument("--K_cvae", type=int, default=16)
    p.add_argument("--knn_key", default="ostium8",
                   choices=["ostium8", "vessel_chamfer", "cond_embed"],
                   help="Retrieval space for kNN: 8-D ostium (default), Chamfer of vessel_pts, or conditioner embedding.")
    p.add_argument("--alpha_grid", default="0.25,0.5,0.75",
                   help="Comma-separated blend weights w in [0,1] for alpha*cvae + (1-alpha)*knn.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--flow_steps", type=int, default=64)
    p.add_argument("--flow_sampler", default="heun")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    print(f"[load] cvae ({args.cvae_tag}): {args.cvae_ckpt}")
    ck, sa, cond_net, sample_fn = METHOD_LOADERS[args.cvae_tag](args.cvae_ckpt, device)

    train_cases = json.load(open(args.train_cases_file))
    test_cases = json.load(open(args.test_cases_file))
    print(f"[data] train={len(train_cases)}  test={len(test_cases)}")

    print("[build] train dataset")
    train_ds = _build_dataset(train_cases, sa)
    _copy_stats(train_ds, ck)
    if "orig_ghd_mean" in ck:
        train_ds.ghd_mean = ck["orig_ghd_mean"].cpu()
        train_ds.ghd_std  = ck["orig_ghd_std"].cpu()
    G_tr, O_tr, V_tr = _gather_ghd_ostium_vessel(train_ds, device)
    train_batch = next(iter(DataLoader(train_ds, batch_size=len(train_ds), shuffle=False, collate_fn=collate_fn)))
    print(f"  train tensors: ghd {tuple(G_tr.shape)}  ost {tuple(O_tr.shape)}")

    print("[build] test dataset (TRAIN stats)")
    test_ds = _build_dataset(test_cases, sa)
    _copy_stats(test_ds, ck)
    if "orig_ghd_mean" in ck:
        test_ds.ghd_mean = ck["orig_ghd_mean"].cpu()
        test_ds.ghd_std  = ck["orig_ghd_std"].cpu()
    G_te, O_te, V_te = _gather_ghd_ostium_vessel(test_ds, device)
    test_batch = next(iter(DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, collate_fn=collate_fn)))
    print(f"  test  tensors: ghd {tuple(G_te.shape)}  ost {tuple(O_te.shape)}")

    Ntr = G_tr.shape[0]
    Nte = G_te.shape[0]

    # Decoder for vertex MSE
    canonical_norm = float(sa.get("canonical_norm_factor", 1.10))
    canonical_mesh = sa.get("canonical_mesh",
                            "/path/to/SynVA-A1/checkpoints/canonical_average/part_aligned.obj")
    eigen_chk = sa.get("eigen_chk",
                       "/path/to/SynVA-A1/checkpoints/canonical_average/eigen_chk_144.pkl")
    c2m = CoeffToMesh(canonical_mesh, eigen_chk,
                      num_basis=int(sa.get("num_basis", 144)),
                      device=device,
                      canonical_norm_factor=canonical_norm)
    ghd_mean = train_ds.ghd_mean.to(device)
    ghd_std  = train_ds.ghd_std.to(device)

    # ---- kNN retrievals: find K_knn nearest train cases for each test case ----
    if args.knn_key == "ostium8":
        d_key = torch.cdist(O_te, O_tr)                   # [Nte, Ntr]
    elif args.knn_key == "vessel_chamfer":
        # symmetric mean Chamfer between 256-point vessel clouds
        # V_te: [Nte, P, 3], V_tr: [Ntr, P, 3]; chunked over test for memory
        Nte_, P, _ = V_te.shape
        Ntr_ = V_tr.shape[0]
        d_key = torch.empty(Nte_, Ntr_, device=device)
        with torch.no_grad():
            for i in range(Nte_):
                # [P, Ntr*P] would be huge; loop train in chunks
                vi = V_te[i]                                        # [P,3]
                CHUNK = 64
                row = torch.empty(Ntr_, device=device)
                for j0 in range(0, Ntr_, CHUNK):
                    j1 = min(Ntr_, j0 + CHUNK)
                    vj = V_tr[j0:j1]                                # [c,P,3]
                    # pairwise distances: [c, P_te, P_tr]
                    dpw = torch.cdist(vi.unsqueeze(0).expand(j1 - j0, -1, -1), vj)
                    # symmetric mean chamfer
                    d_te2tr = dpw.min(dim=2).values.mean(dim=1)     # [c]
                    d_tr2te = dpw.min(dim=1).values.mean(dim=1)     # [c]
                    row[j0:j1] = 0.5 * (d_te2tr + d_tr2te)
                d_key[i] = row
    elif args.knn_key == "cond_embed":
        with torch.no_grad():
            E_tr = condition_from_batch(cond_net, train_batch, device)  # [Ntr, cond_dim]
            E_te = condition_from_batch(cond_net, test_batch, device)   # [Nte, cond_dim]
        d_key = torch.cdist(E_te, E_tr)
    else:
        raise ValueError(args.knn_key)
    knn_idx = d_key.topk(args.K_knn, largest=False).indices        # [Nte, K_knn]
    KNN = G_tr[knn_idx].transpose(0, 1).contiguous()               # [K_knn, Nte, 432]

    # ---- CVAE samples per test case ----
    cond = condition_from_batch(cond_net, test_batch, device)  # [Nte, cond_dim]
    with torch.no_grad():
        CVAE = sample_fn(cond, args.K_cvae, args)          # [K_cvae, Nte, 432]

    pools = {}
    pools["knn_only"]  = KNN
    pools["cvae_only"] = CVAE
    pools["knn_plus_cvae"] = torch.cat([KNN, CVAE], dim=0)

    # blends: align by index, blend each (cvae[k], knn[k]) for k in 0..min(K_knn,K_cvae)-1
    alphas = [float(s) for s in args.alpha_grid.split(",") if s.strip()]
    K_blend = min(args.K_knn, args.K_cvae)
    blend_list = []
    for a in alphas:
        b = a * CVAE[:K_blend] + (1.0 - a) * KNN[:K_blend]
        blend_list.append(b)
    if blend_list:
        BLENDS = torch.cat(blend_list, dim=0)              # [len(alphas)*K_blend, Nte, 432]
        pools["blend_only"] = BLENDS
        pools["knn_plus_cvae_blend"] = torch.cat([KNN, CVAE, BLENDS], dim=0)

    print("\n=== NN-Hybrid pools (ORACLE best-of-pool on test) ===")
    print("  WARNING: best_of_pool uses GT to select. This is an ORACLE upper bound,")
    print("  not a deployable predictor. A reranker is required for honest inference.")
    print(f"  {'pool':>22s}  {'K':>4s}  {'GHD-432 RMSE':>14s}   {'vert MSE (1e-3)':>18s}")
    rmse_results, vert_results = {}, {}
    for tag, pool in pools.items():
        r = best_of_k_rmse(pool, G_te)
        v = best_of_k_vert_mse(pool, G_te, c2m, ghd_mean, ghd_std)
        rmse_results[tag] = r
        vert_results[tag] = v
        print(f"  {tag:>22s}  {pool.shape[0]:>4d}  {r:>14.4f}   {1000*v:>18.4f}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        json.dump({"K_knn": args.K_knn, "K_cvae": args.K_cvae,
                   "alpha_grid": alphas,
                   "knn_key": args.knn_key,
                   "cvae_ckpt": args.cvae_ckpt, "cvae_tag": args.cvae_tag,
                   "train_cases_file": args.train_cases_file,
                   "test_cases_file": args.test_cases_file,
                   "ghd_rmse": rmse_results,
                   "vert_mse": vert_results},
                  open(args.out_json, "w"), indent=2)
        print(f"[out] {args.out_json}")


if __name__ == "__main__":
    main()
