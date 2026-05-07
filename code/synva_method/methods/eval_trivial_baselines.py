#!/usr/bin/env python
"""Trivial-baseline best-of-K RMSE on the same eval protocol as methods/eval_all.py.

Reports per-method best-of-K RMSE in the normalized 432-D GHD space:
  - mean      : predict the train-mean (zero in normalized space). Deterministic, K=1.
  - gauss     : sample N(0, I) per case (matches the unconditional std-normal prior).
  - rand_train: K random GHDs sampled from train.
  - knn_ost   : k=K nearest train cases by 8-D ostium-param L2 distance.
  - knn_ves   : k=K nearest train cases by mean-vessel-point Chamfer-ish L2 distance.

This defines the data floor: any method that does not beat these is not learning.

Usage:
  python methods/eval_trivial_baselines.py \
    --train_cases_file <split>/cases_train.json \
    --test_cases_file  <split>/cases_test.json \
    --ref_ckpt <any A/B/C/D ckpt that gives us the dataset config> \
    --num_samples 16 --out_json out.json
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_cases_file", required=True)
    p.add_argument("--test_cases_file", required=True)
    p.add_argument("--ref_ckpt", required=True,
                   help="Any trained ckpt from methods/* — only used to read dataset config + normalization stats.")
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", default=None)
    return p.parse_args()


def _sa(ck):
    out = {}
    for key in ("args", "saved_args"):
        a = ck.get(key, None)
        if a is None:
            continue
        if not isinstance(a, dict):
            a = vars(a)
        out.update(a)
    return out


def _build_dataset(cases, sa):
    return VesselAwareGHDDataset(
        ghd_chk_root=sa.get("ghd_chk_root"),
        ghd_run=sa.get("ghd_run"),
        ghd_chk_name=sa.get("ghd_chk_name", "ghb_fitting_checkpoint.pkl"),
        data_root=sa.get("data_root", "/path/to/prepared_meshes_3"),
        cases=cases,
        num_vessel_pts=int(sa.get("num_vessel_pts", 256)),
        num_ostium_pts=int(sa.get("num_ostium_pts", 64)),
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
        normalize=True,
    )


def _copy_stats(ds, ck):
    for n in ("ghd_mean", "ghd_std", "ostium_mean", "ostium_std",
              "ostium_ring_mean", "ostium_ring_std",
              "vessel_center", "vessel_scale"):
        if n in ck:
            setattr(ds, n, ck[n].cpu())


def _gather_ghd_ostium_vessel(ds, device):
    """Stack all (ghd, ostium, vessel) tensors of a dataset."""
    loader = DataLoader(ds, batch_size=200, shuffle=False, collate_fn=collate_fn)
    ghds, osts, vess = [], [], []
    for batch in loader:
        ghds.append(batch["ghd"])
        osts.append(batch["ostium_params"])
        vess.append(batch["vessel_pts"])
    G = torch.cat(ghds, 0).to(device)         # [N, 432]
    O = torch.cat(osts, 0).to(device)         # [N, 8]
    V = torch.cat(vess, 0).to(device)         # [N, P, 3]
    return G, O, V


def best_of_k_rmse(samples_KBn: torch.Tensor, gt_Bn: torch.Tensor) -> float:
    """samples: [K, B, 432]; gt: [B, 432]; returns mean per-case best-of-K RMSE."""
    rmse = torch.sqrt(((samples_KBn - gt_Bn.unsqueeze(0)) ** 2).mean(-1) + 1e-8)
    return float(rmse.min(0).values.mean().item())


def best_of_k_vert_mse(samples_KBn: torch.Tensor, gt_Bn: torch.Tensor,
                       c2m: CoeffToMesh, ghd_mean, ghd_std) -> float:
    """Decode K samples + GT to vertices, return mean per-case best-of-K vertex-MSE."""
    K, B, _ = samples_KBn.shape
    v_gt, _ = c2m(gt_Bn, ghd_mean, ghd_std, want_normals=False)         # [B, V, 3]
    bests = torch.full((B,), float("inf"), device=gt_Bn.device)
    for k in range(K):
        v_pred, _ = c2m(samples_KBn[k], ghd_mean, ghd_std, want_normals=False)
        mse = ((v_pred - v_gt) ** 2).mean(dim=(1, 2))                    # [B]
        bests = torch.minimum(bests, mse)
    return float(bests.mean().item())


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    K = args.num_samples

    print(f"[load] ref ckpt: {args.ref_ckpt}")
    ck = torch.load(args.ref_ckpt, map_location="cpu", weights_only=False)
    sa = _sa(ck)

    train_cases = json.load(open(args.train_cases_file))
    test_cases = json.load(open(args.test_cases_file))
    print(f"[data] train={len(train_cases)}  test={len(test_cases)}")

    print("[build] train dataset")
    train_ds = _build_dataset(train_cases, sa)
    _copy_stats(train_ds, ck)
    G_tr, O_tr, V_tr = _gather_ghd_ostium_vessel(train_ds, device)
    print(f"  train tensors: ghd {tuple(G_tr.shape)}  ost {tuple(O_tr.shape)}  ves {tuple(V_tr.shape)}")

    print("[build] test dataset (normalized with TRAIN stats)")
    test_ds = _build_dataset(test_cases, sa)
    _copy_stats(test_ds, ck)
    G_te, O_te, V_te = _gather_ghd_ostium_vessel(test_ds, device)
    print(f"  test  tensors: ghd {tuple(G_te.shape)}  ost {tuple(O_te.shape)}  ves {tuple(V_te.shape)}")

    Ntr = G_tr.shape[0]
    Nte = G_te.shape[0]

    # CoeffToMesh for vertex-space evaluation
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

    # Train mean / std in normalized space (should be ~0 / ~1 because of dataset norm).
    train_mean = G_tr.mean(0, keepdim=True)        # [1, 432]
    train_std  = G_tr.std(0, keepdim=True)         # [1, 432]
    print(f"[stats] train_mean ‖.‖ = {train_mean.norm().item():.4f}   "
          f"train_std mean = {train_std.mean().item():.4f}")

    results = {}
    vert_results = {}

    # 1. Constant train mean (K=1, deterministic)
    pred = train_mean.expand(Nte, -1).unsqueeze(0)   # [1, Nte, 432]
    results["mean"] = best_of_k_rmse(pred, G_te)
    vert_results["mean"] = best_of_k_vert_mse(pred, G_te, c2m, ghd_mean, ghd_std)

    # 2. Gaussian samples from N(train_mean, train_std), K samples per case
    g = torch.randn(K, Nte, 432, device=device) * train_std + train_mean
    results["gauss"] = best_of_k_rmse(g, G_te)
    vert_results["gauss"] = best_of_k_vert_mse(g, G_te, c2m, ghd_mean, ghd_std)

    # 3. Random K training GHDs per case
    idx = torch.randint(0, Ntr, (K, Nte), device=device)
    pred = G_tr[idx]                                 # [K, Nte, 432]
    results["rand_train"] = best_of_k_rmse(pred, G_te)
    vert_results["rand_train"] = best_of_k_vert_mse(pred, G_te, c2m, ghd_mean, ghd_std)

    # 4. kNN by ostium params (8-D)
    d_ost = torch.cdist(O_te, O_tr)                  # [Nte, Ntr]
    knn_idx = d_ost.topk(K, largest=False).indices   # [Nte, K]
    pred = G_tr[knn_idx].transpose(0, 1)             # [K, Nte, 432]
    results["knn_ostium"] = best_of_k_rmse(pred, G_te)
    vert_results["knn_ostium"] = best_of_k_vert_mse(pred, G_te, c2m, ghd_mean, ghd_std)

    # 5. kNN by mean vessel point (cheap surrogate for full set distance)
    Vtr_mean = V_tr.mean(1)                          # [Ntr, 3]
    Vte_mean = V_te.mean(1)                          # [Nte, 3]
    d_ves = torch.cdist(Vte_mean, Vtr_mean)
    knn_idx = d_ves.topk(K, largest=False).indices
    pred = G_tr[knn_idx].transpose(0, 1)
    results["knn_vessel_mean"] = best_of_k_rmse(pred, G_te)
    vert_results["knn_vessel_mean"] = best_of_k_vert_mse(pred, G_te, c2m, ghd_mean, ghd_std)

    # 6. kNN by combined ostium+vessel-mean (z-scored)
    feat_tr = torch.cat([O_tr, Vtr_mean], dim=1)     # [Ntr, 11]
    feat_te = torch.cat([O_te, Vte_mean], dim=1)
    f_mean = feat_tr.mean(0, keepdim=True)
    f_std  = feat_tr.std(0, keepdim=True) + 1e-8
    feat_tr_n = (feat_tr - f_mean) / f_std
    feat_te_n = (feat_te - f_mean) / f_std
    d_combo = torch.cdist(feat_te_n, feat_tr_n)
    knn_idx = d_combo.topk(K, largest=False).indices
    pred = G_tr[knn_idx].transpose(0, 1)
    results["knn_combo"] = best_of_k_rmse(pred, G_te)
    vert_results["knn_combo"] = best_of_k_vert_mse(pred, G_te, c2m, ghd_mean, ghd_std)

    print("\n=== Trivial baselines (best-of-{} on test) ===".format(K))
    print(f"  {'tag':>20s} : {'GHD-432 RMSE':>14s}   {'vert MSE (1e-3)':>18s}")
    for tag, v in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {tag:>20s} : {v:>14.4f}   {1000*vert_results[tag]:>18.4f}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        json.dump({"K": K,
                   "train_cases_file": args.train_cases_file,
                   "test_cases_file": args.test_cases_file,
                   "ref_ckpt": args.ref_ckpt,
                   "ghd_rmse": results,
                   "vert_mse": vert_results},
                  open(args.out_json, "w"), indent=2)
        print(f"[out] {args.out_json}")


if __name__ == "__main__":
    main()
