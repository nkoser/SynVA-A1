"""Two-stage trainer for FSQ-VAE + AR Transformer prior.

Stage 1: Train FSQVAE with reconstruction MSE only.
         Conditioner is co-trained so that 'cond' represents the ostium.
Stage 2: Freeze stage-1 modules, encode each train example to discrete tokens,
         train autoregressive transformer prior over tokens conditioned on cond.

Saves stage-1 + stage-2 weights, normalization stats, and args.
Inference is in `sample.py`.
"""
from __future__ import annotations
import argparse, json, os, sys, time

import torch
import torch.nn.functional as F

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
sys.path.insert(0, ROOT)

from methods._common.data import (
    add_common_args, set_seed, make_loaders, build_conditioner, encode_cond,
)
from methods._common.mesh_loss import (
    CoeffToMesh, mesh_recon_losses, apply_loss_mix, add_mesh_loss_args,
)
from methods.C_fsq_ar.model import FSQVAE
from methods.C_fsq_ar.ar_prior import FSQARPrior


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p)
    add_mesh_loss_args(p)
    # FSQVAE
    p.add_argument("--hidden_dim", type=int, default=384)
    p.add_argument("--encoder_blocks", type=int, default=3)
    p.add_argument("--decoder_blocks", type=int, default=6)
    p.add_argument("--num_tokens", type=int, default=8)
    p.add_argument("--levels", type=str, default="8,8,5,5,5",
                   help="Comma-separated FSQ levels per token dim.")
    p.add_argument("--dropout", type=float, default=0.05)
    # Stage-1
    p.add_argument("--stage1_epochs", type=int, default=2000)
    p.add_argument("--stage1_lr", type=float, default=7e-4)
    p.add_argument("--stage1_wd", type=float, default=1e-4)
    # Stage-2
    p.add_argument("--ar_dim", type=int, default=256)
    p.add_argument("--ar_depth", type=int, default=4)
    p.add_argument("--ar_heads", type=int, default=4)
    p.add_argument("--ar_dropout", type=float, default=0.1)
    p.add_argument("--stage2_epochs", type=int, default=2000)
    p.add_argument("--stage2_lr", type=float, default=5e-4)
    p.add_argument("--stage2_wd", type=float, default=1e-4)
    p.add_argument("--cond_dropout", type=float, default=0.10)
    return p.parse_args()


class _Tee:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1) if path else None
    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        print(msg, flush=True)
        if self.f: self.f.write(msg + "\n")


def stage1(args, device, log):
    train_ds, val_ds, train_dl, val_dl = make_loaders(args, device)
    input_dim = train_ds[0]["ghd"].numel()
    conditioner = build_conditioner(args, device)
    levels = [int(x) for x in args.levels.split(",")]
    model = FSQVAE(
        input_dim=input_dim, cond_dim=args.vessel_cond_dim,
        hidden_dim=args.hidden_dim, num_tokens=args.num_tokens,
        levels=levels, encoder_blocks=args.encoder_blocks,
        decoder_blocks=args.decoder_blocks, dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(conditioner.parameters()),
        lr=args.stage1_lr, weight_decay=args.stage1_wd,
    )
    log(f"[S1] FSQVAE input_dim={input_dim} levels={levels} codebook={model.quant.codebook_size} T={args.num_tokens}")
    coeff2mesh = CoeffToMesh(
        canonical_mesh_path=args.canonical_mesh_obj,
        eigen_chk=args.eigen_chk,
        num_basis=args.num_basis, device=device,
        canonical_norm_factor=args.canonical_norm_factor,
    )
    ghd_mean = train_ds.ghd_mean.to(device)
    ghd_std  = train_ds.ghd_std.to(device)
    use_normals = (not args.no_normals) and (args.w_normal > 0)
    log(f"[S1] mesh-aware loss: w_mse={args.w_mse} w_vert={args.w_vert} w_normal={args.w_normal} use_normals={use_normals}")
    best_val = float("inf"); best = None
    for ep in range(args.stage1_epochs):
        model.train(); conditioner.train()
        tr = []
        for batch in train_dl:
            ghd = batch["ghd"].to(device)
            cond = encode_cond(conditioner, batch, device,
                               no_vessel_pts=args.no_vessel_pts,
                               no_conditioning=args.no_conditioning)
            recon, _ = model(ghd, cond)
            ghd_mse, vert_mse, normal_mse = mesh_recon_losses(
                coeff2mesh, ghd, recon, ghd_mean, ghd_std, want_normals=use_normals,
            )
            loss = apply_loss_mix(ghd_mse, vert_mse, normal_mse,
                                  w_mse=args.w_mse, w_vert=args.w_vert, w_normal=args.w_normal)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(conditioner.parameters()), 1.0)
            opt.step()
            tr.append((loss.item(), ghd_mse.item(), vert_mse.item(),
                      normal_mse.item() if use_normals else 0.0))
        if (ep + 1) % 25 == 0 or ep == args.stage1_epochs - 1:
            model.eval(); conditioner.eval()
            with torch.no_grad():
                vl = []
                for batch in val_dl:
                    ghd = batch["ghd"].to(device)
                    cond = encode_cond(conditioner, batch, device,
                                       no_vessel_pts=args.no_vessel_pts,
                                       no_conditioning=args.no_conditioning)
                    recon, _ = model(ghd, cond)
                    g_mse, v_mse, n_mse = mesh_recon_losses(
                        coeff2mesh, ghd, recon, ghd_mean, ghd_std, want_normals=use_normals,
                    )
                    vl.append((g_mse.item(), v_mse.item(),
                               n_mse.item() if use_normals else 0.0))
            t_loss = sum(x[0] for x in tr) / len(tr)
            t_ghd  = sum(x[1] for x in tr) / len(tr)
            t_vert = sum(x[2] for x in tr) / len(tr)
            v_ghd  = sum(x[0] for x in vl) / len(vl)
            v_vert = sum(x[1] for x in vl) / len(vl)
            log(f"[S1] ep {ep+1:4d} train_loss {t_loss:.4f} train_ghd {t_ghd:.5f} train_vert {t_vert:.6f} | val_ghd {v_ghd:.5f} val_vert {v_vert:.6f}")
            # use vert MSE as model selection metric (matches works VAE)
            if v_vert < best_val:
                best_val = v_vert
                best = {
                    "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "conditioner": {k: v.detach().cpu().clone() for k, v in conditioner.state_dict().items()},
                    "epoch": ep + 1, "val_vert": v_vert, "val_ghd": v_ghd,
                }
    return model, conditioner, best, train_ds, val_ds, train_dl, val_dl, levels, input_dim


def collect_tokens(model, conditioner, loader, device, no_vp, no_cond):
    model.eval(); conditioner.eval()
    ids_list, cond_list = [], []
    with torch.no_grad():
        for batch in loader:
            ghd = batch["ghd"].to(device)
            cond = encode_cond(conditioner, batch, device, no_vessel_pts=no_vp, no_conditioning=no_cond)
            _, ids = model.encode(ghd, cond)
            ids_list.append(ids.cpu()); cond_list.append(cond.cpu())
    return torch.cat(ids_list, 0), torch.cat(cond_list, 0)


def stage2(args, device, log, model, conditioner, train_dl, val_dl, levels):
    vocab = 1
    for L in levels: vocab *= L
    log(f"[S2] AR prior over T={args.num_tokens} positions, vocab={vocab}")
    train_ids, train_cond = collect_tokens(model, conditioner, train_dl, device, args.no_vessel_pts, args.no_conditioning)
    val_ids,   val_cond   = collect_tokens(model, conditioner, val_dl,   device, args.no_vessel_pts, args.no_conditioning)
    train_ids, train_cond = train_ids.to(device), train_cond.to(device)
    val_ids,   val_cond   = val_ids.to(device),   val_cond.to(device)

    ar = FSQARPrior(vocab_size=vocab, num_tokens=args.num_tokens,
                    cond_dim=args.vessel_cond_dim, dim=args.ar_dim,
                    depth=args.ar_depth, heads=args.ar_heads, dropout=args.ar_dropout).to(device)
    opt = torch.optim.AdamW(ar.parameters(), lr=args.stage2_lr, weight_decay=args.stage2_wd)

    N = train_ids.size(0)
    best_val = float("inf"); best = None
    for ep in range(args.stage2_epochs):
        perm = torch.randperm(N, device=device)
        ar.train(); tr = []
        for i in range(0, N, args.batch_size):
            sl = perm[i:i + args.batch_size]
            ids_b = train_ids[sl]
            cond_b = train_cond[sl].clone()
            if args.cond_dropout > 0:
                drop = (torch.rand(cond_b.size(0), device=device) < args.cond_dropout).view(-1, 1)
                cond_b = torch.where(drop, torch.zeros_like(cond_b), cond_b)
            loss = ar.loss(ids_b, cond_b)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ar.parameters(), 1.0)
            opt.step()
            tr.append(loss.item())
        if (ep + 1) % 25 == 0 or ep == args.stage2_epochs - 1:
            ar.eval()
            with torch.no_grad():
                vloss = ar.loss(val_ids, val_cond).item()
            tmean = sum(tr) / len(tr)
            log(f"[S2] ep {ep+1:4d} train_ce {tmean:.4f} val_ce {vloss:.4f}")
            if vloss < best_val:
                best_val = vloss
                best = {
                    "ar": {k: v.detach().cpu().clone() for k, v in ar.state_dict().items()},
                    "epoch": ep + 1, "val_ce": vloss,
                }
    return ar, best


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.save_root, args.meta)
    os.makedirs(out_dir, exist_ok=True)
    log = _Tee(args.log_file or os.path.join(out_dir, "train.log"))
    log(f"=== Method C: FSQ-VAE + AR Prior ===")
    log(f"args: {json.dumps(vars(args), indent=2)}")

    t0 = time.time()
    model, conditioner, best1, train_ds, val_ds, train_dl, val_dl, levels, input_dim = stage1(args, device, log)
    log(f"[S1] done best_val_vert {best1['val_vert']:.5f} ep {best1['epoch']} time {time.time()-t0:.0f}s")

    # restore best stage-1 weights before tokenizing
    model.load_state_dict(best1["model"]); conditioner.load_state_dict(best1["conditioner"])
    t1 = time.time()
    ar, best2 = stage2(args, device, log, model, conditioner, train_dl, val_dl, levels)
    log(f"[S2] done best_val_ce {best2['val_ce']:.4f} ep {best2['epoch']} time {time.time()-t1:.0f}s")

    payload = {
        "args": vars(args),
        "saved_args": {  # for ensemble loader compat
            "model_type": "fsq_ar",
            "num_tokens": args.num_tokens, "levels": levels,
            "vessel_cond_dim": args.vessel_cond_dim,
            "vessel_feat_dim": args.vessel_feat_dim,
            "num_vessel_pts": args.num_vessel_pts,
            "num_ostium_pts": args.num_ostium_pts,
            "ring_points": args.ring_points,
            "canonical_opa_checkpoint": args.canonical_opa_checkpoint,
            "ostium_source": args.ostium_source,
            "use_ring_pts": args.use_ring_pts,
            "ring_feat_dim": args.ring_feat_dim,
            "use_ordered_ring": args.use_ordered_ring,
            "ordered_ring_feat_dim": args.ordered_ring_feat_dim,
            "no_vessel_pts": args.no_vessel_pts,
            "no_conditioning": args.no_conditioning,
            "ghd_chk_root": args.ghd_chk_root,
            "ghd_run": args.ghd_run,
            "ghd_chk_name": args.ghd_chk_name,
            "data_root": args.data_root,
            "aligned_data_root": args.aligned_data_root,
            "canonical_mesh": args.canonical_mesh,
            "canonical_norm_factor": args.canonical_norm_factor,
            "condition_space": args.condition_space,
            "condition_data_mode": args.condition_data_mode,
            "withscale": args.withscale,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "encoder_blocks": args.encoder_blocks,
            "decoder_blocks": args.decoder_blocks,
            "ar_dim": args.ar_dim, "ar_depth": args.ar_depth, "ar_heads": args.ar_heads,
        },
        "fsqvae": best1["model"],
        "conditioner": best1["conditioner"],
        "ar_prior": best2["ar"],
        "ghd_mean": train_ds.ghd_mean,
        "ghd_std": train_ds.ghd_std,
        "ostium_mean": train_ds.ostium_mean,
        "ostium_std": train_ds.ostium_std,
        "ostium_ring_mean": train_ds.ostium_ring_mean,
        "ostium_ring_std": train_ds.ostium_ring_std,
        "vessel_center": train_ds.vessel_center,
        "vessel_scale": train_ds.vessel_scale,
        "stage1_val_vert": best1["val_vert"],
        "stage1_val_ghd":  best1["val_ghd"],
        "stage2_val_ce": best2["val_ce"],
    }
    out_path = os.path.join(out_dir, "best.pt")
    torch.save(payload, out_path)
    log(f"saved {out_path}")


if __name__ == "__main__":
    main()
