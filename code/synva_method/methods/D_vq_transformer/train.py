"""Two-stage trainer for VQ-VAE + AR Transformer prior. Mirrors Method C."""
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
    CoeffToMesh, mesh_recon_losses, opening_ring_losses,
    apply_loss_mix, add_mesh_loss_args,
)
from methods.E_collision.collision_loss import load_opening_idx
from methods.D_vq_transformer.model import VQVAE
from methods.C_fsq_ar.ar_prior import FSQARPrior  # reused (vocab differs)


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p)
    add_mesh_loss_args(p)
    p.add_argument("--hidden_dim", type=int, default=384)
    p.add_argument("--encoder_blocks", type=int, default=3)
    p.add_argument("--decoder_blocks", type=int, default=6)
    p.add_argument("--num_tokens", type=int, default=8)
    p.add_argument("--code_dim", type=int, default=32)
    p.add_argument("--num_codes", type=int, default=256)
    p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--commitment_beta", type=float, default=0.25)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--dead_reset_every", type=int, default=50)
    p.add_argument("--stage1_epochs", type=int, default=2000)
    p.add_argument("--stage1_lr", type=float, default=7e-4)
    p.add_argument("--stage1_wd", type=float, default=1e-4)
    p.add_argument("--ar_dim", type=int, default=256)
    p.add_argument("--ar_depth", type=int, default=4)
    p.add_argument("--ar_heads", type=int, default=4)
    p.add_argument("--ar_dropout", type=float, default=0.10)
    p.add_argument("--stage2_epochs", type=int, default=2000)
    p.add_argument("--stage2_lr", type=float, default=5e-4)
    p.add_argument("--stage2_wd", type=float, default=1e-4)
    p.add_argument("--cond_dropout", type=float, default=0.10)
    return p.parse_args()


class _Tee:
    def __init__(self, path): self.f = open(path, "a", buffering=1) if path else None
    def __call__(self, *a):
        m = " ".join(str(x) for x in a); print(m, flush=True)
        if self.f: self.f.write(m + "\n")


def stage1(args, device, log):
    train_ds, val_ds, train_dl, val_dl = make_loaders(args, device)
    input_dim = train_ds[0]["ghd"].numel()
    conditioner = build_conditioner(args, device)
    model = VQVAE(
        input_dim=input_dim, cond_dim=args.vessel_cond_dim,
        hidden_dim=args.hidden_dim, num_tokens=args.num_tokens,
        code_dim=args.code_dim, num_codes=args.num_codes,
        encoder_blocks=args.encoder_blocks, decoder_blocks=args.decoder_blocks,
        dropout=args.dropout, ema_decay=args.ema_decay,
    ).to(device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(conditioner.parameters()),
        lr=args.stage1_lr, weight_decay=args.stage1_wd,
    )
    log(f"[S1] VQVAE input_dim={input_dim} K={args.num_codes} D_code={args.code_dim} T={args.num_tokens}")
    coeff2mesh = CoeffToMesh(
        canonical_mesh_path=args.canonical_mesh_obj,
        eigen_chk=args.eigen_chk,
        num_basis=args.num_basis, device=device,
        canonical_norm_factor=args.canonical_norm_factor,
    )
    opening_idx = load_opening_idx(args.canonical_mesh_obj, args.canonical_opa_checkpoint)
    if opening_idx is not None:
        opening_idx = opening_idx.to(device)
        log(f"[S1] opening-ring loss: idx={int(opening_idx.numel())} "
            f"w_ring={args.w_ring} w_ring_chamfer={args.w_ring_chamfer}")
    else:
        log("[S1] opening-ring loss: disabled (no opening indices found)")
    ghd_mean = train_ds.ghd_mean.to(device); ghd_std = train_ds.ghd_std.to(device)
    ring_mean = train_ds.ostium_ring_mean.to(device)
    ring_std = train_ds.ostium_ring_std.to(device)
    use_normals = (not args.no_normals) and (args.w_normal > 0)
    log(f"[S1] mesh-aware loss: w_mse={args.w_mse} w_vert={args.w_vert} w_normal={args.w_normal} use_normals={use_normals}")
    best_val = float("inf"); best = None; last_flat = None
    for ep in range(args.stage1_epochs):
        model.train(); conditioner.train()
        tr_loss, tr_ghd, tr_vert, tr_ring, tr_chamfer, tr_cm, tr_perp = [], [], [], [], [], [], []
        for batch in train_dl:
            ghd = batch["ghd"].to(device)
            cond = encode_cond(conditioner, batch, device,
                               no_vessel_pts=args.no_vessel_pts,
                               no_conditioning=args.no_conditioning)
            recon, ids, commitment, perp, flat = model(ghd, cond)
            ghd_mse, vert_mse, normal_mse = mesh_recon_losses(
                coeff2mesh, ghd, recon, ghd_mean, ghd_std, want_normals=use_normals,
            )
            recon_loss = apply_loss_mix(
                ghd_mse, vert_mse, normal_mse,
                w_mse=args.w_mse, w_vert=args.w_vert, w_normal=args.w_normal,
            )
            ring_mse = torch.zeros((), device=device)
            ring_chamfer = torch.zeros((), device=device)
            if opening_idx is not None and (args.w_ring > 0 or args.w_ring_chamfer > 0):
                pred_verts, _ = coeff2mesh(recon, ghd_mean, ghd_std, want_normals=False)
                target_ring = batch["ostium_ring"].to(device).reshape(ghd.shape[0], -1)
                target_ring = (target_ring * ring_std + ring_mean).view(ghd.shape[0], args.ring_points, 3)
                ring_mse, ring_chamfer = opening_ring_losses(pred_verts, target_ring, opening_idx)
                recon_loss = recon_loss + args.w_ring * ring_mse + args.w_ring_chamfer * ring_chamfer
            loss = recon_loss + args.commitment_beta * commitment
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(conditioner.parameters()), 1.0)
            opt.step()
            tr_loss.append(loss.item()); tr_ghd.append(ghd_mse.item())
            tr_vert.append(vert_mse.item())
            tr_ring.append(ring_mse.item()); tr_chamfer.append(ring_chamfer.item())
            tr_cm.append(commitment.item()); tr_perp.append(perp.item())
            last_flat = flat.detach()
        if args.dead_reset_every > 0 and (ep + 1) % args.dead_reset_every == 0 and last_flat is not None:
            n = model.vq.reset_dead_codes(last_flat)
            if n > 0: log(f"[S1] ep {ep+1}: reset {n} dead codes")
        if (ep + 1) % 25 == 0 or ep == args.stage1_epochs - 1:
            model.eval(); conditioner.eval()
            with torch.no_grad():
                vl_ghd, vl_vert, vl_ring, vl_chamfer = [], [], [], []
                for batch in val_dl:
                    ghd = batch["ghd"].to(device)
                    cond = encode_cond(conditioner, batch, device,
                                       no_vessel_pts=args.no_vessel_pts,
                                       no_conditioning=args.no_conditioning)
                    recon, *_ = model(ghd, cond)
                    g_mse, v_mse, _ = mesh_recon_losses(
                        coeff2mesh, ghd, recon, ghd_mean, ghd_std, want_normals=use_normals,
                    )
                    r_mse = torch.zeros((), device=device)
                    r_ch = torch.zeros((), device=device)
                    if opening_idx is not None:
                        pred_verts, _ = coeff2mesh(recon, ghd_mean, ghd_std, want_normals=False)
                        target_ring = batch["ostium_ring"].to(device).reshape(ghd.shape[0], -1)
                        target_ring = (target_ring * ring_std + ring_mean).view(ghd.shape[0], args.ring_points, 3)
                        r_mse, r_ch = opening_ring_losses(pred_verts, target_ring, opening_idx)
                    vl_ghd.append(g_mse.item()); vl_vert.append(v_mse.item())
                    vl_ring.append(r_mse.item()); vl_chamfer.append(r_ch.item())
            v_ghd  = sum(vl_ghd)/len(vl_ghd)
            v_vert = sum(vl_vert)/len(vl_vert)
            v_ring = sum(vl_ring)/len(vl_ring)
            v_chamfer = sum(vl_chamfer)/len(vl_chamfer)
            log(f"[S1] ep {ep+1:4d} train_loss {sum(tr_loss)/len(tr_loss):.4f} "
                f"train_vert {sum(tr_vert)/len(tr_vert):.6f} "
                f"train_ring {sum(tr_ring)/len(tr_ring):.6f} train_ring_ch {sum(tr_chamfer)/len(tr_chamfer):.6f} "
                f"cm {sum(tr_cm)/len(tr_cm):.4f} perp {sum(tr_perp)/len(tr_perp):.1f}/{args.num_codes} "
                f"| val_ghd {v_ghd:.5f} val_vert {v_vert:.6f} "
                f"val_ring {v_ring:.6f} val_ring_ch {v_chamfer:.6f}")
            if v_vert < best_val:
                best_val = v_vert
                best = {
                    "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "conditioner": {k: v.detach().cpu().clone() for k, v in conditioner.state_dict().items()},
                    "epoch": ep + 1, "val_vert": v_vert, "val_ghd": v_ghd,
                    "val_ring": v_ring, "val_ring_chamfer": v_chamfer,
                }
    return model, conditioner, best, train_ds, val_ds, train_dl, val_dl, input_dim


def collect_tokens(model, conditioner, loader, device, no_vp, no_cond):
    model.eval(); conditioner.eval()
    ids_list, cond_list = [], []
    with torch.no_grad():
        for batch in loader:
            ghd = batch["ghd"].to(device)
            cond = encode_cond(conditioner, batch, device, no_vessel_pts=no_vp, no_conditioning=no_cond)
            _, ids, *_ = model.encode(ghd, cond)
            ids_list.append(ids.cpu()); cond_list.append(cond.cpu())
    return torch.cat(ids_list, 0), torch.cat(cond_list, 0)


def stage2(args, device, log, model, conditioner, train_dl, val_dl):
    train_ids, train_cond = collect_tokens(model, conditioner, train_dl, device, args.no_vessel_pts, args.no_conditioning)
    val_ids,   val_cond   = collect_tokens(model, conditioner, val_dl,   device, args.no_vessel_pts, args.no_conditioning)
    train_ids, train_cond = train_ids.to(device), train_cond.to(device)
    val_ids,   val_cond   = val_ids.to(device),   val_cond.to(device)

    ar = FSQARPrior(vocab_size=args.num_codes, num_tokens=args.num_tokens,
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
            opt.step(); tr.append(loss.item())
        if (ep + 1) % 25 == 0 or ep == args.stage2_epochs - 1:
            ar.eval()
            with torch.no_grad():
                vloss = ar.loss(val_ids, val_cond).item()
            log(f"[S2] ep {ep+1:4d} train_ce {sum(tr)/len(tr):.4f} val_ce {vloss:.4f}")
            if vloss < best_val:
                best_val = vloss
                best = {"ar": {k: v.detach().cpu().clone() for k, v in ar.state_dict().items()},
                        "epoch": ep + 1, "val_ce": vloss}
    return ar, best


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.save_root, args.meta)
    os.makedirs(out_dir, exist_ok=True)
    log = _Tee(args.log_file or os.path.join(out_dir, "train.log"))
    log(f"=== Method D: VQ-VAE + AR Prior ===")
    log(f"args: {json.dumps(vars(args), indent=2)}")

    t0 = time.time()
    model, conditioner, best1, train_ds, val_ds, train_dl, val_dl, input_dim = stage1(args, device, log)
    log(f"[S1] done best_val_vert {best1['val_vert']:.5f} ep {best1['epoch']} time {time.time()-t0:.0f}s")
    model.load_state_dict(best1["model"]); conditioner.load_state_dict(best1["conditioner"])

    t1 = time.time()
    ar, best2 = stage2(args, device, log, model, conditioner, train_dl, val_dl)
    log(f"[S2] done best_val_ce {best2['val_ce']:.4f} ep {best2['epoch']} time {time.time()-t1:.0f}s")

    payload = {
        "args": vars(args),
        "saved_args": {
            "model_type": "vq_ar",
            "num_tokens": args.num_tokens,
            "code_dim": args.code_dim, "num_codes": args.num_codes,
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
            "no_vessel_pts": args.no_vessel_pts, "no_conditioning": args.no_conditioning,
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
            "w_ring": args.w_ring,
            "w_ring_chamfer": args.w_ring_chamfer,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "encoder_blocks": args.encoder_blocks, "decoder_blocks": args.decoder_blocks,
            "ar_dim": args.ar_dim, "ar_depth": args.ar_depth, "ar_heads": args.ar_heads,
        },
        "vqvae": best1["model"],
        "conditioner": best1["conditioner"],
        "ar_prior": best2["ar"],
        "ghd_mean": train_ds.ghd_mean, "ghd_std": train_ds.ghd_std,
        "ostium_mean": train_ds.ostium_mean, "ostium_std": train_ds.ostium_std,
        "ostium_ring_mean": train_ds.ostium_ring_mean, "ostium_ring_std": train_ds.ostium_ring_std,
        "vessel_center": train_ds.vessel_center, "vessel_scale": train_ds.vessel_scale,
        "stage1_val_vert": best1["val_vert"], "stage1_val_ghd": best1["val_ghd"],
        "stage2_val_ce": best2["val_ce"],
    }
    out_path = os.path.join(out_dir, "best.pt")
    torch.save(payload, out_path)
    log(f"saved {out_path}")


if __name__ == "__main__":
    main()
