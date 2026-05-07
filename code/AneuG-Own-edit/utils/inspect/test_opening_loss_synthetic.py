#!/usr/bin/env python3
"""Synthetic test for opening-loss sensitivity to triangulation artifacts.

This script compares:
1) fan-triangulated disk vs fan-triangulated disk (clean baseline)
2) fan-triangulated disk vs fan disk with only center-vertex shifted
3) fan-triangulated disk vs two-ring disk with same boundary shape

It reports opening surface-sampled losses (paper style) and gradient hotspots,
plus boundary-ring losses for reference.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes


@dataclass
class DiskMesh:
    verts: torch.Tensor
    faces: torch.Tensor
    outer_ring_idx: torch.Tensor
    center_idx: int | None


def make_outer_ring(n: int, radius: float, device: torch.device) -> torch.Tensor:
    t = torch.linspace(0.0, 2.0 * torch.pi, n + 1, device=device)[:-1]
    x = radius * torch.cos(t)
    y = radius * torch.sin(t)
    z = torch.zeros_like(x)
    return torch.stack([x, y, z], dim=1)


def make_fan_disk(
    n: int,
    radius: float,
    center_offset: tuple[float, float, float],
    device: torch.device,
) -> DiskMesh:
    outer = make_outer_ring(n, radius, device)
    center = torch.tensor(center_offset, dtype=outer.dtype, device=device).view(1, 3)
    verts = torch.cat([outer, center], dim=0)
    c = n
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([c, i, j])
    faces = torch.tensor(faces, dtype=torch.int64, device=device)
    return DiskMesh(
        verts=verts,
        faces=faces,
        outer_ring_idx=torch.arange(n, device=device, dtype=torch.int64),
        center_idx=c,
    )


def make_two_ring_disk(
    n: int,
    radius_outer: float,
    radius_inner: float,
    device: torch.device,
) -> DiskMesh:
    outer = make_outer_ring(n, radius_outer, device)
    inner = make_outer_ring(n, radius_inner, device)
    verts = torch.cat([outer, inner], dim=0)
    faces = []
    # Annulus band (outer-inner strips)
    for i in range(n):
        j = (i + 1) % n
        oi, oj = i, j
        ii, ij = n + i, n + j
        faces.append([oi, oj, ii])
        faces.append([oj, ij, ii])
    # Fill inner polygon (no single center vertex)
    for i in range(1, n - 1):
        faces.append([n + 0, n + i, n + i + 1])
    faces = torch.tensor(faces, dtype=torch.int64, device=device)
    return DiskMesh(
        verts=verts,
        faces=faces,
        outer_ring_idx=torch.arange(n, device=device, dtype=torch.int64),
        center_idx=None,
    )


def ring_mean_normal(ring: torch.Tensor) -> torch.Tensor:
    c = ring.mean(dim=0, keepdim=True)
    a = ring - c
    b = torch.roll(a, shifts=-1, dims=0)
    n = torch.cross(a, b, dim=1).sum(dim=0)
    return n / (torch.norm(n) + 1e-12)


def compute_surface_losses_and_grads(
    src: DiskMesh,
    tgt: DiskMesh,
    sample_num: int,
    seed: int,
    device: torch.device,
) -> dict:
    src_verts = src.verts.clone().detach().requires_grad_(True)
    src_mesh = Meshes(verts=[src_verts], faces=[src.faces])
    tgt_mesh = Meshes(verts=[tgt.verts], faces=[tgt.faces])

    torch.manual_seed(seed)
    p_src, n_src = sample_points_from_meshes(src_mesh, sample_num, return_normals=True)
    torch.manual_seed(seed + 1)
    p_tgt, n_tgt = sample_points_from_meshes(tgt_mesh, sample_num, return_normals=True)
    loss_p, loss_n = chamfer_distance(p_src, p_tgt, x_normals=n_src, y_normals=n_tgt)
    total = loss_p + loss_n
    total.backward()

    g = src_verts.grad.norm(dim=1).detach()
    outer_g = g[src.outer_ring_idx]
    out = {
        "loss_openings_p_surface": float(loss_p.detach().cpu().item()),
        "loss_openings_n_surface": float(loss_n.detach().cpu().item()),
        "loss_surface_total": float(total.detach().cpu().item()),
        "grad_max": float(g.max().cpu().item()),
        "grad_outer_mean": float(outer_g.mean().cpu().item()),
        "grad_outer_max": float(outer_g.max().cpu().item()),
    }
    if src.center_idx is not None:
        gc = float(g[src.center_idx].cpu().item())
        out["grad_center"] = gc
        out["grad_center_over_outer_mean"] = gc / (out["grad_outer_mean"] + 1e-12)
    return out


def compute_boundary_losses(src: DiskMesh, tgt: DiskMesh) -> dict:
    r_src = src.verts[src.outer_ring_idx].unsqueeze(0)
    r_tgt = tgt.verts[tgt.outer_ring_idx].unsqueeze(0)
    loss_p, _ = chamfer_distance(r_src, r_tgt)
    n_src = ring_mean_normal(src.verts[src.outer_ring_idx])
    n_tgt = ring_mean_normal(tgt.verts[tgt.outer_ring_idx])
    loss_n = 1.0 - torch.abs(torch.clamp(torch.sum(n_src * n_tgt), -1.0, 1.0))
    return {
        "loss_openings_p_boundary": float(loss_p.detach().cpu().item()),
        "loss_openings_n_boundary": float(loss_n.detach().cpu().item()),
    }


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    src = make_fan_disk(
        n=args.n_boundary,
        radius=args.radius,
        center_offset=(0.0, 0.0, 0.0),
        device=device,
    )
    tgt_clean = make_fan_disk(
        n=args.n_boundary,
        radius=args.radius,
        center_offset=(0.0, 0.0, 0.0),
        device=device,
    )
    tgt_center_shift = make_fan_disk(
        n=args.n_boundary,
        radius=args.radius,
        center_offset=(args.center_shift, 0.0, 0.0),
        device=device,
    )
    tgt_two_ring = make_two_ring_disk(
        n=args.n_boundary,
        radius_outer=args.radius,
        radius_inner=args.radius * args.inner_ratio,
        device=device,
    )

    res = {
        "config": {
            "n_boundary": args.n_boundary,
            "radius": args.radius,
            "sample_num": args.sample_num,
            "center_shift": args.center_shift,
            "inner_ratio": args.inner_ratio,
            "device": args.device,
        },
        "scenarios": {},
    }

    scenarios = {
        "A_clean_fan_vs_fan": tgt_clean,
        "B_center_shift_fan_vs_fan": tgt_center_shift,
        "C_topology_mismatch_fan_vs_two_ring": tgt_two_ring,
    }
    for i, (name, tgt) in enumerate(scenarios.items()):
        surf = compute_surface_losses_and_grads(
            src=src,
            tgt=tgt,
            sample_num=args.sample_num,
            seed=args.seed + 100 * i,
            device=device,
        )
        bnd = compute_boundary_losses(src, tgt)
        m = {}
        m.update(surf)
        m.update(bnd)
        res["scenarios"][name] = m
    return res


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic opening-loss triangulation sensitivity test.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-boundary", type=int, default=96)
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--inner-ratio", type=float, default=0.45)
    p.add_argument("--center-shift", type=float, default=0.02)
    p.add_argument("--sample-num", type=int, default=3000)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out-json", type=str, default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    res = run(args)
    print(json.dumps(res, indent=2))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
