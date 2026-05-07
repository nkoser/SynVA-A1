"""Shared helpers for methods/* — dataset, conditioner, normalization."""
from __future__ import annotations
import argparse
import json, os, random
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.vae_datasets_vessel import VesselAwareGHDDataset
from models.vessel_conditioner import OstiumConditioner
from first_stage_vessel_aware import collate_fn
from train_vessel_flow_matching import build_conditioner as _build_conditioner
from train_vessel_flow_matching import condition_from_batch


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cases(path: str):
    with open(path, "r") as f:
        if path.endswith(".json"):
            return json.load(f)
        return [line.strip() for line in f if line.strip()]


def make_dataset(cases, args) -> VesselAwareGHDDataset:
    return VesselAwareGHDDataset(
        ghd_chk_root=args.ghd_chk_root,
        ghd_run=args.ghd_run,
        ghd_chk_name=args.ghd_chk_name,
        data_root=args.data_root,
        cases=cases,
        num_vessel_pts=args.num_vessel_pts,
        num_ostium_pts=getattr(args, "num_ostium_pts", 64),
        num_label2_pts=getattr(args, "num_label2_pts", 256),
        ring_points=getattr(args, "ring_points", 20),
        canonical_opa_checkpoint=getattr(
            args,
            "canonical_opa_checkpoint",
            "/path/to/SynVA-A1/checkpoints/canonical_average/opa_checkpoint_1op.pkl",
        ),
        ostium_source=getattr(args, "ostium_source", "opa_checkpoint"),
        condition_space=args.condition_space,
        aligned_data_root=args.aligned_data_root,
        canonical_mesh=args.canonical_mesh,
        canonical_norm_factor=args.canonical_norm_factor,
        condition_data_mode=getattr(args, "condition_data_mode", "prepared"),
        morphology_root=getattr(args, "morphology_root", None),
        morphology_keys=getattr(args, "morphology_keys", None),
        withscale=getattr(args, "withscale", False),
        normalize=True,
    )


def copy_norm(src, dst):
    for n in ("ghd_mean", "ghd_std", "ostium_mean", "ostium_std",
              "ostium_ring_mean", "ostium_ring_std",
              "vessel_center", "vessel_scale",
              "morphology_mean", "morphology_std", "morphology_feature_names"):
        if not hasattr(src, n):
            continue
        v = getattr(src, n)
        setattr(dst, n, v.clone() if torch.is_tensor(v) else v)


def make_loaders(args, device):
    train_cases = load_cases(args.train_cases_file)
    val_cases = load_cases(args.val_cases_file)
    train_ds = make_dataset(train_cases, args)
    val_ds = make_dataset(val_cases, args)
    copy_norm(train_ds, val_ds)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, collate_fn=collate_fn, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn, drop_last=False)
    return train_ds, val_ds, train_dl, val_dl


def build_conditioner(args, device) -> OstiumConditioner:
    return _build_conditioner(args, device)


def encode_cond(conditioner, batch, device, no_vessel_pts: bool = False, no_conditioning: bool = False):
    return condition_from_batch(
        conditioner,
        batch,
        device,
        zero_vessel=no_vessel_pts,
        zero_all=no_conditioning,
    )


def add_common_args(p):
    p.add_argument("--ghd_chk_root", type=str,
                   default="/path/to/SynVA-A1/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999")
    p.add_argument("--ghd_run", type=str, default="prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3")
    p.add_argument("--ghd_chk_name", type=str, default="ghb_fitting_checkpoint.pkl")
    p.add_argument("--data_root", type=str, default="/path/to/prepared_meshes_3")
    p.add_argument("--aligned_data_root", type=str, default="/path/to/ghd_prepared_meshes_3_aneurysm_1op_new")
    p.add_argument("--canonical_mesh", type=str,
                   default="/path/to/SynVA-A1/checkpoints/canonical_average/part_aligned.obj")
    p.add_argument("--canonical_norm_factor", type=float, default=1.10)
    p.add_argument("--withscale", action=argparse.BooleanOptionalAction, default=False,
                   help="Train on GHD coefficients plus fitted scale s as the final target dimension.")
    p.add_argument("--condition_space", type=str, default="ghd_local")
    p.add_argument("--condition_data_mode", choices=["prepared", "opa_only", "alignment_vessel"],
                   default="prepared")
    p.add_argument("--morphology_root", type=str, default=None,
                   help="Optional root with */07_other/morphological_parameters.npy to append as condition.")
    p.add_argument("--morphology_keys", type=str, default="default",
                   help="Comma-separated morphology keys. default uses scalar aneurysm morphology without C_O.")
    p.add_argument("--num_vessel_pts", type=int, default=256)
    p.add_argument("--num_ostium_pts", type=int, default=64)
    p.add_argument("--num_label2_pts", type=int, default=256)
    p.add_argument("--ring_points", type=int, default=20)
    p.add_argument("--canonical_opa_checkpoint", type=str,
                   default="/path/to/SynVA-A1/checkpoints/canonical_average/opa_checkpoint_1op.pkl")
    p.add_argument("--ostium_source", choices=["opa_checkpoint", "vessel_boundary", "label2", "label1"],
                   default="opa_checkpoint")
    p.add_argument("--vessel_cond_dim", type=int, default=32)
    p.add_argument("--vessel_feat_dim", type=int, default=64)
    p.add_argument("--use_ring_pts", action="store_true")
    p.add_argument("--ring_feat_dim", type=int, default=32)
    p.add_argument("--use_ordered_ring", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ordered_ring_feat_dim", type=int, default=64)
    p.add_argument("--train_cases_file", type=str, required=True)
    p.add_argument("--val_cases_file", type=str, required=True)
    p.add_argument("--save_root", type=str, required=True)
    p.add_argument("--meta", type=str, required=True)
    p.add_argument("--log_file", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no_vessel_pts", action="store_true")
    p.add_argument("--no_conditioning", action="store_true")
