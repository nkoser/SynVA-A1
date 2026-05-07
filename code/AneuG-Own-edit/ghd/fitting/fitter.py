import pytorch3d.io
import pytorch3d
import torch
import numpy as np
from pytorch3d.io import load_objs_as_meshes, save_obj
import os
import sys
import pickle
import json
import copy
import time
from ghd.base.mesh_geometry import MeshThickness
import torch.nn.functional as F
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ghd.fitting.registration import RegistrationwOpeningAlignmentwDifferentiableCentreline
from ghd.base.graph_harmonic_deformation import (
    Graph_Harmonic_Deform_opening_alignment_dynamic,
    Graph_Harmonic_Deform,
)
from ghd.losses import (
    Mesh_loss_opening_alignment,
    Mesh_loss_differentiable_occupancy,
    Mesh_loss_do_differentiable_centreline,
    Mesh_loss_pouch_only,
    Mesh_loss,
)
from torch.utils.tensorboard import SummaryWriter
from ghd.fitting.logger import log_dict_printer
from ghd.fitting.logger import viz_fitting_static, viz_fitting_debug
from ghd.fitting.weighter import base_loss_weighter
from ghd.fitting.dropper import Do_Dropper
from ghd.base.mesh_geometry3 import Winding_Occupancy
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes


def _tri_mesh_area(verts: np.ndarray, faces: np.ndarray) -> float:
    if verts is None or faces is None:
        return 0.0
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if verts.size == 0 or faces.size == 0 or faces.ndim != 2 or faces.shape[1] < 3:
        return 0.0
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def _safe_unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(vec)
    return vec / n if n > 1e-12 else vec


def _mesh_radius(mesh: Meshes) -> float:
    try:
        return float(torch.max(torch.norm(mesh.verts_packed(), dim=-1)).detach().cpu().item())
    except Exception:
        return 1.0


def estimate_opening_difficulty(canonical, target, num_op: int):
    pair_num = min(num_op, len(getattr(canonical, "op_rec_v", [])), len(getattr(target, "op_rec_v", [])))
    if pair_num <= 0:
        return None
    scale = max(_mesh_radius(getattr(canonical, "mesh_target_p3d", None)),
                _mesh_radius(getattr(target, "mesh_target_p3d", None)),
                1e-6)
    per_opening = []
    for idx in range(pair_num):
        c_verts = np.asarray(canonical.op_rec_v[idx], dtype=np.float64)
        t_verts = np.asarray(target.op_rec_v[idx], dtype=np.float64)
        c_faces = np.asarray(canonical.op_rec_f[idx], dtype=np.int64)
        t_faces = np.asarray(target.op_rec_f[idx], dtype=np.int64)
        c_centroid = c_verts.mean(axis=0) if c_verts.size else np.zeros(3, dtype=np.float64)
        t_centroid = t_verts.mean(axis=0) if t_verts.size else np.zeros(3, dtype=np.float64)
        centroid_dist = float(np.linalg.norm(c_centroid - t_centroid) / scale)
        centroid_dist = float(min(max(centroid_dist, 0.0), 2.0))
        c_area = _tri_mesh_area(c_verts, c_faces)
        t_area = _tri_mesh_area(t_verts, t_faces)
        area_ratio = float(c_area / (t_area + 1e-8))
        area_mismatch = float(abs(area_ratio - 1.0))
        area_mismatch = float(min(max(area_mismatch, 0.0), 2.0))
        c_n = _safe_unit(getattr(canonical, "op_n_mean", [np.array([1.0, 0.0, 0.0])])[idx])
        t_n = _safe_unit(getattr(target, "op_n_mean", [np.array([1.0, 0.0, 0.0])])[idx])
        normal_mismatch = float(1.0 - abs(float(np.dot(c_n, t_n))))
        normal_mismatch = float(min(max(normal_mismatch, 0.0), 1.0))
        per_opening.append({
            "opening_index": int(idx),
            "centroid_dist": centroid_dist,
            "area_ratio": area_ratio,
            "area_mismatch": area_mismatch,
            "normal_mismatch": normal_mismatch,
        })
    avg_centroid_dist = float(np.mean([m["centroid_dist"] for m in per_opening]))
    avg_area_mismatch = float(np.mean([m["area_mismatch"] for m in per_opening]))
    avg_normal_mismatch = float(np.mean([m["normal_mismatch"] for m in per_opening]))
    score = float(0.6 * avg_centroid_dist + 0.3 * avg_area_mismatch + 0.1 * avg_normal_mismatch)
    return {
        "score": score,
        "avg_centroid_dist": avg_centroid_dist,
        "avg_area_mismatch": avg_area_mismatch,
        "avg_normal_mismatch": avg_normal_mismatch,
        "pair_num": int(pair_num),
        "scale": float(scale),
        "per_opening": per_opening,
    }


def _apply_adaptive_policy(args, loss_weighting, canonical, target):
    if int(getattr(args, "adaptive_fitting", 0)) != 1:
        return loss_weighting, None, None, int(getattr(args, "epochs", 0))
    difficulty = estimate_opening_difficulty(canonical, target, int(getattr(args, "num_op", 0)))
    if difficulty is None:
        return loss_weighting, None, None, int(getattr(args, "epochs", 0))
    score = float(difficulty["score"])
    easy_thr = float(getattr(args, "difficulty_easy", 0.25))
    hard_thr = float(getattr(args, "difficulty_hard", 0.6))
    level = "easy"
    epoch_factor = 1.0
    opening_factor = 1.0
    if score < easy_thr:
        level = "easy"
    elif score < hard_thr:
        level = "medium"
        epoch_factor = float(getattr(args, "adaptive_epochs_medium", 1.5))
        opening_factor = float(getattr(args, "adaptive_opening_weight_medium", 1.5))
    else:
        level = "hard"
        epoch_factor = float(getattr(args, "adaptive_epochs_hard", 2.0))
        opening_factor = float(getattr(args, "adaptive_opening_weight_hard", 2.0))
    base_epochs = int(getattr(args, "epochs", 0))
    adapted_epochs = int(round(base_epochs * epoch_factor))
    min_epochs = int(getattr(args, "adaptive_min_epochs", 1))
    max_epochs = int(getattr(args, "adaptive_max_epochs", adapted_epochs))
    adapted_epochs = max(min_epochs, min(adapted_epochs, max_epochs))

    adapted_loss_weighting = copy.deepcopy(loss_weighting or {})
    for key in (
        "loss_openings_p",
        "loss_openings_n",
        "loss_opening_area",
        "loss_opening_boundary_smooth",
        "loss_opening_overlap",
        "loss_grid_occupancy",
    ):
        if key in adapted_loss_weighting:
            adapted_loss_weighting[key] = float(adapted_loss_weighting[key]) * opening_factor

    difficulty["level"] = level
    adaptation = {
        "level": level,
        "epoch_factor": float(epoch_factor),
        "opening_weight_factor": float(opening_factor),
        "epochs_base": int(base_epochs),
        "epochs_adapted": int(adapted_epochs),
    }
    return adapted_loss_weighting, difficulty, adaptation, adapted_epochs


def _prune_zero_weight_losses(loss_weighting, eps: float = 1e-12):
    kept = {}
    removed = []
    for key, val in (loss_weighting or {}).items():
        w = float(val)
        if abs(w) <= eps:
            removed.append(key)
            continue
        kept[key] = w
    if removed:
        print("[two_stage_fitting] Pruned zero-weight losses: {}".format(", ".join(sorted(removed))))
    return kept


def _apply_consistency_scale_floor(loss_weighting, scale: float):
    if "loss_consistency" not in loss_weighting:
        return
    base = float(loss_weighting["loss_consistency"])
    if abs(base) <= 1e-12:
        return
    # Keep consistency high in both stages; users can only scale it up (>=1.0).
    eff = max(float(scale), 1.0)
    loss_weighting["loss_consistency"] = base * eff


def _build_stage_schedule(args, base_loss_weighting):
    total_epochs = int(getattr(args, "epochs", 0))
    default_schedule = [{
        "name": "single_stage",
        "epochs": total_epochs,
        "loss_weighting": _prune_zero_weight_losses(copy.deepcopy(base_loss_weighting or {})),
    }]
    if int(getattr(args, "two_stage_fitting", 0)) != 1 or total_epochs < 2:
        return default_schedule

    ratio = float(getattr(args, "stage1_epoch_ratio", 0.35))
    ratio = min(max(ratio, 0.05), 0.95)
    stage1_epochs = int(round(total_epochs * ratio))
    stage1_epochs = max(1, min(stage1_epochs, total_epochs - 1))
    stage2_epochs = total_epochs - stage1_epochs

    stage1 = copy.deepcopy(base_loss_weighting or {})
    stage2 = copy.deepcopy(base_loss_weighting or {})

    shape_match_keys = (
        "loss_p0",
        "loss_n1",
        "loss_do",
        "loss_diff_centreline",
    )
    regularizer_keys = (
        "loss_rigid",
        "loss_laplacian",
        "loss_edge",
        "loss_volume",
    )
    opening_keys = (
        "loss_openings_p",
        "loss_openings_n",
        "loss_opening_area",
        "loss_opening_boundary_smooth",
    )

    stage1_shape_scale = float(getattr(args, "stage1_shape_match_weight_scale", 1.8))
    stage1_reg_scale = float(getattr(args, "stage1_regularizer_weight_scale", 0.65))
    stage1_opening_scale = float(getattr(args, "stage1_opening_weight_scale", 0.2))
    stage2_shape_scale = float(getattr(args, "stage2_shape_match_weight_scale", 0.8))
    stage2_opening_scale = float(getattr(args, "stage2_opening_weight_scale", 2.0))

    for key in shape_match_keys:
        if key in stage1:
            stage1[key] = float(stage1[key]) * stage1_shape_scale
        if key in stage2:
            stage2[key] = float(stage2[key]) * stage2_shape_scale
    for key in regularizer_keys:
        if key in stage1:
            stage1[key] = float(stage1[key]) * stage1_reg_scale
    for key in opening_keys:
        if key in stage1:
            stage1[key] = float(stage1[key]) * stage1_opening_scale
        if key in stage2:
            stage2[key] = float(stage2[key]) * stage2_opening_scale
    if "loss_opening_overlap" in stage1:
        stage1["loss_opening_overlap"] = (
            float(stage1["loss_opening_overlap"]) * float(getattr(args, "stage1_overlap_weight_scale", 0.1))
        )
    if "loss_opening_overlap" in stage2:
        stage2["loss_opening_overlap"] = (
            float(stage2["loss_opening_overlap"]) * float(getattr(args, "stage2_overlap_weight_scale", 2.5))
        )
    if "loss_grid_occupancy" in stage1:
        stage1["loss_grid_occupancy"] = (
            float(stage1["loss_grid_occupancy"]) * float(getattr(args, "stage1_grid_occupancy_weight_scale", 0.0))
        )
    if "loss_grid_occupancy" in stage2:
        stage2["loss_grid_occupancy"] = (
            float(stage2["loss_grid_occupancy"]) * float(getattr(args, "stage2_grid_occupancy_weight_scale", 1.8))
        )

    _apply_consistency_scale_floor(stage1, float(getattr(args, "stage1_consistency_scale", 1.0)))
    _apply_consistency_scale_floor(stage2, float(getattr(args, "stage2_consistency_scale", 1.0)))

    stage1 = _prune_zero_weight_losses(stage1)
    stage2 = _prune_zero_weight_losses(stage2)

    if not stage1 and stage1_epochs > 0:
        print("[two_stage_fitting] stage1 has no active losses; moving its epochs to stage2.")
        stage2_epochs += stage1_epochs
        stage1_epochs = 0
    if not stage2 and stage2_epochs > 0:
        print("[two_stage_fitting] stage2 has no active losses; moving its epochs to stage1.")
        stage1_epochs += stage2_epochs
        stage2_epochs = 0

    schedule = []
    if stage1_epochs > 0 and stage1:
        schedule.append({
            "name": "stage1_coarse",
            "epochs": stage1_epochs,
            "loss_weighting": stage1,
            "lr_scale": float(getattr(args, "stage1_lr_scale", 1.6)),
        })
    if stage2_epochs > 0 and stage2:
        schedule.append({
            "name": "stage2_refine",
            "epochs": stage2_epochs,
            "loss_weighting": stage2,
            "lr_scale": float(getattr(args, "stage2_lr_scale", 0.7)),
        })

    if not schedule:
        return default_schedule
    return schedule


def _dump_run_config(
    log_path: str,
    args,
    loss_weighting,
    base_loss_weighting,
    base_epochs: int,
    base_chk_freq: int,
    difficulty,
    adaptation,
    stage_schedule,
    lr_base,
) -> None:
    cfg = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "case": str(getattr(args, "name_target", "")),
        "meta": str(getattr(args, "meta", "")),
        "args": dict(vars(args)),
        "loss_weighting": loss_weighting,
        "loss_weighting_base": base_loss_weighting,
        "epochs_base": int(base_epochs),
        "chk_freq_base": int(base_chk_freq),
        "difficulty": difficulty,
        "adaptive": adaptation,
        "stage_schedule": stage_schedule,
        "lr_base": float(lr_base),
    }
    cfg_path = os.path.join(log_path, "run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)



def fit_ghd(args, loss_weighting, hard_normalize=True, keep_size=True, canonical_chk=None):
    base_epochs = int(getattr(args, "epochs", 0))
    base_chk_freq = int(getattr(args, "chk_freq", 0))
    base_loss_weighting = copy.deepcopy(loss_weighting or {})
    # intialize registration
    canonical, target = initailize_registration(args, hard_normalize=hard_normalize, keep_size=keep_size)

    adapted_epochs = base_epochs
    difficulty = None
    adaptation = None
    loss_weighting, difficulty, adaptation, adapted_epochs = _apply_adaptive_policy(
        args, loss_weighting, canonical, target
    )
    if adaptation is not None:
        args.epochs = adapted_epochs
        if int(getattr(args, "adaptive_adjust_chk_freq", 1)) == 1:
            chk_num = int(getattr(args, "chk_num", 0))
            if chk_num > 0:
                default_base = int(round(base_epochs / chk_num))
                if int(base_chk_freq) == int(default_base):
                    args.chk_freq = max(1, int(round(adapted_epochs / chk_num)))
        print(
            "[adaptive_fitting] level={} score={:.4f} epochs {} -> {} opening_weight_factor={:.2f}".format(
                adaptation.get("level"),
                float(difficulty.get("score", 0.0)),
                base_epochs,
                adapted_epochs,
                float(adaptation.get("opening_weight_factor", 1.0)),
            )
        )
    elif int(getattr(args, "adaptive_fitting", 0)) == 1:
        print("[adaptive_fitting] opening difficulty unavailable; using base epochs/losses")

    stage_schedule = _build_stage_schedule(args, loss_weighting)
    total_stage_epochs = int(sum(int(s.get("epochs", 0)) for s in stage_schedule))
    if total_stage_epochs <= 0:
        print("No epochs scheduled after stage setup. Nothing to optimize.")
        args.epochs = base_epochs
        args.chk_freq = base_chk_freq
        return
    if len(stage_schedule) > 1:
        for idx, st in enumerate(stage_schedule):
            active = ",".join(sorted(st.get("loss_weighting", {}).keys()))
            print(
                f"[two_stage_fitting] {idx+1}/{len(stage_schedule)} {st.get('name')} "
                f"epochs={int(st.get('epochs', 0))} active_losses=[{active}]"
            )
    else:
        st = stage_schedule[0]
        active = ",".join(sorted(st.get("loss_weighting", {}).keys()))
        print(
            f"[two_stage_fitting] single stage epochs={int(st.get('epochs', 0))} "
            f"active_losses=[{active}]"
        )

    # create graph fitter and losser
    _eigen_chk_path = canonical_chk if (canonical_chk is not None and os.path.exists(canonical_chk)) else None
    canonical_fitter = Graph_Harmonic_Deform_opening_alignment_dynamic(args, canonical, eigen_chk=_eigen_chk_path)
    if canonical_chk is not None and not os.path.exists(canonical_chk):
        chk = {
            "GBH_eigval": getattr(canonical_fitter, "GBH_eigval").detach().cpu(),
            "GBH_eigvec": getattr(canonical_fitter, "GBH_eigvec").detach().cpu(),
        }
        with open(canonical_chk, "wb") as f:
            pickle.dump(chk, f)
        print(f"[fit_ghd] Saved canonical eigenvector checkpoint to {canonical_chk}")

    pouch_only = bool(getattr(args, "pouch_only", 0))
    need_do_loss = any(
        ("loss_do" in st.get("loss_weighting", {}))
        and (float(st.get("loss_weighting", {}).get("loss_do", 0.0)) > 0.0)
        for st in stage_schedule
    )
    if pouch_only:
        mesh_losser = Mesh_loss_pouch_only(args, canonical, target)
        query_points = None
        do_gt = None
    else:
        mesh_losser = Mesh_loss_do_differentiable_centreline(args, canonical, target)
        if need_do_loss:
            query_points, do_gt = mesh_losser.get_static_mask_and_gt(style=args.do_style)
            if args.do_loss_type == "dice_loss_attention":
                print("using attention dice loss, calculating attention weight map now")
                mesh_losser.get_weights_attention(
                    query_points, min_w=1.0, max_w=args.attention_max_w, smooth=args.attention_smooth, inspect=False
                )
            query_points = query_points.to(torch.device(args.device))
            do_gt = do_gt.to(torch.device(args.device))
        else:
            query_points = None
            do_gt = None
            print("loss_do is disabled: skipping occupancy query generation")
    if pouch_only and hasattr(mesh_losser, "pouch_target_opening_idx"):
        debug_target_openings = [
            mesh_losser.target_openings[mesh_losser.pouch_target_opening_idx].to(torch.device(args.device))
        ]
    else:
        debug_target_openings = [x.to(torch.device(args.device)) for x in getattr(mesh_losser, "target_openings", [])]

    # thickness loss
    thinknesser = MeshThickness(r=0.2, num_bundle_filtered=100, innerp_threshold=0.6, num_sel=25)

    # training manager
    log_path = os.path.join(args.save_root, args.name_target, args.meta)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    loss_log_path = os.path.join(log_path, "loss_log.jsonl")
    with open(loss_log_path, "w", encoding="utf-8") as f:
        f.write("")
    print(f"Loss log file: {loss_log_path}")
    if int(getattr(args, "save_run_config", 1)) == 1:
        _dump_run_config(
            log_path,
            args,
            loss_weighting,
            base_loss_weighting,
            base_epochs,
            base_chk_freq,
            difficulty,
            adaptation,
            stage_schedule,
            args.lr,
        )
    optimizer = torch.optim.AdamW(
        [canonical_fitter.deformation_param, canonical_fitter.s, canonical_fitter.T, canonical_fitter.R], lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.75)
    writer = SummaryWriter(log_path)
    if pouch_only:
        do_dropper = None
        use_dropper = 0
        do_index = None
        query_points_update, do_gt_update = None, None
        print("pouch_only mode: DO/DC losses disabled")
    else:
        if need_do_loss and query_points is not None:
            do_dropper = Do_Dropper(
                args,
                getattr(mesh_losser, "weights_attention"),
                num_queries=query_points.shape[0],
                drop_num=25,
                drop_rate=0.75,
            )
            use_dropper = getattr(args, "use_do_dropper", 0)
            print("using do dropper") if use_dropper == 1 else print("using static do")
            query_points_update, do_gt_update = query_points, do_gt
        else:
            do_dropper = None
            use_dropper = 0
            do_index = None
            query_points_update, do_gt_update = None, None
            print("loss_do is disabled: skipping Do_Dropper updates")

    # main_loop
    global_epoch = 0
    warped_mesh, warped_openings = None, None
    for stage_idx, stage in enumerate(stage_schedule):
        stage_name = str(stage.get("name", f"stage_{stage_idx+1}"))
        stage_epochs = int(stage.get("epochs", 0))
        stage_loss_weighting = copy.deepcopy(stage.get("loss_weighting", {}))
        stage_lr_scale = float(stage.get("lr_scale", 1.0))
        if stage_epochs <= 0 or not stage_loss_weighting:
            print(f"[two_stage_fitting] skipping {stage_name}: epochs={stage_epochs}, active_losses={len(stage_loss_weighting)}")
            continue
        stage_args = copy.copy(args)
        stage_args.epochs = max(stage_epochs, 1)
        stage_weighter = base_loss_weighter(stage_args, glo_loss_weighting=stage_loss_weighting, style=args.weighter_style)
        stage_lr = float(args.lr) * stage_lr_scale
        for pg in optimizer.param_groups:
            pg["lr"] = stage_lr
        print(
            f"[two_stage_fitting] starting {stage_name} ({stage_idx+1}/{len(stage_schedule)}), "
            f"epochs={stage_epochs}, lr={stage_lr:.6g}, "
            f"global_epoch_range=[{global_epoch}, {global_epoch + stage_epochs - 1}]"
        )
        for local_epoch in range(stage_epochs):
            warped_mesh, warped_openings = canonical_fitter.forward_with_opening_alignment()
            loc_loss_weighting = stage_weighter.easy_weighting(local_epoch)
            if pouch_only:
                loss_dict = mesh_losser.forward_pouch_only(warped_mesh, warped_openings, loc_loss_weighting)
            else:
                if need_do_loss and do_dropper is not None and use_dropper == 1:
                    do_index, update_do = do_dropper.forward(global_epoch)
                elif need_do_loss and do_dropper is not None:
                    do_index, update_do = do_dropper.forward(global_epoch)
                else:
                    do_index, update_do = None, False
                if update_do and query_points is not None and do_gt is not None and do_index is not None:
                    query_points_update = query_points[do_index].clone()
                    do_gt_update = do_gt[do_index].clone()
                loss_dict = mesh_losser.forward_do_dcforward_opa_do(
                    warped_mesh, warped_openings, loc_loss_weighting, query_points_update, do_gt_update, do_index
                )
            # thickness loss
            if "loss_thickness" in stage_loss_weighting:
                thickness_dict, thickness, _, sign = thinknesser.forward(warped_mesh)
                mask_thickness = torch.where(thickness.abs() > 0.1, torch.zeros_like(thickness), torch.ones_like(thickness))
                signed = torch.sign(sign)
                loss_thickness = (F.relu(0.04 - thickness * signed) + F.relu(0.01 - thickness_dict * signed)) * mask_thickness
                loss_thickness = loss_thickness.mean() + (1e-4 / (sign ** 2 + 1e-6) * mask_thickness).mean()
                loss_dict["loss_thickness"] = loss_thickness

            total_loss = torch.zeros(1, device=torch.device(args.device))
            term_log_dict = {}
            for term, loss in loss_dict.items():
                weight = float(loc_loss_weighting.get(term, 0.0))
                if term not in ["loss_openings_p", "loss_openings_n"]:
                    weighted_loss = loss * weight
                else:
                    weighted_loss = torch.sum(torch.stack(loss), dim=0) * weight
                total_loss += weighted_loss
                writer.add_scalar("Train/" + term, weighted_loss.cpu().item(), global_epoch)
                term_log_dict[term] = weighted_loss.cpu().item()
            log_dict = {
                "epoch": global_epoch,
                "stage": stage_name,
                "stage_epoch": local_epoch,
                "total_loss": float(total_loss.detach().cpu().item()),
            }
            log_dict.update(term_log_dict)
            writer.add_scalar("Train/total_loss", float(total_loss.detach().cpu().item()), global_epoch)
            # Persist per-epoch loss entries to case-specific JSONL file.
            if int(global_epoch) % 100 == 0:
                log_entry = {
                    "epoch": int(global_epoch),
                    "stage": str(stage_name),
                    "stage_epoch": int(local_epoch),
                    "case": str(args.name_target),
                    "total_loss": float(total_loss.detach().cpu().item()),
                }
                for k, v in log_dict.items():
                    if k not in {"epoch", "stage", "stage_epoch"}:
                        log_entry[k] = float(v)
                with open(loss_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
            if global_epoch % args.log_freq == 0:
                log_dict_printer(log_dict)
            if global_epoch % (4 * args.log_freq) == 0:
                print(args.name_target)

            # logging
            viz_fitting_static(
                global_epoch,
                log_path,
                warped_mesh,
                getattr(mesh_losser, "target_mesh"),
                args,
                warped_openings=warped_openings,
                target_openings=debug_target_openings,
            )

            # gradient descent
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # saving chk
            if global_epoch % args.chk_freq == 0 and global_epoch != 0:
                chk_path = os.path.join(log_path, "ghb_fitting_checkpoint_" + str(round(global_epoch / args.chk_freq)) + ".pkl")
                chk = {
                    "R": getattr(canonical_fitter, "R").detach().cpu(),
                    "s": getattr(canonical_fitter, "s").detach().cpu().abs(),
                    "T": getattr(canonical_fitter, "T").detach().cpu(),
                    "GHD_coefficient": getattr(canonical_fitter, "deformation_param").detach().cpu(),
                }
                with open(chk_path, "wb") as f:
                    pickle.dump(chk, f)
                print("GHB fitting results have been saved to {}".format(chk_path))
            global_epoch += 1

    # saving
    # Recompute after the final optimizer step so the saved mesh matches the final checkpoint.
    warped_mesh, warped_openings = canonical_fitter.forward_with_opening_alignment()
    # Always dump a final mesh snapshot, even when (epochs-1) is not a viz_freq multiple.
    final_epoch = max(global_epoch - 1, 0)
    viz_fitting_static(
        final_epoch,
        log_path,
        warped_mesh,
        getattr(mesh_losser, "target_mesh"),
        args,
        force=True,
        warped_openings=warped_openings,
        target_openings=debug_target_openings,
    )

    chk_path = os.path.join(log_path, "ghb_fitting_checkpoint.pkl")
    chk = {
        "R": getattr(canonical_fitter, "R").detach().cpu(),
        "s": getattr(canonical_fitter, "s").detach().cpu().abs(),
        "T": getattr(canonical_fitter, "T").detach().cpu(),
        "GHD_coefficient": getattr(canonical_fitter, "deformation_param").detach().cpu(),
    }
    with open(chk_path, "wb") as f:
        pickle.dump(chk, f)
    print("GHB fitting results have been saved to {}".format(chk_path))

    args.epochs = base_epochs
    args.chk_freq = base_chk_freq

def initailize_registration(args, hard_normalize=True, keep_size=True):
    print("Bold opening normal sorting = {}".format(True if args.op_bold == 1 else False))
    pouch_only = bool(getattr(args, "pouch_only", 0))
    center_opening_at_origin = bool(getattr(args, "center_opening_at_origin", 0))
    center_opening_index = int(getattr(args, "center_opening_index", 0))

    def _center_on_opening(reg_class, reg_name):
        if not hasattr(reg_class, "op_rec_v") or len(reg_class.op_rec_v) == 0:
            print(f"[center_opening_at_origin] {reg_name}: no openings found, skip.")
            return
        op_idx = min(max(center_opening_index, 0), len(reg_class.op_rec_v) - 1)
        centroid = np.asarray(reg_class.op_rec_v[op_idx], dtype=np.float64).mean(axis=0)
        reg_class.class_translate(-centroid)
        print(
            f"[center_opening_at_origin] {reg_name}: opening {op_idx} centroid "
            f"{centroid.tolist()} -> [0, 0, 0]"
        )

    canonical = RegistrationwOpeningAlignmentwDifferentiableCentreline(args, args.root_template, args.name_canonical)
    canonical.load_checkpoint_opa(None)
    canonical.sort_opening_normals(inspect_true_normal=False, clean_threshold=0.2, bold=True if args.op_bold == 1 else False)
    if center_opening_at_origin:
        _center_on_opening(canonical, "canonical")
    if not pouch_only:
        canonical.load_checkpoint_centreline(None, redo=False)
    norm_canonical = torch.max(torch.norm(getattr(canonical, "mesh_target_p3d").verts_packed(), dim=-1)).detach().item() * 1.10 if hard_normalize else 10.0
    if keep_size:
        norm_canonical = 2.50 * norm_canonical
        print("keeping same size ratio, which means canonical is normalized using 2.50 * radius")
    canonical.class_normalize(norm=norm_canonical)
    canonical.centreline_clean(radius=0.5 / norm_canonical)

    target = RegistrationwOpeningAlignmentwDifferentiableCentreline(args, args.root_target, args.name_target)
    target.load_checkpoint_opa(None)
    target.sort_opening_normals(inspect_true_normal=False, clean_threshold=0.2, bold=True if args.op_bold == 1 else False)
    if center_opening_at_origin:
        _center_on_opening(target, "target")
    if not pouch_only:
        target.load_checkpoint_centreline(None, redo=False)
    norm_target = torch.max(torch.norm(getattr(target, "mesh_target_p3d").verts_packed(),
                                       dim=-1)).detach().item() * 1.10 if hard_normalize else 7.5
    norm_target = norm_canonical if keep_size else norm_target
    target.class_normalize(norm=norm_target)
    target.centreline_clean(radius=0.5 / norm_target)
    print("canonical and target Meshes have been normalized using radius={} and {}".format(norm_canonical, norm_target))
    return canonical, target

def Mesh_normalize(mesh: Meshes, extra_factor=0.1):
    norm = torch.max(torch.norm(mesh.verts_packed(), dim=-1)).detach().item() * (1+extra_factor)
    original_mesh_verts = mesh.verts_padded().float()
    updated_mesh_verts = original_mesh_verts / norm
    normalized_mesh = mesh.update_padded(updated_mesh_verts)
    return normalized_mesh
