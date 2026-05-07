import argparse
import copy
import os
import random
import time

import torch
import torch.multiprocessing as mp

from ghd.fitting.fitter import fit_ghd
from ghd.fitting.registration import RegistrationwOpeningAlignmentwDifferentiableCentreline

# conf
epochs = 2000
chk_num = 4  # number of checkpoints during fitting
register = True


def build_parser():
    parser = argparse.ArgumentParser("ghb_fitting_oa")
    parser.add_argument("--device",type=str,default="cuda:0",help='Single device (e.g. cuda:0) or multi-device list (e.g. "0,1,2,3" or "all")',)
    parser.add_argument("--root_template", type=str, default="./checkpoints-v2/alignment")
    parser.add_argument("--root_target", type=str, default="./checkpoints-v2/alignment")
    parser.add_argument("--name_canonical", type=str, default="canonical_model")
    parser.add_argument("--name_target", type=str, default="AN213_full_clean")
    parser.add_argument("--viz_freq", type=int, default=200)
    parser.add_argument("--chk_freq", type=int, default=round(epochs / chk_num))
    parser.add_argument("--chk_num",type=int,default=chk_num,help="Number of checkpoints during fitting (used to recompute chk_freq when adaptive_fitting is on)",)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_op", type=int, default=3)
    parser.add_argument("--num_Basis", type=int, default=12 ** 2)
    parser.add_argument("--mix_lap_weights", type=list, default=[1.0, 0.1, 0.1])
    parser.add_argument("--sample_num", type=int, default=int(2.5e5))
    parser.add_argument("--op_sample_num", type=int, default=int(1e3))
    parser.add_argument("--op_clean_threshold", type=float, default=0.2)
    parser.add_argument("--op_bold",type=int,default=1,help="confidence that trimesh offers uniform mesh normal directions",)
    parser.add_argument("--save_root", type=str, default="./checkpoints-v2/ghd_fitting_output1")
    parser.add_argument("--meta", type=str, default="vanilla")
    parser.add_argument("--epochs", type=int, default=int(epochs))
    parser.add_argument("--num_sp", type=int, default=2)
    parser.add_argument("--do_dpi", type=int, default=4)
    parser.add_argument("--do_style", type=str, default="number_control_v2")
    parser.add_argument("--do_loss_type", type=str, default="dice_loss_attention")
    parser.add_argument("--use_do_dropper", type=int, default=0)
    parser.add_argument("--attention_max_w", type=float, default=3.0)
    parser.add_argument("--attention_smooth", type=float, default=0.02)
    parser.add_argument("--do_number", type=int, default=25000)
    parser.add_argument("--weighter_style", type=str, default="strategy_v1_linear")
    # parser.add_argument('--weighter_warmup', type=int, default=1500)
    parser.add_argument("--pouch_only",type=int,default=1,help="1: aneurysm-only mode (single opening, no DO/DC losses)",)
    parser.add_argument("--opening_min_ratio",type=float,default=0.5,help="Minimum opening area ratio for pouch_only opening-area penalty",)
    parser.add_argument("--target_label",type=str,default="",help="If provided, only this single target case is processed",)

    #---------cpu-threads---------------------
    parser.add_argument("--cpu_threads",type=int,default=0,help="Set torch.set_num_threads(N). 0 keeps the default.",)
    parser.add_argument("--cpu_interop_threads",type=int,default=0,help="Set torch.set_num_interop_threads(N). 0 keeps the default.",)


    #----------loss functions-----------------
    parser.add_argument("--loss_rigid",type=float,default=100.0,help="Weight for rigidity regularization term",)
    parser.add_argument("--loss_openings_p",type=float,default=10,help="Weight for opening position/shape matching term in pouch_only mode",)
    parser.add_argument("--loss_p0n1_scale",type=float,default=20,help="Shared scale for loss_p0/loss_n1 (loss_p0=1.0*scale, loss_n1=0.8*scale)",)
    parser.add_argument("--loss_consistency",type=float,default=350,help="normal consistency between neighboring faces; discourages folds/flips",)
    parser.add_argument("--loss_volume",type=float,default=0.05,help="Weight for relative volume loss term in pouch_only mode",)
    parser.add_argument("--loss_opening_boundary_smooth",type=float,default=0.5,help="Weight for boundary-only opening smoothness term in pouch_only mode",)
    parser.add_argument("--loss_opening_overlap",type=float,default=1,help="Weight for opening overlap loss in pouch_only mode (0=off).",)
    parser.add_argument("--loss_grid_occupancy",type=float,default=0,help="Weight for 3D-grid inside/outside occupancy loss in pouch_only mode (0=off).",)

    #----------Debug options-------------------
    parser.add_argument("--opening_boundary_smooth_width",type=int,default=3,help="Approximate smoothing width (in vertex-ring steps) from opening boundary",)
    parser.add_argument("--debug_opening_losses",type=int,default=0,help="1: save opening loss debug OBJ (sampled points + normal arrows) with each warped_epoch OBJ",)
    parser.add_argument("--debug_opening_normal_scale",type=float,default=0.01,help="Arrow length scale for opening normal visualization in debug OBJ",)
    parser.add_argument("--debug_opening_point_marker_scale",type=float,default=0.0035,help="Cross-marker size for opening vertices in debug OBJ",)
    parser.add_argument("--opening_overlap_sigma_ratio",type=float,default=0.1,help="Scale for overlap-kernel sigma relative to target opening radius.",)

    parser.add_argument("--grid_occupancy_dpi",type=int,default=18,help="Grid density for occupancy-loss point generation over bounding box.",)
    parser.add_argument("--grid_occupancy_max_points",type=int,default=12000,help="Maximum cached occupancy grid points (uniform downsample if exceeded).",)
    parser.add_argument("--grid_occupancy_samples_per_step",type=int,default=0,help="Number of occupancy grid points sampled per training step (<=0 uses all).",)
    parser.add_argument("--grid_occupancy_chunk_size",type=int,default=1024,help="Chunk size for winding occupancy evaluation to avoid OOM.",)
    parser.add_argument("--grid_occupancy_loss_type",type=str,default="mse",choices=["mse", "dice"],help="Loss type for occupancy grid term.",)

    parser.add_argument("--center_opening_at_origin",type=int,default=0,help="1: translate each case so selected opening centroid is at world origin before normalization",)
    parser.add_argument("--center_opening_index",type=int,default=0,help="Opening index used for center_opening_at_origin",)

    #------------------two-stage fitting--------------
    parser.add_argument("--two_stage_fitting",type=int,default=0,help="1: run a 2-stage fitting schedule (coarse->refine).",)
    parser.add_argument("--stage1_epoch_ratio",type=float,default=0.35,help="Fraction of epochs used in stage 1 (coarse fit).",)
    parser.add_argument("--stage1_shape_match_weight_scale",type=float,default=1.8,help="Stage-1 multiplier for global shape-matching losses (p0/n1/do/centreline).",)
    parser.add_argument("--stage1_regularizer_weight_scale",type=float,default=0.65,help="Stage-1 multiplier for regularization losses (rigid/laplacian/edge/volume).",)
    parser.add_argument("--stage1_opening_weight_scale",type=float,default=0.2,help="Stage-1 multiplier for opening-related losses (except overlap/grid).",)
    parser.add_argument("--stage1_overlap_weight_scale",type=float,default=0.1,help="Stage-1 multiplier for loss_opening_overlap.",)
    parser.add_argument("--stage1_grid_occupancy_weight_scale",type=float,default=0.0,help="Stage-1 multiplier for loss_grid_occupancy.",)
    parser.add_argument("--stage1_lr_scale",type=float,default=1.6,help="Stage-1 learning-rate multiplier (larger -> more drastic updates).",)
    parser.add_argument("--stage2_shape_match_weight_scale",type=float,default=0.8,help="Stage-2 multiplier for global shape-matching losses (p0/n1/do/centreline).",)
    parser.add_argument("--stage2_opening_weight_scale",type=float,default=2.0,help="Stage-2 multiplier for opening-related losses (except overlap/grid).",)
    parser.add_argument("--stage2_overlap_weight_scale",type=float,default=2.5,help="Stage-2 multiplier for loss_opening_overlap.",)
    parser.add_argument("--stage2_grid_occupancy_weight_scale",type=float,default=1.8,help="Stage-2 multiplier for loss_grid_occupancy.",)
    parser.add_argument("--stage2_lr_scale",type=float,default=0.7,help="Stage-2 learning-rate multiplier.",)
    parser.add_argument("--stage1_consistency_scale",type=float,default=1.0,help="Stage-1 multiplier for loss_consistency (clamped to >=1.0).",)
    parser.add_argument("--stage2_consistency_scale",type=float,default=1.0,help="Stage-2 multiplier for loss_consistency (clamped to >=1.0).",)

    #------------------adaptive training--------------
    parser.add_argument("--adaptive_fitting",type=int,default=0,help="1: adapt epochs + opening-related loss weights based on opening mismatch difficulty",)
    parser.add_argument("--difficulty_easy",type=float,default=0.25,help="Difficulty score threshold for easy cases",)
    parser.add_argument("--difficulty_hard",type=float,default=0.6,help="Difficulty score threshold for hard cases",)
    parser.add_argument("--adaptive_epochs_medium",type=float,default=1.5,help="Epoch multiplier for medium difficulty cases",)
    parser.add_argument("--adaptive_epochs_hard",type=float,default=2.0,help="Epoch multiplier for hard difficulty cases",)
    parser.add_argument("--adaptive_opening_weight_medium",type=float,default=1.5,help="Opening-related loss weight multiplier for medium difficulty cases",)
    parser.add_argument("--adaptive_opening_weight_hard",type=float,default=2.0,help="Opening-related loss weight multiplier for hard difficulty cases",)
    parser.add_argument("--adaptive_min_epochs",type=int,default=500,help="Minimum epochs when adaptive_fitting is enabled",)
    parser.add_argument("--adaptive_max_epochs",type=int,default=30000,help="Maximum epochs when adaptive_fitting is enabled",)
    parser.add_argument("--adaptive_adjust_chk_freq",type=int,default=1,help="1: recompute chk_freq when adaptive epochs are used (if chk_freq was defaulted)",)
    parser.add_argument("--save_run_config",type=int,default=1,help="1: save per-case run_config.json in the output folder",)
    parser.add_argument(
        "--canonical_eigen_chk",
        type=str,
        default="checkpoints-new/canonical_model/canonical_model_144_normed.pkl",
        help=(
            "Path to pre-computed canonical eigenvector checkpoint (.pkl). "
            "If provided and file exists, GHD fitting loads these eigenvectors instead of computing fresh ones. "
            "If provided but file does not exist, eigenvectors are computed once and saved to this path. "
            "Use create_canonical_eigen_checkpoint.py to generate this file."
        ),
    )
    return parser


def _unique_keep_order(seq):
    seen = set()
    out = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _parse_device_list(device_arg):
    if not device_arg:
        return []
    device_str = device_arg.strip().lower()
    if device_str in {"all", "cuda:all"}:
        count = torch.cuda.device_count()
        return list(range(count))
    if "," not in device_arg:
        return []
    parts = []
    for token in device_arg.split(","):
        tok = token.strip()
        if not tok:
            continue
        if tok.lower().startswith("cuda:"):
            tok = tok.split("cuda:")[-1]
        if tok.isdigit():
            parts.append(int(tok))
        else:
            raise ValueError(f"Unrecognized device token '{token}' in --device")
    return _unique_keep_order(parts)


def _normalize_single_device(device_arg):
    if device_arg and device_arg.isdigit():
        return f"cuda:{device_arg}"
    return device_arg


def _prune_zero_weight_losses(loss_weighting, eps=1e-12):
    if not loss_weighting:
        return {}
    kept = {}
    removed = []
    for key, val in loss_weighting.items():
        weight = float(val)
        if abs(weight) <= eps:
            removed.append(key)
            continue
        kept[key] = weight
    if removed:
        print("[loss_weighting] Skipping zero-weight losses: {}".format(", ".join(sorted(removed))))
    return kept


def _build_loss_weighting(args):
    if args.pouch_only == 1:
        args.num_op = 1
        print("Running pouch_only mode: using 1 opening, disabling DO/DC losses.")
        p0_w = 1.0 * args.loss_p0n1_scale
        n1_w = 0.8 * args.loss_p0n1_scale
        loss_weighting = {
            "loss_p0": p0_w,  # point-to-point shape distance
            "loss_n1": n1_w,  # normal alignment term from Chamfer normals
            "loss_laplacian": 0.1,  # cot-Laplacian smoothing regularizer; reduces noisy local deformations
            "loss_edge": 0.1,  # keeps edge lengths close to a reference average edge length
            "loss_consistency": args.loss_consistency,  # normal consistency between neighboring faces; discourages folds/flips
            "loss_rigid": args.loss_rigid,  # ARAP-like local rigidity term (tries to preserve local shape under near-rigid transforms)
            "loss_openings_p": args.loss_openings_p,  # matches opening geometry between warped and target
            "loss_openings_n": 0.1,  # opening normal alignment (single opening in pouch_only mode)
            "loss_opening_area": 1,  # prevents ostium collapse if area too small
            "loss_volume": args.loss_volume,  # relative absolute-volume error between warped and target mesh
            "loss_opening_boundary_smooth": args.loss_opening_boundary_smooth,
            "loss_opening_overlap": args.loss_opening_overlap,  # overlap-aware opening alignment: 0 best, 1 worst
            "loss_grid_occupancy": args.loss_grid_occupancy,  # compares inside/outside on shared 3D grid points
        }
    else:
        p0_w = 1.0 * args.loss_p0n1_scale
        n1_w = 0.8 * args.loss_p0n1_scale
        loss_weighting = {
            "loss_do": 1.0,
            "loss_p0": p0_w,
            "loss_n1": n1_w,
            "loss_laplacian": 0.1,
            "loss_edge": 0.1,
            "loss_consistency": 0.1,
            "loss_rigid": args.loss_rigid,
            "loss_openings_p": args.loss_openings_p,
            "loss_openings_n": 0.1,
            "loss_diff_centreline": 10.0,
        }
    loss_weighting = _prune_zero_weight_losses(loss_weighting)
    return loss_weighting


def _configure_torch_threads(args, prefix=None):
    cpu_threads = int(getattr(args, "cpu_threads", 0) or 0)
    cpu_interop = int(getattr(args, "cpu_interop_threads", 0) or 0)
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
    if cpu_interop > 0:
        torch.set_num_interop_threads(cpu_interop)
    if cpu_threads > 0 or cpu_interop > 0:
        head = f"{prefix} " if prefix else ""
        print(
            f"{head}Torch CPU threads set to intra={torch.get_num_threads()}, "
            f"interop={torch.get_num_interop_threads()}"
        )


def _discover_valid_labels(args):
    label_list = [
        label
        for label in os.listdir(args.root_target)
        if os.path.isdir(os.path.join(args.root_target, label)) and label != args.name_canonical
    ]
    if args.target_label:
        if args.target_label in label_list:
            label_list = [args.target_label]
        else:
            print(
                f"Requested target_label '{args.target_label}' not found under {args.root_target}. Nothing to process."
            )
            label_list = []
    random.shuffle(label_list)
    valid_labels = []

    if register:
        for label in label_list:
            args.name_target = label
            has_opa = os.path.exists(os.path.join(args.root_target, args.name_target, "opa_checkpoint.pkl"))
            has_diff = os.path.exists(os.path.join(args.root_target, args.name_target, "diff_centreline_checkpoint.pkl"))
            is_valid = has_opa if args.pouch_only == 1 else (has_opa and has_diff)
            if is_valid:
                valid_labels.append(label)
                print("Registration for case {} has been found, skipping".format(label))
            else:
                print("Registration for case {} not found (opa={}, diff={})".format(label, has_opa, has_diff))
                continue
                target = RegistrationwOpeningAlignmentwDifferentiableCentreline(
                    args, args.root_target, args.name_target
                )
                target.load_checkpoint_opa(None, redo=False)
                target.load_checkpoint_centreline(None, redo=False)
                norm_target = (
                    torch.max(torch.norm(getattr(target, "mesh_target_p3d").verts_packed(), dim=-1))
                    .detach()
                    .item()
                )
                target.class_normalize(norm=norm_target)
                target.centreline_clean(radius=0.5 / norm_target)
                target.visualize_centreline(norm_target)
    return valid_labels


def _run_single_case(args, loss_weighting, label, idx, total, device_id=None):
    args.name_target = label
    prefix = f"[GPU {device_id}] " if device_id is not None else ""
    print(f"{prefix}Now performing ghd fitting for case {label} ({idx+1}/{total})")
    chk_path = os.path.join(args.save_root, args.name_target, args.meta, "ghb_fitting_checkpoint.pkl")
    if not os.path.exists(chk_path):
        copied_loss_weighting = loss_weighting.copy()
        start_time = time.time()
        canonical_chk = args.canonical_eigen_chk if args.canonical_eigen_chk else None
        fit_ghd(args, copied_loss_weighting, hard_normalize=True, keep_size=False, canonical_chk=canonical_chk)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{prefix}Sample {label} fitting time: {elapsed:.2f} seconds ({elapsed/60:.2f} min)")
        return elapsed
    print(f"{prefix}Skipping ghd fitting for case {label}")
    return None


def _fit_worker(device_id, task_queue, result_queue, base_args, loss_weighting, total_cases):
    _configure_torch_threads(base_args, prefix=f"[GPU {device_id}]")
    torch.cuda.set_device(device_id)
    while True:
        item = task_queue.get()
        if item is None:
            break
        idx, label = item
        args = copy.deepcopy(base_args)
        args.device = f"cuda:{device_id}"
        try:
            elapsed = _run_single_case(args, loss_weighting, label, idx, total_cases, device_id=device_id)
            result_queue.put(("ok", label, device_id, elapsed, None))
        except Exception as exc:
            result_queue.put(("err", label, device_id, None, repr(exc)))


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.device = _normalize_single_device(args.device)
    _configure_torch_threads(args)

    loss_weighting = _build_loss_weighting(args)
    if not loss_weighting:
        print("No active losses after pruning zero-weight terms. Nothing to optimize.")
        return
    valid_labels = _discover_valid_labels(args)

    if not valid_labels:
        print("No valid labels to process.")
        return

    device_list = _parse_device_list(args.device)
    if device_list:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but multiple devices were requested.")
        max_device = max(device_list) if device_list else -1
        count = torch.cuda.device_count()
        if max_device >= count:
            raise RuntimeError(f"Requested GPU {max_device} but only {count} CUDA devices are visible.")

    # perform ghd fitting
    if device_list and len(device_list) > 1:
        total_cases = len(valid_labels)
        print(f"Launching parallel fitting on devices: {device_list} ({total_cases} cases)")
        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()
        for idx, label in enumerate(valid_labels):
            task_queue.put((idx, label))
        for _ in device_list:
            task_queue.put(None)

        procs = []
        for device_id in device_list:
            p = ctx.Process(
                target=_fit_worker,
                args=(device_id, task_queue, result_queue, args, loss_weighting, total_cases),
            )
            p.start()
            procs.append(p)

        sample_times = []
        finished = 0
        while finished < total_cases:
            status, label, device_id, elapsed, err = result_queue.get()
            finished += 1
            if status == "err":
                print(f"[GPU {device_id}] Error while processing {label}: {err}")
            elif elapsed is not None:
                sample_times.append(elapsed)

        for p in procs:
            p.join()
    else:
        sample_times = []
        for idx, label in enumerate(valid_labels):
            elapsed = _run_single_case(args, loss_weighting, label, idx, len(valid_labels))
            if elapsed is not None:
                sample_times.append(elapsed)

    if sample_times:
        avg_time = sum(sample_times) / len(sample_times)
        total_time = avg_time * len(valid_labels)
        print(f"\nAverage time per sample: {avg_time:.2f} seconds ({avg_time/60:.2f} min)")
        print(
            f"Estimated total time for {len(valid_labels)} samples: {total_time:.2f} seconds "
            f"({total_time/60:.2f} min, {total_time/3600:.2f} h)"
        )


if __name__ == "__main__":
    main()
