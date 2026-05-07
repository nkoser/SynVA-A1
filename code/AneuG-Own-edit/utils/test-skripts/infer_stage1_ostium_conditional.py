import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


class _AneuGWandbStub:
    def __getattr__(self, _name):
        def _noop(*_args, **_kwargs):
            return None
        return _noop


sys.modules.setdefault("wandb", _AneuGWandbStub())

import torch
from pytorch3d.io import save_obj
from pytorch3d.ops import taubin_smoothing
from pytorch3d.structures import join_meshes_as_batch

from first_stage_ostium_conditional import (
    collect_available_cases,
    compute_fitting_norm_canonical,
    load_or_create_split,
    load_split_from_folders,
    ensure_canonical_diff_checkpoint,
    prepare_ghd_condition_opa_checkpoints,
    resolve_case_identifier,
)
from ghd.base.graph_harmonic_deformation import Graph_Harmonic_Deform_opening_alignment_dynamic
from ghd.fitting.fitter import initailize_registration
from models.ghd_reconstruct import GHD_Reconstruct
from models.vae_datasets import OstiumGHDDataset
from models.vae_models import ConditionalGHDVAE
from utils.utils import safe_load_mesh
from aneug_method_adapter import ExternalAneuGSampler


def infer_model_hparams(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int, int, str]:
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


def denormalize_target(dataset: OstiumGHDDataset, target_norm: torch.Tensor) -> torch.Tensor:
    return target_norm * dataset.target_std.to(target_norm.device) + dataset.target_mean.to(target_norm.device)


def mesh_from_target_vector(
    ghd_reconstruct: GHD_Reconstruct,
    target_raw: torch.Tensor,
    apply_scale: bool,
) -> torch.Tensor:
    ghd = target_raw[:, :-1]
    scale = target_raw[:, -1:] if apply_scale else None
    return ghd_reconstruct.ghd_forward_as_Meshes(ghd, denormalize_shape=False, scale=scale)


def _zero_pose(batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rotation = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)
    translation = torch.zeros((batch_size, 3), dtype=torch.float32, device=device)
    return rotation, translation


def build_replay_args(alignment_root: Path, case_name: str, device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        device=str(device),
        root_template=str(alignment_root),
        root_target=str(alignment_root),
        name_canonical="canonical_model",
        name_target=case_name,
        num_op=1,
        num_cep=3,
        num_waves=5,
        step_size=2,
        op_bold=1,
        pouch_only=1,
        center_opening_at_origin=0,
        center_opening_index=0,
        num_Basis=12 ** 2,
        mix_lap_weights=[1.0, 0.1, 0.1],
    )


def resolve_canonical_eigen_chk(canonical_root: Path, ghd_chk_root: Path, explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    candidates.extend(
        [
            canonical_root / "canonical_model_144_normed.pkl",
            ghd_chk_root / "canonical_model_144_normed.pkl",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find canonical_model_144_normed.pkl. "
        f"Tried: {[str(p) for p in candidates]}"
    )


def build_case_fitter(
    alignment_root: Path,
    case_name: str,
    device: torch.device,
    canonical_eigen_chk: Path,
):
    args = build_replay_args(alignment_root=alignment_root, case_name=case_name, device=device)
    canonical, _ = initailize_registration(args, hard_normalize=True, keep_size=False)
    fitter = Graph_Harmonic_Deform_opening_alignment_dynamic(args, canonical, eigen_chk=str(canonical_eigen_chk))
    return fitter.to(device)


def meshes_from_target_vector_current_fitter(
    case_fitter,
    target_raw: torch.Tensor,
    rotation_axis_angle: torch.Tensor,
    translation: torch.Tensor,
    apply_scale: bool,
):
    target_raw = target_raw.to(case_fitter.device)
    rotation_axis_angle = rotation_axis_angle.to(case_fitter.device)
    translation = translation.to(case_fitter.device)
    meshes = []
    ghd_dim = target_raw.shape[1] - 1
    with torch.no_grad():
        for idx in range(target_raw.shape[0]):
            case_fitter.R.data = rotation_axis_angle[idx:idx + 1].reshape(1, 3).float()
            case_fitter.T.data = translation[idx:idx + 1].reshape(1, 3).float()
            if apply_scale:
                case_fitter.s.data = target_raw[idx:idx + 1, ghd_dim:].abs().reshape(1, 1).float()
            else:
                case_fitter.s.data = torch.ones((1, 1), device=case_fitter.device, dtype=torch.float32)
            mesh, _ = case_fitter.forward_with_opening_alignment(
                target_raw[idx, :ghd_dim].reshape(-1, 3).float()
            )
            meshes.append(mesh)
    return join_meshes_as_batch(meshes)


def parse_args():
    project_root = Path(__file__).resolve().parent
    default_checkpoints_root = project_root / "checkpoint-v2"
    parser = argparse.ArgumentParser(description="Infer pouch meshes from one ostium-conditioned Stage 1 checkpoint.")
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=default_checkpoints_root,
        help="Root directory containing checkpoint folders. Default: <project_root>/checkpoint-v2",
    )
    parser.add_argument(
        "--ghd-chk-root",
        type=Path,
        default=default_checkpoints_root / "ghd_fitting",
        help="Directory containing per-case GHD fitting outputs. Default: <checkpoints_root>/ghd_fitting",
    )
    parser.add_argument(
        "--alignment-root",
        type=Path,
        default=default_checkpoints_root / "alignment",
        help="Legacy condition root. Ignored when --condition-root is set. Default: <checkpoints_root>/alignment",
    )
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=default_checkpoints_root / "canonical_model",
        help="Directory containing canonical assets. Default: <checkpoints_root>/canonical_model",
    )
    parser.add_argument(
        "--condition-root",
        type=Path,
        default=None,
        help=(
            "Root containing per-case condition opa_checkpoint.pkl. "
            "Default: <ghd-chk-root> (generated from fitted GHD outputs)."
        ),
    )
    parser.add_argument(
        "--prepare-condition-from-ghd",
        type=int,
        default=1,
        help="1: generate missing per-case opa_checkpoint.pkl from ghd_fitting_output, 0: skip.",
    )
    parser.add_argument(
        "--force-prepare-condition-from-ghd",
        type=int,
        default=0,
        help="1: overwrite existing generated condition checkpoints, 0: keep existing files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a trained Stage 1 ostium checkpoint, e.g. checkpoint-v2/first_stage_ostium_conditional/ostium_pouch_v1/models_epoch_1000.pth",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Case name under condition-root to use as the example ostium.",
    )
    parser.add_argument(
        "--opa-path",
        type=Path,
        default=None,
        help="Optional direct path to an opa_checkpoint.pkl. Overrides --case if provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where generated meshes and metadata will be written.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sample",
        choices=["sample", "reconstruct"],
        help="sample: random generation from condition, reconstruct: encode/decode the requested case.",
    )
    parser.add_argument("--num-samples", type=int, default=4, help="Number of pouch samples to generate.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for latent sampling. If omitted, sampling is non-deterministic.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=108)
    parser.add_argument("--cond-embed-dim", type=int, default=64)
    parser.add_argument("--ring-points", type=int, default=20)
    parser.add_argument("--split-train-ratio", type=float, default=0.8)
    parser.add_argument("--split-val-ratio", type=float, default=0.1)
    parser.add_argument("--split-test-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--force-resplit", type=int, default=0)
    parser.add_argument("--train-subset-limit", type=int, default=None)
    parser.add_argument("--apply-scale", type=int, default=1, help="1 to apply predicted scale to exported meshes.")
    parser.add_argument("--denormalize-shape", type=int, default=1, help="1 to export meshes in denormalized canonical space.")
    parser.add_argument(
        "--canonical-eigen-chk",
        type=Path,
        default=None,
        help="Optional explicit path to canonical_model_144_normed.pkl. Default: canonical_root or ghd_chk_root.",
    )
    parser.add_argument(
        "--posterior-noise-scale",
        type=float,
        default=0.0,
        help="Noise scale used in reconstruct mode. 0 decodes the posterior mean deterministically.",
    )
    parser.add_argument("--taubin-iter", type=int, default=10, help="Number of Taubin smoothing iterations (0 to disable).")
    parser.add_argument("--taubin-lambda", type=float, default=0.53, help="Taubin smoothing lambda (positive step).")
    parser.add_argument("--taubin-mu", type=float, default=-0.53, help="Taubin smoothing mu (negative step, prevents shrinkage).")
    parser.add_argument("--external-method-type", choices=["A", "B", "C", "D", "E", "W", "baseline"], default=None,
                        help="Use a /path/to/SynVA-A1 method instead of this repo's ConditionalGHDVAE for Stage-1 sampling.")
    parser.add_argument("--external-method-checkpoint", type=Path, default=None,
                        help="Checkpoint for --external-method-type.")
    parser.add_argument("--external-aneug-root", type=Path, default=Path("/path/to/SynVA-A1"),
                        help="Path to the AneuG repo containing methods/eval_all.py.")
    parser.add_argument("--external-temperature", type=float, default=0.8)
    parser.add_argument("--external-top-k", type=int, default=0)
    parser.add_argument("--external-flow-steps", type=int, default=64)
    parser.add_argument("--external-flow-sampler", choices=["euler", "heun"], default="heun")
    return parser.parse_args()


def resolve_opa_path(args, condition_root: Path, available_cases) -> tuple[Path, str]:
    if args.opa_path is not None:
        return args.opa_path, args.opa_path.parent.name
    if args.case is None:
        raise ValueError("Provide either --case or --opa-path.")
    resolved_case = resolve_case_identifier(args.case, available_cases)
    return condition_root / resolved_case / "opa_checkpoint.pkl", resolved_case


def maybe_apply_training_stats(dataset, checkpoint, available_cases, args, checkpoints_root, ghd_chk_root, condition_root):
    if all(key in checkpoint for key in ("target_mean", "target_std", "cond_mean", "cond_std")):
        dataset.target_mean = checkpoint["target_mean"].float()
        dataset.target_std = checkpoint["target_std"].float()
        dataset.cond_mean = checkpoint["cond_mean"].float()
        dataset.cond_std = checkpoint["cond_std"].float()
        print("Loaded normalization statistics from checkpoint.")
        return

    split_cases = load_split_from_folders(
        ghd_chk_root=ghd_chk_root,
        available_cases=available_cases,
        split_val_ratio=float(args.split_val_ratio),
        split_seed=int(args.split_seed),
    )
    if split_cases is None:
        if args.split_file is None:
            split_file = checkpoints_root / "dataset_splits" / f"ostium_conditional_split_seed{args.split_seed}.json"
        else:
            split_file = args.split_file.expanduser()
        split_cases = load_or_create_split(split_file, available_cases, args)

    if args.train_subset_limit is not None:
        split_cases = {
            "train": split_cases["train"][: int(args.train_subset_limit)],
            "val": split_cases["val"],
            "test": split_cases["test"],
        }

    case_to_index = {case: idx for idx, case in enumerate(dataset.updated_cases)}
    train_indices = [case_to_index[case] for case in split_cases["train"] if case in case_to_index]
    if not train_indices:
        raise RuntimeError("Could not recover any train cases to rebuild normalization statistics.")

    if dataset.withscale:
        train_targets = torch.stack(
            [torch.cat([dataset.ghd[idx], dataset.scale[idx]]) for idx in train_indices], dim=0
        )
    else:
        train_targets = torch.stack([dataset.ghd[idx] for idx in train_indices], dim=0)
    train_conditions = torch.stack([dataset.ostium_condition[idx] for idx in train_indices], dim=0)
    dataset.target_mean = train_targets.mean(dim=0, keepdim=True)
    dataset.target_std = train_targets.std(dim=0, keepdim=True, unbiased=False) + 0.01
    dataset.cond_mean = train_conditions.mean(dim=0, keepdim=True)
    dataset.cond_std = train_conditions.std(dim=0, keepdim=True, unbiased=False) + 0.01
    print(
        "Rebuilt normalization statistics from train split "
        f"(train_cases={len(train_indices)}, train_subset_limit={args.train_subset_limit})."
    )


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    project_root = Path(__file__).resolve().parent
    checkpoints_root = args.checkpoints_root.expanduser()
    ghd_chk_root = args.ghd_chk_root.expanduser()
    alignment_root = args.alignment_root.expanduser()
    canonical_root = args.canonical_root.expanduser()
    condition_root = args.condition_root.expanduser() if args.condition_root is not None else ghd_chk_root
    canonical_eigen_chk = resolve_canonical_eigen_chk(canonical_root, ghd_chk_root, args.canonical_eigen_chk)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    canonical_meshes_raw = safe_load_mesh(str(canonical_root / "part_aligned.obj"))
    ensure_canonical_diff_checkpoint(canonical_root, canonical_meshes_raw)

    # Build NORMALIZED canonical (matching GHD fitting's normalization)
    from pytorch3d.structures import Meshes as P3dMeshes
    v_raw = canonical_meshes_raw.verts_packed()
    norm_canonical_val = compute_fitting_norm_canonical(canonical_meshes_raw)
    v_normed = v_raw / norm_canonical_val
    canonical_meshes = P3dMeshes(verts=[v_normed], faces=canonical_meshes_raw.faces_list())

    ghd_reconstruct = GHD_Reconstruct(
        canonical_meshes,
        str(canonical_root / "canonical_model_144_normed.pkl"),
        num_Basis=12**2,
        device=device,
        skip_normalize=True,
        norm_canonical_override=norm_canonical_val,
    )

    if int(args.prepare_condition_from_ghd) == 1:
        prep_summary = prepare_ghd_condition_opa_checkpoints(
            ghd_chk_root=ghd_chk_root,
            canonical_opa_chk=canonical_root / "opa_checkpoint.pkl",
            ghd_reconstruct=ghd_reconstruct,
            ghd_run="vanilla",
            ghd_chk_name="ghb_fitting_checkpoint.pkl",
            output_root=condition_root,
            force=bool(args.force_prepare_condition_from_ghd),
            condition_filename="opa_checkpoint.pkl",
            device=device,
        )
        print(
            "Condition OPA preparation: "
            f"created={prep_summary['created']}, skipped={prep_summary['skipped']}, "
            f"failed={len(prep_summary['failed'])}, root={prep_summary['output_root']}"
        )
        if prep_summary["failed"]:
            first_fail = prep_summary["failed"][:5]
            raise RuntimeError(
                f"Failed to create condition checkpoints for {len(prep_summary['failed'])} cases. "
                f"First failures: {first_fail}"
            )

    cases = collect_available_cases(
        ghd_chk_root=ghd_chk_root,
        condition_root=condition_root,
        ghd_run="vanilla",
        ghd_chk_name="ghb_fitting_checkpoint.pkl",
        condition_filename="opa_checkpoint.pkl",
    )
    if len(cases) == 0:
        raise RuntimeError(f"No valid cases found under {ghd_chk_root} with condition root {condition_root}.")

    opa_path, case_name = resolve_opa_path(args, condition_root, cases)
    if not opa_path.exists():
        raise FileNotFoundError(f"Ostium checkpoint not found: {opa_path}")

    dataset = OstiumGHDDataset(
        str(ghd_chk_root),
        str(condition_root),
        str(canonical_root / "opa_checkpoint.pkl"),
        cases,
        ghd_run="vanilla",
        ghd_chk_name="ghb_fitting_checkpoint.pkl",
        withscale=True,
        normalize=True,
        ring_points=args.ring_points,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    external_sampler = None
    if args.external_method_type is not None or args.external_method_checkpoint is not None:
        if args.external_method_type is None or args.external_method_checkpoint is None:
            raise ValueError("Both --external-method-type and --external-method-checkpoint are required for external sampling.")
        external_sampler = ExternalAneuGSampler(
            method_type=args.external_method_type,
            checkpoint=args.external_method_checkpoint,
            aneug_root=args.external_aneug_root,
            device=device,
            temperature=float(args.external_temperature),
            top_k=int(args.external_top_k),
            flow_steps=int(args.external_flow_steps),
            flow_sampler=str(args.external_flow_sampler),
        )
        print(f"Loaded external AneuG sampler: {args.external_method_type} from {args.external_method_checkpoint}")
    maybe_apply_training_stats(dataset, checkpoint, cases, args, checkpoints_root, ghd_chk_root, condition_root)
    dataset.target_mean = dataset.target_mean.cpu()
    dataset.target_std = dataset.target_std.cpu()
    dataset.cond_mean = dataset.cond_mean.cpu()
    dataset.cond_std = dataset.cond_std.cpu()

    generator = None
    latent_dim = None
    if external_sampler is None:
        vae_state = checkpoint["generator"]
        input_dim, hidden_dim, latent_dim, cond_embed_dim, norm_type = infer_model_hparams(vae_state)
        generator = ConditionalGHDVAE(
            input_dim,
            hidden_dim,
            latent_dim,
            cond_dim=dataset.get_cond_dim(),
            cond_embed_dim=cond_embed_dim,
            norm_type=norm_type,
        ).to(device)
        generator.load_state_dict(vae_state)
        generator.eval()

    ghd_dim = dataset.get_ghd_dim()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    case_name_for_file = case_name.replace("/", "__")
    metadata = {
        "checkpoint": str(args.checkpoint),
        "case": case_name,
        "opa_path": str(opa_path),
        "mode": args.mode,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "ring_points": args.ring_points,
        "apply_scale": bool(args.apply_scale),
        "denormalize_shape": bool(args.denormalize_shape),
        "canonical_eigen_chk": str(canonical_eigen_chk),
        "taubin_iter": args.taubin_iter,
        "taubin_lambda": args.taubin_lambda,
        "taubin_mu": args.taubin_mu,
        "outputs": written,
        "external_method_type": args.external_method_type,
        "external_method_checkpoint": str(args.external_method_checkpoint) if args.external_method_checkpoint is not None else None,
    }

    if args.mode == "sample":
        case_to_index = {name: idx for idx, name in enumerate(dataset.updated_cases)}
        case_fitter = build_case_fitter(alignment_root, case_name, device, canonical_eigen_chk)

        if external_sampler is not None:
            target_fake_raw = external_sampler.sample_raw_for_case(case_name, args.num_samples)
            target_fake_raw = target_fake_raw.to(device)
        else:
            cond = dataset.get_condition_from_opa_checkpoint(str(opa_path), normalize=True).to(device).unsqueeze(0)
            cond = cond.repeat(args.num_samples, 1)
            z = torch.randn(args.num_samples, latent_dim, device=device)
            with torch.no_grad():
                target_fake_norm = generator.decode(z, cond)
            target_fake_raw = denormalize_target(dataset, target_fake_norm)
        if case_name in case_to_index:
            sample = dataset[case_to_index[case_name]]
            rot = sample["alignment_rotation"].unsqueeze(0).to(device).repeat(args.num_samples, 1)
            trans = sample["alignment_translation"].unsqueeze(0).to(device).repeat(args.num_samples, 1)
            metadata["pose_source"] = "dataset_case_alignment"
        else:
            rot, trans = _zero_pose(args.num_samples, device)
            metadata["pose_source"] = "identity_pose_external_condition"
        meshes = meshes_from_target_vector_current_fitter(
            case_fitter,
            target_fake_raw,
            rotation_axis_angle=rot,
            translation=trans,
            apply_scale=dataset.withscale and int(args.apply_scale) == 1,
        )

        if args.taubin_iter > 0:
            meshes_smooth = taubin_smoothing(
                meshes, lambd=args.taubin_lambda, mu=args.taubin_mu, num_iter=args.taubin_iter
            )
            print(f"Applied {args.taubin_iter} iterations of Taubin smoothing (λ={args.taubin_lambda}, μ={args.taubin_mu})")
        else:
            meshes_smooth = None

        for idx in range(args.num_samples):
            raw_path = args.output_dir / f"{case_name_for_file}_sample_{idx:03d}_raw.obj"
            save_obj(raw_path, meshes.verts_padded()[idx].detach().cpu(), meshes.faces_padded()[idx].detach().cpu())
            written.append(str(raw_path))
            if meshes_smooth is not None:
                smooth_path = args.output_dir / f"{case_name_for_file}_sample_{idx:03d}_smooth.obj"
                save_obj(
                    smooth_path,
                    meshes_smooth.verts_padded()[idx].detach().cpu(),
                    meshes_smooth.faces_padded()[idx].detach().cpu(),
                )
                written.append(str(smooth_path))

        metadata["predicted_scale_mean"] = (
            float(target_fake_raw[:, ghd_dim:].mean().item()) if dataset.withscale else None
        )
    else:
        case_to_index = {name: idx for idx, name in enumerate(dataset.updated_cases)}
        if case_name not in case_to_index:
            raise KeyError(f"Case {case_name} not found in dataset assembly.")

        sample = dataset[case_to_index[case_name]]
        target = sample["target"].unsqueeze(0).to(device)
        cond = sample["condition"].unsqueeze(0).to(device)
        case_fitter = build_case_fitter(alignment_root, case_name, device, canonical_eigen_chk)

        with torch.no_grad():
            target_recon_norm, mu, logvar = generator(
                target, cond, noise_scale=float(args.posterior_noise_scale)
            )
        target_raw = denormalize_target(dataset, target)
        target_recon_raw = denormalize_target(dataset, target_recon_norm)
        rot = sample["alignment_rotation"].unsqueeze(0).to(device)
        trans = sample["alignment_translation"].unsqueeze(0).to(device)
        target_mesh = meshes_from_target_vector_current_fitter(
            case_fitter,
            target_raw,
            rotation_axis_angle=rot,
            translation=trans,
            apply_scale=dataset.withscale and int(args.apply_scale) == 1,
        )
        recon_mesh = meshes_from_target_vector_current_fitter(
            case_fitter,
            target_recon_raw,
            rotation_axis_angle=rot,
            translation=trans,
            apply_scale=dataset.withscale and int(args.apply_scale) == 1,
        )

        if args.taubin_iter > 0:
            recon_mesh_smooth = taubin_smoothing(
                recon_mesh, lambd=args.taubin_lambda, mu=args.taubin_mu, num_iter=args.taubin_iter
            )
            print(f"Applied {args.taubin_iter} iterations of Taubin smoothing (λ={args.taubin_lambda}, μ={args.taubin_mu})")
        else:
            recon_mesh_smooth = None

        target_path = args.output_dir / f"{case_name_for_file}_target_raw.obj"
        recon_path = args.output_dir / f"{case_name_for_file}_recon_raw.obj"
        save_obj(target_path, target_mesh.verts_padded()[0].detach().cpu(), target_mesh.faces_padded()[0].detach().cpu())
        save_obj(recon_path, recon_mesh.verts_padded()[0].detach().cpu(), recon_mesh.faces_padded()[0].detach().cpu())
        written.extend([str(target_path), str(recon_path)])

        if recon_mesh_smooth is not None:
            smooth_path = args.output_dir / f"{case_name_for_file}_recon_smooth.obj"
            save_obj(
                smooth_path,
                recon_mesh_smooth.verts_padded()[0].detach().cpu(),
                recon_mesh_smooth.faces_padded()[0].detach().cpu(),
            )
            written.append(str(smooth_path))

        metadata.update(
            {
                "posterior_noise_scale": float(args.posterior_noise_scale),
                "latent_mu_norm": float(mu.norm(dim=1).mean().item()),
                "latent_logvar_mean": float(logvar.mean().item()),
                "target_mse": float(torch.mean((target_recon_norm - target) ** 2).item()),
                "ghd_mse": float(torch.mean((target_recon_raw[:, :ghd_dim] - target_raw[:, :ghd_dim]) ** 2).item()),
                "scale_mse": (
                    float(torch.mean((target_recon_raw[:, ghd_dim:] - target_raw[:, ghd_dim:]) ** 2).item()) if dataset.withscale else None
                ),
                "predicted_scale": (
                    float(target_recon_raw[:, ghd_dim:].squeeze().item()) if dataset.withscale else None
                ),
                "target_scale": (
                    float(target_raw[:, ghd_dim:].squeeze().item()) if dataset.withscale else None
                ),
            }
        )

    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote {len(written)} mesh(es) to {args.output_dir}")


if __name__ == "__main__":
    main()
