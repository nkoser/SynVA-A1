#!/usr/bin/env python
"""Generate test-set aneurysm samples and stitch them with the reference pipeline.

This wrapper keeps all outputs below --out-root.  It prepares the minimal case
folder layout expected by vessel-mesh-editing-master's run_inference_pipeline.py
step3, samples one raw canonical pouch per case from one of our method
checkpoints, then calls the reference bridge stitching code.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import trimesh
from pytorch3d.io import save_obj
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from first_stage_vessel_aware import collate_fn
from ghd.base.graph_harmonic_deformation import Graph_Harmonic_Deform_opening_alignment_dynamic
from ghd.fitting.fitter import initailize_registration
from methods._common.mesh_loss import CoeffToMesh
from methods.eval_all import METHOD_LOADERS, _build_dataset, _copy_stats
from train_vessel_flow_matching import condition_from_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cases_file", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--method", choices=["A", "B", "C", "D", "E", "W", "baseline"], default="E")
    p.add_argument("--out_root", default="outputs/reference_stitch_aneug_ghds_test")
    p.add_argument("--alignment_root", default="/path/to/aneug-ghds/data/alignment")
    p.add_argument("--ghd_root", default="/path/to/aneug-ghds/data/ghd_fitting")
    p.add_argument("--prepared_cases_root", default="/path/to/prepared_meshes_3",
                   help="Root containing original case folders with 04_subpointclouds/05_submeshes/07_other.")
    p.add_argument("--reference_pipeline", default="/path/to/SynVA-A1/code/inference/run_inference_pipeline.py")
    p.add_argument("--canonical_root", default="/path/to/aneug-ghds/data/alignment/canonical_model")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_cases", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--flow_steps", type=int, default=64)
    p.add_argument("--flow_sampler", choices=["euler", "heun"], default="heun")
    p.add_argument("--num_candidates", type=int, default=1,
                   help="Samples drawn per case before stitching.")
    p.add_argument("--select_by", choices=["first", "ostium"], default="first",
                   help="How to choose the sample passed to reference step3.")
    p.add_argument("--save_candidates", action="store_true",
                   help="Also export every candidate as candidate_XXX.obj.")
    p.add_argument("--ring_points", type=int, default=20)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--continue_on_error", action="store_true")
    p.add_argument("--no_remesh", action="store_true")
    p.add_argument("--no_smooth_transition", action="store_true")
    p.add_argument("--stitch_method", choices=["bridge", "snap"], default="bridge")
    p.add_argument("--decode_backend", choices=["reference", "coeff"], default="reference",
                   help="reference replays vessel-mesh-editing-master's per-case opening-alignment fitter; coeff uses the lightweight coeff-to-mesh path.")
    p.add_argument("--decode_canonical_norm_factor", type=float, default=2.75,
                   help="Canonical radius multiplier for GHD->mesh export. The reference Stage-1 fitter uses 1.10*2.50=2.75.")
    return p.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def ring_normal(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    center = pts.mean(axis=0)
    centered = pts - center.reshape(1, 3)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    return normal.astype(np.float64)


def _resample_closed_ring(points: np.ndarray, num_points: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] == num_points:
        return pts.copy()
    nxt = np.roll(pts, -1, axis=0)
    seg = np.linalg.norm(nxt - pts, axis=1)
    total = float(seg.sum())
    if total <= 1e-12:
        return np.repeat(pts[:1], num_points, axis=0)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    samples = np.linspace(0.0, total, num_points, endpoint=False)
    out = []
    for sample in samples:
        idx = min(np.searchsorted(cumulative, sample, side="right") - 1, len(seg) - 1)
        length = max(float(seg[idx]), 1e-12)
        alpha = (sample - cumulative[idx]) / length
        out.append((1.0 - alpha) * pts[idx] + alpha * pts[(idx + 1) % len(pts)])
    return np.asarray(out, dtype=np.float64)


def _similarity_fit_mse(source: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Return Procrustes MSE and scale fitting source -> target."""
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    src_c = src.mean(axis=0, keepdims=True)
    tgt_c = tgt.mean(axis=0, keepdims=True)
    x = src - src_c
    y = tgt - tgt_c
    cov = x.T @ y / max(1, src.shape[0])
    u, s, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    var = float((x * x).sum() / max(1, src.shape[0]))
    scale = float(s.sum() / max(var, 1e-12))
    fitted = scale * (x @ r.T) + tgt_c
    mse = float(np.mean(np.sum((fitted - tgt) ** 2, axis=1)))
    return mse, scale


def _best_ring_fit_score(pred_ring: np.ndarray, target_ring: np.ndarray) -> dict[str, object]:
    pred = _resample_closed_ring(np.asarray(pred_ring, dtype=np.float64), target_ring.shape[0])
    target = np.asarray(target_ring, dtype=np.float64)
    best = None
    for reversed_order, base in ((False, target), (True, target[::-1].copy())):
        for shift in range(target.shape[0]):
            shifted = np.roll(base, shift, axis=0)
            mse, scale = _similarity_fit_mse(pred, shifted)
            rec = {
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "scale": scale,
                "shift": int(shift),
                "reversed_order": bool(reversed_order),
            }
            if best is None or rec["mse"] < best["mse"]:
                best = rec
    assert best is not None
    pred_n = ring_normal(pred)
    target_n = ring_normal(target)
    best["abs_normal_dot"] = float(abs(np.dot(pred_n, target_n)))
    return best


def write_pointcloud_ply(points: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = trimesh.points.PointCloud(np.asarray(points, dtype=np.float64))
    cloud.export(path)


def resolve_prepared_case_root(prepared_root: Path, case: str) -> Path:
    candidates = [
        prepared_root / case,
        prepared_root / f"aneux_{case}",
        prepared_root / case.replace("cmha_", "cmch_"),
        prepared_root / f"aneux_{case.replace('cmha_', 'cmch_')}",
    ]
    for candidate in candidates:
        if (candidate / "05_submeshes" / "vessel_submesh.obj").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find prepared case with 05_submeshes/vessel_submesh.obj. Tried:\n"
        + "\n".join(str(x) for x in candidates)
    )


def prepare_case_layout(
    out_root: Path,
    case: str,
    alignment_root: Path,
    ghd_root: Path,
    canonical_root: Path,
    prepared_cases_root: Path,
    overwrite: bool,
) -> Path:
    case_root = out_root / "cases" / "test" / case
    if overwrite and case_root.exists():
        shutil.rmtree(case_root)

    align_case = alignment_root / case
    ghd_case = ghd_root / case
    align_opa_src = align_case / "opa_checkpoint.pkl"
    cond_opa_src = ghd_case / "opa_checkpoint.pkl"
    prepared_case = resolve_prepared_case_root(prepared_cases_root, case)
    vessel_src = prepared_case / "05_submeshes" / "vessel_submesh.obj"
    label2_src = prepared_case / "04_subpointclouds" / "subpointcloud_label_2.ply"
    centroid_src = prepared_case / "07_other" / "centroid_ostium.npy"
    normal_src = prepared_case / "07_other" / "normal_vector.npy"
    required = [
        vessel_src,
        label2_src,
        centroid_src,
        normal_src,
        align_opa_src,
        cond_opa_src,
        canonical_root / "opa_checkpoint.pkl",
    ]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError("Missing case assets:\n" + "\n".join(missing))

    for rel in ["04_subpointclouds", "05_submeshes", "07_other", "outputs/stage1_sample"]:
        (case_root / rel).mkdir(parents=True, exist_ok=True)
    runtime_align = case_root / "_runtime" / "alignment_vc"
    (runtime_align / case).mkdir(parents=True, exist_ok=True)
    (runtime_align / "canonical_model").mkdir(parents=True, exist_ok=True)

    shutil.copy2(vessel_src, case_root / "05_submeshes" / "vessel_submesh.obj")
    shutil.copy2(label2_src, case_root / "04_subpointclouds" / "subpointcloud_label_2.ply")
    shutil.copy2(centroid_src, case_root / "07_other" / "centroid_ostium.npy")
    shutil.copy2(normal_src, case_root / "07_other" / "normal_vector.npy")
    shutil.copy2(align_opa_src, runtime_align / case / "opa_checkpoint.pkl")
    shutil.copy2(canonical_root / "opa_checkpoint.pkl", runtime_align / "canonical_model" / "opa_checkpoint.pkl")
    shutil.copy2(canonical_root / "part_aligned.obj", runtime_align / "canonical_model" / "part_aligned.obj")

    return case_root


def build_reference_fitter(alignment_root: Path, case: str, eigen_chk: Path, device: torch.device):
    fitter_args = SimpleNamespace(
        device=str(device),
        root_template=str(alignment_root),
        root_target=str(alignment_root),
        name_canonical="canonical_model",
        name_target=case,
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
    canonical, _ = initailize_registration(fitter_args, hard_normalize=True, keep_size=False)
    fitter = Graph_Harmonic_Deform_opening_alignment_dynamic(
        fitter_args,
        canonical,
        eigen_chk=str(eigen_chk),
    )
    return fitter.to(device)


def decode_with_reference_fitter(
    samples_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    case: str,
    alignment_root: Path,
    ghd_root: Path,
    ghd_run: str,
    ghd_chk_name: str,
    eigen_chk: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay the Stage-1 mesh decode used by vessel-mesh-editing-master."""
    target_raw = samples_norm * std.to(device) + mean.to(device)
    ghd_dim = min(432, target_raw.shape[-1])
    ghd_checkpoint = ghd_root / case / ghd_run / ghd_chk_name
    ghd_chk = load_pickle(ghd_checkpoint)
    rot = ghd_chk["R"].reshape(1, 3).to(device).float()
    trans = ghd_chk["T"].reshape(1, 3).to(device).float()

    fitter = build_reference_fitter(alignment_root, case, eigen_chk, device)
    verts_out = []
    faces_out = None
    with torch.no_grad():
        for idx in range(target_raw.shape[0]):
            fitter.R.data = rot.clone()
            fitter.T.data = trans.clone()
            if target_raw.shape[-1] > ghd_dim:
                fitter.s.data = target_raw[idx:idx + 1, ghd_dim:ghd_dim + 1].abs().reshape(1, 1).float()
            else:
                fitter.s.data = torch.ones((1, 1), device=device, dtype=torch.float32)
            mesh, _ = fitter.forward_with_opening_alignment(target_raw[idx, :ghd_dim].reshape(-1, 3).float())
            verts_out.append(mesh.verts_padded()[0])
            if faces_out is None:
                faces_out = mesh.faces_padded()[0].detach().cpu()
    return torch.stack(verts_out, dim=0), faces_out


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_root = (ROOT / args.out_root).resolve() if not os.path.isabs(args.out_root) else Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cases_all = json.loads(Path(args.cases_file).read_text())
    cases = cases_all[: args.max_cases] if args.max_cases > 0 else cases_all
    device = torch.device(args.device)

    ck, sa, cond_net, sampler = METHOD_LOADERS[args.method](args.ckpt, device)
    ds = _build_dataset(cases, sa)
    _copy_stats(ds, ck)
    if "orig_ghd_mean" in ck:
        ds.ghd_mean = ck["orig_ghd_mean"].cpu()
        ds.ghd_std = ck["orig_ghd_std"].cpu()

    canonical_mesh = sa.get("canonical_mesh", str(Path(args.canonical_root) / "part_aligned.obj"))
    eigen_chk = Path(sa.get("eigen_chk", str(Path(args.canonical_root) / "canonical_model_144_normed.pkl")))
    c2m = None
    faces = None
    if args.decode_backend == "coeff":
        canonical_norm = float(args.decode_canonical_norm_factor)
        c2m = CoeffToMesh(canonical_mesh, str(eigen_chk), num_basis=int(sa.get("num_basis", 144)),
                          device=device, canonical_norm_factor=canonical_norm)
        faces = c2m.canonical.faces_packed().detach().cpu()

    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    ok: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []

    for case, batch in zip(cases, loader):
        item: dict[str, object] = {"case": case}
        try:
            case_root = prepare_case_layout(
                out_root=out_root,
                case=case,
                alignment_root=Path(args.alignment_root),
                ghd_root=Path(args.ghd_root),
                canonical_root=Path(args.canonical_root),
                prepared_cases_root=Path(args.prepared_cases_root),
                overwrite=bool(args.overwrite),
            )
            cond = condition_from_batch(
                cond_net,
                batch,
                device,
                zero_vessel=bool(sa.get("no_vessel_pts", False)),
                zero_all=bool(sa.get("no_conditioning", False)),
            )
            with torch.no_grad():
                samples = sampler(cond, max(1, int(args.num_candidates)), args)[:, 0, :]
                if args.decode_backend == "reference":
                    verts, faces = decode_with_reference_fitter(
                        samples,
                        ds.ghd_mean,
                        ds.ghd_std,
                        case=case,
                        alignment_root=Path(args.alignment_root),
                        ghd_root=Path(args.ghd_root),
                        ghd_run=str(sa.get("ghd_run", "vanilla")),
                        ghd_chk_name=str(sa.get("ghd_chk_name", "ghb_fitting_checkpoint.pkl")),
                        eigen_chk=eigen_chk,
                        device=device,
                    )
                else:
                    assert c2m is not None and faces is not None
                    verts, _ = c2m(samples, ds.ghd_mean.to(device), ds.ghd_std.to(device), want_normals=False)
                    if samples.shape[-1] > 432 and ds.ghd_mean.shape[-1] > 432:
                        scale = samples[:, 432:433] * ds.ghd_std[:, 432:433].to(device) + ds.ghd_mean[:, 432:433].to(device)
                        verts = verts * scale.abs().view(-1, 1, 1)

            candidate_scores = []
            selected_idx = 0
            if args.select_by == "ostium" and int(args.num_candidates) > 1:
                ref_opa = load_pickle(Path(args.ghd_root) / case / "opa_checkpoint.pkl")
                opening_idx = np.asarray(ref_opa["op_v_indices"][0], dtype=np.int64)
                opening_idx = opening_idx[(opening_idx >= 0) & (opening_idx < verts.shape[1])]
                target_ring = batch["ostium_ring"][0].reshape(1, -1)
                target_ring = target_ring * ds.ostium_ring_std + ds.ostium_ring_mean
                target_ring_np = target_ring.reshape(-1, 3).detach().cpu().numpy()
                if opening_idx.shape[0] >= 3 and target_ring_np.shape[0] >= 3:
                    verts_np = verts.detach().cpu().numpy()
                    for cand_idx in range(verts_np.shape[0]):
                        score = _best_ring_fit_score(verts_np[cand_idx, opening_idx], target_ring_np)
                        score["candidate"] = int(cand_idx)
                        candidate_scores.append(score)
                    selected_idx = min(candidate_scores, key=lambda r: (r["mse"], -r["abs_normal_dot"]))["candidate"]

            raw_path = case_root / "outputs" / "stage1_sample" / f"{case}_sample_000_raw.obj"
            save_obj(str(raw_path), verts=verts[selected_idx].detach().cpu(), faces=faces)
            if args.save_candidates and verts.shape[0] > 1:
                cand_dir = case_root / "outputs" / "stage1_sample" / "candidates"
                cand_dir.mkdir(parents=True, exist_ok=True)
                for cand_idx in range(verts.shape[0]):
                    save_obj(str(cand_dir / f"{case}_candidate_{cand_idx:03d}.obj"),
                             verts=verts[cand_idx].detach().cpu(), faces=faces)

            cmd = [
                sys.executable,
                str(Path(args.reference_pipeline)),
                "step3",
                "--cases-root",
                str(out_root / "cases"),
                "--case-split",
                "test",
                "--case-name",
                case,
                "--stage1-ghd-root",
                str(Path(args.ghd_root)),
                "--stage1-canonical-root",
                str(Path(args.canonical_root)),
                "--ring-points",
                str(int(args.ring_points)),
                "--stitch",
                "--stitch-method",
                args.stitch_method,
                "--stitch-loop-source",
                "auto",
            ]
            if not args.no_remesh:
                cmd.append("--resample-aneurysm-to-vessel-resolution")
            if not args.no_smooth_transition:
                cmd.append("--smooth-ostium-transition")

            run_env = os.environ.copy()
            conda_prefix = Path(sys.executable).resolve().parents[1]
            conda_lib = conda_prefix / "lib"
            if conda_lib.exists():
                old_ld = run_env.get("LD_LIBRARY_PATH", "")
                parts = [str(conda_lib)] + ([old_ld] if old_ld else [])
                run_env["LD_LIBRARY_PATH"] = ":".join(parts)

            completed = subprocess.run(cmd, text=True, capture_output=True, check=False, env=run_env)
            item.update({
                "raw_sample": str(raw_path),
                "num_candidates": int(args.num_candidates),
                "select_by": args.select_by,
                "selected_candidate": int(selected_idx),
                "candidate_scores": candidate_scores,
                "command": " ".join(cmd),
                "returncode": int(completed.returncode),
            })
            (case_root / "outputs" / "reference_step3_stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
            (case_root / "outputs" / "reference_step3_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError((completed.stderr or completed.stdout or "reference step3 failed").strip()[-2000:])
            summary_path = case_root / "outputs" / "step3_compose_summary.json"
            if summary_path.exists():
                item["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
            ok.append(item)
            print(f"[ok] {case}")
        except Exception as exc:
            item["error"] = str(exc)
            failed.append(item)
            print(f"[failed] {case}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                break

    summary = {
        "cases_file": args.cases_file,
        "ckpt": args.ckpt,
        "method": args.method,
        "out_root": str(out_root),
        "ok_count": len(ok),
        "failed_count": len(failed),
        "ok": ok,
        "failed": failed,
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"ok_count": len(ok), "failed_count": len(failed), "summary": str(summary_path)}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
