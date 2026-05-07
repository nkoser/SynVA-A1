#!/usr/bin/env python3
"""Run a label_2-conditioned Stage-1 inference pipeline for any case.

The script keeps shared Stage-1 assets below inference/shared and writes all
case-specific runtime/output files below inference/cases/<split>/<case_name>.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
DEFAULT_CASES_ROOT = SCRIPT_DIR / "cases"
DEFAULT_CASE_SPLIT = "test"
DEFAULT_BUNDLE_ROOT = SCRIPT_DIR / "shared" / "stage1_inference_bundle"
DEFAULT_ANEUG_ROOT = CODE_ROOT / "AneuG-Own-edit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Case inference pipeline from subpointcloud_label_2.ply.")
    parser.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=["all", "step1", "step2", "step3"],
        help="Pipeline step to run. Default: all.",
    )
    parser.add_argument(
        "--case-root",
        type=Path,
        default=None,
        help="Direct case directory override. If unset, uses --cases-root/--case-split/--case-name.",
    )
    parser.add_argument("--cases-root", type=Path, default=DEFAULT_CASES_ROOT)
    parser.add_argument("--case-split", type=str, default=DEFAULT_CASE_SPLIT, help="Case split folder, e.g. train/test.")
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--aneug-root", type=Path, default=DEFAULT_ANEUG_ROOT)
    parser.add_argument("--case-name", type=str, default="C0084")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--ring-points", type=int, default=20)
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        default=DEFAULT_ANEUG_ROOT
        / "checkpoint-v2"
        / "first_stage_ostium_conditional"
        / "ostium_boundary20_retrain_0429"
        / "models_epoch_2000.pth",
        help="Stage-1 VAE checkpoint. Default: latest 20-point OPA retrain.",
    )
    parser.add_argument(
        "--stage1-ghd-root",
        type=Path,
        default=DEFAULT_ANEUG_ROOT / "checkpoint-v2" / "ghd_fitting_split_real",
        help="Reference GHD/condition root used by the Stage-1 VAE.",
    )
    parser.add_argument(
        "--stage1-alignment-root",
        type=Path,
        default=DEFAULT_ANEUG_ROOT / "alignment_vc",
        help="Reference alignment root used by the Stage-1 VAE.",
    )
    parser.add_argument(
        "--stage1-canonical-root",
        type=Path,
        default=DEFAULT_ANEUG_ROOT / "alignment_vc" / "canonical_model",
        help="Reference canonical root used by the Stage-1 VAE.",
    )
    parser.add_argument(
        "--stage1-split-file",
        type=Path,
        default=DEFAULT_ANEUG_ROOT / "checkpoint-v2" / "dataset_splits" / "data_split_real.json",
        help="Dataset split file used for Stage-1 normalization fallback.",
    )
    parser.add_argument(
        "--stage1-train-subset-limit",
        type=int,
        default=8,
        help="Training subset limit used for Stage-1 normalization fallback.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove runtime/output dirs before running.")
    parser.add_argument("--skip-reconstruct", action="store_true", help="In step2, only run sample mode.")
    parser.add_argument("--external-method-type", choices=["A", "B", "C", "D", "E", "W", "baseline"], default=None,
                        help="Use a /path/to/SynVA-A1 method checkpoint for Stage-1 sampling in step2.")
    parser.add_argument("--external-method-checkpoint", type=Path, default=None,
                        help="Checkpoint for --external-method-type.")
    parser.add_argument("--external-aneug-root", type=Path, default=Path("/path/to/SynVA-A1"),
                        help="Path to the AneuG repo containing methods/eval_all.py.")
    parser.add_argument("--external-temperature", type=float, default=0.8)
    parser.add_argument("--external-top-k", type=int, default=0)
    parser.add_argument("--external-flow-steps", type=int, default=64)
    parser.add_argument("--external-flow-sampler", choices=["euler", "heun"], default="heun")
    parser.add_argument("--disable-opening-refine", action="store_true", help="Disable automatic in-plane ring refinement in step3.")
    parser.add_argument(
        "--opening-refine-objective",
        choices=["ring", "surface"],
        default="ring",
        help="Objective for automatic in-plane opening refinement. Default: ring.",
    )
    parser.add_argument("--opening-refine-max-shift", type=float, default=0.04)
    parser.add_argument("--stitch", action="store_true", help="Also export a stitched vessel+pouch mesh. Default: off.")
    parser.add_argument("--stitch-method", choices=["bridge", "snap"], default="bridge")
    parser.add_argument("--stitch-bridge-steps", type=int, default=1, help="Intermediate rings for bridge stitching.")
    parser.add_argument("--stitch-k-candidates", type=int, default=20)
    parser.add_argument("--stitch-merge-digits", type=int, default=12)
    parser.add_argument(
        "--stitch-loop-source",
        choices=["auto", "nearest", "opa"],
        default="auto",
        help="Bridge stitching ring source. Default auto selects the smoothest output ring.",
    )
    parser.add_argument(
        "--stitch-smooth-intersection",
        action="store_true",
        help="Apply local Taubin smoothing after bridge stitching. Default off to avoid spike artifacts.",
    )
    parser.add_argument(
        "--smooth-ostium-transition",
        action="store_true",
        help=(
            "After stitching, locally smooth the ostium transition with simple N-hop neighbor averaging "
            "around bridge/ostium seeds (bridge, nearby vessel, nearby aneurysm). Requires --stitch."
        ),
    )
    parser.add_argument(
        "--smooth-ostium-radius",
        type=float,
        default=None,
        help="Legacy parameter kept for compatibility. Ignored by --smooth-ostium-transition N-hop smoothing.",
    )
    parser.add_argument(
        "--smooth-ostium-radius-scale",
        type=float,
        default=4.0,
        help="Legacy parameter kept for compatibility. Ignored by --smooth-ostium-transition N-hop smoothing.",
    )
    parser.add_argument("--smooth-ostium-iterations", type=int, default=10)
    parser.add_argument(
        "--smooth-ostium-hops",
        type=int,
        default=2,
        help="Topological neighborhood depth around bridge/ostium seeds for --smooth-ostium-transition.",
    )
    parser.add_argument(
        "--smooth-ostium-lambda",
        type=float,
        default=0.5,
        help="Legacy parameter kept for compatibility. Ignored by --smooth-ostium-transition N-hop smoothing.",
    )
    parser.add_argument(
        "--smooth-ostium-nu",
        type=float,
        default=0.53,
        help="Legacy parameter kept for compatibility. Ignored by --smooth-ostium-transition N-hop smoothing.",
    )
    parser.add_argument(
        "--stitch-legacy-mode",
        action="store_true",
        help="Use the previous stable bridge behavior: automatic output-neck ring selection without smoothing.",
    )
    parser.add_argument(
        "--keep-output-normal",
        action="store_true",
        help="Do not flip the final pouch direction in step3. Default flips to the opposite ostium side.",
    )
    parser.add_argument(
        "--shift-mode",
        type=str,
        choices=["normal-flush", "opening-center", "nearest-one-axis", "none"],
        default="none",
        help=(
            "Final aneurysm-only translation after rotation/scale. "
            "normal-flush: shift along output normal so the neck starts at the ostium plane; "
            "opening-center: shift generated neck center to centroid_ostium; "
            "nearest-one-axis: one-axis nearest-point correction against fixed label_2 ostium; none: no extra shift."
        ),
    )
    parser.add_argument(
        "--opening-align-mode",
        type=str,
        choices=["ring-fit", "legacy"],
        default="ring-fit",
        help="How step3 aligns the generated opening to the vessel ostium. Default ring-fit uses the 20-point OPA ring.",
    )
    parser.add_argument(
        "--resample-aneurysm-to-vessel-resolution",
        action="store_true",
        help="After step3 alignment, remesh the generated aneurysm to the vessel median edge length.",
    )
    parser.add_argument(
        "--aneurysm-remesh-target-edge-scale",
        type=float,
        default=1.0,
        help="Multiplier for the vessel median edge length used by --resample-aneurysm-to-vessel-resolution.",
    )
    parser.add_argument(
        "--aneurysm-remesh-iterations",
        type=int,
        default=5,
        help="PyMeshLab isotropic remeshing iterations for aneurysm resolution matching.",
    )
    return parser.parse_args()


def resolve_case_root(args: argparse.Namespace) -> Path:
    if args.case_root is not None:
        return args.case_root.expanduser().resolve()
    return (args.cases_root.expanduser() / args.case_split / args.case_name).resolve()


def require_files(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required file(s):\n" + "\n".join(missing))


def case_dir_with_prefix_fallback(root: Path, case_name: str) -> Path:
    direct = root / case_name
    if direct.exists():
        return direct
    if case_name.startswith("cmch_"):
        candidate = root / ("cmha_" + case_name[len("cmch_") :])
        if candidate.exists():
            return candidate
    if case_name.startswith("cmha_"):
        candidate = root / ("cmch_" + case_name[len("cmha_") :])
        if candidate.exists():
            return candidate
    return direct


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected mesh at {path}, got {type(mesh)!r}")
    return mesh


def save_mesh(mesh: trimesh.Trimesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def load_pointcloud_vertices(path: Path) -> np.ndarray:
    cloud = trimesh.load(path, process=False)
    if isinstance(cloud, trimesh.Scene):
        vertices = []
        for geometry in cloud.geometry.values():
            vertices.append(np.asarray(geometry.vertices, dtype=np.float64))
        if not vertices:
            raise ValueError(f"No vertices found in point cloud scene: {path}")
        return np.vstack(vertices)
    return np.asarray(cloud.vertices, dtype=np.float64)


def save_colored_sample_ostium_ply(
    sample_mesh: trimesh.Trimesh,
    ostium_points: np.ndarray,
    output_path: Path,
) -> None:
    pouch_points = np.asarray(sample_mesh.vertices, dtype=np.float64)
    ostium_points = np.asarray(ostium_points, dtype=np.float64)
    vertices = np.vstack([pouch_points, ostium_points])
    colors = np.vstack(
        [
            np.tile(np.array([[220, 64, 46, 255]], dtype=np.uint8), (pouch_points.shape[0], 1)),
            np.tile(np.array([[35, 160, 255, 255]], dtype=np.uint8), (ostium_points.shape[0], 1)),
        ]
    )
    cloud = trimesh.points.PointCloud(vertices, colors=colors)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cloud.export(output_path)


def mesh_summary(path: Path) -> dict[str, object]:
    mesh = load_mesh(path)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edge_stats = mesh_edge_stats(mesh)
    return {
        "path": str(path),
        "vertices": int(verts.shape[0]),
        "faces": int(faces.shape[0]),
        "edge_length_mean": edge_stats["mean"],
        "edge_length_median": edge_stats["median"],
        "edge_length_p90": edge_stats["p90"],
        "bounds_min": verts.min(axis=0).round(8).tolist(),
        "bounds_max": verts.max(axis=0).round(8).tolist(),
        "centroid": verts.mean(axis=0).round(8).tolist(),
    }


def mesh_edge_stats(mesh: trimesh.Trimesh) -> dict[str, float | None]:
    lengths = np.asarray(mesh.edges_unique_length, dtype=np.float64)
    if lengths.size == 0:
        return {"mean": None, "median": None, "p10": None, "p90": None}
    return {
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "p10": float(np.percentile(lengths, 10)),
        "p90": float(np.percentile(lengths, 90)),
    }


def unit_vector(value: np.ndarray, label: str) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError(f"{label} is degenerate.")
    return vec / norm


def rotation_from_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = unit_vector(src, "src")
    dst = unit_vector(dst, "dst")
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if dot > 1.0 - 1e-6:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-6:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(src, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(src, ref)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        angle = np.pi
    else:
        axis = np.cross(src, dst)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        angle = float(np.arccos(dot))

    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def copytree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def import_aneug_helpers(aneug_root: Path):
    sys.path.insert(0, str(aneug_root))
    from utils.create_opa_checkpoint_from_ostium import create_opa_checkpoint_for_case

    return create_opa_checkpoint_for_case


def translate_case_mesh_and_opa(case_dir: Path, translation: np.ndarray) -> None:
    translation = np.asarray(translation, dtype=np.float64).reshape(1, 3)
    mesh_path = case_dir / "part_aligned.obj"
    opa_path = case_dir / "opa_checkpoint.pkl"

    mesh = load_mesh(mesh_path)
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) + translation
    save_mesh(mesh, mesh_path)

    with opa_path.open("rb") as handle:
        chk = pickle.load(handle)
    for key in ("op_v_coords", "op_rec_v"):
        if key in chk:
            chk[key] = [np.asarray(value, dtype=np.float64) + translation for value in chk[key]]
    with opa_path.open("wb") as handle:
        pickle.dump(chk, handle)


def load_canonical_normal(canonical_opa_path: Path) -> np.ndarray:
    with canonical_opa_path.open("rb") as handle:
        chk = pickle.load(handle)
    return unit_vector(np.asarray(chk["op_n_mean"][0], dtype=np.float64), "canonical op_n_mean")


def load_opa(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_opa(path: Path, chk: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(chk, handle)


def estimate_similarity_transform(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or source.shape[0] < 3:
        raise ValueError(f"Expected matching [N, 3] rings, got {source.shape} and {target.shape}.")

    src_center = source.mean(axis=0)
    tgt_center = target.mean(axis=0)
    src0 = source - src_center.reshape(1, 3)
    tgt0 = target - tgt_center.reshape(1, 3)
    covariance = src0.T @ tgt0 / float(source.shape[0])
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(vt.T @ u.T) < 0:
        correction[-1, -1] = -1.0
    rotation = vt.T @ correction @ u.T
    variance = float(np.mean(np.sum(src0 * src0, axis=1)))
    if variance <= 1e-12:
        raise ValueError("Source ring is degenerate.")
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = tgt_center - scale * (src_center @ rotation.T)
    return scale, rotation, translation


def ring_radius(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0, keepdims=True)
    return float(np.mean(np.linalg.norm(points - center, axis=1)))


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = unit_vector(normal, "normal")
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = unit_vector(np.cross(n, ref), "plane basis u")
    v = unit_vector(np.cross(n, u), "plane basis v")
    return u, v


def nearest_stats(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(target).query(source, k=1)
    return {
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "max": float(np.max(distances)),
        "min": float(np.min(distances)),
    }


def symmetric_chamfer(source: np.ndarray, target: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    a = cKDTree(target).query(source, k=1)[0]
    b = cKDTree(source).query(target, k=1)[0]
    return float(0.5 * (np.mean(a * a) + np.mean(b * b)))


def optimize_in_plane_shift(
    source_points: np.ndarray,
    target_points: np.ndarray,
    normal: np.ndarray,
    max_shift: float,
) -> tuple[np.ndarray, dict[str, object]]:
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    u, v = plane_basis(normal)

    def shift_from_x(x: np.ndarray) -> np.ndarray:
        return float(x[0]) * u + float(x[1]) * v

    def objective(x: np.ndarray) -> float:
        shifted = source_points + shift_from_x(x).reshape(1, 3)
        return symmetric_chamfer(shifted, target_points) + 1e-5 * float(np.dot(x, x))

    best_x = np.zeros(2, dtype=np.float64)
    best_score = objective(best_x)
    evaluations = 1
    half_width = float(max_shift)
    for _ in range(5):
        values0 = np.clip(np.linspace(best_x[0] - half_width, best_x[0] + half_width, 17), -max_shift, max_shift)
        values1 = np.clip(np.linspace(best_x[1] - half_width, best_x[1] + half_width, 17), -max_shift, max_shift)
        for x0 in values0:
            for x1 in values1:
                x = np.array([x0, x1], dtype=np.float64)
                score = objective(x)
                evaluations += 1
                if score < best_score:
                    best_score = score
                    best_x = x
        half_width *= 0.25

    shift = shift_from_x(best_x)
    return shift, {
        "success": True,
        "message": "deterministic coarse-to-fine grid search",
        "basis_coefficients": [float(best_x[0]), float(best_x[1])],
        "objective": float(best_score),
        "evaluations": int(evaluations),
        "plane_basis_u": u.round(10).tolist(),
        "plane_basis_v": v.round(10).tolist(),
    }


def refine_opening_in_plane(
    pouch_mesh: trimesh.Trimesh,
    opening_indices: np.ndarray,
    ostium_points: np.ndarray,
    normal: np.ndarray,
    objective: str,
    max_shift: float,
) -> tuple[trimesh.Trimesh, dict[str, object]]:
    vertices = np.asarray(pouch_mesh.vertices, dtype=np.float64)
    opening = vertices[opening_indices]
    if objective == "ring":
        source_for_fit = opening
    elif objective == "surface":
        ostium_center = ostium_points.mean(axis=0, keepdims=True)
        radius = float(np.mean(np.linalg.norm(ostium_points - ostium_center, axis=1)))
        distances = np.linalg.norm(vertices - ostium_center, axis=1)
        source_for_fit = vertices[distances <= max(radius * 1.6, 1e-6)]
        if source_for_fit.shape[0] < opening_indices.shape[0]:
            source_for_fit = opening
    else:
        raise ValueError(f"Unsupported opening refinement objective: {objective}")

    before = {
        "ostium_to_opening": nearest_stats(ostium_points, opening),
        "opening_to_ostium": nearest_stats(opening, ostium_points),
        "ostium_to_aneurysm": nearest_stats(ostium_points, vertices),
    }
    shift, optimization = optimize_in_plane_shift(
        source_points=source_for_fit,
        target_points=ostium_points,
        normal=normal,
        max_shift=float(max_shift),
    )
    refined = pouch_mesh.copy()
    refined.vertices = vertices + shift.reshape(1, 3)
    refined_vertices = np.asarray(refined.vertices, dtype=np.float64)
    refined_opening = refined_vertices[opening_indices]
    normal_component = float(np.dot(shift, unit_vector(normal, "normal")))
    after = {
        "ostium_to_opening": nearest_stats(ostium_points, refined_opening),
        "opening_to_ostium": nearest_stats(refined_opening, ostium_points),
        "ostium_to_aneurysm": nearest_stats(ostium_points, refined_vertices),
    }
    return refined, {
        "enabled": True,
        "objective": objective,
        "max_shift": float(max_shift),
        "shift_vector": shift.round(10).tolist(),
        "shift_norm": float(np.linalg.norm(shift)),
        "shift_normal_component": normal_component,
        "shift_in_plane_residual": float(np.linalg.norm(shift - normal_component * unit_vector(normal, "normal"))),
        "optimization": optimization,
        "before": before,
        "after": after,
    }


def infer_vessel_labels_from_ostium(vessel_mesh: trimesh.Trimesh, ostium_points: np.ndarray, ostium_label: int = 2) -> np.ndarray:
    from scipy.spatial import cKDTree

    labels = np.zeros(len(vessel_mesh.vertices), dtype=np.int64)
    _, nearest = cKDTree(np.asarray(vessel_mesh.vertices, dtype=np.float64)).query(ostium_points, k=1)
    labels[np.unique(np.asarray(nearest, dtype=np.int64))] = int(ostium_label)
    return labels


def smooth_intersection(mesh: trimesh.Trimesh, labels: np.ndarray, ostium_label: int = 2) -> trimesh.Trimesh:
    from trimesh.smoothing import filter_taubin

    mesh_smoothed = mesh.copy()
    labels = np.asarray(labels)
    ostium_vertex_idx = np.where(labels == ostium_label)[0]
    if ostium_vertex_idx.size == 0:
        return mesh_smoothed

    neighbors: set[int] = set()
    for vertex_idx in ostium_vertex_idx:
        neighbors.update(mesh_smoothed.vertex_neighbors[int(vertex_idx)])
    vertices_to_smooth_idx = np.unique(np.concatenate([ostium_vertex_idx, np.asarray(list(neighbors), dtype=np.int64)]))

    smoothed_mesh = mesh_smoothed.copy()
    filter_taubin(smoothed_mesh, lamb=0.5, nu=0.53, iterations=10)
    if smoothed_mesh.vertices.shape != mesh_smoothed.vertices.shape or smoothed_mesh.faces.shape != mesh_smoothed.faces.shape:
        print("Warning: Number of vertices or faces changed during smoothing. Returning unsmoothed stitched mesh.")
        return mesh_smoothed
    mesh_smoothed.vertices[vertices_to_smooth_idx] = smoothed_mesh.vertices[vertices_to_smooth_idx]
    return mesh_smoothed


def smooth_ostium_transition_band(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    ostium_points: np.ndarray,
    iterations: int,
    hops: int,
    ostium_label: int = 2,
    radius: float | None = None,
    lamb: float | None = None,
    nu: float | None = None,
) -> tuple[trimesh.Trimesh, dict[str, object]]:
    from scipy.spatial import cKDTree
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape[0] != vertices.shape[0]:
        raise ValueError("Stitched labels must match stitched vertex count for ostium transition smoothing.")
    if int(hops) < 0:
        raise ValueError(f"Ostium smoothing hops must be >= 0, got {hops}.")

    seed_from_labels = np.where(labels == int(ostium_label))[0]
    _, nearest = cKDTree(vertices).query(np.asarray(ostium_points, dtype=np.float64), k=1)
    seed_from_points = np.unique(np.asarray(nearest, dtype=np.int64))
    seed = np.unique(np.concatenate([seed_from_labels, seed_from_points]))
    if seed.size == 0:
        return mesh.copy(), {
            "enabled": True,
            "method": "neighbor_average_n_hop",
            "seed_vertices": 0,
            "selected_vertices": 0,
            "iterations": int(iterations),
            "hops": int(hops),
            "message": "No seed vertices for ostium transition smoothing.",
        }

    selected = np.zeros(vertices.shape[0], dtype=bool)
    selected[seed] = True
    frontier = seed.tolist()
    for _ in range(int(hops)):
        if not frontier:
            break
        next_frontier: list[int] = []
        for vertex_idx in frontier:
            for neighbor_idx in mesh.vertex_neighbors[int(vertex_idx)]:
                neighbor_idx = int(neighbor_idx)
                if not selected[neighbor_idx]:
                    selected[neighbor_idx] = True
                    next_frontier.append(neighbor_idx)
        frontier = next_frontier

    selected_idx = np.flatnonzero(selected)
    selected_count = int(selected_idx.shape[0])
    if selected_count == 0:
        return mesh.copy(), {
            "enabled": True,
            "method": "neighbor_average_n_hop",
            "seed_vertices": int(seed.shape[0]),
            "selected_vertices": 0,
            "iterations": int(iterations),
            "hops": int(hops),
            "message": "No vertices selected for ostium transition smoothing.",
        }

    out = mesh.copy()
    current_vertices = vertices.copy()
    effective_iterations = max(1, int(iterations))
    for _ in range(effective_iterations):
        prev_vertices = current_vertices.copy()
        for vertex_idx in selected_idx:
            neighbors = mesh.vertex_neighbors[int(vertex_idx)]
            if not neighbors:
                continue
            current_vertices[int(vertex_idx)] = prev_vertices[np.asarray(neighbors, dtype=np.int64)].mean(axis=0)
    out.vertices = current_vertices

    label_counts = {
        str(int(label)): int(np.count_nonzero(selected & (labels == label)))
        for label in np.unique(labels)
    }
    displacement = np.linalg.norm(current_vertices - vertices, axis=1)
    return out, {
        "enabled": True,
        "method": "neighbor_average_n_hop",
        "seed_vertices": int(seed.shape[0]),
        "iterations": int(effective_iterations),
        "hops": int(hops),
        "selected_vertices": selected_count,
        "selected_label_counts": label_counts,
        "max_displacement": float(np.max(displacement[selected])),
        "mean_displacement": float(np.mean(displacement[selected])),
        "legacy_parameters": {
            "radius": None if radius is None else float(radius),
            "lambda": None if lamb is None else float(lamb),
            "nu": None if nu is None else float(nu),
            "ignored": True,
        },
    }


def _cleanup_mesh_with_labels(mesh: trimesh.Trimesh, labels: np.ndarray) -> tuple[trimesh.Trimesh, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    referenced = np.unique(mesh.faces.reshape(-1))
    remap = np.full(len(mesh.vertices), -1, dtype=np.int64)
    remap[referenced] = np.arange(len(referenced), dtype=np.int64)
    mesh.vertices = mesh.vertices[referenced]
    mesh.faces = remap[mesh.faces]
    labels = labels[referenced]
    return mesh, labels


def stitch_meshes(
    vessel_submesh: trimesh.Trimesh,
    labels_vessel_submesh: np.ndarray,
    transformed_mesh: trimesh.Trimesh,
    ostium_points: np.ndarray,
    opening_indices: np.ndarray,
    ostium_label: int = 2,
    transformed_label: int = 1,
    k_candidates: int = 20,
    merge_digits: int = 12,
) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray, dict[str, object]]:
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    vessel_submesh = vessel_submesh.copy()
    transformed_mesh = transformed_mesh.copy()
    labels = np.asarray(labels_vessel_submesh, dtype=np.int64)
    ostium_idx = np.where(labels == ostium_label)[0]
    if ostium_idx.size < 3:
        labels = infer_vessel_labels_from_ostium(vessel_submesh, ostium_points, ostium_label=ostium_label)
        ostium_idx = np.where(labels == ostium_label)[0]
    if ostium_idx.size < 3:
        raise RuntimeError("Could not infer enough vessel ostium vertices for stitching.")

    ostium_vertices = np.asarray(vessel_submesh.vertices, dtype=np.float64)[ostium_idx]
    transformed_vertices = np.asarray(transformed_mesh.vertices, dtype=np.float64).copy()
    opening_indices = np.asarray(opening_indices, dtype=np.int64)
    opening_indices = opening_indices[(opening_indices >= 0) & (opening_indices < len(transformed_vertices))]

    tree = cKDTree(transformed_vertices)
    k = min(max(1, int(k_candidates)), len(transformed_vertices))
    _, candidate_inds = tree.query(ostium_vertices, k=k)
    candidate_pool = set(np.unique(np.atleast_2d(candidate_inds).reshape(-1)).astype(np.int64).tolist())
    candidate_pool.update(opening_indices.tolist())
    candidate_pool = np.asarray(sorted(candidate_pool), dtype=np.int64)
    if candidate_pool.size < ostium_idx.size:
        _, expanded = tree.query(ostium_vertices, k=min(len(transformed_vertices), max(k, ostium_idx.size)))
        candidate_pool = np.asarray(sorted(set(np.unique(np.atleast_2d(expanded).reshape(-1)).astype(np.int64).tolist()) | set(opening_indices.tolist())), dtype=np.int64)
    if candidate_pool.size < ostium_idx.size:
        raise RuntimeError("Not enough transformed candidate vertices for stitching.")

    candidate_vertices = transformed_vertices[candidate_pool]
    cost = np.linalg.norm(ostium_vertices[:, None, :] - candidate_vertices[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_ostium_idx = ostium_idx[row_ind]
    matched_transformed_idx = candidate_pool[col_ind]

    transformed_mesh.vertices[matched_transformed_idx] = np.asarray(vessel_submesh.vertices)[matched_ostium_idx]
    vessel_labels = labels.copy()
    transformed_labels = np.full(len(transformed_mesh.vertices), transformed_label, dtype=np.int64)
    transformed_labels[matched_transformed_idx] = ostium_label

    stitched = trimesh.util.concatenate([vessel_submesh, transformed_mesh])
    combined_labels = np.concatenate([vessel_labels, transformed_labels])

    rounded = np.round(stitched.vertices, decimals=int(merge_digits))
    _, unique_idx, inverse = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
    stitched.vertices = stitched.vertices[unique_idx]
    stitched.faces = inverse[stitched.faces]

    new_labels = np.zeros(len(unique_idx), dtype=np.int64)
    for old_idx, new_idx in enumerate(inverse):
        new_labels[new_idx] = max(new_labels[new_idx], combined_labels[old_idx])

    stitched, new_labels = _cleanup_mesh_with_labels(stitched, new_labels)
    stitched.fix_normals()
    _ = stitched.vertex_normals

    matches = np.column_stack([matched_ostium_idx, matched_transformed_idx])
    matched_distances = np.linalg.norm(
        np.asarray(vessel_submesh.vertices)[matched_ostium_idx] - transformed_vertices[matched_transformed_idx],
        axis=1,
    )
    metadata = {
        "ostium_vertex_count": int(ostium_idx.size),
        "candidate_count": int(candidate_pool.size),
        "match_count": int(matches.shape[0]),
        "match_distance_mean_before_snap": float(np.mean(matched_distances)),
        "match_distance_max_before_snap": float(np.max(matched_distances)),
        "merge_digits": int(merge_digits),
    }
    return stitched, new_labels, matches, metadata


def order_ring_points(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    u, v = plane_basis(normal)
    rel = points - center.reshape(1, 3)
    coords = np.stack([rel @ u, rel @ v], axis=1)
    angles = np.arctan2(coords[:, 1], coords[:, 0])
    return np.argsort(angles)


def align_ordered_loop_indices_to_reference(candidate_indices: np.ndarray, candidate_points: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_points = np.asarray(candidate_points, dtype=np.float64)
    reference = resample_closed_ring(np.asarray(reference_points, dtype=np.float64), candidate_points.shape[0])
    best_error = None
    best_indices = candidate_indices
    best_shift = 0
    best_reverse = False
    for reverse, base_indices in [(False, candidate_indices), (True, candidate_indices[::-1].copy())]:
        base = candidate_points[::-1].copy() if reverse else candidate_points
        for shift in range(candidate_points.shape[0]):
            shifted = np.roll(base, -shift, axis=0)
            error = float(np.mean(np.sum((shifted - reference) ** 2, axis=1)))
            if best_error is None or error < best_error:
                best_error = error
                best_reverse = reverse
                best_shift = shift
                best_indices = np.roll(base_indices, -shift)
    return best_indices


def ordered_ring_edge_stats(points: np.ndarray) -> dict[str, float]:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 3:
        return {"mean": float("inf"), "p99": float("inf"), "max": float("inf")}
    edge_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    return {
        "mean": float(np.mean(edge_lengths)),
        "p99": float(np.percentile(edge_lengths, 99)),
        "max": float(np.max(edge_lengths)),
    }


def boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )
    edges = np.sort(edges, axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    return unique_edges[counts == 1]


def ordered_boundary_loops(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    edges = boundary_edges(mesh)
    if edges.size == 0:
        return []

    adjacency: dict[int, set[int]] = {}
    for a, b in edges.astype(np.int64):
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))

    unvisited = {tuple(map(int, edge)) for edge in edges}
    loops: list[np.ndarray] = []
    while unvisited:
        start, current = next(iter(unvisited))
        previous = start
        unvisited.discard(tuple(sorted((start, current))))
        loop = [start, current]

        while True:
            candidates = [
                neighbor
                for neighbor in sorted(adjacency.get(current, ()))
                if neighbor != previous and tuple(sorted((current, neighbor))) in unvisited
            ]
            if not candidates:
                break
            next_vertex = candidates[0]
            unvisited.discard(tuple(sorted((current, next_vertex))))
            if next_vertex == start:
                break
            loop.append(next_vertex)
            previous, current = current, next_vertex

        if len(loop) >= 3:
            loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def ordered_mesh_boundary_loop_candidates(
    mesh: trimesh.Trimesh,
    target_points: np.ndarray,
    normal: np.ndarray,
    reference_points: np.ndarray | None = None,
    source_prefix: str = "boundary_loop",
) -> list[dict[str, object]]:
    from scipy.spatial import cKDTree

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    target_tree = cKDTree(target_points)
    candidates: list[dict[str, object]] = []
    for loop_idx, loop in enumerate(ordered_boundary_loops(mesh)):
        loop_points = vertices[loop]
        if float(np.dot(ring_normal(loop_points), normal)) < 0.0:
            loop = loop[::-1].copy()
            loop_points = vertices[loop]
        if reference_points is not None:
            loop = align_ordered_loop_indices_to_reference(loop, loop_points, reference_points)
            loop_points = vertices[loop]
        edge_stats = ordered_ring_edge_stats(loop_points)
        target_distances = target_tree.query(loop_points, k=1)[0]
        candidates.append(
            {
                "source": f"{source_prefix}_{loop_idx}",
                "indices": loop,
                "is_boundary": True,
                "edge_stats": edge_stats,
                "ostium_distance_mean": float(np.mean(target_distances)),
                "ostium_distance_max": float(np.max(target_distances)),
            }
        )
    return candidates


def ordered_pouch_loop_candidates(
    pouch_mesh: trimesh.Trimesh,
    pouch_vertices: np.ndarray,
    opening_indices: np.ndarray,
    ostium_points: np.ndarray,
    vessel_loop_points: np.ndarray,
    normal: np.ndarray,
) -> list[dict[str, object]]:
    from scipy.spatial import cKDTree

    candidates: list[tuple[str, np.ndarray]] = []
    boundary_candidates = ordered_mesh_boundary_loop_candidates(
        mesh=pouch_mesh,
        target_points=ostium_points,
        normal=normal,
        reference_points=vessel_loop_points,
        source_prefix="pouch_boundary_loop",
    )
    opa_loop = np.asarray(opening_indices, dtype=np.int64)
    opa_loop = opa_loop[(opa_loop >= 0) & (opa_loop < len(pouch_vertices))]
    if opa_loop.shape[0] >= 3:
        candidates.append(("opa_indices", opa_loop))

    _, nearest_loop = cKDTree(pouch_vertices).query(ostium_points, k=1)
    nearest_loop = np.unique(np.asarray(nearest_loop, dtype=np.int64))
    if nearest_loop.shape[0] >= 3:
        candidates.append(("nearest_ostium_vertices", nearest_loop))

    ordered: list[dict[str, object]] = list(boundary_candidates)
    ostium_tree = cKDTree(np.asarray(ostium_points, dtype=np.float64))
    for source, loop in candidates:
        loop_points = pouch_vertices[loop]
        loop = loop[order_ring_points(loop_points, normal)]
        loop_points = pouch_vertices[loop]
        loop = align_ordered_loop_indices_to_reference(loop, loop_points, vessel_loop_points)
        loop_points = pouch_vertices[loop]
        edge_stats = ordered_ring_edge_stats(loop_points)
        ostium_distances = ostium_tree.query(loop_points, k=1)[0]
        ordered.append(
            {
                "source": source,
                "indices": loop,
                "is_boundary": False,
                "edge_stats": edge_stats,
                "ostium_distance_mean": float(np.mean(ostium_distances)),
                "ostium_distance_max": float(np.max(ostium_distances)),
            }
        )
    return ordered


def bridge_faces_between_loops(loop_a: np.ndarray, loop_b: np.ndarray, flip: bool = False) -> np.ndarray:
    loop_a = np.asarray(loop_a, dtype=np.int64)
    loop_b = np.asarray(loop_b, dtype=np.int64)
    n = int(loop_a.shape[0])
    m = int(loop_b.shape[0])
    if n < 3 or m < 3:
        raise ValueError("Bridge loops need at least three vertices each.")

    faces = []
    i = 0
    j = 0
    while i < n or j < m:
        next_i = (i + 1) % n
        next_j = (j + 1) % m
        can_a = i < n
        can_b = j < m
        if not can_b or (can_a and ((i + 1) / n <= (j + 1) / m)):
            face = [loop_a[i % n], loop_b[j % m], loop_a[next_i]]
            i += 1
        else:
            face = [loop_a[i % n], loop_b[j % m], loop_b[next_j]]
            j += 1
        if flip:
            face = [face[0], face[2], face[1]]
        faces.append(face)
        if i >= n and j >= m:
            break
    return np.asarray(faces, dtype=np.int64)


def stitch_meshes_bridge(
    vessel_submesh: trimesh.Trimesh,
    labels_vessel_submesh: np.ndarray,
    transformed_mesh: trimesh.Trimesh,
    ostium_points: np.ndarray,
    opening_indices: np.ndarray,
    normal: np.ndarray,
    bridge_steps: int = 4,
    ostium_label: int = 2,
    transformed_label: int = 1,
    merge_digits: int = 12,
    loop_source: str = "auto",
    smooth_intersection_enabled: bool = False,
) -> tuple[trimesh.Trimesh, np.ndarray, np.ndarray, dict[str, object]]:
    from scipy.spatial import cKDTree

    vessel = vessel_submesh.copy()
    pouch = transformed_mesh.copy()
    labels = np.asarray(labels_vessel_submesh, dtype=np.int64)
    if labels.shape[0] != len(vessel.vertices):
        raise ValueError("Vessel labels must match vessel vertex count.")

    vessel_vertices = np.asarray(vessel.vertices, dtype=np.float64)
    pouch_vertices = np.asarray(pouch.vertices, dtype=np.float64)
    vessel_boundary_candidates = ordered_mesh_boundary_loop_candidates(
        mesh=vessel,
        target_points=ostium_points,
        normal=normal,
        source_prefix="vessel_boundary_loop",
    )
    if vessel_boundary_candidates:
        vessel_candidate = min(
            vessel_boundary_candidates,
            key=lambda item: (
                item["ostium_distance_mean"],
                item["ostium_distance_max"],
                item["edge_stats"]["max"],
            ),
        )
        vessel_loop = np.asarray(vessel_candidate["indices"], dtype=np.int64)
    else:
        _, vessel_loop = cKDTree(vessel_vertices).query(ostium_points, k=1)
        vessel_loop = np.unique(np.asarray(vessel_loop, dtype=np.int64))
        vessel_loop_points = vessel_vertices[vessel_loop]
        vessel_loop = vessel_loop[order_ring_points(vessel_loop_points, normal)]
        vessel_candidate = {
            "source": "nearest_ostium_vertices",
            "is_boundary": False,
            "edge_stats": ordered_ring_edge_stats(vessel_vertices[vessel_loop]),
            "ostium_distance_mean": 0.0,
            "ostium_distance_max": 0.0,
        }
    vessel_loop_points = vessel_vertices[vessel_loop]

    if vessel_loop.shape[0] < 3:
        raise RuntimeError("Could not recover valid vessel/pouch loops for bridge stitching.")
    pouch_candidates = ordered_pouch_loop_candidates(
        pouch_mesh=pouch,
        pouch_vertices=pouch_vertices,
        opening_indices=opening_indices,
        ostium_points=ostium_points,
        vessel_loop_points=vessel_loop_points,
        normal=normal,
    )
    if not pouch_candidates:
        raise RuntimeError("Could not recover valid vessel/pouch loops for bridge stitching.")
    requested_loop_source = str(loop_source)
    if requested_loop_source == "auto":
        # Prefer true open mesh boundaries. A nearest/OPA subset can leave remeshed
        # boundary vertices unstitched, which shows up as visible holes.
        pouch_candidate = min(
            pouch_candidates,
            key=lambda item: (
                not bool(item.get("is_boundary", False)),
                item["ostium_distance_mean"],
                item["edge_stats"]["max"],
                item["ostium_distance_max"],
            ),
        )
    else:
        source_map = {"nearest": "nearest_ostium_vertices", "opa": "opa_indices"}
        expected_source = source_map[requested_loop_source]
        matching = [candidate for candidate in pouch_candidates if candidate["source"] == expected_source]
        if not matching:
            raise RuntimeError(f"Requested stitch loop source {requested_loop_source!r} is not available.")
        pouch_candidate = matching[0]
    pouch_loop = np.asarray(pouch_candidate["indices"], dtype=np.int64)
    pouch_loop_points = pouch_vertices[pouch_loop]

    all_vertices = [vessel_vertices, pouch_vertices]
    all_faces = [np.asarray(vessel.faces, dtype=np.int64), np.asarray(pouch.faces, dtype=np.int64) + len(vessel_vertices)]
    all_labels = [labels.copy(), np.full(len(pouch_vertices), transformed_label, dtype=np.int64)]
    all_labels[1][pouch_loop] = ostium_label

    ring_indices: list[np.ndarray] = [vessel_loop.copy()]
    ring_counts = np.rint(np.linspace(vessel_loop.shape[0], pouch_loop.shape[0], max(0, int(bridge_steps)) + 2)).astype(int)
    ring_counts[0] = vessel_loop.shape[0]
    ring_counts[-1] = pouch_loop.shape[0]

    vertex_offset = len(vessel_vertices) + len(pouch_vertices)
    bridge_vertices = []
    bridge_labels = []
    for step_idx, count in enumerate(ring_counts[1:-1], start=1):
        t = step_idx / float(len(ring_counts) - 1)
        vessel_resampled = resample_closed_ring(vessel_loop_points, int(count))
        pouch_resampled = resample_closed_ring(pouch_loop_points, int(count))
        ring = (1.0 - t) * vessel_resampled + t * pouch_resampled
        indices = np.arange(vertex_offset, vertex_offset + int(count), dtype=np.int64)
        vertex_offset += int(count)
        bridge_vertices.append(ring)
        bridge_labels.append(np.full(int(count), ostium_label, dtype=np.int64))
        ring_indices.append(indices)
    ring_indices.append(pouch_loop + len(vessel_vertices))

    if bridge_vertices:
        all_vertices.append(np.vstack(bridge_vertices))
        all_labels.append(np.concatenate(bridge_labels))

    bridge_faces = []
    for left, right in zip(ring_indices[:-1], ring_indices[1:]):
        bridge_faces.append(bridge_faces_between_loops(left, right))
    all_faces.append(np.vstack(bridge_faces))

    stitched = trimesh.Trimesh(
        vertices=np.vstack(all_vertices),
        faces=np.vstack(all_faces),
        process=False,
    )
    combined_labels = np.concatenate(all_labels)

    rounded = np.round(stitched.vertices, decimals=int(merge_digits))
    _, unique_idx, inverse = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
    stitched.vertices = stitched.vertices[unique_idx]
    stitched.faces = inverse[stitched.faces]

    new_labels = np.zeros(len(unique_idx), dtype=np.int64)
    for old_idx, new_idx in enumerate(inverse):
        new_labels[new_idx] = max(new_labels[new_idx], combined_labels[old_idx])

    stitched, new_labels = _cleanup_mesh_with_labels(stitched, new_labels)
    if smooth_intersection_enabled:
        stitched = smooth_intersection(stitched, new_labels, ostium_label=ostium_label)
    stitched.fix_normals()
    _ = stitched.vertex_normals

    edge_lengths = stitched.edges_unique_length
    metadata = {
        "method": "bridge",
        "vessel_loop_count": int(vessel_loop.shape[0]),
        "pouch_loop_count": int(pouch_loop.shape[0]),
        "vessel_loop_source": str(vessel_candidate["source"]),
        "vessel_loop_is_boundary": bool(vessel_candidate.get("is_boundary", False)),
        "pouch_loop_is_boundary": bool(pouch_candidate.get("is_boundary", False)),
        "bridge_steps": int(bridge_steps),
        "ring_counts": ring_counts.astype(int).tolist(),
        "bridge_face_count": int(sum(len(x) for x in bridge_faces)),
        "requested_loop_source": requested_loop_source,
        "pouch_loop_source": str(pouch_candidate["source"]),
        "smooth_intersection_enabled": bool(smooth_intersection_enabled),
        "pouch_loop_edge_stats": pouch_candidate["edge_stats"],
        "pouch_loop_ostium_distance_mean": float(pouch_candidate["ostium_distance_mean"]),
        "pouch_loop_ostium_distance_max": float(pouch_candidate["ostium_distance_max"]),
        "pouch_loop_candidates": [
            {
                "source": str(candidate["source"]),
                "count": int(np.asarray(candidate["indices"]).shape[0]),
                "is_boundary": bool(candidate.get("is_boundary", False)),
                "edge_stats": candidate["edge_stats"],
                "ostium_distance_mean": float(candidate["ostium_distance_mean"]),
                "ostium_distance_max": float(candidate["ostium_distance_max"]),
            }
            for candidate in pouch_candidates
        ],
        "edge_length_max": float(np.max(edge_lengths)) if edge_lengths.size else None,
        "edge_length_p99": float(np.percentile(edge_lengths, 99)) if edge_lengths.size else None,
        "merge_digits": int(merge_digits),
    }
    matches = np.empty((0, 2), dtype=np.int64)
    return stitched, new_labels, matches, metadata


def resample_closed_ring(points: np.ndarray, num_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        raise ValueError(f"Expected ring points with shape [N, 3], got {points.shape}.")
    if num_points < 3:
        raise ValueError("num_points must be at least 3.")
    diffs = np.roll(points, -1, axis=0) - points
    seg_lengths = np.linalg.norm(diffs, axis=1)
    if np.all(seg_lengths < 1e-12):
        raise ValueError("Cannot resample degenerate closed ring.")
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(cumulative[-1])
    samples = np.linspace(0.0, total, num=num_points, endpoint=False)
    out = np.zeros((num_points, 3), dtype=np.float64)
    for idx, sample in enumerate(samples):
        seg_idx = min(np.searchsorted(cumulative, sample, side="right") - 1, points.shape[0] - 1)
        seg_len = seg_lengths[seg_idx]
        if seg_len <= 1e-12:
            out[idx] = points[seg_idx]
            continue
        alpha = (sample - cumulative[seg_idx]) / seg_len
        out[idx] = (1.0 - alpha) * points[seg_idx] + alpha * points[(seg_idx + 1) % points.shape[0]]
    return out


def ring_normal(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0, keepdims=True)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    poly_normal = np.zeros(3, dtype=np.float64)
    for cur, nxt in zip(points, np.roll(points, -1, axis=0)):
        poly_normal += np.cross(cur, nxt)
    if np.linalg.norm(poly_normal) > 1e-12 and np.dot(normal, poly_normal) < 0:
        normal = -normal
    return unit_vector(normal, "ring normal")


def transform_points(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return float(scale) * (np.asarray(points, dtype=np.float64) @ np.asarray(rotation, dtype=np.float64).T) + np.asarray(
        translation, dtype=np.float64
    ).reshape(1, 3)


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = unit_vector(axis, "rotation axis")
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def plane_basis_from_normal(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = unit_vector(normal, "plane normal")
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(n, ref)
    u = unit_vector(u, "plane basis u")
    v = np.cross(n, u)
    v = unit_vector(v, "plane basis v")
    return u, v


def fit_ring_similarity_to_target(
    source_ring: np.ndarray,
    target_ring: np.ndarray,
    desired_output_normal: np.ndarray,
    source_mesh_vertices: np.ndarray,
    samples: int = 96,
) -> dict[str, object]:
    source_sampled = resample_closed_ring(source_ring, samples)
    target_sampled = resample_closed_ring(target_ring, samples)
    desired_output_normal = unit_vector(desired_output_normal, "desired_output_normal")

    # Make the target traversal consistent with the intended output side, then
    # still evaluate both traversals because generated mesh indexing can differ.
    if float(np.dot(ring_normal(target_sampled), desired_output_normal)) < 0.0:
        target_sampled = target_sampled[::-1].copy()

    best = None
    best_normal = None
    best_normal_with_side = None
    total_candidates = 0
    normal_valid_candidates = 0
    normal_and_side_candidates = 0
    target_center = np.asarray(target_ring, dtype=np.float64).mean(axis=0)
    source_mesh_vertices = np.asarray(source_mesh_vertices, dtype=np.float64)
    target_variants = [(False, target_sampled), (True, target_sampled[::-1].copy())]
    for reversed_order, target_variant in target_variants:
        for shift in range(samples):
            total_candidates += 1
            target_shifted = np.roll(target_variant, -shift, axis=0)
            scale, rotation, translation = estimate_similarity_transform(source_sampled, target_shifted)
            fitted_ring = transform_points(source_sampled, scale, rotation, translation)
            mse = float(np.mean(np.sum((fitted_ring - target_shifted) ** 2, axis=1)))
            fitted_normal = ring_normal(fitted_ring)
            normal_dot = float(np.dot(fitted_normal, desired_output_normal))
            transformed_mesh_center = transform_points(
                source_mesh_vertices.mean(axis=0, keepdims=True), scale, rotation, translation
            )[0]
            side_dot = float(np.dot(transformed_mesh_center - target_center, desired_output_normal))
            candidate = {
                "scale": float(scale),
                "rotation": rotation,
                "translation": translation,
                "mse": mse,
                "shift": int(shift),
                "reversed_order": bool(reversed_order),
                "normal_dot": normal_dot,
                "side_dot": side_dot,
            }
            if best is None or mse < best["mse"]:
                best = candidate
            if normal_dot >= 0.0:
                normal_valid_candidates += 1
                if best_normal is None or mse < best_normal["mse"]:
                    best_normal = candidate
                if side_dot >= 0.0:
                    normal_and_side_candidates += 1
                    if best_normal_with_side is None or mse < best_normal_with_side["mse"]:
                        best_normal_with_side = candidate

    if best is None:
        raise RuntimeError("Could not fit generated opening ring to target ostium ring.")
    if best_normal_with_side is not None:
        selected = dict(best_normal_with_side)
        selected["selection_reason"] = "normal_and_side"
    elif best_normal is not None:
        selected = dict(best_normal)
        selected["selection_reason"] = "normal_only"
    else:
        raise RuntimeError(
            "Could not fit generated opening ring with outward-facing normal. "
            f"Best candidate had normal_dot={float(best['normal_dot']):.6f}, side_dot={float(best['side_dot']):.6f}."
        )
    selected["candidate_counts"] = {
        "total": int(total_candidates),
        "normal_valid": int(normal_valid_candidates),
        "normal_and_side": int(normal_and_side_candidates),
    }
    return selected


def fit_ring_scaled_directional_to_target(
    source_ring: np.ndarray,
    target_ring: np.ndarray,
    desired_output_normal: np.ndarray,
    source_mesh_vertices: np.ndarray,
    samples: int = 96,
) -> dict[str, object]:
    source_ring = np.asarray(source_ring, dtype=np.float64)
    target_ring = np.asarray(target_ring, dtype=np.float64)
    source_mesh_vertices = np.asarray(source_mesh_vertices, dtype=np.float64)
    desired_output_normal = unit_vector(desired_output_normal, "desired_output_normal")

    source_center = source_ring.mean(axis=0)
    target_center = target_ring.mean(axis=0)
    source_radius = ring_radius(source_ring)
    target_radius = ring_radius(target_ring)
    if source_radius <= 1e-12 or target_radius <= 1e-12:
        raise RuntimeError("Cannot fit rings with degenerate radius.")
    scale = target_radius / source_radius

    source_side = source_mesh_vertices.mean(axis=0) - source_center
    if np.linalg.norm(source_side) <= 1e-12:
        source_side = ring_normal(source_ring)
    base_rotation = rotation_from_vectors(source_side, desired_output_normal)

    source_sampled = resample_closed_ring(source_ring, samples)
    target_sampled = resample_closed_ring(target_ring, samples)
    if float(np.dot(ring_normal(target_sampled), desired_output_normal)) < 0.0:
        target_sampled = target_sampled[::-1].copy()

    source_base = scale * ((source_sampled - source_center.reshape(1, 3)) @ base_rotation.T)
    target_centered_base = target_sampled - target_center.reshape(1, 3)
    u, v = plane_basis_from_normal(desired_output_normal)
    basis = np.stack([u, v], axis=1)
    source2 = source_base @ basis

    best = None
    for reversed_order, target_variant in [(False, target_centered_base), (True, target_centered_base[::-1].copy())]:
        for shift in range(samples):
            target_shifted = np.roll(target_variant, -shift, axis=0)
            target2 = target_shifted @ basis
            covariance = source2.T @ target2
            u2, _, vt2 = np.linalg.svd(covariance)
            rot2 = vt2.T @ u2.T
            if np.linalg.det(rot2) < 0.0:
                vt2[-1, :] *= -1.0
                rot2 = vt2.T @ u2.T
            angle = float(np.arctan2(rot2[1, 0], rot2[0, 0]))
            around = rotation_about_axis(desired_output_normal, angle)
            full_rotation = around @ base_rotation
            fitted = scale * ((source_sampled - source_center.reshape(1, 3)) @ full_rotation.T) + target_center.reshape(1, 3)
            mse = float(np.mean(np.sum((fitted - (target_shifted + target_center.reshape(1, 3))) ** 2, axis=1)))
            transformed_mesh_center = scale * (
                (source_mesh_vertices.mean(axis=0) - source_center) @ full_rotation.T
            ) + target_center
            side_dot = float(np.dot(transformed_mesh_center - target_center, desired_output_normal))
            fitted_normal = ring_normal(fitted)
            normal_dot = float(np.dot(fitted_normal, desired_output_normal))
            candidate = {
                "scale": float(scale),
                "rotation": full_rotation,
                "translation": target_center - scale * (source_center @ full_rotation.T),
                "mse": mse,
                "shift": int(shift),
                "reversed_order": bool(reversed_order),
                "normal_dot": normal_dot,
                "side_dot": side_dot,
                "in_plane_angle_rad": angle,
            }
            if best is None or mse < best["mse"]:
                best = candidate

    if best is None:
        raise RuntimeError("Could not fit generated opening ring to target ostium ring.")
    return best


def align_target_ring_order_to_source(source_ring: np.ndarray, target_ring: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    source_ring = np.asarray(source_ring, dtype=np.float64)
    target_resampled = resample_closed_ring(target_ring, source_ring.shape[0])
    best = None
    for reversed_order, candidate_base in [(False, target_resampled), (True, target_resampled[::-1].copy())]:
        for shift in range(source_ring.shape[0]):
            candidate = np.roll(candidate_base, -shift, axis=0)
            mse = float(np.mean(np.sum((source_ring - candidate) ** 2, axis=1)))
            if best is None or mse < best["mse"]:
                best = {
                    "ring": candidate,
                    "mse": mse,
                    "shift": int(shift),
                    "reversed_order": bool(reversed_order),
                }
    if best is None:
        raise RuntimeError("Could not align target ring order to generated source ring.")
    return best["ring"], {k: v for k, v in best.items() if k != "ring"}


def estimate_center_radius_normal_transform(source_chk: dict, target_chk: dict) -> tuple[float, np.ndarray, np.ndarray]:
    source = np.asarray(source_chk["op_v_coords"][0], dtype=np.float64)
    target = np.asarray(target_chk["op_v_coords"][0], dtype=np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_radius = ring_radius(source)
    target_radius = ring_radius(target)
    if source_radius <= 1e-12 or target_radius <= 1e-12:
        raise ValueError("Cannot estimate condition transform from degenerate ring radius.")
    source_normal = unit_vector(np.asarray(source_chk["op_n_mean"][0], dtype=np.float64), "source op_n_mean")
    target_normal = unit_vector(np.asarray(target_chk["op_n_mean"][0], dtype=np.float64), "target op_n_mean")
    scale = target_radius / source_radius
    rotation = rotation_from_vectors(source_normal, target_normal)
    translation = target_center - scale * (source_center @ rotation.T)
    return float(scale), rotation, translation


def transform_opa_checkpoint(chk: dict, scale: float, rotation: np.ndarray, translation: np.ndarray) -> dict:
    out = dict(chk)
    translation = np.asarray(translation, dtype=np.float64).reshape(1, 3)
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)

    for key in ("op_v_coords", "op_rec_v"):
        if key in out:
            out[key] = [
                float(scale) * (np.asarray(value, dtype=np.float64) @ rotation.T) + translation
                for value in out[key]
            ]
    if "op_v_normal" in out:
        out["op_v_normal"] = [
            np.asarray(value, dtype=np.float64) @ rotation.T
            for value in out["op_v_normal"]
        ]
    if "op_n_mean" in out:
        out["op_n_mean"] = [
            unit_vector(np.asarray(value, dtype=np.float64) @ rotation.T, "transformed op_n_mean")
            for value in out["op_n_mean"]
        ]
    out["source"] = "label2_alignment_opa_transformed_to_training_condition_space"
    return out


def step1_create_opa(args: argparse.Namespace) -> dict[str, object]:
    case_root = resolve_case_root(args)
    bundle_root = args.bundle_root.resolve()
    aneug_root = args.aneug_root.resolve()
    stage1_ghd_root = args.stage1_ghd_root.resolve()
    stage1_alignment_root = args.stage1_alignment_root.resolve()
    stage1_canonical_root = args.stage1_canonical_root.resolve()
    runtime_root = case_root / "_runtime"
    zero_case_dir = runtime_root / "zero_cases" / args.case_name
    alignment_root = runtime_root / "alignment_vc"
    condition_root = runtime_root / "condition_opa"
    alignment_case_dir = alignment_root / args.case_name
    condition_case_dir = condition_root / args.case_name
    canonical_src = stage1_canonical_root
    canonical_dst = alignment_root / "canonical_model"
    reference_alignment_case_dir = case_dir_with_prefix_fallback(stage1_alignment_root, args.case_name)
    reference_alignment_opa = reference_alignment_case_dir / "opa_checkpoint.pkl"
    reference_condition_opa = stage1_ghd_root / args.case_name / "opa_checkpoint.pkl"
    canonical_eigen = stage1_canonical_root / "canonical_model_144_normed.pkl"
    if not canonical_eigen.exists():
        canonical_eigen = stage1_ghd_root / "canonical_model_144_normed.pkl"

    ostium_ply = case_root / "04_subpointclouds" / "subpointcloud_label_2.ply"
    vessel_mesh = case_root / "05_submeshes" / "vessel_submesh.obj"
    centroid_path = case_root / "07_other" / "centroid_ostium.npy"
    normal_path = case_root / "07_other" / "normal_vector.npy"
    require_files(
        [
            ostium_ply,
            vessel_mesh,
            centroid_path,
            normal_path,
            canonical_src / "part_aligned.obj",
            canonical_src / "opa_checkpoint.pkl",
            canonical_eigen,
            reference_alignment_opa,
            reference_condition_opa,
        ]
    )

    if args.overwrite and runtime_root.exists():
        shutil.rmtree(runtime_root)

    (zero_case_dir / "04_subpointclouds").mkdir(parents=True, exist_ok=True)
    (zero_case_dir / "07_other").mkdir(parents=True, exist_ok=True)
    alignment_case_dir.mkdir(parents=True, exist_ok=True)
    condition_case_dir.mkdir(parents=True, exist_ok=True)

    copytree_clean(canonical_src, canonical_dst)
    shutil.copy2(canonical_eigen, canonical_dst / "canonical_model_144_normed.pkl")
    shutil.copy2(ostium_ply, zero_case_dir / "04_subpointclouds" / "subpointcloud_label_2.ply")
    shutil.copy2(centroid_path, zero_case_dir / "07_other" / "centroid_ostium.npy")
    shutil.copy2(normal_path, zero_case_dir / "07_other" / "normal_vector.npy")
    shutil.copy2(vessel_mesh, zero_case_dir / "aneurysm_aligned.obj")
    shutil.copy2(vessel_mesh, alignment_case_dir / "part_aligned.obj")

    create_opa_checkpoint_for_case = import_aneug_helpers(aneug_root)
    chk = create_opa_checkpoint_for_case(
        zero_case_dir=zero_case_dir,
        alignment_case_dir=alignment_case_dir,
        ostium_label=2,
        flip_inside_normal=False,
        smooth_iters=2,
        smooth_alpha=0.25,
        target_opening_triangles=max(1, int(args.ring_points) - 2),
    )
    save_opa(alignment_case_dir / "opa_checkpoint.pkl", chk)

    centroid = np.asarray(np.load(centroid_path), dtype=np.float64).reshape(3)
    translate_case_mesh_and_opa(alignment_case_dir, -centroid)

    alignment_chk = load_opa(alignment_case_dir / "opa_checkpoint.pkl")
    reference_alignment_chk = load_opa(reference_alignment_opa)
    reference_condition_chk = load_opa(reference_condition_opa)
    scale, condition_rotation, condition_translation = estimate_center_radius_normal_transform(
        reference_alignment_chk,
        reference_condition_chk,
    )
    condition_chk = transform_opa_checkpoint(
        alignment_chk,
        scale=scale,
        rotation=condition_rotation,
        translation=condition_translation,
    )
    save_opa(condition_case_dir / "opa_checkpoint.pkl", condition_chk)

    debug_opa = case_root / "07_other" / "opa_checkpoint_stage1.pkl"
    shutil.copy2(condition_case_dir / "opa_checkpoint.pkl", debug_opa)

    ring = np.asarray(alignment_chk["op_v_coords"][0], dtype=np.float64)
    condition_ring = np.asarray(condition_chk["op_v_coords"][0], dtype=np.float64)
    rec = np.asarray(alignment_chk["op_rec_v"][0], dtype=np.float64)
    normal = unit_vector(np.load(normal_path), "normal_vector")
    canonical_normal = load_canonical_normal(canonical_dst / "opa_checkpoint.pkl")
    rotation = rotation_from_vectors(canonical_normal, normal)

    summary = {
        "step": "step1_create_opa",
        "case_name": args.case_name,
        "case_root": str(case_root),
        "zero_case_dir": str(zero_case_dir),
        "alignment_root": str(alignment_root),
        "condition_root": str(condition_root),
        "reference_alignment_opa": str(reference_alignment_opa),
        "reference_condition_opa": str(reference_condition_opa),
        "opa_path": str(alignment_case_dir / "opa_checkpoint.pkl"),
        "condition_opa_path": str(condition_case_dir / "opa_checkpoint.pkl"),
        "debug_opa_path": str(debug_opa),
        "centroid": centroid.round(8).tolist(),
        "normal_vector": normal.round(8).tolist(),
        "canonical_normal": canonical_normal.round(8).tolist(),
        "canonical_to_target_rotation": rotation.round(10).tolist(),
        "ring_points": int(ring.shape[0]),
        "ring_bounds_min": ring.min(axis=0).round(8).tolist(),
        "ring_bounds_max": ring.max(axis=0).round(8).tolist(),
        "ring_center": ring.mean(axis=0).round(8).tolist(),
        "condition_transform_scale": float(scale),
        "condition_ring_bounds_min": condition_ring.min(axis=0).round(8).tolist(),
        "condition_ring_bounds_max": condition_ring.max(axis=0).round(8).tolist(),
        "condition_ring_center": condition_ring.mean(axis=0).round(8).tolist(),
        "rec_points": int(rec.shape[0]),
        "part_aligned": mesh_summary(alignment_case_dir / "part_aligned.obj"),
    }
    out_path = case_root / "outputs" / "step1_opa_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def run_checked(cmd: list[str], cwd: Path) -> None:
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def step2_infer(args: argparse.Namespace) -> dict[str, object]:
    case_root = resolve_case_root(args)
    aneug_root = args.aneug_root.resolve()
    runtime_root = case_root / "_runtime"
    alignment_root = runtime_root / "alignment_vc"
    condition_root = runtime_root / "condition_opa"
    ghd_root = args.stage1_ghd_root.resolve()
    checkpoint = args.stage1_checkpoint.resolve()
    canonical_root = args.stage1_canonical_root.resolve()
    canonical_eigen = canonical_root / "canonical_model_144_normed.pkl"
    if not canonical_eigen.exists():
        canonical_eigen = ghd_root / "canonical_model_144_normed.pkl"
    opa_path = condition_root / args.case_name / "opa_checkpoint.pkl"
    infer_script = aneug_root / "infer_stage1_ostium_conditional.py"
    if not infer_script.exists():
        infer_script = aneug_root / "utils" / "test-skripts" / "infer_stage1_ostium_conditional.py"

    require_files(
        [
            checkpoint,
            canonical_eigen,
            opa_path,
            alignment_root / "canonical_model" / "part_aligned.obj",
            alignment_root / "canonical_model" / "opa_checkpoint.pkl",
            condition_root / args.case_name / "opa_checkpoint.pkl",
            infer_script,
            ghd_root / args.case_name / "vanilla" / "ghb_fitting_checkpoint.pkl",
        ]
    )

    if args.overwrite:
        for folder in (case_root / "outputs" / "stage1_reconstruct", case_root / "outputs" / "stage1_sample"):
            if folder.exists():
                shutil.rmtree(folder)

    base_cmd = [
        sys.executable,
        str(infer_script),
        "--checkpoint",
        str(checkpoint),
        "--ghd-chk-root",
        str(ghd_root),
        "--condition-root",
        str(condition_root),
        "--alignment-root",
        str(alignment_root),
        "--canonical-root",
        str(alignment_root / "canonical_model"),
        "--canonical-eigen-chk",
        str(canonical_eigen),
        "--prepare-condition-from-ghd",
        "0",
        "--case",
        args.case_name,
        "--ring-points",
        str(int(args.ring_points)),
        "--taubin-iter",
        "0",
    ]
    if args.stage1_split_file is not None and args.stage1_split_file.exists():
        base_cmd += ["--split-file", str(args.stage1_split_file.resolve())]
    if args.stage1_train_subset_limit is not None:
        base_cmd += ["--train-subset-limit", str(int(args.stage1_train_subset_limit))]
    if args.external_method_type is not None:
        if args.external_method_checkpoint is None:
            raise ValueError("--external-method-checkpoint is required with --external-method-type")
        base_cmd += [
            "--external-method-type",
            str(args.external_method_type),
            "--external-method-checkpoint",
            str(args.external_method_checkpoint.resolve()),
            "--external-aneug-root",
            str(args.external_aneug_root.resolve()),
            "--external-temperature",
            str(float(args.external_temperature)),
            "--external-top-k",
            str(int(args.external_top_k)),
            "--external-flow-steps",
            str(int(args.external_flow_steps)),
            "--external-flow-sampler",
            str(args.external_flow_sampler),
        ]

    outputs: dict[str, object] = {"step": "step2_infer", "commands": []}
    if not args.skip_reconstruct:
        reconstruct_dir = case_root / "outputs" / "stage1_reconstruct"
        cmd = base_cmd + [
            "--mode",
            "reconstruct",
            "--posterior-noise-scale",
            "0",
            "--output-dir",
            str(reconstruct_dir),
        ]
        run_checked(cmd, cwd=aneug_root)
        outputs["commands"].append({"mode": "reconstruct", "output_dir": str(reconstruct_dir)})

    sample_dir = case_root / "outputs" / "stage1_sample"
    cmd = base_cmd + [
        "--mode",
        "sample",
        "--num-samples",
        str(int(args.num_samples)),
        "--seed",
        str(int(args.seed)),
        "--output-dir",
        str(sample_dir),
    ]
    run_checked(cmd, cwd=aneug_root)
    outputs["commands"].append({"mode": "sample", "output_dir": str(sample_dir)})

    for key, folder in (("reconstruct", case_root / "outputs" / "stage1_reconstruct"), ("sample", sample_dir)):
        metadata = folder / "metadata.json"
        if metadata.exists():
            outputs[f"{key}_metadata"] = json.loads(metadata.read_text(encoding="utf-8"))
        mesh_paths = sorted(folder.glob("*.obj"))
        outputs[f"{key}_meshes"] = [mesh_summary(path) for path in mesh_paths]

    out_path = case_root / "outputs" / "step2_infer_summary.json"
    out_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(json.dumps(outputs, indent=2))
    return outputs


def transform_mesh(mesh: trimesh.Trimesh, rotation: np.ndarray, translation: np.ndarray) -> trimesh.Trimesh:
    out = mesh.copy()
    vertices = np.asarray(out.vertices, dtype=np.float64)
    out.vertices = vertices @ rotation.T + np.asarray(translation, dtype=np.float64).reshape(1, 3)
    return out


def remesh_to_target_edge_length(
    mesh: trimesh.Trimesh,
    target_edge_length: float,
    iterations: int,
) -> tuple[trimesh.Trimesh, dict[str, object]]:
    if target_edge_length <= 0:
        raise ValueError(f"target_edge_length must be positive, got {target_edge_length}.")
    try:
        import pymeshlab
    except ImportError as exc:
        raise RuntimeError(
            "--resample-aneurysm-to-vessel-resolution requires pymeshlab in the active Python environment."
        ) from exc

    before = mesh_edge_stats(mesh)
    meshset = pymeshlab.MeshSet()
    meshset.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        ),
        "aneurysm",
    )
    meshset.meshing_isotropic_explicit_remeshing(
        iterations=max(1, int(iterations)),
        adaptive=False,
        selectedonly=False,
        targetlen=pymeshlab.PureValue(float(target_edge_length)),
        splitflag=True,
        collapseflag=True,
        swapflag=True,
        smoothflag=True,
        reprojectflag=True,
    )
    remeshed_raw = meshset.current_mesh()
    remeshed = trimesh.Trimesh(
        vertices=np.asarray(remeshed_raw.vertex_matrix(), dtype=np.float64),
        faces=np.asarray(remeshed_raw.face_matrix(), dtype=np.int64),
        process=False,
    )
    remeshed.remove_unreferenced_vertices()
    remeshed.fix_normals()
    after = mesh_edge_stats(remeshed)
    return remeshed, {
        "enabled": True,
        "method": "pymeshlab.meshing_isotropic_explicit_remeshing",
        "target_edge_length": float(target_edge_length),
        "iterations": int(iterations),
        "vertices_before": int(len(mesh.vertices)),
        "faces_before": int(len(mesh.faces)),
        "vertices_after": int(len(remeshed.vertices)),
        "faces_after": int(len(remeshed.faces)),
        "edge_stats_before": before,
        "edge_stats_after": after,
    }


def recover_opening_indices_from_ostium(mesh: trimesh.Trimesh, ostium_points: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    boundary_candidates = ordered_mesh_boundary_loop_candidates(
        mesh=mesh,
        target_points=ostium_points,
        normal=ring_normal(np.asarray(ostium_points, dtype=np.float64)),
        source_prefix="recovered_boundary_loop",
    )
    if boundary_candidates:
        best = min(
            boundary_candidates,
            key=lambda item: (
                item["ostium_distance_mean"],
                item["ostium_distance_max"],
                item["edge_stats"]["max"],
            ),
        )
        return np.asarray(best["indices"], dtype=np.int64)

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    _, nearest = cKDTree(vertices).query(np.asarray(ostium_points, dtype=np.float64), k=1)
    return np.unique(np.asarray(nearest, dtype=np.int64))


def transform_mesh_opening_to_target(
    mesh: trimesh.Trimesh,
    raw_opening_center: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    target_center: np.ndarray,
) -> trimesh.Trimesh:
    out = mesh.copy()
    vertices = np.asarray(out.vertices, dtype=np.float64)
    centered = vertices - np.asarray(raw_opening_center, dtype=np.float64).reshape(1, 3)
    out.vertices = float(scale) * (centered @ np.asarray(rotation, dtype=np.float64).reshape(3, 3).T) + np.asarray(
        target_center, dtype=np.float64
    ).reshape(1, 3)
    return out


def step3_compose(args: argparse.Namespace) -> dict[str, object]:
    case_root = resolve_case_root(args)
    runtime_root = case_root / "_runtime"
    alignment_root = runtime_root / "alignment_vc"
    sample_path = case_root / "outputs" / "stage1_sample" / f"{args.case_name}_sample_000_raw.obj"
    vessel_path = case_root / "05_submeshes" / "vessel_submesh.obj"
    ostium_ply = case_root / "04_subpointclouds" / "subpointcloud_label_2.ply"
    centroid_path = case_root / "07_other" / "centroid_ostium.npy"
    normal_path = case_root / "07_other" / "normal_vector.npy"
    canonical_opa = alignment_root / "canonical_model" / "opa_checkpoint.pkl"
    alignment_opa = alignment_root / args.case_name / "opa_checkpoint.pkl"
    reference_condition_opa = args.stage1_ghd_root.resolve() / args.case_name / "opa_checkpoint.pkl"
    require_files([sample_path, vessel_path, ostium_ply, centroid_path, normal_path, canonical_opa, alignment_opa, reference_condition_opa])

    centroid = np.asarray(np.load(centroid_path), dtype=np.float64).reshape(3)
    target_normal = unit_vector(np.load(normal_path), "normal_vector")
    output_normal = target_normal if args.keep_output_normal else -target_normal
    canonical_normal = load_canonical_normal(canonical_opa)
    rotation = rotation_from_vectors(canonical_normal, output_normal)

    vessel_mesh = load_mesh(vessel_path)
    pouch_mesh = load_mesh(sample_path)
    ref_condition = load_opa(reference_condition_opa)
    opening_indices = np.asarray(ref_condition["op_v_indices"][0], dtype=np.int64)
    raw_vertices = np.asarray(pouch_mesh.vertices, dtype=np.float64)
    opening_indices = opening_indices[(opening_indices >= 0) & (opening_indices < raw_vertices.shape[0])]
    if opening_indices.shape[0] < 3:
        raise RuntimeError(f"Could not recover valid generated opening indices from {reference_condition_opa}.")
    raw_opening = raw_vertices[opening_indices]
    raw_opening_center = raw_opening.mean(axis=0)
    raw_opening_radius = ring_radius(raw_opening)
    alignment_chk = load_opa(alignment_opa)
    target_opening = np.asarray(alignment_chk["op_v_coords"][0], dtype=np.float64) + centroid.reshape(1, 3)
    target_opening_radius = ring_radius(target_opening)
    if raw_opening_radius <= 1e-12 or target_opening_radius <= 1e-12:
        raise RuntimeError("Cannot align generated pouch because an opening radius is degenerate.")

    if args.opening_align_mode == "ring-fit":
        opening_fit = fit_ring_similarity_to_target(
            source_ring=raw_opening,
            target_ring=target_opening,
            desired_output_normal=output_normal,
            source_mesh_vertices=raw_vertices,
            samples=max(96, int(args.ring_points) * 4),
        )
        output_scale = float(opening_fit["scale"])
        transformed_pouch = pouch_mesh.copy()
        transformed_pouch.vertices = transform_points(
            raw_vertices,
            scale=output_scale,
            rotation=np.asarray(opening_fit["rotation"], dtype=np.float64),
            translation=np.asarray(opening_fit["translation"], dtype=np.float64),
        )
        opening_alignment = {
            "mode": "ring-fit",
            "mse": float(opening_fit["mse"]),
            "shift": int(opening_fit["shift"]),
            "reversed_order": bool(opening_fit["reversed_order"]),
            "normal_dot": float(opening_fit["normal_dot"]),
            "side_dot": float(opening_fit["side_dot"]),
            "selection_reason": str(opening_fit.get("selection_reason", "unknown")),
            "candidate_counts": dict(opening_fit.get("candidate_counts", {})),
            "rotation": np.asarray(opening_fit["rotation"], dtype=np.float64).round(10).tolist(),
            "translation": np.asarray(opening_fit["translation"], dtype=np.float64).round(10).tolist(),
        }
    else:
        output_scale = target_opening_radius / raw_opening_radius
        transformed_pouch = transform_mesh_opening_to_target(
            pouch_mesh,
            raw_opening_center=raw_opening_center,
            scale=output_scale,
            rotation=rotation,
            target_center=centroid,
        )
        opening_alignment = {
            "mode": "legacy",
            "rotation": rotation.round(10).tolist(),
            "translation": centroid.round(10).tolist(),
        }

    ostium_points = load_pointcloud_vertices(ostium_ply)
    shift_vector = np.zeros(3, dtype=np.float64)
    shift_axis = None
    shift_raw = np.zeros(3, dtype=np.float64)
    transformed_vertices_before_shift = np.asarray(transformed_pouch.vertices, dtype=np.float64)
    normal_flush_anchor_before = None
    normal_flush_anchor_after = None
    normal_flush_projection_range_before = None
    normal_flush_projection_range_after = None
    if args.shift_mode == "normal-flush":
        transformed_opening_before_shift = transformed_vertices_before_shift[opening_indices]
        opening_projection = (transformed_opening_before_shift - centroid.reshape(1, 3)) @ output_normal
        normal_flush_projection_range_before = [
            float(np.min(opening_projection)),
            float(np.mean(opening_projection)),
            float(np.max(opening_projection)),
        ]
        # Put the neck-side boundary on the ostium plane without changing lateral alignment.
        if np.mean((transformed_vertices_before_shift - centroid.reshape(1, 3)) @ output_normal) >= 0.0:
            normal_flush_anchor_before = float(np.min(opening_projection))
        else:
            normal_flush_anchor_before = float(np.max(opening_projection))
        shift_vector = -normal_flush_anchor_before * output_normal
        shift_raw = shift_vector.copy()
        transformed_vertices_after_shift = transformed_vertices_before_shift + shift_vector.reshape(1, 3)
        transformed_pouch.vertices = transformed_vertices_after_shift
        opening_projection_after = (transformed_vertices_after_shift[opening_indices] - centroid.reshape(1, 3)) @ output_normal
        normal_flush_projection_range_after = [
            float(np.min(opening_projection_after)),
            float(np.mean(opening_projection_after)),
            float(np.max(opening_projection_after)),
        ]
        if np.mean((transformed_vertices_after_shift - centroid.reshape(1, 3)) @ output_normal) >= 0.0:
            normal_flush_anchor_after = float(np.min(opening_projection_after))
        else:
            normal_flush_anchor_after = float(np.max(opening_projection_after))
    elif args.shift_mode == "opening-center":
        transformed_opening_before_shift = transformed_vertices_before_shift[opening_indices]
        shift_raw = centroid - transformed_opening_before_shift.mean(axis=0)
        shift_vector = shift_raw.copy()
        transformed_pouch.vertices = transformed_vertices_before_shift + shift_vector.reshape(1, 3)
    elif args.shift_mode == "nearest-one-axis":
        from scipy.spatial import cKDTree

        nearest_idx = cKDTree(transformed_vertices_before_shift).query(ostium_points, k=1)[1]
        nearest_points = transformed_vertices_before_shift[np.asarray(nearest_idx, dtype=np.int64)]
        shift_raw = np.asarray(ostium_points - nearest_points, dtype=np.float64).mean(axis=0)
        shift_axis = int(np.argmax(np.abs(shift_raw)))
        shift_vector[shift_axis] = shift_raw[shift_axis]
        transformed_pouch.vertices = transformed_vertices_before_shift + shift_vector.reshape(1, 3)
    elif args.shift_mode == "none":
        pass
    else:
        raise ValueError(f"Unsupported shift mode: {args.shift_mode}")

    opening_refinement = {"enabled": False}
    if not args.disable_opening_refine:
        transformed_pouch, opening_refinement = refine_opening_in_plane(
            pouch_mesh=transformed_pouch,
            opening_indices=opening_indices,
            ostium_points=ostium_points,
            normal=target_normal,
            objective=args.opening_refine_objective,
            max_shift=float(args.opening_refine_max_shift),
        )

    remesh_summary = {"enabled": False}
    if args.resample_aneurysm_to_vessel_resolution:
        vessel_stats = mesh_edge_stats(vessel_mesh)
        vessel_median_edge = vessel_stats["median"]
        if vessel_median_edge is None:
            raise RuntimeError("Cannot remesh aneurysm because the vessel mesh has no valid edge lengths.")
        target_edge_length = float(vessel_median_edge) * float(args.aneurysm_remesh_target_edge_scale)
        transformed_pouch, remesh_summary = remesh_to_target_edge_length(
            transformed_pouch,
            target_edge_length=target_edge_length,
            iterations=int(args.aneurysm_remesh_iterations),
        )
        remesh_summary["vessel_edge_stats"] = vessel_stats
        remesh_summary["target_edge_scale"] = float(args.aneurysm_remesh_target_edge_scale)
        opening_indices = recover_opening_indices_from_ostium(transformed_pouch, ostium_points)
        remesh_summary["recovered_opening_indices"] = int(opening_indices.shape[0])

    combined = trimesh.util.concatenate([vessel_mesh, transformed_pouch])

    out_dir = case_root / "outputs" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    pouch_world_path = out_dir / f"{args.case_name}_generated_aneurysm_world.obj"
    combined_path = out_dir / f"{args.case_name}_vessel_with_generated_aneurysm_unstitched.obj"
    colored_debug_path = out_dir / f"{args.case_name}_sample_with_ostium_colored.ply"
    stitched_path = out_dir / f"{args.case_name}_vessel_with_generated_aneurysm_stitched.obj"
    stitched_labels_path = out_dir / f"{args.case_name}_vessel_with_generated_aneurysm_stitched_labels.npy"
    save_mesh(transformed_pouch, pouch_world_path)
    save_mesh(combined, combined_path)
    save_colored_sample_ostium_ply(transformed_pouch, ostium_points, colored_debug_path)

    stitch_summary = {"enabled": False}
    if args.smooth_ostium_transition and not args.stitch:
        raise RuntimeError("--smooth-ostium-transition requires --stitch because the bridge exists only in stitched output.")
    if args.stitch:
        vessel_labels = infer_vessel_labels_from_ostium(vessel_mesh, ostium_points, ostium_label=2)
        if args.stitch_method == "bridge":
            if args.stitch_legacy_mode:
                stitch_loop_source = "auto"
                stitch_smooth_intersection = False
            else:
                stitch_loop_source = args.stitch_loop_source
                stitch_smooth_intersection = bool(args.stitch_smooth_intersection)
            stitched, stitched_labels, matches, stitch_metadata = stitch_meshes_bridge(
                vessel_submesh=vessel_mesh,
                labels_vessel_submesh=vessel_labels,
                transformed_mesh=transformed_pouch,
                ostium_points=ostium_points,
                opening_indices=opening_indices,
                normal=target_normal,
                bridge_steps=int(args.stitch_bridge_steps),
                ostium_label=2,
                transformed_label=1,
                merge_digits=int(args.stitch_merge_digits),
                loop_source=stitch_loop_source,
                smooth_intersection_enabled=stitch_smooth_intersection,
            )
            stitch_metadata["legacy_mode_requested"] = bool(args.stitch_legacy_mode)
        else:
            stitched, stitched_labels, matches, stitch_metadata = stitch_meshes(
                vessel_submesh=vessel_mesh,
                labels_vessel_submesh=vessel_labels,
                transformed_mesh=transformed_pouch,
                ostium_points=ostium_points,
                opening_indices=opening_indices,
                ostium_label=2,
                transformed_label=1,
                k_candidates=int(args.stitch_k_candidates),
                merge_digits=int(args.stitch_merge_digits),
            )
        if args.smooth_ostium_transition:
            stitched, transition_smoothing = smooth_ostium_transition_band(
                mesh=stitched,
                labels=stitched_labels,
                ostium_points=ostium_points,
                iterations=int(args.smooth_ostium_iterations),
                hops=int(args.smooth_ostium_hops),
                ostium_label=2,
                radius=None if args.smooth_ostium_radius is None else float(args.smooth_ostium_radius),
                lamb=float(args.smooth_ostium_lambda),
                nu=float(args.smooth_ostium_nu),
            )
            transition_smoothing.setdefault("legacy_parameters", {})
            transition_smoothing["legacy_parameters"]["radius_scale"] = float(args.smooth_ostium_radius_scale)
            stitch_metadata["ostium_transition_smoothing"] = transition_smoothing
        else:
            stitch_metadata["ostium_transition_smoothing"] = {"enabled": False}
        save_mesh(stitched, stitched_path)
        np.save(stitched_labels_path, stitched_labels)
        stitch_summary = {
            "enabled": True,
            "method": args.stitch_method,
            "stitched_path": str(stitched_path),
            "labels_path": str(stitched_labels_path),
            "metadata": stitch_metadata,
            "matches_preview": matches[:10].astype(int).tolist(),
            "stitched": mesh_summary(stitched_path),
        }

    pouch_center = np.asarray(transformed_pouch.vertices, dtype=np.float64).mean(axis=0)
    transformed_opening = np.asarray(transformed_pouch.vertices, dtype=np.float64)[opening_indices]
    opening_center_world = transformed_opening.mean(axis=0)
    nearest_distance = float(np.min(np.linalg.norm(np.asarray(transformed_pouch.vertices) - centroid.reshape(1, 3), axis=1)))
    from scipy.spatial import cKDTree

    target_tree = cKDTree(target_opening)
    ring_to_target_dist, _ = target_tree.query(transformed_opening, k=1)
    label2_tree = cKDTree(ostium_points)
    ring_to_label2_dist, _ = label2_tree.query(transformed_opening, k=1)
    label2_to_pouch_dist, _ = cKDTree(np.asarray(transformed_pouch.vertices, dtype=np.float64)).query(ostium_points, k=1)
    summary = {
        "step": "step3_compose",
        "source_sample": str(sample_path),
        "pouch_world_path": str(pouch_world_path),
        "combined_path": str(combined_path),
        "colored_sample_ostium_path": str(colored_debug_path),
        "centroid_ostium": centroid.round(8).tolist(),
        "target_normal": target_normal.round(8).tolist(),
        "output_normal_flipped": not bool(args.keep_output_normal),
        "output_normal": output_normal.round(8).tolist(),
        "canonical_normal": canonical_normal.round(8).tolist(),
        "raw_opening_center": raw_opening_center.round(8).tolist(),
        "raw_opening_radius": raw_opening_radius,
        "target_opening_radius": target_opening_radius,
        "output_scale_to_target_ostium": output_scale,
        "opening_alignment": opening_alignment,
        "shift_mode": args.shift_mode,
        "shift_raw": shift_raw.round(8).tolist(),
        "shift_axis": shift_axis,
        "shift_vector": shift_vector.round(8).tolist(),
        "normal_flush_anchor_before": normal_flush_anchor_before,
        "normal_flush_anchor_after": normal_flush_anchor_after,
        "normal_flush_projection_range_before": normal_flush_projection_range_before,
        "normal_flush_projection_range_after": normal_flush_projection_range_after,
        "one_axis_shift_enabled": args.shift_mode == "nearest-one-axis",
        "one_axis_shift_raw": shift_raw.round(8).tolist(),
        "one_axis_shift_axis": shift_axis,
        "one_axis_shift_vector": shift_vector.round(8).tolist(),
        "neck_snap_enabled": False,
        "opening_refinement": opening_refinement,
        "aneurysm_resolution_remesh": remesh_summary,
        "stitching": stitch_summary,
        "ring_to_target_mean_distance": float(np.mean(ring_to_target_dist)),
        "ring_to_target_max_distance": float(np.max(ring_to_target_dist)),
        "ring_to_label2_mean_distance": float(np.mean(ring_to_label2_dist)),
        "ring_to_label2_max_distance": float(np.max(ring_to_label2_dist)),
        "label2_to_pouch_mean_distance": float(np.mean(label2_to_pouch_dist)),
        "label2_to_pouch_max_distance": float(np.max(label2_to_pouch_dist)),
        "world_opening_center": opening_center_world.round(8).tolist(),
        "opening_center_to_ostium_distance": float(np.linalg.norm(opening_center_world - centroid)),
        "nearest_vertex_to_ostium_distance": nearest_distance,
        "pouch_center_to_ostium_distance": float(np.linalg.norm(pouch_center - centroid)),
        "vessel": mesh_summary(vessel_path),
        "pouch_world": mesh_summary(pouch_world_path),
        "combined": mesh_summary(combined_path),
    }
    out_path = case_root / "outputs" / "step3_compose_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    args = parse_args()
    if args.case_root is not None:
        args.case_root = args.case_root.expanduser()
    args.cases_root = args.cases_root.expanduser()
    args.bundle_root = args.bundle_root.expanduser()
    args.aneug_root = args.aneug_root.expanduser()
    args.stage1_checkpoint = args.stage1_checkpoint.expanduser()
    args.stage1_ghd_root = args.stage1_ghd_root.expanduser()
    args.stage1_alignment_root = args.stage1_alignment_root.expanduser()
    args.stage1_canonical_root = args.stage1_canonical_root.expanduser()
    if args.stage1_split_file is not None:
        args.stage1_split_file = args.stage1_split_file.expanduser()

    if args.step in ("all", "step1"):
        step1_create_opa(args)
    if args.step in ("all", "step2"):
        step2_infer(args)
    if args.step in ("all", "step3"):
        step3_compose(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
