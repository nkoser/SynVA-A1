"""
Vessel-aware GHD Dataset.

Extends GHDDataset with per-case ostium / vessel conditioning data loaded from
prepared_meshes_3.  Each sample returns:
    ghd:           [D]          normalised GHD coefficients  (same as original)
    ostium_params: [8]          centroid(3) + normal(3) + radius(1) + ecc(1)
    vessel_pts:    [N_vessel,3] local vessel surface points near the ostium

By default, conditioning geometry stays in the raw prepared_meshes_3 frame for
backward compatibility.  With condition_space="ghd_local", ostium/vessel points
are transformed into the same local canonical GHD frame as the coefficients.
"""
import os
import io
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Sequence, Tuple
from models.vessel_conditioner import OstiumFeatureExtractor


class _TorchCPUUnpickler(pickle.Unpickler):
    """Load pickle files containing torch CUDA storages on CPU-only sessions."""

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module, name)


def _load_checkpoint_cpu(path: str):
    with open(path, 'rb') as f:
        return _TorchCPUUnpickler(f).load()


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _axis_angle_to_matrix_np(axis_angle) -> np.ndarray:
    """Rodrigues formula matching pytorch3d.transforms.axis_angle_to_matrix."""
    vec = _to_numpy(axis_angle).reshape(-1)[:3].astype(np.float64)
    theta = float(np.linalg.norm(vec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)

    axis = vec / theta
    x, y, z = axis
    k = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    return eye * np.cos(theta) + (1.0 - np.cos(theta)) * np.outer(axis, axis) + np.sin(theta) * k


def _apply_homogeneous(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    return vec / (np.linalg.norm(vec) + 1e-12)


def _ostium_radius_ecc(ostium_verts: np.ndarray, centroid: np.ndarray) -> Tuple[float, float]:
    if ostium_verts is None or len(ostium_verts) < 3:
        return 0.1, 1.0

    centered = np.asarray(ostium_verts, dtype=np.float64) - centroid.reshape(1, 3)
    radius = float(np.linalg.norm(centered, axis=1).mean())
    try:
        _, svals, _ = np.linalg.svd(centered, full_matrices=False)
        ecc = float(svals[1] / (svals[0] + 1e-8))
    except np.linalg.LinAlgError:
        ecc = 1.0
    return radius, ecc


def _resample_closed_ring(points: np.ndarray, num_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        raise ValueError(f"Expected ring points with shape [N, 3], got {points.shape}.")
    if num_points < 3:
        raise ValueError("num_points must be at least 3.")
    diffs = np.roll(points, -1, axis=0) - points
    seg_lengths = np.linalg.norm(diffs, axis=1)
    if np.all(seg_lengths < 1e-12):
        raise ValueError("Cannot resample degenerate ostium ring.")
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(cumulative[-1])
    samples = np.linspace(0.0, total, num=num_points, endpoint=False)
    out = np.zeros((num_points, 3), dtype=np.float32)
    for idx, sample in enumerate(samples):
        seg_idx = min(np.searchsorted(cumulative, sample, side="right") - 1, points.shape[0] - 1)
        seg_len = seg_lengths[seg_idx]
        if seg_len <= 1e-12:
            out[idx] = points[seg_idx]
            continue
        alpha = (sample - cumulative[seg_idx]) / seg_len
        out[idx] = (1.0 - alpha) * points[seg_idx] + alpha * points[(seg_idx + 1) % points.shape[0]]
    return out


def _align_ring_to_reference(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if points.shape != reference.shape:
        raise ValueError(f"points and reference must have matching shapes, got {points.shape} vs {reference.shape}.")
    best = None
    best_error = None
    for candidate in (points, points[::-1].copy()):
        for shift in range(candidate.shape[0]):
            shifted = np.roll(candidate, -shift, axis=0)
            error = float(np.mean((shifted - reference) ** 2))
            if best_error is None or error < best_error:
                best = shifted.copy()
                best_error = error
    return best


def _load_reference_ring(canonical_opa_checkpoint: Optional[str], ring_points: int) -> Optional[np.ndarray]:
    if canonical_opa_checkpoint is None or not os.path.exists(canonical_opa_checkpoint):
        return None
    with open(canonical_opa_checkpoint, "rb") as f:
        chk = pickle.load(f)
    ring = np.asarray(chk["op_v_coords"][0], dtype=np.float32)
    return _resample_closed_ring(ring, ring_points)


def _load_case_opa_checkpoint(
    aligned_data_root: Optional[str],
    case: str,
    canonical_opa_checkpoint: Optional[str],
) -> Dict:
    if aligned_data_root is None:
        raise ValueError("ostium_source='opa_checkpoint' requires aligned_data_root")
    candidates = []
    if canonical_opa_checkpoint:
        candidates.append(os.path.basename(canonical_opa_checkpoint))
    candidates.extend(["opa_checkpoint_1op.pkl", "opa_checkpoint.pkl"])
    for name in dict.fromkeys(candidates):
        path = os.path.join(aligned_data_root, case, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(
        f"No OPA checkpoint found for {case} under {aligned_data_root}; tried {candidates}"
    )


def _load_case_opa_ring(aligned_data_root: Optional[str], case: str, canonical_opa_checkpoint: Optional[str]) -> np.ndarray:
    chk = _load_case_opa_checkpoint(aligned_data_root, case, canonical_opa_checkpoint)
    return np.asarray(chk["op_v_coords"][0], dtype=np.float32)


DEFAULT_MORPHOLOGY_KEYS = (
    "A_A",
    "V_A",
    "A_O1",
    "A_O2",
    "D_max",
    "H_max",
    "W_max",
    "H_ortho",
    "W_ortho",
    "N_max",
    "N_avg",
    "AR_1",
    "AR_2",
    "V_CH",
    "A_CH",
    "EI",
    "NSI",
    "UI",
)


def _parse_morphology_keys(keys: Optional[Sequence[str] | str]) -> List[str]:
    if keys is None or keys == "" or keys == "default":
        return list(DEFAULT_MORPHOLOGY_KEYS)
    if isinstance(keys, str):
        return [k.strip() for k in keys.split(",") if k.strip()]
    return [str(k).strip() for k in keys if str(k).strip()]


def _resolve_morphology_path(root: Optional[str], case: str) -> Optional[str]:
    if not root:
        return None
    prefixes = ("", "aneux_", "cmha_", "cmch_", "intra_")
    candidates = []
    for prefix in prefixes:
        candidates.append(os.path.join(root, f"{prefix}{case}", "07_other", "morphological_parameters.npy"))
    for prefix in prefixes[1:]:
        if case.startswith(prefix):
            candidates.append(os.path.join(root, case[len(prefix):], "07_other", "morphological_parameters.npy"))
    for path in dict.fromkeys(candidates):
        if os.path.exists(path):
            return path
    return None


def _morphology_vector_from_file(path: str, keys: Sequence[str]) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()
    if not isinstance(data, dict):
        raise ValueError(f"Expected morphology dict in {path}, got {type(data)}")
    values = []
    for key in keys:
        if key not in data:
            raise KeyError(f"Missing morphology key '{key}' in {path}")
        value = data[key]
        if key == "C_O":
            values.extend(np.asarray(value, dtype=np.float64).reshape(-1)[:3].tolist())
            continue
        if key in ("V_A", "V_CH"):
            values.append(abs(float(np.asarray(value).reshape(-1)[0])))
            continue
        if key == "NSI":
            values.append(abs(complex(np.asarray(value).reshape(-1)[0])))
            continue
        values.append(float(np.asarray(value).reshape(-1)[0]))
    out = np.asarray(values, dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError(f"Non-finite morphology vector in {path}: {out}")
    return out


def _ostium_from_opa_checkpoint(chk: Dict, num_vessel_pts: int, num_ostium_pts: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ring = np.asarray(chk["op_v_coords"][0], dtype=np.float32)
    centroid = ring.mean(axis=0).astype(np.float32)
    if "op_n_mean" in chk:
        normal = np.asarray(chk["op_n_mean"][0], dtype=np.float32).reshape(3)
    else:
        centered = ring.astype(np.float64) - centroid.reshape(1, 3)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[-1].astype(np.float32)
        except np.linalg.LinAlgError:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    radius, ecc = _ostium_radius_ecc(ring, centroid)
    ostium_params = np.concatenate([
        centroid,
        normal.astype(np.float32),
        np.array([radius], dtype=np.float32),
        np.array([ecc], dtype=np.float32),
    ]).astype(np.float32)
    ostium_pts = _fps_subsample(ring, num_ostium_pts)
    vessel_pts = _fps_subsample(ring, num_vessel_pts)
    return ostium_params, vessel_pts, ostium_pts.astype(np.float32), ring.astype(np.float32)


def _fps_subsample(pts: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    """Greedy farthest-point subsample of `pts` ([N,3]) to exactly K points.
    If N < K, repeat-pad. If N == K, return as-is."""
    pts = np.asarray(pts, dtype=np.float32)
    N = pts.shape[0]
    if N == 0:
        return np.zeros((K, 3), dtype=np.float32)
    if N == K:
        return pts
    if N < K:
        # pad with random repeats so chamfer is still well-defined
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, N, size=K - N)
        return np.concatenate([pts, pts[idx]], axis=0)
    # FPS
    rng = np.random.default_rng(seed)
    sel = np.empty(K, dtype=np.int64)
    sel[0] = int(rng.integers(0, N))
    d2 = ((pts - pts[sel[0]]) ** 2).sum(-1)
    for i in range(1, K):
        j = int(np.argmax(d2))
        sel[i] = j
        d2 = np.minimum(d2, ((pts - pts[j]) ** 2).sum(-1))
    return pts[sel]


def _resolve_prepared_case_dir(data_root: str, case: str) -> Optional[str]:
    candidates = [
        os.path.join(data_root, case),
        os.path.join(data_root, f"aneux_{case}"),
        os.path.join(data_root, case.replace("cmha_", "cmch_")),
        os.path.join(data_root, f"aneux_{case.replace('cmha_', 'cmch_')}"),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "07_other", "centroid_ostium.npy")):
            return candidate
    return None


def _load_pointcloud_vertices(path: str) -> np.ndarray:
    import trimesh

    cloud = trimesh.load(path, process=False)
    if isinstance(cloud, trimesh.Scene):
        verts = [np.asarray(geom.vertices, dtype=np.float32) for geom in cloud.geometry.values()]
        return np.concatenate(verts, axis=0) if verts else np.empty((0, 3), dtype=np.float32)
    return np.asarray(cloud.vertices, dtype=np.float32)


def _load_stage3_targets(
    data_root: str,
    aligned_data_root: Optional[str],
    case: str,
    ring_points: int,
    num_label2_pts: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    case_dir = _resolve_prepared_case_dir(data_root, case)
    if case_dir is None:
        raise FileNotFoundError(f"No prepared case dir for {case} under {data_root}")

    centroid = np.load(os.path.join(case_dir, "07_other", "centroid_ostium.npy")).astype(np.float32).reshape(3)
    normal = np.load(os.path.join(case_dir, "07_other", "normal_vector.npy")).astype(np.float32).reshape(3)
    normal = normal / (np.linalg.norm(normal) + 1e-8)

    label2_path = os.path.join(case_dir, "04_subpointclouds", "subpointcloud_label_2.ply")
    label2 = _load_pointcloud_vertices(label2_path) if os.path.exists(label2_path) else centroid.reshape(1, 3)
    label2 = _fps_subsample(label2.astype(np.float32), int(num_label2_pts))

    target_ring = None
    if aligned_data_root is not None:
        opa_path = os.path.join(aligned_data_root, case, "opa_checkpoint.pkl")
        if os.path.exists(opa_path):
            with open(opa_path, "rb") as f:
                opa_chk = pickle.load(f)
            target_ring = np.asarray(opa_chk["op_v_coords"][0], dtype=np.float32) + centroid.reshape(1, 3)
    if target_ring is None:
        target_ring = label2
    target_ring = _resample_closed_ring(target_ring, int(ring_points)).astype(np.float32)
    return label2.astype(np.float32), centroid.astype(np.float32), normal.astype(np.float32), target_ring


def _canonical_norm(canonical_mesh: str, norm_factor: float) -> float:
    import trimesh

    mesh = trimesh.load(canonical_mesh, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    return float(np.linalg.norm(verts, axis=1).max() * norm_factor)


def _condition_to_ghd_local(
    vf: Dict[str, np.ndarray],
    chk: Dict,
    prealign_transform: np.ndarray,
    canonical_norm: float,
    num_ostium_pts: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert raw prepared_meshes_3 condition geometry into GHD-local coordinates.

    Fitting forward path:
        p_target_norm = (p_ghd_local @ R.T) * s + T

    Therefore inverse path:
        p_ghd_local = ((p_target_norm - T) / s) @ R
    """
    r_mat = _axis_angle_to_matrix_np(chk['R'])
    s = float(np.abs(_to_numpy(chk['s']).reshape(-1)[0])) + 1e-12
    t = _to_numpy(chk['T']).reshape(-1, 3)[0].astype(np.float64)

    def points_to_local(points: np.ndarray) -> np.ndarray:
        target = _apply_homogeneous(points, prealign_transform) / canonical_norm
        return ((target - t.reshape(1, 3)) / s) @ r_mat

    def normal_to_local(normal: np.ndarray) -> np.ndarray:
        target_normal = np.asarray(normal, dtype=np.float64).reshape(3) @ prealign_transform[:3, :3].T
        return _normalize_vector(target_normal @ r_mat)

    vessel_pts = points_to_local(vf['vessel_local_pts']).astype(np.float32)
    ostium_centroid = points_to_local(vf['ostium_centroid'].reshape(1, 3))[0]
    ostium_normal = normal_to_local(vf['ostium_normal'])

    ostium_verts_raw = vf.get('ostium_verts')
    if ostium_verts_raw is not None and len(ostium_verts_raw) >= 3:
        ostium_verts = points_to_local(ostium_verts_raw)
        ostium_radius, ostium_ecc = _ostium_radius_ecc(ostium_verts, ostium_centroid)
        ostium_pts = _fps_subsample(ostium_verts.astype(np.float32), num_ostium_pts)
    else:
        ostium_radius = float(np.asarray(vf['ostium_radius']).reshape(-1)[0]) / canonical_norm / s
        ostium_ecc = float(np.asarray(vf['ostium_ecc']).reshape(-1)[0])
        # No raw ring available: fall back to centroid-tile (chamfer will be bad,
        # which correctly reflects the missing data).
        ostium_pts = np.tile(ostium_centroid.astype(np.float32).reshape(1, 3),
                             (num_ostium_pts, 1))

    ostium_ring_raw = vf.get('ostium_ring_verts', ostium_verts_raw)
    if ostium_ring_raw is not None and len(ostium_ring_raw) >= 3:
        ostium_ring = points_to_local(ostium_ring_raw).astype(np.float32)
    else:
        ostium_ring = ostium_pts.astype(np.float32)

    ostium_params = np.concatenate([
        ostium_centroid.astype(np.float32),
        ostium_normal.astype(np.float32),
        np.array([ostium_radius], dtype=np.float32),
        np.array([ostium_ecc], dtype=np.float32),
    ])
    return ostium_params, vessel_pts, ostium_pts.astype(np.float32), ostium_ring.astype(np.float32)


def _target_norm_points_to_ghd_local(points: np.ndarray, chk: Dict) -> np.ndarray:
    r_mat = _axis_angle_to_matrix_np(chk['R'])
    s = float(np.abs(_to_numpy(chk['s']).reshape(-1)[0])) + 1e-12
    t = _to_numpy(chk['T']).reshape(-1, 3)[0].astype(np.float64)
    points = np.asarray(points, dtype=np.float64)
    return ((points - t.reshape(1, 3)) / s) @ r_mat


def _alignment_vessel_condition(
    case: str,
    chk: Dict,
    ghd_chk_root: str,
    alignment_root: str,
    canonical_norm: float,
    num_vessel_pts: int,
    num_ostium_pts: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import trimesh

    opa_path = os.path.join(ghd_chk_root, case, "opa_checkpoint.pkl")
    if not os.path.exists(opa_path):
        opa_path = os.path.join(alignment_root, case, "opa_checkpoint.pkl")
    with open(opa_path, "rb") as f:
        opa_chk = pickle.load(f)

    ring_target_norm = np.asarray(opa_chk["op_v_coords"][0], dtype=np.float32)
    ring_local = _target_norm_points_to_ghd_local(ring_target_norm, chk).astype(np.float32)

    part_path = os.path.join(alignment_root, case, "part_aligned.obj")
    mesh = trimesh.load(part_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    verts_target_norm = np.asarray(mesh.vertices, dtype=np.float64) / float(canonical_norm)
    vessel_local_all = _target_norm_points_to_ghd_local(verts_target_norm, chk).astype(np.float32)

    centroid = ring_local.mean(axis=0).astype(np.float32)
    normal = np.asarray(opa_chk.get("op_n_mean", [np.array([0.0, 0.0, 1.0])])[0], dtype=np.float64)
    normal = (normal @ _axis_angle_to_matrix_np(chk['R'])).astype(np.float32)
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    radius, ecc = _ostium_radius_ecc(ring_local, centroid)

    d = np.linalg.norm(vessel_local_all - centroid.reshape(1, 3), axis=1)
    cutoff = max(3.0 * radius, np.percentile(d, 25))
    near = vessel_local_all[d <= cutoff]
    if len(near) < 8:
        near = vessel_local_all
    vessel_pts = _fps_subsample(near, num_vessel_pts)
    ostium_pts = _fps_subsample(ring_local, num_ostium_pts)
    ostium_params = np.concatenate([
        centroid,
        normal.astype(np.float32),
        np.array([radius], dtype=np.float32),
        np.array([ecc], dtype=np.float32),
    ]).astype(np.float32)
    return ostium_params, vessel_pts, ostium_pts.astype(np.float32), ring_local.astype(np.float32)


class VesselAwareGHDDataset(Dataset):
    """
    Loads:
      1) GHD coefficients from ghd checkpoint root  (same logic as GHDDataset)
      2) Ostium/vessel features from prepared_meshes_3 via OstiumFeatureExtractor
    Only cases present in BOTH roots are kept.
    """

    def __init__(
        self,
        ghd_chk_root: str,
        ghd_run: str,
        ghd_chk_name: str,
        data_root: str,                # e.g. /path/to/prepared_meshes_3
        cases: List[str],
        num_vessel_pts: int = 256,
        vessel_radius_factor: float = 3.0,
        normalize: bool = True,
        withscale: bool = False,
        condition_space: str = 'raw',
        aligned_data_root: Optional[str] = None,
        canonical_mesh: Optional[str] = None,
        canonical_norm_factor: float = 1.10,
        canonical_norm: Optional[float] = None,
        num_ostium_pts: int = 64,
        num_label2_pts: int = 256,
        ring_points: int = 20,
        canonical_opa_checkpoint: Optional[str] = "/path/to/SynVA-A1/checkpoints/canonical_average/opa_checkpoint_1op.pkl",
        ostium_source: str = "vessel_boundary",
        condition_data_mode: str = "prepared",
        morphology_root: Optional[str] = None,
        morphology_keys: Optional[Sequence[str] | str] = None,
    ):
        self.ghd_chk_root = ghd_chk_root
        self.ghd_run = ghd_run
        self.ghd_chk_name = ghd_chk_name
        self.data_root = data_root
        self.normalize_flag = normalize
        self.withscale = withscale
        self.num_vessel_pts = num_vessel_pts
        self.condition_space = condition_space
        self.aligned_data_root = aligned_data_root
        self.num_ostium_pts = int(num_ostium_pts)
        self.num_label2_pts = int(num_label2_pts)
        self.ring_points = int(ring_points)
        self.canonical_opa_checkpoint = canonical_opa_checkpoint
        if ostium_source not in ('opa_checkpoint', 'vessel_boundary', 'label2', 'label1'):
            raise ValueError("ostium_source must be 'opa_checkpoint', 'vessel_boundary', 'label2', or 'label1'")
        self.ostium_source = ostium_source
        if condition_data_mode not in ("prepared", "opa_only", "alignment_vessel"):
            raise ValueError("condition_data_mode must be 'prepared', 'opa_only', or 'alignment_vessel'")
        self.condition_data_mode = condition_data_mode
        self.morphology_root = morphology_root
        self.morphology_keys = _parse_morphology_keys(morphology_keys)
        self.reference_ring = (
            _load_reference_ring(canonical_opa_checkpoint, self.ring_points)
            if self.condition_space == 'ghd_local'
            else None
        )

        if self.condition_space not in ('raw', 'ghd_local'):
            raise ValueError("condition_space must be 'raw' or 'ghd_local'")
        if self.condition_space == 'ghd_local':
            if aligned_data_root is None:
                raise ValueError("condition_space='ghd_local' requires aligned_data_root")
            if canonical_norm is None:
                if canonical_mesh is None:
                    raise ValueError("condition_space='ghd_local' requires canonical_mesh or canonical_norm")
                canonical_norm = _canonical_norm(canonical_mesh, canonical_norm_factor)
            self.condition_canonical_norm = float(canonical_norm)
            print(
                "VesselAwareGHDDataset: transforming vessel/ostium conditions "
                f"to GHD-local frame (canonical_norm={self.condition_canonical_norm:.8f})"
            )
        else:
            self.condition_canonical_norm = None

        # ---- 1) extract ostium features for all candidate cases ----
        vessel_feats_all = None
        if condition_data_mode in ("opa_only", "alignment_vessel"):
            if ostium_source != "opa_checkpoint":
                raise ValueError(f"condition_data_mode='{condition_data_mode}' requires ostium_source='opa_checkpoint'")
            if condition_data_mode == "opa_only":
                if self.condition_space != "raw":
                    raise ValueError("condition_data_mode='opa_only' currently requires condition_space='raw'")
                print(
                    "VesselAwareGHDDataset: condition_data_mode=opa_only; "
                    "using OPA checkpoint geometry for ostium params, ordered ring, and placeholder vessel pts"
                )
            else:
                if aligned_data_root is None:
                    raise ValueError("condition_data_mode='alignment_vessel' requires aligned_data_root")
                if canonical_norm is None:
                    if canonical_mesh is None:
                        raise ValueError("condition_data_mode='alignment_vessel' requires canonical_mesh or canonical_norm")
                    canonical_norm = _canonical_norm(canonical_mesh, canonical_norm_factor)
                self.condition_canonical_norm = float(canonical_norm)
                print(
                    "VesselAwareGHDDataset: condition_data_mode=alignment_vessel; "
                    "transforming alignment part_aligned.obj vessel pts into GHD-local frame "
                    f"(canonical_norm={self.condition_canonical_norm:.8f})"
                )
        else:
            extractor_source = "vessel_boundary" if ostium_source == "opa_checkpoint" else ostium_source
            if ostium_source == "opa_checkpoint":
                print(
                    "VesselAwareGHDDataset: ordered ostium ring source=opa_checkpoint "
                    f"(ring_points={self.ring_points}); base vessel features source={extractor_source}"
                )
            extractor = OstiumFeatureExtractor(data_root, num_vessel_pts=num_vessel_pts,
                                               radius_factor=vessel_radius_factor,
                                               ostium_source=extractor_source)
            vessel_feats_all = extractor.extract_all(cases, verbose=True)

        # ---- 2) load GHD checkpoints, keep only matched cases ----
        self.case_names: List[str] = []
        self.ghd_list: List[torch.Tensor] = []
        self.scale_list: List[torch.Tensor] = []
        self.alignment_list: List[torch.Tensor] = []
        self.ostium_params_list: List[torch.Tensor] = []    # [8]
        self.vessel_pts_list: List[torch.Tensor] = []       # [N, 3]
        self.ostium_pts_list: List[torch.Tensor] = []       # [num_ostium_pts, 3]
        self.ostium_ring_list: List[torch.Tensor] = []      # [ring_points, 3]
        self.label2_pts_list: List[torch.Tensor] = []
        self.target_center_list: List[torch.Tensor] = []
        self.target_normal_list: List[torch.Tensor] = []
        self.target_ring_world_list: List[torch.Tensor] = []
        self.morphology_list: List[torch.Tensor] = []
        skipped_transform = 0
        skipped_morphology = 0

        for case in cases:
            ghd_path = os.path.join(ghd_chk_root, case, ghd_run, ghd_chk_name)
            if not os.path.exists(ghd_path):
                continue
            if condition_data_mode == "prepared" and case not in vessel_feats_all:
                continue
            # load ghd
            chk = _load_checkpoint_cpu(ghd_path)
            ghd_coeff = chk['GHD_coefficient'].view(-1)
            if torch.isnan(ghd_coeff).any() or torch.isinf(ghd_coeff).any():
                continue  # skip corrupted GHD fits
            R, s, T = chk['R'], chk['s'].abs(), chk['T']
            alignment = torch.cat((R.view(-1), s.view(-1), T.view(-1))).detach()
            scale = s.view(-1).detach()

            # load vessel/ostium
            if condition_data_mode == "opa_only":
                try:
                    opa_chk = _load_case_opa_checkpoint(
                        self.aligned_data_root,
                        case,
                        self.canonical_opa_checkpoint,
                    )
                    ostium_params, vessel_pts, ostium_pts, ostium_ring_raw = _ostium_from_opa_checkpoint(
                        opa_chk,
                        self.num_vessel_pts,
                        self.num_ostium_pts,
                    )
                except Exception as e:
                    print(f"[VesselAwareGHDDataset] skip {case}: OPA-only condition failed: {e}")
                    skipped_transform += 1
                    continue
            elif condition_data_mode == "alignment_vessel":
                try:
                    ostium_params, vessel_pts, ostium_pts, ostium_ring_raw = _alignment_vessel_condition(
                        case,
                        chk,
                        self.ghd_chk_root,
                        self.aligned_data_root,
                        self.condition_canonical_norm,
                        self.num_vessel_pts,
                        self.num_ostium_pts,
                    )
                except Exception as e:
                    print(f"[VesselAwareGHDDataset] skip {case}: alignment-vessel condition failed: {e}")
                    skipped_transform += 1
                    continue
            else:
                vf = vessel_feats_all[case]
                if self.condition_space == 'ghd_local':
                    prealign_path = os.path.join(aligned_data_root, case, 'prealign_transform.npy')
                    if not os.path.exists(prealign_path):
                        skipped_transform += 1
                        continue
                    try:
                        prealign_transform = np.load(prealign_path).astype(np.float64)
                        ostium_params, vessel_pts, ostium_pts, ostium_ring_raw = _condition_to_ghd_local(
                            vf, chk, prealign_transform, self.condition_canonical_norm,
                            num_ostium_pts=self.num_ostium_pts,
                        )
                        if self.ostium_source == "opa_checkpoint":
                            ostium_ring_raw = _load_case_opa_ring(
                                self.aligned_data_root,
                                case,
                                self.canonical_opa_checkpoint,
                            )
                    except Exception as e:
                        print(f"[VesselAwareGHDDataset] skip {case}: condition transform failed: {e}")
                        skipped_transform += 1
                        continue
                else:
                    ostium_params = np.concatenate([
                        vf['ostium_centroid'],   # 3
                        vf['ostium_normal'],     # 3
                        vf['ostium_radius'],     # 1
                        vf['ostium_ecc'],        # 1
                    ])  # → [8]
                    vessel_pts = vf['vessel_local_pts']
                    ov = vf.get('ostium_verts')
                    if ov is not None and len(ov) >= 3:
                        ostium_pts = _fps_subsample(np.asarray(ov, dtype=np.float32), self.num_ostium_pts)
                    else:
                        ostium_pts = np.tile(np.asarray(vf['ostium_centroid'], dtype=np.float32).reshape(1, 3),
                                             (self.num_ostium_pts, 1))
                    ostium_ring_raw = vf.get('ostium_ring_verts', ov)
                    if self.ostium_source == "opa_checkpoint":
                        ostium_ring_raw = _load_case_opa_ring(
                            self.aligned_data_root,
                            case,
                            self.canonical_opa_checkpoint,
                        )

            try:
                ostium_ring = _resample_closed_ring(ostium_ring_raw, self.ring_points)
                if self.reference_ring is not None:
                    ostium_ring = _align_ring_to_reference(ostium_ring, self.reference_ring)
                label2_pts, target_center, target_normal, target_ring_world = _load_stage3_targets(
                    self.data_root,
                    self.aligned_data_root,
                    case,
                    self.ring_points,
                    self.num_label2_pts,
                )
            except Exception as e:
                print(f"[VesselAwareGHDDataset] skip {case}: ordered ring/stage3 targets failed: {e}")
                skipped_transform += 1
                continue

            self.case_names.append(case)
            self.ghd_list.append(ghd_coeff)
            self.scale_list.append(scale)
            self.alignment_list.append(alignment)
            self.ostium_params_list.append(torch.from_numpy(ostium_params))
            self.vessel_pts_list.append(torch.from_numpy(vessel_pts))
            self.ostium_pts_list.append(torch.from_numpy(ostium_pts.astype(np.float32)))
            self.ostium_ring_list.append(torch.from_numpy(ostium_ring.astype(np.float32)))
            self.label2_pts_list.append(torch.from_numpy(label2_pts.astype(np.float32)))
            self.target_center_list.append(torch.from_numpy(target_center.astype(np.float32)))
            self.target_normal_list.append(torch.from_numpy(target_normal.astype(np.float32)))
            self.target_ring_world_list.append(torch.from_numpy(target_ring_world.astype(np.float32)))
            if self.morphology_root:
                morph_path = _resolve_morphology_path(self.morphology_root, case)
                if morph_path is None:
                    self.case_names.pop()
                    self.ghd_list.pop()
                    self.scale_list.pop()
                    self.alignment_list.pop()
                    self.ostium_params_list.pop()
                    self.vessel_pts_list.pop()
                    self.ostium_pts_list.pop()
                    self.ostium_ring_list.pop()
                    self.label2_pts_list.pop()
                    self.target_center_list.pop()
                    self.target_normal_list.pop()
                    self.target_ring_world_list.pop()
                    skipped_morphology += 1
                    continue
                try:
                    morph = _morphology_vector_from_file(morph_path, self.morphology_keys)
                except Exception as e:
                    self.case_names.pop()
                    self.ghd_list.pop()
                    self.scale_list.pop()
                    self.alignment_list.pop()
                    self.ostium_params_list.pop()
                    self.vessel_pts_list.pop()
                    self.ostium_pts_list.pop()
                    self.ostium_ring_list.pop()
                    self.label2_pts_list.pop()
                    self.target_center_list.pop()
                    self.target_normal_list.pop()
                    self.target_ring_world_list.pop()
                    skipped_morphology += 1
                    print(f"[VesselAwareGHDDataset] skip {case}: morphology load failed: {e}")
                    continue
                self.morphology_list.append(torch.from_numpy(morph))

        print(f"VesselAwareGHDDataset: {len(self.case_names)} cases loaded "
              f"(from {len(cases)} candidates)")
        if skipped_transform:
            print(f"VesselAwareGHDDataset: skipped {skipped_transform} cases during condition transform")
        if skipped_morphology:
            print(f"VesselAwareGHDDataset: skipped {skipped_morphology} cases due to missing/bad morphology")

        # ---- 3) normalise GHD (same as original GHDDataset) ----
        if self.withscale:
            stacked = torch.stack([torch.cat([g, s]) for g, s in
                                   zip(self.ghd_list, self.scale_list)], dim=0)
        else:
            stacked = torch.stack(self.ghd_list, dim=0)
        self.ghd_mean = stacked.mean(dim=0, keepdim=True)
        self.ghd_std  = stacked.std(dim=0, keepdim=True) + 0.01

        # ---- 4) normalise ostium params ----
        ostium_stack = torch.stack(self.ostium_params_list, dim=0)  # [N_cases, 8]
        self.ostium_mean = ostium_stack.mean(dim=0, keepdim=True)
        self.ostium_std  = ostium_stack.std(dim=0, keepdim=True) + 1e-6

        # ---- 5) normalise vessel points (per dataset, global centering + scale) ----
        all_vpts = torch.cat(self.vessel_pts_list, dim=0)           # [N_total, 3]
        self.vessel_center = all_vpts.mean(dim=0, keepdim=True)     # [1, 3]
        self.vessel_scale  = all_vpts.std() + 1e-6                  # scalar

        # ---- 6) normalise ordered ostium ring as its own condition vector ----
        ring_stack = torch.stack([ring.reshape(-1) for ring in self.ostium_ring_list], dim=0)
        self.ostium_ring_mean = ring_stack.mean(dim=0, keepdim=True)
        self.ostium_ring_std = ring_stack.std(dim=0, keepdim=True, unbiased=False) + 0.01

        if self.morphology_root:
            morphology_stack = torch.stack(self.morphology_list, dim=0)
            self.morphology_mean = morphology_stack.mean(dim=0, keepdim=True)
            self.morphology_std = morphology_stack.std(dim=0, keepdim=True, unbiased=False) + 1e-6
            self.morphology_feature_names = list(self.morphology_keys)

    # ----- public API (mirrors GHDDataset) ----- #

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, idx):
        ghd = self.ghd_list[idx]
        scale = self.scale_list[idx]
        x = torch.cat([ghd, scale]) if self.withscale else ghd
        if self.normalize_flag:
            x = (x - self.ghd_mean) / self.ghd_std

        ostium_params = self.ostium_params_list[idx]
        if self.normalize_flag:
            ostium_params = (ostium_params - self.ostium_mean) / self.ostium_std

        vessel_pts = self.vessel_pts_list[idx]
        if self.normalize_flag:
            vessel_pts = (vessel_pts - self.vessel_center) / self.vessel_scale

        ostium_pts = self.ostium_pts_list[idx]
        if self.normalize_flag:
            # Reuse vessel centering so ring + vessel share one frame.
            ostium_pts = (ostium_pts - self.vessel_center) / self.vessel_scale

        ostium_ring = self.ostium_ring_list[idx]
        if self.normalize_flag:
            flat_ring = ostium_ring.reshape(1, -1)
            ostium_ring = ((flat_ring - self.ostium_ring_mean) / self.ostium_ring_std).view(self.ring_points, 3)

        out = {
            'ghd':           x.view(-1),
            'ostium_params': ostium_params.view(-1),     # [8]
            'vessel_pts':    vessel_pts,                  # [N_vessel, 3]
            'ostium_pts':    ostium_pts,                  # [K_ring, 3]
            'ostium_ring':   ostium_ring,                 # [ring_points, 3]
            'alignment_rotation': self.alignment_list[idx][:3].view(-1),
            'alignment_scale': self.alignment_list[idx][3:4].view(-1),
            'alignment_translation': self.alignment_list[idx][4:].view(-1),
            'label2_pts': self.label2_pts_list[idx],
            'target_ostium_center': self.target_center_list[idx],
            'target_ostium_normal': self.target_normal_list[idx],
            'target_ring_world': self.target_ring_world_list[idx],
        }
        if self.morphology_root:
            morphology = self.morphology_list[idx]
            if self.normalize_flag:
                morphology = (morphology - self.morphology_mean.view(-1)) / self.morphology_std.view(-1)
            out['morphology'] = morphology.view(-1)
        return out

    def get_dim(self) -> int:
        if self.withscale:
            return self.ghd_list[0].numel() + self.scale_list[0].numel()
        return self.ghd_list[0].numel()

    def get_morphology_dim(self) -> int:
        if not self.morphology_root:
            return 0
        return int(self.morphology_list[0].numel())

    def get_mean_std(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.withscale:
            return self.ghd_mean[:, :-1], self.ghd_std[:, :-1]
        return self.ghd_mean, self.ghd_std

    def get_scale_mean_std(self):
        if self.withscale:
            return self.ghd_mean[:, -1], self.ghd_std[:, -1]
        return None, None

    def get_ostium_mean_std(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.ostium_mean, self.ostium_std

    def get_ostium_ring_mean_std(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.ostium_ring_mean, self.ostium_ring_std

    def de_normalize_ghd(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_flag:
            return x
        return x * self.ghd_std.to(x.device) + self.ghd_mean.to(x.device)

    def de_normalize_ostium(self, o: torch.Tensor) -> torch.Tensor:
        if not self.normalize_flag:
            return o
        return o * self.ostium_std.to(o.device) + self.ostium_mean.to(o.device)
