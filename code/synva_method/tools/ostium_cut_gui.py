#!/usr/bin/env python
"""Local Dash GUI for cutting an ostium hole into a healthy vessel mesh."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.attach_aneurysm_to_healthy import (  # noqa: E402
    _apply_h,
    _boundary_edges,
    _canonical_norm,
    _cut_healthy,
    _load_pickle,
    _jagged_radius,
    _normalize,
    _order_loop_by_angle,
    _order_loop_by_edges,
    _plane_basis,
    _select_loop,
    _axis_angle_to_matrix,
    _to_numpy,
)

try:
    from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - runtime environment hint
    raise SystemExit(
        "Dash/Plotly are missing. Start this app from the project env, e.g.:\n"
        "  conda run --no-capture-output -n unified_env python tools/ostium_cut_gui.py"
    ) from exc


APP_STATE: Dict[str, Dict[str, Any]] = {}
MAX_VIEW_FACES = 65000
MAX_PICK_POINTS = 9000
DEFAULT_ALIGNED_DATA_ROOT = "/path/to/ghd_prepared_meshes_3_aneurysm_1op_new"
DEFAULT_GHD_CHK_ROOT = "/path/to/SynVA-A1/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999"
DEFAULT_GHD_RUN = "prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3"
DEFAULT_GHD_CHK_NAME = "ghb_fitting_checkpoint.pkl"


CSS = """
:root {
  --bg: #f6f7f4;
  --panel: #ffffff;
  --ink: #18201c;
  --muted: #637068;
  --line: #d7ded5;
  --accent: #147d79;
  --accent-2: #c5653b;
  --warn: #a34b29;
  --good: #27724f;
  --shadow: 0 18px 50px rgba(24, 32, 28, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.app {
  min-height: 100vh;
  display: grid;
  grid-template-rows: 58px minmax(0, 1fr);
}
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 22px;
  border-bottom: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.88);
  backdrop-filter: blur(10px);
}
.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  font-weight: 740;
  font-size: 17px;
}
.brand-mark {
  width: 28px;
  height: 28px;
  border-radius: 7px;
  background:
    linear-gradient(135deg, rgba(20, 125, 121, 0.95), rgba(39, 114, 79, 0.92)),
    radial-gradient(circle at 75% 25%, rgba(246, 247, 244, 0.8), transparent 40%);
}
.top-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--muted);
  font-size: 13px;
}
.workspace {
  display: grid;
  grid-template-columns: minmax(330px, 390px) minmax(0, 1fr);
  min-height: 0;
}
.sidebar {
  overflow: auto;
  border-right: 1px solid var(--line);
  background: var(--panel);
  padding: 18px;
}
.viewer-wrap {
  min-width: 0;
  min-height: 0;
  padding: 18px;
}
.viewer-shell {
  height: calc(100vh - 94px);
  min-height: 560px;
  border: 1px solid var(--line);
  border-radius: 8px;
  overflow: hidden;
  background: #fbfcfa;
  box-shadow: var(--shadow);
}
.section {
  padding: 17px 0;
  border-bottom: 1px solid var(--line);
}
.section:first-child { padding-top: 0; }
.section-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
  font-size: 12px;
  font-weight: 760;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
.grid-3 {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 8px;
}
label {
  display: block;
  margin: 8px 0 5px;
  color: #3c4740;
  font-size: 12px;
  font-weight: 680;
}
input, select {
  width: 100%;
  min-height: 36px;
  border: 1px solid #cbd5cc;
  border-radius: 7px;
  padding: 8px 10px;
  color: var(--ink);
  background: #fbfcfa;
  font: inherit;
  font-size: 13px;
  outline: none;
}
input:focus, select:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(20, 125, 121, 0.12);
}
.button-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
  margin-top: 12px;
}
.button-row.three {
  grid-template-columns: 1fr 1fr 1fr;
}
button {
  min-height: 38px;
  border: 1px solid #bfcac1;
  border-radius: 7px;
  padding: 8px 11px;
  background: #ffffff;
  color: var(--ink);
  font-weight: 760;
  cursor: pointer;
}
button:hover {
  border-color: var(--accent);
  color: var(--accent);
}
.primary {
  border-color: var(--accent);
  background: var(--accent);
  color: white;
}
.primary:hover {
  background: #116d69;
  color: white;
}
.danger {
  border-color: rgba(163, 75, 41, 0.45);
  color: var(--warn);
}
.status {
  margin-top: 12px;
  min-height: 32px;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.35;
}
.status.good { color: var(--good); }
.status.warn { color: var(--warn); }
.stats {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-top: 10px;
}
.metric {
  min-height: 48px;
  padding: 9px 10px;
  border: 1px solid var(--line);
  border-radius: 7px;
  background: #fbfcfa;
}
.metric .k {
  color: var(--muted);
  font-size: 11px;
  font-weight: 700;
}
.metric .v {
  margin-top: 2px;
  font-size: 15px;
  font-weight: 760;
}
details {
  margin-top: 10px;
}
summary {
  cursor: pointer;
  color: var(--muted);
  font-size: 12px;
  font-weight: 720;
}
.mini {
  color: var(--muted);
  font-size: 12px;
}
.dash-graph, .js-plotly-plot {
  height: 100% !important;
}
@media (max-width: 980px) {
  .workspace {
    grid-template-columns: 1fr;
  }
  .sidebar {
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }
  .viewer-shell {
    height: 70vh;
  }
}
"""


def _state_for(session_id: Optional[str]) -> Dict[str, Any]:
    if not session_id:
        session_id = "default"
    return APP_STATE.setdefault(session_id, {})


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        if value is None or value == "":
            return float(fallback)
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        if value is None or value == "":
            return int(fallback)
        return int(value)
    except (TypeError, ValueError):
        return int(fallback)


def _load_mesh(path: str, merge: bool = True) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"Scene is empty: {path}")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected a triangle mesh, got {type(mesh).__name__}")
    mesh = mesh.copy()
    if merge:
        mesh.merge_vertices(digits_vertex=8, merge_tex=True, merge_norm=True)
        if hasattr(mesh, "unique_faces"):
            mesh.update_faces(mesh.unique_faces())
        if hasattr(mesh, "nondegenerate_faces"):
            mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    return mesh


def _case_paths(case: str, healthy_root: str, prepared_root: str) -> Dict[str, str]:
    case = (case or "").strip()
    return {
        "healthy": os.path.join(
            healthy_root,
            f"{case}_vessel_submesh_closed",
            f"{case}_vessel_submesh_closed.obj",
        ),
        "centroid": os.path.join(prepared_root, case, "07_other", "centroid_ostium.npy"),
        "normal": os.path.join(prepared_root, case, "07_other", "normal_vector.npy"),
        "prealign": os.path.join(DEFAULT_ALIGNED_DATA_ROOT, case, "prealign_transform.npy"),
        "ghd_checkpoint": os.path.join(DEFAULT_GHD_CHK_ROOT, case, DEFAULT_GHD_RUN, DEFAULT_GHD_CHK_NAME),
    }


def _mesh_defaults(mesh: trimesh.Trimesh) -> Dict[str, float]:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    diag = float(np.linalg.norm(bounds[1] - bounds[0]))
    diag = max(diag, 1e-6)
    center = verts.mean(axis=0)
    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "normal_x": 0.0,
        "normal_y": 0.0,
        "normal_z": 1.0,
        "radius": diag * 0.035,
        "slab": diag * 0.018,
    }


def _read_ostium_paths(
    case: str,
    prepared_root: str,
    centroid_path: str,
    normal_path: str,
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    defaults = _case_paths(case, "", prepared_root)
    centroid_path = (centroid_path or defaults["centroid"]).strip()
    normal_path = (normal_path or defaults["normal"]).strip()
    center = np.load(centroid_path).astype(np.float64).reshape(3)
    normal = _normalize(np.load(normal_path).astype(np.float64).reshape(3))
    return center, normal, centroid_path, normal_path


def _read_params(
    cx: Any,
    cy: Any,
    cz: Any,
    nx: Any,
    ny: Any,
    nz: Any,
    radius: Any,
    radius_scale: Any,
    slab: Any,
    jagged_amp: Any,
    harmonics: Any,
    seed: Any,
) -> Dict[str, Any]:
    center = np.array(
        [
            _coerce_float(cx, 0.0),
            _coerce_float(cy, 0.0),
            _coerce_float(cz, 0.0),
        ],
        dtype=np.float64,
    )
    normal = np.array(
        [
            _coerce_float(nx, 0.0),
            _coerce_float(ny, 0.0),
            _coerce_float(nz, 1.0),
        ],
        dtype=np.float64,
    )
    normal = _normalize(normal)
    return {
        "center": center,
        "normal": normal,
        "radius": max(_coerce_float(radius, 1.0), 1e-9),
        "radius_scale": max(_coerce_float(radius_scale, 1.0), 1e-9),
        "slab": max(_coerce_float(slab, 0.1), 1e-9),
        "jagged_amp": max(_coerce_float(jagged_amp, 0.0), 0.0),
        "harmonics": max(_coerce_int(harmonics, 7), 0),
        "seed": _coerce_int(seed, 17),
    }


def _read_condition_params(
    cx: Any,
    cy: Any,
    cz: Any,
    nx: Any,
    ny: Any,
    nz: Any,
    radius: Any,
    ecc: Any,
    num_points: Any,
    radius_factor: Any,
    seed: Any,
) -> Dict[str, Any]:
    center = np.array(
        [
            _coerce_float(cx, 0.0),
            _coerce_float(cy, 0.0),
            _coerce_float(cz, 0.0),
        ],
        dtype=np.float64,
    )
    normal = _normalize(
        np.array(
            [
                _coerce_float(nx, 0.0),
                _coerce_float(ny, 0.0),
                _coerce_float(nz, 1.0),
            ],
            dtype=np.float64,
        )
    )
    return {
        "center": center,
        "normal": normal,
        "radius": max(_coerce_float(radius, 1.0), 1e-9),
        "ecc": min(max(_coerce_float(ecc, 1.0), 1e-3), 1.0),
        "num_points": max(_coerce_int(num_points, 256), 1),
        "radius_factor": max(_coerce_float(radius_factor, 3.0), 1e-6),
        "seed": _coerce_int(seed, 17),
    }


def _params_key(params: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        tuple(np.round(params["center"], 8).tolist()),
        tuple(np.round(params["normal"], 8).tolist()),
        round(float(params["radius"]), 8),
        round(float(params["radius_scale"]), 8),
        round(float(params["slab"]), 8),
        round(float(params["jagged_amp"]), 8),
        int(params["harmonics"]),
        int(params["seed"]),
    )


def _compute_cut(mesh: trimesh.Trimesh, params: Dict[str, Any]) -> Dict[str, Any]:
    cut_mesh, removed = _cut_healthy(
        mesh,
        params["center"],
        params["normal"],
        params["radius"],
        params["radius_scale"],
        params["slab"],
        params["jagged_amp"],
        params["harmonics"],
        params["seed"],
    )
    loop = np.zeros((0,), dtype=np.int64)
    try:
        loop_idx, loop_edges = _select_loop(cut_mesh, params["center"])
        ordered = _order_loop_by_edges(loop_idx, loop_edges)
        if ordered is None:
            ordered = _order_loop_by_angle(
                np.asarray(cut_mesh.vertices),
                loop_idx,
                params["center"],
                params["normal"],
            )
        loop = np.asarray(ordered, dtype=np.int64)
    except Exception:
        loop = np.zeros((0,), dtype=np.int64)
    return {
        "mesh": cut_mesh,
        "removed": np.asarray(removed, dtype=bool),
        "loop": loop,
        "params_key": _params_key(params),
    }


def _ostium_radius_ecc(points: np.ndarray, center: np.ndarray) -> Tuple[float, float]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    if points.shape[0] < 3:
        return 0.1, 1.0
    rel = points - center.reshape(1, 3)
    radius = float(np.linalg.norm(rel, axis=1).mean())
    try:
        _, svals, _ = np.linalg.svd(rel, full_matrices=False)
        ecc = float(svals[1] / (svals[0] + 1e-8))
    except np.linalg.LinAlgError:
        ecc = 1.0
    return radius, min(max(ecc, 1e-3), 1.0)


def _ellipse_points(center: np.ndarray, normal: np.ndarray, radius: float, ecc: float, samples: int = 180) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=True)
    u, v = _plane_basis(normal)
    return (
        center.reshape(1, 3)
        + np.cos(theta).reshape(-1, 1) * float(radius) * u.reshape(1, 3)
        + np.sin(theta).reshape(-1, 1) * float(radius) * float(ecc) * v.reshape(1, 3)
    )


def _sample_vessel_condition_points(
    mesh: trimesh.Trimesh,
    center: np.ndarray,
    radius: float,
    radius_factor: float,
    num_points: int,
    seed: int,
) -> np.ndarray:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.shape[0] == 0:
        raise ValueError("Loaded vessel mesh has no vertices")
    dists = np.linalg.norm(verts - center.reshape(1, 3), axis=1)
    cutoff = float(radius_factor) * float(radius)
    local = verts[dists < cutoff]
    if local.shape[0] == 0:
        local = verts
    rng = np.random.default_rng(int(seed))
    replace = local.shape[0] < int(num_points)
    idx = rng.choice(local.shape[0], int(num_points), replace=replace)
    return local[idx].astype(np.float32)


def _condition_to_ghd_local(
    ostium_params: np.ndarray,
    vessel_pts: np.ndarray,
    contour: np.ndarray,
    prealign_path: str,
    ghd_checkpoint_path: str,
    canonical_mesh_path: str,
    canonical_norm_factor: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not prealign_path or not os.path.exists(prealign_path):
        raise ValueError("condition_space='ghd_local' needs a valid prealign_transform.npy")
    if not ghd_checkpoint_path or not os.path.exists(ghd_checkpoint_path):
        raise ValueError("condition_space='ghd_local' needs a valid GHD checkpoint")
    if not canonical_mesh_path or not os.path.exists(canonical_mesh_path):
        raise ValueError("condition_space='ghd_local' needs a valid canonical mesh")

    prealign = np.load(prealign_path).astype(np.float64)
    chk = _load_pickle(ghd_checkpoint_path)
    r_mat = _axis_angle_to_matrix(chk["R"])
    scale = float(np.abs(_to_numpy(chk["s"]).reshape(-1)[0])) + 1e-12
    trans = _to_numpy(chk["T"]).reshape(-1, 3)[0].astype(np.float64)
    c_norm = _canonical_norm(canonical_mesh_path, float(canonical_norm_factor))

    def points_to_local(points: np.ndarray) -> np.ndarray:
        target = _apply_h(np.asarray(points, dtype=np.float64), prealign) / c_norm
        return ((target - trans.reshape(1, 3)) / scale) @ r_mat

    def normal_to_local(normal: np.ndarray) -> np.ndarray:
        target_normal = np.asarray(normal, dtype=np.float64).reshape(3) @ prealign[:3, :3].T
        return _normalize(target_normal @ r_mat)

    center_raw = np.asarray(ostium_params[:3], dtype=np.float64)
    normal_raw = np.asarray(ostium_params[3:6], dtype=np.float64)
    center_local = points_to_local(center_raw.reshape(1, 3))[0]
    normal_local = normal_to_local(normal_raw)
    contour_local = points_to_local(contour)
    radius_local, ecc_local = _ostium_radius_ecc(contour_local, center_local)
    ostium_local = np.concatenate(
        [
            center_local.astype(np.float32),
            normal_local.astype(np.float32),
            np.array([radius_local], dtype=np.float32),
            np.array([ecc_local], dtype=np.float32),
        ]
    )
    return ostium_local, points_to_local(vessel_pts).astype(np.float32), contour_local.astype(np.float32)


def _build_cvae_condition(
    mesh: trimesh.Trimesh,
    condition_params: Dict[str, Any],
    condition_space: str,
    prealign_path: str,
    ghd_checkpoint_path: str,
    canonical_mesh_path: str,
    canonical_norm_factor: float,
) -> Dict[str, Any]:
    contour_raw = _ellipse_points(
        condition_params["center"],
        condition_params["normal"],
        condition_params["radius"],
        condition_params["ecc"],
        samples=180,
    ).astype(np.float32)
    vessel_raw = _sample_vessel_condition_points(
        mesh,
        condition_params["center"],
        condition_params["radius"],
        condition_params["radius_factor"],
        condition_params["num_points"],
        condition_params["seed"],
    )
    ostium_raw = np.concatenate(
        [
            condition_params["center"].astype(np.float32),
            condition_params["normal"].astype(np.float32),
            np.array([condition_params["radius"]], dtype=np.float32),
            np.array([condition_params["ecc"]], dtype=np.float32),
        ]
    )

    if condition_space == "ghd_local":
        ostium_params, vessel_pts, contour = _condition_to_ghd_local(
            ostium_raw,
            vessel_raw,
            contour_raw,
            prealign_path,
            ghd_checkpoint_path,
            canonical_mesh_path,
            canonical_norm_factor,
        )
    elif condition_space == "raw":
        ostium_params, vessel_pts, contour = ostium_raw, vessel_raw, contour_raw
    else:
        raise ValueError("condition_space must be 'raw' or 'ghd_local'")

    return {
        "ostium_params": ostium_params.astype(np.float32),
        "vessel_pts": vessel_pts.astype(np.float32),
        "ostium_contour": contour.astype(np.float32),
        "ostium_params_raw": ostium_raw.astype(np.float32),
        "vessel_pts_raw": vessel_raw.astype(np.float32),
        "ostium_contour_raw": contour_raw.astype(np.float32),
        "condition_space": condition_space,
    }


def _sample_faces(faces: np.ndarray, max_faces: int = MAX_VIEW_FACES) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.shape[0] <= max_faces:
        return faces
    step = max(1, int(math.ceil(faces.shape[0] / max_faces)))
    return faces[::step]


def _sample_vertices(vertices: np.ndarray, max_points: int = MAX_PICK_POINTS) -> np.ndarray:
    count = int(vertices.shape[0])
    if count <= max_points:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.linspace(0, count - 1, max_points, dtype=np.int64))


def _ring_points(params: Dict[str, Any], samples: int = 180) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=True)
    radius = _jagged_radius(
        theta,
        params["radius"] * params["radius_scale"],
        params["jagged_amp"],
        params["harmonics"],
        params["seed"],
    )
    u, v = _plane_basis(params["normal"])
    return (
        params["center"].reshape(1, 3)
        + np.cos(theta).reshape(-1, 1) * radius.reshape(-1, 1) * u.reshape(1, 3)
        + np.sin(theta).reshape(-1, 1) * radius.reshape(-1, 1) * v.reshape(1, 3)
    )


def _empty_figure(message: str = "Load a vessel mesh") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 17, "color": "#637068"},
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="#fbfcfa",
        plot_bgcolor="#fbfcfa",
    )
    return fig


def _mesh_trace(mesh: trimesh.Trimesh, name: str, color: str, opacity: float) -> go.Mesh3d:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = _sample_faces(np.asarray(mesh.faces, dtype=np.int64))
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=opacity,
        flatshading=True,
        lighting={"ambient": 0.54, "diffuse": 0.68, "specular": 0.18, "roughness": 0.72},
        lightposition={"x": 0, "y": 0, "z": 100},
        name=name,
        hoverinfo="skip",
    )


def _figure_for(
    state: Dict[str, Any],
    mesh_info: Optional[Dict[str, Any]],
    preview_info: Optional[Dict[str, Any]],
    params: Dict[str, Any],
    show_removed: list,
) -> go.Figure:
    mesh = state.get("mesh")
    if mesh is None or not mesh_info:
        return _empty_figure()

    preview = state.get("preview")
    render_mesh = preview["mesh"] if preview and preview_info else mesh
    fig = go.Figure()
    fig.add_trace(_mesh_trace(render_mesh, "cut vessel" if preview else "vessel", "#b8c7bd", 0.82))

    if preview and preview_info and "removed" in (show_removed or []):
        removed = np.asarray(preview["removed"], dtype=bool)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if removed.shape[0] == faces.shape[0] and np.any(removed):
            removed_mesh = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices, dtype=np.float64),
                faces=faces[removed],
                process=False,
            )
            fig.add_trace(_mesh_trace(removed_mesh, "removed faces", "#c5653b", 0.42))

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    pick_idx = _sample_vertices(verts)
    fig.add_trace(
        go.Scatter3d(
            x=verts[pick_idx, 0],
            y=verts[pick_idx, 1],
            z=verts[pick_idx, 2],
            mode="markers",
            marker={"size": 2.2, "color": "#18201c", "opacity": 0.24},
            customdata=pick_idx,
            name="pick points",
            hovertemplate="vertex %{customdata}<extra></extra>",
        )
    )

    center = params["center"]
    normal = params["normal"]
    arrow_len = max(params["radius"] * params["radius_scale"] * 2.2, params["slab"] * 3.0)
    arrow = np.vstack([center, center + normal * arrow_len])
    fig.add_trace(
        go.Scatter3d(
            x=[center[0]],
            y=[center[1]],
            z=[center[2]],
            mode="markers",
            marker={"size": 6.5, "color": "#e1a732", "line": {"color": "#18201c", "width": 1}},
            name="center",
            hovertemplate="center<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=arrow[:, 0],
            y=arrow[:, 1],
            z=arrow[:, 2],
            mode="lines",
            line={"color": "#147d79", "width": 7},
            name="normal",
            hoverinfo="skip",
        )
    )

    ring = _ring_points(params)
    fig.add_trace(
        go.Scatter3d(
            x=ring[:, 0],
            y=ring[:, 1],
            z=ring[:, 2],
            mode="lines",
            line={"color": "#a34b29", "width": 5, "dash": "dash"},
            name="planned contour",
            hoverinfo="skip",
        )
    )

    if preview and preview_info and preview.get("loop") is not None and len(preview["loop"]) > 2:
        cut_verts = np.asarray(preview["mesh"].vertices, dtype=np.float64)
        loop = np.asarray(preview["loop"], dtype=np.int64)
        loop_points = cut_verts[np.r_[loop, loop[0]]]
        fig.add_trace(
            go.Scatter3d(
                x=loop_points[:, 0],
                y=loop_points[:, 1],
                z=loop_points[:, 2],
                mode="lines",
                line={"color": "#27724f", "width": 8},
                name="cut boundary",
                hoverinfo="skip",
            )
        )

    condition = state.get("condition")
    if condition is not None:
        vpts = np.asarray(condition["vessel_pts_raw"], dtype=np.float64)
        if vpts.shape[0] > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=vpts[:, 0],
                    y=vpts[:, 1],
                    z=vpts[:, 2],
                    mode="markers",
                    marker={"size": 3.0, "color": "#147d79", "opacity": 0.72},
                    name="CVAE vessel pts",
                    hoverinfo="skip",
                )
            )
        contour = np.asarray(condition["ostium_contour_raw"], dtype=np.float64)
        if contour.shape[0] > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=contour[:, 0],
                    y=contour[:, 1],
                    z=contour[:, 2],
                    mode="lines",
                    line={"color": "#111827", "width": 6},
                    name="CVAE ostium",
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="#fbfcfa",
        plot_bgcolor="#fbfcfa",
        clickmode="event+select",
        legend={
            "orientation": "h",
            "x": 0.02,
            "y": 0.98,
            "bgcolor": "rgba(251,252,250,0.72)",
            "bordercolor": "#d7ded5",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        scene={
            "aspectmode": "data",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "bgcolor": "#fbfcfa",
        },
    )
    return fig


def _metric(label: str, value: str) -> html.Div:
    return html.Div([html.Div(label, className="k"), html.Div(value, className="v")], className="metric")


def _stats_block(mesh_info: Optional[Dict[str, Any]], preview_info: Optional[Dict[str, Any]]) -> html.Div:
    if not mesh_info:
        return html.Div(className="stats")
    metrics = [
        _metric("vertices", f"{int(mesh_info.get('vertices', 0)):,}"),
        _metric("faces", f"{int(mesh_info.get('faces', 0)):,}"),
    ]
    if preview_info:
        metrics.extend(
            [
                _metric("removed", f"{int(preview_info.get('removed_faces', 0)):,}"),
                _metric("boundary", f"{int(preview_info.get('loop_vertices', 0)):,}"),
            ]
        )
    return html.Div(metrics, className="stats")


def _default_output_paths(case: str, healthy_path: str) -> Tuple[str, str, str]:
    label = (case or "").strip()
    if not label:
        label = Path(healthy_path or "vessel").stem
    out_dir = REPO_ROOT / "checkpoints" / "ostium_cut_gui" / label
    return (
        str(out_dir / "cvae_condition.npz"),
        str(out_dir / f"{label}_cut_vessel.obj"),
        str(out_dir / "cut_report.json"),
    )


def build_app(args: argparse.Namespace) -> Dash:
    app = Dash(__name__, title="Ostium Cutter", suppress_callback_exceptions=True)
    app.index_string = f"""<!DOCTYPE html>
<html>
  <head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <style>{CSS}</style>
  </head>
  <body>
    {{%app_entry%}}
    <footer>
      {{%config%}}
      {{%scripts%}}
      {{%renderer%}}
    </footer>
  </body>
</html>"""

    def serve_layout() -> html.Div:
        session_id = uuid.uuid4().hex
        _state_for(session_id)
        return html.Div(
            [
                dcc.Store(id="session-id", data=session_id),
                dcc.Store(id="mesh-info"),
                dcc.Store(id="preview-info"),
                dcc.Store(id="condition-info"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(className="brand-mark"),
                                html.Div("Ostium Cutter"),
                            ],
                            className="brand",
                        ),
                        html.Div("local mesh editor", className="top-actions"),
                    ],
                    className="topbar",
                ),
                html.Div(
                    [
                        html.Aside(
                            [
                                html.Div(
                                    [
                                        html.Div("Input", className="section-title"),
                                        html.Label("Case"),
                                        dcc.Input(id="case-input", value=args.case or "", placeholder="aneux_C0075"),
                                        html.Label("Healthy vessel OBJ"),
                                        dcc.Input(
                                            id="healthy-path",
                                            value=args.healthy_mesh or "",
                                            placeholder="/path/to/healthy_vessel/.../vessel.obj",
                                        ),
                                        html.Div(
                                            [
                                                html.Button("Load", id="load-vessel", className="primary"),
                                                html.Button("Load ostium", id="load-ostium"),
                                            ],
                                            className="button-row",
                                        ),
                                        html.Div(id="load-status", className="status"),
                                        html.Div(id="pick-status", className="status"),
                                        html.Div(id="stats-block"),
                                        html.Details(
                                            [
                                                html.Summary("Paths"),
                                                html.Label("Healthy root"),
                                                dcc.Input(id="healthy-root", value=args.healthy_root),
                                                html.Label("Prepared root"),
                                                dcc.Input(id="prepared-root", value=args.prepared_root),
                                                html.Label("Centroid NPY"),
                                                dcc.Input(id="centroid-path", value=args.ostium_centroid or ""),
                                                html.Label("Normal NPY"),
                                                dcc.Input(id="normal-path", value=args.ostium_normal or ""),
                                            ]
                                        ),
                                    ],
                                    className="section",
                                ),
                                html.Div(
                                    [
                                        html.Div("Ostium", className="section-title"),
                                        html.Div(
                                            [
                                                html.Div([html.Label("X"), dcc.Input(id="center-x", type="number")]),
                                                html.Div([html.Label("Y"), dcc.Input(id="center-y", type="number")]),
                                                html.Div([html.Label("Z"), dcc.Input(id="center-z", type="number")]),
                                            ],
                                            className="grid-3",
                                        ),
                                        html.Div(
                                            [
                                                html.Div([html.Label("NX"), dcc.Input(id="normal-x", type="number")]),
                                                html.Div([html.Label("NY"), dcc.Input(id="normal-y", type="number")]),
                                                html.Div([html.Label("NZ"), dcc.Input(id="normal-z", type="number")]),
                                            ],
                                            className="grid-3",
                                        ),
                                        html.Div(
                                            [
                                                html.Button("Flip", id="flip-normal", title="Invert normal"),
                                                html.Button("Normalize", id="normalize-normal"),
                                                html.Button("Clear preview", id="clear-preview"),
                                            ],
                                            className="button-row three",
                                        ),
                                    ],
                                    className="section",
                                ),
                                html.Div(
                                    [
                                        html.Div("CVAE Condition", className="section-title"),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label("Ecc"),
                                                        dcc.Input(id="ostium-ecc", type="number", value=1.0, min=0.001, max=1.0, step=0.01),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label("Vessel pts"),
                                                        dcc.Input(id="num-vessel-pts", type="number", value=256, step=1),
                                                    ]
                                                ),
                                            ],
                                            className="grid-2",
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label("Radius factor"),
                                                        dcc.Input(id="vessel-radius-factor", type="number", value=3.0, step=0.1),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label("Space"),
                                                        dcc.Dropdown(
                                                            id="condition-space",
                                                            options=[
                                                                {"label": "raw", "value": "raw"},
                                                                {"label": "ghd_local", "value": "ghd_local"},
                                                            ],
                                                            value="raw",
                                                            clearable=False,
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            className="grid-2",
                                        ),
                                        html.Div(
                                            [
                                                html.Button("Preview condition", id="preview-condition", className="primary"),
                                                html.Button("Export condition", id="export-condition"),
                                            ],
                                            className="button-row",
                                        ),
                                        html.Div(id="condition-status", className="status"),
                                        html.Details(
                                            [
                                                html.Summary("GHD-local transform"),
                                                html.Label("Prealign transform"),
                                                dcc.Input(id="prealign-path", value=""),
                                                html.Label("GHD checkpoint"),
                                                dcc.Input(id="ghd-checkpoint-path", value=""),
                                                html.Label("Canonical mesh"),
                                                dcc.Input(id="canonical-mesh-path", value=args.canonical_mesh),
                                                html.Label("Canonical norm factor"),
                                                dcc.Input(id="canonical-norm-factor", type="number", value=1.10, step=0.01),
                                            ]
                                        ),
                                    ],
                                    className="section",
                                ),
                                html.Div(
                                    [
                                        html.Div("Cut", className="section-title"),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label("Radius"),
                                                        dcc.Input(id="cut-radius", type="number", step=0.001),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label("Scale"),
                                                        dcc.Input(id="radius-scale", type="number", value=1.1, step=0.01),
                                                    ]
                                                ),
                                            ],
                                            className="grid-2",
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label("Slab"),
                                                        dcc.Input(id="cut-slab", type="number", step=0.001),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label("Jagged"),
                                                        dcc.Input(id="jagged-amp", type="number", value=0.12, step=0.01),
                                                    ]
                                                ),
                                            ],
                                            className="grid-2",
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label("Harmonics"),
                                                        dcc.Input(id="jagged-harmonics", type="number", value=7, step=1),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        html.Label("Seed"),
                                                        dcc.Input(id="seed", type="number", value=17, step=1),
                                                    ]
                                                ),
                                            ],
                                            className="grid-2",
                                        ),
                                        dcc.Checklist(
                                            id="show-removed",
                                            options=[{"label": " removed faces", "value": "removed"}],
                                            value=["removed"],
                                            className="mini",
                                        ),
                                        html.Div(
                                            [
                                                html.Button("Preview", id="preview-cut", className="primary"),
                                                html.Button("Export", id="export-cut"),
                                            ],
                                            className="button-row",
                                        ),
                                        html.Div(id="preview-status", className="status"),
                                    ],
                                    className="section",
                                ),
                                html.Div(
                                    [
                                        html.Div("Output", className="section-title"),
                                        html.Label("CVAE condition NPZ"),
                                        dcc.Input(id="out-condition-path", value=""),
                                        html.Label("Cut vessel OBJ"),
                                        dcc.Input(id="out-cut-path", value=""),
                                        html.Label("Report JSON"),
                                        dcc.Input(id="out-report-path", value=""),
                                        html.Div(id="export-status", className="status"),
                                    ],
                                    className="section",
                                ),
                            ],
                            className="sidebar",
                        ),
                        html.Main(
                            [
                                html.Div(
                                    dcc.Loading(
                                        dcc.Graph(
                                            id="mesh-viewer",
                                            figure=_empty_figure(),
                                            config={"displaylogo": False, "scrollZoom": True, "responsive": True},
                                            style={"height": "100%"},
                                        ),
                                        type="circle",
                                    ),
                                    className="viewer-shell",
                                )
                            ],
                            className="viewer-wrap",
                        ),
                    ],
                    className="workspace",
                ),
            ],
            className="app",
        )

    app.layout = serve_layout

    @app.callback(
        Output("mesh-info", "data"),
        Output("preview-info", "data", allow_duplicate=True),
        Output("condition-info", "data", allow_duplicate=True),
        Output("healthy-path", "value"),
        Output("centroid-path", "value"),
        Output("normal-path", "value"),
        Output("prealign-path", "value"),
        Output("ghd-checkpoint-path", "value"),
        Output("cut-radius", "value"),
        Output("cut-slab", "value"),
        Output("out-condition-path", "value"),
        Output("out-cut-path", "value"),
        Output("out-report-path", "value"),
        Output("load-status", "children"),
        Output("load-status", "className"),
        Input("load-vessel", "n_clicks"),
        State("session-id", "data"),
        State("case-input", "value"),
        State("healthy-path", "value"),
        State("healthy-root", "value"),
        State("prepared-root", "value"),
        prevent_initial_call=True,
    )
    def load_vessel(n_clicks, session_id, case, healthy_path, healthy_root, prepared_root):
        del n_clicks
        try:
            paths = _case_paths(case, healthy_root, prepared_root)
            resolved_healthy = (healthy_path or "").strip() or paths["healthy"]
            if not resolved_healthy or not os.path.isfile(resolved_healthy):
                raise FileNotFoundError(f"Healthy vessel not found: {resolved_healthy}")
            mesh = _load_mesh(resolved_healthy, merge=True)
            state = _state_for(session_id)
            state.clear()
            state["mesh"] = mesh
            state["mesh_path"] = resolved_healthy
            state["preview"] = None
            defaults = _mesh_defaults(mesh)
            out_condition, out_cut, out_report = _default_output_paths(case, resolved_healthy)
            info = {
                "path": resolved_healthy,
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "bounds": np.asarray(mesh.bounds, dtype=float).tolist(),
                "defaults": defaults,
            }
            return (
                info,
                None,
                None,
                resolved_healthy,
                paths["centroid"],
                paths["normal"],
                paths["prealign"],
                paths["ghd_checkpoint"],
                round(defaults["radius"], 6),
                round(defaults["slab"], 6),
                out_condition,
                out_cut,
                out_report,
                f"Loaded {Path(resolved_healthy).name}",
                "status good",
            )
        except Exception as exc:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                str(exc),
                "status warn",
            )

    @app.callback(
        Output("center-x", "value"),
        Output("center-y", "value"),
        Output("center-z", "value"),
        Output("normal-x", "value"),
        Output("normal-y", "value"),
        Output("normal-z", "value"),
        Output("pick-status", "children"),
        Output("pick-status", "className"),
        Input("mesh-info", "data"),
        Input("load-ostium", "n_clicks"),
        Input("mesh-viewer", "clickData"),
        Input("flip-normal", "n_clicks"),
        Input("normalize-normal", "n_clicks"),
        State("session-id", "data"),
        State("case-input", "value"),
        State("prepared-root", "value"),
        State("centroid-path", "value"),
        State("normal-path", "value"),
        State("center-x", "value"),
        State("center-y", "value"),
        State("center-z", "value"),
        State("normal-x", "value"),
        State("normal-y", "value"),
        State("normal-z", "value"),
        prevent_initial_call=True,
    )
    def update_selection(
        mesh_info,
        load_ostium_clicks,
        click_data,
        flip_clicks,
        normalize_clicks,
        session_id,
        case,
        prepared_root,
        centroid_path,
        normal_path,
        cx,
        cy,
        cz,
        nx,
        ny,
        nz,
    ):
        del load_ostium_clicks, flip_clicks, normalize_clicks
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        state = _state_for(session_id)
        mesh = state.get("mesh")
        try:
            if trigger == "mesh-info" and mesh_info and mesh is not None:
                defaults = mesh_info.get("defaults") or _mesh_defaults(mesh)
                try:
                    center, normal, _, _ = _read_ostium_paths(case, prepared_root, centroid_path, normal_path)
                    msg = "Ostium files loaded"
                except Exception:
                    center = np.array([defaults["center_x"], defaults["center_y"], defaults["center_z"]])
                    normal = np.array([defaults["normal_x"], defaults["normal_y"], defaults["normal_z"]])
                    msg = "Default center ready"
                return (
                    round(float(center[0]), 6),
                    round(float(center[1]), 6),
                    round(float(center[2]), 6),
                    round(float(normal[0]), 6),
                    round(float(normal[1]), 6),
                    round(float(normal[2]), 6),
                    msg,
                    "status good",
                )

            if trigger == "load-ostium":
                center, normal, _, _ = _read_ostium_paths(case, prepared_root, centroid_path, normal_path)
                return (
                    round(float(center[0]), 6),
                    round(float(center[1]), 6),
                    round(float(center[2]), 6),
                    round(float(normal[0]), 6),
                    round(float(normal[1]), 6),
                    round(float(normal[2]), 6),
                    "Ostium files loaded",
                    "status good",
                )

            if trigger == "mesh-viewer" and click_data and mesh is not None:
                point = click_data.get("points", [{}])[0]
                idx = point.get("customdata")
                verts = np.asarray(mesh.vertices, dtype=np.float64)
                if idx is None:
                    xyz = np.array([point.get("x"), point.get("y"), point.get("z")], dtype=np.float64)
                    idx = int(np.argmin(np.linalg.norm(verts - xyz.reshape(1, 3), axis=1)))
                idx = int(idx)
                idx = max(0, min(idx, len(verts) - 1))
                center = verts[idx]
                normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
                normal = _normalize(normals[idx] if len(normals) == len(verts) else np.array([0.0, 0.0, 1.0]))
                return (
                    round(float(center[0]), 6),
                    round(float(center[1]), 6),
                    round(float(center[2]), 6),
                    round(float(normal[0]), 6),
                    round(float(normal[1]), 6),
                    round(float(normal[2]), 6),
                    f"Picked vertex {idx}",
                    "status good",
                )

            current_center = np.array(
                [_coerce_float(cx, 0.0), _coerce_float(cy, 0.0), _coerce_float(cz, 0.0)],
                dtype=np.float64,
            )
            current_normal = _normalize(
                np.array(
                    [_coerce_float(nx, 0.0), _coerce_float(ny, 0.0), _coerce_float(nz, 1.0)],
                    dtype=np.float64,
                )
            )
            if trigger == "flip-normal":
                current_normal = -current_normal
                msg = "Normal flipped"
            else:
                msg = "Normal normalized"
            return (
                round(float(current_center[0]), 6),
                round(float(current_center[1]), 6),
                round(float(current_center[2]), 6),
                round(float(current_normal[0]), 6),
                round(float(current_normal[1]), 6),
                round(float(current_normal[2]), 6),
                msg,
                "status good",
            )
        except Exception as exc:
            return (
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                str(exc),
                "status warn",
            )

    @app.callback(
        Output("preview-info", "data", allow_duplicate=True),
        Output("preview-status", "children"),
        Output("preview-status", "className"),
        Input("preview-cut", "n_clicks"),
        Input("clear-preview", "n_clicks"),
        State("session-id", "data"),
        State("mesh-info", "data"),
        State("center-x", "value"),
        State("center-y", "value"),
        State("center-z", "value"),
        State("normal-x", "value"),
        State("normal-y", "value"),
        State("normal-z", "value"),
        State("cut-radius", "value"),
        State("radius-scale", "value"),
        State("cut-slab", "value"),
        State("jagged-amp", "value"),
        State("jagged-harmonics", "value"),
        State("seed", "value"),
        prevent_initial_call=True,
    )
    def preview_cut(
        preview_clicks,
        clear_clicks,
        session_id,
        mesh_info,
        cx,
        cy,
        cz,
        nx,
        ny,
        nz,
        radius,
        radius_scale,
        slab,
        jagged_amp,
        harmonics,
        seed,
    ):
        del preview_clicks, clear_clicks
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        state = _state_for(session_id)
        if trigger == "clear-preview":
            state["preview"] = None
            return None, "Preview cleared", "status"
        if not mesh_info or state.get("mesh") is None:
            return no_update, "Load a vessel first", "status warn"
        try:
            params = _read_params(cx, cy, cz, nx, ny, nz, radius, radius_scale, slab, jagged_amp, harmonics, seed)
            preview = _compute_cut(state["mesh"], params)
            state["preview"] = preview
            removed_faces = int(np.asarray(preview["removed"], dtype=bool).sum())
            boundary_edges = int(_boundary_edges(np.asarray(preview["mesh"].faces, dtype=np.int64)).shape[0])
            info = {
                "removed_faces": removed_faces,
                "loop_vertices": int(len(preview["loop"])),
                "boundary_edges": boundary_edges,
                "params_key": repr(preview["params_key"]),
            }
            cls = "status good" if removed_faces > 0 and len(preview["loop"]) > 0 else "status warn"
            msg = (
                f"Removed {removed_faces:,} faces | "
                f"loop {len(preview['loop']):,} vertices | "
                f"boundary edges {boundary_edges:,}"
            )
            return info, msg, cls
        except Exception as exc:
            state["preview"] = None
            return None, str(exc), "status warn"

    @app.callback(
        Output("condition-info", "data", allow_duplicate=True),
        Output("condition-status", "children"),
        Output("condition-status", "className"),
        Input("preview-condition", "n_clicks"),
        Input("export-condition", "n_clicks"),
        State("session-id", "data"),
        State("mesh-info", "data"),
        State("case-input", "value"),
        State("center-x", "value"),
        State("center-y", "value"),
        State("center-z", "value"),
        State("normal-x", "value"),
        State("normal-y", "value"),
        State("normal-z", "value"),
        State("cut-radius", "value"),
        State("ostium-ecc", "value"),
        State("num-vessel-pts", "value"),
        State("vessel-radius-factor", "value"),
        State("seed", "value"),
        State("condition-space", "value"),
        State("prealign-path", "value"),
        State("ghd-checkpoint-path", "value"),
        State("canonical-mesh-path", "value"),
        State("canonical-norm-factor", "value"),
        State("out-condition-path", "value"),
        prevent_initial_call=True,
    )
    def preview_or_export_condition(
        preview_clicks,
        export_clicks,
        session_id,
        mesh_info,
        case,
        cx,
        cy,
        cz,
        nx,
        ny,
        nz,
        radius,
        ecc,
        num_points,
        radius_factor,
        seed,
        condition_space,
        prealign_path,
        ghd_checkpoint_path,
        canonical_mesh_path,
        canonical_norm_factor,
        out_condition_path,
    ):
        del preview_clicks, export_clicks
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        state = _state_for(session_id)
        mesh = state.get("mesh")
        if not mesh_info or mesh is None:
            return no_update, "Load a vessel first", "status warn"
        try:
            cparams = _read_condition_params(
                cx,
                cy,
                cz,
                nx,
                ny,
                nz,
                radius,
                ecc,
                num_points,
                radius_factor,
                seed,
            )
            condition = _build_cvae_condition(
                mesh,
                cparams,
                condition_space or "raw",
                prealign_path or "",
                ghd_checkpoint_path or "",
                canonical_mesh_path or "",
                _coerce_float(canonical_norm_factor, 1.10),
            )
            state["condition"] = condition
            info = {
                "condition_space": condition["condition_space"],
                "ostium_params": condition["ostium_params"].astype(float).tolist(),
                "vessel_pts": int(condition["vessel_pts"].shape[0]),
                "radius": float(condition["ostium_params"][6]),
                "ecc": float(condition["ostium_params"][7]),
            }
            msg = (
                f"CVAE condition ready | {info['condition_space']} | "
                f"ostium_params [8] | vessel_pts [{info['vessel_pts']}, 3]"
            )
            if trigger == "export-condition":
                out_path = (out_condition_path or "").strip()
                if not out_path:
                    out_path, _, _ = _default_output_paths(case, mesh_info.get("path", "vessel"))
                out = Path(out_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out,
                    ostium_params=condition["ostium_params"],
                    vessel_pts=condition["vessel_pts"],
                    ostium_contour=condition["ostium_contour"],
                    ostium_params_raw=condition["ostium_params_raw"],
                    vessel_pts_raw=condition["vessel_pts_raw"],
                    ostium_contour_raw=condition["ostium_contour_raw"],
                    condition_space=np.array(condition["condition_space"]),
                )
                np.save(out.parent / "ostium_params.npy", condition["ostium_params"])
                np.save(out.parent / "vessel_pts.npy", condition["vessel_pts"])
                metadata = {
                    "case": case,
                    "healthy_mesh": mesh_info.get("path"),
                    "condition_npz": str(out),
                    "condition_space": condition["condition_space"],
                    "ostium_params_layout": "centroid_xyz, normal_xyz, radius, eccentricity",
                    "ostium_params": condition["ostium_params"].astype(float).tolist(),
                    "vessel_pts_shape": list(condition["vessel_pts"].shape),
                    "raw_center": cparams["center"].astype(float).tolist(),
                    "raw_normal": cparams["normal"].astype(float).tolist(),
                    "raw_radius": float(cparams["radius"]),
                    "raw_ecc": float(cparams["ecc"]),
                    "radius_factor": float(cparams["radius_factor"]),
                    "seed": int(cparams["seed"]),
                    "prealign_transform": prealign_path or "",
                    "ghd_checkpoint": ghd_checkpoint_path or "",
                    "canonical_mesh": canonical_mesh_path or "",
                    "note": "Pass ostium_params[None] and vessel_pts[None] to VesselAwareCVAEEnsemble.sample(..., normalize_inputs=True).",
                }
                with open(out.parent / "condition_metadata.json", "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, sort_keys=True)
                msg = f"Exported CVAE condition: {out}"
            return info, msg, "status good"
        except Exception as exc:
            return None, str(exc), "status warn"

    @app.callback(
        Output("mesh-viewer", "figure"),
        Input("mesh-info", "data"),
        Input("preview-info", "data"),
        Input("condition-info", "data"),
        Input("center-x", "value"),
        Input("center-y", "value"),
        Input("center-z", "value"),
        Input("normal-x", "value"),
        Input("normal-y", "value"),
        Input("normal-z", "value"),
        Input("cut-radius", "value"),
        Input("radius-scale", "value"),
        Input("cut-slab", "value"),
        Input("jagged-amp", "value"),
        Input("jagged-harmonics", "value"),
        Input("seed", "value"),
        Input("show-removed", "value"),
        State("session-id", "data"),
    )
    def render_viewer(
        mesh_info,
        preview_info,
        condition_info,
        cx,
        cy,
        cz,
        nx,
        ny,
        nz,
        radius,
        radius_scale,
        slab,
        jagged_amp,
        harmonics,
        seed,
        show_removed,
        session_id,
    ):
        del condition_info
        try:
            state = _state_for(session_id)
            if state.get("mesh") is None:
                return _empty_figure()
            defaults = (mesh_info or {}).get("defaults", {})
            params = _read_params(
                cx if cx is not None else defaults.get("center_x", 0.0),
                cy if cy is not None else defaults.get("center_y", 0.0),
                cz if cz is not None else defaults.get("center_z", 0.0),
                nx if nx is not None else defaults.get("normal_x", 0.0),
                ny if ny is not None else defaults.get("normal_y", 0.0),
                nz if nz is not None else defaults.get("normal_z", 1.0),
                radius if radius is not None else defaults.get("radius", 1.0),
                radius_scale,
                slab if slab is not None else defaults.get("slab", 0.1),
                jagged_amp,
                harmonics,
                seed,
            )
            return _figure_for(state, mesh_info, preview_info, params, show_removed or [])
        except Exception:
            return _empty_figure(traceback.format_exc(limit=1))

    @app.callback(
        Output("stats-block", "children"),
        Input("mesh-info", "data"),
        Input("preview-info", "data"),
    )
    def update_stats(mesh_info, preview_info):
        return _stats_block(mesh_info, preview_info)

    @app.callback(
        Output("export-status", "children"),
        Output("export-status", "className"),
        Input("export-cut", "n_clicks"),
        State("session-id", "data"),
        State("mesh-info", "data"),
        State("preview-info", "data"),
        State("out-cut-path", "value"),
        State("out-report-path", "value"),
        State("center-x", "value"),
        State("center-y", "value"),
        State("center-z", "value"),
        State("normal-x", "value"),
        State("normal-y", "value"),
        State("normal-z", "value"),
        State("cut-radius", "value"),
        State("radius-scale", "value"),
        State("cut-slab", "value"),
        State("jagged-amp", "value"),
        State("jagged-harmonics", "value"),
        State("seed", "value"),
        prevent_initial_call=True,
    )
    def export_cut(
        n_clicks,
        session_id,
        mesh_info,
        preview_info,
        out_cut_path,
        out_report_path,
        cx,
        cy,
        cz,
        nx,
        ny,
        nz,
        radius,
        radius_scale,
        slab,
        jagged_amp,
        harmonics,
        seed,
    ):
        del n_clicks, preview_info
        state = _state_for(session_id)
        if not mesh_info or state.get("mesh") is None:
            return "Load a vessel first", "status warn"
        try:
            params = _read_params(cx, cy, cz, nx, ny, nz, radius, radius_scale, slab, jagged_amp, harmonics, seed)
            preview = state.get("preview")
            if preview is None or preview.get("params_key") != _params_key(params):
                preview = _compute_cut(state["mesh"], params)
                state["preview"] = preview
            out_cut_path = (out_cut_path or "").strip()
            if not out_cut_path:
                raise ValueError("Missing output OBJ path")
            Path(out_cut_path).parent.mkdir(parents=True, exist_ok=True)
            preview["mesh"].export(out_cut_path)

            report = {
                "healthy_mesh": mesh_info.get("path"),
                "out_cut_mesh": out_cut_path,
                "cut_radius": float(params["radius"]),
                "radius_scale": float(params["radius_scale"]),
                "cut_slab": float(params["slab"]),
                "jagged_amp": float(params["jagged_amp"]),
                "jagged_harmonics": int(params["harmonics"]),
                "seed": int(params["seed"]),
                "center": params["center"].tolist(),
                "normal": params["normal"].tolist(),
                "input_vertices": int(len(state["mesh"].vertices)),
                "input_faces": int(len(state["mesh"].faces)),
                "cut_vertices": int(len(preview["mesh"].vertices)),
                "cut_faces": int(len(preview["mesh"].faces)),
                "removed_faces": int(np.asarray(preview["removed"], dtype=bool).sum()),
                "loop_vertices": int(len(preview["loop"])),
                "cut_boundary_edges": int(_boundary_edges(np.asarray(preview["mesh"].faces, dtype=np.int64)).shape[0]),
                "cut_watertight": bool(preview["mesh"].is_watertight),
            }
            if out_report_path:
                Path(out_report_path).parent.mkdir(parents=True, exist_ok=True)
                with open(out_report_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2, sort_keys=True)
            return f"Exported {out_cut_path}", "status good"
        except Exception as exc:
            return str(exc), "status warn"

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a local GUI for cutting ostium holes into vessel meshes.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--case", default="")
    parser.add_argument("--healthy_mesh", default="")
    parser.add_argument("--healthy_root", default="/path/to/healthy_vessel")
    parser.add_argument("--prepared_root", default="/path/to/prepared_meshes_3")
    parser.add_argument("--ostium_centroid", default="")
    parser.add_argument("--ostium_normal", default="")
    parser.add_argument("--canonical_mesh", default="/path/to/SynVA-A1/checkpoints/canonical_average/part_aligned.obj")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(args)
    run_kwargs = {"host": args.host, "port": int(args.port), "debug": bool(args.debug)}
    if hasattr(app, "run"):
        app.run(**run_kwargs)
    else:  # pragma: no cover - old Dash compatibility
        app.run_server(**run_kwargs)


if __name__ == "__main__":
    main()
