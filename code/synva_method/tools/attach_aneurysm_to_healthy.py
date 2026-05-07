#!/usr/bin/env python
"""Prototype: cut a jagged ostium hole into a healthy vessel and stitch an aneurysm mesh to it."""
import argparse
import io
import json
import os
import pickle
from collections import defaultdict, deque

import numpy as np
import torch
import trimesh


class _TorchCPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def _load_clean_mesh(path, merge=False):
    mesh = trimesh.load(path, process=False)
    if merge:
        mesh.merge_vertices(digits_vertex=8, merge_tex=True, merge_norm=True)
        if hasattr(mesh, "unique_faces"):
            mesh.update_faces(mesh.unique_faces())
        if hasattr(mesh, "nondegenerate_faces"):
            mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    return mesh


def _normalize(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v / (np.linalg.norm(v) + 1e-12)


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _axis_angle_to_matrix(axis_angle):
    aa = _to_numpy(axis_angle).reshape(3).astype(np.float64)
    angle = float(np.linalg.norm(aa))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = aa / angle
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    return np.array([
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
    ], dtype=np.float64)


def _load_pickle(path):
    with open(path, "rb") as f:
        return _TorchCPUUnpickler(f).load()


def _apply_h(points, transform):
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def _apply_h_inverse(points, transform):
    points = np.asarray(points, dtype=np.float64)
    inv = np.linalg.inv(transform)
    return _apply_h(points, inv)


def _canonical_norm(canonical_mesh, factor):
    mesh = trimesh.load(canonical_mesh, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return float(np.linalg.norm(np.asarray(mesh.vertices), axis=1).max() * factor)


def _plane_basis(normal):
    normal = _normalize(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _normalize(np.cross(normal, helper))
    v = _normalize(np.cross(normal, u))
    return u, v


def _project(points, center, normal):
    u, v = _plane_basis(normal)
    rel = np.asarray(points, dtype=np.float64) - center.reshape(1, 3)
    return rel @ u, rel @ v, rel @ normal.reshape(3)


def _jagged_radius(theta, base_radius, amplitude, harmonics, seed):
    if amplitude <= 0.0:
        return np.full_like(theta, base_radius, dtype=np.float64)
    rng = np.random.default_rng(seed)
    signal = np.zeros_like(theta, dtype=np.float64)
    for k in range(2, int(harmonics) + 2):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        weight = rng.uniform(0.35, 1.0) / k
        signal += weight * np.sin(k * theta + phase)
    signal /= np.max(np.abs(signal)) + 1e-12
    radius = base_radius * (1.0 + amplitude * signal)
    return np.maximum(radius, base_radius * 0.35)


def _boundary_edges(faces):
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ])
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1]


def _boundary_components(edges):
    adj = defaultdict(list)
    for a, b in edges:
        a = int(a)
        b = int(b)
        adj[a].append(b)
        adj[b].append(a)
    components = []
    seen = set()
    for start in adj:
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp = []
        while q:
            node = q.popleft()
            comp.append(node)
            for nb in adj[node]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        components.append(np.asarray(comp, dtype=np.int64))
    return components


def _select_loop(mesh, center, min_vertices=16):
    edges = _boundary_edges(np.asarray(mesh.faces, dtype=np.int64))
    if len(edges) == 0:
        raise ValueError("mesh has no boundary after cutting")
    comps = [c for c in _boundary_components(edges) if len(c) >= min_vertices]
    if not comps:
        raise ValueError("no sufficiently large boundary component found")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    comps.sort(key=lambda c: np.linalg.norm(verts[c].mean(axis=0) - center))
    return comps[0], edges


def _order_loop_by_edges(indices, boundary_edges):
    index_set = set(int(i) for i in indices)
    adj = defaultdict(list)
    for a, b in boundary_edges:
        a = int(a)
        b = int(b)
        if a in index_set and b in index_set:
            adj[a].append(b)
            adj[b].append(a)
    if not adj or any(len(v) != 2 for v in adj.values()):
        return None

    start = min(adj)
    ordered = [start]
    prev = None
    cur = start
    for _ in range(len(adj) + 1):
        nxt_candidates = [n for n in adj[cur] if n != prev]
        if not nxt_candidates:
            return None
        nxt = nxt_candidates[0]
        if nxt == start:
            return np.asarray(ordered, dtype=np.int64)
        ordered.append(nxt)
        prev, cur = cur, nxt
    return None


def _order_loop_by_angle(vertices, indices, center, normal):
    pts = vertices[indices]
    x, y, _ = _project(pts, center, normal)
    angles = np.arctan2(y, x)
    order = np.argsort(angles)
    return indices[order]


def _signed_area_xy(points, center, normal):
    x, y, _ = _project(points, center, normal)
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ensure_same_orientation(vertices, loop_a, loop_b, center, normal):
    area_a = _signed_area_xy(vertices[loop_a], center, normal)
    area_b = _signed_area_xy(vertices[loop_b], center, normal)
    if area_a * area_b < 0:
        return loop_b[::-1]
    return loop_b


def _rotate_align_loop(vertices, loop_a, loop_b):
    """Cyclically rotate loop_b so its first vertex is closest (in 3D) to
    loop_a[0]. This removes arbitrary loop start offsets that otherwise cause
    the bridge zipper to produce long fan triangles spanning across the hole."""
    pa0 = vertices[loop_a[0]]
    pts_b = vertices[loop_b]
    d = np.linalg.norm(pts_b - pa0.reshape(1, 3), axis=1)
    k = int(np.argmin(d))
    if k == 0:
        return loop_b
    return np.concatenate([loop_b[k:], loop_b[:k]], axis=0)


def _densify_boundary_loop(verts, faces, loop_ordered, target_count):
    """Iteratively subdivide the longest edge of an ordered boundary loop until
    `len(loop) >= target_count`. Each subdivision is a proper edge-split:

      1. Find the longest loop edge a-b (boundary edge → exactly 1 incident face).
      2. Add midpoint vertex m on a-b.
      3. Replace face (a,b,c) with (a,m,c) and (m,b,c), preserving winding.
      4. Insert m into the loop between a and b.

    Result: no T-junctions, fully manifold mesh, denser loop with edges that are
    monotonically shorter than (or equal to) the original ones."""
    verts = np.asarray(verts, dtype=np.float64).copy()
    faces = np.asarray(faces, dtype=np.int64).copy()
    loop = [int(x) for x in loop_ordered]
    n_target = int(max(target_count, len(loop)))
    safety = 8 * n_target  # avoid pathological loops
    while len(loop) < n_target and safety > 0:
        safety -= 1
        # longest loop edge
        L = len(loop)
        best_i = 0
        best_d = -1.0
        for i in range(L):
            a = loop[i]; b = loop[(i + 1) % L]
            d = float(np.linalg.norm(verts[a] - verts[b]))
            if d > best_d:
                best_d = d
                best_i = i
        a = loop[best_i]; b = loop[(best_i + 1) % L]
        # boundary edge → unique incident face
        mask = ((faces == a).any(axis=1)) & ((faces == b).any(axis=1))
        face_idxs = np.where(mask)[0]
        if len(face_idxs) == 0:
            break  # shouldn't happen for a valid boundary loop
        fi = int(face_idxs[0])
        f = faces[fi].tolist()
        c = next((v for v in f if v != a and v != b), None)
        if c is None:
            break
        # midpoint
        m_idx = len(verts)
        verts = np.vstack([verts, 0.5 * (verts[a] + verts[b])])
        # winding-preserving split: find rotation of f s.t. f[r]→f[r+1] is a→b
        new_f1 = new_f2 = None
        for r in range(3):
            if f[r] == a and f[(r + 1) % 3] == b:
                # cycle is (a, b, c)
                new_f1 = [a, m_idx, c]
                new_f2 = [m_idx, b, c]
                break
            if f[r] == b and f[(r + 1) % 3] == a:
                # cycle is (b, a, c) → opposite winding
                new_f1 = [b, m_idx, c]
                new_f2 = [m_idx, a, c]
                break
        if new_f1 is None:
            break
        faces[fi] = new_f1
        faces = np.vstack([faces, new_f2])
        # insert m between a and b in loop
        loop = loop[:best_i + 1] + [m_idx] + loop[best_i + 1:]
    return verts, faces, np.asarray(loop, dtype=np.int64)


def _falloff_displace(vertices, faces, anchor_indices, anchor_targets,
                      band_rings=4, sigma_rings=2.0):
    """Move `anchor_indices` to `anchor_targets` and propagate the per-anchor
    displacement to graph-neighbours using exp(-(d/sigma)^2) falloff,
    where d is BFS distance (ring count) from the anchor.

    - anchor vertices receive full displacement (weight 1).
    - vertices not within `band_rings` BFS-rings of any anchor are unchanged.
    - For each non-anchor vertex within the band, the displacement is the
      gaussian-weighted average over anchors within reach.

    Returns new vertex array.
    """
    v_out = vertices.copy()
    n_v = len(vertices)
    # adjacency
    adj = [[] for _ in range(n_v)]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adj[a].append(b); adj[a].append(c)
        adj[b].append(a); adj[b].append(c)
        adj[c].append(a); adj[c].append(b)
    for i in range(n_v):
        adj[i] = list(set(adj[i]))

    anchors = np.asarray(anchor_indices, dtype=np.int64)
    targets = np.asarray(anchor_targets, dtype=np.float64)
    deltas = targets - vertices[anchors]

    # multi-source BFS: distance to nearest anchor + index of that anchor (in `anchors`)
    INF = band_rings + 1
    dist = np.full(n_v, INF, dtype=np.int32)
    src = np.full(n_v, -1, dtype=np.int64)
    q = deque()
    for k, a_idx in enumerate(anchors):
        dist[a_idx] = 0
        src[a_idx] = k
        q.append(int(a_idx))
    while q:
        u = q.popleft()
        if dist[u] >= band_rings:
            continue
        for w in adj[u]:
            if dist[w] > dist[u] + 1:
                dist[w] = dist[u] + 1
                src[w] = src[u]
                q.append(w)

    # apply: each vertex within band gets its source-anchor's delta scaled by gaussian
    sigma = max(sigma_rings, 1e-6)
    for v in range(n_v):
        if dist[v] == 0:
            v_out[v] = targets[src[v]]
        elif dist[v] <= band_rings and src[v] >= 0:
            w = float(np.exp(-(dist[v] / sigma) ** 2))
            v_out[v] = vertices[v] + w * deltas[src[v]]
    return v_out


def _laplacian_smooth_band(vertices, faces, band_indices, iters=5, lam=0.5):
    """Legacy uniform Laplacian (kept for backward-compat). Causes shrinkage at
    the rim and should be replaced by `_taubin_smooth_band` for visible-edge
    elimination at the ostium."""
    if iters <= 0 or len(band_indices) == 0:
        return vertices
    n_v = len(vertices)
    adj = [[] for _ in range(n_v)]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adj[a].append(b); adj[a].append(c)
        adj[b].append(a); adj[b].append(c)
        adj[c].append(a); adj[c].append(b)
    for i in range(n_v):
        adj[i] = list(set(adj[i]))
    v = vertices.copy()
    for _ in range(int(iters)):
        v_new = v.copy()
        for i in band_indices:
            ni = adj[int(i)]
            if not ni:
                continue
            mean = v[ni].mean(axis=0)
            v_new[int(i)] = (1.0 - lam) * v[int(i)] + lam * mean
        v = v_new
    return v


def _taubin_smooth_band(vertices, faces, band_indices, iters=10,
                         lamb=0.5, nu=0.53):
    """Local Taubin smoothing: alternates a positive uniform-Laplacian step
    (lam) with a negative one (-nu) so the result is shrink-free. Smoothing is
    restricted to `band_indices`; vertices outside the band are anchors so the
    band blends smoothly into the surrounding mesh.

    Including the rim itself in band_indices is safe (and recommended) because
    Taubin does not shrink the rim toward the centroid the way uniform
    Laplacian does.
    """
    if iters <= 0 or len(band_indices) == 0:
        return vertices
    n_v = len(vertices)
    adj = [[] for _ in range(n_v)]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adj[a].append(b); adj[a].append(c)
        adj[b].append(a); adj[b].append(c)
        adj[c].append(a); adj[c].append(b)
    for i in range(n_v):
        adj[i] = list(set(adj[i]))
    band_idx = np.asarray(list(band_indices), dtype=np.int64)
    v = vertices.copy()

    def _step(v_in, step):
        v_out = v_in.copy()
        for i in band_idx:
            ni = adj[int(i)]
            if not ni:
                continue
            mean = v_in[ni].mean(axis=0)
            v_out[int(i)] = v_in[int(i)] + step * (mean - v_in[int(i)])
        return v_out

    for _ in range(int(iters)):
        v = _step(v, float(lamb))    # shrinking step
        v = _step(v, -float(nu))     # un-shrinking step (|nu|>|lam| -> volume preserving)
    return v


def _ladder_faces(loop_a, loop_b):
    """Triangulate two same-length, index-aligned loops with two triangles per
    quad. `loop_a[i]` connects to `loop_b[i]`."""
    n = len(loop_a)
    assert n == len(loop_b), "ladder requires equal loop lengths"
    faces = np.empty((2 * n, 3), dtype=np.int64)
    for i in range(n):
        j = (i + 1) % n
        a0, a1 = int(loop_a[i]), int(loop_a[j])
        b0, b1 = int(loop_b[i]), int(loop_b[j])
        faces[2 * i]     = (a0, b0, b1)
        faces[2 * i + 1] = (a0, b1, a1)
    return faces


# ---------------------------------------------------------------------------
# Open-boundary bridge stitch (Weg B)
#
# Reference: vessel-mesh-editing-master/code/inference/run_inference_pipeline.py
# (functions resample_closed_ring, align_ordered_loop_indices_to_reference,
#  bridge_faces_between_loops, stitch_meshes_bridge).
#
# Treats both the vessel hole loop and the aneurysm rim as OPEN boundaries.
# No vertex is displaced; N intermediate rings are inserted by arc-length
# resampling + linear interpolation, then ladder-triangulated with a greedy
# bridger that handles unequal vertex counts. This eliminates the "pasted"
# bridge band that the legacy snap+fuse pipeline produced at the ostium.
# ---------------------------------------------------------------------------

def _resample_closed_ring(points, num_points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
        raise ValueError(f"Expected closed ring [N,3], got {points.shape}.")
    if num_points < 3:
        raise ValueError("num_points must be >= 3")
    diffs = np.roll(points, -1, axis=0) - points
    seg_lengths = np.linalg.norm(diffs, axis=1)
    if np.all(seg_lengths < 1e-12):
        raise ValueError("Cannot resample degenerate ring.")
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(cumulative[-1])
    samples = np.linspace(0.0, total, num=num_points, endpoint=False)
    out = np.zeros((num_points, 3), dtype=np.float64)
    for idx, s in enumerate(samples):
        seg_idx = min(np.searchsorted(cumulative, s, side="right") - 1,
                      points.shape[0] - 1)
        seg_len = seg_lengths[seg_idx]
        if seg_len <= 1e-12:
            out[idx] = points[seg_idx]
            continue
        alpha = (s - cumulative[seg_idx]) / seg_len
        out[idx] = ((1.0 - alpha) * points[seg_idx]
                    + alpha * points[(seg_idx + 1) % points.shape[0]])
    return out


def _align_ordered_loop_to_reference(candidate_indices, candidate_points, reference_points):
    """Rotate / reverse candidate loop so its vertices line up with reference order."""
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    candidate_points = np.asarray(candidate_points, dtype=np.float64)
    reference = _resample_closed_ring(
        np.asarray(reference_points, dtype=np.float64),
        candidate_points.shape[0],
    )
    best_err = None
    best_indices = candidate_indices
    for reverse, base_indices in [(False, candidate_indices),
                                  (True, candidate_indices[::-1].copy())]:
        base = candidate_points[::-1].copy() if reverse else candidate_points
        for shift in range(candidate_points.shape[0]):
            shifted = np.roll(base, -shift, axis=0)
            err = float(np.mean(np.sum((shifted - reference) ** 2, axis=1)))
            if best_err is None or err < best_err:
                best_err = err
                best_indices = np.roll(base_indices, -shift)
    return best_indices


def _bridge_faces_between_loops(loop_a, loop_b, flip=False):
    """Greedy ladder triangulation that handles unequal-length loops.

    Direct port of ref code's ``bridge_faces_between_loops``.
    """
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
            face = [int(loop_a[i % n]), int(loop_b[j % m]), int(loop_a[next_i])]
            i += 1
        else:
            face = [int(loop_a[i % n]), int(loop_b[j % m]), int(loop_b[next_j])]
            j += 1
        if flip:
            face = [face[0], face[2], face[1]]
        faces.append(face)
        if i >= n and j >= m:
            break
    return np.asarray(faces, dtype=np.int64)


def _ordered_boundary_loops_from_faces(faces):
    """Return list of ordered loop index arrays (closed loops) from faces."""
    edges = _boundary_edges(np.asarray(faces, dtype=np.int64))
    if edges.size == 0:
        return []
    adj = defaultdict(set)
    for a, b in edges.astype(np.int64):
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    unvisited = {tuple(sorted((int(a), int(b)))) for a, b in edges}
    loops = []
    while unvisited:
        start, current = next(iter(unvisited))
        previous = start
        unvisited.discard(tuple(sorted((start, current))))
        loop = [start, current]
        while True:
            cands = [nb for nb in sorted(adj.get(current, ()))
                     if nb != previous and tuple(sorted((current, nb))) in unvisited]
            if not cands:
                break
            nxt = cands[0]
            unvisited.discard(tuple(sorted((current, nxt))))
            if nxt == start:
                break
            loop.append(nxt)
            previous, current = current, nxt
        if len(loop) >= 3:
            loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def _pick_loop_at_centroid(loops, vertices, centroid):
    """Return the boundary loop whose mean position is closest to centroid."""
    if not loops:
        return None
    centroid = np.asarray(centroid, dtype=np.float64).reshape(3)
    best = None
    best_d = None
    for loop in loops:
        d = float(np.linalg.norm(vertices[loop].mean(axis=0) - centroid))
        if best_d is None or d < best_d:
            best_d = d
            best = loop
    return best


def _bfs_rings_from_seeds(faces, n_v, seed_indices, max_rings):
    """Return (dist_array_of_size_n_v, ring_lists). dist=INF if unreachable
    within max_rings. ring_lists[k] = vertices with BFS distance k (k=0..max_rings).
    """
    INF = max_rings + 1
    dist = np.full(n_v, INF, dtype=np.int32)
    adj = [[] for _ in range(n_v)]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adj[a].append(b); adj[a].append(c)
        adj[b].append(a); adj[b].append(c)
        adj[c].append(a); adj[c].append(b)
    for i in range(n_v):
        adj[i] = list(set(adj[i]))
    q = deque()
    for s in seed_indices:
        s = int(s)
        if 0 <= s < n_v:
            dist[s] = 0
            q.append(s)
    while q:
        u = q.popleft()
        if dist[u] >= max_rings:
            continue
        for v in adj[u]:
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                q.append(v)
    rings = [np.where(dist == k)[0] for k in range(max_rings + 1)]
    return dist, rings


def _rim_presmooth_aneurysm(aneurysm_v, aneurysm_f, aneurysm_rim,
                             rings=4, iters=8, lam=0.5, nu=0.53,
                             keep_rim_fixed=True):
    """Locally Taubin-smooth the first `rings` BFS-vertex-rings inside the
    aneurysm rim to remove the high-frequency "ringing" that the GHD fit
    introduces around the ostium boundary.

    Diagnosis on aneux_C0099 baseline.obj: 91/91 dihedral>120° edges live at
    distance <0.015 from the rim; sac interior is essentially flat. The
    Stage-2 ``rim_refine_*`` block in the GHD config (large opening_p_weight,
    small laplacian_weight) lets the rim wrinkle at high frequency. We undo
    that locally before stitching.

    Parameters
    ----------
    aneurysm_v, aneurysm_f : aneurysm-local vertices/faces
    aneurysm_rim : ordered indices into ``aneurysm_v`` (rim/boundary loop)
    rings : how many BFS rings inside the rim to smooth
    iters, lam, nu : Taubin params (shrink-free)
    keep_rim_fixed : if True (default) the rim vertices themselves are anchors
        — required when the open-bridge will tie its outer ring to that rim.
    """
    n_v = len(aneurysm_v)
    if rings <= 0 or iters <= 0 or n_v == 0:
        return aneurysm_v
    rim_idx = np.asarray(aneurysm_rim, dtype=np.int64)
    _, ring_lists = _bfs_rings_from_seeds(aneurysm_f, n_v, rim_idx, max_rings=rings)
    band_set = set()
    start = 1 if keep_rim_fixed else 0
    for k in range(start, rings + 1):
        band_set.update(int(i) for i in ring_lists[k])
    if not band_set:
        return aneurysm_v
    return _taubin_smooth_band(aneurysm_v, aneurysm_f, band_set,
                                iters=iters, lamb=lam, nu=nu)


def _open_boundary_bridge_stitch(base_v, base_f, vessel_loop,
                                 aneurysm_v, aneurysm_f, aneurysm_loop,
                                 normal, bridge_steps=4,
                                 bridge_smooth_iters=0,
                                 bridge_smooth_lam=0.5,
                                 bridge_smooth_nu=0.53):
    """Build combined mesh by inserting N intermediate rings between the two
    open boundary loops and ladder-triangulating, with NO vertex displacement.

    Parameters
    ----------
    base_v, base_f : vessel/healthy vertices and faces (loop is open boundary).
    vessel_loop : ordered indices into base_v of the ostium boundary loop.
    aneurysm_v, aneurysm_f : aneurysm vertices and (local) faces;
        aneurysm_loop is open boundary in aneurysm-local index space.
    normal : ostium normal (used only to enforce consistent loop orientation).
    bridge_steps : number of intermediate rings (>=0). 0 = direct bridge.
    bridge_smooth_iters : if >0, run that many Taubin iterations restricted to
        the intermediate bridge ring vertices (both anchor rings — vessel and
        aneurysm — stay pinned). Removes high-frequency ladder folds without
        touching any pre-existing geometry.
    """
    base_v = np.asarray(base_v, dtype=np.float64)
    base_f = np.asarray(base_f, dtype=np.int64)
    aneurysm_v = np.asarray(aneurysm_v, dtype=np.float64)
    aneurysm_f = np.asarray(aneurysm_f, dtype=np.int64)
    vessel_loop = np.asarray(vessel_loop, dtype=np.int64)
    aneurysm_loop = np.asarray(aneurysm_loop, dtype=np.int64)
    bridge_steps = int(max(0, bridge_steps))

    vessel_loop_points = base_v[vessel_loop]
    aneurysm_loop_points = aneurysm_v[aneurysm_loop]

    # Orient aneurysm loop so that, going around, its plane normal matches the
    # ostium normal. If not, reverse it (this controls bridge winding).
    def _ring_normal(pts):
        c = pts.mean(axis=0)
        rel = pts - c
        n_acc = np.zeros(3, dtype=np.float64)
        for k in range(rel.shape[0]):
            n_acc += np.cross(rel[k], rel[(k + 1) % rel.shape[0]])
        nn = np.linalg.norm(n_acc)
        return n_acc / nn if nn > 1e-12 else n_acc

    n_vessel = _ring_normal(vessel_loop_points)
    n_aneur = _ring_normal(aneurysm_loop_points)
    if float(n_vessel @ n_aneur) < 0.0:
        aneurysm_loop = aneurysm_loop[::-1].copy()
        aneurysm_loop_points = aneurysm_v[aneurysm_loop]

    # Align aneurysm loop start index against vessel loop (rotation + reverse).
    aneurysm_loop = _align_ordered_loop_to_reference(
        aneurysm_loop, aneurysm_loop_points, vessel_loop_points,
    )
    aneurysm_loop_points = aneurysm_v[aneurysm_loop]

    # Stack vessel + aneurysm vertices first.
    all_vertices = [base_v, aneurysm_v]
    all_faces = [base_f, aneurysm_f + len(base_v)]

    n_v = vessel_loop.shape[0]
    n_a = aneurysm_loop.shape[0]
    ring_counts = np.rint(
        np.linspace(n_v, n_a, bridge_steps + 2)
    ).astype(int)
    ring_counts[0] = n_v
    ring_counts[-1] = n_a

    ring_indices = [vessel_loop.copy()]
    vertex_offset = len(base_v) + len(aneurysm_v)
    for step_idx, count in enumerate(ring_counts[1:-1], start=1):
        t = step_idx / float(len(ring_counts) - 1)
        v_resamp = _resample_closed_ring(vessel_loop_points, int(count))
        a_resamp = _resample_closed_ring(aneurysm_loop_points, int(count))
        ring = (1.0 - t) * v_resamp + t * a_resamp
        idx = np.arange(vertex_offset, vertex_offset + int(count), dtype=np.int64)
        vertex_offset += int(count)
        all_vertices.append(ring)
        ring_indices.append(idx)
    ring_indices.append(aneurysm_loop + len(base_v))

    bridge_faces = []
    for left, right in zip(ring_indices[:-1], ring_indices[1:]):
        bridge_faces.append(_bridge_faces_between_loops(left, right))
    if bridge_faces:
        bridge_face_arr = np.vstack(bridge_faces)

        # Robust post-build orientation check: ensure bridge faces are
        # winding-consistent with both anchor surfaces. A manifold-consistent
        # shared edge appears in opposite directions in the two faces it
        # belongs to (one face has [a,b], the other has [b,a]). Count for
        # each bridge face its directed half-edges that already exist with
        # the SAME direction in the anchor surfaces (= inconsistent), vs
        # the OPPOSITE direction (= consistent). If the majority is
        # inconsistent, flip all bridge faces. The `_ring_normal`-based
        # heuristic above only covers the easy cases; this check fixes the
        # ~70% of submeshes whose ring normal sign happens to be opposite
        # to what's needed.
        anchor_directed = set()
        for face_block in (base_f, aneurysm_f + len(base_v)):
            for fa in face_block:
                anchor_directed.add((int(fa[0]), int(fa[1])))
                anchor_directed.add((int(fa[1]), int(fa[2])))
                anchor_directed.add((int(fa[2]), int(fa[0])))
        same_dir = 0
        opp_dir = 0
        for fa in bridge_face_arr:
            for a, b in ((int(fa[0]), int(fa[1])),
                         (int(fa[1]), int(fa[2])),
                         (int(fa[2]), int(fa[0]))):
                if (a, b) in anchor_directed:
                    same_dir += 1  # same direction in both faces -> inconsistent
                elif (b, a) in anchor_directed:
                    opp_dir += 1   # opposite direction -> consistent
        if same_dir > opp_dir:
            bridge_face_arr = bridge_face_arr[:, [0, 2, 1]]
        all_faces.append(bridge_face_arr)

    combined_v = np.vstack(all_vertices)
    combined_f = np.vstack(all_faces)

    # Optional Taubin pass restricted to intermediate bridge rings only.
    bridge_v_start = len(base_v) + len(aneurysm_v)
    bridge_v_stop = vertex_offset  # exclusive; intermediate rings live here
    if bridge_smooth_iters > 0 and bridge_v_stop > bridge_v_start:
        band = set(range(bridge_v_start, bridge_v_stop))
        combined_v = _taubin_smooth_band(
            combined_v, combined_f, band,
            iters=int(bridge_smooth_iters),
            lamb=float(bridge_smooth_lam),
            nu=float(bridge_smooth_nu),
        )
    return combined_v, combined_f, ring_counts.tolist(), int(sum(len(b) for b in bridge_faces))


def _topological_fuse_rims(base_v, base_f, hole_loop,
                           aneurysm_v, aneurysm_f, aneurysm_loop,
                           band_rings=4, sigma_rings=2.0,
                           smooth_iters=10, smooth_lam=0.5, smooth_nu=0.53,
                           smoother="taubin", bridge_steps=0, bridge_alpha=0.6):
    """Merge aneurysm into healthy vessel.

    Two modes (selected by `bridge_steps`):

    * `bridge_steps == 0`: hard remap. Aneurysm rim is falloff-snapped onto the
      hole, then aneurysm faces are reindexed so they reference hole vertices
      directly. Zero-thickness seam → fast but the curvature jump shows as a
      visible edge.

    * `bridge_steps >= 1`: insert N intermediate rings linearly interpolated
      between the hole rim (in `base_v`) and the (un-snapped) aneurysm rim.
      Each adjacent ring pair is stitched with a same-length ladder. The
      curvature change is spread over `bridge_steps + 1` triangle layers, which
      Taubin smoothing can then turn into a C¹-looking transition. This is the
      construction used in vessel-mesh-editing-master.

    Both modes finish with a Taubin smoothing band (BFS depth `band_rings`)
    around the seam. Requires `len(hole_loop) == len(aneurysm_loop)`.

    Returns (merged_v, merged_f).
    """
    assert len(hole_loop) == len(aneurysm_loop), \
        f"rim fuse requires equal loop lengths (got {len(hole_loop)} vs {len(aneurysm_loop)})"
    bridge_steps = max(0, int(bridge_steps))
    n_base = len(base_v)
    aneurysm_loop = np.asarray(aneurysm_loop, dtype=np.int64)
    hole_loop = np.asarray(hole_loop, dtype=np.int64)

    if bridge_steps == 0:
        # ---- legacy hard-remap mode ----
        targets = base_v[hole_loop]
        aneurysm_v_disp = _falloff_displace(
            aneurysm_v, aneurysm_f,
            anchor_indices=aneurysm_loop,
            anchor_targets=targets,
            band_rings=band_rings, sigma_rings=sigma_rings,
        )
        merged_v = np.vstack([base_v, aneurysm_v_disp])
        remap = np.arange(len(aneurysm_v_disp)) + n_base
        for i, an_idx in enumerate(aneurysm_loop):
            remap[int(an_idx)] = int(hole_loop[i])
        f_remap = remap[aneurysm_f]
        merged_f = np.vstack([base_f, f_remap])
        a_, b_, c_ = merged_f[:, 0], merged_f[:, 1], merged_f[:, 2]
        merged_f = merged_f[(a_ != b_) & (b_ != c_) & (a_ != c_)]
        seam_seeds = hole_loop.tolist()
    else:
        # ---- bridge mode (Option 3: curvature-continuity Hermite bridge) ----
        # Linear interpolation produced sharp dihedral steps between bridge
        # rings (shown by smoke test: BS=2..8 made p99 dihedral worse, not
        # better, because rings sat on a straight chord between rim points
        # while the surrounding mesh curved tangentially away). Instead we
        # build a cubic Hermite curve per rim correspondence using estimated
        # tangents from each side, then sample N intermediate ring points.
        hole_pts = base_v[hole_loop]
        rim_pts  = aneurysm_v[aneurysm_loop]   # unsnapped rim, in aneurysm coords
        # estimate per-vertex outward tangents from each side via 1-ring
        # neighbors of the rim vertices. The "tangent" is the average vector
        # from the rim vertex to its non-rim neighbors, which points into the
        # interior of that side. Negate so it points AWAY from the side
        # (i.e. toward the bridge).
        def _interior_dir(verts, faces, loop_idx):
            n = len(verts)
            adj_local = [set() for _ in range(n)]
            for f in faces:
                a_, b_, c_ = int(f[0]), int(f[1]), int(f[2])
                adj_local[a_].add(b_); adj_local[a_].add(c_)
                adj_local[b_].add(a_); adj_local[b_].add(c_)
                adj_local[c_].add(a_); adj_local[c_].add(b_)
            loop_set = set(int(x) for x in loop_idx)
            dirs = np.zeros((len(loop_idx), 3), dtype=np.float64)
            for i, vi in enumerate(loop_idx):
                vi_ = int(vi)
                interior = [w for w in adj_local[vi_] if w not in loop_set]
                if not interior:
                    interior = list(adj_local[vi_])
                if not interior:
                    continue
                vec = verts[interior].mean(axis=0) - verts[vi_]
                dirs[i] = vec
            return dirs
        vessel_in = _interior_dir(base_v, base_f, hole_loop)       # into vessel
        aneur_in  = _interior_dir(aneurysm_v, aneurysm_f, aneurysm_loop)  # into aneurysm
        # tangent at hole side should point toward bridge = AWAY from vessel
        # interior = -vessel_in. Same logic on aneurysm side.
        T0 = -vessel_in
        T1 = -aneur_in   # points away from aneurysm interior (i.e. toward vessel)
        # Scale tangents by chord length so curve magnitude is comparable to
        # the gap. alpha controls how much the tangent biases the cubic.
        chord = np.linalg.norm(rim_pts - hole_pts, axis=1, keepdims=True)
        # normalize tangent directions and scale by chord
        def _unit_scale(vecs, scale):
            n = np.linalg.norm(vecs, axis=1, keepdims=True)
            n = np.where(n < 1e-9, 1.0, n)
            return (vecs / n) * scale
        alpha = float(bridge_alpha)  # 0 = linear, 1 = full chord-length tangent push
        T0s = _unit_scale(T0, alpha * chord)
        T1s = _unit_scale(T1, alpha * chord)
        # Cubic Hermite: H(t) = (2t³-3t²+1)P0 + (t³-2t²+t)T0 + (-2t³+3t²)P1 + (t³-t²)T1
        # Tangents: H'(0)=T0, H'(1)=T1. With T1 pointing TOWARD vessel the
        # curve enters the aneurysm rim with the correct outward-bend.
        # Note: T1 points "away from aneurysm interior" = toward vessel = same
        # direction as the curve travels at t=1, i.e. positive sign on T1.
        merged_v = np.vstack([base_v, aneurysm_v])
        merged_f = np.vstack([base_f, aneurysm_f + n_base])
        n_offset = len(merged_v)
        loops_in_merged = [hole_loop.copy()]
        ring_count = len(hole_loop)
        new_ring_blocks = []
        for k in range(1, bridge_steps + 1):
            t = k / float(bridge_steps + 1)
            h00 = 2*t**3 - 3*t**2 + 1
            h10 = t**3 - 2*t**2 + t
            h01 = -2*t**3 + 3*t**2
            h11 = t**3 - t**2
            ring_pts = h00 * hole_pts + h10 * T0s + h01 * rim_pts + h11 * T1s
            new_idx = np.arange(n_offset, n_offset + ring_count, dtype=np.int64)
            n_offset += ring_count
            new_ring_blocks.append(ring_pts)
            loops_in_merged.append(new_idx)
        loops_in_merged.append(aneurysm_loop + n_base)
        if new_ring_blocks:
            merged_v = np.vstack([merged_v, np.vstack(new_ring_blocks)])
        # ladder-triangulate between consecutive rings
        bridge_faces = []
        for la, lb in zip(loops_in_merged[:-1], loops_in_merged[1:]):
            bridge_faces.append(_ladder_faces(la, lb))
        merged_f = np.vstack([merged_f] + bridge_faces)
        # seeds for the smoothing-band BFS = ALL bridge vertices (and rim
        # neighbors on both sides via band_rings)
        seam_seeds = []
        for ring in loops_in_merged:
            seam_seeds.extend(int(x) for x in ring)

    # ---- shared smoothing pass ----
    n_v = len(merged_v)
    adj = [[] for _ in range(n_v)]
    for f in merged_f:
        a_, b_, c_ = int(f[0]), int(f[1]), int(f[2])
        adj[a_].append(b_); adj[a_].append(c_)
        adj[b_].append(a_); adj[b_].append(c_)
        adj[c_].append(a_); adj[c_].append(b_)
    for i in range(n_v):
        adj[i] = list(set(adj[i]))
    INF = band_rings + 1
    dist = np.full(n_v, INF, dtype=np.int32)
    q = deque()
    for hi in seam_seeds:
        if dist[int(hi)] > 0:
            dist[int(hi)] = 0
            q.append(int(hi))
    while q:
        u = q.popleft()
        if dist[u] >= band_rings:
            continue
        for w in adj[u]:
            if dist[w] > dist[u] + 1:
                dist[w] = dist[u] + 1
                q.append(w)
    band_indices = [int(i) for i in range(n_v) if dist[i] <= band_rings]
    if smoother == "taubin":
        merged_v = _taubin_smooth_band(
            merged_v, merged_f, band_indices,
            iters=smooth_iters, lamb=smooth_lam, nu=smooth_nu,
        )
    else:
        band_indices_no_rim = [i for i in band_indices if dist[i] > 0]
        merged_v = _laplacian_smooth_band(
            merged_v, merged_f, band_indices_no_rim,
            iters=smooth_iters, lam=smooth_lam,
        )
    return merged_v, merged_f


def _bridge_loops(loop_a, loop_b, vertices=None):
    """Triangulate the band between two ordered, like-oriented boundary loops.

    If `vertices` is provided, uses a greedy zipper that, at each step, advances
    the side whose next diagonal is shorter. This avoids long fan triangles when
    the loops have very different vertex counts (e.g. healthy hole ~75 vs
    aneurysm rim ~139). Falls back to parametric advancement otherwise.
    """
    n = len(loop_a)
    m = len(loop_b)
    faces = []
    if vertices is None:
        # legacy parametric path (fallback)
        i = 0
        j = 0
        while i < n or j < m:
            next_a = (i + 1) / n if i < n else np.inf
            next_b = (j + 1) / m if j < m else np.inf
            a0 = int(loop_a[i % n])
            b0 = int(loop_b[j % m])
            if next_a <= next_b:
                a1 = int(loop_a[(i + 1) % n])
                if a1 != a0 and b0 != a0 and b0 != a1:
                    faces.append([a0, a1, b0])
                i += 1
            else:
                b1 = int(loop_b[(j + 1) % m])
                if b1 != b0 and a0 != b0 and a0 != b1:
                    faces.append([a0, b1, b0])
                j += 1
        return np.asarray(faces, dtype=np.int64)

    # Greedy shortest-diagonal zipper.
    i = 0
    j = 0
    while i < n or j < m:
        a0 = int(loop_a[i % n])
        b0 = int(loop_b[j % m])
        a1 = int(loop_a[(i + 1) % n])
        b1 = int(loop_b[(j + 1) % m])
        # If one side is exhausted, only the other can advance.
        if i >= n:
            advance_a = False
        elif j >= m:
            advance_a = True
        else:
            # Compare the two candidate diagonals: (a1,b0) vs (a0,b1). Pick the
            # advance that introduces the shorter new edge.
            d_a = float(np.linalg.norm(vertices[a1] - vertices[b0]))
            d_b = float(np.linalg.norm(vertices[a0] - vertices[b1]))
            advance_a = d_a <= d_b
        if advance_a:
            if a1 != a0 and b0 != a0 and b0 != a1:
                faces.append([a0, a1, b0])
            i += 1
        else:
            if b1 != b0 and a0 != b0 and a0 != b1:
                faces.append([a0, b1, b0])
            j += 1
    return np.asarray(faces, dtype=np.int64)


def _rim_from_labels(mesh, labels_path, rim_label):
    labels = np.load(labels_path)
    if len(labels) != len(mesh.vertices):
        raise ValueError(
            f"label count {len(labels)} does not match aneurysm vertices {len(mesh.vertices)}"
        )
    rim = np.where(labels == rim_label)[0]
    if len(rim) < 8:
        raise ValueError(f"found only {len(rim)} rim vertices for label {rim_label}")
    return rim.astype(np.int64)


def _rim_from_boundary(mesh, center):
    loop, _ = _select_loop(mesh, center)
    return loop


def _order_mesh_rim(mesh, rim_indices, center, normal, vertices=None):
    edges = _boundary_edges(np.asarray(mesh.faces, dtype=np.int64))
    ordered = _order_loop_by_edges(rim_indices, edges)
    if ordered is not None:
        return ordered
    if vertices is None:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return _order_loop_by_angle(vertices, rim_indices, center, normal)


def _cut_healthy(healthy, center, normal, radius, radius_scale, slab, jagged_amp, jagged_harmonics, seed):
    verts = np.asarray(healthy.vertices, dtype=np.float64)
    faces = np.asarray(healthy.faces, dtype=np.int64)
    face_centers = verts[faces].mean(axis=1)
    x, y, d = _project(face_centers, center, normal)
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    cut_radius = _jagged_radius(theta, radius * radius_scale, jagged_amp, jagged_harmonics, seed)
    remove = (r <= cut_radius) & (np.abs(d) <= slab)
    cut = trimesh.Trimesh(vertices=verts.copy(), faces=faces[~remove].copy(), process=False)
    cut.remove_unreferenced_vertices()
    return cut, remove


def _snap_aneurysm_to_hole(aneurysm_vertices, aneurysm_rim, hole_vertices, scale=False):
    verts = aneurysm_vertices.copy()
    src = verts[aneurysm_rim]
    dst = hole_vertices
    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)
    if scale:
        src_r = np.linalg.norm(src - src_center, axis=1).mean()
        dst_r = np.linalg.norm(dst - dst_center, axis=1).mean()
        factor = dst_r / (src_r + 1e-12)
        verts = (verts - src_center.reshape(1, 3)) * factor + src_center.reshape(1, 3)
        src_center = verts[aneurysm_rim].mean(axis=0)
    verts = verts + (dst_center - src_center).reshape(1, 3)
    return verts


def _procrustes_rigid_align(aneurysm_vertices, aneurysm_rim, hole_vertices, allow_scale=False):
    """Rigid (or rigid+scale) Procrustes alignment of the aneurysm so that its rim
    matches the healthy hole loop as closely as possible.

    Correspondence is built by matching each rim vertex to the nearest hole vertex
    (loops have different sizes, so we use nearest-neighbour pairing).
    Returns transformed verts.
    """
    verts = aneurysm_vertices.copy()
    src = verts[aneurysm_rim].astype(np.float64)
    dst = hole_vertices.astype(np.float64)
    # nearest-neighbour pairs in both directions then average the two transforms by
    # using src->dst correspondences (smaller / sparser side) for stability.
    # src ~= 139, dst ~= 75 (typical). Pair each src to its nearest dst.
    diff = src[:, None, :] - dst[None, :, :]
    nn_idx = np.argmin(np.linalg.norm(diff, axis=-1), axis=1)
    paired_dst = dst[nn_idx]

    src_c = src.mean(axis=0)
    dst_c = paired_dst.mean(axis=0)
    src_centered = src - src_c
    dst_centered = paired_dst - dst_c
    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    if allow_scale:
        var_src = float((src_centered ** 2).sum())
        s = float(np.trace(np.diag(S) @ D)) / max(var_src, 1e-12)
    else:
        s = 1.0
    t = dst_c - s * (R @ src_c)
    return (s * (verts @ R.T)) + t


def _aneurysm_shape_metrics(vertices, rim_indices, center, normal):
    vertices = np.asarray(vertices, dtype=np.float64)
    rim_center = vertices[rim_indices].mean(axis=0)
    rel = vertices - rim_center.reshape(1, 3)
    axial = rel @ normal.reshape(3)
    radial_vec = rel - axial.reshape(-1, 1) * normal.reshape(1, 3)
    radial = np.linalg.norm(radial_vec, axis=1)
    rim_radial = radial[rim_indices]
    return {
        "rim_radius_mean": float(rim_radial.mean()),
        "rim_radius_std": float(rim_radial.std()),
        "sac_axial_min": float(axial.min()),
        "sac_axial_max": float(axial.max()),
        "sac_radial_max": float(radial.max()),
    }


def _scale_aneurysm_shape(vertices, rim_indices, normal, uniform_scale, sac_radial_scale, sac_axial_scale, falloff_power):
    verts = np.asarray(vertices, dtype=np.float64).copy()
    if (
        abs(uniform_scale - 1.0) < 1e-12
        and abs(sac_radial_scale - 1.0) < 1e-12
        and abs(sac_axial_scale - 1.0) < 1e-12
    ):
        return verts

    normal = normal.reshape(3)
    rim_center = verts[rim_indices].mean(axis=0)
    rel = verts - rim_center.reshape(1, 3)
    rel = rel * float(uniform_scale)

    axial = rel @ normal
    radial_vec = rel - axial.reshape(-1, 1) * normal.reshape(1, 3)
    axial_vec = axial.reshape(-1, 1) * normal.reshape(1, 3)

    max_abs_axial = float(np.max(np.abs(axial))) + 1e-12
    falloff = np.clip(np.abs(axial) / max_abs_axial, 0.0, 1.0)
    falloff = falloff ** max(float(falloff_power), 1e-6)
    radial_factor = 1.0 + (float(sac_radial_scale) - 1.0) * falloff

    scaled_rel = radial_vec * radial_factor.reshape(-1, 1) + axial_vec * float(sac_axial_scale)
    return rim_center.reshape(1, 3) + scaled_rel


def _default_paths(args):
    case = args.case
    healthy = args.healthy_mesh
    aneurysm = args.aneurysm_mesh
    labels = args.aneurysm_labels
    centroid = args.ostium_centroid
    normal = args.ostium_normal
    prealign = args.prealign_transform
    ghd_checkpoint = args.ghd_checkpoint
    if case:
        if healthy is None:
            healthy = os.path.join(
                args.healthy_root,
                f"{case}_vessel_submesh_closed",
                f"{case}_vessel_submesh_closed.obj",
            )
        if aneurysm is None:
            aneurysm = os.path.join(args.prepared_root, case, "05_submeshes", "aneurysm_submesh.obj")
        if labels is None and args.aneurysm_mesh is None:
            labels = os.path.join(args.prepared_root, case, "06_submesh_labels", "labels_aneurysm.npy")
        if centroid is None:
            centroid = os.path.join(args.prepared_root, case, "07_other", "centroid_ostium.npy")
        if normal is None:
            normal = os.path.join(args.prepared_root, case, "07_other", "normal_vector.npy")
        if prealign is None:
            prealign = os.path.join(args.aligned_data_root, case, "prealign_transform.npy")
        if ghd_checkpoint is None:
            ghd_checkpoint = os.path.join(
                args.ghd_chk_root,
                case,
                args.ghd_run,
                args.ghd_chk_name,
            )
    return healthy, aneurysm, labels, centroid, normal, prealign, ghd_checkpoint


def _transform_aneurysm_vertices_to_raw(vertices, args, prealign_path, ghd_checkpoint_path):
    vertices = np.asarray(vertices, dtype=np.float64)
    if args.aneurysm_space == "raw":
        return vertices

    if not prealign_path or not os.path.exists(prealign_path):
        raise ValueError("--aneurysm_space requires --prealign_transform or --case")
    prealign = np.load(prealign_path).astype(np.float64)

    if args.aneurysm_space == "aligned":
        return _apply_h_inverse(vertices, prealign)

    if args.aneurysm_space == "ghd_local":
        if not ghd_checkpoint_path or not os.path.exists(ghd_checkpoint_path):
            raise ValueError("--aneurysm_space ghd_local requires --ghd_checkpoint or --case")
        chk = _load_pickle(ghd_checkpoint_path)
        r_mat = _axis_angle_to_matrix(chk["R"])
        s = float(np.abs(_to_numpy(chk["s"]).reshape(-1)[0])) + 1e-12
        t = _to_numpy(chk["T"]).reshape(-1, 3)[0].astype(np.float64)
        canonical_norm = _canonical_norm(args.canonical_mesh, args.canonical_norm_factor)

        target_norm = (vertices @ r_mat.T) * s + t.reshape(1, 3)
        target_aligned = target_norm * canonical_norm
        return _apply_h_inverse(target_aligned, prealign)

    raise ValueError(f"unknown aneurysm_space: {args.aneurysm_space}")


def parse_args():
    p = argparse.ArgumentParser("attach_aneurysm_to_healthy")
    p.add_argument("--case", default=None)
    p.add_argument("--prepared_root", default="/path/to/prepared_meshes_3")
    p.add_argument("--healthy_root", default="/path/to/healthy_vessel")
    p.add_argument("--aligned_data_root", default="/path/to/ghd_prepared_meshes_3_aneurysm_1op_new")
    p.add_argument("--ghd_chk_root", default="/path/to/SynVA-A1/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999")
    p.add_argument("--ghd_run", default="prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3")
    p.add_argument("--ghd_chk_name", default="ghb_fitting_checkpoint.pkl")
    p.add_argument("--canonical_mesh", default="/path/to/SynVA-A1/checkpoints/canonical_average/part_aligned.obj")
    p.add_argument("--canonical_norm_factor", type=float, default=1.10)
    p.add_argument("--healthy_mesh", default=None)
    p.add_argument("--aneurysm_mesh", default=None)
    p.add_argument("--aneurysm_space", choices=["raw", "aligned", "ghd_local"], default="raw")
    p.add_argument("--aneurysm_labels", default=None)
    p.add_argument("--rim_label", type=int, default=2)
    p.add_argument("--ostium_centroid", default=None)
    p.add_argument("--ostium_normal", default=None)
    p.add_argument("--prealign_transform", default=None)
    p.add_argument("--ghd_checkpoint", default=None)
    p.add_argument("--cut_radius", type=float, default=None)
    p.add_argument("--radius_scale", type=float, default=1.10)
    p.add_argument("--cut_slab", type=float, default=0.06)
    p.add_argument("--jagged_amp", type=float, default=0.12)
    p.add_argument("--jagged_harmonics", type=int, default=7)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--aneurysm_scale", type=float, default=1.0)
    p.add_argument("--sac_radial_scale", type=float, default=1.0)
    p.add_argument("--sac_axial_scale", type=float, default=1.0)
    p.add_argument("--sac_falloff_power", type=float, default=1.0)
    p.add_argument("--snap_rim", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--scale_rim", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--procrustes_align", action=argparse.BooleanOptionalAction, default=True,
                   help="Rigid Procrustes alignment of aneurysm rim to vessel hole loop "
                        "(translation + rotation; with --procrustes_scale also uniform scale).")
    p.add_argument("--procrustes_scale", action="store_true",
                   help="Allow uniform scale in Procrustes alignment.")
    p.add_argument("--use_vessel_submesh", action="store_true",
                   help="Use prepared_root/<case>/05_submeshes/vessel_submesh.obj "
                        "(already has the ostium hole) and skip cutting.")
    p.add_argument("--remesh_loops", action=argparse.BooleanOptionalAction, default=True,
                   help="Densify the sparser of the two boundary loops by mid-edge "
                        "splits + face subdivision so both loops have similar vertex "
                        "counts before bridging. Removes long fan triangles in the seam.")
    p.add_argument("--remesh_target_count", type=int, default=0,
                   help="Target vertex count for both boundary loops after densification. "
                        "0 = max(len(hole), len(rim)).")
    p.add_argument("--fuse_rims", action=argparse.BooleanOptionalAction, default=False,
                   help="Topologically merge the aneurysm rim into the healthy hole "
                        "(per-vertex remap, no bridge band). Requires loops with equal "
                        "length, achieved via --remesh_loops or --use_vessel_submesh.")
    p.add_argument("--fuse_band_rings", type=int, default=6,
                   help="Falloff propagation depth (BFS rings) when fusing rims. "
                        "Smoothing band uses this same depth on both sides of the seam.")
    p.add_argument("--fuse_sigma_rings", type=float, default=2.5,
                   help="Gaussian sigma (in BFS rings) for falloff displacement.")
    p.add_argument("--fuse_smoother", choices=["taubin", "laplacian"], default="taubin",
                   help="Smoothing method for the fused band. Taubin (default) is "
                        "shrink-free and eliminates the visible seam at the rim.")
    p.add_argument("--fuse_smooth_iters", type=int, default=10,
                   help="Smoothing iterations on the fused band.")
    p.add_argument("--fuse_smooth_lam", type=float, default=0.5,
                   help="Smoothing lambda (Taubin positive step / Laplacian step).")
    p.add_argument("--fuse_smooth_nu", type=float, default=0.53,
                   help="Taubin negative-step parameter (only used by Taubin smoother).")
    p.add_argument("--fuse_bridge_steps", type=int, default=0,
                   help="Insert N intermediate rings between hole and aneurysm rim "
                        "(cubic Hermite interpolation along estimated rim tangents, "
                        "ladder triangulation). 0 = legacy hard remap. "
                        "Recommended 3-5 for visibly seamless ostium.")
    p.add_argument("--fuse_bridge_alpha", type=float, default=0.6,
                   help="Tangent magnitude for Hermite bridge (0=linear, 1=full chord-length).")
    p.add_argument("--open_bridge", action=argparse.BooleanOptionalAction, default=False,
                   help="Use open-boundary bridge stitch (Weg B): treat both vessel "
                        "ostium loop and aneurysm rim as OPEN boundaries, insert N "
                        "intermediate rings via arc-length resampling + linear interp, "
                        "and ladder-triangulate. NO vertex displacement, NO falloff snap. "
                        "Implies --no-snap_rim, --no-remesh_loops, --no-fuse_rims. "
                        "Mirrors vessel-mesh-editing reference pipeline.")
    p.add_argument("--open_bridge_steps", type=int, default=4,
                   help="Number of intermediate rings inserted between vessel and "
                        "aneurysm boundary loops in --open_bridge mode (>=0).")
    p.add_argument("--rim_presmooth", action=argparse.BooleanOptionalAction, default=False,
                   help="Local Taubin smoothing of the first N BFS-rings inside "
                        "the aneurysm rim (rim itself fixed) before stitching. "
                        "Targets the GHD rim-ringing artifact (high-frequency "
                        "wrinkles concentrated within ~1 cm of the ostium).")
    p.add_argument("--rim_presmooth_rings", type=int, default=4,
                   help="Number of BFS-vertex-rings inside the rim to smooth.")
    p.add_argument("--rim_presmooth_iters", type=int, default=8,
                   help="Number of Taubin iterations.")
    p.add_argument("--rim_presmooth_lam", type=float, default=0.5,
                   help="Taubin lambda (positive shrinking step).")
    p.add_argument("--rim_presmooth_nu", type=float, default=0.53,
                   help="Taubin nu (negative un-shrinking step). Must satisfy nu>lam.")
    p.add_argument("--bridge_smooth_iters", type=int, default=4,
                   help="Taubin iterations restricted to intermediate bridge "
                        "rings only (anchors pinned). 0 disables. Removes "
                        "ladder-triangle folds at the bridge.")
    p.add_argument("--bridge_smooth_lam", type=float, default=0.5)
    p.add_argument("--bridge_smooth_nu", type=float, default=0.53)
    p.add_argument("--out_mesh", required=True)
    p.add_argument("--out_cut_mesh", default=None)
    p.add_argument("--out_report", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.open_bridge:
        # open-bridge mode forbids any pre-snap/remesh/fuse manipulation.
        if args.snap_rim:
            print("[open_bridge] disabling --snap_rim (vertices are not displaced).")
            args.snap_rim = False
        if args.remesh_loops:
            print("[open_bridge] disabling --remesh_loops (loops kept native).")
            args.remesh_loops = False
        if args.fuse_rims:
            print("[open_bridge] disabling --fuse_rims (replaced by ring bridge).")
            args.fuse_rims = False
    healthy_path, aneurysm_path, labels_path, centroid_path, normal_path, prealign_path, ghd_checkpoint_path = _default_paths(args)
    if args.use_vessel_submesh:
        if not args.case:
            raise ValueError("--use_vessel_submesh requires --case")
        healthy_path = os.path.join(args.prepared_root, args.case, "05_submeshes", "vessel_submesh.obj")
    if not all([healthy_path, aneurysm_path, centroid_path, normal_path]):
        raise ValueError("provide --case or explicit mesh/ostium paths")

    center = np.load(centroid_path).astype(np.float64).reshape(3)
    normal = _normalize(np.load(normal_path).astype(np.float64).reshape(3))
    healthy = _load_clean_mesh(healthy_path, merge=True)
    # Some prepared vessel_submesh.obj files come with internally
    # winding-inconsistent face orientations (~16% of the corpus). Repair
    # them in-place via trimesh.repair.fix_winding (BFS edge-orientation
    # propagation) so the bridge orientation check downstream has a
    # consistent anchor surface to align against.
    if not bool(getattr(healthy, "is_winding_consistent", True)):
        try:
            trimesh.repair.fix_winding(healthy)
        except Exception as _err:
            print(f"[warn] fix_winding failed on healthy mesh: {_err}")
    aneurysm = _load_clean_mesh(aneurysm_path, merge=False)
    aneurysm.vertices = _transform_aneurysm_vertices_to_raw(
        np.asarray(aneurysm.vertices, dtype=np.float64),
        args,
        prealign_path,
        ghd_checkpoint_path,
    )

    if labels_path and os.path.exists(labels_path):
        aneurysm_rim = _rim_from_labels(aneurysm, labels_path, args.rim_label)
    else:
        aneurysm_rim = _rim_from_boundary(aneurysm, center)

    aneurysm_vertices = np.asarray(aneurysm.vertices, dtype=np.float64)
    shape_before = _aneurysm_shape_metrics(aneurysm_vertices, aneurysm_rim, center, normal)
    aneurysm_vertices = _scale_aneurysm_shape(
        aneurysm_vertices,
        aneurysm_rim,
        normal,
        args.aneurysm_scale,
        args.sac_radial_scale,
        args.sac_axial_scale,
        args.sac_falloff_power,
    )
    aneurysm.vertices = aneurysm_vertices
    shape_after_scale = _aneurysm_shape_metrics(aneurysm_vertices, aneurysm_rim, center, normal)

    rim_points = aneurysm_vertices[aneurysm_rim]
    x, y, _ = _project(rim_points, center, normal)
    inferred_radius = float(np.sqrt(x * x + y * y).mean())
    cut_radius = float(args.cut_radius) if args.cut_radius is not None else inferred_radius

    if args.use_vessel_submesh:
        # vessel_submesh already has the ostium as its (single) open boundary loop;
        # use it as-is and skip the cut.
        cut_healthy = healthy
        removed = np.zeros(len(healthy.faces), dtype=bool)
    else:
        cut_healthy, removed = _cut_healthy(
            healthy,
            center,
            normal,
            cut_radius,
            args.radius_scale,
            args.cut_slab,
            args.jagged_amp,
            args.jagged_harmonics,
            args.seed,
        )
    hole_loop, hole_edges = _select_loop(cut_healthy, center)
    hole_loop_ordered = _order_loop_by_edges(hole_loop, hole_edges)
    if hole_loop_ordered is None:
        hole_loop_ordered = _order_loop_by_angle(np.asarray(cut_healthy.vertices), hole_loop, center, normal)
    hole_loop = hole_loop_ordered

    if args.snap_rim:
        if args.procrustes_align:
            aneurysm_vertices = _procrustes_rigid_align(
                aneurysm_vertices,
                aneurysm_rim,
                np.asarray(cut_healthy.vertices)[hole_loop],
                allow_scale=args.procrustes_scale,
            )
        else:
            aneurysm_vertices = _snap_aneurysm_to_hole(
                aneurysm_vertices,
                aneurysm_rim,
                np.asarray(cut_healthy.vertices)[hole_loop],
                scale=args.scale_rim,
            )
    aneurysm_rim = _order_mesh_rim(aneurysm, aneurysm_rim, center, normal, vertices=aneurysm_vertices)

    base_v = np.asarray(cut_healthy.vertices, dtype=np.float64)
    base_f = np.asarray(cut_healthy.faces, dtype=np.int64)

    # Robust orientation check on the vessel side (open submesh: ostium hole
    # is small relative to the closed vessel surface, so the divergence-
    # theorem signed volume is still a reliable sign indicator). Some
    # prepared vessel_submesh.obj files come with inverted face winding
    # which then propagates an inconsistent combined mesh. Flip if needed.
    def _signed_volume(verts, faces):
        v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
        return float(np.einsum('ij,ij->i', v0, np.cross(v1, v2)).sum() / 6.0)
    vessel_vol = _signed_volume(base_v, base_f)
    if vessel_vol < 0.0:
        base_f = base_f[:, [0, 2, 1]]

    # Match aneurysm face winding to vessel.
    #
    # OLD heuristic (broken): average face normal of faces incident to the rim,
    # then flip if vessel-side and aneurysm-side averages had negative dot
    # product. Diagnostic on aneux_C0099 showed the result was essentially
    # noise — radial outward normals around a closed ring sum to ~0, so the
    # sign of the dot product is unstable. Empirically this consistently
    # flipped the aneurysm by 180° in the wrong direction (combined mesh ended
    # with `is_winding_consistent=False`, while both raw inputs were already
    # consistent and outward-oriented).
    #
    # NEW: trust the source meshes and use a robust *global* signed-volume
    # criterion. Both vessel-submesh and GHD aneurysm are produced with
    # outward-facing normals (positive divergence-theorem volume). We only
    # flip the aneurysm if its own signed volume is negative (i.e. inverted).
    # If after merging the combined mesh is still winding-inconsistent across
    # the seam, the open-bridge ladder direction is what matters — that is
    # already corrected inside `_open_boundary_bridge_stitch` via the loop
    # `_ring_normal` test, so no per-face flip is needed here.
    aneurysm_faces_local = np.asarray(aneurysm.faces, dtype=np.int64).copy()

    aneur_vol = _signed_volume(aneurysm_vertices, aneurysm_faces_local)
    flip = aneur_vol < 0.0
    if flip:
        aneurysm_faces_local = aneurysm_faces_local[:, [0, 2, 1]]

    aneurysm_offset = len(base_v)
    aneurysm_f = aneurysm_faces_local + aneurysm_offset
    aneurysm_loop = aneurysm_rim + aneurysm_offset

    remesh_info = {"hole_before": int(len(hole_loop)),
                   "rim_before": int(len(aneurysm_loop)),
                   "hole_after": int(len(hole_loop)),
                   "rim_after": int(len(aneurysm_loop))}
    if args.remesh_loops:
        target = (int(args.remesh_target_count)
                  if args.remesh_target_count > 0
                  else max(len(hole_loop), len(aneurysm_loop)))
        if len(hole_loop) < target:
            base_v, base_f, hole_loop = _densify_boundary_loop(
                base_v, base_f, hole_loop, target)
            # offsets shift only for aneurysm-side because base_v grew
            new_offset = len(base_v)
            shift = new_offset - aneurysm_offset
            aneurysm_offset = new_offset
            aneurysm_f = aneurysm_faces_local + aneurysm_offset
            aneurysm_loop = aneurysm_rim + aneurysm_offset
            remesh_info["hole_after"] = int(len(hole_loop))
        if len(aneurysm_loop) < target:
            # densify on the aneurysm-local index space, then re-offset
            an_v, an_f, an_loop_local = _densify_boundary_loop(
                aneurysm_vertices, aneurysm_faces_local, aneurysm_rim, target)
            aneurysm_vertices = an_v
            aneurysm_faces_local = an_f
            aneurysm_rim = an_loop_local
            aneurysm_f = aneurysm_faces_local + aneurysm_offset
            aneurysm_loop = aneurysm_rim + aneurysm_offset
            remesh_info["rim_after"] = int(len(aneurysm_loop))

    all_vertices = np.vstack([base_v, aneurysm_vertices])
    aneurysm_loop = _ensure_same_orientation(all_vertices, hole_loop, aneurysm_loop, center, normal)
    aneurysm_loop = _rotate_align_loop(all_vertices, hole_loop, aneurysm_loop)

    # Local rim indices in aneurysm space (before adding base offset)
    aneurysm_loop_local = aneurysm_loop - len(base_v)

    base_boundary_edges_before_attach = _boundary_edges(base_f)
    aneurysm_boundary_edges_before_attach = _boundary_edges(aneurysm_faces_local)
    selected_ostium_loop_edges = int(len(hole_loop))
    selected_aneurysm_rim_edges = int(len(aneurysm_loop_local))
    expected_remaining_boundary_edges = int(
        len(base_boundary_edges_before_attach)
        + len(aneurysm_boundary_edges_before_attach)
        - selected_ostium_loop_edges
        - selected_aneurysm_rim_edges
    )

    fuse_used = bool(args.fuse_rims and len(hole_loop) == len(aneurysm_loop_local))
    open_bridge_used = bool(args.open_bridge)
    open_bridge_info = {}
    rim_presmooth_info = {}
    if open_bridge_used and args.rim_presmooth:
        before_v = aneurysm_vertices.copy()
        aneurysm_vertices = _rim_presmooth_aneurysm(
            aneurysm_vertices, aneurysm_faces_local, aneurysm_rim,
            rings=args.rim_presmooth_rings,
            iters=args.rim_presmooth_iters,
            lam=args.rim_presmooth_lam,
            nu=args.rim_presmooth_nu,
            keep_rim_fixed=True,
        )
        max_disp = float(np.max(np.linalg.norm(aneurysm_vertices - before_v, axis=1)))
        rim_presmooth_info = {
            "rings": int(args.rim_presmooth_rings),
            "iters": int(args.rim_presmooth_iters),
            "lam": float(args.rim_presmooth_lam),
            "nu": float(args.rim_presmooth_nu),
            "max_vertex_displacement": max_disp,
        }
    if open_bridge_used:
        all_vertices, all_faces, ring_counts, n_bridge_faces = _open_boundary_bridge_stitch(
            base_v, base_f, hole_loop,
            aneurysm_vertices, aneurysm_faces_local, aneurysm_rim,
            normal=normal,
            bridge_steps=args.open_bridge_steps,
            bridge_smooth_iters=args.bridge_smooth_iters,
            bridge_smooth_lam=args.bridge_smooth_lam,
            bridge_smooth_nu=args.bridge_smooth_nu,
        )
        bridge_f = np.zeros((n_bridge_faces, 3), dtype=np.int64)  # placeholder for report counter
        open_bridge_info = {
            "bridge_steps": int(args.open_bridge_steps),
            "ring_counts": [int(c) for c in ring_counts],
            "bridge_face_count": int(n_bridge_faces),
            "bridge_smooth_iters": int(args.bridge_smooth_iters),
        }
    elif fuse_used:
        all_vertices, all_faces = _topological_fuse_rims(
            base_v, base_f, hole_loop,
            aneurysm_vertices, aneurysm_faces_local, aneurysm_loop_local,
            band_rings=args.fuse_band_rings,
            sigma_rings=args.fuse_sigma_rings,
            smooth_iters=args.fuse_smooth_iters,
            smooth_lam=args.fuse_smooth_lam,
            smooth_nu=args.fuse_smooth_nu,
            smoother=args.fuse_smoother,
            bridge_steps=args.fuse_bridge_steps,
            bridge_alpha=args.fuse_bridge_alpha,
        )
        bridge_f = np.zeros((0, 3), dtype=np.int64)  # no bridge needed
    else:
        bridge_f = _bridge_loops(hole_loop, aneurysm_loop, vertices=all_vertices)
        all_faces = np.vstack([base_f, aneurysm_f, bridge_f])

    combined = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)
    combined.remove_unreferenced_vertices()
    os.makedirs(os.path.dirname(args.out_mesh) or ".", exist_ok=True)
    combined.export(args.out_mesh)
    if args.out_cut_mesh:
        os.makedirs(os.path.dirname(args.out_cut_mesh) or ".", exist_ok=True)
        cut_healthy.export(args.out_cut_mesh)

    combined_boundary_edges = int((_boundary_edges(np.asarray(combined.faces)).shape[0]))
    seam_boundary_delta = int(combined_boundary_edges - expected_remaining_boundary_edges)
    seam_closed = bool(seam_boundary_delta == 0)

    report = {
        "case": args.case,
        "healthy_mesh": healthy_path,
        "aneurysm_mesh": aneurysm_path,
        "aneurysm_space": args.aneurysm_space,
        "out_mesh": args.out_mesh,
        "healthy_vertices": int(len(healthy.vertices)),
        "healthy_faces": int(len(healthy.faces)),
        "cut_removed_faces": int(removed.sum()),
        "hole_loop_vertices": int(len(hole_loop)),
        "aneurysm_rim_vertices": int(len(aneurysm_rim)),
        "bridge_faces": int(len(bridge_f)),
        "attach_mode": "open_bridge" if open_bridge_used else ("fuse_rims" if fuse_used else "bridge"),
        "fuse_requested": bool(args.fuse_rims),
        "fuse_used": bool(fuse_used),
        "open_bridge_requested": bool(args.open_bridge),
        "open_bridge_used": bool(open_bridge_used),
        "open_bridge_info": open_bridge_info,
        "rim_presmooth_requested": bool(args.rim_presmooth),
        "rim_presmooth_info": rim_presmooth_info,
        "fuse_smoother": args.fuse_smoother,
        "fuse_band_rings": int(args.fuse_band_rings),
        "fuse_sigma_rings": float(args.fuse_sigma_rings),
        "fuse_smooth_iters": int(args.fuse_smooth_iters),
        "fuse_smooth_lam": float(args.fuse_smooth_lam),
        "fuse_smooth_nu": float(args.fuse_smooth_nu),
        "remesh_info": remesh_info,
        "input_boundary_edges": int(len(base_boundary_edges_before_attach)),
        "input_aneurysm_boundary_edges": int(len(aneurysm_boundary_edges_before_attach)),
        "selected_ostium_loop_edges": selected_ostium_loop_edges,
        "selected_aneurysm_rim_edges": selected_aneurysm_rim_edges,
        "expected_remaining_boundary_edges": expected_remaining_boundary_edges,
        "seam_boundary_delta": seam_boundary_delta,
        "seam_closed": seam_closed,
        "aneurysm_faces_flipped": bool(flip),
        "cut_radius": cut_radius,
        "radius_scale": args.radius_scale,
        "cut_slab": args.cut_slab,
        "jagged_amp": args.jagged_amp,
        "aneurysm_scale": args.aneurysm_scale,
        "sac_radial_scale": args.sac_radial_scale,
        "sac_axial_scale": args.sac_axial_scale,
        "sac_falloff_power": args.sac_falloff_power,
        "shape_before": shape_before,
        "shape_after_scale": shape_after_scale,
        "shape_after_snap": _aneurysm_shape_metrics(aneurysm_vertices, aneurysm_rim, center, normal),
        "combined_vertices": int(len(combined.vertices)),
        "combined_faces": int(len(combined.faces)),
        "combined_watertight": bool(combined.is_watertight),
        "combined_boundary_edges": combined_boundary_edges,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out_report:
        os.makedirs(os.path.dirname(args.out_report) or ".", exist_ok=True)
        with open(args.out_report, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
