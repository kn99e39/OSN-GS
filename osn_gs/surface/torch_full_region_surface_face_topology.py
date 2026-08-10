from __future__ import annotations

"""Full-region observed surface faces from local covariance frames.

The input adjacency is the already-established Worklog 82 same-surface
graph.  No neighbour, threshold, normal source, or relation is changed here.
This module only supplies the requested combinatorial embedding: each actual
neighbour direction is projected into its source vertex's covariance tangent
plane, cyclically ordered by local angle, and all half-edge face orbits are
traced before any chart-unit membership is considered.
"""

import math
from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-12


@dataclass(frozen=True)
class ObservedSurfaceFace:
    face_id: int
    ordered_region_indices: tuple[int, ...]
    halfedges: tuple[tuple[int, int], ...]
    orientation_score: float


@dataclass(frozen=True)
class FullRegionSurfaceFaceTopology:
    oriented_normals: Any
    tangent_u: Any
    tangent_v: Any
    rotation_system: tuple[tuple[int, ...], ...]
    observed_faces: tuple[ObservedSurfaceFace, ...]
    face_incidence_by_edge: dict[tuple[int, int], tuple[int, ...]]
    invalid_topology_nodes: frozenset[int]


def edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _stable_key(index: int, stable_ids: Sequence[Any]) -> tuple[str, int]:
    return str(stable_ids[index]), index


def _orient_covariance_frames(
    normals: Any,
    tangent_u: Any,
    tangent_v_fallback: Any,
    adjacency: Sequence[set[int] | frozenset[int]],
    stable_ids: Sequence[Any],
) -> tuple[Any, Any, Any, frozenset[int]]:
    """Choose consistent +/- eigenvector signs per same-surface component."""

    torch = require_torch()
    count = int(normals.shape[0])
    signs = [0] * count
    invalid: set[int] = set()
    for seed in sorted(range(count), key=lambda index: _stable_key(index, stable_ids)):
        if signs[seed] != 0:
            continue
        signs[seed] = 1
        stack = [seed]
        component: set[int] = set()
        conflict = False
        while stack:
            node = stack.pop()
            component.add(node)
            for neighbor in sorted(adjacency[node], key=lambda index: _stable_key(index, stable_ids)):
                relation = 1 if float(normals[node] @ normals[neighbor]) >= 0.0 else -1
                desired = signs[node] * relation
                if signs[neighbor] == 0:
                    signs[neighbor] = desired
                    stack.append(neighbor)
                elif signs[neighbor] != desired:
                    conflict = True
        if conflict:
            invalid.update(component)

    sign_tensor = torch.tensor(signs, dtype=normals.dtype, device=normals.device)[:, None]
    oriented_normals = torch.nn.functional.normalize(normals * sign_tensor, dim=1, eps=_EPS)
    local_u = tangent_u - (tangent_u * oriented_normals).sum(dim=1, keepdim=True) * oriented_normals
    fallback = tangent_v_fallback - (tangent_v_fallback * oriented_normals).sum(dim=1, keepdim=True) * oriented_normals
    use_fallback = local_u.norm(dim=1) <= _EPS
    local_u = torch.where(use_fallback[:, None], fallback, local_u)
    degenerate = local_u.norm(dim=1) <= _EPS
    invalid.update(torch.nonzero(degenerate, as_tuple=False).reshape(-1).tolist())
    local_u = torch.nn.functional.normalize(local_u, dim=1, eps=_EPS)
    local_v = torch.nn.functional.normalize(torch.cross(oriented_normals, local_u, dim=1), dim=1, eps=_EPS)
    return oriented_normals, local_u, local_v, frozenset(invalid)


def _build_local_rotation_system(
    positions: Any,
    tangent_u: Any,
    tangent_v: Any,
    adjacency: Sequence[set[int] | frozenset[int]],
    stable_ids: Sequence[Any],
) -> tuple[tuple[tuple[int, ...], ...], frozenset[int]]:
    """Order actual same-surface neighbours by vertex-local tangent angle."""

    rotation: list[tuple[int, ...]] = []
    invalid: set[int] = set()
    for node, neighbors in enumerate(adjacency):
        ranked: list[tuple[float, float, str, int]] = []
        for neighbor in neighbors:
            delta = positions[neighbor] - positions[node]
            x = float(delta @ tangent_u[node])
            y = float(delta @ tangent_v[node])
            radius = math.hypot(x, y)
            if radius <= _EPS:
                invalid.update((node, neighbor))
                continue
            ranked.append((math.atan2(y, x), radius, str(stable_ids[neighbor]), neighbor))
        rotation.append(tuple(row[-1] for row in sorted(ranked)))
    return tuple(rotation), frozenset(invalid)


def loop_orientation_score(
    indices: Sequence[int], positions: Any, oriented_normals: Any,
) -> float:
    if len(indices) < 3:
        return 0.0
    torch = require_torch()
    score = 0.0
    for offset, current in enumerate(indices):
        previous = indices[(offset - 1) % len(indices)]
        following = indices[(offset + 1) % len(indices)]
        to_next = positions[following] - positions[current]
        to_previous = positions[previous] - positions[current]
        score += float(oriented_normals[current] @ torch.cross(to_next, to_previous, dim=0))
    return score


def _recover_observed_faces(
    positions: Any,
    oriented_normals: Any,
    adjacency: Sequence[set[int] | frozenset[int]],
    rotation_system: Sequence[Sequence[int]],
    stable_ids: Sequence[Any],
) -> tuple[tuple[ObservedSurfaceFace, ...], dict[tuple[int, int], tuple[int, ...]], frozenset[int]]:
    """Trace every full-graph half-edge orbit; keep observed-side faces."""

    all_halfedges = {(a, b) for a, neighbors in enumerate(adjacency) for b in neighbors}
    unvisited = set(all_halfedges)
    faces: list[ObservedSurfaceFace] = []
    invalid: set[int] = set()
    budget = len(all_halfedges) + 1
    while unvisited:
        start = min(
            unvisited,
            key=lambda edge: (_stable_key(edge[0], stable_ids), _stable_key(edge[1], stable_ids)),
        )
        current = start
        orbit: list[tuple[int, int]] = []
        local_seen: set[tuple[int, int]] = set()
        closed = False
        for _ in range(budget):
            if current in local_seen:
                closed = current == start
                break
            if current not in all_halfedges:
                break
            local_seen.add(current)
            orbit.append(current)
            a, b = current
            around = rotation_system[b]
            if a not in around or len(around) < 2:
                break
            current = (b, around[(around.index(a) - 1) % len(around)])
        unvisited.difference_update(local_seen)
        vertices = tuple(edge[0] for edge in orbit)
        if not closed or len(vertices) < 3 or len(set(vertices)) != len(vertices):
            invalid.update(vertices)
            continue
        score = loop_orientation_score(vertices, positions, oriented_normals)
        if abs(score) <= _EPS:
            invalid.update(vertices)
            continue
        if score < 0.0:
            continue
        faces.append(ObservedSurfaceFace(len(faces), vertices, tuple(orbit), score))

    incidence: dict[tuple[int, int], list[int]] = {}
    for face in faces:
        for a, b in face.halfedges:
            incidence.setdefault(edge_key(a, b), []).append(face.face_id)
    for edge, face_ids in incidence.items():
        if len(face_ids) > 2:
            invalid.update(edge)
    return tuple(faces), {edge: tuple(ids) for edge, ids in incidence.items()}, frozenset(invalid)


def build_full_region_surface_face_topology(
    positions: Any,
    covariance: Any,
    stable_ids: Sequence[Any],
    adjacency: Sequence[set[int] | frozenset[int]],
) -> FullRegionSurfaceFaceTopology:
    frame = extract_covariance_frame(covariance)
    oriented_normals, tangent_u, tangent_v, orientation_invalid = _orient_covariance_frames(
        frame.normal_candidate, frame.tangent_u, frame.tangent_v, adjacency, stable_ids,
    )
    rotation, rotation_invalid = _build_local_rotation_system(
        positions, tangent_u, tangent_v, adjacency, stable_ids,
    )
    faces, incidence, face_invalid = _recover_observed_faces(
        positions, oriented_normals, adjacency, rotation, stable_ids,
    )
    return FullRegionSurfaceFaceTopology(
        oriented_normals=oriented_normals,
        tangent_u=tangent_u,
        tangent_v=tangent_v,
        rotation_system=rotation,
        observed_faces=faces,
        face_incidence_by_edge=incidence,
        invalid_topology_nodes=frozenset(
            set(orientation_invalid) | set(rotation_invalid) | set(face_invalid)
        ),
    )
