from __future__ import annotations

"""Canonical, topology-transported tangent frames for boundary sectors.

The frame is deliberately constructed from the accepted region graph. In
particular, an eigensolver's arbitrary tangent-vector sign is never a sector
origin: it is only an optional line observation aligned to transported local
geometry.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_gaussian_covariance_frame import GaussianCovarianceFrame
from osn_gs.surface.torch_gaussian_structural_reliability import StructuralReliabilityResult
from osn_gs.surface.torch_gaussian_surface_region_formation import RegionFormationResult

POLICY_VERSION = "worklog122_canonical_transport_v2"


@dataclass(frozen=True)
class CanonicalRegionTangentFrame:
    region_id: int
    gaussian_id: Any
    oriented_normal: Any
    tangent_axis_0: Any
    tangent_axis_1: Any
    seed_id: Any
    transport_parent_id: Any | None
    axis_source: str
    anisotropy: float
    transport_residual: float
    ambiguity_reason: str | None
    policy_version: str = POLICY_VERSION


def _unit(vector: Any):
    return vector / vector.norm().clamp_min(1e-12)


def _geometric_axis(positions: Any, node: int, neighbors: Sequence[int], normal: Any):
    """Return a sign-fixed structural axis, or None for an isotropic neighborhood."""
    import torch
    if not neighbors:
        return None
    offsets = torch.stack([positions[item] - positions[node] for item in neighbors])
    offsets = offsets - (offsets @ normal).unsqueeze(1) * normal
    lengths = offsets.norm(dim=1)
    valid = lengths > 1e-8
    if not bool(valid.any()):
        return None
    directions = offsets[valid] / lengths[valid].unsqueeze(1)
    weights = lengths[valid].reciprocal()
    tensor = (directions * weights.unsqueeze(1)).transpose(0, 1) @ directions
    values, vectors = torch.linalg.eigh(tensor)
    if float(values[-1] - values[-2]) <= max(float(values[-1]), 1.0) * 1e-5:
        return None
    axis = vectors[:, -1]
    first = (directions * weights.unsqueeze(1)).sum(dim=0)
    if float(first.norm()) <= 1e-8:
        return None
    if float(axis @ first) < 0.0:
        axis = -axis
    return _unit(axis)


def construct_canonical_region_tangent_frames(positions: Any, frame: GaussianCovarianceFrame, reliability: StructuralReliabilityResult, regions: RegionFormationResult, *, ids: Sequence[Any]) -> tuple[CanonicalRegionTangentFrame | None, ...]:
    """Transport a canonical tangent reference through accepted local topology."""
    import torch
    count = len(ids)
    index = {item: i for i, item in enumerate(ids)}
    output: list[CanonicalRegionTangentFrame | None] = [None] * count
    for region in regions.regions:
        members = [index[item] for item in region.member_ids if item in index]
        if not members:
            continue
        adjacency = {node: [] for node in members}
        for left, right in region.internal_accepted_edge_ids:
            if left in index and right in index and index[left] in adjacency and index[right] in adjacency:
                adjacency[index[left]].append(index[right])
                adjacency[index[right]].append(index[left])

        def anisotropy(node: int) -> float:
            return float(frame.tangent_major_scale[node] / frame.tangent_minor_scale[node].clamp_min(1e-12))

        # Stable IDs are only the final deterministic tie-break.
        seed = max(members, key=lambda node: (
            float(reliability.planarity_score[node]), len(adjacency[node]), anisotropy(node), str(ids[node])
        ))
        seed_normal = _unit(frame.normal_candidate[seed].clone())
        geometric = _geometric_axis(positions, seed, adjacency[seed], seed_normal)
        raw_major = frame.tangent_u[seed] - seed_normal * (frame.tangent_u[seed] @ seed_normal)
        raw_major = _unit(raw_major)
        if geometric is not None:
            major, source, ambiguity = geometric, "neighborhood_geometry", None
        elif anisotropy(seed) > 1.05:
            major, source, ambiguity = raw_major, "covariance_anisotropy", "unoriented_seed_line"
        else:
            # No world-axis fallback. Consumers must not use this raw line as a
            # sector decision; support extraction below is reference invariant.
            major, source, ambiguity = raw_major, "ambiguous", "rotationally_ambiguous_seed"

        queue = [seed]
        parents = {seed: None}
        normals = {seed: seed_normal}
        tangents = {seed: major}
        residuals = {seed: 0.0}
        while queue:
            current = queue.pop(0)
            for target in sorted(adjacency[current], key=lambda node: str(ids[node])):
                target_normal = _unit(frame.normal_candidate[target].clone())
                if float(target_normal @ normals[current]) < 0.0:
                    target_normal = -target_normal
                transported = tangents[current] - target_normal * (tangents[current] @ target_normal)
                if float(transported.norm()) <= 1e-8:
                    continue
                transported = _unit(transported)
                local = _geometric_axis(positions, target, adjacency[target], target_normal)
                if local is None and anisotropy(target) > 1.05:
                    local = frame.tangent_u[target] - target_normal * (frame.tangent_u[target] @ target_normal)
                    local = _unit(local)
                if local is not None and float(local @ transported) < 0.0:
                    local = -local
                residual = 0.0 if local is None else float(1.0 - max(min(float(local @ transported), 1.0), -1.0))
                if target not in parents:
                    parents[target] = current
                    normals[target] = target_normal
                    tangents[target] = transported
                    residuals[target] = residual
                    queue.append(target)

        for node in members:
            if node not in normals:
                continue
            normal = normals[node]
            axis0 = tangents[node]
            axis1 = _unit(torch.linalg.cross(normal, axis0))
            output[node] = CanonicalRegionTangentFrame(
                region.region_id, ids[node], normal, axis0, axis1, ids[seed],
                ids[parents[node]] if parents[node] is not None else None,
                source if node == seed else "transported", anisotropy(node),
                residuals[node], ambiguity if node == seed else None,
            )
    return tuple(output)

