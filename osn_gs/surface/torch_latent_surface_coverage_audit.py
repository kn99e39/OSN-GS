from __future__ import annotations

"""Worklog 103 -- factual latent-surface spatial-coverage audit.

Distinguishes, explicitly (per the Worklog 103 directive section 5):

RAW REGION EVIDENCE
    original visible Gaussian-center observations owned by a region.

LATENT-SUPPORTED OBSERVATION
    a source observation for which the EXISTING Worklog 95 support query
    (:meth:`~osn_gs.surface.torch_latent_surface_support.LatentSurfaceSupport.query_batch`)
    succeeds (``query.supported``).

LATENT-PROJECTED POSITION
    the projected ``query.positions`` coordinate for a supported
    observation -- the authoritative latent-surface coordinate.

LATENT SUPPORT UNIT
    a connected component of the SAME continuously-supported kNN graph
    Worklog 98's own field builder already uses
    (:func:`~osn_gs.surface.torch_latent_surface_tangent_frame_field._knn_edges`
    + :func:`~osn_gs.surface.torch_latent_surface_curve_tracer.sample_segment_continuous_support`),
    used ONLY to organize materialization -- this module never requires
    Worklog 98 frame coherence (holonomy), Worklog 100 UV validity,
    Worklog 101 chart membership, or Worklog 102 patch identifiability to
    include a unit. This is NOT synonymous with a Worklog 101 "chart" and
    NOT synonymous with "surface area coverage" -- it is evidence/node
    coverage only.

No bridging, convex hull, bounding-box fill, or fabricated boundary is ever
used to connect otherwise-disconnected support.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_latent_surface_curve_tracer import sample_segment_continuous_support
from osn_gs.surface.torch_latent_surface_support import LatentSurfaceSupport
from osn_gs.surface.torch_latent_surface_tangent_frame_field import FIELD_NEIGHBOR_COUNT, _knn_edges
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9

# Minimal floor purely to avoid reporting single isolated points as their
# own "unit" when they're genuinely disconnected noise -- NOT a coherence,
# chart-validity, or NURBS-related gate. A single node is still counted in
# every RAW/SUPPORTED accounting figure regardless of this floor; it only
# affects how many distinct UNITS are reported.
MIN_UNIT_SIZE = 1


@dataclass(frozen=True)
class LatentSupportUnit:
    unit_id: int
    node_indices: tuple[int, ...]  # indices into the region's raw evidence tensor
    raw_positions: Any  # (M, 3)
    latent_positions: Any  # (M, 3) -- query.positions, authoritative
    projection_displacement: Any  # (M, 3)
    normals: Any  # (M, 3) -- query.normals
    edges: tuple[tuple[int, int], ...]  # LOCAL indices (0..M-1), continuously-supported


@dataclass(frozen=True)
class RegionCoverageAudit:
    region_id: int
    raw_evidence_count: int
    latent_supported_count: int
    latent_unsupported_count: int
    unsupported_raw_positions: Any  # (K, 3)
    units: tuple[LatentSupportUnit, ...]
    projection_displacement_all_supported: Any  # (S, 3) -- every supported node, not just unit members
    median_spacing: float


def _union_find(count: int, edges: Any) -> list[list[int]]:
    parent = list(range(count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_x] = root_y

    for a, b in edges.tolist():
        union(a, b)

    groups: dict[int, list[int]] = {}
    for node in range(count):
        groups.setdefault(find(node), []).append(node)
    return list(groups.values())


def audit_region_latent_coverage(
    region_id: int, raw_evidence: Any, support: LatentSurfaceSupport, k: int = FIELD_NEIGHBOR_COUNT,
) -> RegionCoverageAudit:
    """Materialize ALL supported latent-projected positions for one
    region's raw evidence, organized into connectivity-only support units.
    Never requires downstream (Worklog 98-102) acceptance."""

    torch = require_torch()
    count = int(raw_evidence.shape[0])
    query = support.query_batch(raw_evidence)
    supported_mask = query.supported
    supported_count = int(supported_mask.sum().item())
    unsupported_count = count - supported_count

    candidate_edges = _knn_edges(raw_evidence, k)
    edge_supported_mask = []
    for a, b in candidate_edges.tolist():
        _seg_points, fully_supported = sample_segment_continuous_support(support, raw_evidence[a], raw_evidence[b])
        edge_supported_mask.append(fully_supported)
    edge_supported_tensor = torch.tensor(edge_supported_mask, dtype=torch.bool)
    supported_edges = candidate_edges[edge_supported_tensor] if candidate_edges.shape[0] else candidate_edges

    groups = _union_find(count, supported_edges)

    units: list[LatentSupportUnit] = []
    for group in groups:
        # A unit is meaningful only if it actually contains at least one
        # continuously-supported edge OR is itself a single supported node
        # -- a lone UNSUPPORTED node with no edges is not a "unit" (it
        # belongs in unsupported_raw_positions instead).
        node_indices = [node for node in group if bool(supported_mask[node].item())]
        if len(node_indices) < MIN_UNIT_SIZE:
            continue
        selector = torch.tensor(sorted(node_indices), dtype=torch.long, device=raw_evidence.device)
        local_to_new = {old: new for new, old in enumerate(sorted(node_indices))}
        node_set = set(node_indices)
        local_edges = tuple(
            (local_to_new[a], local_to_new[b])
            for a, b in supported_edges.tolist() if a in node_set and b in node_set
        )
        units.append(LatentSupportUnit(
            unit_id=len(units),
            node_indices=tuple(sorted(node_indices)),
            raw_positions=raw_evidence[selector],
            latent_positions=query.positions[selector],
            projection_displacement=(query.positions[selector] - raw_evidence[selector]),
            normals=query.normals[selector],
            edges=local_edges,
        ))

    unsupported_selector = (~supported_mask).nonzero(as_tuple=True)[0]
    unsupported_positions = raw_evidence[unsupported_selector]

    supported_selector = supported_mask.nonzero(as_tuple=True)[0]
    displacement_all_supported = (
        query.positions[supported_selector] - raw_evidence[supported_selector]
    )

    distance = torch.cdist(raw_evidence, raw_evidence)
    if count >= 2:
        distance.fill_diagonal_(float("inf"))
        median_spacing = float(distance.min(dim=1).values.median().item())
    else:
        median_spacing = 1e-6

    return RegionCoverageAudit(
        region_id=region_id,
        raw_evidence_count=count,
        latent_supported_count=supported_count,
        latent_unsupported_count=unsupported_count,
        unsupported_raw_positions=unsupported_positions,
        units=tuple(units),
        projection_displacement_all_supported=displacement_all_supported,
        median_spacing=median_spacing,
    )
