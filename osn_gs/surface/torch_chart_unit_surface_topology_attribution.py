from __future__ import annotations

"""Read-only attribution for failed Worklog 89 full-region face topology.

This diagnostic deliberately does not create an edge, alter Worklog 82
relations, order a boundary, or construct a face.  It compares the fixed
center-point graph with the existing oriented covariance footprints at one
standard deviation.  The latter is a measurement convention (the covariance
itself), not a parameter sweep or proposed production admission rule.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_dense_surface_consistency_components import (
    DEFAULT_SAME_SURFACE_MIN_NORMAL_ALIGNMENT,
    DenseConsistencyEdge,
    RELATION_AMBIGUOUS,
    RELATION_CREASE_VETOED,
    RELATION_SAME_SURFACE,
)
from osn_gs.surface.torch_gaussian_covariance_frame import extract_covariance_frame
from osn_gs.utils.torch_ops import require_torch

CENTER_UNDERSAMPLING = "CENTER_UNDERSAMPLING"
RELATION_FALSE_NEGATIVE = "RELATION_FALSE_NEGATIVE"
TRUE_SUPPORT_GAP = "TRUE_SUPPORT_GAP"
MULTILAYER_OR_VOLUMETRIC = "MULTILAYER_OR_VOLUMETRIC"
GRAPH_TO_SURFACE_TOPOLOGY_MISMATCH = "GRAPH_TO_SURFACE_TOPOLOGY_MISMATCH"

ATTRIBUTION_CLASSES = (
    CENTER_UNDERSAMPLING,
    RELATION_FALSE_NEGATIVE,
    TRUE_SUPPORT_GAP,
    MULTILAYER_OR_VOLUMETRIC,
    GRAPH_TO_SURFACE_TOPOLOGY_MISMATCH,
)

# A covariance footprint is measured at its native one-standard-deviation
# ellipse/thickness.  This is not an acceptance threshold and is intentionally
# fixed: changing it would be a support-scale ablation, prohibited here.
FOOTPRINT_SIGMA = 1.0
_EPS = 1e-12


@dataclass(frozen=True)
class SurfaceTopologyAttribution:
    primary_cause: str
    cause_node_fractions: dict[str, float]
    center_spacing_over_tangent_scale_median: float | None
    compatible_footprint_overlap_coverage: float
    missing_same_surface_edge_fraction_despite_footprint: float
    relation_false_negative_fraction: float
    layer_ambiguity_fraction: float
    valid_local_surface_complex_plausible: bool
    footprint_compatible_pair_count: int
    missing_center_graph_pair_count: int
    rejected_relation_pair_count: int
    accepted_same_surface_pair_count: int
    layer_conflict_pair_count: int
    true_gap_node_count: int
    provenance_veto_pair_count: int


def _edge_relation_lookup(
    relation_edges: Sequence[DenseConsistencyEdge],
) -> dict[tuple[int, int], str]:
    return {
        (edge.a, edge.b) if edge.a < edge.b else (edge.b, edge.a): edge.relation
        for edge in relation_edges
    }


def _primary_cause(masses: dict[str, float]) -> str:
    # The tie order only gives a deterministic presentation to equal evidence
    # masses; it never changes a graph relation or topology decision.
    order = (
        GRAPH_TO_SURFACE_TOPOLOGY_MISMATCH,
        RELATION_FALSE_NEGATIVE,
        CENTER_UNDERSAMPLING,
        MULTILAYER_OR_VOLUMETRIC,
        TRUE_SUPPORT_GAP,
    )
    return max(order, key=lambda cause: (masses[cause], -order.index(cause)))


def attribute_failed_chart_unit_surface_topology(
    positions: Any,
    covariance: Any,
    member_indices: Sequence[int],
    relation_edges: Sequence[DenseConsistencyEdge],
) -> SurfaceTopologyAttribution:
    """Attribute one failed coherent unit without changing any constructor state.

    ``relation_edges`` must be the exact existing Worklog 82 bounded-kNN edge
    classification for this *full region*.  Pairs absent from it are therefore
    absent from the fixed center graph; they are not proposed additions.
    """

    torch = require_torch()
    members = tuple(dict.fromkeys(int(index) for index in member_indices))
    if not members:
        raise ValueError("member_indices must not be empty")
    selector = torch.tensor(members, dtype=torch.long, device=positions.device)
    points = positions[selector]
    frame = extract_covariance_frame(covariance[selector])
    count = len(members)
    if count == 1:
        masses = {
            CENTER_UNDERSAMPLING: 0.0,
            RELATION_FALSE_NEGATIVE: 0.0,
            TRUE_SUPPORT_GAP: 1.0,
            MULTILAYER_OR_VOLUMETRIC: 0.0,
            GRAPH_TO_SURFACE_TOPOLOGY_MISMATCH: 0.0,
        }
        return SurfaceTopologyAttribution(
            TRUE_SUPPORT_GAP, masses, None, 0.0, 0.0, 0.0, 0.0, False,
            0, 0, 0, 0, 0, 1, 0,
        )

    # Pairwise local footprint measurement only.  M is a chart unit (not the
    # full training cloud), so this cannot alter or broaden production kNN.
    delta = points[None, :, :] - points[:, None, :]
    distance = delta.norm(dim=2)
    off_diagonal = ~torch.eye(count, dtype=torch.bool, device=points.device)
    safe_distance = distance.clamp_min(_EPS)

    normal = frame.normal_candidate
    normal_alignment = (normal @ normal.T).abs()
    signed_normal_offset_a = (delta * normal[:, None, :]).sum(dim=2)
    signed_normal_offset_b = (delta * normal[None, :, :]).sum(dim=2)
    normal_offset_a = signed_normal_offset_a.abs()
    normal_offset_b = signed_normal_offset_b.abs()
    depth_a = normal_offset_a / frame.normal_thickness[:, None].clamp_min(_EPS)
    depth_b = normal_offset_b / frame.normal_thickness[None, :].clamp_min(_EPS)

    tangent_delta_a = delta - signed_normal_offset_a[..., None] * normal[:, None, :]
    tangent_delta_b = delta - signed_normal_offset_b[..., None] * normal[None, :, :]
    tangent_distance = 0.5 * (tangent_delta_a.norm(dim=2) + tangent_delta_b.norm(dim=2))
    direction_a = tangent_delta_a / tangent_delta_a.norm(dim=2, keepdim=True).clamp_min(_EPS)
    direction_b = tangent_delta_b / tangent_delta_b.norm(dim=2, keepdim=True).clamp_min(_EPS)
    reach_a = torch.sqrt(
        (direction_a * frame.tangent_u[:, None, :]).sum(dim=2).square()
        * frame.tangent_major_scale[:, None].square()
        + (direction_a * frame.tangent_v[:, None, :]).sum(dim=2).square()
        * frame.tangent_minor_scale[:, None].square()
    )
    reach_b = torch.sqrt(
        (direction_b * frame.tangent_u[None, :, :]).sum(dim=2).square()
        * frame.tangent_major_scale[None, :].square()
        + (direction_b * frame.tangent_v[None, :, :]).sum(dim=2).square()
        * frame.tangent_minor_scale[None, :].square()
    )
    tangent_overlap = tangent_distance <= FOOTPRINT_SIGMA * (reach_a + reach_b)
    normal_depth_compatible = (depth_a <= FOOTPRINT_SIGMA) & (depth_b <= FOOTPRINT_SIGMA)
    normal_compatible = normal_alignment >= DEFAULT_SAME_SURFACE_MIN_NORMAL_ALIGNMENT
    footprint_compatible = off_diagonal & tangent_overlap & normal_depth_compatible & normal_compatible
    layer_conflict = off_diagonal & tangent_overlap & ~(
        normal_depth_compatible & normal_compatible
    )

    relation = _edge_relation_lookup(relation_edges)
    local_relation: list[list[str | None]] = [[None] * count for _ in range(count)]
    for local_a, global_a in enumerate(members):
        for local_b in range(local_a + 1, count):
            global_b = members[local_b]
            key = (global_a, global_b) if global_a < global_b else (global_b, global_a)
            value = relation.get(key)
            local_relation[local_a][local_b] = value
            local_relation[local_b][local_a] = value

    compatible_any = footprint_compatible.any(dim=1)
    center_missing = torch.zeros(count, dtype=torch.bool, device=points.device)
    relation_false_negative = torch.zeros_like(center_missing)
    accepted_same_surface = torch.zeros_like(center_missing)
    accepted_matrix = torch.zeros((count, count), dtype=torch.bool, device=points.device)
    provenance_veto = torch.zeros_like(center_missing)
    compatible_pair_count = missing_pair_count = rejected_pair_count = accepted_pair_count = 0
    veto_pair_count = 0
    for a in range(count):
        for b in range(a + 1, count):
            if not bool(footprint_compatible[a, b]):
                continue
            compatible_pair_count += 1
            value = local_relation[a][b]
            if value is None:
                missing_pair_count += 1
                center_missing[a] = center_missing[b] = True
            elif value == RELATION_AMBIGUOUS:
                rejected_pair_count += 1
                relation_false_negative[a] = relation_false_negative[b] = True
            elif value == RELATION_SAME_SURFACE:
                accepted_pair_count += 1
                accepted_same_surface[a] = accepted_same_surface[b] = True
                accepted_matrix[a, b] = accepted_matrix[b, a] = True
            elif value == RELATION_CREASE_VETOED:
                veto_pair_count += 1
                provenance_veto[a] = provenance_veto[b] = True

    layer_nodes = layer_conflict.any(dim=1)
    # A graph-to-surface mismatch requires more than a single accepted edge:
    # every marked node has an accepted compatible triangle in the bounded
    # center graph, yet this unit still failed Worklog 89 face recovery.
    graph_complex_nodes = torch.zeros_like(center_missing)
    for node in range(count):
        neighbors = torch.nonzero(accepted_matrix[node], as_tuple=False).reshape(-1).tolist()
        graph_complex_nodes[node] = any(
            bool(accepted_matrix[a, b])
            for offset, a in enumerate(neighbors)
            for b in neighbors[offset + 1:]
        )
    # A true gap has neither a one-sigma compatible footprint continuation nor
    # an overlapping competing layer.  Typed-veto-only evidence is disclosed
    # separately rather than reclassified as a relation false negative.
    true_gap = ~compatible_any & ~layer_nodes
    # Compatible footprints with no local center-graph triangle are the
    # center-point undersampling case, including a lone accepted edge that
    # cannot constitute a surface face complex by itself.
    center_undersampling = compatible_any & ~graph_complex_nodes
    masses = {
        CENTER_UNDERSAMPLING: float(center_undersampling.float().mean()),
        RELATION_FALSE_NEGATIVE: float(relation_false_negative.float().mean()),
        TRUE_SUPPORT_GAP: float(true_gap.float().mean()),
        MULTILAYER_OR_VOLUMETRIC: float(layer_nodes.float().mean()),
        GRAPH_TO_SURFACE_TOPOLOGY_MISMATCH: float(graph_complex_nodes.float().mean()),
    }
    primary = _primary_cause(masses)
    nearest = distance.masked_fill(~off_diagonal, float("inf")).min(dim=1).values
    spacing_ratio = nearest / frame.equivalent_tangent_scale.clamp_min(_EPS)
    plausible = primary in {
        CENTER_UNDERSAMPLING,
        RELATION_FALSE_NEGATIVE,
        GRAPH_TO_SURFACE_TOPOLOGY_MISMATCH,
    }
    missing_fraction = (
        float((missing_pair_count + rejected_pair_count) / compatible_pair_count)
        if compatible_pair_count else 0.0
    )
    return SurfaceTopologyAttribution(
        primary_cause=primary,
        cause_node_fractions=masses,
        center_spacing_over_tangent_scale_median=float(spacing_ratio.median()),
        compatible_footprint_overlap_coverage=float(compatible_any.float().mean()),
        missing_same_surface_edge_fraction_despite_footprint=missing_fraction,
        relation_false_negative_fraction=float(relation_false_negative.float().mean()),
        layer_ambiguity_fraction=float(layer_nodes.float().mean()),
        valid_local_surface_complex_plausible=plausible,
        footprint_compatible_pair_count=compatible_pair_count,
        missing_center_graph_pair_count=missing_pair_count,
        rejected_relation_pair_count=rejected_pair_count,
        accepted_same_surface_pair_count=accepted_pair_count,
        layer_conflict_pair_count=int(layer_conflict.triu(diagonal=1).sum()),
        true_gap_node_count=int(true_gap.sum()),
        provenance_veto_pair_count=veto_pair_count,
    )
