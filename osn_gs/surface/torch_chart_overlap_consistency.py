from __future__ import annotations

"""Worklog 102 -- pre-reconciliation overlap-consistency EVALUATION for a
Worklog 101 chart atlas.

Worklog 101's atlas allows neighboring charts to overlap (chart growth
never excludes already-covered nodes). This module evaluates, for every
pair of charts that share at least one source node, how much their
INDEPENDENTLY fitted patches disagree over that shared evidence --
positional disagreement, tangent/normal disagreement -- and whether both
patches remain supported there. This is purely a reporting metric in this
batch: nothing here merges, modifies, or re-fits a patch, and nothing
tunes patch capacity to reduce disagreement.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9


@dataclass(frozen=True)
class OverlapPairConsistency:
    chart_id_a: int
    chart_id_b: int
    shared_node_count: int
    both_fitted: bool
    position_disagreement_p50: float | None
    position_disagreement_p95: float | None
    normal_disagreement_degrees_p50: float | None
    normal_disagreement_degrees_p95: float | None


def _surface_normal(surface: Any, uv: Any) -> Any:
    torch = require_torch()
    _point, deriv_u, deriv_v = surface.evaluate_with_derivatives(uv)
    normal = torch.linalg.cross(deriv_u, deriv_v)
    norm = normal.norm(dim=1, keepdim=True).clamp_min(_EPS)
    return normal / norm


def evaluate_overlap_consistency(
    charts: list[Any],  # Chart objects (torch_intrinsic_chart_atlas.Chart)
    surfaces_by_chart_id: dict[int, Any],  # chart_id -> TorchNURBSSurface or None (fit failed/not attempted)
    uv_by_chart_id: dict[int, Any],  # chart_id -> (M, 2) uv aligned to chart.node_indices order
    scale: float,
) -> tuple[OverlapPairConsistency, ...]:
    """Evaluate every pair of charts sharing at least one source node.
    Reporting only -- callers must not use this to merge/modify/re-tune
    any chart or patch capacity in this batch."""

    torch = require_torch()
    node_membership: dict[int, list[int]] = {}
    for chart in charts:
        for node in chart.node_indices:
            node_membership.setdefault(node, []).append(chart.chart_id)

    chart_by_id = {chart.chart_id: chart for chart in charts}
    pair_shared_nodes: dict[tuple[int, int], list[int]] = {}
    for node, chart_ids in node_membership.items():
        if len(chart_ids) < 2:
            continue
        unique_ids = sorted(set(chart_ids))
        for i in range(len(unique_ids)):
            for j in range(i + 1, len(unique_ids)):
                pair_shared_nodes.setdefault((unique_ids[i], unique_ids[j]), []).append(node)

    results: list[OverlapPairConsistency] = []
    for (chart_id_a, chart_id_b), shared_nodes in sorted(pair_shared_nodes.items()):
        surface_a = surfaces_by_chart_id.get(chart_id_a)
        surface_b = surfaces_by_chart_id.get(chart_id_b)
        both_fitted = surface_a is not None and surface_b is not None
        if not both_fitted:
            results.append(OverlapPairConsistency(
                chart_id_a, chart_id_b, len(shared_nodes), False, None, None, None, None,
            ))
            continue

        chart_a, chart_b = chart_by_id[chart_id_a], chart_by_id[chart_id_b]
        local_a = {node: local for local, node in enumerate(chart_a.node_indices)}
        local_b = {node: local for local, node in enumerate(chart_b.node_indices)}
        uv_a = uv_by_chart_id[chart_id_a]
        uv_b = uv_by_chart_id[chart_id_b]

        rows_a = torch.tensor([local_a[node] for node in shared_nodes], dtype=torch.long)
        rows_b = torch.tensor([local_b[node] for node in shared_nodes], dtype=torch.long)
        query_uv_a = uv_a[rows_a]
        query_uv_b = uv_b[rows_b]

        point_a, _du_a, _dv_a = surface_a.evaluate_with_derivatives(query_uv_a)
        point_b, _du_b, _dv_b = surface_b.evaluate_with_derivatives(query_uv_b)
        position_disagreement = (point_a - point_b).norm(dim=1) / max(scale, _EPS)

        normal_a = _surface_normal(surface_a, query_uv_a)
        normal_b = _surface_normal(surface_b, query_uv_b)
        cosine = (normal_a * normal_b).sum(dim=1).clamp(-1.0, 1.0)
        angle_degrees = torch.rad2deg(torch.arccos(cosine))

        results.append(OverlapPairConsistency(
            chart_id_a, chart_id_b, len(shared_nodes), True,
            float(torch.quantile(position_disagreement, 0.50).item()),
            float(torch.quantile(position_disagreement, 0.95).item()),
            float(torch.quantile(angle_degrees, 0.50).item()),
            float(torch.quantile(angle_degrees, 0.95).item()),
        ))
    return tuple(results)
