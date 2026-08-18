from __future__ import annotations

"""Worklog 100 -- global differential (u, v) integration over a Worklog 98
synchronized tangent frame field.

Worklog 99 found that 80.4% of Worklog 98's orientation-coherent components
already fail a pre-fit parameter-domain check before any surface is even
attempted. Worklog 98's own ``(u, v)`` is accumulated along ONE Dijkstra
spanning-tree path per component; every other supported edge is only ever
checked for consistency, never used to determine the actual coordinate.
This module answers the architecture question directly: does solving for
``(u, v)`` from ALL supported edges simultaneously (a weighted least-
squares / discrete-Poisson-style integration), rather than accumulating
along one path, remove the path-dependent drift that Worklog 99's
corrected validator (:mod:`~osn_gs.surface.torch_parametric_domain_validity`)
can now detect as spurious local folds?

CANDIDATE_B (this module): global least-squares differential integration.
``min sum_ij w_ij [(u_j - u_i - du_ij)^2 + (v_j - v_i - dv_ij)^2]`` over
every :class:`~osn_gs.surface.torch_latent_surface_edge_differential.EdgeDifferential`
in a coherent component, with ONLY the minimal required gauge freedom
fixed (one point's ``(u, v)`` pinned to the origin, removing the
translation-only null space -- nothing else is renormalized or
PCA-aligned, so the differential metric the synchronized frame already
establishes is preserved exactly).

CANDIDATE_C lives in
:mod:`~osn_gs.surface.torch_orientation_preserving_uv_integration` and is
always initialized strictly from this module's result.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_latent_surface_edge_differential import EdgeDifferential
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9


@dataclass(frozen=True)
class GlobalIntegrationResult:
    valid: bool
    invalid_reason: str | None
    uv: Any | None  # (M, 2), aligned to component.node order (local indices)
    overall_residual_rms: float | None
    per_edge_residual_p50: float | None
    per_edge_residual_p95: float | None
    cycle_edge_residual_rms: float | None
    gauge_node_index: int | None


def integrate_global_differential_uv(
    component: Any, edge_differentials: tuple[EdgeDifferential, ...],
) -> GlobalIntegrationResult:
    """Solve the weighted least-squares system for ``(u, v)`` over every
    node touched by ``edge_differentials`` simultaneously. Fails closed
    (``valid=False``) if there are too few edges/nodes to determine a
    non-trivial solution, or if the resulting system is singular even
    after gauge-fixing."""

    torch = require_torch()
    if not edge_differentials:
        return GlobalIntegrationResult(False, "no_supported_edges", None, None, None, None, None, None)

    node_set = sorted({edge.node_a for edge in edge_differentials} | {edge.node_b for edge in edge_differentials})
    if len(node_set) < 3:
        return GlobalIntegrationResult(False, "insufficient_connected_nodes", None, None, None, None, None, None)

    index_of = {node: position for position, node in enumerate(node_set)}
    n = len(node_set)
    m = len(edge_differentials)

    # Minimal gauge fix: pin the first node's (u, v) to the origin. This
    # removes exactly the one-dimensional-per-coordinate translation null
    # space of the graph Laplacian system without touching the actual
    # differential metric on any edge.
    gauge_node = node_set[0]
    gauge_index = index_of[gauge_node]
    free_indices = [position for position in range(n) if position != gauge_index]
    free_index_of = {position: column for column, position in enumerate(free_indices)}

    design = torch.zeros((m, n - 1), dtype=component.positions.dtype, device=component.positions.device)
    target_u = torch.zeros(m, dtype=component.positions.dtype, device=component.positions.device)
    target_v = torch.zeros(m, dtype=component.positions.dtype, device=component.positions.device)
    weights = torch.zeros(m, dtype=component.positions.dtype, device=component.positions.device)
    for row, edge in enumerate(edge_differentials):
        pos_a, pos_b = index_of[edge.node_a], index_of[edge.node_b]
        if pos_b != gauge_index:
            design[row, free_index_of[pos_b]] += 1.0
        if pos_a != gauge_index:
            design[row, free_index_of[pos_a]] -= 1.0
        target_u[row] = edge.du
        target_v[row] = edge.dv
        weights[row] = edge.weight

    sqrt_weights = weights.clamp_min(_EPS).sqrt().unsqueeze(1)
    weighted_design = design * sqrt_weights
    weighted_target = torch.stack([target_u, target_v], dim=1) * sqrt_weights

    try:
        solution = torch.linalg.lstsq(weighted_design, weighted_target).solution  # (n-1, 2)
    except Exception as exc:  # pragma: no cover - defensive
        return GlobalIntegrationResult(False, f"singular_system:{type(exc).__name__}", None, None, None, None, None, None)
    if bool(torch.isnan(solution).any().item()) or bool(torch.isinf(solution).any().item()):
        return GlobalIntegrationResult(False, "non_finite_solution", None, None, None, None, None, None)

    uv = torch.zeros((n, 2), dtype=component.positions.dtype, device=component.positions.device)
    for position in free_indices:
        uv[position] = solution[free_index_of[position]]
    # gauge_index row already zero.

    # Map back to local node-index order (0..M-1 over the component).
    full_uv = torch.zeros((int(component.positions.shape[0]), 2), dtype=uv.dtype, device=uv.device)
    for node, position in index_of.items():
        full_uv[node] = uv[position]

    residuals = design @ solution - torch.stack([target_u, target_v], dim=1)
    per_edge_residual = residuals.norm(dim=1)
    overall_rms = float(torch.sqrt((weights * per_edge_residual.square()).sum() / weights.sum().clamp_min(_EPS)).item())
    per_edge_p50 = float(torch.quantile(per_edge_residual, 0.50).item())
    per_edge_p95 = float(torch.quantile(per_edge_residual, 0.95).item())

    tree_edge_set = {tuple(sorted(edge)) for edge in component.tree_edges}
    cycle_rows = [
        row for row, edge in enumerate(edge_differentials)
        if tuple(sorted((edge.node_a, edge.node_b))) not in tree_edge_set
    ]
    cycle_residual_rms = (
        float(per_edge_residual[cycle_rows].square().mean().sqrt().item()) if cycle_rows else None
    )

    return GlobalIntegrationResult(
        valid=True, invalid_reason=None, uv=full_uv,
        overall_residual_rms=overall_rms,
        per_edge_residual_p50=per_edge_p50, per_edge_residual_p95=per_edge_p95,
        cycle_edge_residual_rms=cycle_residual_rms,
        gauge_node_index=gauge_node,
    )
