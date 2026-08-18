from __future__ import annotations

"""Worklog 100 -- orientation-preserving refinement of a global differential
(u, v) integration.

CANDIDATE_C: initialized STRICTLY from
:func:`~osn_gs.surface.torch_global_differential_uv_integration.integrate_global_differential_uv`
(candidate B) -- never a new or different parameterization family. Retains
the exact same edge differential objective, plus a local injectivity /
orientation-preservation term: points whose local source-tangent -> UV
Jacobian determinant collapses toward zero or reverses sign (using the
SAME source-graph neighborhoods and synchronized frame the corrected
:mod:`~osn_gs.surface.torch_parametric_domain_validity` validator uses) get
their incident edge weights boosted for one further weighted least-squares
resolve of the SAME global objective. This is one fixed formulation
(fixed iteration count, fixed boost factor) applied uniformly to every
component -- never tuned from held-out NURBS performance.

If, after the fixed refinement schedule, a component still contains a
genuine local fold (spatially inconsistent orientation relative to the
synchronized frame, exactly as the corrected validator defines it), this
fails CLOSED: the component is reported not globally parameterizable at
its current scale. There is no PCA repair and no fit-driven splitting
here.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_global_differential_uv_integration import (
    GlobalIntegrationResult,
    integrate_global_differential_uv,
)
from osn_gs.surface.torch_latent_surface_edge_differential import EdgeDifferential
from osn_gs.surface.torch_parametric_domain_validity import (
    ParametricDomainValidityReport,
    assess_parametric_domain_validity,
)
from osn_gs.utils.torch_ops import require_torch

# Fixed refinement schedule -- not tuned from any held-out/fit outcome.
REFINEMENT_ITERATIONS = 3
FOLD_EDGE_WEIGHT_BOOST = 8.0


@dataclass(frozen=True)
class OrientationPreservingResult:
    valid: bool
    invalid_reason: str | None
    uv: Any | None
    refinement_iterations_used: int
    domain_report: ParametricDomainValidityReport | None
    base_result: GlobalIntegrationResult


def _reweight_for_folds(
    component: Any, edge_differentials: tuple[EdgeDifferential, ...], domain_report: ParametricDomainValidityReport,
    fold_nodes: set[int],
) -> tuple[EdgeDifferential, ...]:
    if not fold_nodes:
        return edge_differentials
    boosted = []
    for edge in edge_differentials:
        if edge.node_a in fold_nodes or edge.node_b in fold_nodes:
            boosted.append(EdgeDifferential(edge.node_a, edge.node_b, edge.du, edge.dv, edge.weight * FOLD_EDGE_WEIGHT_BOOST))
        else:
            boosted.append(edge)
    return tuple(boosted)


def _fold_nodes(component: Any, uv: Any, median_spacing: float) -> tuple[set[int], ParametricDomainValidityReport]:
    """Reuses the corrected validator's own source-graph adjacency and
    synchronized-frame orientation check to find which specific nodes
    participate in a genuine local fold (after global-flip
    canonicalization) -- the same definition used for the final pass/fail
    decision, so the refinement penalizes exactly the failure mode the
    validator reports."""

    from osn_gs.surface.torch_parametric_domain_validity import _local_source_jacobian, _source_graph_adjacency

    torch = require_torch()
    report = assess_parametric_domain_validity(component, uv, median_spacing)
    if "uv_orientation_reversal_or_foldover" not in report.invalid_reasons:
        return set(), report

    adjacency = _source_graph_adjacency(component)
    determinants: dict[int, float] = {}
    for index in range(int(component.positions.shape[0])):
        jacobian, _singular_values = _local_source_jacobian(component, uv, index, adjacency.get(index, []))
        if jacobian is None:
            continue
        determinants[index] = float(torch.linalg.det(jacobian).item())
    if not determinants:
        return set(), report
    positive_count = sum(1 for value in determinants.values() if value > 0)
    negative_count = len(determinants) - positive_count
    if negative_count > positive_count:
        determinants = {index: -value for index, value in determinants.items()}

    fold_nodes: set[int] = set()
    for index, sign in determinants.items():
        own_positive = sign > 0
        for neighbor in adjacency.get(index, []):
            neighbor_sign = determinants.get(neighbor)
            if neighbor_sign is None:
                continue
            if (neighbor_sign > 0) != own_positive:
                fold_nodes.add(index)
                fold_nodes.add(neighbor)
                break
    return fold_nodes, report


def integrate_orientation_preserving_uv(
    component: Any, edge_differentials: tuple[EdgeDifferential, ...], median_spacing: float,
) -> OrientationPreservingResult:
    base_result = integrate_global_differential_uv(component, edge_differentials)
    if not base_result.valid:
        return OrientationPreservingResult(
            False, f"base_integration_failed:{base_result.invalid_reason}", None, 0, None, base_result,
        )

    uv = base_result.uv
    working_edges = edge_differentials
    domain_report: ParametricDomainValidityReport | None = None
    iterations_used = 0
    for iteration in range(REFINEMENT_ITERATIONS):
        fold_nodes, domain_report = _fold_nodes(component, uv, median_spacing)
        if not fold_nodes:
            return OrientationPreservingResult(True, None, uv, iterations_used, domain_report, base_result)
        working_edges = _reweight_for_folds(component, working_edges, domain_report, fold_nodes)
        refined = integrate_global_differential_uv(component, working_edges)
        iterations_used = iteration + 1
        if not refined.valid:
            # Cannot even re-solve under the boosted weighting -- fail
            # closed rather than falling back to a different family.
            return OrientationPreservingResult(
                False, f"refinement_resolve_failed:{refined.invalid_reason}", None, iterations_used, domain_report, base_result,
            )
        uv = refined.uv

    fold_nodes, domain_report = _fold_nodes(component, uv, median_spacing)
    if fold_nodes:
        # Fixed schedule exhausted and a genuine local fold still remains:
        # fail closed rather than repair via PCA or fit-driven splitting.
        return OrientationPreservingResult(
            False, "not_globally_parameterizable_at_current_scale", None, iterations_used, domain_report, base_result,
        )
    return OrientationPreservingResult(True, None, uv, iterations_used, domain_report, base_result)
