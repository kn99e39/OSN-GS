from __future__ import annotations

"""Phase F — Minimal Constrained Occluded NURBS Bridge.

docs/Urgent_Work/OSN_GS_Phase_F_Constrained_Occluded_Chart_Design.md,
impl plan docs/Urgent_Work/OSN_GS_Boundary_Conditioned_Occlusion_Impl_Plan.md section 8.

Converts one Phase E pairwise bounded ribbon `OccludedRegionCandidate` into a
real constrained occluded NURBS chart (`TorchNURBSSurface`) via:

    candidate + ContinuationDomain registry
    -> open quadrilateral parameterization (correspondence-edge paired resampling)
    -> Coons/transfinite seed
    -> single-chart constrained LSQ (support high-weight, connector low-weight, interior fairness)
    -> validity gate -> OccludedChartResult

Scope limits (design sections 2, 20): open pairwise quadrilateral ribbon only.
Cyclic candidates return `unsupported`; no multi-sided topology, no global
selection, no Gaussian proposal, no production wiring, no robust
self-intersection / visible-surface-penetration (cheap proxies only). Phase E
evidence is used for post-fit validation / metadata / rejection provenance
ONLY, never as a solver weight.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

from osn_gs.surface.torch_coons_patch import coons_bilinear_patch
from osn_gs.surface.torch_constrained_chart_lsq import solve_constrained_chart
from osn_gs.surface.torch_continuation_domain import ContinuationDomain
from osn_gs.surface.torch_occluded_region_candidate import OccludedRegionCandidate
from osn_gs.surface.torch_parametric_diagnostics import (
    compute_orientation_consistency,
    compute_parametric_jacobian_metrics,
)
from osn_gs.surface.torch_patch_boundary import PatchBoundarySegment
from osn_gs.utils.torch_ops import require_torch

STATE_FITTED = "fitted"
STATE_VALIDATED = "validated"
STATE_UNSUPPORTED = "unsupported"
STATE_REJECTED = "rejected"
CHART_STATES = {STATE_FITTED, STATE_VALIDATED, STATE_UNSUPPORTED, STATE_REJECTED}

TOPOLOGY_OPEN_QUAD = "open_quadrilateral"

_EPS = 1e-9


@dataclass
class OccludedChartFitConfig:
    resolution_u: int = 7
    resolution_v: int = 5
    degree_u: int = 3
    degree_v: int = 3
    boundary_sample_count: int = 0  # 0 -> derive as max(2*resolution_u, correspondence length)
    connector_sample_count: int = 5
    support_weight: float = 1.0
    connector_weight: float = 0.03
    fairness_weight: float = 1e-3
    interior_seed_weight: float = 1e-4
    c0_residual_rel_tolerance: float = 0.15  # fraction of local surface scale
    jacobian_eps: float = 1e-8
    diagnostic_grid: int = 12
    run_validation: bool = True

    def fingerprint(self) -> str:
        parts = [
            f"{self.resolution_u}x{self.resolution_v}",
            f"deg{self.degree_u}x{self.degree_v}",
            f"bs{self.boundary_sample_count}",
            f"cs{self.connector_sample_count}",
            f"sw{self.support_weight:.6g}",
            f"cw{self.connector_weight:.6g}",
            f"fw{self.fairness_weight:.6g}",
            f"iw{self.interior_seed_weight:.6g}",
        ]
        return "|".join(parts)

    def payload(self) -> dict[str, Any]:
        return {
            "resolution_u": self.resolution_u,
            "resolution_v": self.resolution_v,
            "degree_u": self.degree_u,
            "degree_v": self.degree_v,
            "boundary_sample_count": self.boundary_sample_count,
            "connector_sample_count": self.connector_sample_count,
            "support_weight": self.support_weight,
            "connector_weight": self.connector_weight,
            "fairness_weight": self.fairness_weight,
            "interior_seed_weight": self.interior_seed_weight,
            "c0_residual_rel_tolerance": self.c0_residual_rel_tolerance,
            "jacobian_eps": self.jacobian_eps,
            "diagnostic_grid": self.diagnostic_grid,
            "run_validation": self.run_validation,
        }


@dataclass
class OccludedChartResult:
    chart_id: str
    source_candidate_id: str

    supporting_domain_ids: list[str]
    supporting_boundary_ids: list[str]
    supporting_patch_ids: list[int]

    topology: str
    surface: Any  # TorchNURBSSurface | None

    common_parameter: Any
    support_samples_a: Any
    support_samples_b: Any
    connector_samples_start: Any
    connector_samples_end: Any
    parameter_correspondence: dict[str, Any]

    constraint_config: dict[str, Any]
    fit_diagnostics: dict[str, Any]
    boundary_conformance: dict[str, Any]
    tangent_mismatch: dict[str, Any]
    normal_mismatch: dict[str, Any]
    second_order_mismatch: dict[str, Any]

    jacobian_diagnostics: dict[str, Any]
    orientation_diagnostics: dict[str, Any]
    parameter_quality: dict[str, Any]

    self_intersection_status: dict[str, Any]
    visible_surface_penetration_status: dict[str, Any]
    evidence_consistency: dict[str, Any]
    conflict_provenance: dict[str, Any]

    state: str
    reason: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.state not in CHART_STATES:
            raise ValueError(f"Unknown occluded-chart state: {self.state!r}")

    def payload(self) -> dict[str, Any]:
        def _t(x: Any) -> Any:
            return None if x is None else x.detach().cpu().tolist()

        return {
            "chart_id": self.chart_id,
            "source_candidate_id": self.source_candidate_id,
            "supporting_domain_ids": list(self.supporting_domain_ids),
            "supporting_boundary_ids": list(self.supporting_boundary_ids),
            "supporting_patch_ids": [int(p) for p in self.supporting_patch_ids],
            "topology": self.topology,
            "surface_control_grid": None if self.surface is None else self.surface.control_grid.detach().cpu().tolist(),
            "common_parameter": _t(self.common_parameter),
            "support_samples_a": _t(self.support_samples_a),
            "support_samples_b": _t(self.support_samples_b),
            "connector_samples_start": _t(self.connector_samples_start),
            "connector_samples_end": _t(self.connector_samples_end),
            "parameter_correspondence": dict(self.parameter_correspondence),
            "constraint_config": dict(self.constraint_config),
            "fit_diagnostics": dict(self.fit_diagnostics),
            "boundary_conformance": dict(self.boundary_conformance),
            "tangent_mismatch": dict(self.tangent_mismatch),
            "normal_mismatch": dict(self.normal_mismatch),
            "second_order_mismatch": dict(self.second_order_mismatch),
            "jacobian_diagnostics": dict(self.jacobian_diagnostics),
            "orientation_diagnostics": dict(self.orientation_diagnostics),
            "parameter_quality": dict(self.parameter_quality),
            "self_intersection_status": dict(self.self_intersection_status),
            "visible_surface_penetration_status": dict(self.visible_surface_penetration_status),
            "evidence_consistency": dict(self.evidence_consistency),
            "conflict_provenance": dict(self.conflict_provenance),
            "state": self.state,
            "reason": self.reason,
            "provenance": dict(self.provenance),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _resample_polyline(points: Any, param: Any, query: Any) -> Any:
    """Piecewise-linear resample of an ordered polyline at normalized ``query``."""

    torch = require_torch()
    n = int(points.shape[0])
    if n == 1:
        return points.expand(int(query.shape[0]), 3).clone()
    idx = torch.searchsorted(param.contiguous(), query.contiguous(), right=False).clamp(1, n - 1)
    lo = idx - 1
    p_lo, p_hi = param[lo], param[idx]
    denom = (p_hi - p_lo).clamp_min(1e-12)
    frac = ((query - p_lo) / denom).clamp(0.0, 1.0)
    return points[lo] + frac[:, None] * (points[idx] - points[lo])


def _paired_arclength(a_points: Any, b_points: Any) -> Any:
    torch = require_torch()
    n = int(a_points.shape[0])
    if n < 2:
        return a_points.new_zeros(n)
    seg = 0.5 * ((a_points[1:] - a_points[:-1]).norm(dim=1) + (b_points[1:] - b_points[:-1]).norm(dim=1))
    cumulative = torch.cat([a_points.new_zeros(1), torch.cumsum(seg, dim=0)])
    total = cumulative[-1].clamp_min(1e-12)
    return cumulative / total


def _chamfer(edge: Any, reference: Any) -> dict[str, float]:
    torch = require_torch()
    d = torch.cdist(edge, reference)
    edge_to_ref = d.min(dim=1).values
    ref_to_edge = d.min(dim=0).values
    return {
        "edge_to_reference_mean": float(edge_to_ref.mean()),
        "edge_to_reference_max": float(edge_to_ref.max()),
        "reference_to_edge_mean": float(ref_to_edge.mean()),
        "reference_to_edge_max": float(ref_to_edge.max()),
        "symmetric_max": float(torch.maximum(edge_to_ref.max(), ref_to_edge.max())),
    }


def _angle_stats(a_unit: Any, b_unit: Any) -> dict[str, float]:
    torch = require_torch()
    dot = (a_unit * b_unit).sum(dim=1).clamp(-1.0, 1.0)
    angle = torch.arccos(dot.abs())  # undirected: chart normal sign is arbitrary
    return {
        "angle_deg_mean": float(torch.rad2deg(angle).mean()),
        "angle_deg_max": float(torch.rad2deg(angle).max()),
        "abs_dot_min": float(dot.abs().min()),
    }


def _unit(v: Any, eps: float) -> Any:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def _early_result(
    candidate: OccludedRegionCandidate, config: OccludedChartFitConfig, state: str, reason: str
) -> OccludedChartResult:
    empty: dict[str, Any] = {}
    chart_id = _chart_id(candidate.candidate_id, config, "none")
    return OccludedChartResult(
        chart_id=chart_id,
        source_candidate_id=candidate.candidate_id,
        supporting_domain_ids=list(candidate.supporting_domain_ids),
        supporting_boundary_ids=list(candidate.supporting_boundary_ids),
        supporting_patch_ids=list(candidate.supporting_patch_ids),
        topology=TOPOLOGY_OPEN_QUAD,
        surface=None,
        common_parameter=None,
        support_samples_a=None,
        support_samples_b=None,
        connector_samples_start=None,
        connector_samples_end=None,
        parameter_correspondence=dict(empty),
        constraint_config=config.payload(),
        fit_diagnostics=dict(empty),
        boundary_conformance=dict(empty),
        tangent_mismatch=dict(empty),
        normal_mismatch=dict(empty),
        second_order_mismatch=dict(empty),
        jacobian_diagnostics=dict(empty),
        orientation_diagnostics=dict(empty),
        parameter_quality=dict(empty),
        self_intersection_status={"checked": False},
        visible_surface_penetration_status={"checked": False},
        evidence_consistency=_evidence_consistency(candidate),
        conflict_provenance={"candidate_reason": candidate.reason, "candidate_state": candidate.state},
        state=state,
        reason=reason,
        provenance={"candidate_state": candidate.state, "solver_run": False},
    )


def _evidence_consistency(candidate: OccludedRegionCandidate) -> dict[str, Any]:
    fc = candidate.free_space_contradiction
    return {
        "candidate_hard_contradiction": bool(fc.get("candidate_hard_contradiction", False)) if fc else False,
        "free_space_contradiction": dict(candidate.free_space_contradiction),
        "behind_observation_support": dict(candidate.behind_observation_support),
        "conflicting_evidence": dict(candidate.conflicting_evidence),
        "empty_voxel_support": dict(candidate.empty_voxel_support),
        "used_as_solver_weight": False,
    }


def _chart_id(candidate_id: str, config: OccludedChartFitConfig, param_fp: str) -> str:
    digest = hashlib.sha256(f"{candidate_id}::{config.fingerprint()}::{param_fp}".encode()).hexdigest()[:16]
    return f"{candidate_id}#chart:{digest}"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def fit_occluded_chart(
    candidate: OccludedRegionCandidate,
    domains_by_id: Mapping[str, ContinuationDomain],
    boundaries_by_id: Mapping[str, PatchBoundarySegment],
    surfaces_by_patch_id: Mapping[int, Any] | None = None,
    *,
    config: OccludedChartFitConfig | None = None,
) -> OccludedChartResult:
    """Fit a constrained occluded NURBS chart for one Phase E candidate."""

    torch = require_torch()
    config = config or OccludedChartFitConfig()

    # --- State propagation (no solve) ---
    if candidate.state == "rejected":
        return _early_result(candidate, config, STATE_REJECTED, f"candidate_rejected:{candidate.reason}")
    if candidate.state == "unsupported":
        return _early_result(candidate, config, STATE_UNSUPPORTED, f"candidate_unsupported:{candidate.reason}")
    if candidate.connector_end is None or bool(candidate.provenance.get("cyclic", False)):
        return _early_result(candidate, config, STATE_UNSUPPORTED, "cyclic_topology_deferred")
    if candidate.free_space_contradiction and bool(
        candidate.free_space_contradiction.get("candidate_hard_contradiction", False)
    ):
        return _early_result(candidate, config, STATE_REJECTED, "evidence_full_known_free_space")

    # --- Input contract (eager errors) ---
    if len(candidate.supporting_domain_ids) != 2:
        raise ValueError("Phase F supports exactly two supporting domains per candidate.")
    domain_a_id, domain_b_id = candidate.supporting_domain_ids
    if domain_a_id not in domains_by_id or domain_b_id not in domains_by_id:
        raise ValueError("Both supporting domains must be present in domains_by_id.")
    domain_a = domains_by_id[domain_a_id]
    domain_b = domains_by_id[domain_b_id]

    edges = candidate.correspondence_edges
    if len(edges) < 2:
        return _early_result(candidate, config, STATE_UNSUPPORTED, "fewer_than_two_correspondence_pairs")

    dtype = domain_a.world.dtype
    device = domain_a.world.device

    # --- Recover matched (continuation-supported) chains + visible reference ---
    a_matched = torch.stack([domain_a.world[e.s_a, e.t_a] for e in edges])
    b_matched = torch.stack([domain_b.world[e.s_b, e.t_b] for e in edges])
    a_visible = torch.stack([domain_a.world[e.s_a, 0] for e in edges])
    b_visible = torch.stack([domain_b.world[e.s_b, 0] for e in edges])
    a_outward = torch.stack([domain_a.outward_tangent_world[e.s_a] for e in edges])
    b_outward = torch.stack([domain_b.outward_tangent_world[e.s_b] for e in edges])
    a_normal = torch.stack([domain_a.normal[e.s_a, e.t_a] for e in edges])
    b_normal = torch.stack([domain_b.normal[e.s_b, e.t_b] for e in edges])

    # --- Paired resampling to a common u grid (design section 6) ---
    param = _paired_arclength(a_matched, b_matched)
    n_edges = int(a_matched.shape[0])
    fit_count = config.boundary_sample_count if config.boundary_sample_count > 0 else max(2 * config.resolution_u, n_edges)
    u_fit = torch.linspace(0.0, 1.0, int(fit_count), dtype=dtype, device=device)
    u_seed = torch.linspace(0.0, 1.0, int(config.resolution_u), dtype=dtype, device=device)

    support_a_fit = _resample_polyline(a_matched, param, u_fit)
    support_b_fit = _resample_polyline(b_matched, param, u_fit)
    support_a_seed = _resample_polyline(a_matched, param, u_seed)
    support_b_seed = _resample_polyline(b_matched, param, u_seed)

    support_interval_a = float((support_a_fit[1:] - support_a_fit[:-1]).norm(dim=1).sum())
    support_interval_b = float((support_b_fit[1:] - support_b_fit[:-1]).norm(dim=1).sum())
    if support_interval_a <= _EPS or support_interval_b <= _EPS:
        return _early_result(candidate, config, STATE_UNSUPPORTED, "zero_support_interval")

    # --- Connectors as straight lines between resampled endpoints ---
    v_seed = torch.linspace(0.0, 1.0, int(config.resolution_v), dtype=dtype, device=device)
    v_conn = torch.linspace(0.0, 1.0, int(config.connector_sample_count), dtype=dtype, device=device)

    def _line(p0: Any, p1: Any, t: Any) -> Any:
        return p0[None, :] + t[:, None] * (p1 - p0)[None, :]

    connector_start_seed = _line(support_a_seed[0], support_b_seed[0], v_seed)
    connector_end_seed = _line(support_a_seed[-1], support_b_seed[-1], v_seed)
    connector_start_fit = _line(support_a_fit[0], support_b_fit[0], v_conn)
    connector_end_fit = _line(support_a_fit[-1], support_b_fit[-1], v_conn)

    start_sep = float((support_a_fit[0] - support_b_fit[0]).norm())
    end_sep = float((support_a_fit[-1] - support_b_fit[-1]).norm())
    if start_sep <= _EPS or end_sep <= _EPS:
        return _early_result(candidate, config, STATE_REJECTED, "zero_connector_separation")

    # --- Coons transfinite seed at control-grid resolution ---
    try:
        seed_grid = coons_bilinear_patch(
            support_a_seed, support_b_seed, connector_start_seed, connector_end_seed, atol=1e-5
        )
    except ValueError as exc:
        return _early_result(candidate, config, STATE_REJECTED, f"coons_seed_failed:{exc}")

    # --- Constrained single-chart LSQ ---
    support_a_uv = torch.stack([u_fit, torch.zeros_like(u_fit)], dim=1)
    support_b_uv = torch.stack([u_fit, torch.ones_like(u_fit)], dim=1)
    connector_uv = torch.cat(
        [
            torch.stack([torch.zeros_like(v_conn), v_conn], dim=1),
            torch.stack([torch.ones_like(v_conn), v_conn], dim=1),
        ],
        dim=0,
    )
    connector_points = torch.cat([connector_start_fit, connector_end_fit], dim=0)

    surface, solve_diag = solve_constrained_chart(
        seed_grid,
        support_a_uv=support_a_uv,
        support_a_points=support_a_fit,
        support_b_uv=support_b_uv,
        support_b_points=support_b_fit,
        connector_uv=connector_uv,
        connector_points=connector_points,
        degree_u=config.degree_u,
        degree_v=config.degree_v,
        support_weight=config.support_weight,
        connector_weight=config.connector_weight,
        fairness_weight=config.fairness_weight,
        interior_seed_weight=config.interior_seed_weight,
    )

    param_fp = f"{config.resolution_u}x{config.resolution_v}:{fit_count}:{n_edges}"
    chart_id = _chart_id(candidate.candidate_id, config, param_fp)

    parameter_correspondence = {
        "correspondence_pair_count": n_edges,
        "paired_arclength_param": param.detach().cpu().tolist(),
        "fit_sample_count": int(fit_count),
        "reversed_correspondence": bool(edges[0].s_b > edges[-1].s_b),
        "common_parameter_kind": "normalized_paired_arclength",
    }

    if not solve_diag["control_grid_finite"]:
        result = _assemble_result(
            candidate, config, chart_id, param_fp, surface=None, topology=TOPOLOGY_OPEN_QUAD,
            common_parameter=u_fit, support_a=support_a_fit, support_b=support_b_fit,
            connector_start=connector_start_fit, connector_end=connector_end_fit,
            parameter_correspondence=parameter_correspondence, fit_diagnostics=solve_diag,
            diagnostics=None, state=STATE_REJECTED, reason="non_finite_solve",
        )
        return result

    # --- Diagnostics ---
    diagnostics = _chart_diagnostics(
        surface, config, support_a_fit, support_b_fit, connector_start_fit, connector_end_fit,
        a_outward, b_outward, a_normal, b_normal, param, u_fit, domain_a, domain_b,
        candidate, surfaces_by_patch_id,
    )

    # --- Validity gate ---
    reference_scale = max(
        float(domain_a.local_surface_scale), float(domain_b.local_surface_scale), _EPS
    )
    c0_tolerance = float(config.c0_residual_rel_tolerance) * reference_scale
    c0_residual = diagnostics["boundary_conformance"]["c0_residual_max"]
    jac = diagnostics["jacobian_diagnostics"]
    orient = diagnostics["orientation_diagnostics"]
    zero_area = float(jac.get("min_area_jacobian", 0.0)) <= float(config.jacobian_eps)
    jacobian_collapse = float(jac.get("min_jacobian_singular_value_normalized", 0.0)) <= float(config.jacobian_eps)
    orientation_flip = int(orient.get("orientation_flip_count", 0)) > 0

    reasons: list[str] = []
    if zero_area:
        reasons.append("zero_area_chart")
    if jacobian_collapse:
        reasons.append("jacobian_collapse")
    if orientation_flip:
        reasons.append("orientation_flip")
    if c0_residual > c0_tolerance:
        reasons.append("c0_residual_exceeds_tolerance")

    if reasons:
        state, reason = STATE_REJECTED, ";".join(reasons)
    elif not config.run_validation:
        state, reason = STATE_FITTED, "solve_ok_validation_skipped"
    else:
        state, reason = STATE_VALIDATED, "ok"

    diagnostics["boundary_conformance"]["c0_tolerance"] = c0_tolerance
    diagnostics["boundary_conformance"]["reference_scale"] = reference_scale

    return _assemble_result(
        candidate, config, chart_id, param_fp, surface=surface, topology=TOPOLOGY_OPEN_QUAD,
        common_parameter=u_fit, support_a=support_a_fit, support_b=support_b_fit,
        connector_start=connector_start_fit, connector_end=connector_end_fit,
        parameter_correspondence=parameter_correspondence, fit_diagnostics=solve_diag,
        diagnostics=diagnostics, state=state, reason=reason,
    )


def _chart_diagnostics(
    surface: Any, config: OccludedChartFitConfig,
    support_a_fit: Any, support_b_fit: Any, connector_start_fit: Any, connector_end_fit: Any,
    a_outward: Any, b_outward: Any, a_normal: Any, b_normal: Any,
    param: Any, u_fit: Any, domain_a: ContinuationDomain, domain_b: ContinuationDomain,
    candidate: OccludedRegionCandidate, surfaces_by_patch_id: Mapping[int, Any] | None,
) -> dict[str, Any]:
    torch = require_torch()
    dtype, device = surface.control_grid.dtype, surface.control_grid.device
    eps = float(config.jacobian_eps)

    # Chart boundary edges.
    edge_v0_uv = torch.stack([u_fit, torch.zeros_like(u_fit)], dim=1)
    edge_v1_uv = torch.stack([u_fit, torch.ones_like(u_fit)], dim=1)
    _, su_v0, sv_v0 = surface.evaluate_with_derivatives(edge_v0_uv)
    chart_v0 = surface.evaluate(edge_v0_uv)
    chart_v1 = surface.evaluate(edge_v1_uv)
    _, su_v1, sv_v1 = surface.evaluate_with_derivatives(edge_v1_uv)

    v_conn = torch.linspace(0.0, 1.0, int(config.connector_sample_count), dtype=dtype, device=device)
    edge_u0 = surface.evaluate(torch.stack([torch.zeros_like(v_conn), v_conn], dim=1))
    edge_u1 = surface.evaluate(torch.stack([torch.ones_like(v_conn), v_conn], dim=1))

    conf_a = _chamfer(chart_v0, support_a_fit)
    conf_b = _chamfer(chart_v1, support_b_fit)
    conf_start = _chamfer(edge_u0, connector_start_fit)
    conf_end = _chamfer(edge_u1, connector_end_fit)
    boundary_conformance = {
        "support_a": conf_a,
        "support_b": conf_b,
        "c0_residual_max": max(conf_a["symmetric_max"], conf_b["symmetric_max"]),
        "connector_start": conf_start,
        "connector_end": conf_end,
        "connector_deviation_max": max(conf_start["symmetric_max"], conf_end["symmetric_max"]),
    }

    # Tangent / normal mismatch (soft, undirected) vs resampled domain frames.
    a_out_rs = _unit(_resample_polyline(a_outward, param, u_fit), eps)
    b_out_rs = _unit(_resample_polyline(b_outward, param, u_fit), eps)
    a_nrm_rs = _unit(_resample_polyline(a_normal, param, u_fit), eps)
    b_nrm_rs = _unit(_resample_polyline(b_normal, param, u_fit), eps)
    chart_out_v0 = _unit(sv_v0, eps)
    chart_out_v1 = _unit(-sv_v1, eps)  # inward-pointing at v=1 to compare against B outward
    chart_nrm_v0 = _unit(torch.cross(su_v0, sv_v0, dim=1), eps)
    chart_nrm_v1 = _unit(torch.cross(su_v1, sv_v1, dim=1), eps)
    tangent_mismatch = {
        "support_a": _angle_stats(chart_out_v0, a_out_rs),
        "support_b": _angle_stats(chart_out_v1, b_out_rs),
        "note": "undirected angle between chart cross-boundary tangent and domain outward direction (soft/G1 only)",
    }
    normal_mismatch = {
        "support_a": _angle_stats(chart_nrm_v0, a_nrm_rs),
        "support_b": _angle_stats(chart_nrm_v1, b_nrm_rs),
        "note": "undirected; chart is NOT forced to follow either visible patch normal",
    }

    # Second-order mismatch proxy: chart S_vv magnitude at boundaries.
    _, _, _, _, _, svv_v0 = surface.evaluate_with_second_derivatives(edge_v0_uv)
    _, _, _, _, _, svv_v1 = surface.evaluate_with_second_derivatives(edge_v1_uv)
    second_order_mismatch = {
        "chart_svv_norm_v0_mean": float(svv_v0.norm(dim=1).mean()),
        "chart_svv_norm_v1_mean": float(svv_v1.norm(dim=1).mean()),
        "note": "chart cross-boundary curvature proxy; NOT intrinsic curvature, diagnostic only",
    }

    # Interior grid Jacobian / orientation / parameter quality.
    g = max(3, int(config.diagnostic_grid))
    gu = torch.linspace(0.0, 1.0, g, dtype=dtype, device=device)
    gv = torch.linspace(0.0, 1.0, g, dtype=dtype, device=device)
    uu, vv = torch.meshgrid(gu, gv, indexing="ij")
    grid_uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
    grid_pts, su, sv = surface.evaluate_with_derivatives(grid_uv)
    jac = compute_parametric_jacobian_metrics(su, sv, eps=eps, scale=max(float(domain_a.local_surface_scale), eps))
    normals = torch.cross(su, sv, dim=1)
    orient = compute_orientation_consistency(normals, eps=eps)

    norm_su = su.norm(dim=1)
    norm_sv = sv.norm(dim=1)
    anisotropy = torch.minimum(norm_su, norm_sv) / torch.maximum(norm_su, norm_sv).clamp_min(eps)
    orthogonality = (su * sv).sum(dim=1).abs() / (norm_su * norm_sv).clamp_min(eps)
    parameter_quality = {
        "anisotropy_mean": float(anisotropy.mean()),
        "anisotropy_min": float(anisotropy.min()),
        "orthogonality_mean": float(orthogonality.mean()),
        "orthogonality_max": float(orthogonality.max()),
    }

    jacobian_diagnostics = {
        k: v for k, v in jac.items()
        if k in {
            "min_area_jacobian", "min_jacobian_singular_value", "min_jacobian_singular_value_normalized",
            "jacobian_condition_mean", "jacobian_condition_p95", "max_jacobian_condition",
            "near_degenerate_count", "sample_count",
        }
    }
    orientation_diagnostics = {
        "orientation_flip_count": int(orient.get("orientation_flip_count", 0)),
        "valid_sample_count": int(orient.get("valid_sample_count", 0)),
    }

    # Cheap self-intersection / penetration proxies (NOT a complete check).
    normal_grid = _unit(normals, eps).reshape(g, g, 3)
    flip_proxy = 0
    for i in range(g - 1):
        row_dot = (normal_grid[i] * normal_grid[i + 1]).sum(dim=1)
        flip_proxy += int((row_dot < 0.0).sum())
    chart_min = grid_pts.min(dim=0).values
    chart_max = grid_pts.max(dim=0).values

    def _aabb_overlap(a_min, a_max, b_min, b_max):
        return bool((a_min <= b_max).all() and (b_min <= a_max).all())

    penetration_overlaps = 0
    for dom in (domain_a, domain_b):
        # source visible patch AABB (t=0 boundary) as a coarse proxy region.
        vis = dom.world[:, 0]
        if _aabb_overlap(chart_min, chart_max, vis.min(dim=0).values, vis.max(dim=0).values):
            penetration_overlaps += 1

    self_intersection_status = {
        "checked": False,
        "proxy_grid_normal_flip_count": flip_proxy,
        "note": "cheap proxy only; complete self-intersection check deferred to Phase F+1",
    }
    visible_surface_penetration_status = {
        "checked": False,
        "proxy_visible_aabb_overlap_count": penetration_overlaps,
        "note": "cheap AABB proxy only; complete penetration check deferred to Phase F+1",
    }

    # Optional analytic-surface tangent/normal diagnostic (never a constraint).
    analytic_note: dict[str, Any] = {"available": False}
    if surfaces_by_patch_id is not None:
        pid_a = int(domain_a.source_patch_id)
        analytic_note = {"available": pid_a in surfaces_by_patch_id, "used_as_constraint": False}

    return {
        "boundary_conformance": boundary_conformance,
        "tangent_mismatch": tangent_mismatch,
        "normal_mismatch": normal_mismatch,
        "second_order_mismatch": second_order_mismatch,
        "jacobian_diagnostics": jacobian_diagnostics,
        "orientation_diagnostics": orientation_diagnostics,
        "parameter_quality": parameter_quality,
        "self_intersection_status": self_intersection_status,
        "visible_surface_penetration_status": visible_surface_penetration_status,
        "analytic_surface_diagnostic": analytic_note,
    }


def _assemble_result(
    candidate: OccludedRegionCandidate, config: OccludedChartFitConfig, chart_id: str, param_fp: str,
    *, surface: Any, topology: str, common_parameter: Any, support_a: Any, support_b: Any,
    connector_start: Any, connector_end: Any, parameter_correspondence: dict[str, Any],
    fit_diagnostics: dict[str, Any], diagnostics: dict[str, Any] | None, state: str, reason: str,
) -> OccludedChartResult:
    empty: dict[str, Any] = {}
    d = diagnostics or {}
    return OccludedChartResult(
        chart_id=chart_id,
        source_candidate_id=candidate.candidate_id,
        supporting_domain_ids=list(candidate.supporting_domain_ids),
        supporting_boundary_ids=list(candidate.supporting_boundary_ids),
        supporting_patch_ids=list(candidate.supporting_patch_ids),
        topology=topology,
        surface=surface,
        common_parameter=common_parameter,
        support_samples_a=support_a,
        support_samples_b=support_b,
        connector_samples_start=connector_start,
        connector_samples_end=connector_end,
        parameter_correspondence=parameter_correspondence,
        constraint_config=config.payload(),
        fit_diagnostics=fit_diagnostics,
        boundary_conformance=d.get("boundary_conformance", dict(empty)),
        tangent_mismatch=d.get("tangent_mismatch", dict(empty)),
        normal_mismatch=d.get("normal_mismatch", dict(empty)),
        second_order_mismatch=d.get("second_order_mismatch", dict(empty)),
        jacobian_diagnostics=d.get("jacobian_diagnostics", dict(empty)),
        orientation_diagnostics=d.get("orientation_diagnostics", dict(empty)),
        parameter_quality=d.get("parameter_quality", dict(empty)),
        self_intersection_status=d.get("self_intersection_status", {"checked": False}),
        visible_surface_penetration_status=d.get("visible_surface_penetration_status", {"checked": False}),
        evidence_consistency=_evidence_consistency(candidate),
        conflict_provenance={"candidate_reason": candidate.reason, "candidate_state": candidate.state},
        state=state,
        reason=reason,
        provenance={
            "candidate_state": candidate.state,
            "solver_run": surface is not None,
            "param_fingerprint": param_fp,
            "analytic_surface_diagnostic": d.get("analytic_surface_diagnostic", {"available": False}),
        },
    )
