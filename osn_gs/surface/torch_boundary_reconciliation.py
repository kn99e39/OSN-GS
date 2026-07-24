from __future__ import annotations

"""Isolated artificial patch-boundary reconciliation prototype.

Callers provide local patch adjacency explicitly. Scale-normalized proximity
and curve overlap decide whether a joint fit is attempted; tangent and normal
agreement are recorded as soft evidence only. Accepted pairs are fit with
shared control variables from the first solve onward and are marked internal
only after a patch-wide Jacobian validity check.
"""

from dataclasses import dataclass, field
from typing import Any

from osn_gs.surface.torch_nurbs import (
    SharedBoundaryConstraint,
    TorchNURBSSurface,
    boundary_control_indices,
    fit_coupled_patch_graph_lsq,
    fit_torch_visible_surface_lsq,
)
from osn_gs.surface.torch_patch_boundary import (
    BOUNDARY_RECONCILED_INTERNAL,
    BOUNDARY_UNCLASSIFIED,
)
from osn_gs.utils.torch_ops import require_torch


@dataclass(frozen=True)
class PatchEdgePair:
    pair_id: str
    patch_a: int
    edge_a: str
    patch_b: int
    edge_b: str
    start_a: int = 0
    stop_a: int | None = None
    start_b: int = 0
    stop_b: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class BoundaryPairEvidence:
    pair_id: str
    patch_a: int
    edge_a: str
    patch_b: int
    edge_b: str
    reverse: bool
    curve_scale: float
    gap_rms: float
    gap_max: float
    normalized_gap_rms: float
    normalized_gap_max: float
    length_ratio: float
    tangent_angle_deg_mean: float
    normal_angle_deg_mean: float
    finite: bool

    def payload(self) -> dict[str, Any]:
        return dict(vars(self))


@dataclass
class PatchReconciliationDecision:
    pair_id: str
    state: str
    reason: str
    pre_fit: BoundaryPairEvidence
    post_fit: BoundaryPairEvidence | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "state": self.state,
            "reason": self.reason,
            "pre_fit": self.pre_fit.payload(),
            "post_fit": None if self.post_fit is None else self.post_fit.payload(),
        }


@dataclass
class PatchReconciliationResult:
    surfaces: list[TorchNURBSSurface]
    uv: list[Any]
    decisions: list[PatchReconciliationDecision]
    constraints: list[SharedBoundaryConstraint]
    used_joint_fit: bool
    jacobian_validity: list[dict[str, Any]]

    def payload(self) -> dict[str, Any]:
        return {
            "used_joint_fit": self.used_joint_fit,
            "constraint_count": len(self.constraints),
            "decisions": [decision.payload() for decision in self.decisions],
            "jacobian_validity": self.jacobian_validity,
        }


def _edge_uv(surface: TorchNURBSSurface, edge: str, sample_count: int) -> Any:
    torch = require_torch()
    t = torch.linspace(
        0.0,
        1.0,
        int(sample_count),
        dtype=surface.control_grid.dtype,
        device=surface.control_grid.device,
    )
    edge = str(edge).lower()
    if edge == "u0":
        return torch.stack((torch.zeros_like(t), t), dim=1)
    if edge == "u1":
        return torch.stack((torch.ones_like(t), t), dim=1)
    if edge == "v0":
        return torch.stack((t, torch.zeros_like(t)), dim=1)
    if edge == "v1":
        return torch.stack((t, torch.ones_like(t)), dim=1)
    raise ValueError(f"Unknown patch edge: {edge!r}")


def _curve_length(points: Any) -> float:
    if int(points.shape[0]) < 2:
        return 0.0
    return float((points[1:] - points[:-1]).norm(dim=1).sum().detach().cpu())


def _angle_mean(a: Any, b: Any, unsigned: bool = False) -> float:
    torch = require_torch()
    cosine = (a * b).sum(dim=1).clamp(-1.0, 1.0)
    if unsigned:
        cosine = cosine.abs()
    return float(torch.rad2deg(torch.arccos(cosine)).mean().detach().cpu())


def evaluate_patch_edge_pair(
    surfaces: list[TorchNURBSSurface],
    pair: PatchEdgePair,
    sample_count: int = 33,
) -> BoundaryPairEvidence:
    """Measure both parameter directions and keep the lower-gap correspondence."""

    torch = require_torch()
    surface_a, surface_b = surfaces[int(pair.patch_a)], surfaces[int(pair.patch_b)]
    uv_a = _edge_uv(surface_a, pair.edge_a, sample_count)
    uv_b = _edge_uv(surface_b, pair.edge_b, sample_count)
    world_a, du_a, dv_a = surface_a.evaluate_with_derivatives(uv_a)
    world_b, du_b, dv_b = surface_b.evaluate_with_derivatives(uv_b)
    tangent_a = dv_a if str(pair.edge_a).lower().startswith("u") else du_a
    tangent_b = dv_b if str(pair.edge_b).lower().startswith("u") else du_b
    tangent_a = torch.nn.functional.normalize(tangent_a, dim=1, eps=1e-12)
    tangent_b = torch.nn.functional.normalize(tangent_b, dim=1, eps=1e-12)
    normal_a = surface_a.normals(uv_a)
    normal_b = surface_b.normals(uv_b)

    direct = (world_a - world_b).norm(dim=1)
    reversed_gap = (world_a - world_b.flip(0)).norm(dim=1)
    reverse = float(reversed_gap.square().mean()) < float(direct.square().mean())
    if reverse:
        world_b = world_b.flip(0)
        tangent_b = -tangent_b.flip(0)
        normal_b = normal_b.flip(0)
    gap = (world_a - world_b).norm(dim=1)
    length_a, length_b = _curve_length(world_a), _curve_length(world_b)
    curve_scale = max(0.5 * (length_a + length_b), 1e-8)
    length_ratio = min(length_a, length_b) / max(length_a, length_b, 1e-8)
    finite = bool(
        torch.isfinite(world_a).all()
        and torch.isfinite(world_b).all()
        and torch.isfinite(gap).all()
    )
    gap_rms = float(gap.square().mean().sqrt().detach().cpu())
    gap_max = float(gap.max().detach().cpu())
    return BoundaryPairEvidence(
        pair_id=pair.pair_id,
        patch_a=int(pair.patch_a),
        edge_a=str(pair.edge_a),
        patch_b=int(pair.patch_b),
        edge_b=str(pair.edge_b),
        reverse=bool(reverse),
        curve_scale=curve_scale,
        gap_rms=gap_rms,
        gap_max=gap_max,
        normalized_gap_rms=gap_rms / curve_scale,
        normalized_gap_max=gap_max / curve_scale,
        length_ratio=length_ratio,
        tangent_angle_deg_mean=_angle_mean(tangent_a, tangent_b),
        normal_angle_deg_mean=_angle_mean(normal_a, normal_b, unsigned=True),
        finite=finite,
    )


def surface_jacobian_validity(
    surface: TorchNURBSSurface,
    resolution: int = 12,
    relative_minimum: float = 1e-5,
) -> dict[str, Any]:
    torch = require_torch()
    t = torch.linspace(0.0, 1.0, int(resolution), dtype=surface.control_grid.dtype, device=surface.control_grid.device)
    u, v = torch.meshgrid(t, t, indexing="ij")
    uv = torch.stack((u.flatten(), v.flatten()), dim=1)
    _, deriv_u, deriv_v = surface.evaluate_with_derivatives(uv)
    cross = torch.cross(deriv_u, deriv_v, dim=1)
    area = cross.norm(dim=1)
    median = area.median().clamp_min(1e-12)
    unit = torch.nn.functional.normalize(cross, dim=1, eps=1e-12)
    reference = unit[torch.argmax(area)]
    flips = int(((unit * reference).sum(dim=1) < 0.0).sum().detach().cpu())
    degenerate = int((area <= median * float(relative_minimum)).sum().detach().cpu())
    finite = bool(torch.isfinite(area).all())
    return {
        "finite": finite,
        "jacobian_min": float(area.min().detach().cpu()),
        "jacobian_median": float(median.detach().cpu()),
        "orientation_flip_count": flips,
        "near_degenerate_count": degenerate,
        "valid": finite and flips == 0 and degenerate == 0,
    }


def fit_reconciled_patch_graph(
    patch_points: list[Any],
    patch_initial_uv: list[Any],
    adjacency: list[PatchEdgePair],
    resolution_u: int = 8,
    resolution_v: int = 6,
    degree_u: int = 2,
    degree_v: int = 2,
    max_normalized_gap: float = 0.05,
    min_length_ratio: float = 0.5,
    sample_count: int = 33,
) -> PatchReconciliationResult:
    """Fit an explicit local patch graph and reconcile geometrically coincident edges.

    Normal and tangent angles are never used as hard gates. If any jointly fit
    patch fails the post-fit Jacobian check, the entire prototype transaction
    rolls back to the independently fitted surfaces.
    """

    if len(patch_points) < 2 or len(patch_points) != len(patch_initial_uv):
        raise ValueError("Patch reconciliation needs matching point/UV lists for at least two patches.")
    independent = [
        fit_torch_visible_surface_lsq(
            patch_points[index],
            resolution_u=resolution_u,
            resolution_v=resolution_v,
            degree_u=degree_u,
            degree_v=degree_v,
            initial_uv=patch_initial_uv[index],
        )
        for index in range(len(patch_points))
    ]
    independent_surfaces = [item[0] for item in independent]
    independent_uv = [item[1] for item in independent]

    decisions: list[PatchReconciliationDecision] = []
    constraints: list[SharedBoundaryConstraint] = []
    selected_pairs: list[PatchEdgePair] = []
    for pair in sorted(adjacency, key=lambda item: item.pair_id):
        evidence = evaluate_patch_edge_pair(independent_surfaces, pair, sample_count=sample_count)
        reason = "joint_fit_candidate"
        if not evidence.finite:
            reason = "non_finite_boundary"
        elif evidence.length_ratio < float(min_length_ratio):
            reason = "insufficient_curve_overlap"
        elif evidence.normalized_gap_rms > float(max_normalized_gap):
            reason = "scale_normalized_gap"
        if reason != "joint_fit_candidate":
            decisions.append(PatchReconciliationDecision(pair.pair_id, BOUNDARY_UNCLASSIFIED, reason, evidence))
            continue
        indices_a = boundary_control_indices(
            resolution_u, resolution_v, pair.edge_a, pair.start_a, pair.stop_a
        )
        indices_b = boundary_control_indices(
            resolution_u, resolution_v, pair.edge_b, pair.start_b, pair.stop_b
        )
        constraints.append(
            SharedBoundaryConstraint(
                patch_a=int(pair.patch_a),
                control_indices_a=indices_a,
                patch_b=int(pair.patch_b),
                control_indices_b=indices_b,
                reverse=evidence.reverse,
                constraint_id=pair.pair_id,
            )
        )
        selected_pairs.append(pair)
        decisions.append(PatchReconciliationDecision(pair.pair_id, BOUNDARY_UNCLASSIFIED, reason, evidence))

    if not constraints:
        validity = [surface_jacobian_validity(surface) for surface in independent_surfaces]
        return PatchReconciliationResult(
            independent_surfaces, independent_uv, decisions, [], False, validity
        )

    coupled = fit_coupled_patch_graph_lsq(
        patch_points,
        patch_initial_uv,
        constraints,
        resolution_u=resolution_u,
        resolution_v=resolution_v,
        degree_u=degree_u,
        degree_v=degree_v,
    )
    coupled_surfaces = [item[0] for item in coupled]
    coupled_uv = [item[1] for item in coupled]
    validity = [surface_jacobian_validity(surface) for surface in coupled_surfaces]
    if not all(item["valid"] for item in validity):
        for decision in decisions:
            if decision.reason == "joint_fit_candidate":
                decision.reason = "post_fit_jacobian_invalid"
        return PatchReconciliationResult(
            independent_surfaces,
            independent_uv,
            decisions,
            [],
            False,
            [surface_jacobian_validity(surface) for surface in independent_surfaces],
        )

    pair_by_id = {pair.pair_id: pair for pair in selected_pairs}
    for decision in decisions:
        if decision.reason != "joint_fit_candidate":
            continue
        decision.post_fit = evaluate_patch_edge_pair(
            coupled_surfaces, pair_by_id[decision.pair_id], sample_count=sample_count
        )
        numerical_tolerance = 1e-5 * max(decision.post_fit.curve_scale, 1.0)
        if decision.post_fit.gap_max <= numerical_tolerance:
            decision.state = BOUNDARY_RECONCILED_INTERNAL
            decision.reason = "joint_shared_control_valid"
        else:
            decision.reason = "post_fit_c0_error"

    if not all(decision.state == BOUNDARY_RECONCILED_INTERNAL for decision in decisions if decision.post_fit is not None):
        for decision in decisions:
            if decision.post_fit is not None:
                decision.state = BOUNDARY_UNCLASSIFIED
                decision.reason = "joint_transaction_rolled_back"
        return PatchReconciliationResult(
            independent_surfaces,
            independent_uv,
            decisions,
            [],
            False,
            [surface_jacobian_validity(surface) for surface in independent_surfaces],
        )
    return PatchReconciliationResult(
        coupled_surfaces, coupled_uv, decisions, constraints, True, validity
    )