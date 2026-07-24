from __future__ import annotations

"""Diagnostics-only local quadratic surface proxies.

This module implements the diagnostics-only Stage 1 recorded in
``docs/worklogs/61_proxy_decomposition_stage1_quadratic_diagnostics.md``. It
does not participate in the production component builder. The proxy is a local
decomposition diagnostic, never a replacement for the final NURBS surface.

Coordinates are normalized by one isotropic tangent-space support scale before
fitting ``z = ax^2 + bxy + cy^2 + dx + ey + f``.  Error increase for a pair is
computed from world-space child SSEs and normalized once by the *merged*
support scale, avoiding subtraction of unrelated per-region RMS values.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.utils.torch_ops import require_torch


@dataclass
class QuadraticSurfaceProxy:
    """One PCA-framed quadratic height-field fit over a local point set."""

    origin: Any  # (3,)
    tangent_u: Any  # (3,)
    tangent_v: Any  # (3,)
    normal: Any  # (3,)
    coefficients: Any  # (6,), normalized z = [x2, xy, y2, x, y, 1] @ coefficients
    support_scale: float
    support_extent: Any  # (2,), world-space tangent ranges
    covariance_eigenvalues: Any  # (3,), ascending
    point_count: int
    effective_weight_sum: float
    world_rms_residual: float
    world_max_residual: float
    normalized_rms_residual: float
    normalized_max_residual: float
    plane_world_rms_residual: float
    plane_normalized_rms_residual: float
    condition_number: float
    planarity_score: float
    local_curvature_proxy: float
    residual_concentration: float
    regularization: float
    valid: bool
    invalid_reason: str | None = None

    def payload(self) -> dict[str, Any]:
        """Return a compact JSON-serializable diagnostic payload."""

        return {
            "point_count": self.point_count,
            "effective_weight_sum": self.effective_weight_sum,
            "origin": self.origin.detach().cpu().tolist(),
            "tangent_u": self.tangent_u.detach().cpu().tolist(),
            "tangent_v": self.tangent_v.detach().cpu().tolist(),
            "normal": self.normal.detach().cpu().tolist(),
            "coefficients": self.coefficients.detach().cpu().tolist(),
            "support_scale": self.support_scale,
            "support_extent": self.support_extent.detach().cpu().tolist(),
            "covariance_eigenvalues": self.covariance_eigenvalues.detach().cpu().tolist(),
            "world_rms_residual": self.world_rms_residual,
            "world_max_residual": self.world_max_residual,
            "normalized_rms_residual": self.normalized_rms_residual,
            "normalized_max_residual": self.normalized_max_residual,
            "plane_world_rms_residual": self.plane_world_rms_residual,
            "plane_normalized_rms_residual": self.plane_normalized_rms_residual,
            "condition_number": self.condition_number,
            "planarity_score": self.planarity_score,
            "local_curvature_proxy": self.local_curvature_proxy,
            "residual_concentration": self.residual_concentration,
            "regularization": self.regularization,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
        }


@dataclass
class ProxyMergeDiagnostics:
    """Independent signals for one possible region merge.

    No weighted admissibility score is defined in Stage 1.  Keeping the
    signals separate is intentional: Stage 2/3 must not hide a scene-specific
    compromise inside one opaque scalar.
    """

    proxy_a: QuadraticSurfaceProxy
    proxy_b: QuadraticSurfaceProxy
    merged_proxy: QuadraticSurfaceProxy
    child_pooled_world_rms: float
    normalized_error_increase: float
    merged_to_child_rms_ratio: float
    support_gap_quantile: float
    scale_normalized_support_gap: float
    sampling_scale: float
    centroid_distance: float
    normal_angle_degrees: float
    normal_change_rate: float
    normal_offset_a: float
    normal_offset_b: float
    layer_separation_score: float
    valid: bool

    def payload(self) -> dict[str, Any]:
        return {
            "proxy_a": self.proxy_a.payload(),
            "proxy_b": self.proxy_b.payload(),
            "merged_proxy": self.merged_proxy.payload(),
            "child_pooled_world_rms": self.child_pooled_world_rms,
            "normalized_error_increase": self.normalized_error_increase,
            "merged_to_child_rms_ratio": self.merged_to_child_rms_ratio,
            "support_gap_quantile": self.support_gap_quantile,
            "scale_normalized_support_gap": self.scale_normalized_support_gap,
            "sampling_scale": self.sampling_scale,
            "centroid_distance": self.centroid_distance,
            "normal_angle_degrees": self.normal_angle_degrees,
            "normal_change_rate": self.normal_change_rate,
            "normal_offset_a": self.normal_offset_a,
            "normal_offset_b": self.normal_offset_b,
            "layer_separation_score": self.layer_separation_score,
            "valid": self.valid,
        }


def _canonicalize_axis(axis: Any) -> Any:
    """Resolve eigenvector sign ambiguity by its largest-magnitude element."""

    index = int(axis.abs().argmax())
    return -axis if float(axis[index]) < 0.0 else axis


def _validate_points_and_weights(points: Any, weights: Any | None) -> tuple[Any, Any]:
    torch = require_torch()
    if points.ndim != 2 or tuple(points.shape[1:]) != (3,):
        raise ValueError(f"points must have shape (N, 3), got {tuple(points.shape)}")
    work_points = points.detach().to(dtype=torch.float64)
    if weights is None:
        work_weights = torch.ones(
            (int(points.shape[0]),), dtype=work_points.dtype, device=work_points.device
        )
    else:
        if weights.ndim != 1 or int(weights.shape[0]) != int(points.shape[0]):
            raise ValueError("weights must have shape (N,) matching points")
        work_weights = weights.detach().to(dtype=work_points.dtype, device=work_points.device)
    if not bool(torch.isfinite(work_points).all()):
        raise ValueError("points contain non-finite values")
    if not bool(torch.isfinite(work_weights).all()) or bool((work_weights < 0).any()):
        raise ValueError("weights must be finite and non-negative")
    if float(work_weights.sum()) <= 0.0:
        raise ValueError("weights must have a positive sum")
    return work_points, work_weights


def _invalid_proxy(
    points: Any,
    weights: Any,
    reason: str,
    regularization: float,
    origin: Any | None = None,
    tangent_u: Any | None = None,
    tangent_v: Any | None = None,
    normal: Any | None = None,
    eigenvalues: Any | None = None,
) -> QuadraticSurfaceProxy:
    torch = require_torch()
    dtype, device = points.dtype, points.device
    zero3 = torch.zeros((3,), dtype=dtype, device=device)
    origin = points.mean(dim=0) if origin is None and int(points.shape[0]) else (origin if origin is not None else zero3)
    tangent_u = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device) if tangent_u is None else tangent_u
    tangent_v = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device) if tangent_v is None else tangent_v
    normal = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device) if normal is None else normal
    eigenvalues = torch.zeros((3,), dtype=dtype, device=device) if eigenvalues is None else eigenvalues
    return QuadraticSurfaceProxy(
        origin=origin,
        tangent_u=tangent_u,
        tangent_v=tangent_v,
        normal=normal,
        coefficients=torch.zeros((6,), dtype=dtype, device=device),
        support_scale=0.0,
        support_extent=torch.zeros((2,), dtype=dtype, device=device),
        covariance_eigenvalues=eigenvalues,
        point_count=int(points.shape[0]),
        effective_weight_sum=float(weights.sum()) if int(weights.numel()) else 0.0,
        world_rms_residual=float("inf"),
        world_max_residual=float("inf"),
        normalized_rms_residual=float("inf"),
        normalized_max_residual=float("inf"),
        plane_world_rms_residual=float("inf"),
        plane_normalized_rms_residual=float("inf"),
        condition_number=float("inf"),
        planarity_score=0.0,
        local_curvature_proxy=float("inf"),
        residual_concentration=float("inf"),
        regularization=float(regularization),
        valid=False,
        invalid_reason=reason,
    )


def fit_quadratic_surface_proxy(
    points: Any,
    weights: Any | None = None,
    regularization: float = 1e-6,
    min_points: int = 6,
    degeneracy_ratio: float = 1e-8,
    max_condition_number: float = 1e10,
) -> QuadraticSurfaceProxy:
    """Fit a regularized quadratic height field in a normalized PCA frame."""

    torch = require_torch()
    work_points, work_weights = _validate_points_and_weights(points, weights)
    if int(work_points.shape[0]) < int(min_points):
        return _invalid_proxy(
            work_points, work_weights, "insufficient_support", regularization
        )

    with torch.no_grad():
        weight_sum = work_weights.sum()
        normalized_weights = work_weights / weight_sum
        origin = (work_points * normalized_weights[:, None]).sum(dim=0)
        centered = work_points - origin
        covariance = centered.T @ (normalized_weights[:, None] * centered)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        largest = max(float(eigenvalues[-1]), 1e-15)
        if float(eigenvalues[1]) <= float(degeneracy_ratio) * largest:
            return _invalid_proxy(
                work_points,
                work_weights,
                "degenerate_tangent_support",
                regularization,
                origin=origin,
                eigenvalues=eigenvalues,
            )

        normal = _canonicalize_axis(eigenvectors[:, 0])
        tangent_u = _canonicalize_axis(eigenvectors[:, 2])
        tangent_v = torch.cross(normal, tangent_u, dim=0)
        tangent_v = tangent_v / torch.linalg.norm(tangent_v).clamp_min(1e-15)
        # Recompute the normal so {u,v,n} is right-handed after sign fixing.
        normal = torch.cross(tangent_u, tangent_v, dim=0)
        normal = normal / torch.linalg.norm(normal).clamp_min(1e-15)

        local_x = centered @ tangent_u
        local_y = centered @ tangent_v
        local_z = centered @ normal
        tangent_radius_sq = local_x.square() + local_y.square()
        support_scale = float((normalized_weights * tangent_radius_sq).sum().sqrt())
        if support_scale <= 1e-12:
            return _invalid_proxy(
                work_points,
                work_weights,
                "zero_support_scale",
                regularization,
                origin=origin,
                tangent_u=tangent_u,
                tangent_v=tangent_v,
                normal=normal,
                eigenvalues=eigenvalues,
            )

        x = local_x / support_scale
        y = local_y / support_scale
        z = local_z / support_scale
        design = torch.stack(
            [x.square(), x * y, y.square(), x, y, torch.ones_like(x)], dim=1
        )
        sqrt_weights = normalized_weights.sqrt()
        weighted_design = sqrt_weights[:, None] * design
        singular_values = torch.linalg.svdvals(weighted_design)
        min_singular = float(singular_values[-1])
        condition_number = (
            float(singular_values[0]) / min_singular
            if min_singular > 1e-15
            else float("inf")
        )

        penalty = torch.diag(
            torch.tensor([1.0, 1.0, 1.0, 0.1, 0.1, 0.0], dtype=design.dtype, device=design.device)
        )
        normal_matrix = design.T @ (normalized_weights[:, None] * design)
        normal_matrix = normal_matrix + float(regularization) * penalty
        rhs = design.T @ (normalized_weights * z)
        try:
            coefficients = torch.linalg.solve(normal_matrix, rhs)
        except RuntimeError:
            coefficients = torch.linalg.lstsq(normal_matrix, rhs[:, None]).solution[:, 0]

        predicted = design @ coefficients
        normalized_residual = z - predicted
        normalized_mse = float((normalized_weights * normalized_residual.square()).sum())
        normalized_rms = normalized_mse ** 0.5
        normalized_max = float(normalized_residual.abs().max())
        plane_normalized_rms = float((normalized_weights * z.square()).sum().sqrt())
        world_rms = normalized_rms * support_scale
        world_max = normalized_max * support_scale
        support_extent = torch.stack(
            [local_x.max() - local_x.min(), local_y.max() - local_y.min()]
        )
        mid_eigenvalue = max(float(eigenvalues[1]), 1e-15)
        planarity_score = 1.0 - min(float(eigenvalues[0]) / mid_eigenvalue, 1.0)
        a, b, c = coefficients[:3]
        hessian = torch.stack(
            [torch.stack([2.0 * a, b]), torch.stack([b, 2.0 * c])]
        )
        curvature_proxy = float(torch.linalg.norm(hessian))
        concentration = normalized_max / max(normalized_rms, 1e-15)
        valid = bool(torch.isfinite(coefficients).all()) and condition_number <= float(max_condition_number)
        invalid_reason = None if valid else "ill_conditioned_fit"

        return QuadraticSurfaceProxy(
            origin=origin,
            tangent_u=tangent_u,
            tangent_v=tangent_v,
            normal=normal,
            coefficients=coefficients,
            support_scale=support_scale,
            support_extent=support_extent,
            covariance_eigenvalues=eigenvalues,
            point_count=int(work_points.shape[0]),
            effective_weight_sum=float(weight_sum),
            world_rms_residual=world_rms,
            world_max_residual=world_max,
            normalized_rms_residual=normalized_rms,
            normalized_max_residual=normalized_max,
            plane_world_rms_residual=plane_normalized_rms * support_scale,
            plane_normalized_rms_residual=plane_normalized_rms,
            condition_number=condition_number,
            planarity_score=planarity_score,
            local_curvature_proxy=curvature_proxy,
            residual_concentration=concentration,
            regularization=float(regularization),
            valid=valid,
            invalid_reason=invalid_reason,
        )


def evaluate_quadratic_proxy(proxy: QuadraticSurfaceProxy, points: Any) -> Any:
    """Project points to the proxy at their tangent coordinates."""

    centered = points.to(dtype=proxy.origin.dtype, device=proxy.origin.device) - proxy.origin
    local_x = centered @ proxy.tangent_u
    local_y = centered @ proxy.tangent_v
    x = local_x / max(proxy.support_scale, 1e-15)
    y = local_y / max(proxy.support_scale, 1e-15)
    design = require_torch().stack(
        [x.square(), x * y, y.square(), x, y, require_torch().ones_like(x)], dim=1
    )
    local_z = (design @ proxy.coefficients) * proxy.support_scale
    return (
        proxy.origin
        + local_x[:, None] * proxy.tangent_u
        + local_y[:, None] * proxy.tangent_v
        + local_z[:, None] * proxy.normal
    )


def quadratic_proxy_signed_residuals(proxy: QuadraticSurfaceProxy, points: Any) -> Any:
    """Signed normal-direction residual to ``proxy`` in world units."""

    projected = evaluate_quadratic_proxy(proxy, points)
    work_points = points.to(dtype=proxy.origin.dtype, device=proxy.origin.device)
    return ((work_points - projected) * proxy.normal).sum(dim=1)


def _median_nn_spacing(points: Any) -> float:
    torch = require_torch()
    count = int(points.shape[0])
    if count < 2:
        return 0.0
    distances = torch.cdist(points, points)
    distances.fill_diagonal_(float("inf"))
    return float(distances.min(dim=1).values.median())


def merge_proxy_diagnostics(
    points_a: Any,
    points_b: Any,
    weights_a: Any | None = None,
    weights_b: Any | None = None,
    regularization: float = 1e-6,
    support_gap_quantile: float = 0.02,
) -> ProxyMergeDiagnostics:
    """Measure independent geometric signals for a possible region merge."""

    torch = require_torch()
    if not 0.0 <= float(support_gap_quantile) <= 1.0:
        raise ValueError("support_gap_quantile must be in [0, 1]")
    work_a, work_weights_a = _validate_points_and_weights(points_a, weights_a)
    work_b, work_weights_b = _validate_points_and_weights(points_b, weights_b)
    proxy_a = fit_quadratic_surface_proxy(work_a, work_weights_a, regularization)
    proxy_b = fit_quadratic_surface_proxy(work_b, work_weights_b, regularization)
    merged_points = torch.cat([work_a, work_b], dim=0)
    merged_weights = torch.cat([work_weights_a, work_weights_b], dim=0)
    merged_proxy = fit_quadratic_surface_proxy(
        merged_points, merged_weights, regularization
    )

    weight_a = float(work_weights_a.sum())
    weight_b = float(work_weights_b.sum())
    total_weight = max(weight_a + weight_b, 1e-15)
    child_world_mse = (
        weight_a * proxy_a.world_rms_residual ** 2
        + weight_b * proxy_b.world_rms_residual ** 2
    ) / total_weight
    child_pooled_world_rms = child_world_mse ** 0.5
    merged_scale_sq = max(merged_proxy.support_scale ** 2, 1e-30)
    normalized_error_increase = (
        merged_proxy.world_rms_residual ** 2 - child_world_mse
    ) / merged_scale_sq
    merged_to_child_ratio = merged_proxy.world_rms_residual / max(
        child_pooled_world_rms, 1e-15
    )

    cross_distances = torch.cdist(work_a, work_b)
    nearest_a = cross_distances.min(dim=1).values
    nearest_b = cross_distances.min(dim=0).values
    gap_a = float(torch.quantile(nearest_a, float(support_gap_quantile)))
    gap_b = float(torch.quantile(nearest_b, float(support_gap_quantile)))
    support_gap = max(gap_a, gap_b)
    sampling_scale = max(
        _median_nn_spacing(work_a), _median_nn_spacing(work_b), 1e-12
    )
    normalized_gap = support_gap / sampling_scale

    centroid_delta = proxy_b.origin - proxy_a.origin
    centroid_distance = float(torch.linalg.norm(centroid_delta))
    normal_dot = float((proxy_a.normal @ proxy_b.normal).abs().clamp(0.0, 1.0))
    normal_angle_radians = float(torch.arccos(torch.tensor(normal_dot, dtype=torch.float64)))
    normal_angle_degrees = float(torch.rad2deg(torch.tensor(normal_angle_radians)))
    normal_change_rate = normal_angle_radians / max(centroid_distance, sampling_scale)
    signed_offset_a = float(centroid_delta @ proxy_a.normal)
    signed_offset_b = float(centroid_delta @ proxy_b.normal)
    normal_offset_a = abs(signed_offset_a)
    normal_offset_b = abs(signed_offset_b)
    tangent_distance_a = float(
        torch.linalg.norm(centroid_delta - signed_offset_a * proxy_a.normal)
    )
    tangent_distance_b = float(
        torch.linalg.norm(centroid_delta - signed_offset_b * proxy_b.normal)
    )
    # A true parallel layer displacement is predominantly normal to both
    # proxies. Smooth neighbors are separated mainly along their tangent
    # directions even when curvature gives them a non-zero normal offset.
    layer_separation_score = normal_dot * min(
        normal_offset_a / max(tangent_distance_a, sampling_scale),
        normal_offset_b / max(tangent_distance_b, sampling_scale),
    )

    return ProxyMergeDiagnostics(
        proxy_a=proxy_a,
        proxy_b=proxy_b,
        merged_proxy=merged_proxy,
        child_pooled_world_rms=child_pooled_world_rms,
        normalized_error_increase=normalized_error_increase,
        merged_to_child_rms_ratio=merged_to_child_ratio,
        support_gap_quantile=support_gap,
        scale_normalized_support_gap=normalized_gap,
        sampling_scale=sampling_scale,
        centroid_distance=centroid_distance,
        normal_angle_degrees=normal_angle_degrees,
        normal_change_rate=normal_change_rate,
        normal_offset_a=normal_offset_a,
        normal_offset_b=normal_offset_b,
        layer_separation_score=layer_separation_score,
        valid=proxy_a.valid and proxy_b.valid and merged_proxy.valid,
    )


__all__ = [
    "ProxyMergeDiagnostics",
    "QuadraticSurfaceProxy",
    "evaluate_quadratic_proxy",
    "fit_quadratic_surface_proxy",
    "merge_proxy_diagnostics",
    "quadratic_proxy_signed_residuals",
]
