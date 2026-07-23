from __future__ import annotations

"""Gaussian-native support continuity diagnostics.

This module is the diagnostics-only Stage 3-R investigation.  It is not
imported by the production component builder or by Stage 3 agglomeration.
Every returned quantity is an independent raw signal; no merge decision,
weighted score, or ground-truth input exists here.
"""

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from osn_gs.surface.torch_surface_proxy import merge_proxy_diagnostics
from osn_gs.utils.torch_ops import require_torch


@dataclass(frozen=True)
class GaussianSupportContinuityConfig:
    covariance_eigenvalue_floor: float = 1e-10
    covariance_relative_eigenvalue_floor: float = 1e-6
    ellipsoid_sigma_factors: tuple[float, ...] = (1.0, 2.0, 3.0)
    support_quantiles: tuple[float, ...] = (0.02, 0.1, 0.5)
    bridge_sample_count: int = 33
    kernel_truncation_radius: float = 4.0
    opacity_weighting_modes: tuple[bool, ...] = (False, True)
    boundary_facing_selection_quantile: float = 0.1
    max_boundary_pairs: int = 32
    tangent_normal_projection_mode: str = "covariance_principal_axis"
    proxy_regularization: float = 1e-6
    dtype: str = "float64"

    def __post_init__(self) -> None:
        for name in (
            "covariance_eigenvalue_floor",
            "covariance_relative_eigenvalue_floor",
            "kernel_truncation_radius",
            "proxy_regularization",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if int(self.bridge_sample_count) < 3:
            raise ValueError("bridge_sample_count must be >= 3")
        if int(self.max_boundary_pairs) < 1:
            raise ValueError("max_boundary_pairs must be >= 1")
        if not 0.0 < float(self.boundary_facing_selection_quantile) <= 1.0:
            raise ValueError("boundary_facing_selection_quantile must be in (0, 1]")
        if not self.ellipsoid_sigma_factors or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in self.ellipsoid_sigma_factors
        ):
            raise ValueError("ellipsoid_sigma_factors must be finite and positive")
        if not self.support_quantiles or any(
            not 0.0 <= float(value) <= 1.0 for value in self.support_quantiles
        ):
            raise ValueError("support_quantiles must be in [0, 1]")
        if not self.opacity_weighting_modes:
            raise ValueError("opacity_weighting_modes must not be empty")
        if self.tangent_normal_projection_mode != "covariance_principal_axis":
            raise ValueError(
                "only covariance_principal_axis projection is implemented"
            )
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")

    def payload(self) -> dict[str, Any]:
        return {
            "covariance_eigenvalue_floor": float(self.covariance_eigenvalue_floor),
            "covariance_relative_eigenvalue_floor": float(
                self.covariance_relative_eigenvalue_floor
            ),
            "ellipsoid_sigma_factors": [
                float(value) for value in self.ellipsoid_sigma_factors
            ],
            "support_quantiles": [float(value) for value in self.support_quantiles],
            "bridge_sample_count": int(self.bridge_sample_count),
            "kernel_truncation_radius": float(self.kernel_truncation_radius),
            "opacity_weighting_modes": [
                bool(value) for value in self.opacity_weighting_modes
            ],
            "boundary_facing_selection_quantile": float(
                self.boundary_facing_selection_quantile
            ),
            "max_boundary_pairs": int(self.max_boundary_pairs),
            "tangent_normal_projection_mode": self.tangent_normal_projection_mode,
            "proxy_regularization": float(self.proxy_regularization),
            "dtype": self.dtype,
        }


@dataclass
class GaussianSupportContinuityDiagnostics:
    config: GaussianSupportContinuityConfig
    region_a_indices: list[int]
    region_b_indices: list[int]
    validity_flags: dict[str, Any]
    covariance_diagnostics: dict[str, Any]
    existing_point_diagnostics: dict[str, Any]
    mahalanobis: dict[str, Any]
    ellipsoid_overlap: dict[str, Any]
    projected_reach: dict[str, Any]
    bridge_density: dict[str, Any]
    facing_support: dict[str, Any]
    boundary_pair_metrics: list[dict[str, Any]]
    computational_cost: dict[str, int]

    def payload(self) -> dict[str, Any]:
        return _json_safe(
            {
                "schema_version": 1,
                "stage": "gaussian_native_support_continuity_stage3r",
                "production_membership_changed": False,
                "config": self.config.payload(),
                "region_a_indices": list(self.region_a_indices),
                "region_b_indices": list(self.region_b_indices),
                "validity_flags": self.validity_flags,
                "covariance_diagnostics": self.covariance_diagnostics,
                "existing_point_diagnostics": self.existing_point_diagnostics,
                "mahalanobis": self.mahalanobis,
                "ellipsoid_overlap": self.ellipsoid_overlap,
                "projected_reach": self.projected_reach,
                "bridge_density": self.bridge_density,
                "facing_support": self.facing_support,
                "boundary_pair_metrics": self.boundary_pair_metrics,
                "computational_cost": self.computational_cost,
            }
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _coerce_config(
    config: GaussianSupportContinuityConfig | Mapping[str, Any] | None,
) -> GaussianSupportContinuityConfig:
    if config is None:
        return GaussianSupportContinuityConfig()
    if isinstance(config, GaussianSupportContinuityConfig):
        return config
    if isinstance(config, Mapping):
        return GaussianSupportContinuityConfig(**dict(config))
    raise TypeError("config must be GaussianSupportContinuityConfig, mapping, or None")


def _dtype(config: GaussianSupportContinuityConfig) -> Any:
    torch = require_torch()
    return torch.float64 if config.dtype == "float64" else torch.float32


def covariance_from_scale_rotation(scales: Any, rotations: Any, dtype: str = "float64") -> Any:
    """Convert 3DGS scale + WXYZ quaternion to world covariance.

    The multiplication order matches the vendored rasterizer's ``M=S*R`` and
    ``Sigma=M^T*M`` convention.
    """

    torch = require_torch()
    work_dtype = torch.float64 if dtype == "float64" else torch.float32
    scales = torch.as_tensor(scales, dtype=work_dtype)
    rotations = torch.as_tensor(rotations, dtype=work_dtype, device=scales.device)
    if scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError("scales must have shape (N, 3)")
    if rotations.shape != (scales.shape[0], 4):
        raise ValueError("rotations must have shape (N, 4)")
    rotations = torch.nn.functional.normalize(rotations, dim=1)
    r, x, y, z = rotations.unbind(dim=1)
    matrix = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - r * z),
            2 * (x * z + r * y),
            2 * (x * y + r * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - r * x),
            2 * (x * z - r * y),
            2 * (y * z + r * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=1,
    ).reshape(-1, 3, 3)
    scaled = torch.diag_embed(scales) @ matrix
    return scaled.transpose(1, 2) @ scaled


def _indices(values: Any, count: int, device: Any) -> Any:
    torch = require_torch()
    result = torch.as_tensor(values, dtype=torch.long, device=device).reshape(-1)
    if result.numel() == 0:
        raise ValueError("regions must contain at least one Gaussian")
    result = torch.unique(result, sorted=True)
    if int(result.min()) < 0 or int(result.max()) >= count:
        raise ValueError("region index is outside Gaussian arrays")
    return result


def _quantile_payload(values: Any, quantiles: Sequence[float]) -> dict[str, Any]:
    torch = require_torch()
    flat = values.reshape(-1)
    if flat.numel() == 0:
        return {"count": 0, "values": [], "quantiles": {}}
    return {
        "count": int(flat.numel()),
        "minimum": float(flat.min()),
        "maximum": float(flat.max()),
        "mean": float(flat.mean()),
        "median": float(flat.median()),
        "quantiles": {
            f"q{float(q):g}": float(torch.quantile(flat, float(q)))
            for q in quantiles
        },
        "values": [float(value) for value in flat],
    }


def _median_spacing(points: Any) -> float:
    torch = require_torch()
    if int(points.shape[0]) < 2:
        return 0.0
    distances = torch.cdist(points, points)
    distances.fill_diagonal_(float("inf"))
    return float(distances.min(dim=1).values.median())


def _regularize_covariances(covariances: Any, config: GaussianSupportContinuityConfig):
    torch = require_torch()
    symmetric = 0.5 * (covariances + covariances.transpose(1, 2))
    raw_values, vectors = torch.linalg.eigh(symmetric)
    relative_floor = raw_values[:, -1:].clamp_min(0.0) * float(
        config.covariance_relative_eigenvalue_floor
    )
    floor = torch.maximum(
        torch.full_like(relative_floor, float(config.covariance_eigenvalue_floor)),
        relative_floor,
    )
    values = torch.maximum(raw_values, floor)
    regularized = vectors @ torch.diag_embed(values) @ vectors.transpose(1, 2)
    inverse = vectors @ torch.diag_embed(values.reciprocal()) @ vectors.transpose(1, 2)
    condition = values[:, -1] / values[:, 0]
    axis_confidence = (values[:, 1] - values[:, 0]) / values[:, -1].clamp_min(
        float(config.covariance_eigenvalue_floor)
    )
    floored = (raw_values < floor).any(dim=1)
    return regularized, inverse, values, vectors, condition, axis_confidence, floored


def _boundary_pairs(points_a: Any, points_b: Any, indices_a: Any, indices_b: Any, config: GaussianSupportContinuityConfig):
    torch = require_torch()
    distances = torch.cdist(points_a, points_b)
    nearest_b_distance, nearest_b = distances.min(dim=1)
    nearest_a_distance, nearest_a = distances.min(dim=0)
    records: dict[tuple[int, int], float] = {}
    for local_a in range(int(points_a.shape[0])):
        local_b = int(nearest_b[local_a])
        records[(local_a, local_b)] = float(nearest_b_distance[local_a])
    for local_b in range(int(points_b.shape[0])):
        local_a = int(nearest_a[local_b])
        key = (local_a, local_b)
        records[key] = min(records.get(key, float("inf")), float(nearest_a_distance[local_b]))
    ordered = sorted(
        ((distance, local_a, local_b) for (local_a, local_b), distance in records.items()),
        key=lambda item: (item[0], int(indices_a[item[1]]), int(indices_b[item[2]])),
    )
    all_distances = torch.tensor([item[0] for item in ordered], dtype=points_a.dtype, device=points_a.device)
    cutoff = float(
        torch.quantile(all_distances, float(config.boundary_facing_selection_quantile))
    )
    selected = [item for item in ordered if item[0] <= cutoff + 1e-15]
    selected = selected[: int(config.max_boundary_pairs)] or ordered[:1]
    local_a = torch.tensor([item[1] for item in selected], dtype=torch.long, device=points_a.device)
    local_b = torch.tensor([item[2] for item in selected], dtype=torch.long, device=points_a.device)
    return distances, local_a, local_b, cutoff, len(ordered)


def _kernel_density(samples: Any, means: Any, inverse_covariances: Any, opacities: Any, truncation: float):
    torch = require_torch()
    delta = samples[:, None, :] - means[None, :, :]
    mahalanobis_sq = torch.einsum("smk,mkl,sml->sm", delta, inverse_covariances, delta)
    kernels = torch.exp(-0.5 * mahalanobis_sq)
    kernels = torch.where(mahalanobis_sq <= float(truncation) ** 2, kernels, torch.zeros_like(kernels))
    return kernels @ opacities


def evaluate_gaussian_support_continuity(
    region_a: Any,
    region_b: Any,
    gaussian_means: Any,
    gaussian_covariances: Any,
    gaussian_opacities: Any,
    config: GaussianSupportContinuityConfig | Mapping[str, Any] | None = None,
) -> GaussianSupportContinuityDiagnostics:
    """Evaluate independent Gaussian-native continuity signals for two regions.

    Region arguments are Gaussian index collections.  No scene name, GT label,
    topology, component count, or merge/admissibility state is accepted.
    """

    torch = require_torch()
    resolved = _coerce_config(config)
    work_dtype = _dtype(resolved)
    means = torch.as_tensor(gaussian_means, dtype=work_dtype)
    covariances = torch.as_tensor(
        gaussian_covariances, dtype=work_dtype, device=means.device
    )
    opacities = torch.as_tensor(
        gaussian_opacities, dtype=work_dtype, device=means.device
    ).reshape(-1)
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("gaussian_means must have shape (N, 3)")
    count = int(means.shape[0])
    if covariances.shape != (count, 3, 3):
        raise ValueError("gaussian_covariances must have shape (N, 3, 3)")
    if opacities.shape != (count,):
        raise ValueError("gaussian_opacities must have shape (N,) or (N, 1)")
    if not bool(torch.isfinite(means).all()):
        raise ValueError("gaussian_means must be finite")
    if not bool(torch.isfinite(covariances).all()):
        raise ValueError("gaussian_covariances must be finite")
    if not bool(torch.isfinite(opacities).all()):
        raise ValueError("gaussian_opacities must be finite")
    if bool((opacities < 0.0).any()):
        raise ValueError("gaussian_opacities must be non-negative")

    indices_a = _indices(region_a, count, means.device)
    indices_b = _indices(region_b, count, means.device)
    if bool(torch.isin(indices_a, indices_b).any()):
        raise ValueError("region_a and region_b must be disjoint")
    points_a, points_b = means[indices_a], means[indices_b]

    regularized, inverse, eigenvalues, eigenvectors, condition, axis_confidence, floored = _regularize_covariances(
        covariances, resolved
    )
    cross_distances, local_a, local_b, facing_cutoff, directed_pair_count = _boundary_pairs(
        points_a, points_b, indices_a, indices_b, resolved
    )
    global_a, global_b = indices_a[local_a], indices_b[local_b]
    delta = means[global_b] - means[global_a]
    distance = torch.linalg.norm(delta, dim=1).clamp_min(1e-15)
    direction = delta / distance[:, None]
    cov_a, cov_b = regularized[global_a], regularized[global_b]
    inv_a, inv_b = inverse[global_a], inverse[global_b]
    one_a = torch.sqrt(torch.einsum("pi,pij,pj->p", delta, inv_a, delta).clamp_min(0.0))
    one_b = torch.sqrt(torch.einsum("pi,pij,pj->p", delta, inv_b, delta).clamp_min(0.0))
    pooled_inverse = torch.linalg.inv(cov_a + cov_b)
    pooled = torch.sqrt(torch.einsum("pi,pij,pj->p", delta, pooled_inverse, delta).clamp_min(0.0))
    symmetric_mean = 0.5 * (one_a + one_b)
    symmetric_max = torch.maximum(one_a, one_b)

    sigma_a = torch.sqrt(torch.einsum("pi,pij,pj->p", direction, cov_a, direction).clamp_min(0.0))
    sigma_b = torch.sqrt(torch.einsum("pi,pij,pj->p", direction, cov_b, direction).clamp_min(0.0))
    reach_sum = (sigma_a + sigma_b).clamp_min(1e-15)
    reach_ratio = distance / reach_sum

    normal_a = eigenvectors[global_a, :, 0]
    normal_b = eigenvectors[global_b, :, 0]
    signs = torch.where((normal_a * normal_b).sum(dim=1, keepdim=True) < 0.0, -1.0, 1.0)
    average_normal = torch.nn.functional.normalize(normal_a + signs * normal_b, dim=1)
    normal_gap = (delta * average_normal).sum(dim=1).abs()
    tangent_vector = delta - (delta * average_normal).sum(dim=1, keepdim=True) * average_normal
    tangent_gap = torch.linalg.norm(tangent_vector, dim=1)
    tangent_direction = tangent_vector / tangent_gap.clamp_min(1e-15)[:, None]
    tangent_sigma_a = torch.sqrt(torch.einsum("pi,pij,pj->p", tangent_direction, cov_a, tangent_direction).clamp_min(0.0))
    tangent_sigma_b = torch.sqrt(torch.einsum("pi,pij,pj->p", tangent_direction, cov_b, tangent_direction).clamp_min(0.0))
    normal_sigma_a = torch.sqrt(torch.einsum("pi,pij,pj->p", average_normal, cov_a, average_normal).clamp_min(0.0))
    normal_sigma_b = torch.sqrt(torch.einsum("pi,pij,pj->p", average_normal, cov_b, average_normal).clamp_min(0.0))
    tangent_reach_ratio = tangent_gap / (tangent_sigma_a + tangent_sigma_b).clamp_min(1e-15)
    normal_reach_ratio = normal_gap / (normal_sigma_a + normal_sigma_b).clamp_min(1e-15)

    quantiles = resolved.support_quantiles
    ellipsoid: dict[str, Any] = {}
    pair_rows: list[dict[str, Any]] = []
    for pair_index in range(int(global_a.numel())):
        pair_rows.append(
            {
                "gaussian_a": int(global_a[pair_index]),
                "gaussian_b": int(global_b[pair_index]),
                "euclidean_distance": float(distance[pair_index]),
                "one_sided_mahalanobis_a": float(one_a[pair_index]),
                "one_sided_mahalanobis_b": float(one_b[pair_index]),
                "symmetric_mahalanobis_mean": float(symmetric_mean[pair_index]),
                "pooled_mahalanobis": float(pooled[pair_index]),
                "directional_sigma_a": float(sigma_a[pair_index]),
                "directional_sigma_b": float(sigma_b[pair_index]),
                "center_gap_over_directional_reach": float(reach_ratio[pair_index]),
                "tangent_gap": float(tangent_gap[pair_index]),
                "normal_gap": float(normal_gap[pair_index]),
                "tangent_reach_ratio": float(tangent_reach_ratio[pair_index]),
                "normal_reach_ratio": float(normal_reach_ratio[pair_index]),
            }
        )
    for sigma_factor in resolved.ellipsoid_sigma_factors:
        margin = float(sigma_factor) * reach_sum - distance
        key = f"k{float(sigma_factor):g}"
        ellipsoid[key] = {
            "sigma_factor": float(sigma_factor),
            "signed_overlap_margin": _quantile_payload(margin, quantiles),
            "normalized_overlap_margin": _quantile_payload(
                margin / reach_sum, quantiles
            ),
            "overlap_fraction": float((margin >= 0.0).to(work_dtype).mean()),
        }
        for pair_index, row in enumerate(pair_rows):
            row[f"ellipsoid_overlap_margin_{key}"] = float(margin[pair_index])

    support_indices = torch.cat([indices_a, indices_b])
    support_means = means[support_indices]
    support_inverse = inverse[support_indices]
    support_opacity = opacities[support_indices]
    bridge_by_mode: dict[str, Any] = {}
    bridge_kernel_evaluations = 0
    sample_t = torch.linspace(
        0.0, 1.0, int(resolved.bridge_sample_count), dtype=work_dtype, device=means.device
    )
    for opacity_weighted in resolved.opacity_weighting_modes:
        weights = support_opacity if opacity_weighted else torch.ones_like(support_opacity)
        records = []
        for pair_index in range(int(global_a.numel())):
            start, end = means[global_a[pair_index]], means[global_b[pair_index]]
            samples = start[None, :] + sample_t[:, None] * (end - start)[None, :]
            density = _kernel_density(
                samples,
                support_means,
                support_inverse,
                weights,
                float(resolved.kernel_truncation_radius),
            )
            endpoint_reference = torch.minimum(density[0], density[-1]).clamp_min(1e-15)
            minimum = density.min()
            records.append(
                {
                    "gaussian_a": int(global_a[pair_index]),
                    "gaussian_b": int(global_b[pair_index]),
                    "minimum": float(minimum),
                    "mean": float(density.mean()),
                    "integral_t": float(torch.trapezoid(density, sample_t)),
                    "endpoint_minimum_ratio": float(minimum / endpoint_reference),
                    "normalized_valley_depth": float(1.0 - minimum / endpoint_reference),
                    "samples": [float(value) for value in density],
                }
            )
            bridge_kernel_evaluations += int(samples.shape[0] * support_means.shape[0])
        mode_key = "opacity_weighted" if opacity_weighted else "unweighted"
        bridge_by_mode[mode_key] = {
            "pair_count": len(records),
            "minimum": _quantile_payload(
                torch.tensor([item["minimum"] for item in records], dtype=work_dtype), quantiles
            ),
            "mean": _quantile_payload(
                torch.tensor([item["mean"] for item in records], dtype=work_dtype), quantiles
            ),
            "integral_t": _quantile_payload(
                torch.tensor([item["integral_t"] for item in records], dtype=work_dtype), quantiles
            ),
            "endpoint_minimum_ratio": _quantile_payload(
                torch.tensor([item["endpoint_minimum_ratio"] for item in records], dtype=work_dtype), quantiles
            ),
            "normalized_valley_depth": _quantile_payload(
                torch.tensor([item["normalized_valley_depth"] for item in records], dtype=work_dtype), quantiles
            ),
            "pairs": records,
        }

    point_diag = merge_proxy_diagnostics(
        points_a,
        points_b,
        regularization=float(resolved.proxy_regularization),
        support_gap_quantile=min(float(value) for value in resolved.support_quantiles),
    )
    unique_a, unique_b = torch.unique(global_a), torch.unique(global_b)
    spacing = max(_median_spacing(points_a), _median_spacing(points_b), 1e-15)
    facing_opacity_a = float(opacities[unique_a].sum())
    facing_opacity_b = float(opacities[unique_b].sum())
    centroid_gap = float(torch.linalg.norm(means[unique_a].mean(dim=0) - means[unique_b].mean(dim=0)))
    facing_support = {
        "gaussian_count_a": int(unique_a.numel()),
        "gaussian_count_b": int(unique_b.numel()),
        "opacity_mass_a": facing_opacity_a,
        "opacity_mass_b": facing_opacity_b,
        "total_opacity_mass": facing_opacity_a + facing_opacity_b,
        "mean_directional_sigma_a": float(sigma_a.mean()),
        "mean_directional_sigma_b": float(sigma_b.mean()),
        "support_centroid_gap": centroid_gap,
        "support_centroid_gap_over_spacing": centroid_gap / spacing,
        "opacity_mass_per_normalized_gap": (facing_opacity_a + facing_opacity_b)
        / max(float(point_diag.scale_normalized_support_gap), 1e-15),
        "selection_distance_cutoff": float(facing_cutoff),
    }

    finite_covariance = bool(torch.isfinite(regularized[support_indices]).all())
    support_floored = floored[support_indices]
    support_axis_confidence = axis_confidence[support_indices]
    support_condition = condition[support_indices]
    support_eigenvalues = eigenvalues[support_indices]
    principal_axis_ambiguous = support_axis_confidence < 0.05
    return GaussianSupportContinuityDiagnostics(
        config=resolved,
        region_a_indices=[int(value) for value in indices_a],
        region_b_indices=[int(value) for value in indices_b],
        validity_flags={
            "valid": finite_covariance,
            "finite_covariance": finite_covariance,
            "covariance_eigenvalue_floored_count": int(support_floored.sum()),
            "principal_axis_ambiguous_count": int(principal_axis_ambiguous.sum()),
            "principal_axis_meaningful_for_all": not bool(principal_axis_ambiguous.any()),
            "opacity_nonzero_count": int((opacities > 0.0).sum()),
        },
        covariance_diagnostics={
            "eigenvalue_min": _quantile_payload(support_eigenvalues[:, 0], quantiles),
            "eigenvalue_middle": _quantile_payload(support_eigenvalues[:, 1], quantiles),
            "eigenvalue_max": _quantile_payload(support_eigenvalues[:, 2], quantiles),
            "condition_number": _quantile_payload(support_condition, quantiles),
            "principal_axis_confidence": _quantile_payload(support_axis_confidence, quantiles),
        },
        existing_point_diagnostics={
            "euclidean_support_gap": float(point_diag.support_gap_quantile),
            "support_gap_over_local_spacing": float(point_diag.scale_normalized_support_gap),
            "sampling_scale": float(point_diag.sampling_scale),
            "merged_normalized_quadratic_rms": float(point_diag.merged_proxy.normalized_rms_residual),
            "normalized_error_increase": float(point_diag.normalized_error_increase),
            "normal_angle_degrees": float(point_diag.normal_angle_degrees),
            "layer_separation_score": float(point_diag.layer_separation_score),
        },
        mahalanobis={
            "one_sided_a": _quantile_payload(one_a, quantiles),
            "one_sided_b": _quantile_payload(one_b, quantiles),
            "symmetric_mean": _quantile_payload(symmetric_mean, quantiles),
            "symmetric_max": _quantile_payload(symmetric_max, quantiles),
            "pooled": _quantile_payload(pooled, quantiles),
        },
        ellipsoid_overlap=ellipsoid,
        projected_reach={
            "center_gap_over_directional_reach": _quantile_payload(reach_ratio, quantiles),
            "signed_directional_reach_margin": _quantile_payload(reach_sum - distance, quantiles),
            "tangent_gap": _quantile_payload(tangent_gap, quantiles),
            "normal_gap": _quantile_payload(normal_gap, quantiles),
            "tangent_reach_ratio": _quantile_payload(tangent_reach_ratio, quantiles),
            "normal_reach_ratio": _quantile_payload(normal_reach_ratio, quantiles),
        },
        bridge_density=bridge_by_mode,
        facing_support=facing_support,
        boundary_pair_metrics=pair_rows,
        computational_cost={
            "cross_distance_evaluations": int(points_a.shape[0] * points_b.shape[0]),
            "directed_boundary_pair_count": int(directed_pair_count),
            "selected_boundary_pair_count": int(global_a.numel()),
            "bridge_sample_count_per_pair": int(resolved.bridge_sample_count),
            "bridge_kernel_evaluations": int(bridge_kernel_evaluations),
        },
    )


__all__ = [
    "GaussianSupportContinuityConfig",
    "GaussianSupportContinuityDiagnostics",
    "covariance_from_scale_rotation",
    "evaluate_gaussian_support_continuity",
]
