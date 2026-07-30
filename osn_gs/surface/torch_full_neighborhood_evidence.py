from __future__ import annotations

"""Full-observed-cloud contextual evidence for a bounded representative set.

Worklog 129 problem statement: the production canonical construction path
bounds its O(N^2) topology stages to a "representative" sample (voxel-nearest
selection, see ``torch_density_preserving_representative_selection.py``), but
prior to this module the CONTEXTUAL reliability check
(``evaluate_contextual_consistency``) also computed its k-nearest-neighbor
evidence only among that same sparse representative set. On a real ADC-trained
scene with ~139k Gaussians and a representative cap of a few hundred to ~2k,
representative spacing is far larger than the true local Gaussian density, so
every representative's "8 nearest OTHER representatives" read as a sparse,
disagreeing cloud -- collapsing every representative to
``contextual_insufficient``/``contextual_mixed`` regardless of how coherent
the real, full-density local surface actually is (worklog 126:
``reliable_count=0`` at every cap tried).

This module computes, for each representative, aggregate evidence over the
FULL observed Gaussian cloud (not the sparse representative set): every full
Gaussian is assigned to its nearest representative (a full-cloud Voronoi
partition), then per-representative statistics (support count, opacity mass,
normal consensus, tangent-plane residual, local density, competing/ rejected
mass) are computed by scatter-reduction over that assignment. Topology
(pairwise same-surface/crease classification) still only ever runs on the
bounded representative set -- this module never introduces an O(N^2) full-
cloud operation, only a chunked nearest-representative assignment (the same
complexity class already used by ``TorchOSNGSPipeline._propagate_canonical_patch_ids``).
"""

from dataclasses import dataclass
from typing import Any, Sequence

from osn_gs.surface.torch_gaussian_covariance_frame import (
    GaussianCovarianceFrame,
    orientation_insensitive_alignment,
)
from osn_gs.surface.torch_gaussian_structural_reliability import (
    INTRINSIC_REJECTED,
    IntrinsicReliabilityResult,
)
from osn_gs.utils.torch_ops import require_torch


@dataclass(frozen=True)
class FullNeighborhoodEvidenceConfig:
    """Configurable policy, not a confirmed canonical threshold set."""

    chunk_size: int = 4096
    # Alignment below this is counted as "competing" (conflicting-orientation)
    # evidence rather than same-surface support.
    competing_mode_alignment_threshold: float = 0.5


@dataclass(frozen=True)
class FullNeighborhoodEvidence:
    """Per-representative aggregate evidence over its full-cloud Voronoi cell."""

    representative_ids: tuple[Any, ...]
    support_count: Any  # (M,) long -- full Gaussians assigned to this representative
    opacity_sum: Any  # (M,)
    opacity_weighted_centroid: Any  # (M, 3)
    mean_spacing: Any  # (M,) mean distance of assigned full Gaussians to the representative
    spacing_std: Any  # (M,)
    normal_consensus: Any  # (M,) in [0, 1] -- resultant length of sign-corrected assigned normals
    tangent_residual_mean: Any  # (M,) mean |offset . rep_normal| / rep tangent scale
    tangent_residual_std: Any  # (M,)
    eigenvalue_ratio_mean: Any  # (M,) mean assigned-Gaussian planarity ratio (lambda2/lambda3)
    eigenvalue_ratio_std: Any  # (M,)
    competing_mode_mass: Any  # (M,) fraction of assigned members with low normal alignment to the representative
    rejected_neighbor_mass: Any  # (M,) fraction of assigned members that are full-cloud intrinsic-rejected
    local_density: Any  # (M,) support_count / representative footprint area
    config: FullNeighborhoodEvidenceConfig

    def payload(self) -> list[dict[str, Any]]:
        count = int(self.support_count.shape[0])
        return [
            {
                "representative_id": self.representative_ids[i],
                "support_count": int(self.support_count[i]),
                "opacity_sum": float(self.opacity_sum[i]),
                "mean_spacing": float(self.mean_spacing[i]),
                "spacing_std": float(self.spacing_std[i]),
                "normal_consensus": float(self.normal_consensus[i]),
                "tangent_residual_mean": float(self.tangent_residual_mean[i]),
                "tangent_residual_std": float(self.tangent_residual_std[i]),
                "eigenvalue_ratio_mean": float(self.eigenvalue_ratio_mean[i]),
                "eigenvalue_ratio_std": float(self.eigenvalue_ratio_std[i]),
                "competing_mode_mass": float(self.competing_mode_mass[i]),
                "rejected_neighbor_mass": float(self.rejected_neighbor_mass[i]),
                "local_density": float(self.local_density[i]),
            }
            for i in range(count)
        ]


def assign_nearest_representative(
    full_positions: Any,
    representative_positions: Any,
    *,
    chunk_size: int = 4096,
) -> tuple[Any, Any]:
    """Return ``(nearest_index, distance)`` of the closest representative per full Gaussian.

    Chunked over the (potentially large) full cloud against the (bounded)
    representative set -- same complexity class as
    ``TorchOSNGSPipeline._nearest_canonical_sample_indices``, kept standalone
    here so it has no dependency on pipeline/model state.
    """
    torch = require_torch()
    full_count = int(full_positions.shape[0])
    nearest_index = torch.empty((full_count,), dtype=torch.long, device=full_positions.device)
    nearest_distance = torch.empty((full_count,), dtype=full_positions.dtype, device=full_positions.device)
    chunk = max(64, min(int(chunk_size), 65536))
    for start in range(0, full_count, chunk):
        end = min(start + chunk, full_count)
        distances = torch.cdist(full_positions[start:end], representative_positions)
        values, indices = distances.min(dim=1)
        nearest_index[start:end] = indices
        nearest_distance[start:end] = values
    return nearest_index, nearest_distance


def compute_full_neighborhood_evidence(
    full_positions: Any,
    full_frame: GaussianCovarianceFrame,
    full_opacity: Any,
    full_intrinsic: IntrinsicReliabilityResult,
    representative_positions: Any,
    representative_frame: GaussianCovarianceFrame,
    representative_ids: Sequence[Any],
    *,
    config: FullNeighborhoodEvidenceConfig | None = None,
) -> FullNeighborhoodEvidence:
    """Aggregate full-cloud evidence into each representative's Voronoi cell.

    ``representative_positions``/``representative_frame`` describe the
    bounded topology sample; ``full_positions``/``full_frame``/``full_opacity``
    describe the entire observed (eligible) Gaussian cloud the representative
    was drawn from. Every full Gaussian nearest to a given representative
    contributes to that representative's aggregate -- this is what lets a
    dense ADC-grown region translate into higher representative confidence
    even though the representative COUNT stays capped.
    """
    torch = require_torch()
    config = config or FullNeighborhoodEvidenceConfig()
    full_positions = torch.as_tensor(full_positions)
    representative_positions = torch.as_tensor(representative_positions)
    m = int(representative_positions.shape[0])
    device = full_positions.device

    nearest, spacing = assign_nearest_representative(
        full_positions, representative_positions, chunk_size=config.chunk_size
    )

    support_count = torch.zeros((m,), dtype=torch.long, device=device)
    support_count.index_add_(0, nearest, torch.ones_like(nearest, dtype=torch.long))
    safe_support = support_count.clamp_min(1).to(full_positions.dtype)

    opacity_flat = torch.as_tensor(full_opacity).reshape(-1)
    opacity_sum = torch.zeros((m,), dtype=full_positions.dtype, device=device)
    opacity_sum.index_add_(0, nearest, opacity_flat)

    weighted_positions = full_positions * opacity_flat.unsqueeze(-1)
    centroid_sum = torch.zeros((m, 3), dtype=full_positions.dtype, device=device)
    centroid_sum.index_add_(0, nearest, weighted_positions)
    safe_opacity_sum = opacity_sum.clamp_min(1e-12)
    opacity_weighted_centroid = centroid_sum / safe_opacity_sum.unsqueeze(-1)
    # Gaussians with (near-)zero opacity everywhere in a cell would otherwise
    # divide by ~0; fall back to the plain (unweighted) centroid for those.
    plain_sum = torch.zeros((m, 3), dtype=full_positions.dtype, device=device)
    plain_sum.index_add_(0, nearest, full_positions)
    plain_centroid = plain_sum / safe_support.unsqueeze(-1)
    opacity_weighted_centroid = torch.where(
        (opacity_sum > 1e-9).unsqueeze(-1), opacity_weighted_centroid, plain_centroid
    )

    spacing_sum = torch.zeros((m,), dtype=full_positions.dtype, device=device)
    spacing_sum.index_add_(0, nearest, spacing)
    mean_spacing = spacing_sum / safe_support
    spacing_sq_sum = torch.zeros((m,), dtype=full_positions.dtype, device=device)
    spacing_sq_sum.index_add_(0, nearest, spacing.square())
    spacing_variance = (spacing_sq_sum / safe_support - mean_spacing.square()).clamp_min(0.0)
    spacing_std = torch.sqrt(spacing_variance)

    rep_normal_per_full = representative_frame.normal_candidate[nearest]
    full_normal = full_frame.normal_candidate
    alignment = orientation_insensitive_alignment(full_normal, rep_normal_per_full)
    sign = torch.where((full_normal * rep_normal_per_full).sum(dim=-1) < 0.0, -1.0, 1.0).unsqueeze(-1)
    corrected_normal = full_normal * sign
    normal_sum = torch.zeros((m, 3), dtype=full_positions.dtype, device=device)
    normal_sum.index_add_(0, nearest, corrected_normal)
    mean_normal = normal_sum / safe_support.unsqueeze(-1)
    normal_consensus = torch.linalg.norm(mean_normal, dim=-1).clamp(max=1.0)

    rep_position_per_full = representative_positions[nearest]
    rep_tangent_scale_per_full = representative_frame.tangent_major_scale[nearest].clamp_min(1e-12)
    offset = full_positions - rep_position_per_full
    residual = (offset * rep_normal_per_full).sum(dim=-1).abs() / rep_tangent_scale_per_full
    residual_sum = torch.zeros((m,), dtype=full_positions.dtype, device=device)
    residual_sum.index_add_(0, nearest, residual)
    tangent_residual_mean = residual_sum / safe_support
    residual_sq_sum = torch.zeros((m,), dtype=full_positions.dtype, device=device)
    residual_sq_sum.index_add_(0, nearest, residual.square())
    residual_variance = (residual_sq_sum / safe_support - tangent_residual_mean.square()).clamp_min(0.0)
    tangent_residual_std = torch.sqrt(residual_variance)

    planarity = full_frame.planarity
    planarity_sum = torch.zeros((m,), dtype=full_positions.dtype, device=device)
    planarity_sum.index_add_(0, nearest, planarity)
    eigenvalue_ratio_mean = planarity_sum / safe_support
    planarity_sq_sum = torch.zeros((m,), dtype=full_positions.dtype, device=device)
    planarity_sq_sum.index_add_(0, nearest, planarity.square())
    planarity_variance = (planarity_sq_sum / safe_support - eigenvalue_ratio_mean.square()).clamp_min(0.0)
    eigenvalue_ratio_std = torch.sqrt(planarity_variance)

    competing_mask = (alignment < config.competing_mode_alignment_threshold).to(full_positions.dtype)
    competing_sum = torch.zeros((m,), dtype=full_positions.dtype, device=device)
    competing_sum.index_add_(0, nearest, competing_mask)
    competing_mode_mass = competing_sum / safe_support

    rejected_mask = torch.tensor(
        [c == INTRINSIC_REJECTED for c in full_intrinsic.intrinsic_class],
        device=device,
    ).to(full_positions.dtype)
    rejected_sum = torch.zeros((m,), dtype=full_positions.dtype, device=device)
    rejected_sum.index_add_(0, nearest, rejected_mask)
    rejected_neighbor_mass = rejected_sum / safe_support

    footprint_area = representative_frame.footprint_area.clamp_min(1e-12)
    local_density = support_count.to(full_positions.dtype) / footprint_area

    return FullNeighborhoodEvidence(
        representative_ids=tuple(representative_ids),
        support_count=support_count.detach(),
        opacity_sum=opacity_sum.detach(),
        opacity_weighted_centroid=opacity_weighted_centroid.detach(),
        mean_spacing=mean_spacing.detach(),
        spacing_std=spacing_std.detach(),
        normal_consensus=normal_consensus.detach(),
        tangent_residual_mean=tangent_residual_mean.detach(),
        tangent_residual_std=tangent_residual_std.detach(),
        eigenvalue_ratio_mean=eigenvalue_ratio_mean.detach(),
        eigenvalue_ratio_std=eigenvalue_ratio_std.detach(),
        competing_mode_mass=competing_mode_mass.detach(),
        rejected_neighbor_mass=rejected_neighbor_mass.detach(),
        local_density=local_density.detach(),
        config=config,
    )
