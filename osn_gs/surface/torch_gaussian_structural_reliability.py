from __future__ import annotations

"""Per-Gaussian structural reliability for covariance-guided boundary evidence.

Canonical principle (worklog 111/113): observation status and geometric
reliability are different axes. An observed Gaussian is not automatically
reliable surface evidence -- an isolated floater, an isotropic blob, or an
oversized bridge Gaussian can all be "observed" while contributing nothing (or
actively misleading) structural evidence.

Worklog 114 correction: reliability itself is not one axis either. This
module now separates:

- :class:`IntrinsicReliabilityResult` -- can THIS Gaussian's own covariance be
  used as surface-normal evidence at all? Computed with NO neighbor/position
  dependency (covariance shape/conditioning/scale validity only).
- :class:`ContextualConsistencyResult` -- given its neighborhood, does this
  Gaussian's evidence agree with a single consistent local surface? This
  DOES depend on neighbor positions/covariances, and is explicitly allowed to
  be "mixed" (crease/multi-surface neighborhood) WITHOUT that meaning the
  Gaussian's own intrinsic evidence is bad.

A Gaussian at a genuine wall/floor crease is exactly
``intrinsic_reliable + contextual_mixed`` -- it must never collapse to the
same state as an intrinsically bad (isotropic/degenerate) Gaussian just
because both end up "not fully clean". ``evaluate_structural_reliability()``
combines both into the OLD 3-tier ``reliable``/``ambiguous``/``rejected``
class purely as a backward-compatible projection; new code should prefer the
two-axis result directly.

This is explicitly a DIAGNOSTIC/POLICY foundation, not a finished algorithm:
thresholds are configurable defaults, not confirmed canonical constants.
"""

from dataclasses import dataclass, field
from typing import Any

from osn_gs.surface.torch_gaussian_covariance_frame import (
    GaussianCovarianceFrame,
    covariance_conditioning_score,
    orientation_insensitive_alignment,
)
from osn_gs.utils.torch_ops import require_torch

# --- Intrinsic axis: can this Gaussian's OWN covariance be used as evidence? ---
INTRINSIC_RELIABLE = "intrinsic_reliable"
INTRINSIC_AMBIGUOUS = "intrinsic_ambiguous"
INTRINSIC_REJECTED = "intrinsic_rejected"

# --- Contextual axis: does the NEIGHBORHOOD agree on one consistent surface? ---
CONTEXTUAL_CONSISTENT = "contextual_consistent"
CONTEXTUAL_MIXED = "contextual_mixed"
CONTEXTUAL_INSUFFICIENT = "contextual_insufficient"

# --- Compatibility projection (worklog 111/113 vocabulary, kept for old callers) ---
RELIABLE = "reliable_structural_evidence"
AMBIGUOUS = "ambiguous_structural_evidence"
REJECTED = "rejected_structural_evidence"

# --- Neighbor aggregation policy (worklog 114 §10) ---
AGGREGATION_MEAN = "mean"
AGGREGATION_MEDIAN = "median"
AGGREGATION_TRIMMED_MEAN = "trimmed_mean"
AGGREGATION_RELIABILITY_WEIGHTED = "reliability_weighted"
AGGREGATION_REJECTED_EXCLUDED = "rejected_excluded"


@dataclass(frozen=True)
class IntrinsicReliabilityConfig:
    """Configurable policy, not a confirmed canonical threshold set. No neighbor dependency."""

    reliable_min_planar_likelihood: float = 0.6
    rejected_min_isotropic_likelihood: float = 0.85
    # Optional absolute bound on equivalent_tangent_scale, e.g. (min, max) in
    # scene units. None means "no absolute bound checked" -- only relative
    # (own-eigenvalue-ratio) validity is enforced.
    expected_scale_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class IntrinsicReliabilityResult:
    intrinsic_class: tuple[str, ...]
    conditioning_score: Any
    planar_likelihood: Any
    needle_likelihood: Any
    isotropic_likelihood: Any
    scale_validity: Any
    reasons: tuple[tuple[str, ...], ...]
    config: IntrinsicReliabilityConfig

    def payload(self) -> list[dict[str, Any]]:
        count = int(self.conditioning_score.shape[0])
        return [
            {
                "intrinsic_class": self.intrinsic_class[i],
                "conditioning_score": float(self.conditioning_score[i]),
                "planar_likelihood": float(self.planar_likelihood[i]),
                "needle_likelihood": float(self.needle_likelihood[i]),
                "isotropic_likelihood": float(self.isotropic_likelihood[i]),
                "scale_validity": float(self.scale_validity[i]),
                "reasons": list(self.reasons[i]),
            }
            for i in range(count)
        ]


@dataclass(frozen=True)
class ContextualConsistencyConfig:
    """Configurable policy, not a confirmed canonical threshold set."""

    neighbor_count: int = 8
    aggregation_method: str = AGGREGATION_REJECTED_EXCLUDED
    trimmed_fraction: float = 0.2
    consistent_min_neighbor_normal_agreement: float = 0.85
    consistent_max_mutual_tangent_residual: float = 0.35
    consistent_min_support_sufficiency: float = 0.4
    consistent_max_multi_surface_ambiguity: float = 0.25
    insufficient_min_valid_neighbor_count: int = 2
    local_support_close_ratio: float = 1.0
    local_support_far_ratio: float = 10.0
    rejected_max_scale_consistency: float = 0.2
    # --- full-observed-neighborhood evidence policy (worklog 129) ---
    # Support sufficiency for the full-cloud-evidence path is driven by an
    # absolute assigned-Gaussian COUNT rather than a representative-only
    # normalized distance, since the whole point of full-neighborhood
    # evidence is that representative spacing is no longer a proxy for real
    # local density.
    full_evidence_min_support_count: int = 6
    full_evidence_saturating_support_count: int = 24
    full_evidence_max_rejected_mass: float = 0.6


@dataclass(frozen=True)
class ContextualConsistencyResult:
    contextual_class: tuple[str, ...]
    neighbor_normal_agreement: Any
    mutual_tangent_residual: Any
    local_curvature_agreement: Any
    neighborhood_support_sufficiency: Any
    multi_surface_neighborhood_ambiguity: Any
    density_variation_sensitivity: Any
    scale_consistency: Any
    valid_neighbor_count: Any
    neighbor_spacing: Any  # (N,) aggregated raw (world-unit) distance to valid neighbors -- a scale-contract reference distinct from any single Gaussian's own covariance scale, see torch_gaussian_manifold_affinity.py
    reasons: tuple[tuple[str, ...], ...]
    config: ContextualConsistencyConfig

    def payload(self) -> list[dict[str, Any]]:
        count = int(self.neighbor_normal_agreement.shape[0])
        return [
            {
                "contextual_class": self.contextual_class[i],
                "neighbor_normal_agreement": float(self.neighbor_normal_agreement[i]),
                "mutual_tangent_residual": float(self.mutual_tangent_residual[i]),
                "local_curvature_agreement": float(self.local_curvature_agreement[i]),
                "neighborhood_support_sufficiency": float(self.neighborhood_support_sufficiency[i]),
                "multi_surface_neighborhood_ambiguity": float(self.multi_surface_neighborhood_ambiguity[i]),
                "density_variation_sensitivity": float(self.density_variation_sensitivity[i]),
                "scale_consistency": float(self.scale_consistency[i]),
                "valid_neighbor_count": int(self.valid_neighbor_count[i]),
                "neighbor_spacing": float(self.neighbor_spacing[i]),
                "reasons": list(self.reasons[i]),
            }
            for i in range(count)
        ]


@dataclass(frozen=True)
class StructuralReliabilityConfig:
    """Aggregates the two independent-axis configs -- kept so
    ``evaluate_structural_reliability(positions, frame)`` (no config) still
    works exactly as before; new callers should prefer passing
    ``IntrinsicReliabilityConfig``/``ContextualConsistencyConfig`` directly.
    """

    intrinsic: IntrinsicReliabilityConfig = field(default_factory=IntrinsicReliabilityConfig)
    contextual: ContextualConsistencyConfig = field(default_factory=ContextualConsistencyConfig)


@dataclass(frozen=True)
class StructuralReliabilityResult:
    """Backward-compatible projection of (intrinsic, contextual) onto the old
    3-tier vocabulary. NOT the canonical source of truth any more -- prefer
    ``.intrinsic``/``.contextual`` for anything new."""

    reliability_class: tuple[str, ...]
    intrinsic: IntrinsicReliabilityResult
    contextual: ContextualConsistencyResult
    # Old flat field names, sourced from the two sub-results, kept for
    # existing callers/tests written against worklog 113's API.
    planarity_score: Any
    neighbor_normal_agreement: Any
    mutual_tangent_residual: Any
    scale_consistency: Any
    local_support_score: Any
    reasons: tuple[tuple[str, ...], ...]
    config: StructuralReliabilityConfig

    def payload(self) -> list[dict[str, Any]]:
        count = int(self.planarity_score.shape[0])
        rows = []
        for index in range(count):
            rows.append(
                {
                    "final_reliability_class": self.reliability_class[index],
                    "intrinsic_class": self.intrinsic.intrinsic_class[index],
                    "contextual_class": self.contextual.contextual_class[index],
                    "planarity_score": float(self.planarity_score[index]),
                    "neighbor_normal_agreement": float(self.neighbor_normal_agreement[index]),
                    "mutual_tangent_residual": float(self.mutual_tangent_residual[index]),
                    "scale_consistency": float(self.scale_consistency[index]),
                    "local_support_score": float(self.local_support_score[index]),
                    "reasons": list(self.reasons[index]),
                }
            )
        return rows


def _pairwise_neighbors(positions: Any, k: int) -> tuple[Any, Any]:
    """Return ``(indices, distances)`` of the ``k`` nearest OTHER points per row."""
    torch = require_torch()
    count = int(positions.shape[0])
    k = max(1, min(k, count - 1))
    distances = torch.cdist(positions, positions)
    distances.fill_diagonal_(float("inf"))
    nearest_distances, nearest_indices = torch.topk(distances, k=k, largest=False, dim=1)
    return nearest_indices, nearest_distances


def _aggregate(values: Any, weights: Any, *, method: str, trimmed_fraction: float) -> Any:
    """Aggregate ``(N, k)`` per-neighbor values into ``(N,)`` per the chosen policy."""
    torch = require_torch()
    if method == AGGREGATION_MEAN:
        return values.mean(dim=1)
    if method in (AGGREGATION_REJECTED_EXCLUDED, AGGREGATION_RELIABILITY_WEIGHTED):
        weight_sum = weights.sum(dim=1).clamp_min(1e-12)
        return (values * weights).sum(dim=1) / weight_sum
    if method == AGGREGATION_MEDIAN:
        return values.median(dim=1).values
    if method == AGGREGATION_TRIMMED_MEAN:
        k = int(values.shape[1])
        trim_count = int(k * trimmed_fraction)
        sorted_values, _ = torch.sort(values, dim=1)
        if trim_count > 0 and k - 2 * trim_count > 0:
            sorted_values = sorted_values[:, trim_count : k - trim_count]
        return sorted_values.mean(dim=1)
    raise ValueError(f"Unknown aggregation method: {method!r}")


def _aggregate_dispersion(values: Any, weights: Any, *, method: str, trimmed_fraction: float) -> Any:
    """Weighted/robust standard deviation of ``(N, k)`` per-neighbor values, matching ``_aggregate``'s policy."""
    torch = require_torch()
    center = _aggregate(values, weights, method=method, trimmed_fraction=trimmed_fraction).unsqueeze(1)
    squared_deviation = (values - center).square()
    return torch.sqrt(torch.clamp(_aggregate(squared_deviation, weights, method=method, trimmed_fraction=trimmed_fraction), min=0.0))


def evaluate_intrinsic_reliability(
    frame: GaussianCovarianceFrame,
    *,
    config: IntrinsicReliabilityConfig | None = None,
) -> IntrinsicReliabilityResult:
    """Per-Gaussian intrinsic evidence quality -- covariance shape/conditioning/scale ONLY.

    No position or neighbor input: this answers "is this Gaussian's own
    covariance usable as surface-normal evidence", independent of anything
    else in the scene.
    """
    torch = require_torch()
    config = config or IntrinsicReliabilityConfig()
    count = int(frame.eigenvalues.shape[0])

    conditioning_score = covariance_conditioning_score(frame.eigenvalues, frame.eigenvalues)
    planar_likelihood = torch.clamp(frame.planarity / (frame.planarity + 3.0), max=1.0)
    needle_likelihood = torch.clamp(frame.elongation / (frame.elongation + 3.0), max=1.0)
    isotropic_likelihood = frame.isotropy

    finite_scale = (
        torch.isfinite(frame.tangent_major_scale)
        & torch.isfinite(frame.tangent_minor_scale)
        & torch.isfinite(frame.normal_thickness)
        & (frame.tangent_major_scale > 0)
        & (frame.tangent_minor_scale > 0)
        & (frame.normal_thickness > 0)
    )
    if config.expected_scale_range is not None:
        low, high = config.expected_scale_range
        finite_scale = finite_scale & (frame.equivalent_tangent_scale >= low) & (frame.equivalent_tangent_scale <= high)
    scale_validity = finite_scale.float()

    # Keep the decision policy exactly as before, but evaluate it as tensor
    # masks.  The old per-row Python ``if`` read CUDA scalars 138k+ times on
    # real ADC snapshots, forcing one host synchronization per Gaussian.
    # Metadata still uses the same Python tuple contract, constructed after a
    # single bulk transfer instead of scalar-by-scalar transfers.
    conditioning_bad = conditioning_score < 1.0
    scale_bad = (~conditioning_bad) & (scale_validity < 1.0)
    isotropic_bad = (~conditioning_bad) & (~scale_bad) & (
        isotropic_likelihood >= config.rejected_min_isotropic_likelihood
    )
    rejected = conditioning_bad | scale_bad | isotropic_bad
    reliable = (~rejected) & (planar_likelihood >= config.reliable_min_planar_likelihood)
    ambiguous = ~(rejected | reliable)

    conditioning_bad_cpu = conditioning_bad.detach().cpu().tolist()
    scale_bad_cpu = scale_bad.detach().cpu().tolist()
    isotropic_bad_cpu = isotropic_bad.detach().cpu().tolist()
    planar_low_cpu = (planar_likelihood < config.reliable_min_planar_likelihood).detach().cpu().tolist()
    needle_high_cpu = (needle_likelihood >= config.reliable_min_planar_likelihood).detach().cpu().tolist()
    rejected_cpu = rejected.detach().cpu().tolist()
    reliable_cpu = reliable.detach().cpu().tolist()
    intrinsic_class: list[str] = []
    reasons: list[tuple[str, ...]] = []
    for index in range(count):
        if rejected_cpu[index]:
            intrinsic_class.append(INTRINSIC_REJECTED)
            if conditioning_bad_cpu[index]:
                reasons.append(("degenerate_or_nonfinite_covariance",))
            elif scale_bad_cpu[index]:
                reasons.append(("scale_outside_expected_range_or_nonfinite",))
            else:
                reasons.append(("isotropic_no_normal_evidence",))
        elif reliable_cpu[index]:
            intrinsic_class.append(INTRINSIC_RELIABLE)
            reasons.append(())
        else:
            intrinsic_class.append(INTRINSIC_AMBIGUOUS)
            row_reasons: list[str] = []
            if planar_low_cpu[index]:
                row_reasons.append("insufficient_planar_likelihood")
            if needle_high_cpu[index]:
                row_reasons.append("needle_like_ambiguous_normal_direction")
            reasons.append(tuple(row_reasons))

    return IntrinsicReliabilityResult(
        intrinsic_class=tuple(intrinsic_class),
        conditioning_score=conditioning_score.detach(),
        planar_likelihood=planar_likelihood.detach(),
        needle_likelihood=needle_likelihood.detach(),
        isotropic_likelihood=isotropic_likelihood.detach(),
        scale_validity=scale_validity.detach(),
        reasons=tuple(reasons),
        config=config,
    )


def evaluate_contextual_consistency(
    positions: Any,
    frame: GaussianCovarianceFrame,
    intrinsic: IntrinsicReliabilityResult,
    *,
    config: ContextualConsistencyConfig | None = None,
) -> ContextualConsistencyResult:
    """Per-Gaussian neighborhood consistency -- REQUIRES positions/neighbors.

    Neighbor aggregation is policy-configurable (``config.aggregation_method``,
    worklog 114 §10): a rejected-neighbor's evidence can silently dominate a
    naive mean, so the default (``rejected_excluded``) drops REJECTED
    neighbors from every aggregate rather than averaging them in. See the
    module docstring / worklog 114 for the comparison that motivated this
    default.
    """
    torch = require_torch()
    config = config or ContextualConsistencyConfig()
    positions = torch.as_tensor(positions)
    count = int(positions.shape[0])
    if count < 2:
        raise ValueError("Contextual consistency requires at least two Gaussians.")

    neighbor_indices, neighbor_distances = _pairwise_neighbors(positions, config.neighbor_count)
    k = int(neighbor_indices.shape[1])
    tangent_major_scale = frame.tangent_major_scale
    neighbor_tangent_major_scale = tangent_major_scale[neighbor_indices]

    # ``neighbor_indices`` lives on the same device as ``positions``. Keep
    # these policy masks there so CUDA canonical reconstruction does not
    # attempt to index a CPU tensor with CUDA indices.
    is_rejected = torch.tensor(
        [c == "intrinsic_rejected" for c in intrinsic.intrinsic_class],
        device=positions.device,
    )
    is_reliable = torch.tensor(
        [c == "intrinsic_reliable" for c in intrinsic.intrinsic_class],
        device=positions.device,
    )
    neighbor_rejected = is_rejected[neighbor_indices]
    neighbor_reliable = is_reliable[neighbor_indices]
    valid_neighbor_mask = ~neighbor_rejected
    valid_neighbor_count = valid_neighbor_mask.sum(dim=1)

    if config.aggregation_method == AGGREGATION_REJECTED_EXCLUDED:
        weights = valid_neighbor_mask.float()
    elif config.aggregation_method == AGGREGATION_RELIABILITY_WEIGHTED:
        weights = torch.where(neighbor_rejected, 0.0, torch.where(neighbor_reliable, 1.0, 0.5))
    else:
        weights = torch.ones_like(neighbor_rejected, dtype=torch.float32)
    # Every aggregation method still hard-excludes REJECTED neighbors from
    # mean/median/trimmed_mean too -- a rejected Gaussian is not "possibly
    # low-weight evidence", it is excluded evidence, regardless of averaging
    # style. Only the WEIGHT SHAPE among non-rejected neighbors differs.
    safe_weights = torch.where(neighbor_rejected, torch.zeros_like(weights), weights.clamp_min(1e-6))

    neighbor_normals = frame.normal_candidate[neighbor_indices]
    self_normals = frame.normal_candidate.unsqueeze(1).expand_as(neighbor_normals)
    per_neighbor_alignment = orientation_insensitive_alignment(self_normals, neighbor_normals)

    neighbor_positions = positions[neighbor_indices]
    displacement = neighbor_positions - positions.unsqueeze(1)
    self_normal_expanded = frame.normal_candidate.unsqueeze(1).expand_as(displacement)
    residual_from_self = (displacement * self_normal_expanded).sum(dim=-1).abs() / tangent_major_scale.unsqueeze(1).clamp_min(1e-12)
    residual_from_neighbor = (-displacement * neighbor_normals).sum(dim=-1).abs() / neighbor_tangent_major_scale.clamp_min(1e-12)
    per_neighbor_residual = torch.maximum(residual_from_self, residual_from_neighbor)

    average_scale = (tangent_major_scale.unsqueeze(1) + neighbor_tangent_major_scale) / 2.0
    per_neighbor_normalized_distance = neighbor_distances / average_scale.clamp_min(1e-12)
    per_neighbor_curvature_rate = (1.0 - per_neighbor_alignment) / per_neighbor_normalized_distance.clamp_min(1e-6)

    scale_ratio = tangent_major_scale.unsqueeze(1) / neighbor_tangent_major_scale.clamp_min(1e-12)
    per_neighbor_scale_log_ratio = scale_ratio.log().abs()

    def agg(values: Any) -> Any:
        return _aggregate(values, safe_weights, method=config.aggregation_method, trimmed_fraction=config.trimmed_fraction)

    neighbor_spacing = agg(neighbor_distances)
    neighbor_normal_agreement = agg(per_neighbor_alignment)
    mutual_tangent_residual = agg(per_neighbor_residual)
    scale_consistency = torch.exp(-torch.clamp(agg(per_neighbor_scale_log_ratio), max=10.0))
    normalized_support_distance = agg(per_neighbor_normalized_distance)
    support_span = max(config.local_support_far_ratio - config.local_support_close_ratio, 1e-12)
    neighborhood_support_sufficiency = torch.clamp(
        1.0 - (normalized_support_distance - config.local_support_close_ratio) / support_span, min=0.0, max=1.0
    )
    # Curvature *agreement*: how CONSISTENT the per-neighbor curvature rate is
    # (low dispersion -> smoothly-explained local curvature; high dispersion
    # -> some neighbors jump while others don't, a crease/discontinuity signal
    # rather than uniform curvature).
    curvature_dispersion = _aggregate_dispersion(per_neighbor_curvature_rate, safe_weights, method=config.aggregation_method, trimmed_fraction=config.trimmed_fraction)
    local_curvature_agreement = 1.0 / (1.0 + curvature_dispersion)
    # Multi-surface ambiguity: dispersion of the alignment scores themselves --
    # a bimodal neighbor set (some near-parallel, some near-perpendicular)
    # spikes this even if the MEAN alignment looks moderate.
    multi_surface_neighborhood_ambiguity = _aggregate_dispersion(per_neighbor_alignment, safe_weights, method=config.aggregation_method, trimmed_fraction=config.trimmed_fraction)

    # Density-variation sensitivity: how much the support estimate would swing
    # between a naive mean and a robust median aggregation -- independent of
    # whichever method is actually configured as default.
    mean_weights = torch.ones_like(safe_weights)
    naive_support_distance = _aggregate(per_neighbor_normalized_distance, mean_weights, method=AGGREGATION_MEAN, trimmed_fraction=config.trimmed_fraction)
    robust_support_distance = _aggregate(per_neighbor_normalized_distance, safe_weights, method=AGGREGATION_MEDIAN, trimmed_fraction=config.trimmed_fraction)
    density_variation_sensitivity = (naive_support_distance - robust_support_distance).abs() / support_span

    contextual_class = []
    reasons: list[tuple[str, ...]] = []
    for index in range(count):
        row_reasons: list[str] = []
        if int(valid_neighbor_count[index]) < config.insufficient_min_valid_neighbor_count:
            row_reasons.append("insufficient_non_rejected_neighbors")
            contextual_class.append(CONTEXTUAL_INSUFFICIENT)
            reasons.append(tuple(row_reasons))
            continue
        consistent = (
            neighbor_normal_agreement[index] >= config.consistent_min_neighbor_normal_agreement
            and mutual_tangent_residual[index] <= config.consistent_max_mutual_tangent_residual
            and neighborhood_support_sufficiency[index] >= config.consistent_min_support_sufficiency
            and multi_surface_neighborhood_ambiguity[index] <= config.consistent_max_multi_surface_ambiguity
        )
        if consistent:
            contextual_class.append(CONTEXTUAL_CONSISTENT)
        else:
            if neighbor_normal_agreement[index] < config.consistent_min_neighbor_normal_agreement:
                row_reasons.append("neighbor_normal_disagreement")
            if mutual_tangent_residual[index] > config.consistent_max_mutual_tangent_residual:
                row_reasons.append("mutual_tangent_residual_above_threshold")
            if neighborhood_support_sufficiency[index] < config.consistent_min_support_sufficiency:
                row_reasons.append("insufficient_neighborhood_support")
            if multi_surface_neighborhood_ambiguity[index] > config.consistent_max_multi_surface_ambiguity:
                row_reasons.append("multi_surface_neighborhood_ambiguity")
            contextual_class.append(CONTEXTUAL_MIXED)
        reasons.append(tuple(row_reasons))

    return ContextualConsistencyResult(
        contextual_class=tuple(contextual_class),
        neighbor_normal_agreement=neighbor_normal_agreement.detach(),
        mutual_tangent_residual=mutual_tangent_residual.detach(),
        local_curvature_agreement=local_curvature_agreement.detach(),
        neighborhood_support_sufficiency=neighborhood_support_sufficiency.detach(),
        multi_surface_neighborhood_ambiguity=multi_surface_neighborhood_ambiguity.detach(),
        density_variation_sensitivity=density_variation_sensitivity.detach(),
        scale_consistency=scale_consistency.detach(),
        valid_neighbor_count=valid_neighbor_count.detach(),
        neighbor_spacing=neighbor_spacing.detach(),
        reasons=tuple(reasons),
        config=config,
    )


def combine_reliability(
    intrinsic: IntrinsicReliabilityResult,
    contextual: ContextualConsistencyResult,
    *,
    config: StructuralReliabilityConfig,
) -> StructuralReliabilityResult:
    """Project (intrinsic, contextual) onto the OLD 3-tier class.

    Shared by :func:`evaluate_structural_reliability` (representative-only
    contextual evidence) and
    :func:`evaluate_structural_reliability_from_full_evidence` (full-observed-
    cloud contextual evidence, worklog 129) -- the projection rule itself does
    not depend on WHERE the contextual evidence came from.

    Projection rule (worklog 114 §2):

    - ``intrinsic_rejected`` -> ``rejected`` (never usable as primary evidence,
      regardless of neighborhood).
    - ``intrinsic_reliable`` + ``contextual_consistent`` -> ``reliable`` (clean
      interior evidence).
    - everything else (``intrinsic_reliable`` + mixed/insufficient context, or
      ``intrinsic_ambiguous`` with any context) -> ``ambiguous``. This is
      deliberately NOT further split here -- e.g. a real crease Gaussian
      (intrinsic reliable, contextual mixed) lands here, which is why callers
      that need to distinguish "crease candidate" from "genuinely uncertain"
      should read ``.intrinsic``/``.contextual`` directly instead of relying
      on this compatibility field.
    """
    count = len(intrinsic.intrinsic_class)
    reliability_class = []
    reasons: list[tuple[str, ...]] = []
    for index in range(count):
        if intrinsic.intrinsic_class[index] == INTRINSIC_REJECTED:
            reliability_class.append(REJECTED)
        elif intrinsic.intrinsic_class[index] == INTRINSIC_RELIABLE and contextual.contextual_class[index] == CONTEXTUAL_CONSISTENT:
            reliability_class.append(RELIABLE)
        else:
            reliability_class.append(AMBIGUOUS)
        reasons.append(tuple(intrinsic.reasons[index]) + tuple(contextual.reasons[index]))

    return StructuralReliabilityResult(
        reliability_class=tuple(reliability_class),
        intrinsic=intrinsic,
        contextual=contextual,
        planarity_score=intrinsic.planar_likelihood,
        neighbor_normal_agreement=contextual.neighbor_normal_agreement,
        mutual_tangent_residual=contextual.mutual_tangent_residual,
        scale_consistency=contextual.scale_consistency,
        local_support_score=contextual.neighborhood_support_sufficiency,
        reasons=tuple(reasons),
        config=config,
    )


def evaluate_structural_reliability(
    positions: Any,
    frame: GaussianCovarianceFrame,
    *,
    config: StructuralReliabilityConfig | None = None,
) -> StructuralReliabilityResult:
    """Canonical entry point: intrinsic + contextual, projected to the OLD 3-tier class.

    Contextual evidence here comes from the representative-only k-nearest-
    neighbor search (see :func:`evaluate_contextual_consistency`). For a
    bounded representative set drawn from a much denser full observed cloud,
    prefer :func:`evaluate_structural_reliability_from_full_evidence` instead
    (worklog 129) -- representative-only neighbor search under-counts real
    local density and can never see support that full-cloud aggregation
    would.
    """
    config = config or StructuralReliabilityConfig()
    intrinsic = evaluate_intrinsic_reliability(frame, config=config.intrinsic)
    contextual = evaluate_contextual_consistency(positions, frame, intrinsic, config=config.contextual)
    return combine_reliability(intrinsic, contextual, config=config)


def evaluate_contextual_consistency_from_full_evidence(
    evidence: "FullNeighborhoodEvidence",
    *,
    config: ContextualConsistencyConfig | None = None,
) -> ContextualConsistencyResult:
    """Contextual consistency driven by full-observed-cloud aggregate evidence.

    Worklog 129: replaces the representative-only 8-nearest-neighbor search
    with statistics aggregated over every full-cloud Gaussian assigned to a
    representative's Voronoi cell (see
    ``torch_full_neighborhood_evidence.compute_full_neighborhood_evidence``).
    Support sufficiency is judged by an absolute assigned-Gaussian count
    (``full_evidence_min_support_count``/``full_evidence_saturating_support_count``)
    rather than representative-only normalized spacing, since representative
    spacing is no longer a valid density proxy once representatives are
    capped far below the true Gaussian count.

    Produces the SAME :class:`ContextualConsistencyResult` contract as the
    representative-only path so downstream code (``combine_reliability``,
    region formation, etc.) is unchanged.
    """
    torch = require_torch()
    config = config or ContextualConsistencyConfig()
    count = int(evidence.support_count.shape[0])

    non_rejected_support = evidence.support_count.to(evidence.opacity_sum.dtype) * (
        1.0 - evidence.rejected_neighbor_mass
    )
    support_span = max(
        config.full_evidence_saturating_support_count - config.full_evidence_min_support_count, 1
    )
    neighborhood_support_sufficiency = torch.clamp(
        (non_rejected_support - config.full_evidence_min_support_count) / support_span, min=0.0, max=1.0
    )
    # No-support representatives contribute no consensus signal; treat a
    # zero-support cell as fully disagreeing/ambiguous rather than as
    # spuriously "aligned" (an empty mean would otherwise read as 0, which
    # already happens to be the right answer for normal_consensus, but is
    # made explicit here for tangent residual / eigenvalue-ratio std, which
    # can be exactly 0 with zero support and must not be read as "perfectly
    # consistent").
    has_support = evidence.support_count > 0

    contextual_class = []
    reasons: list[tuple[str, ...]] = []
    for index in range(count):
        row_reasons: list[str] = []
        if not bool(has_support[index]) or int(evidence.support_count[index]) < config.insufficient_min_valid_neighbor_count:
            row_reasons.append("insufficient_full_neighborhood_support")
            contextual_class.append(CONTEXTUAL_INSUFFICIENT)
            reasons.append(tuple(row_reasons))
            continue
        if float(evidence.rejected_neighbor_mass[index]) > config.full_evidence_max_rejected_mass:
            row_reasons.append("rejected_neighbor_dominant")
        consistent = (
            float(evidence.normal_consensus[index]) >= config.consistent_min_neighbor_normal_agreement
            and float(evidence.tangent_residual_mean[index]) <= config.consistent_max_mutual_tangent_residual
            and float(neighborhood_support_sufficiency[index]) >= config.consistent_min_support_sufficiency
            and float(evidence.competing_mode_mass[index]) <= config.consistent_max_multi_surface_ambiguity
            and not row_reasons
        )
        if consistent:
            contextual_class.append(CONTEXTUAL_CONSISTENT)
        else:
            if float(evidence.normal_consensus[index]) < config.consistent_min_neighbor_normal_agreement:
                row_reasons.append("neighbor_normal_disagreement")
            if float(evidence.tangent_residual_mean[index]) > config.consistent_max_mutual_tangent_residual:
                row_reasons.append("mutual_tangent_residual_above_threshold")
            if float(neighborhood_support_sufficiency[index]) < config.consistent_min_support_sufficiency:
                row_reasons.append("insufficient_neighborhood_support")
            if float(evidence.competing_mode_mass[index]) > config.consistent_max_multi_surface_ambiguity:
                row_reasons.append("multi_surface_neighborhood_ambiguity")
            contextual_class.append(CONTEXTUAL_MIXED)
        reasons.append(tuple(row_reasons))

    scale_consistency = torch.exp(
        -torch.clamp(evidence.eigenvalue_ratio_std / evidence.eigenvalue_ratio_mean.clamp_min(1e-6), max=10.0)
    )
    local_curvature_agreement = 1.0 / (1.0 + evidence.tangent_residual_std)

    return ContextualConsistencyResult(
        contextual_class=tuple(contextual_class),
        neighbor_normal_agreement=evidence.normal_consensus.detach(),
        mutual_tangent_residual=evidence.tangent_residual_mean.detach(),
        local_curvature_agreement=local_curvature_agreement.detach(),
        neighborhood_support_sufficiency=neighborhood_support_sufficiency.detach(),
        multi_surface_neighborhood_ambiguity=evidence.competing_mode_mass.detach(),
        density_variation_sensitivity=(
            evidence.spacing_std / evidence.mean_spacing.clamp_min(1e-12)
        ).detach(),
        scale_consistency=scale_consistency.detach(),
        valid_neighbor_count=non_rejected_support.round().long().detach(),
        neighbor_spacing=evidence.mean_spacing.detach(),
        reasons=tuple(reasons),
        config=config,
    )


def evaluate_structural_reliability_from_full_evidence(
    frame: GaussianCovarianceFrame,
    evidence: "FullNeighborhoodEvidence",
    *,
    config: StructuralReliabilityConfig | None = None,
) -> StructuralReliabilityResult:
    """Canonical full-observed-cloud entry point (worklog 129).

    Intrinsic evidence is unchanged (representative's own learned covariance,
    no neighbor dependency -- already the production primary-evidence source
    for the ADC-post-commit path). Contextual evidence comes from
    :func:`evaluate_contextual_consistency_from_full_evidence` instead of the
    representative-only k-nearest-neighbor search.
    """
    config = config or StructuralReliabilityConfig()
    intrinsic = evaluate_intrinsic_reliability(frame, config=config.intrinsic)
    contextual = evaluate_contextual_consistency_from_full_evidence(evidence, config=config.contextual)
    return combine_reliability(intrinsic, contextual, config=config)
