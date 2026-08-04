from __future__ import annotations

"""Production orchestration from bounded occluded candidates to safe proposals.

This is deliberately downstream-only: Worklog 56's visible/Phase D/E bridge is
consumed as an opaque producer.  Each candidate is accounted for exactly once,
and only a Phase-E ``candidate`` supported exclusively by Phase-D ``valid``
domains can enter constrained fitting.  In particular, a ``degenerate`` domain
is useful Phase-E provenance but is never silently promoted to a Phase-F/G input.
"""

from dataclasses import dataclass
from typing import Any, Mapping

from osn_gs.surface.torch_chart_conflict import attach_conflict_edges, build_occluded_chart_conflicts
from osn_gs.surface.torch_eligible_boundary_continuation_bridge import (
    EligibleBoundaryContinuationBridgeResult,
    GaussianToOccludedCandidateBridgeResult,
    run_eligible_boundary_continuation_bridge_from_gaussians,
)
from osn_gs.surface.torch_occluded_chart import OccludedChartFitConfig, OccludedChartResult, fit_occluded_chart
from osn_gs.surface.torch_occluded_chart_hardening import (
    OccludedChartHardeningConfig,
    OccludedChartSafetyResult,
    evaluate_occluded_chart_safety,
)
from osn_gs.surface.torch_uncertain_gaussian_proposal import (
    UncertainGaussianProposalBatch,
    UncertainGaussianProposalConfig,
    default_target_spacing,
    generate_uncertain_gaussian_proposals,
)


STATUS_PROPOSED = "proposed"
STATUS_CANDIDATE_REJECTED = "candidate_rejected"
STATUS_DOMAIN_NOT_CANDIDATE_READY = "domain_not_candidate_ready"
STATUS_SOURCE_PROVENANCE_REJECTED = "source_provenance_rejected"
STATUS_FIT_FAILED = "fit_failed"
STATUS_CHART_REJECTED = "chart_rejected"
STATUS_SAFETY_REJECTED = "safety_rejected"
STATUS_PROPOSAL_REJECTED = "proposal_rejected"


@dataclass(frozen=True)
class SafeUncertainProposalAttempt:
    """Complete accounting record for one Phase-E candidate."""

    candidate_id: str
    supporting_domain_ids: tuple[str, ...]
    supporting_boundary_ids: tuple[str, ...]
    supporting_patch_ids: tuple[int, ...]
    status: str
    reasons: tuple[str, ...]
    chart: OccludedChartResult | None = None
    safety: OccludedChartSafetyResult | None = None
    proposal: UncertainGaussianProposalBatch | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "supporting_domain_ids": list(self.supporting_domain_ids),
            "supporting_boundary_ids": list(self.supporting_boundary_ids),
            "supporting_patch_ids": list(self.supporting_patch_ids),
            "status": self.status,
            "reasons": list(self.reasons),
            "chart_id": None if self.chart is None else self.chart.chart_id,
            "chart_state": None if self.chart is None else self.chart.state,
            "safety_eligibility": None if self.safety is None else self.safety.eligibility,
            "proposal_batch_id": None if self.proposal is None else self.proposal.proposal_batch_id,
            "proposal_sample_count": None if self.proposal is None else int(self.proposal.uv.shape[0]),
        }


@dataclass(frozen=True)
class SafeUncertainProposalProductionResult:
    bridge: EligibleBoundaryContinuationBridgeResult
    attempts: tuple[SafeUncertainProposalAttempt, ...]

    def diagnostic_summary(self) -> dict[str, Any]:
        return {
            "candidate_count": len(self.attempts),
            "proposed_count": sum(item.status == STATUS_PROPOSED for item in self.attempts),
            "rejected_count": sum(item.status != STATUS_PROPOSED for item in self.attempts),
            "proposal_sample_count": sum(
                0 if item.proposal is None else int(item.proposal.valid_mask.sum()) for item in self.attempts
            ),
            "all_candidates_accounted": len(self.attempts) == len(self.bridge.occluded_region_candidates),
        }


@dataclass(frozen=True)
class GaussianToSafeUncertainProposalResult:
    candidate_bridge: GaussianToOccludedCandidateBridgeResult
    production: SafeUncertainProposalProductionResult


def _common(candidate: Any) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "supporting_domain_ids": tuple(candidate.supporting_domain_ids),
        "supporting_boundary_ids": tuple(candidate.supporting_boundary_ids),
        "supporting_patch_ids": tuple(int(item) for item in candidate.supporting_patch_ids),
    }


def _candidate_input_reasons(candidate: Any, domains: Mapping[str, Any], boundaries: Mapping[str, Any]) -> tuple[str, ...]:
    reasons: list[str] = []
    if candidate.state != "candidate":
        reasons.append(f"candidate_state:{candidate.state}")
    if len(candidate.supporting_domain_ids) != 2:
        reasons.append("supporting_domain_count_not_pairwise")
    if not (
        len(candidate.supporting_domain_ids) == len(candidate.supporting_boundary_ids) == len(candidate.supporting_patch_ids)
    ):
        reasons.append("supporting_source_cardinality_mismatch")
        return tuple(reasons)
    for domain_id, boundary_id, patch_id in zip(
        candidate.supporting_domain_ids, candidate.supporting_boundary_ids, candidate.supporting_patch_ids
    ):
        domain = domains.get(domain_id)
        boundary = boundaries.get(boundary_id)
        if domain is None or boundary is None:
            reasons.append("source_domain_or_boundary_missing")
            continue
        if domain.state != "valid":
            reasons.append(f"continuation_domain_state:{domain.state}")
        if domain.source_boundary_id != boundary_id or int(domain.source_patch_id) != int(patch_id):
            reasons.append("source_domain_boundary_provenance_mismatch")
    return tuple(sorted(set(reasons)))


def build_safe_uncertain_proposals_from_bridge(
    bridge: EligibleBoundaryContinuationBridgeResult,
    *,
    surfaces_by_patch_id: Mapping[int, Any],
    chart_config: OccludedChartFitConfig | None = None,
    safety_config: OccludedChartHardeningConfig | None = None,
    proposal_config: UncertainGaussianProposalConfig | None = None,
) -> SafeUncertainProposalProductionResult:
    """Run existing Phase F, F.1 and G implementations for every bridge candidate.

    No model append, appearance assignment, or opacity assignment occurs here.
    """

    domains = {domain.domain_id: domain for domain in bridge.continuation_domains}
    boundaries = bridge.boundaries_by_id
    staged: list[tuple[Any, OccludedChartResult]] = []
    attempts: dict[str, SafeUncertainProposalAttempt] = {}

    for candidate in bridge.occluded_region_candidates:
        common = _common(candidate)
        reasons = _candidate_input_reasons(candidate, domains, boundaries)
        if candidate.state != "candidate":
            attempts[candidate.candidate_id] = SafeUncertainProposalAttempt(
                **common, status=STATUS_CANDIDATE_REJECTED, reasons=reasons,
            )
            continue
        if any(reason.startswith("continuation_domain_state:") for reason in reasons):
            attempts[candidate.candidate_id] = SafeUncertainProposalAttempt(
                **common, status=STATUS_DOMAIN_NOT_CANDIDATE_READY, reasons=reasons,
            )
            continue
        if reasons:
            attempts[candidate.candidate_id] = SafeUncertainProposalAttempt(
                **common, status=STATUS_SOURCE_PROVENANCE_REJECTED, reasons=reasons,
            )
            continue
        try:
            chart = fit_occluded_chart(candidate, domains, boundaries, surfaces_by_patch_id, config=chart_config)
        except Exception as exc:  # fail closed; one candidate cannot suppress accounting for its peers
            attempts[candidate.candidate_id] = SafeUncertainProposalAttempt(
                **common, status=STATUS_FIT_FAILED, reasons=(f"{type(exc).__name__}:{exc}",),
            )
            continue
        if chart.state != "validated" or chart.surface is None:
            attempts[candidate.candidate_id] = SafeUncertainProposalAttempt(
                **common, status=STATUS_CHART_REJECTED,
                reasons=(f"chart_state:{chart.state}", chart.reason), chart=chart,
            )
            continue
        staged.append((candidate, chart))

    conflicts = build_occluded_chart_conflicts([chart for _, chart in staged])
    safeties: dict[str, OccludedChartSafetyResult] = {}
    for candidate, chart in staged:
        safeties[chart.chart_id] = evaluate_occluded_chart_safety(
            chart, surfaces_by_patch_id, config=safety_config, candidate=candidate,
            domains_by_id=domains, boundaries_by_id=boundaries,
        )
    attach_conflict_edges(list(safeties.values()), conflicts)

    for candidate, chart in staged:
        common = _common(candidate)
        safety = safeties[chart.chart_id]
        if safety.eligibility != "eligible":
            attempts[candidate.candidate_id] = SafeUncertainProposalAttempt(
                **common, status=STATUS_SAFETY_REJECTED,
                reasons=tuple(sorted(set(safety.reasons) | set(safety.uncertainty))), chart=chart, safety=safety,
            )
            continue
        try:
            config = proposal_config or UncertainGaussianProposalConfig(
                target_spacing=default_target_spacing(domains, candidate.supporting_domain_ids),
            )
            proposal = generate_uncertain_gaussian_proposals(chart, safety, config=config, conflict_edges=conflicts)
        except Exception as exc:  # no partial or implicitly approved proposal batch
            attempts[candidate.candidate_id] = SafeUncertainProposalAttempt(
                **common, status=STATUS_PROPOSAL_REJECTED, reasons=(f"{type(exc).__name__}:{exc}",),
                chart=chart, safety=safety,
            )
            continue
        if proposal.metadata["eligibility"] != "eligible":
            attempts[candidate.candidate_id] = SafeUncertainProposalAttempt(
                **common, status=STATUS_PROPOSAL_REJECTED,
                reasons=tuple(proposal.metadata["safety_reasons"]), chart=chart, safety=safety, proposal=proposal,
            )
            continue
        attempts[candidate.candidate_id] = SafeUncertainProposalAttempt(
            **common, status=STATUS_PROPOSED, reasons=(), chart=chart, safety=safety, proposal=proposal,
        )

    return SafeUncertainProposalProductionResult(
        bridge=bridge,
        attempts=tuple(attempts[candidate.candidate_id] for candidate in bridge.occluded_region_candidates),
    )


def run_safe_uncertain_proposals_from_gaussians(
    positions: Any,
    *,
    covariance: Any | None = None,
    log_scales: Any | None = None,
    rotations: Any | None = None,
    stable_ids: Any | None = None,
    config: Any | None = None,
    reliability: Any | None = None,
    continuation_input: Any | None = None,
    candidate_scale: Any | None = None,
    residual_scale: Any | None = None,
    extent_multiplier: float = 1.0,
    correspondence_threshold: float = 1.0,
    chart_config: OccludedChartFitConfig | None = None,
    safety_config: OccludedChartHardeningConfig | None = None,
    proposal_config: UncertainGaussianProposalConfig | None = None,
) -> GaussianToSafeUncertainProposalResult:
    """Single production call: Gaussian evidence → safe uncertain proposals."""

    candidate_bridge = run_eligible_boundary_continuation_bridge_from_gaussians(
        positions, covariance=covariance, log_scales=log_scales, rotations=rotations,
        stable_ids=stable_ids, config=config, reliability=reliability,
        continuation_input=continuation_input, candidate_scale=candidate_scale,
        residual_scale=residual_scale, extent_multiplier=extent_multiplier,
        correspondence_threshold=correspondence_threshold,
    )
    surfaces = {
        int(item.input.source_region_id): item.surface
        for item in candidate_bridge.construction.eligible_materialized_surfaces()
        if item.state == "materialized" and item.surface is not None
    }
    production = build_safe_uncertain_proposals_from_bridge(
        candidate_bridge.bridge, surfaces_by_patch_id=surfaces, chart_config=chart_config,
        safety_config=safety_config, proposal_config=proposal_config,
    )
    return GaussianToSafeUncertainProposalResult(candidate_bridge=candidate_bridge, production=production)
