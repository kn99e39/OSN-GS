from __future__ import annotations

"""Production hand-off from Worklog 57 safe proposals to atomic model append.

This module does not alter visible construction or Phase D--G.  It only
forwards a proposal that Worklog 57 already marked ``proposed`` to the existing
``UncertainGaussianAppendAdapter`` transaction, with caller-supplied (never
synthesized) appearance/opacity initialization.
"""

from dataclasses import dataclass
from typing import Any, Callable

from osn_gs.gaussian.torch_uncertain_append_adapter import (
    UncertainAppendInitialization,
    UncertainAppendReceipt,
    UncertainGaussianAppendAdapter,
)
from osn_gs.surface.torch_safe_uncertain_proposal_production import (
    GaussianToSafeUncertainProposalResult,
    SafeUncertainProposalAttempt,
    SafeUncertainProposalProductionResult,
    run_safe_uncertain_proposals_from_gaussians,
)

APPENDED = "appended"
REJECTED = "rejected"
DUPLICATE = "duplicate"
ROLLED_BACK = "rolled_back"

InitializationProvider = Callable[[Any, SafeUncertainProposalAttempt], UncertainAppendInitialization | None]


@dataclass(frozen=True)
class SafeUncertainAppendAttempt:
    candidate_id: str
    chart_id: str | None
    proposal_batch_id: str | None
    status: str
    reasons: tuple[str, ...]
    receipt: UncertainAppendReceipt | None
    source_attempt: SafeUncertainProposalAttempt

    def payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "chart_id": self.chart_id,
            "proposal_batch_id": self.proposal_batch_id,
            "status": self.status,
            "reasons": list(self.reasons),
            "receipt": None if self.receipt is None else self.receipt.stable_payload(),
            "source_attempt": self.source_attempt.payload(),
        }


@dataclass(frozen=True)
class SafeUncertainAppendProductionResult:
    safe_proposals: SafeUncertainProposalProductionResult
    attempts: tuple[SafeUncertainAppendAttempt, ...]
    adapter: UncertainGaussianAppendAdapter

    def diagnostic_summary(self) -> dict[str, Any]:
        return {
            "candidate_count": len(self.attempts),
            "appended_count": sum(item.status == APPENDED for item in self.attempts),
            "rejected_count": sum(item.status == REJECTED for item in self.attempts),
            "duplicate_count": sum(item.status == DUPLICATE for item in self.attempts),
            "rolled_back_count": sum(item.status == ROLLED_BACK for item in self.attempts),
            "all_candidates_accounted": len(self.attempts) == len(self.safe_proposals.attempts),
        }


@dataclass(frozen=True)
class GaussianToAtomicUncertainAppendResult:
    safe_proposal_result: GaussianToSafeUncertainProposalResult
    append_result: SafeUncertainAppendProductionResult


def append_safe_uncertain_proposals(
    safe_proposals: SafeUncertainProposalProductionResult,
    *,
    model: Any,
    initialization_provider: InitializationProvider | None,
    adapter: UncertainGaussianAppendAdapter | None = None,
) -> SafeUncertainAppendProductionResult:
    """Append only already-safe Worklog-57 proposals, one candidate at a time.

    Adapter exceptions are reported as ``rolled_back`` only after the adapter's
    own strong transaction guarantee restores tensors, sidecar, owner registry,
    and ledger.  No catch path attempts a partial manual repair.
    """

    adapter = adapter or UncertainGaussianAppendAdapter()
    outcomes: list[SafeUncertainAppendAttempt] = []
    for source in safe_proposals.attempts:
        chart_id = None if source.chart is None else source.chart.chart_id
        batch = source.proposal
        if source.status != "proposed" or batch is None:
            outcomes.append(SafeUncertainAppendAttempt(
                source.candidate_id, chart_id, None, REJECTED,
                (f"safe_proposal_status:{source.status}",) + tuple(source.reasons), None, source,
            ))
            continue
        if initialization_provider is None:
            outcomes.append(SafeUncertainAppendAttempt(
                source.candidate_id, chart_id, batch.proposal_batch_id, REJECTED,
                ("appearance_initialization_required",), None, source,
            ))
            continue
        initialization = initialization_provider(batch, source)
        if initialization is None:
            outcomes.append(SafeUncertainAppendAttempt(
                source.candidate_id, chart_id, batch.proposal_batch_id, REJECTED,
                ("appearance_initialization_required",), None, source,
            ))
            continue
        try:
            receipt = adapter.append(batch, model, initialization)
        except Exception as exc:  # adapter has already rolled every transaction component back
            outcomes.append(SafeUncertainAppendAttempt(
                source.candidate_id, chart_id, batch.proposal_batch_id, ROLLED_BACK,
                (f"atomic_append_exception:{type(exc).__name__}:{exc}",), None, source,
            ))
            continue
        if receipt.append_state == "appended":
            status, reasons = APPENDED, ()
        elif "duplicate_proposal_batch" in receipt.reasons:
            status, reasons = DUPLICATE, receipt.reasons
        else:
            status, reasons = REJECTED, receipt.reasons
        outcomes.append(SafeUncertainAppendAttempt(
            source.candidate_id, chart_id, batch.proposal_batch_id, status, tuple(reasons), receipt, source,
        ))
    return SafeUncertainAppendProductionResult(safe_proposals, tuple(outcomes), adapter)


def run_safe_uncertain_proposals_and_append_from_gaussians(
    positions: Any,
    *,
    model: Any,
    initialization_provider: InitializationProvider | None,
    adapter: UncertainGaussianAppendAdapter | None = None,
    **safe_proposal_kwargs: Any,
) -> GaussianToAtomicUncertainAppendResult:
    """Single production call: raw Gaussian evidence through atomic append."""

    safe = run_safe_uncertain_proposals_from_gaussians(positions, **safe_proposal_kwargs)
    appended = append_safe_uncertain_proposals(
        safe.production, model=model, initialization_provider=initialization_provider, adapter=adapter,
    )
    return GaussianToAtomicUncertainAppendResult(safe, appended)