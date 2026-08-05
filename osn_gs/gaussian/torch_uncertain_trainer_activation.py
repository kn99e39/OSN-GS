from __future__ import annotations

"""Trainer/optimizer activation for atomically-appended uncertain Gaussians.

Consumes ONLY ``appended`` receipts from Worklog 58's atomic append
(``osn_gs.gaussian.torch_safe_uncertain_append_production``). Never
re-derives, re-runs, or modifies Phase D-G or the append transaction itself
-- every call into ``UncertainGaussianAppendAdapter.append()``
(via the unmodified ``append_safe_uncertain_proposals``) is treated as an
opaque, trusted black box. This module's own job starts strictly AFTER a
successful model-only append and stops at bringing the model back into a
trainable state.

``UncertainGaussianAppendAdapter.append()`` requires ``model.optimizer is
None`` at call time (its own existing precondition, unmodified here). This
module owns the temporary optimizer detach/reattach AROUND that call and the
actual optimizer-state EXTENSION afterward -- reusing
``TorchGaussianModel``'s own existing ``_preserve_optimizer_state``
grow-in-place logic (the same helper ``append_gaussians_raw``'s ADC
clone/split path already relies on, unmodified), never resetting or
discarding existing Adam moments for pre-existing rows. New rows always
start with fresh (zero) Adam state.

Worklog 60 (this file's current round) added two things worklog 59
deliberately left out:

1. **Composite append+activate transaction** (``append_and_activate`` /
   ``run_safe_uncertain_proposals_append_and_activate``): each candidate's
   append and its immediate activation are now treated as ONE atomic unit.
   If activation fails, the append that preceded it is undone too (model
   tensors, adapter-owned provenance sidecar, model-owned owner registry,
   model-owned batch-ID ledger, optimizer) -- restored to the exact
   pre-append snapshot, never leaving a "row exists but nothing references
   it correctly" half-registered state. The production-level result reports
   this as ``ROLLED_BACK``. The lower-level, non-composite
   ``activate_appended_receipts`` (append-only, activation-only,
   optimizer-rollback-only) still exists and can still report the narrower
   ``APPENDED_INACTIVE`` state for a caller that explicitly wants that
   (e.g. a caller managing its own outer transaction).

2. **True row-level training isolation** (``masked_optimizer_step``): a
   from-scratch Adam update applied ONLY to a selected row mask -- for every
   OTHER row, neither the parameter value NOR its Adam moments
   (``exp_avg``/``exp_avg_sq``) are read or written at all. A plain
   ``torch.optim.Adam.step()`` call, even with an exactly-zero gradient on
   excluded rows, still lets their PRE-EXISTING momentum decay them
   (``m <- beta1*m + (1-beta1)*0``) -- that is real, expected Adam behavior,
   but it is NOT isolation. ``masked_optimizer_step`` never even evaluates
   that decay for excluded rows.

Visible/uncertain training separation itself is not reimplemented: every row
this module activates was already marked ``is_uncertain=True`` by the append
adapter (``append_gaussians_model_only``'s own ``uncertain_mask`` argument,
unmodified), and this module's own masked step is what a caller now uses to
train exactly the activated rows (or exactly the visible rows) without
touching the other side's value or momentum at all.

``TorchOSNGSTrainer.activate_and_train_uncertain_step`` (``osn_gs.core.torch_trainer``)
is the real production connection point: it renders through the SAME
rasterizer/loss path ``_train_loop`` uses, then applies
``masked_optimizer_step`` restricted to this call's newly-activated rows.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.gaussian.torch_safe_uncertain_append_production import (
    APPENDED,
    InitializationProvider,
    SafeUncertainAppendAttempt,
    SafeUncertainAppendProductionResult,
    append_safe_uncertain_proposals,
    run_safe_uncertain_proposals_from_gaussians,
)
from osn_gs.gaussian.torch_uncertain_append_adapter import UncertainGaussianAppendAdapter
from osn_gs.surface.torch_safe_uncertain_proposal_production import SafeUncertainProposalProductionResult
from osn_gs.utils.torch_ops import require_torch

ACTIVATED = "activated"
NOT_ACTIVATED = "not_activated"
APPENDED_INACTIVE = "appended_inactive"  # lower-level (`activate_appended_receipts`) only
ROLLED_BACK = "rolled_back"  # production-level (`append_and_activate`) composite-rollback outcome
ACTIVATION_STATES = {ACTIVATED, NOT_ACTIVATED, APPENDED_INACTIVE, ROLLED_BACK}


@dataclass(frozen=True)
class ActivationAttempt:
    candidate_id: str
    proposal_batch_id: str | None
    status: str
    reasons: tuple[str, ...]
    activated_row_count: int
    append_attempt: SafeUncertainAppendAttempt
    activated_index_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.status not in ACTIVATION_STATES:
            raise ValueError(f"Unknown activation status: {self.status!r}")

    def payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "proposal_batch_id": self.proposal_batch_id,
            "status": self.status,
            "reasons": list(self.reasons),
            "activated_row_count": self.activated_row_count,
            "activated_index_range": self.activated_index_range,
            "append": self.append_attempt.payload(),
        }


@dataclass(frozen=True)
class TrainerActivationResult:
    append_result: SafeUncertainAppendProductionResult
    attempts: tuple[ActivationAttempt, ...]

    def diagnostic_summary(self) -> dict[str, Any]:
        return {
            "attempt_count": len(self.attempts),
            "activated_count": sum(item.status == ACTIVATED for item in self.attempts),
            "not_activated_count": sum(item.status == NOT_ACTIVATED for item in self.attempts),
            "appended_inactive_count": sum(item.status == APPENDED_INACTIVE for item in self.attempts),
            "rolled_back_count": sum(item.status == ROLLED_BACK for item in self.attempts),
            "activated_row_total": sum(item.activated_row_count for item in self.attempts),
        }

    def activated_row_mask(self, model: Any) -> Any:
        """Boolean mask, length ``len(model)``, True for every row this
        call's ``ACTIVATED`` attempts newly activated -- the row set a
        subsequent ``masked_optimizer_step`` should train."""

        torch = require_torch()
        mask = torch.zeros((len(model),), dtype=torch.bool, device=model.device)
        for item in self.attempts:
            if item.status == ACTIVATED and item.activated_index_range is not None:
                start, end = item.activated_index_range
                mask[start:end] = True
        return mask


def _snapshot_optimizer(optimizer: Any) -> Any:
    """Cheap structural snapshot: ``_preserve_optimizer_state`` only ever
    reassigns ``group["params"]`` (never mutates the list in place) and
    add/removes ``optimizer.state`` dict entries (never mutates an existing
    inner state dict in place) -- so a shallow copy of each group dict and
    of the outer state dict is a complete, correct undo point."""

    return (
        [dict(group) for group in optimizer.param_groups],
        dict(optimizer.state),
    )


def _restore_optimizer(optimizer: Any, snapshot: Any) -> None:
    group_snapshots, state_snapshot = snapshot
    for group, snap in zip(optimizer.param_groups, group_snapshots):
        group["params"] = snap["params"]
    optimizer.state.clear()
    optimizer.state.update(state_snapshot)


def activate_appended_rows(model: Any, *, old_optimizer: Any, old_count: int) -> tuple[bool, tuple[str, ...]]:
    """Extend ``old_optimizer`` in place to also cover ``model``'s
    newly-appended rows (``model`` must already have been grown by the
    append transaction before this is called).

    Returns ``(succeeded, reasons)``. On success ``model.optimizer`` is the
    (same object, now-extended) ``old_optimizer`` -- every ``param_groups``
    entry already references ``model``'s CURRENT parameter objects (identity
    match), since it reads them fresh via ``_optimizer_named_params()``. On
    failure -- including ``old_optimizer is None`` (no prior training
    session to extend) -- ``model.optimizer`` is left exactly as
    ``old_optimizer`` was before this call (restored via
    ``_snapshot_optimizer``/``_restore_optimizer`` if a partial mutation was
    attempted), never a half-updated param_groups/state.
    """

    if old_optimizer is None:
        model.optimizer = None
        return False, ("no_prior_optimizer_first_activation_requires_training_setup",)

    snapshot = _snapshot_optimizer(old_optimizer)
    old_params = {
        group.get("name"): group["params"][0]
        for group in old_optimizer.param_groups
        if group.get("params")
    }
    model.optimizer = old_optimizer
    try:
        model._preserve_optimizer_state(old_params, None, int(old_count))
    except Exception as exc:  # noqa: BLE001 - fail-closed, exact pre-call state restored below
        _restore_optimizer(old_optimizer, snapshot)
        model.optimizer = old_optimizer
        return False, (f"optimizer_sync_failed:{type(exc).__name__}:{exc}",)
    return True, ()


def activate_appended_receipts(
    model: Any, appended: SafeUncertainAppendProductionResult, old_optimizer: Any,
) -> TrainerActivationResult:
    """Lower-level, append-only entry point: activates every already-committed
    ``appended`` receipt in ``appended`` WITHOUT undoing the append itself on
    an activation failure -- only the optimizer state rolls back, and the
    attempt is reported ``APPENDED_INACTIVE`` (row exists, not yet
    trainable). Production callers should use ``append_and_activate``
    instead, which composes this with the append call itself into one
    all-or-nothing transaction (``ROLLED_BACK`` on failure).
    """

    attempts: list[ActivationAttempt] = []
    for item in appended.attempts:
        if item.status != APPENDED or item.receipt is None:
            model.optimizer = old_optimizer
            attempts.append(ActivationAttempt(
                item.candidate_id, item.proposal_batch_id, NOT_ACTIVATED,
                (f"append_status:{item.status}",), 0, item,
            ))
            continue
        old_count = item.receipt.model_count_before
        succeeded, reasons = activate_appended_rows(model, old_optimizer=old_optimizer, old_count=old_count)
        if succeeded:
            attempts.append(ActivationAttempt(
                item.candidate_id, item.proposal_batch_id, ACTIVATED, (),
                item.receipt.appended_sample_count, item, item.receipt.appended_index_range,
            ))
            old_optimizer = model.optimizer
        else:
            attempts.append(ActivationAttempt(
                item.candidate_id, item.proposal_batch_id, APPENDED_INACTIVE, reasons, 0, item,
            ))
    return TrainerActivationResult(appended, tuple(attempts))


@dataclass
class _TransactionSnapshot:
    model_state: dict[str, Any]
    sidecar: dict[str, dict[str, Any]]
    ledger: frozenset
    owner_registry: dict[int, str]
    optimizer_state_by_name: dict[str, dict[str, Any]] | None


def _snapshot_transaction_state(
    model: Any, adapter: UncertainGaussianAppendAdapter, optimizer: Any,
) -> _TransactionSnapshot:
    """``optimizer`` must be explicitly passed as whatever optimizer was
    active BEFORE this candidate's append -- callers detach
    ``model.optimizer`` to satisfy the append precondition immediately
    around this call, so reading ``model.optimizer`` here would always see
    ``None``."""

    state_by_name = None
    if optimizer is not None:
        state_by_name = {
            group.get("name"): dict(optimizer.state.get(group["params"][0], {}))
            for group in optimizer.param_groups
            if group.get("params")
        }
    return _TransactionSnapshot(
        model_state=model.snapshot_state(),
        sidecar=dict(adapter._sidecar),
        ledger=frozenset(model.appended_uncertain_batch_ids),
        owner_registry=dict(model.occluded_chart_owner_registry),
        optimizer_state_by_name=state_by_name,
    )


def _restore_transaction_state(
    model: Any, adapter: UncertainGaussianAppendAdapter, snapshot: _TransactionSnapshot, *, old_optimizer: Any,
) -> None:
    """Undo everything a (committed) append + failed activation touched.

    ``model.restore_state()`` always builds BRAND NEW ``nn.Parameter``
    objects (never reuses the ones that existed at snapshot time), so any
    optimizer state must be re-keyed by the stable param-group ``name`` --
    never by the old (now-stale) parameter object identity -- to keep
    ``optimizer.param_groups[i]["params"][0] is model.<param>`` true again
    after restore.
    """

    model.restore_state(snapshot.model_state)
    adapter._sidecar.clear()
    adapter._sidecar.update(snapshot.sidecar)
    model.appended_uncertain_batch_ids.clear()
    model.appended_uncertain_batch_ids.update(snapshot.ledger)
    model.occluded_chart_owner_registry.clear()
    model.occluded_chart_owner_registry.update(snapshot.owner_registry)
    if old_optimizer is None:
        model.optimizer = None
        return
    model.optimizer = old_optimizer
    current = model._optimizer_named_params()
    old_optimizer.state.clear()
    for group in old_optimizer.param_groups:
        name = group.get("name")
        new_param = current.get(name)
        if new_param is None:
            continue
        group["params"] = [new_param]
        state = (snapshot.optimizer_state_by_name or {}).get(name)
        if state:
            old_optimizer.state[new_param] = state


def append_and_activate(
    safe_proposals: SafeUncertainProposalProductionResult,
    *,
    model: Any,
    initialization_provider: InitializationProvider,
    adapter: UncertainGaussianAppendAdapter | None = None,
) -> TrainerActivationResult:
    """The production entry point: append + activate as ONE composite
    transaction per candidate. If activation fails, that candidate's append
    is rolled back too (model tensors, sidecar, owner registry, ledger,
    optimizer) -- never a half-registered row. Reported as ``ROLLED_BACK``.
    """

    adapter = adapter or UncertainGaussianAppendAdapter()
    old_optimizer = model.optimizer
    attempts: list[ActivationAttempt] = []
    append_attempts: list[SafeUncertainAppendAttempt] = []
    for source in safe_proposals.attempts:
        txn_snapshot = _snapshot_transaction_state(model, adapter, old_optimizer)
        model.optimizer = None  # required precondition for the (unaudited) append transaction
        single_batch = SafeUncertainProposalProductionResult(safe_proposals.bridge, (source,))
        appended = append_safe_uncertain_proposals(
            single_batch, model=model, initialization_provider=initialization_provider, adapter=adapter,
        )
        append_attempt = appended.attempts[0]
        append_attempts.append(append_attempt)

        if append_attempt.status != APPENDED or append_attempt.receipt is None:
            model.optimizer = old_optimizer
            attempts.append(ActivationAttempt(
                append_attempt.candidate_id, append_attempt.proposal_batch_id, NOT_ACTIVATED,
                (f"append_status:{append_attempt.status}",), 0, append_attempt,
            ))
            continue

        old_count = append_attempt.receipt.model_count_before
        succeeded, reasons = activate_appended_rows(model, old_optimizer=old_optimizer, old_count=old_count)
        if succeeded:
            attempts.append(ActivationAttempt(
                append_attempt.candidate_id, append_attempt.proposal_batch_id, ACTIVATED, (),
                append_attempt.receipt.appended_sample_count, append_attempt,
                append_attempt.receipt.appended_index_range,
            ))
            old_optimizer = model.optimizer
        else:
            _restore_transaction_state(model, adapter, txn_snapshot, old_optimizer=old_optimizer)
            attempts.append(ActivationAttempt(
                append_attempt.candidate_id, append_attempt.proposal_batch_id, ROLLED_BACK,
                ("activation_failed_full_rollback",) + reasons, 0, append_attempt,
            ))

    combined_append_result = SafeUncertainAppendProductionResult(safe_proposals, tuple(append_attempts), adapter)
    return TrainerActivationResult(combined_append_result, tuple(attempts))


def run_safe_uncertain_proposals_append_and_activate(
    positions: Any,
    *,
    model: Any,
    initialization_provider: InitializationProvider,
    adapter: UncertainGaussianAppendAdapter | None = None,
    **safe_proposal_kwargs: Any,
) -> TrainerActivationResult:
    """Single production call: raw Gaussian evidence -> Worklog 57 safe
    proposal -> composite append+activate transaction (see
    ``append_and_activate``)."""

    safe = run_safe_uncertain_proposals_from_gaussians(positions, **safe_proposal_kwargs)
    return append_and_activate(
        safe.production, model=model, initialization_provider=initialization_provider, adapter=adapter,
    )


def masked_optimizer_step(model: Any, row_mask: Any) -> dict[str, Any]:
    """Apply exactly one Adam update restricted to ``row_mask`` -- for every
    OTHER row, neither the parameter value NOR its Adam moments
    (``exp_avg``/``exp_avg_sq``) are read or written at all.

    This is NOT ``torch.optim.Adam.step()`` (which always updates every row
    uniformly, and lets an excluded row's pre-existing momentum keep
    decaying it even at exactly-zero gradient) -- it reimplements Adam's
    per-row update equations directly, gated by ``row_mask``, so isolation
    holds for BOTH the parameter value and the optimizer's own internal
    state. The step/bias-correction counter stays a single shared scalar per
    parameter (matching ``TorchGaussianModel``'s existing Adam state layout,
    including how ``_preserve_optimizer_state`` already treats it) --
    equivalent in spirit to ``torch.optim.SparseAdam``'s own convention of a
    shared step counter with per-row-gated moment updates.

    Requires ``model.optimizer`` to already be active (post-activation).
    Groups whose parameter has no gradient this call are left completely
    untouched (not even a state dict is created for them).
    """

    torch = require_torch()
    optimizer = model.optimizer
    if optimizer is None:
        raise RuntimeError("masked_optimizer_step_requires_active_optimizer")
    mask = torch.as_tensor(row_mask, dtype=torch.bool, device=model.device)
    idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
    touched_names: list[str] = []
    if idx.numel() == 0:
        return {"touched_param_groups": touched_names, "touched_row_count": 0}

    for group in optimizer.param_groups:
        params = group.get("params") or []
        if not params:
            continue
        (param,) = params
        if param.grad is None:
            continue
        state = optimizer.state.setdefault(param, {})
        if "exp_avg" not in state:
            state["step"] = torch.zeros((), dtype=torch.float64, device=param.device)
            state["exp_avg"] = torch.zeros_like(param.data)
            state["exp_avg_sq"] = torch.zeros_like(param.data)
        beta1, beta2 = group.get("betas", (0.9, 0.999))
        eps = float(group.get("eps", 1e-8))
        lr = float(group["lr"])

        step = state["step"] + 1
        state["step"] = step
        grad_selected = param.grad[idx]
        exp_avg_selected = state["exp_avg"][idx]
        exp_avg_sq_selected = state["exp_avg_sq"][idx]

        new_exp_avg = beta1 * exp_avg_selected + (1.0 - beta1) * grad_selected
        new_exp_avg_sq = beta2 * exp_avg_sq_selected + (1.0 - beta2) * grad_selected * grad_selected
        bias_correction1 = 1.0 - beta1 ** float(step)
        bias_correction2 = 1.0 - beta2 ** float(step)
        step_size = lr / bias_correction1
        denom = (new_exp_avg_sq / bias_correction2).sqrt().add_(eps)
        update = step_size * new_exp_avg / denom

        with torch.no_grad():
            param.data[idx] = param.data[idx] - update
        state["exp_avg"][idx] = new_exp_avg
        state["exp_avg_sq"][idx] = new_exp_avg_sq
        touched_names.append(group.get("name"))

    return {"touched_param_groups": touched_names, "touched_row_count": int(idx.numel())}


def run_one_training_step(model: Any, *, loss_fn) -> dict[str, Any]:
    """Minimal forward/backward/optimizer-step harness, independent of
    ``TorchOSNGSTrainer``'s real image-rendering loop -- proves the activated
    optimizer is genuinely differentiable and steps correctly, nothing more.
    Applies a REGULAR (unmasked) ``torch.optim.Adam.step()`` -- for row-level
    isolated training use ``masked_optimizer_step`` instead.
    ``loss_fn(model) -> scalar tensor``.
    """

    if model.optimizer is None:
        raise RuntimeError("run_one_training_step_requires_active_optimizer")
    model.optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model)
    loss.backward()
    model.optimizer.step()
    return {"loss": float(loss.detach().cpu())}


def run_safe_uncertain_proposals_append_activate_and_train_step(
    positions: Any,
    *,
    model: Any,
    initialization_provider: InitializationProvider,
    loss_fn,
    adapter: UncertainGaussianAppendAdapter | None = None,
    masked: bool = True,
    **safe_proposal_kwargs: Any,
) -> tuple[TrainerActivationResult, dict[str, Any] | None]:
    """The full single production entry point: raw Gaussian evidence ->
    proposal -> composite atomic append+activate -> one training step.

    ``masked=True`` (default) restricts the step to exactly this call's
    newly-activated rows via ``masked_optimizer_step`` (true row-level
    isolation). ``masked=False`` runs a regular whole-model
    ``run_one_training_step`` instead.

    Returns ``(activation_result, step_result)``. ``step_result`` is
    ``None`` when nothing became trainable this call.
    """

    activation = run_safe_uncertain_proposals_append_and_activate(
        positions, model=model, initialization_provider=initialization_provider,
        adapter=adapter, **safe_proposal_kwargs,
    )
    if model.optimizer is None:
        return activation, None
    if not masked:
        return activation, run_one_training_step(model, loss_fn=loss_fn)
    row_mask = activation.activated_row_mask(model)
    if not bool(row_mask.any()):
        return activation, None
    model.optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model)
    loss.backward()
    step_stats = masked_optimizer_step(model, row_mask)
    return activation, {"loss": float(loss.detach().cpu()), **step_stats}
