"""Model-only append boundary for approved uncertain Gaussian proposals.

Gate follow-up hardening (docs/worklogs/88_uncertain_gaussian_append_adapter_foundation.md,
docs/worklogs/96_occluded_chart_ownership_foundation.md): this module owns a
single logical transaction across four independent pieces of state --

    model tensor state (TorchGaussianModel)
    proposal provenance sidecar (self._sidecar, adapter-owned)
    occluded-chart owner registry (model.occluded_chart_owner_registry, MODEL-owned)
    appended batch ID ledger (model.appended_uncertain_batch_ids, MODEL-owned)

-- and guarantees all four only ever change together, never partially.

**Ledger ownership and lifecycle**: the duplicate-append ledger is owned by
the `TorchGaussianModel` INSTANCE (see `appended_uncertain_batch_ids` on that
class), not by any `UncertainGaussianAppendAdapter` instance. In-process,
in-memory only -- there is no checkpoint-level persistence. Concretely:

- Duplicate protection holds for ANY adapter instance operating on the SAME
  model object: two different `UncertainGaussianAppendAdapter()` instances
  both consulting the same model will agree on which batch IDs are already
  appended.
- Two DIFFERENT model objects have completely independent ledgers, even for
  the identical batch ID -- this is intentional (a batch ID's unit of
  identity is "already appended to this particular model", not global).
  There is no explicit ledger-reset API; a batch id's presence is tied to the
  model Python object's lifetime, so constructing a new `TorchGaussianModel`
  starts with an empty ledger. Checkpoint save/load persistence of this set
  is a deferred gap (see the worklog's "not started" list below).
- The provenance sidecar (full per-batch chart/candidate/patch/domain/
  boundary IDs, sample IDs, initialization digest) remains adapter-instance-
  owned: it is a convenience trace of what calls THIS adapter instance made,
  not a duplicate-protection mechanism, so unlike the ledger it does not need
  to be shared across adapter instances to satisfy the duplicate contract.

**Receipt vs. exception contract (strong guarantee)**: normal, expected
rejections (failed eligibility, missing provenance, an active optimizer, a
duplicate batch ID, missing initialization, non-finite/invalid proposal
parameters) always return an `UncertainAppendReceipt` with
`append_state="not_appended"` -- never an exception -- and are guaranteed to
leave the model, sidecar, and ledger completely untouched.

For an eligible proposal, the success `UncertainAppendReceipt` is built
COMPLETELY from pre-commit information (batch/preflight/conversion results
and precomputed before/after counts -- nothing that depends on the outcome
of the model/sidecar/ledger commits themselves) before any of those three
commits are attempted. This makes the contract airtight: either
  (a) an exception is raised before any commit began (nothing to roll back),
  (b) an exception is raised during a commit stage and the transaction is
      rolled back to its exact pre-call state before it propagates, or
  (c) all three commits succeed and the already-built receipt is returned.
There is no window where a commit has succeeded but `append()` still might
fail to produce a receipt -- receipt construction never happens after the
point of no return.

**Ownership (docs/worklogs Occluded Chart Ownership Foundation)**: every
Gaussian this adapter appends is assigned `surface_owner_kind =
SURFACE_OWNER_OCCLUDED_CHART` and a deterministic synthetic `surface_owner_id`
derived from the source occluded chart's own stable id (collision-checked
against the target model's own registry -- a truncated hash is
collision-resistant, not collision-proof, so a genuine collision raises
`OccludedChartOwnerCollisionError` instead of silently aliasing two charts).
`cluster_id = min(source_patch_ids)` is still recorded (compatibility only,
see `CLUSTER_ID_PROJECTION_RULE`) but is NEVER the behavioral owner -- the
`surface_owner_kind`/`surface_owner_id` pair is. Full `source_patch_ids`
provenance is preserved unchanged in the sidecar.

**Owner registry is its own transaction stage (Gate final-contract round)**:
projection/validation and the actual registry write are split into three
separate functions in `osn_gs.gaussian.torch_surface_ownership` --
`validate_occluded_owner_binding_read_only` (pure: projects the owner id and
checks for a collision against a DIFFERENT chart, but never mutates the
registry), `commit_occluded_owner_binding` (the actual write), and
`rollback_occluded_owner_binding` (undoes a write, but ONLY if this
transaction newly created it). `_convert()` (transaction stage 1) calls only
the read-only validator, so a later commit-stage failure can never leave the
registry mutated while the model/sidecar/ledger roll back -- the registry
write itself is `append()`'s own explicit stage, positioned after the sidecar
commit and before the ledger commit, with matching rollback wired into every
later stage's exception handler.

Not started here or anywhere in this module: optimizer state expansion,
trainer/renderer/checkpoint integration, appearance/opacity estimation
policy, review workflow, conflict resolution, global ranking, checkpoint-
persistent ledger.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from osn_gs.gaussian.torch_surface_ownership import (
    SURFACE_OWNER_OCCLUDED_CHART,
    commit_occluded_owner_binding,
    rollback_occluded_owner_binding,
    validate_occluded_owner_binding_read_only,
)
from osn_gs.surface.torch_uncertain_gaussian_proposal import ELIGIBLE, UncertainGaussianProposalBatch

SUPPORTED_PROPOSAL_SCHEMA_VERSIONS = frozenset({1})

CLUSTER_ID_PROJECTION_RULE = "min_source_patch_id"


@dataclass(frozen=True)
class UncertainAppendInitialization:
    """Explicit values from an authorized appearance/opacity initialization policy."""

    features_dc: Any
    features_rest: Any
    opacity_logits: Any
    uncertain_confidence_logits: Any


@dataclass(frozen=True)
class UncertainAppendPreflight:
    allowed: bool
    reasons: tuple[str, ...]
    valid_sample_count: int
    target_model_compatible: bool
    required_conversions: tuple[str, ...]


@dataclass(frozen=True)
class UncertainAppendReceipt:
    proposal_batch_id: str
    source_chart_id: str
    requested_sample_count: int
    valid_sample_count: int
    appended_sample_count: int
    rejected_sample_count: int
    model_count_before: int
    model_count_after: int
    appended_index_range: tuple[int, int] | None
    appended_sample_ids: tuple[str, ...]
    conversion_summary: tuple[str, ...]
    append_state: str
    reasons: tuple[str, ...]
    cluster_id: int | None = None
    cluster_id_projection_rule: str | None = None
    initialization_digest: str | None = None
    surface_owner_kind: int | None = None
    surface_owner_id: int | None = None

    def stable_payload(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def stable_json(self) -> str:
        return json.dumps(self.stable_payload(), sort_keys=True, separators=(",", ":"))


def _project_cluster_id(source_patch_ids: Sequence[int]) -> int:
    """Deterministic, input-order-independent canonical cluster id for a
    multi-patch proposal.

    No "canonical owner patch" convention exists anywhere else in the
    codebase to reuse -- Phase E/F candidate/chart provenance stores an
    unordered pair of supporting patch ids (ordered only by incidental
    domain-id string sort, not by patch identity). This adapter therefore
    defines its own explicit rule instead of taking ``source_patch_ids[0]``:
    the numerically smallest patch id. Sorting-then-taking-min makes the
    result independent of whatever order the upstream candidate/chart
    pairing happened to store the two patches in.
    """

    ids = [int(p) for p in source_patch_ids]
    if not ids:
        raise ValueError("source_patch_ids must be non-empty to project a cluster id")
    return min(ids)


def _initialization_digest(initialization: UncertainAppendInitialization) -> str:
    """Content-addressed identity of the appearance/opacity initialization
    actually used, for receipt/provenance traceability without duplicating
    the full tensor values (mirrors the label+shape+dtype+bytes convention
    already used by ``torch_observation_evidence._tensor_digest``)."""

    pieces: list[bytes] = []
    for name in ("features_dc", "features_rest", "opacity_logits", "uncertain_confidence_logits"):
        value = torch.as_tensor(getattr(initialization, name)).detach().cpu().contiguous()
        header = f"{name}|{tuple(value.shape)}|{value.dtype}".encode("utf-8")
        pieces.append(header + value.numpy().tobytes())
    return hashlib.sha256(b"".join(pieces)).hexdigest()[:16]


@dataclass
class _ConvertedAppend:
    xyz: Any
    features_dc: Any
    features_rest: Any
    opacity: Any
    scaling: Any
    rotation: Any
    uncertain_confidence: Any
    uv: Any
    cluster_ids: Any
    uncertain_mask: Any
    sample_ids: tuple[str, ...]
    cluster_id: int
    surface_owner_kind: Any
    surface_owner_id: Any
    owner_id_scalar: int
    owner_binding_preexisted: bool


class UncertainGaussianAppendAdapter:
    """Transactional model-only append boundary. See the module docstring for
    the ledger-ownership and receipt-vs-exception contracts."""

    def __init__(self) -> None:
        self._sidecar: dict[str, dict[str, Any]] = {}

    @property
    def provenance_sidecar(self) -> dict[str, dict[str, Any]]:
        return {key: value.copy() for key, value in self._sidecar.items()}

    def preflight(
        self,
        batch: UncertainGaussianProposalBatch,
        model: Any,
        initialization: UncertainAppendInitialization | None = None,
    ) -> UncertainAppendPreflight:
        reasons: list[str] = []
        meta = batch.metadata
        requested = len(batch.sample_ids)
        valid = int(torch.as_tensor(batch.valid_mask, dtype=torch.bool).sum().item())
        if meta.get("eligibility") != ELIGIBLE:
            reasons.append("proposal_not_eligible")
        if batch.append_state != "not_appended" or meta.get("append_state") != "not_appended":
            reasons.append("proposal_already_appended")
        if batch.schema_version not in SUPPORTED_PROPOSAL_SCHEMA_VERSIONS:
            reasons.append("unsupported_proposal_schema")
        if (
            not isinstance(batch.proposal_batch_id, str)
            or not batch.proposal_batch_id
            or len(batch.sample_ids) != requested
            or any(not isinstance(x, str) or not x for x in batch.sample_ids)
        ):
            reasons.append("invalid_proposal_identifiers")
        if batch.proposal_batch_id in model.appended_uncertain_batch_ids:
            reasons.append("duplicate_proposal_batch")
        if any(
            not meta.get(key)
            for key in (
                "source_chart_id", "source_candidate_id", "source_patch_ids",
                "supporting_domain_ids", "supporting_boundary_ids",
            )
        ):
            reasons.append("proposal_provenance_missing")
        if "full_known_free_contradiction" in meta.get("safety_reasons", []):
            reasons.append("known_free_contradiction")
        if valid <= 0:
            reasons.append("no_valid_samples")
        compatible = getattr(model, "optimizer", None) is None
        if not compatible:
            reasons.append("model_only_append_requires_no_optimizer")
        if initialization is None:
            reasons.append("appearance_initialization_required")
        else:
            mask = torch.as_tensor(batch.valid_mask, dtype=torch.bool)
            values = (batch.position, batch.rotation_quaternion, batch.linear_scale)
            if any(not bool(torch.isfinite(torch.as_tensor(value)[mask]).all()) for value in values):
                reasons.append("nonfinite_proposal_parameter")
            original_scale = torch.as_tensor(batch.linear_scale)[mask]
            if not bool((original_scale > 0).all()):
                reasons.append("nonpositive_proposal_scale")
            else:
                # A scale that is positive in the proposal's own dtype can still
                # underflow to exactly 0 once cast to the model's float32
                # storage, which would make log(scale) = -inf. Reject that
                # case explicitly instead of silently producing a non-finite
                # model parameter.
                cast_scale = original_scale.to(torch.float32)
                if not bool(torch.isfinite(torch.log(cast_scale)).all()):
                    reasons.append("proposal_scale_below_representable_minimum")
            quat = torch.as_tensor(batch.rotation_quaternion)[mask]
            if valid > 0 and not bool(
                torch.allclose(
                    torch.linalg.vector_norm(quat, dim=1),
                    torch.ones((valid,), dtype=quat.dtype),
                    atol=1e-4, rtol=1e-4,
                )
            ):
                reasons.append("unnormalized_proposal_quaternion")
        return UncertainAppendPreflight(
            not reasons,
            tuple(sorted(set(reasons))),
            valid,
            compatible,
            ("linear_scale_to_log_scaling", "canonical_quaternion_to_raw_rotation", "valid_mask_filter", "dtype_device_alignment"),
        )

    def _convert(
        self,
        batch: UncertainGaussianProposalBatch,
        model: Any,
        initialization: UncertainAppendInitialization,
    ) -> _ConvertedAppend:
        """Pure conversion: builds every tensor/id the model append needs,
        including slicing the initialization policy's appearance/opacity
        values down to the same valid-mask subset and ordering as the
        proposal geometry. Raises on malformed input (e.g. an
        initialization tensor shaped for a different sample count) BEFORE
        any model mutation -- transaction stage 1, conversion failure."""

        mask = torch.as_tensor(batch.valid_mask, dtype=torch.bool)
        indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        device, tm = model.device, model.torch
        count = int(indices.numel())

        def _slice(value: Any) -> Any:
            return torch.as_tensor(value)[indices].to(dtype=tm.float32, device=device)

        xyz = _slice(batch.position)
        scaling = torch.log(_slice(batch.linear_scale))
        rotation = _slice(batch.rotation_quaternion)
        uv = _slice(batch.uv)
        features_dc = _slice(initialization.features_dc)
        features_rest = _slice(initialization.features_rest)
        opacity = _slice(initialization.opacity_logits)
        uncertain_confidence = _slice(initialization.uncertain_confidence_logits)
        cluster_id = _project_cluster_id(batch.metadata["source_patch_ids"])
        cluster_ids = torch.full((count,), cluster_id, dtype=tm.long, device=device)
        uncertain_mask = torch.ones((count,), dtype=torch.bool, device=device)
        sample_ids = tuple(batch.sample_ids[i] for i in indices.detach().cpu().tolist())

        # Canonical ownership (NOT the cluster_id compatibility projection
        # above): every appended Gaussian is owned by its source occluded
        # chart, never by one of the two visible patches it bridges. Uses
        # the READ-ONLY validator only -- a hash collision against a
        # DIFFERENT chart already bound to this model raises immediately
        # (transaction stage 1 -- nothing has touched the model OR the
        # registry if this raises). The actual registry write happens later,
        # as its own explicit commit stage in `append()`.
        owner_id_scalar, owner_binding_preexisted = validate_occluded_owner_binding_read_only(
            model, batch.metadata["source_chart_id"]
        )
        surface_owner_kind = torch.full((count,), SURFACE_OWNER_OCCLUDED_CHART, dtype=tm.long, device=device)
        surface_owner_id = torch.full((count,), owner_id_scalar, dtype=tm.long, device=device)

        return _ConvertedAppend(
            xyz=xyz, features_dc=features_dc, features_rest=features_rest, opacity=opacity,
            scaling=scaling, rotation=rotation, uncertain_confidence=uncertain_confidence,
            uv=uv, cluster_ids=cluster_ids, uncertain_mask=uncertain_mask,
            sample_ids=sample_ids, cluster_id=cluster_id,
            surface_owner_kind=surface_owner_kind, surface_owner_id=surface_owner_id,
            owner_id_scalar=owner_id_scalar, owner_binding_preexisted=owner_binding_preexisted,
        )

    def _commit_sidecar(self, batch_id: str, entry: dict[str, Any]) -> None:
        self._sidecar[batch_id] = entry

    def _commit_ledger(self, model: Any, batch_id: str) -> None:
        # Model-owned: see the module docstring's ledger-ownership contract.
        model.appended_uncertain_batch_ids.add(batch_id)

    def _rollback_sidecar(self, batch_id: str) -> None:
        self._sidecar.pop(batch_id, None)

    def _commit_owner_registry(self, model: Any, owner_id: int, source_chart_id: str) -> None:
        commit_occluded_owner_binding(model, owner_id, source_chart_id)

    def _rollback_owner_registry(self, model: Any, owner_id: int, *, was_preexisting: bool) -> None:
        rollback_occluded_owner_binding(model, owner_id, was_preexisting=was_preexisting)

    def _build_sidecar_entry(
        self, batch: UncertainGaussianProposalBatch, converted: _ConvertedAppend,
        init_digest: str, before: int, after: int,
    ) -> dict[str, Any]:
        return {
            "proposal_batch_id": batch.proposal_batch_id,
            "proposal_sample_ids": converted.sample_ids,
            "source_chart_id": batch.metadata["source_chart_id"],
            "source_candidate_id": batch.metadata["source_candidate_id"],
            "source_patch_ids": tuple(batch.metadata["source_patch_ids"]),
            "supporting_domain_ids": tuple(batch.metadata["supporting_domain_ids"]),
            "supporting_boundary_ids": tuple(batch.metadata["supporting_boundary_ids"]),
            "append_origin": "uncertain_gaussian_append_adapter",
            "initialization_digest": init_digest,
            "cluster_id": converted.cluster_id,
            "cluster_id_projection_rule": CLUSTER_ID_PROJECTION_RULE,
            "surface_owner_kind": SURFACE_OWNER_OCCLUDED_CHART,
            "surface_owner_id": converted.owner_id_scalar,
            "appended_index_range": (before, after),
        }

    def _rejection_receipt(
        self, batch: UncertainGaussianProposalBatch, preflight: UncertainAppendPreflight,
        requested: int, before: int,
    ) -> UncertainAppendReceipt:
        return UncertainAppendReceipt(
            proposal_batch_id=batch.proposal_batch_id,
            source_chart_id=str(batch.metadata.get("source_chart_id", "")),
            requested_sample_count=requested,
            valid_sample_count=preflight.valid_sample_count,
            appended_sample_count=0,
            rejected_sample_count=requested,
            model_count_before=before,
            model_count_after=before,
            appended_index_range=None,
            appended_sample_ids=(),
            conversion_summary=(),
            append_state="not_appended",
            reasons=preflight.reasons,
        )

    def append(
        self,
        batch: UncertainGaussianProposalBatch,
        model: Any,
        initialization: UncertainAppendInitialization | None = None,
    ) -> UncertainAppendReceipt:
        """Run the full preflight -> convert -> commit transaction for one
        proposal batch. See the module docstring for the ledger-ownership and
        strong receipt-vs-exception contracts."""

        preflight = self.preflight(batch, model, initialization)
        requested, before = len(batch.sample_ids), len(model)
        if not preflight.allowed:
            return self._rejection_receipt(batch, preflight, requested, before)

        assert initialization is not None  # guaranteed by preflight.allowed above
        # Stage 1: pure conversion + fully-formed success receipt/sidecar
        # entry, built ENTIRELY from pre-commit information. Nothing here
        # touches the model, sidecar, owner registry, or ledger; any raise
        # (malformed initialization shape, an owner-registry collision
        # against a different chart, or even a defect in receipt/sidecar-
        # entry construction itself) leaves all four completely untouched,
        # because none of the four commits below have started yet.
        converted = self._convert(batch, model, initialization)
        init_digest = _initialization_digest(initialization)
        count = int(converted.xyz.shape[0])
        after = before + count
        receipt = self._build_receipt(batch, preflight, converted, requested, before, after, init_digest)
        sidecar_entry = self._build_sidecar_entry(batch, converted, init_digest, before, after)

        # Stage 2: model commit. Snapshot first so any exception here (the
        # commit itself is NOT atomic -- see append_gaussians_model_only's
        # docstring) can be fully undone.
        snapshot = model.snapshot_state()
        try:
            model.append_gaussians_model_only(
                converted.xyz, converted.features_dc, converted.features_rest, converted.opacity,
                converted.scaling, converted.rotation, converted.uncertain_confidence,
                converted.uncertain_mask, converted.uv, converted.cluster_ids,
                converted.surface_owner_kind, converted.surface_owner_id,
            )
        except Exception:
            model.restore_state(snapshot)
            raise

        # Stage 3: sidecar commit (adapter-owned).
        try:
            self._commit_sidecar(batch.proposal_batch_id, sidecar_entry)
        except Exception:
            model.restore_state(snapshot)
            raise

        # Stage 4: owner registry commit (model-owned). Only actually
        # mutates the registry now -- `_convert()` (stage 1) only ever
        # validated. If this raises, nothing has been written to the
        # registry yet, so there is nothing new to roll back there; sidecar
        # and model still unwind.
        try:
            self._commit_owner_registry(model, converted.owner_id_scalar, batch.metadata["source_chart_id"])
        except Exception:
            self._rollback_sidecar(batch.proposal_batch_id)
            model.restore_state(snapshot)
            raise

        # Stage 5: ledger commit (model-owned). If this raises, the owner
        # registry entry must be rolled back too -- but ONLY if THIS
        # transaction newly created it (`owner_binding_preexisted=False`).
        # A binding that already existed before this call (e.g. two batches
        # from the same occluded chart) belongs to whatever earlier,
        # already-committed transaction created it and must survive this
        # transaction's own failure.
        try:
            self._commit_ledger(model, batch.proposal_batch_id)
        except Exception:
            self._rollback_owner_registry(
                model, converted.owner_id_scalar, was_preexisting=converted.owner_binding_preexisted
            )
            self._rollback_sidecar(batch.proposal_batch_id)
            model.restore_state(snapshot)
            raise

        # Transaction complete. `receipt` was built in stage 1, before any
        # commit began, so returning it here cannot itself fail or introduce
        # a window where a successful commit fails to produce a receipt.
        return receipt

    def _build_receipt(
        self, batch: UncertainGaussianProposalBatch, preflight: UncertainAppendPreflight,
        converted: _ConvertedAppend, requested: int, before: int, after: int, init_digest: str,
    ) -> UncertainAppendReceipt:
        count = len(converted.sample_ids)
        return UncertainAppendReceipt(
            proposal_batch_id=batch.proposal_batch_id,
            source_chart_id=batch.metadata["source_chart_id"],
            requested_sample_count=requested,
            valid_sample_count=preflight.valid_sample_count,
            appended_sample_count=count,
            rejected_sample_count=requested - count,
            model_count_before=before,
            model_count_after=after,
            appended_index_range=(before, after),
            appended_sample_ids=converted.sample_ids,
            conversion_summary=preflight.required_conversions,
            append_state="appended",
            reasons=(),
            cluster_id=converted.cluster_id,
            cluster_id_projection_rule=CLUSTER_ID_PROJECTION_RULE,
            initialization_digest=init_digest,
            surface_owner_kind=SURFACE_OWNER_OCCLUDED_CHART,
            surface_owner_id=converted.owner_id_scalar,
        )
