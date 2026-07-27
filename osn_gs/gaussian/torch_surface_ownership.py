from __future__ import annotations

"""Canonical surface-ownership contract for `TorchGaussianModel`.

docs/worklogs (Occluded Chart Ownership Foundation). Fixes an ownership
ambiguity the append-adapter Gate hardening pass (worklog 88) explicitly
flagged as a deferred risk: `cluster_ids` alone cannot express "this Gaussian
was appended because of an occluded NURBS chart that bridges two visible
patches" without arbitrarily picking one of the two patches as an owner.

Canonical ownership (fixed by this module, not re-derived elsewhere):

    Observed/certain Gaussian  -> visible NURBS patch ownership
    Occluded/uncertain Gaussian -> occluded NURBS chart ownership
    Visible source patch/domain/boundary/candidate -> provenance only

`source_patch_ids` (preserved in the append adapter's provenance sidecar) is
the generation basis and provenance for an uncertain Gaussian -- never its
behavioral owner. The append adapter's `cluster_id = min(source_patch_ids)`
projection is a deterministic COMPATIBILITY value only; it must never be
promoted to primary visible-patch ownership, training membership, loss
membership, maintenance membership, or support-mask membership. This module's
`surface_owner_kind`/`surface_owner_id` pair is the one authoritative source
for those behavioral questions.

Design choice (Option A "reserved integer namespace" vs Option B "explicit
owner_kind + owner_id tensors" -- see the Gate worklog for the full
comparison): Option B was adopted, because collapsing two semantically
different ID spaces (a real visible `patch_id` vs. a synthetic occluded-chart
identity) into the single existing `cluster_ids` tensor would force every
read site to re-interpret a bare integer's namespace correctly forever,
which is the kind of implicit contract that erodes over time. A reserved
namespace offset is still used here -- but only as defense-in-depth on TOP of
the explicit `surface_owner_kind` tag (see `project_occluded_chart_owner_id`),
so an accidental read of `surface_owner_id` without checking
`surface_owner_kind` still can't collide with a real, small visible patch id.
"""

import hashlib
from typing import Any

from osn_gs.utils.torch_ops import require_torch

SURFACE_OWNER_UNASSIGNED = 0
SURFACE_OWNER_VISIBLE_PATCH = 1
SURFACE_OWNER_OCCLUDED_CHART = 2

SURFACE_OWNER_KIND_NAMES: dict[int, str] = {
    SURFACE_OWNER_UNASSIGNED: "unassigned",
    SURFACE_OWNER_VISIBLE_PATCH: "visible_patch",
    SURFACE_OWNER_OCCLUDED_CHART: "occluded_chart",
}

# Canonical sentinel `surface_owner_id` for SURFACE_OWNER_UNASSIGNED rows.
# Chosen to equal `cluster_ids`'s own long-standing "no patch assigned"
# sentinel (torch_pipeline.py's `_initialize_stage1`: "Gaussians in
# inactive/skipped leaves stay unassigned (cluster_id -1)") so the migration
# rule below is a direct, unsurprising carry-over rather than a new value.
UNASSIGNED_OWNER_ID = -1

# Bumped only if the owner-id projection formula below changes incompatibly;
# folded into the digest so an old and a new projection can never collide.
OWNERSHIP_SCHEMA_VERSION = 1

# Defense-in-depth namespace floor for synthetic occluded-chart owner ids.
# Real visible `patch_id` values are small, dense, zero-based indices (see
# `TorchPipelineState.surface_patches`); no production scene comes remotely
# close to 10^12 patches, so an owner_id at or above this floor cannot be
# mistaken for a real patch_id even if a caller forgets to check
# `surface_owner_kind` first.
OCCLUDED_CHART_NAMESPACE_BASE = 1_000_000_000_000
_DIGEST_HEX_CHARS = 15  # 16**15 ~= 1.15e18; base + this still fits int64 (~9.2e18) comfortably.


def project_occluded_chart_owner_id(
    source_chart_id: str, *, schema_version: int = OWNERSHIP_SCHEMA_VERSION
) -> int:
    """Deterministic, namespace-safe synthetic owner id for one occluded chart.

    Properties (see the Gate worklog's identity-contract test list):
    - same ``source_chart_id`` (+ same ``schema_version``) -> same id, always.
    - different ``source_chart_id`` -> different id (SHA-256, not a weak hash).
    - independent of Python process hash-seed: uses ``hashlib.sha256``, never
      the built-in salted ``hash()``.
    - independent of input ordering: a single string input, no list/order to
      vary.
    - never collides with a real visible `patch_id`: always
      ``>= OCCLUDED_CHART_NAMESPACE_BASE``, far above any plausible patch count.
    - does not consume floating-point control-point/UV bytes -- only the
      already-stable ``source_chart_id`` string (itself already a stable,
      deterministic identity per `torch_occluded_chart.py`'s `_chart_id`).
    """

    if not isinstance(source_chart_id, str) or not source_chart_id:
        raise ValueError("source_chart_id must be a non-empty string")
    digest = hashlib.sha256(f"{int(schema_version)}|{source_chart_id}".encode("utf-8")).hexdigest()
    compact = int(digest[:_DIGEST_HEX_CHARS], 16)
    return OCCLUDED_CHART_NAMESPACE_BASE + compact


def owner_kind_name(owner_kind: int) -> str:
    """Diagnostic-only human-readable name for a `surface_owner_kind` value."""

    return SURFACE_OWNER_KIND_NAMES.get(int(owner_kind), f"unknown({owner_kind})")


def is_visible_patch_owned(surface_owner_kind: Any) -> Any:
    """Boolean mask: ``surface_owner_kind == SURFACE_OWNER_VISIBLE_PATCH``.

    Small helper so every behavioral read site spells the same authoritative
    ownership check the same way, rather than re-deriving it from
    `is_uncertain` (a correlated but not authoritative proxy) at each site.
    """

    return surface_owner_kind == SURFACE_OWNER_VISIBLE_PATCH


def derive_default_ownership(cluster_ids: Any) -> tuple[Any, Any]:
    """Elementwise migration/default rule shared by `TorchGaussianModel.initialize()`
    and `replace_tensors()`'s optional fallback (old-checkpoint compatibility).

    Canonical rule (audited against `torch_pipeline.py._initialize_stage1`,
    which deliberately leaves Gaussians in inactive/skipped voxel leaves at
    ``cluster_id = -1`` -- a CANONICAL state, not a transient one, per that
    function's own docstring/comments):

        cluster_id >= 0  -> surface_owner_kind = VISIBLE_PATCH, surface_owner_id = cluster_id
        cluster_id <  0  -> surface_owner_kind = UNASSIGNED,     surface_owner_id = UNASSIGNED_OWNER_ID

    A row with ``surface_owner_kind == VISIBLE_PATCH`` and
    ``surface_owner_id == UNASSIGNED_OWNER_ID`` (i.e. -1) is never produced by
    this function and is treated as an invariant violation by
    `validate_surface_ownership_consistency`. Occluded-chart ownership is
    NEVER derived here -- it is only ever assigned explicitly by the append
    adapter, which always has a real source chart id to project from.
    """

    torch = require_torch()
    unassigned = cluster_ids < 0
    kind = torch.where(
        unassigned,
        torch.full_like(cluster_ids, SURFACE_OWNER_UNASSIGNED),
        torch.full_like(cluster_ids, SURFACE_OWNER_VISIBLE_PATCH),
    )
    owner_id = torch.where(unassigned, torch.full_like(cluster_ids, UNASSIGNED_OWNER_ID), cluster_ids)
    return kind, owner_id


class OccludedChartOwnerCollisionError(RuntimeError):
    """Raised when a synthetic owner id maps back to two different chart ids.

    A truncated SHA-256 digest is collision-RESISTANT, not collision-proof --
    this module never claims mathematical uniqueness. This registry is the
    actual enforcement: it remembers every ``owner_id -> source_chart_id``
    binding it has produced and raises loudly the moment a second, different
    chart id would reuse an already-bound owner id, instead of silently
    letting two charts share one identity.
    """


def reject_visible_patch_id_in_occluded_namespace(patch_id: int) -> None:
    """Raise if a real visible ``patch_id`` has strayed into the reserved
    occluded-chart namespace (``>= OCCLUDED_CHART_NAMESPACE_BASE``).

    No production code path currently generates patch ids anywhere near this
    range (they are small, dense, zero-based indices into
    ``TorchPipelineState.surface_patches``) -- this is a defensive assertion
    for diagnostics/tests, not something wired into patch-creation code
    (patch creation belongs to a separate topology/O-grid/planar work area).
    """

    if int(patch_id) >= OCCLUDED_CHART_NAMESPACE_BASE:
        raise ValueError(
            f"visible patch_id {patch_id} collides with the reserved occluded-chart "
            f"namespace (>= {OCCLUDED_CHART_NAMESPACE_BASE})"
        )


def validate_occluded_owner_binding_read_only(
    model: Any, source_chart_id: str, *, schema_version: int = OWNERSHIP_SCHEMA_VERSION
) -> tuple[int, bool]:
    """Pure, READ-ONLY collision check against the model-owned registry
    (``model.occluded_chart_owner_registry``). Does NOT mutate the registry --
    see `commit_occluded_owner_binding` for the actual write, which callers
    that need transactional rollback (the append adapter) run as its own,
    separately-failable commit stage rather than folding into this check.

    Returns ``(owner_id, already_registered)``:
    - ``owner_id``: the projected synthetic owner id.
    - ``already_registered``: True iff this exact ``(owner_id, source_chart_id)``
      binding is already present in the registry (i.e. committing it again
      would be a no-op with nothing new to roll back later).

    Raises `OccludedChartOwnerCollisionError` if ``owner_id`` is already bound
    to a DIFFERENT chart id. Ownership rationale for the registry living on
    the model rather than the adapter or a module-level global: worklog 88
    already found and fixed the exact same class of bug for the duplicate-
    append ledger -- an adapter-owned registry is silently bypassed the
    moment a caller swaps in a fresh adapter instance against the same model.
    A module-level global registry would be worse (shared across unrelated
    models, breaks test isolation, never garbage collected).
    """

    owner_id = project_occluded_chart_owner_id(source_chart_id, schema_version=schema_version)
    existing = model.occluded_chart_owner_registry.get(owner_id)
    if existing is not None and existing != source_chart_id:
        raise OccludedChartOwnerCollisionError(
            f"owner_id {owner_id} is already bound to chart {existing!r}; "
            f"cannot rebind it to {source_chart_id!r}"
        )
    already_registered = existing is not None
    return owner_id, already_registered


def commit_occluded_owner_binding(model: Any, owner_id: int, source_chart_id: str) -> None:
    """Actually register the ``owner_id -> source_chart_id`` binding.

    Idempotent for a binding that already holds (re-writing the same value).
    Callers must have already validated via `validate_occluded_owner_binding_read_only`
    -- this function does not re-check for collisions.
    """

    model.occluded_chart_owner_registry[owner_id] = source_chart_id


def rollback_occluded_owner_binding(model: Any, owner_id: int, *, was_preexisting: bool) -> None:
    """Undo `commit_occluded_owner_binding` for one transaction that later
    failed at a subsequent commit stage (e.g. the append batch ledger).

    Only removes the entry if THIS transaction newly created it
    (``was_preexisting=False``, from `validate_occluded_owner_binding_read_only`'s
    second return value). A binding that already existed before this
    transaction started must never be deleted by that transaction's rollback --
    it belongs to whatever earlier, already-committed transaction created it.
    """

    if not was_preexisting:
        model.occluded_chart_owner_registry.pop(owner_id, None)


def validate_surface_ownership_consistency(model: Any) -> tuple[str, ...]:
    """Pure, read-only ownership invariant check. Returns a tuple of
    human-readable violation strings; an empty tuple means the model's
    ownership state is fully consistent.

    NEVER raises for an invariant violation itself and is NOT invoked
    automatically on any hot path -- callers (tests, diagnostics, or a future
    preflight gate) decide what to do with a non-empty result. Checks:

    - `surface_owner_kind`/`surface_owner_id`/`cluster_ids` row counts match
      ``len(model)``.
    - both ownership tensors are ``torch.long`` and share one device.
    - every `surface_owner_kind` value is a known enum member.
    - every VISIBLE_PATCH-owned row's `surface_owner_id` is in the valid
      visible range ``0 <= id < OCCLUDED_CHART_NAMESPACE_BASE`` (a negative
      id -- e.g. the UNASSIGNED sentinel -- is REJECTED for this kind, per
      the canonical rule "negative/invalid patch membership is not visible
      ownership"), AND equals its `cluster_ids` entry (the canonical
      synchronization invariant).
    - every OCCLUDED_CHART-owned row's `surface_owner_id` sits at or above
      the occluded-chart namespace floor.
    - every UNASSIGNED row's `surface_owner_id` equals the canonical sentinel
      `UNASSIGNED_OWNER_ID` (-1) exactly -- no other value is valid for this kind.
    """

    torch = model.torch
    violations: list[str] = []
    n = len(model)
    kind = model.surface_owner_kind
    owner_id = model.surface_owner_id
    cluster_ids = model.cluster_ids

    for name, tensor in (("surface_owner_kind", kind), ("surface_owner_id", owner_id), ("cluster_ids", cluster_ids)):
        if int(tensor.shape[0]) != n:
            violations.append(f"{name} row count {int(tensor.shape[0])} != model count {n}")
    if violations:
        return tuple(violations)  # further checks assume matching shapes

    if kind.dtype != torch.long:
        violations.append(f"surface_owner_kind dtype {kind.dtype} != torch.long")
    if owner_id.dtype != torch.long:
        violations.append(f"surface_owner_id dtype {owner_id.dtype} != torch.long")

    devices = {str(kind.device), str(owner_id.device), str(cluster_ids.device)}
    if len(devices) > 1:
        violations.append(f"device mismatch across ownership tensors: {sorted(devices)}")

    if n == 0:
        return tuple(violations)

    valid_kinds = {SURFACE_OWNER_UNASSIGNED, SURFACE_OWNER_VISIBLE_PATCH, SURFACE_OWNER_OCCLUDED_CHART}
    unknown = sorted(set(int(x) for x in torch.unique(kind).tolist()) - valid_kinds)
    if unknown:
        violations.append(f"unknown surface_owner_kind values: {unknown}")
        return tuple(violations)  # can't reason about masks below with unknown enum values

    visible_mask = kind == SURFACE_OWNER_VISIBLE_PATCH
    if bool(visible_mask.any()):
        visible_owner_ids = owner_id[visible_mask]
        if bool((visible_owner_ids >= OCCLUDED_CHART_NAMESPACE_BASE).any()):
            violations.append("a VISIBLE_PATCH-owned row has surface_owner_id in the occluded-chart namespace")
        if bool((visible_owner_ids < 0).any()):
            violations.append(
                "a VISIBLE_PATCH-owned row has a negative surface_owner_id "
                "(negative/invalid patch membership is not visible ownership -- use UNASSIGNED)"
            )
        mismatched = visible_mask & (owner_id != cluster_ids)
        if bool(mismatched.any()):
            violations.append(
                f"{int(mismatched.sum())} VISIBLE_PATCH-owned row(s) have surface_owner_id != cluster_ids"
            )

    occluded_mask = kind == SURFACE_OWNER_OCCLUDED_CHART
    if bool(occluded_mask.any()):
        occluded_owner_ids = owner_id[occluded_mask]
        if bool((occluded_owner_ids < OCCLUDED_CHART_NAMESPACE_BASE).any()):
            violations.append("an OCCLUDED_CHART-owned row has surface_owner_id below the reserved namespace")

    unassigned_mask = kind == SURFACE_OWNER_UNASSIGNED
    if bool(unassigned_mask.any()):
        unassigned_owner_ids = owner_id[unassigned_mask]
        if bool((unassigned_owner_ids != UNASSIGNED_OWNER_ID).any()):
            violations.append(
                f"an UNASSIGNED row has surface_owner_id != canonical sentinel {UNASSIGNED_OWNER_ID}"
            )

    return tuple(violations)
