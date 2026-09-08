"""Diagnostic-only audit helpers for Worklog 179.

This module deliberately does *not* implement a production ``F(x)``.  The
existing renderer diagnostics expose several different event subjects:

* ``forward_accepted[g]`` is a primitive/camera event;
* ``representative_id`` and ``contrib_ids`` are pixel-event identities; and
* ``query_reached``/``query_terminated`` are camera-ray point-query events.

Those subjects are not interchangeable.  The helpers below make the raw
event partition testable without relabelling any event as visible-surface
support or surface existence.  A future F contract may consume this module
only after the semantic gate passes.

No surface, TSDF, NURBS, topology, or trainer module is imported here.  The
module is intentionally standard-library-only so the contract tests remain
cheap and deterministic on machines without CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

STATE_OBSERVED = "OBSERVED"
STATE_OCCLUDED = "OCCLUDED"
STATE_UNRESOLVED = "UNRESOLVED"
STATES = frozenset({STATE_OBSERVED, STATE_OCCLUDED, STATE_UNRESOLVED})

RAW_REACHED = "REACHED_ACCEPTED_EVENT"
RAW_TERMINATED = "TERMINATED_BEFORE_QUERY"
RAW_UNRESOLVED = "UNRESOLVED_RENDERER_EVENT"
RAW_EVENTS = frozenset({RAW_REACHED, RAW_TERMINATED, RAW_UNRESOLVED})

STOP_SUBJECT_TYPE_MISMATCH = "SUBJECT_TYPE_MISMATCH"
STOP_NO_THRESHOLD_FREE_CONTRACT = "NO_THRESHOLD_FREE_CONTRACT"
STOP_OCCLUDED_NOT_POSITIVELY_IDENTIFIABLE = "OCCLUDED_NOT_POSITIVELY_IDENTIFIABLE"


@dataclass(frozen=True)
class EvidenceSubjectAudit:
    """Typed description of one already-available renderer event."""

    name: str
    subject_type: str
    event_kind: str
    discrete: bool
    aggregation_scope: str
    semantic_type: str
    fields: tuple[str, ...]
    positive_observed_witness: bool
    positive_blocker_witness: bool
    supports_arbitrary_world_point: bool
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "subject_type": self.subject_type,
            "event_kind": self.event_kind,
            "discrete": self.discrete,
            "aggregation_scope": self.aggregation_scope,
            "semantic_type": self.semantic_type,
            "fields": list(self.fields),
            "positive_observed_witness": self.positive_observed_witness,
            "positive_blocker_witness": self.positive_blocker_witness,
            "supports_arbitrary_world_point": self.supports_arbitrary_world_point,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class RawQueryEvent:
    """One qdepth slot, kept in renderer-native vocabulary."""

    query_terminated: int
    query_reached: int
    query_resolution_depth: float = -1.0
    query_termination_alpha: float = -1.0
    query_T: float = -1.0
    query_prefix_count: int = -1
    query_late_front_count: int = -1
    pixel_inversion_count: int = -1

    def classify(self) -> str:
        return classify_raw_query_event(
            self.query_terminated,
            self.query_reached,
            query_resolution_depth=self.query_resolution_depth,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_terminated": int(self.query_terminated),
            "query_reached": int(self.query_reached),
            "query_resolution_depth": float(self.query_resolution_depth),
            "query_termination_alpha": float(self.query_termination_alpha),
            "query_T": float(self.query_T),
            "query_prefix_count": int(self.query_prefix_count),
            "query_late_front_count": int(self.query_late_front_count),
            "pixel_inversion_count": int(self.pixel_inversion_count),
            "raw_event": self.classify(),
        }


def existing_evidence_subject_audit() -> tuple[EvidenceSubjectAudit, ...]:
    """Return the fixed subject-type audit for the existing diagnostics.

    The booleans are semantic claims about what the field can witness, not
    assumptions about scene geometry.  In particular, a zero
    ``forward_accepted`` bit is not positive blocker evidence.
    """

    return (
        EvidenceSubjectAudit(
            name="forward_accepted",
            subject_type="primitive-camera pair",
            event_kind="accepted contributor in the canonical forward pass",
            discrete=True,
            aggregation_scope="per camera; primitive flag is OR over accepted pixels",
            semantic_type="Primitive Observation Evidence",
            fields=("forward_accepted",),
            positive_observed_witness=True,
            positive_blocker_witness=False,
            supports_arbitrary_world_point=False,
            notes=(
                "A zero bit means no accepted contribution was recorded in that "
                "forward pass; it does not identify a blocker or an occlusion cause."
            ),
        ),
        EvidenceSubjectAudit(
            name="representative_id/contrib_ids",
            subject_type="pixel and pixel-contributor event",
            event_kind="median T=0.5 contributor identity and accepted contributor slots",
            discrete=True,
            aggregation_scope="per pixel per camera",
            semantic_type="Primitive Observation Evidence at a pixel event",
            fields=("representative_id", "contrib_ids", "contrib_count", "contrib_post_median"),
            positive_observed_witness=False,
            positive_blocker_witness=False,
            supports_arbitrary_world_point=False,
            notes=(
                "Identity is a pixel event; it is neither a primitive blocked-state "
                "label nor a point-query occlusion witness."
            ),
        ),
        EvidenceSubjectAudit(
            name="qdepth query event",
            subject_type="camera-ray point query",
            event_kind="accepted traversal reached the query or canonical T termination preceded it",
            discrete=True,
            aggregation_scope="per query slot per camera",
            semantic_type="Point-Query Occlusion Evidence candidate",
            fields=(
                "query_terminated",
                "query_reached",
                "query_resolution_depth",
                "query_termination_alpha",
                "query_T",
                "query_prefix_count",
                "query_late_front_count",
                "pixel_inversion_count",
            ),
            positive_observed_witness=False,
            positive_blocker_witness=True,
            supports_arbitrary_world_point=True,
            notes=(
                "Termination is a positive renderer-traversal blocker event, but it "
                "does not expose a physical blocker identity and reached means an "
                "accepted event at/behind query depth, not a surface at the query."
            ),
        ),
        EvidenceSubjectAudit(
            name="median depth",
            subject_type="pixel",
            event_kind="composited inverse-depth or direct fallback depth",
            discrete=False,
            aggregation_scope="per pixel per camera",
            semantic_type="Scalar depth/transmittance proxy",
            fields=("depth_median", "out_others", "valid_depth_mask"),
            positive_observed_witness=False,
            positive_blocker_witness=False,
            supports_arbitrary_world_point=False,
            notes=(
                "CUDA depth is inverted expected reciprocal depth; the historical "
                "world-sample classifier additionally uses depth_epsilon."
            ),
        ),
    )


def _binary_flag(value: int | bool, name: str) -> int:
    integer = int(value)
    if integer not in (0, 1):
        raise ValueError(f"{name} must be exactly 0 or 1, got {value!r}")
    return integer


def classify_raw_query_event(
    query_terminated: int | bool,
    query_reached: int | bool,
    *,
    query_resolution_depth: float = -1.0,
) -> str:
    """Partition qdepth output without a magnitude threshold.

    This is intentionally a *raw renderer-event* partition, not the paper's
    OBSERVED/OCCLUDED/UNRESOLVED F. ``query_terminated`` and ``query_reached``
    are exact integer flags emitted by the diagnostic kernel.  Both zero is
    the kernel's unresolved/exhausted fill convention; the resolution depth is
    checked only as provenance consistency, never as a classification cutoff.
    """

    terminated = _binary_flag(query_terminated, "query_terminated")
    reached = _binary_flag(query_reached, "query_reached")
    if terminated and reached:
        raise ValueError("query_terminated and query_reached cannot both be 1")
    if terminated:
        return RAW_TERMINATED
    if reached:
        return RAW_REACHED
    if query_resolution_depth >= 0.0:
        raise ValueError("unresolved qdepth event must retain resolution_depth=-1")
    return RAW_UNRESOLVED


def aggregate_raw_query_events(events: Iterable[str]) -> str:
    """Aggregate raw per-camera events with semantic witness rules.

    A reached witness dominates termination.  Termination is global only if
    every relevant camera supplies it.  Any unresolved relevant camera keeps
    the result unresolved.  This is not a majority or confidence vote.
    """

    event_list = list(events)
    if not event_list:
        return RAW_UNRESOLVED
    unknown = set(event_list) - RAW_EVENTS
    if unknown:
        raise ValueError(f"unknown raw event(s): {sorted(unknown)}")
    if RAW_REACHED in event_list:
        return RAW_REACHED
    if all(event == RAW_TERMINATED for event in event_list):
        return RAW_TERMINATED
    return RAW_UNRESOLVED


def primitive_observation_outcome(forward_accepted: int | bool) -> str:
    """Return the only safe primitive-side outcome currently available.

    ``OBSERVED`` is a positive primitive contribution witness.  A false bit is
    deliberately returned as ``UNRESOLVED`` rather than ``OCCLUDED``.  The
    function is not an F implementation and cannot produce a positive
    primitive OCCLUDED result.
    """

    return STATE_OBSERVED if _binary_flag(forward_accepted, "forward_accepted") else STATE_UNRESOLVED


def evaluate_contract_gate() -> dict[str, Any]:
    """Return the Worklog 179 semantic gate without inventing a classifier."""

    audits = existing_evidence_subject_audit()
    primitive = next(item for item in audits if item.name == "forward_accepted")
    point_query = next(item for item in audits if item.name == "qdepth query event")
    return {
        "verdict": "NO_RENDERER_NATIVE_THREE_STATE_CONTRACT",
        "stop_conditions": [STOP_SUBJECT_TYPE_MISMATCH, STOP_OCCLUDED_NOT_POSITIVELY_IDENTIFIABLE],
        "f_implemented": False,
        "primitive_observation": {
            "subject_type": primitive.subject_type,
            "observed_witness": primitive.positive_observed_witness,
            "occluded_witness": primitive.positive_blocker_witness,
            "unresolved_outcome": True,
            "status": "OBSERVED_PLUS_UNRESOLVED_ONLY",
        },
        "point_query": {
            "subject_type": point_query.subject_type,
            "raw_event_partition": sorted(RAW_EVENTS),
            "positive_termination_event": point_query.positive_blocker_witness,
            "direct_point_observation_witness": point_query.positive_observed_witness,
            "status": "RAW_RENDERER_PARTITION_NOT_EPISTEMIC_F",
        },
        "threshold_free": {
            "raw_query_flags_require_no_tuned_magnitude_threshold": True,
            "canonical_termination_constant_is_renderer_semantics_not_new_F_threshold": True,
            "historical_world_sample_classifier_uses_depth_epsilon": True,
            "f_primary_contract_promoted": False,
        },
        "reason": (
            "The available positive primitive event cannot prove primitive occlusion. "
            "The point-query qdepth event is a discrete renderer reachability/termination "
            "partition, but reached is not direct observation at x and termination does "
            "not identify an independent physical blocker. Existing analytic controls "
            "also contain strict median/zero-set counterexamples."
        ),
    }


def validate_state_partition(states: Iterable[str]) -> bool:
    """Small assertion helper for focused property tests."""

    values = list(states)
    return bool(values) and all(value in STATES for value in values)


def validate_audit_schema(audit: Mapping[str, Any]) -> None:
    """Fail fast if a serialized audit drops a required semantic field."""

    required = {
        "subject_type",
        "event_kind",
        "discrete",
        "aggregation_scope",
        "semantic_type",
        "fields",
        "positive_observed_witness",
        "positive_blocker_witness",
        "supports_arbitrary_world_point",
    }
    missing = required - set(audit)
    if missing:
        raise ValueError(f"subject audit missing fields: {sorted(missing)}")

