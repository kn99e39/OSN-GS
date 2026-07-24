from __future__ import annotations

"""Phase E (evidence) — candidate-region ObservationEvidence validation.

docs/Urgent_Work/OSN_GS_Phase_E_Bounded_Candidate_Design.md sections 8-11,
impl plan section 7. Applied AFTER `build_geometric_region_candidates` has
produced purely geometric candidates -- this is the only Phase E module that
imports `torch_observation_evidence`, keeping the geometric builder evidence-free.

Key contracts:
- Bridge cross-sections are sampled per correspondence pair as
  ``[support endpoint A, interior samples..., support endpoint B]``. Support
  endpoints are recorded separately (``support_endpoint_evidence``) and are
  NEVER part of the known-free-space hard-reject computation (design section 8).
- Section hard contradiction (interior only): at least one evaluable interior
  sample, all evaluable interior samples ``known_free_space``, and no
  behind/on_surface/unobserved/conflicting interior evidence. A section with no
  evaluable interior sample is ``insufficient_evidence``, not a contradiction.
- Candidate hard reject: at least one evaluable nondegenerate section AND all
  such sections are hard contradictions -> ``state=rejected``. Partial free
  space is preserved, never auto-rejected (design section 9).
- ``no_observed_support`` and ``conflicting_evidence`` are metadata only, never
  used to promote or reject a candidate (design section 10).
"""

from typing import Any, Callable, Sequence

from osn_gs.surface.torch_observation_evidence import (
    STATUS_CONFLICTING_EVIDENCE,
    STATUS_KNOWN_FREE_SPACE,
    STATUS_OCCLUDED_CANDIDATE,
    STATUS_ON_OBSERVED_SURFACE,
    STATUS_OUTSIDE_VALID_VIEW,
    STATUS_UNOBSERVED,
    ObservationEvidence,
    classify_world_samples,
)
from osn_gs.surface.torch_occluded_region_candidate import (
    STATE_REJECTED,
    OccludedRegionCandidate,
)
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9


def _cross_section_points(a_point: Any, b_point: Any, interior_samples: int) -> tuple[Any, Any]:
    """Return ``(interior (I,3), endpoints (2,3))`` for one A<->B cross-section.

    Interior samples are strictly between the endpoints (exclusive), so the
    boundary itself never enters the interior free-space computation.
    """

    torch = require_torch()
    n = max(1, int(interior_samples))
    fracs = torch.linspace(0.0, 1.0, n + 2, dtype=a_point.dtype, device=a_point.device)[1:-1]
    interior = a_point[None, :] + fracs[:, None] * (b_point - a_point)[None, :]
    endpoints = torch.stack([a_point, b_point])
    return interior, endpoints


def _empty_summary() -> dict[str, Any]:
    return {
        "known_free_sample_count": 0,
        "behind_support_count": 0,
        "on_surface_count": 0,
        "unobserved_count": 0,
        "conflicting_count": 0,
        "outside_count": 0,
        "per_camera": {},
    }


def _accumulate(summary: dict[str, Any], status: str, evidence: Any) -> None:
    if status == STATUS_KNOWN_FREE_SPACE:
        summary["known_free_sample_count"] += 1
    elif status == STATUS_OCCLUDED_CANDIDATE:
        summary["behind_support_count"] += 1
    elif status == STATUS_ON_OBSERVED_SURFACE:
        summary["on_surface_count"] += 1
    elif status == STATUS_UNOBSERVED:
        summary["unobserved_count"] += 1
    elif status == STATUS_CONFLICTING_EVIDENCE:
        summary["conflicting_count"] += 1
    elif status == STATUS_OUTSIDE_VALID_VIEW:
        summary["outside_count"] += 1
    per_camera = summary["per_camera"]
    for cam in (
        list(evidence.free_space_confirmed_by)
        + list(evidence.on_surface_in)
        + list(evidence.behind_surface_in)
        + list(evidence.unobserved_in)
        + list(evidence.outside_in)
    ):
        per_camera[str(cam)] = per_camera.get(str(cam), 0) + 1


def validate_candidate_observation_evidence(
    candidates: Sequence[OccludedRegionCandidate],
    observation_evidence: ObservationEvidence | None,
    *,
    empty_voxel_query: Callable[[Any, Any], Any] | None = None,
    interior_samples: int = 3,
) -> list[OccludedRegionCandidate]:
    """Populate each candidate's evidence fields and apply the interior-only
    known-free-space hard reject in place; returns the same list.

    ``observation_evidence=None`` leaves evidence summaries empty and never
    changes geometric state. ``empty_voxel_query`` (if given) records
    ``no_observed_support`` overlap as metadata only.
    """

    torch = require_torch()
    for candidate in candidates:
        # Empty-voxel metadata (never promotes/rejects).
        if empty_voxel_query is not None:
            result = empty_voxel_query(candidate.aabb_min, candidate.aabb_max)
            candidate.empty_voxel_support = {
                "support": getattr(result, "support", None),
                "overlapping_empty_leaf_ids": list(getattr(result, "overlapping_empty_leaf_ids", []) or []),
                "used_as_evidence": False,
            }
        else:
            candidate.empty_voxel_support = {"support": None, "overlapping_empty_leaf_ids": [], "used_as_evidence": False}

        if observation_evidence is None:
            for name in (
                "free_space_contradiction",
                "behind_observation_support",
                "on_surface_evidence",
                "unobserved_evidence",
                "conflicting_evidence",
            ):
                setattr(candidate, name, {"evaluated": False})
            candidate.provenance["evidence_applied"] = False
            continue

        world_a = candidate.support_chain_a.world
        world_b = candidate.support_chain_b.world
        n_sections = int(world_a.shape[0])

        interior_summary = _empty_summary()
        endpoint_summary = _empty_summary()

        section_flags: list[str] = []  # "hard_contradiction" | "insufficient" | "mixed"
        section_records: list[dict[str, Any]] = []
        nondegenerate = [True] * n_sections  # zero-area handled at bridge-cell level already

        for k in range(n_sections):
            interior_pts, endpoint_pts = _cross_section_points(world_a[k], world_b[k], interior_samples)
            interior_ev = classify_world_samples(observation_evidence, interior_pts)
            endpoint_ev = classify_world_samples(observation_evidence, endpoint_pts)

            for ev in endpoint_ev:
                _accumulate(endpoint_summary, ev.status, ev)

            evaluable = 0
            all_free = True
            has_other = False
            for ev in interior_ev:
                _accumulate(interior_summary, ev.status, ev)
                if ev.status == STATUS_OUTSIDE_VALID_VIEW:
                    continue
                evaluable += 1
                if ev.status == STATUS_KNOWN_FREE_SPACE:
                    continue
                all_free = False
                has_other = True

            if evaluable == 0:
                flag = "insufficient"
            elif all_free and not has_other:
                flag = "hard_contradiction"
            else:
                flag = "mixed"
            section_flags.append(flag)
            section_records.append({"section": k, "flag": flag, "evaluable_interior": evaluable})

        evaluable_sections = [
            i for i in range(n_sections) if section_flags[i] != "insufficient" and nondegenerate[i]
        ]
        contradiction_sections = [i for i in evaluable_sections if section_flags[i] == "hard_contradiction"]
        candidate_hard_reject = (
            len(evaluable_sections) >= 1 and len(contradiction_sections) == len(evaluable_sections)
        )

        candidate.free_space_contradiction = {
            "evaluated": True,
            "known_free_sample_count": interior_summary["known_free_sample_count"],
            "known_free_section_count": len(contradiction_sections),
            "insufficient_section_count": section_flags.count("insufficient"),
            "evaluable_section_count": len(evaluable_sections),
            "candidate_hard_contradiction": bool(candidate_hard_reject),
            "sections": section_records,
        }
        candidate.behind_observation_support = {
            "behind_support_count": interior_summary["behind_support_count"],
        }
        candidate.on_surface_evidence = {
            "interior_on_surface_count": interior_summary["on_surface_count"],
            "endpoint_on_surface_count": endpoint_summary["on_surface_count"],
        }
        candidate.unobserved_evidence = {
            "interior_unobserved_count": interior_summary["unobserved_count"],
        }
        candidate.conflicting_evidence = {
            "interior_conflicting_count": interior_summary["conflicting_count"],
            "used_as_evidence": False,
        }
        candidate.provenance["evidence_applied"] = True
        candidate.provenance["interior_evidence_summary"] = interior_summary
        candidate.provenance["endpoint_evidence_summary"] = endpoint_summary

        if candidate_hard_reject and candidate.state != STATE_REJECTED:
            candidate.state = STATE_REJECTED
            candidate.reason = (
                (candidate.reason + ";" if candidate.reason and candidate.reason != "ok" else "")
                + "full_bridge_interior_known_free_space"
            )

    return list(candidates)
