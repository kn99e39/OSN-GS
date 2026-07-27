from __future__ import annotations

"""Review-only multi-loop evidence for the common Boundary-first role contract.

A single outer boundary must never be duplicated into one annulus per hole;
that would create overlapping charts.  This module reports the exact missing
materialization evidence while retaining every observed boundary role.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MultiLoopCorrespondence:
    state: str
    reason: str | None
    outer_label: int | None
    hole_labels: tuple[int, ...]
    boundary_roles: tuple[str, ...]
    provenance: dict[str, Any]


def assess_multi_loop_correspondence(boundary_result: Any) -> MultiLoopCorrespondence:
    outer = list(getattr(boundary_result, "outer_loops", ()))
    holes = list(getattr(boundary_result, "hole_loops", ()))
    if len(outer) != 1 or len(holes) < 2:
        return MultiLoopCorrespondence("not_applicable", "requires_one_outer_and_multiple_holes", None, (), (), {})
    outer_label = int(outer[0].label)
    nested = tuple(sorted(int(hole.label) for hole in holes if getattr(hole, "nested_in_outer_label", None) == outer_label))
    roles = ("outer_boundary",) + tuple("interior_boundary" for _ in nested)
    loop_evidence = {
        "outer": {
            "label": outer_label,
            "ordered": bool(getattr(outer[0], "ordered_boundary_world_points", ())),
        },
        "interiors": [
            {
                "label": int(hole.label),
                "nested_in_outer_label": getattr(hole, "nested_in_outer_label", None),
                "ordered": bool(getattr(hole, "ordered_boundary_world_points", ())),
            }
            for hole in sorted(holes, key=lambda value: int(value.label))
        ],
    }
    if len(nested) != len(holes):
        return MultiLoopCorrespondence(
            "unsupported",
            "hole_nesting_evidence_incomplete",
            outer_label,
            nested,
            roles,
            {"outer_label": outer_label, "boundary_roles": list(roles), "loop_evidence": loop_evidence},
        )
    ordered = all(bool(getattr(hole, "ordered_boundary_world_points", ())) for hole in holes)
    ordered = ordered and bool(getattr(outer[0], "ordered_boundary_world_points", ()))
    if not ordered:
        return MultiLoopCorrespondence(
            "unsupported",
            "ordered_boundary_required",
            outer_label,
            nested,
            roles,
            {"outer_label": outer_label, "hole_labels": nested, "boundary_roles": list(roles), "loop_evidence": loop_evidence},
        )
    return MultiLoopCorrespondence(
        "review_required",
        "planar_domain_decomposition_required",
        outer_label,
        nested,
        roles,
        {
            "outer_label": outer_label,
            "hole_labels": nested,
            "boundary_roles": list(roles),
            "loop_evidence": loop_evidence,
            "overlap_prevention": "outer_boundary_must_not_be_duplicated_per_hole",
            "missing_materialization_evidence": "non_overlapping_planar_domain_partition",
        },
    )