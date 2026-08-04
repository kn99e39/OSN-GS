"""Scale contract for representative-level termination neighborhoods."""
from __future__ import annotations
from typing import Any


def resolve_termination_neighborhood_scale(
    *, candidate_scale: Any | None, tangent_major_scale: Any,
) -> Any:
    """Return the scale used by representative termination topology only.

    ``candidate_scale`` is the G1 RepresentativeGraphScale: it defines the
    local representative support radius used by graph formation and angular
    termination observation.  It is deliberately distinct from a Gaussian's
    ``equivalent_tangent_scale``, which describes the physical primitive
    footprint and must not be repurposed as representative spacing.

    Legacy callers without an explicit ``candidate_scale`` use the same
    tangent-major default used by the affinity graph.  No synthetic or merged
    scale is created here.
    """
    return tangent_major_scale if candidate_scale is None else candidate_scale
