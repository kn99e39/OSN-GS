from __future__ import annotations

"""Worklog 120 -- CANDIDATE D ONLY. Renderer reachability.

HYPOTHESIS D: a location belongs to the observed domain if the canonical
renderer ray remains contribution-capable through the query depth; it belongs
to the occluded domain in a view if canonical traversal has already terminated
before reaching that depth.

Per-view semantics implemented here and nowhere else:

    OBSERVED    canonical traversal had NOT terminated before the query depth
    OCCLUDED    the canonical termination condition fired at a contributor
                strictly before the query depth was reached
    UNRESOLVED  the probe produced no verdict for this (query, view) pair -- a
                fail-closed state that must not occur for a relevant view, and
                is counted loudly if it ever does

The primitive: the canonical traversal's own termination event,
`if (test_T < 0.0001f) { done = true; ... }` in the vendored `forward.cu`, with
`test_T = T * (1 - alpha)`. RENDERER-NATIVE: the condition, the constant
0.0001, the ordering, the acceptance checks and the depth convention are all the
canonical kernel's, observed at the canonical site by the worklog 120
diagnostic sibling build (`osn_gs/render/vendor/diff_surfel_rasterization_qdepth`)
and re-exposed at arbitrary query depth. NO new occlusion threshold exists here.
This module never compares T against anything.

`T` itself IS carried out of the probe -- but strictly as metadata, so the
report can show what D's verdicts rest on (e.g. how much residual transmittance
a query D calls OBSERVED actually still has). Making T a decision would require
inventing exactly the threshold the directive forbids, and it is not done.

Directive section 8 asked whether existing diagnostic outputs could reconstruct
the query-depth prefix state without new CUDA. They cannot -- see
`osn_gs/render/torch_surfel_query_depth_diagnostics.py`'s module docstring for
the three concrete reasons (final-T only, worklog 110's 97.4% slot truncation,
and no per-contributor alpha/depth anywhere).
"""

from typing import Any

import torch

from .shared import (
    STATE_NON_RELEVANT,
    STATE_OBSERVED,
    STATE_OCCLUDED,
    STATE_UNRESOLVED,
    ViewGeometry,
)

NAME = "D"
TITLE = "RENDERER REACHABILITY"


def classify_view(
    geometry: ViewGeometry,
    terminated: torch.Tensor,
    transmittance: torch.Tensor,
    reached: torch.Tensor,
    prefix_count: torch.Tensor,
) -> dict[str, Any]:
    """Classify every query against ONE view.

    `terminated`/`reached`/`prefix_count` are (N,) int tensors and
    `transmittance` an (N,) float tensor, gathered by the driver from the
    diagnostic probe's per-slot outputs. A -1 in `terminated` means the probe
    never wrote that slot, which is the fail-closed UNRESOLVED case.
    """

    count = int(geometry.depth.shape[0])
    device = geometry.depth.device
    states = torch.full((count,), STATE_NON_RELEVANT, dtype=torch.int8, device=device)
    relevant = geometry.relevant
    if not bool(relevant.any()):
        return {"states": states, "transmittance": transmittance, "reached": reached, "prefix_count": prefix_count}

    written = relevant & (terminated >= 0)
    states = torch.where(relevant, torch.full_like(states, STATE_UNRESOLVED), states)
    occluded = written & (terminated == 1)
    observed = written & (terminated == 0)
    states = torch.where(occluded, torch.full_like(states, STATE_OCCLUDED), states)
    states = torch.where(observed, torch.full_like(states, STATE_OBSERVED), states)
    return {"states": states, "transmittance": transmittance, "reached": reached, "prefix_count": prefix_count}
