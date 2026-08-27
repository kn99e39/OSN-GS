from __future__ import annotations

"""Worklog 127 -- POST-CONSTRUCTION ATTRIBUTION ONLY (directive sections 8, 15, 17).

Everything in this module runs AFTER the mesh exists and may only read
historical quantities. Nothing here can influence voxel creation, fusion, field
authority, the zero crossing or mesh extraction -- those live in `field.py` /
`extraction.py`, which import nothing from the historical families at all.

This is the ONLY module in the package permitted to touch the frozen
observed/occluded candidates and the worklog 107/109 topology, and it does so
read-only.
"""

from typing import Any

import numpy as np
import torch

# Diagnostic mesh-occlusion states (directive section 17). No depth epsilon.
MESH_OCCLUDED = 1
MESH_UNOCCLUDED = 0
MESH_NOT_RELEVANT = -1
MESH_STATE_NAMES = {
    MESH_OCCLUDED: "MESH_OCCLUDED", MESH_UNOCCLUDED: "MESH_UNOCCLUDED",
    MESH_NOT_RELEVANT: "MESH_NOT_RELEVANT",
}


def mesh_occlusion_for_view(
    query_depth: torch.Tensor, pixel_index: torch.Tensor, relevant: torch.Tensor,
    mesh_depth_flat: torch.Tensor,
) -> torch.Tensor:
    """MESH_OCCLUDED iff a mesh first-hit occurs STRICTLY before the query, on
    the same pixel-centre ray candidate B compares against. No epsilon."""

    states = torch.full(query_depth.shape, MESH_NOT_RELEVANT, dtype=torch.int8, device=query_depth.device)
    index = pixel_index.clamp(min=0)
    hit = mesh_depth_flat[index]
    finite = relevant & torch.isfinite(hit)
    occluded = finite & (hit < query_depth)
    unoccluded = finite & ~occluded
    states = torch.where(unoccluded, torch.full_like(states, MESH_UNOCCLUDED), states)
    states = torch.where(occluded, torch.full_like(states, MESH_OCCLUDED), states)
    # A relevant query on a ray with NO mesh hit is unobstructed BY THE MESH.
    no_hit = relevant & ~torch.isfinite(hit)
    return torch.where(no_hit, torch.full_like(states, MESH_UNOCCLUDED), states)


def aggregate_mesh_states(per_view: np.ndarray) -> np.ndarray:
    """The SAME shape of aggregation candidate B's global states use: any view
    that sees the query unobstructed wins. Reproduced here (not imported) so the
    comparison is explicit; `tests` asserts it matches `aggregate_global`'s rule
    on the OBSERVED/OCCLUDED codes."""

    states = np.asarray(per_view)
    any_unoccluded = (states == MESH_UNOCCLUDED).any(axis=1)
    relevant = states != MESH_NOT_RELEVANT
    has_relevant = relevant.any(axis=1)
    all_occluded = has_relevant & ((states == MESH_OCCLUDED) | (~relevant)).all(axis=1)
    result = np.full(states.shape[0], MESH_NOT_RELEVANT, dtype=np.int8)
    result[all_occluded] = MESH_OCCLUDED
    result[any_unoccluded] = MESH_UNOCCLUDED
    return result


def confusion(b_states: np.ndarray, mesh_states: np.ndarray) -> dict[str, int]:
    """The four cells directive section 17 asks for, plus the residue."""

    from observed_occluded.shared import STATE_OBSERVED, STATE_OCCLUDED

    b = np.asarray(b_states).reshape(-1)
    m = np.asarray(mesh_states).reshape(-1)
    return {
        "B_OCCLUDED_and_mesh_OCCLUDED": int(((b == STATE_OCCLUDED) & (m == MESH_OCCLUDED)).sum()),
        "B_OCCLUDED_and_mesh_unobstructed": int(((b == STATE_OCCLUDED) & (m == MESH_UNOCCLUDED)).sum()),
        "B_OBSERVED_and_mesh_OCCLUDED": int(((b == STATE_OBSERVED) & (m == MESH_OCCLUDED)).sum()),
        "B_OBSERVED_and_mesh_unobstructed": int(((b == STATE_OBSERVED) & (m == MESH_UNOCCLUDED)).sum()),
        "B_unresolved_or_irrelevant": int(((b != STATE_OBSERVED) & (b != STATE_OCCLUDED)).sum()),
        "mesh_not_relevant": int((m == MESH_NOT_RELEVANT).sum()),
    }


def historical_component_of_points(
    points: torch.Tensor, model_positions: torch.Tensor, component: torch.Tensor, *, chunk: int = 200_000,
) -> torch.Tensor:
    """Nearest trained surfel's frozen worklog 107/109 component id, for
    READ-ONLY attribution of already-built geometry."""

    out = torch.empty((points.shape[0],), dtype=torch.int64, device=points.device)
    for start in range(0, int(points.shape[0]), chunk):
        block = points[start : start + chunk]
        nearest = torch.cdist(block, model_positions).argmin(dim=1)
        out[start : start + chunk] = component[nearest]
    return out
