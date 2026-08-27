from __future__ import annotations

"""Worklog 120 -- CANDIDATE A ONLY. Direct surface observation / surface-hit.

HYPOTHESIS A: direct renderer-observed surface events are sufficient to define
the observed side of the volumetric partition.

Per-view semantics implemented here and nowhere else:

    OBSERVED   x IS itself the directly observed surface event on this camera
               ray -- i.e. x coincides with the world point of the median
               surface event at the pixel x projects to.
    OCCLUDED   a qualified directly observed surface event exists at that pixel
               and lies in front of x on the same camera ray, and x is not that
               event.
    UNRESOLVED anything else: no qualified surface event at the pixel at all,
               or an event that is at/behind x while x is not the event
               (A has no way to say anything about free space in front of a
               surface -- see the note below).

The primitive: worklog 119's G2, the direct median-surfel local intersection
(`shared.reconstruct_direct_surfel_intersection_world_point`) -- the cleanest
existing surface-event provenance, RENDERER-NATIVE in the sense that `s_u`,
`s_v` and the representative id are values the canonical kernel itself computes
at its own T=0.5 crossing, and RENDERER-GROUNDED in the sense that turning them
into a world point uses the trained surfel's frame. Both labels are reported in
the worklog's Candidate Operational Contract.

"Qualified" means exactly `representative_id >= 0` -- the canonical kernel's own
"some contributor crossed T = 0.5 at this pixel" condition. It is NOT narrowed
to rho3d-dominated events: worklog 119 established rho2d low-pass events must be
kept and classified separately, not rejected, so branch provenance is carried
out as METADATA (`branch` in the returned record) and never as a second
decision rule.

DELIBERATELY NOT REPAIRED (directive section 5): A is not allowed to declare
every point in front of a surface hit OBSERVED, because that conclusion does not
follow from A's own semantics -- an observed surface event at depth d says the
surface was seen, not that the empty space in front of it was. A's poor
volumetric coverage is the hypothesis's own property and is reported, not fixed.

THE ASSOCIATION TOLERANCE PROBLEM, stated rather than tuned (directive section
5): "x IS the surface event" is a coincidence test between an arbitrary 3D
query and a measure-zero point. There is no renderer-native or existing
geometry-derived definition of "near enough to a surface event to BE that
event"; the surfel's own footprint extent is a lateral (in-plane) extent, not a
tolerance along the ray. This module therefore uses only a FLOAT32 ROUND-TRIP
EQUALITY RULE -- `||x - E|| <= 1e-6 * max(1, ||x||)`, a numerical-precision
constant, not a semantic radius -- and the audit reports the measured
`hit_distance` distribution so a reviewer can see directly that results do not
sit near that constant. No coverage-driven widening was performed at any point.
"""

from typing import Any

import numpy as np
import torch

from .shared import (
    STATE_NON_RELEVANT,
    STATE_OBSERVED,
    STATE_OCCLUDED,
    STATE_UNRESOLVED,
    ViewGeometry,
)

NAME = "A"
TITLE = "DIRECT SURFACE OBSERVATION / SURFACE-HIT"

# Float32 round-trip equality only. NOT a semantic association radius: a query
# generated as a surface event reproduces that event's coordinates to float32
# round-off, and nothing else in the scene is within 1e-6 relative of a
# specific event point by accident.
FLOAT32_IDENTITY_RELATIVE_EPSILON = 1e-6


def classify_view(
    geometry: ViewGeometry,
    event_world: torch.Tensor,
    event_depth: torch.Tensor,
    event_valid: torch.Tensor,
    query_positions: torch.Tensor,
) -> dict[str, Any]:
    """Classify every query against ONE view.

    `event_world` (H*W, 3), `event_depth` (H*W,), `event_valid` (H*W,) are the
    per-pixel G2 surface events of this view -- computed by the driver from the
    renderer outputs, never re-decided here.
    """

    count = int(query_positions.shape[0])
    device = query_positions.device
    states = torch.full((count,), STATE_NON_RELEVANT, dtype=torch.int8, device=device)
    hit_distance = torch.full((count,), float("nan"), dtype=torch.float32, device=device)
    event_depth_at_query = torch.full((count,), float("nan"), dtype=torch.float32, device=device)

    relevant = geometry.relevant
    if not bool(relevant.any()):
        return {"states": states, "hit_distance": hit_distance, "event_depth": event_depth_at_query}

    index = geometry.pixel_index.clamp(min=0)
    has_event = event_valid[index] & relevant
    event_xyz = event_world[index]
    event_z = event_depth[index]

    delta = (query_positions - event_xyz).norm(dim=1)
    scale = torch.clamp(query_positions.norm(dim=1), min=1.0)
    is_the_event = has_event & (delta <= FLOAT32_IDENTITY_RELATIVE_EPSILON * scale)
    # Strictly in front on the same ray: the event's own camera-space z is
    # smaller than the query's. Equality is not "in front" and never occludes.
    in_front = has_event & (~is_the_event) & (event_z < geometry.depth)

    states = torch.where(relevant, torch.full_like(states, STATE_UNRESOLVED), states)
    states = torch.where(in_front, torch.full_like(states, STATE_OCCLUDED), states)
    states = torch.where(is_the_event, torch.full_like(states, STATE_OBSERVED), states)

    hit_distance = torch.where(has_event, delta, hit_distance)
    event_depth_at_query = torch.where(has_event, event_z, event_depth_at_query)
    return {"states": states, "hit_distance": hit_distance, "event_depth": event_depth_at_query}


def classify_view_numpy(*args, **kwargs) -> np.ndarray:
    return classify_view(*args, **kwargs)["states"].detach().cpu().numpy()
