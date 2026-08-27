from __future__ import annotations

"""Worklog 120 -- CANDIDATE C ONLY. Geometric visibility.

HYPOTHESIS C: a hard geometric line-of-sight relation to the current scene
geometry is sufficient to define Observed/Occluded space independently of
cumulative alpha transmittance.

Per-view semantics implemented here and nowhere else:

    OBSERVED    the open segment (camera_centre, x) meets no scene geometry
    OCCLUDED    the open segment meets at least one surfel's geometric support
    UNRESOLVED  never returned for a relevant view -- C always has an answer

WHAT COUNTS AS "EXISTING SCENE GEOMETRY" (directive section 7 requires this be
audited before implementing, and forbids manufacturing a new support boundary):

The current 2DGS representation DOES already expose an exact, already-defined
finite geometric support, and it is not a k-sigma choice. The canonical forward
kernel accepts a primitive at a ray/plane intersection iff

    min(0.99, opacity * exp(-0.5 * rho)) >= 1/255,      rho = min(rho3d, rho2d)

Splitting that on its own two branches:
  * `rho3d = s_u^2 + s_v^2` is the TRUE ray-plane intersection distance in the
    surfel's own scaled tangent frame -- a 3D geometric quantity;
  * `rho2d` is a screen-space low-pass floor (`FilterInvSquare * |xy - pixf|^2`),
    a rasterization quantity with no 3D meaning, which worklog 119 measured as a
    systematically different KIND of observation (its G0-vs-G2 displacement is
    33% larger, its residual median 2.1x worse).

So the geometric support of surfel i is exactly the ellipse

    { c_i + a * t_u,i + b * t_v,i : (a/scale_u,i)^2 + (b/scale_v,i)^2 <= rho_max,i }
    rho_max,i = 2 * ln(255 * opacity_i)          (empty when opacity_i <= 1/255)

computed in `shared.canonical_geometric_support_rho_max`. No 3-sigma, no tuned
radius, no hand-selected blocker threshold, no opacity hardening, no opaque
surrogate, no NURBS, no mesh, no new surface reconstruction. The screen-space
`radii`/tile binning is deliberately NOT applied: that is a rasterization
optimization, not geometry, and C is defined as a geometric test.

THE ONE SEMANTIC CAVEAT, disclosed rather than repaired: treating any
intersected support as a HARD blocker regardless of the surfel's opacity
magnitude is hypothesis C's own premise ("independently of cumulative alpha
transmittance"), not an agent-introduced hardening -- no opacity value is
modified, read differently, or replaced anywhere. But it does mean a surfel with
opacity 0.006 blocks exactly as absolutely as one with opacity 0.99. This module
therefore also records, as pure metadata, the blocker count and the maximum
blocker opacity, so the report can show what C's OCCLUDED verdicts actually rest
on. That metadata is never used in the decision.

C is therefore reported as OPERATIONALIZABLE WITH A DISCLOSED SEMANTIC CAVEAT
rather than "NOT CLEANLY OPERATIONALIZABLE"; the audit's verdict section
weighs the caveat explicitly instead of hiding it behind the numbers.
"""

from typing import Any

import torch

from .shared import (
    STATE_NON_RELEVANT,
    STATE_OBSERVED,
    STATE_OCCLUDED,
    ViewGeometry,
)

NAME = "C"
TITLE = "GEOMETRIC VISIBILITY"

# Float32 parametric guard on the OPEN segment (camera, x). This is a
# numerical-precision rule, not a geometric tolerance: without it a query that
# IS a point of some surfel's support (every R1 anchor is) would occlude
# itself through float round-off at t = 1. `nearest_blocker_t` is reported so
# a reviewer can confirm decisions do not pile up against this guard.
SEGMENT_EPSILON = 1e-6


class GeometricSceneSupport:
    """Precomputed canonical geometric support of every surfel, view-independent."""

    def __init__(
        self,
        centers: torch.Tensor,
        normals: torch.Tensor,
        tangent_u: torch.Tensor,
        tangent_v: torch.Tensor,
        scale_u: torch.Tensor,
        scale_v: torch.Tensor,
        rho_max: torch.Tensor,
        opacity: torch.Tensor,
    ) -> None:
        self.centers = centers.to(torch.float32).contiguous()
        self.normals = normals.to(torch.float32).contiguous()
        self.tangent_u = tangent_u.to(torch.float32).contiguous()
        self.tangent_v = tangent_v.to(torch.float32).contiguous()
        self.inv_scale_u = (1.0 / torch.clamp(scale_u.reshape(-1), min=1e-20)).to(torch.float32)
        self.inv_scale_v = (1.0 / torch.clamp(scale_v.reshape(-1), min=1e-20)).to(torch.float32)
        self.rho_max = rho_max.reshape(-1).to(torch.float32)
        self.opacity = opacity.reshape(-1).to(torch.float32)
        # World-space bound on how far a support point can lie from the centre.
        radius = torch.sqrt(torch.clamp(self.rho_max, min=0.0))
        self.support_radius = (radius * torch.maximum(scale_u.reshape(-1), scale_v.reshape(-1))).to(torch.float32)
        self.center_dot_normal = (self.centers * self.normals).sum(dim=1)
        self.center_dot_tu = (self.centers * self.tangent_u).sum(dim=1)
        self.center_dot_tv = (self.centers * self.tangent_v).sum(dim=1)
        self.nonempty = self.rho_max > 0.0

    def __len__(self) -> int:
        return int(self.centers.shape[0])


def classify_view(
    geometry: ViewGeometry,
    query_positions: torch.Tensor,
    camera_center: torch.Tensor,
    world_view_transform: torch.Tensor,
    support: GeometricSceneSupport,
    chunk_bytes: int = 256 * 1024 * 1024,
) -> dict[str, Any]:
    """Exact ray/disc line-of-sight against every surfel's canonical support."""

    device = query_positions.device
    count = int(query_positions.shape[0])
    states = torch.full((count,), STATE_NON_RELEVANT, dtype=torch.int8, device=device)
    blocker_count = torch.zeros((count,), dtype=torch.int32, device=device)
    nearest_blocker_t = torch.full((count,), float("nan"), dtype=torch.float32, device=device)
    max_blocker_opacity = torch.full((count,), float("nan"), dtype=torch.float32, device=device)

    relevant_rows = torch.nonzero(geometry.relevant, as_tuple=False).reshape(-1)
    if relevant_rows.numel() == 0:
        return {
            "states": states, "blocker_count": blocker_count,
            "nearest_blocker_t": nearest_blocker_t, "max_blocker_opacity": max_blocker_opacity,
        }

    origin = camera_center.reshape(3).to(torch.float32)
    max_depth = float(geometry.depth[relevant_rows].max().item())

    # Exactly conservative prefilter: a blocker's support point must lie in
    # front of some query, and every support point is within `support_radius`
    # of its centre, so a surfel whose centre camera-space z minus that radius
    # already exceeds the deepest query in this view cannot block anything
    # here. Nothing else is filtered -- in particular the rasterizer's own
    # screen `radii` are NOT used (see module docstring).
    center_view_z = (
        support.centers @ world_view_transform[:3, 2].reshape(3) + world_view_transform[3, 2]
    )
    candidate = support.nonempty & ((center_view_z - support.support_radius) < max_depth)
    candidate_rows = torch.nonzero(candidate, as_tuple=False).reshape(-1)

    states[relevant_rows] = STATE_OBSERVED  # C only leaves OBSERVED when nothing blocks
    blocker_count[relevant_rows] = 0
    if candidate_rows.numel() == 0:
        return {
            "states": states, "blocker_count": blocker_count,
            "nearest_blocker_t": nearest_blocker_t, "max_blocker_opacity": max_blocker_opacity,
        }

    normals = support.normals[candidate_rows]
    tangent_u = support.tangent_u[candidate_rows]
    tangent_v = support.tangent_v[candidate_rows]
    inv_su = support.inv_scale_u[candidate_rows]
    inv_sv = support.inv_scale_v[candidate_rows]
    rho_max = support.rho_max[candidate_rows]
    opacity = support.opacity[candidate_rows]
    numerator = support.center_dot_normal[candidate_rows] - (normals @ origin)
    base_u = (origin @ tangent_u.T) - support.center_dot_tu[candidate_rows]
    base_v = (origin @ tangent_v.T) - support.center_dot_tv[candidate_rows]

    primitives = int(candidate_rows.shape[0])
    chunk = max(1, min(int(relevant_rows.shape[0]), chunk_bytes // max(1, 4 * primitives)))
    for start in range(0, int(relevant_rows.shape[0]), chunk):
        rows = relevant_rows[start:start + chunk]
        direction = query_positions[rows] - origin.reshape(1, 3)
        denominator = direction @ normals.T
        t = numerator.unsqueeze(0) / denominator
        hit = (denominator != 0) & (t > SEGMENT_EPSILON) & (t < 1.0 - SEGMENT_EPSILON)
        del denominator
        local = ((base_u.unsqueeze(0) + t * (direction @ tangent_u.T)) * inv_su.unsqueeze(0)) ** 2
        local = local + ((base_v.unsqueeze(0) + t * (direction @ tangent_v.T)) * inv_sv.unsqueeze(0)) ** 2
        hit = hit & (local <= rho_max.unsqueeze(0))
        del local
        counts = hit.sum(dim=1)
        blocker_count[rows] = counts.to(torch.int32)
        any_hit = counts > 0
        nearest_blocker_t[rows] = torch.where(any_hit, torch.where(hit, t, torch.full_like(t, -1.0)).max(dim=1).values, torch.full_like(counts, float("nan"), dtype=torch.float32))
        max_blocker_opacity[rows] = torch.where(any_hit, torch.where(hit, opacity.unsqueeze(0).expand_as(hit), torch.zeros_like(t)).max(dim=1).values, torch.full_like(counts, float("nan"), dtype=torch.float32))
        blocked_rows = rows[any_hit]
        states[blocked_rows] = STATE_OCCLUDED
        del hit, t, direction

    return {
        "states": states, "blocker_count": blocker_count,
        "nearest_blocker_t": nearest_blocker_t, "max_blocker_opacity": max_blocker_opacity,
    }
