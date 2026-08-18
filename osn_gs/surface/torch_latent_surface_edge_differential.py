from __future__ import annotations

"""Worklog 100 -- symmetric edge differential constraints over a Worklog 98
synchronized tangent frame field.

Worklog 98's tree-integrated ``(u, v)`` accumulates the parametric
potential along ONE Dijkstra spanning-tree path per component: every edge
not on that tree only ever gets checked for holonomy CONSISTENCY, never
used to actually determine ``(u, v)``. This module builds the intrinsic
per-edge differential (``du_ij``, ``dv_ij``) for EVERY continuously
supported source-graph edge in a coherent component -- the input a global
(not single-path) integration needs.

The edge frame ``(e_u_ij, e_v_ij)`` is deliberately SYMMETRIC: it is built
from BOTH endpoints' own synchronized frames after tangent-plane transport
and sign alignment, then averaged -- never arbitrarily inherited from only
one endpoint (which is what a tree-only propagation effectively does for
its own tree edges). Edge weights follow one fixed, deterministic rule
(inverse-square of 3D edge length relative to the component's own median
spacing) -- never tuned from any fit/held-out outcome.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.surface.torch_latent_surface_curve_tracer import propagate_tangent_onto_plane
from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9


@dataclass(frozen=True)
class EdgeDifferential:
    node_a: int
    node_b: int
    du: float
    dv: float
    weight: float


def _symmetric_edge_frame(component: Any, a: int, b: int) -> tuple[Any, Any, Any] | None:
    """Average tangent plane and in-plane basis for edge (a, b), built from
    BOTH endpoints' synchronized frames after transport and sign alignment.
    Returns ``(midpoint_normal, e_u_ij, e_v_ij)`` or ``None`` if the
    transport is degenerate at this edge."""

    torch = require_torch()
    normal_a, normal_b = component.normals[a], component.normals[b]
    midpoint_normal = normal_a + normal_b
    norm = midpoint_normal.norm()
    if float(norm.item()) < _EPS:
        return None
    midpoint_normal = midpoint_normal / norm

    e_u_a_on_mid = propagate_tangent_onto_plane(component.e_u[a], midpoint_normal)
    e_u_b_on_mid = propagate_tangent_onto_plane(component.e_u[b], midpoint_normal)
    if e_u_a_on_mid is None or e_u_b_on_mid is None:
        return None

    # Sign alignment: the two endpoints' own frames were independently
    # synchronized along (possibly different) transport paths back to a
    # common root, so they already agree up to holonomy drift -- but guard
    # against a spurious antiparallel average degenerating to zero.
    if float((e_u_a_on_mid * e_u_b_on_mid).sum().item()) < 0.0:
        e_u_b_on_mid = -e_u_b_on_mid

    e_u_ij = e_u_a_on_mid + e_u_b_on_mid
    e_u_ij_norm = e_u_ij.norm()
    if float(e_u_ij_norm.item()) < _EPS:
        return None
    e_u_ij = e_u_ij / e_u_ij_norm
    e_v_ij = torch.linalg.cross(midpoint_normal, e_u_ij)
    return midpoint_normal, e_u_ij, e_v_ij


def build_edge_differentials(component: Any, median_spacing: float) -> tuple[EdgeDifferential, ...]:
    """Build one symmetric edge differential per continuously-supported
    source-graph edge (the union of the component's spanning tree and its
    tested holonomy edges -- the same adjacency
    :func:`~osn_gs.surface.torch_parametric_domain_validity._source_graph_adjacency`
    uses)."""

    torch = require_torch()
    edges: set[tuple[int, int]] = set()
    for a, b in component.tree_edges:
        edges.add(tuple(sorted((a, b))))
    for edge in component.holonomy_edges:
        edges.add(tuple(sorted((edge.node_a, edge.node_b))))

    spacing = max(median_spacing, _EPS)
    results: list[EdgeDifferential] = []
    for a, b in sorted(edges):
        frame = _symmetric_edge_frame(component, a, b)
        if frame is None:
            continue
        _midpoint_normal, e_u_ij, e_v_ij = frame
        delta = component.positions[b] - component.positions[a]
        du = float((delta * e_u_ij).sum().item())
        dv = float((delta * e_v_ij).sum().item())
        edge_length = float(delta.norm().item())
        # Fixed, deterministic edge weight: inverse-square of 3D edge
        # length relative to the component's own median spacing -- a
        # standard graph-Laplacian-style weighting choice, clamped so a
        # near-duplicate pair of points cannot blow up the system. Never
        # adjusted from any replay/held-out outcome.
        relative_length = max(edge_length / spacing, 1e-2)
        weight = 1.0 / (relative_length * relative_length)
        results.append(EdgeDifferential(node_a=a, node_b=b, du=du, dv=dv, weight=weight))
    return tuple(results)
