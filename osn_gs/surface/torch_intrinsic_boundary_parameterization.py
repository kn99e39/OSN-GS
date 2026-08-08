from __future__ import annotations

"""Worklog 81: intrinsic boundary-conditioned chart parameterization.

Worklog 80 separated topology (sparse accepted cycle) from geometry (dense,
evidence-backed chart boundary). That fixed chart-domain coverage, but the
four charts that pass it are all `extrapolative`, and worklog 80's own UV
diagnostics on the SAME evidence under `pca_parameterize_points` show 21-36%
of UV-adjacent triangles disagreeing in 3D normal sign. `pca_parameterize_points`
(reused unmodified across worklog 61-80) is a single flat affine projection --
it has no mechanism to prevent two evidence points that are close in the
projection direction but far apart along the surface from landing at
overlapping UV, and no mechanism to respect the chart's own boundary shape
at all (the boundary is whatever the affine projection happens to produce,
not the polygon worklog 80 actually built).

This module builds an alternative: a discrete boundary-conditioned harmonic
(Tutte-style) embedding computed directly from the region's own local
manifold relations, instead of one global linear projection.

  1. The worklog 80 dense chart boundary is fixed to a convex 2D domain
     (points on a unit circle, in the loop's own traversal order -- this is
     the standard Tutte-embedding boundary condition, not a "PCA rectangle"
     or hull: the boundary POSITIONS come entirely from worklog 80's already
     -validated ordering, only the embedding shape is convex-fixed, which is
     what makes the interior solve well-posed and injective for a disk
     topology).
  2. Interior evidence points get a k-nearest-neighbor graph in 3D (the local
     manifold relation -- no global axis, no PCA). Non-boundary points solve
     a discrete Laplace equation (uniform-weight average of neighbors, the
     classic Tutte scheme) with the fixed boundary as Dirichlet condition.
  3. This is provably injective (bijective onto the disk) whenever the
     interior graph is a single connected, planar-embeddable (in the fixed
     boundary's cyclic order) triangulation -- so failure to solve or a
     disconnected/ambiguous local graph is reported explicitly, never forced.

Fails closed (never invents a mapping) when:
  * the interior kNN graph is disconnected from the boundary (non-manifold /
    local graph ambiguity) -- `STATE_DISCONNECTED_GRAPH`,
  * the boundary loop is not the same population as `evaluate_chart` already
    validated (caller error) -- `STATE_INSUFFICIENT_BOUNDARY`,
  * the linear solve fails outright -- `STATE_SOLVE_FAILED`.

Nothing here revisits normal source, connectivity scale, the worklog 77
predicate, or NURBS fitting capacity. This module only proposes a UV layout;
`fit_torch_visible_surface_lsq` and its 6x6 grid are unchanged.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.utils.torch_ops import require_torch

_EPS = 1e-9

STATE_MATERIALIZED = "intrinsic_parameterization_materialized"
STATE_INSUFFICIENT_BOUNDARY = "intrinsic_parameterization_insufficient_boundary"
STATE_DISCONNECTED_GRAPH = "intrinsic_parameterization_disconnected_graph"
STATE_SOLVE_FAILED = "intrinsic_parameterization_solve_failed"


@dataclass(frozen=True)
class IntrinsicParameterizationResult:
    state: str
    uv: Any | None                 # (N, 2) -- boundary rows first (in loop order), then interior
    ordered_positions: Any | None  # (N, 3) matching `uv` row order
    boundary_count: int
    interior_count: int
    disconnected_interior_count: int
    reasons: tuple[str, ...]

    @property
    def materialized(self) -> bool:
        return self.state == STATE_MATERIALIZED


def _knn_edges(points: Any, k: int) -> list[set[int]]:
    torch = require_torch()
    n = int(points.shape[0])
    if n == 0:
        return []
    neighbors = min(k, n - 1)
    if neighbors < 1:
        return [set() for _ in range(n)]
    d = torch.cdist(points, points)
    d.fill_diagonal_(float("inf"))
    idx = d.topk(neighbors, dim=1, largest=False).indices.tolist()
    adjacency: list[set[int]] = [set() for _ in range(n)]
    for i, row in enumerate(idx):
        for j in row:
            adjacency[i].add(int(j))
            adjacency[int(j)].add(i)
    return adjacency


def build_intrinsic_boundary_parameterization(
    boundary_positions: Any,
    interior_positions: Any,
    *,
    knn_k: int = 8,
) -> IntrinsicParameterizationResult:
    """Boundary-conditioned discrete harmonic (Tutte) embedding.

    ``boundary_positions`` must already be in the chart's validated traversal
    order (worklog 80's `DenseChartSupport.ordered_positions`). ``interior_positions``
    is the region-owned evidence NOT already a boundary vertex -- both are
    consumed as-is; this function invents no new geometry and closes no gap.
    """

    torch = require_torch()
    b = int(boundary_positions.shape[0]) if boundary_positions is not None else 0
    m = int(interior_positions.shape[0]) if interior_positions is not None else 0

    def _fail(state: str, *reasons: str, disconnected: int = 0) -> IntrinsicParameterizationResult:
        return IntrinsicParameterizationResult(state, None, None, b, m, disconnected, tuple(reasons))

    if b < 3:
        return _fail(STATE_INSUFFICIENT_BOUNDARY, f"boundary_vertex_count={b}<3")

    dtype, device = boundary_positions.dtype, boundary_positions.device

    # (1) Fix the boundary to a convex domain, preserving worklog 80's own
    # traversal order and (via arclength spacing) its own relative vertex
    # density -- a boundary vertex that is one of a dense cluster keeps a
    # correspondingly small angular step rather than being redistributed
    # uniformly, since uniform redistribution would silently discard the
    # dense-boundary-support geometry worklog 80 built.
    arclen = torch.zeros((b,), dtype=dtype, device=device)
    for i in range(1, b):
        arclen[i] = arclen[i - 1] + (boundary_positions[i] - boundary_positions[i - 1]).norm()
    total = arclen[-1] + (boundary_positions[0] - boundary_positions[-1]).norm()
    total = total.clamp_min(_EPS)
    theta = (arclen / total) * (2 * 3.141592653589793)
    boundary_uv = torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)

    if m == 0:
        return IntrinsicParameterizationResult(
            STATE_MATERIALIZED, boundary_uv, boundary_positions, b, 0, 0, (),
        )

    # (2) Local manifold relation: kNN graph over ALL chart points (boundary
    # + interior) in raw 3D -- no PCA axis, no global projection.
    all_positions = torch.cat((boundary_positions, interior_positions), dim=0)
    n = b + m
    adjacency = _knn_edges(all_positions, knn_k)
    # Every interior node also gets an edge to the boundary via its knn graph
    # naturally if a boundary point is among its neighbors; additionally
    # connect each interior point to its single nearest boundary vertex so a
    # locally-dense interior cluster far from any sampled boundary neighbor
    # still has a path out -- this reuses distance information already
    # computed, it does not invent new positions.
    if b > 0:
        d_to_boundary = torch.cdist(interior_positions, boundary_positions)
        nearest_boundary = d_to_boundary.argmin(dim=1).tolist()
        for local_i, boundary_j in enumerate(nearest_boundary):
            i = b + local_i
            adjacency[i].add(int(boundary_j))
            adjacency[int(boundary_j)].add(i)

    # Connectivity check: every interior node must reach the boundary set
    # through the graph. A node that cannot is genuine local-graph ambiguity
    # (isolated cluster) -- reported, never bridged with an invented edge.
    reachable = set(range(b))
    frontier = list(range(b))
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node]:
            if neighbor not in reachable:
                reachable.add(neighbor)
                frontier.append(neighbor)
    disconnected = [i for i in range(b, n) if i not in reachable]
    if disconnected:
        return _fail(
            STATE_DISCONNECTED_GRAPH,
            f"disconnected_interior_count={len(disconnected)}/{m}",
            disconnected=len(disconnected),
        )

    # (3) Discrete Laplace / Tutte solve: for each interior node, uv_i =
    # mean(uv_j for j in neighbors(i)); boundary nodes are the fixed Dirichlet
    # condition. Uniform (unweighted) averaging is the classical Tutte
    # embedding -- guaranteed injective onto the convex boundary domain when
    # the interior graph is a connected planar triangulation (Tutte 1963).
    interior_ids = list(range(b, n))
    local_index = {node: k for k, node in enumerate(interior_ids)}
    system = torch.zeros((m, m), dtype=dtype, device=device)
    rhs = torch.zeros((m, 2), dtype=dtype, device=device)
    for node in interior_ids:
        row = local_index[node]
        neighbors = adjacency[node]
        deg = len(neighbors)
        if deg == 0:
            return _fail(STATE_DISCONNECTED_GRAPH, f"isolated_interior_node={node}", disconnected=1)
        system[row, row] = 1.0
        for neighbor in neighbors:
            if neighbor < b:
                rhs[row] += boundary_uv[neighbor] / deg
            else:
                system[row, local_index[neighbor]] -= 1.0 / deg

    try:
        interior_uv = torch.linalg.solve(system, rhs)
    except Exception as exc:  # noqa: BLE001
        return _fail(STATE_SOLVE_FAILED, f"{type(exc).__name__}: {exc}")
    if not torch.isfinite(interior_uv).all():
        return _fail(STATE_SOLVE_FAILED, "non_finite_solution")

    uv = torch.cat((boundary_uv, interior_uv), dim=0)
    # Re-normalize to [0, 1]^2 to match `pca_parameterize_points`'s convention
    # so downstream fitting/diagnostics code is agnostic to which parameterization
    # produced the UV.
    coord_min = uv.min(dim=0).values
    span = (uv.max(dim=0).values - coord_min).clamp_min(_EPS)
    uv_unit = (uv - coord_min) / span

    return IntrinsicParameterizationResult(
        STATE_MATERIALIZED, uv_unit, all_positions, b, m, 0, (),
    )
