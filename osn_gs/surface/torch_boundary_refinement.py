from __future__ import annotations

"""Stage 1-F density-refined support boundary primitives.

Boundary leaves (leaves with at least one exterior/unresolved face, see
``compute_leaf_face_adjacency``) get their support refined by a plain 2D
density field over the leaf's tangent-plane UV domain:

- density = unweighted Gaussian KDE over the raw Gaussian centers projected
  into the leaf's UV frame (no opacity/planarity/eligibility weights — those
  are Stage 2 support-mass concepts),
- bandwidth = a configured multiple of the leaf's median UV nearest-neighbor
  spacing,
- threshold = a configured fraction of the median density at the data points
  themselves, so the crossing level is scene-independent by construction.

The sub-voxel contour comes from marching squares with linear edge
interpolation on the density grid (not from binary cell boundaries).
"""

from typing import Any

from osn_gs.utils.torch_ops import require_torch


def median_nn_spacing(uv: Any) -> float:
    """Median nearest-neighbor distance of ``(N, 2)`` UV samples."""

    torch = require_torch()
    count = int(uv.shape[0])
    if count < 2:
        return 1.0
    distances = torch.cdist(uv, uv)
    distances.fill_diagonal_(float("inf"))
    return float(distances.min(dim=1).values.median())


def sample_nn_spacings(uv: Any) -> Any:
    """Per-sample nearest-neighbor distance of ``(N, 2)`` UV samples.

    Feeds the per-sample adaptive KDE bandwidth (§3: a simple multiple of the
    *local* UV nearest-neighbor spacing). A per-sample bandwidth makes the
    density value approximately "neighbors within k local spacings" — invariant
    to density variation — so a single global threshold does not erase sparse
    but uniformly supported regions (density_gradient failure otherwise).
    """

    torch = require_torch()
    count = int(uv.shape[0])
    if count < 2:
        return torch.ones((count,), dtype=uv.dtype, device=uv.device)
    distances = torch.cdist(uv, uv)
    distances.fill_diagonal_(float("inf"))
    spacings = distances.min(dim=1).values
    fallback = spacings[torch.isfinite(spacings)].median()
    return torch.nan_to_num(spacings, nan=float(fallback), posinf=float(fallback)).clamp_min(1e-9)


def kde_density(queries: Any, samples: Any, bandwidth: Any) -> Any:
    """Unweighted Gaussian KDE ``sum_i exp(-0.5 |q - s_i|^2 / h_i^2)`` per query.

    ``bandwidth`` is either a scalar or a per-sample ``(N,)`` tensor (adaptive
    bandwidth; kernels are not 1/h-normalized, so each sample contributes at
    most 1 regardless of its bandwidth).
    """

    torch = require_torch()
    if int(samples.shape[0]) == 0:
        return torch.zeros((int(queries.shape[0]),), dtype=queries.dtype, device=queries.device)
    if torch.is_tensor(bandwidth) and bandwidth.ndim > 0:
        h = bandwidth.reshape(1, -1).clamp_min(1e-9)
    else:
        h = torch.tensor(
            [[max(float(bandwidth), 1e-9)]], dtype=queries.dtype, device=queries.device
        )
    return torch.exp(-0.5 * torch.cdist(queries, samples).square() / h.square()).sum(dim=1)


def density_grid(samples: Any, resolution: int, bandwidth: Any) -> Any:
    """``(resolution, resolution)`` KDE grid at UV cell centers over [0, 1]^2."""

    torch = require_torch()
    resolution = max(2, int(resolution))
    centers = (torch.arange(resolution, dtype=samples.dtype, device=samples.device) + 0.5) / resolution
    grid_u, grid_v = torch.meshgrid(centers, centers, indexing="ij")
    queries = torch.stack([grid_u.reshape(-1), grid_v.reshape(-1)], dim=1)
    return kde_density(queries, samples, bandwidth).reshape(resolution, resolution)


def marching_squares(grid: Any, level: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Iso-contour segments of ``grid`` at ``level`` via linear edge interpolation.

    ``grid[i, j]`` is the field value at UV node ``((i + 0.5) / R, (j + 0.5) / R)``.
    Returns UV-space segments. Saddle cells (4 crossings) are disambiguated by
    the cell-center mean, which keeps the output deterministic.
    """

    values = grid.detach().cpu()
    resolution = int(values.shape[0])
    if resolution < 2:
        return []

    def node_uv(i: int, j: int) -> tuple[float, float]:
        return ((i + 0.5) / resolution, (j + 0.5) / resolution)

    def interpolate(i0: int, j0: int, i1: int, j1: int) -> tuple[float, float]:
        va = float(values[i0, j0]) - level
        vb = float(values[i1, j1]) - level
        t = va / (va - vb) if abs(va - vb) > 1e-12 else 0.5
        a, b = node_uv(i0, j0), node_uv(i1, j1)
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            corner_nodes = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
            inside = [float(values[a, b]) >= level for a, b in corner_nodes]
            if all(inside) or not any(inside):
                continue
            # Cell edges in order: bottom, right, top, left (node index pairs).
            edges = ((0, 1), (1, 2), (2, 3), (3, 0))
            crossings = []
            for edge_index, (a, b) in enumerate(edges):
                if inside[a] != inside[b]:
                    na, nb = corner_nodes[a], corner_nodes[b]
                    crossings.append((edge_index, interpolate(na[0], na[1], nb[0], nb[1])))
            if len(crossings) == 2:
                segments.append((crossings[0][1], crossings[1][1]))
            elif len(crossings) == 4:
                center_inside = sum(float(values[a, b]) for a, b in corner_nodes) / 4.0 >= level
                # Pair crossings so the two segments separate the corners
                # consistently with the center estimate.
                by_edge = dict(crossings)
                if center_inside:
                    segments.append((by_edge[0], by_edge[1]))
                    segments.append((by_edge[2], by_edge[3]))
                else:
                    segments.append((by_edge[3], by_edge[0]))
                    segments.append((by_edge[1], by_edge[2]))
    return segments


def contour_length_uv(segments: list[tuple[tuple[float, float], tuple[float, float]]]) -> float:
    return sum(
        ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in segments
    )
