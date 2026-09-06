from __future__ import annotations

"""W174 reference zero-set surface-complex utilities.

Diagnostic only.  Nothing here changes W154 charting, region construction or
the selected cell population.  These routines recover the marching-cubes
triangle complex that the frozen corner scalars already determine, and report
its topology exactly.

OWNERSHIP RULE (construction-native, verified by the W174 feasibility probe):
each authoritative cell is decoded into its own 2x2x2 scalar block, so a
triangle can only be attributed to the cell whose eight stored corner values
produced it.  No distance, radius, normal or threshold matching is used to
associate triangles with support rows anywhere in this module.

VERTEX WELDING: marching-cubes vertices are generated per cell, so the same
lattice edge is cut once per incident cell.  Triangle-level adjacency is
therefore established by welding vertices on exact lattice-edge identity --
a vertex is keyed by the ordered pair of integer lattice corners it lies
between plus the exact interpolation parameter recomputed from the stored
scalars.  This is an exact structural key, not a spatial tolerance merge.
"""

from collections import deque

import numpy as np

# Corner ordering of the frozen W154 extractor, reproduced verbatim.
CORNER_OFFSETS = np.asarray(
    [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
     (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)],
    dtype=np.int64,
)

# Scale used to turn the interpolation parameter into an exact integer key.
# The parameter is a ratio of stored float32 scalars; quantising at 1e9 keeps
# distinct crossings distinct while making identical crossings compare equal.
_T_QUANT = 1_000_000_000


def cell_scalar_block(corner_values_row):
    """Lay the 8 stored corner scalars out as the 2x2x2 block MC expects."""
    block = np.zeros((2, 2, 2), dtype=np.float64)
    for corner, (dx, dy, dz) in enumerate(CORNER_OFFSETS):
        block[dx, dy, dz] = corner_values_row[corner]
    return block


def _lattice_edge_key(cell, local_vertex):
    """Key a marching-cubes vertex by the global lattice edge it cuts.

    A MC vertex on a unit cell always lies on a cube edge, so exactly two of
    its local coordinates are integers (0 or 1) and one is fractional -- or all
    three are integral when the crossing sits exactly on a lattice corner.
    Both cases are keyed exactly, in global lattice coordinates, so two cells
    that cut the same physical edge produce the same key.
    """
    glob = cell.astype(np.int64) + np.rint(local_vertex).astype(np.int64) * 0
    frac_axis = -1
    lo = np.empty(3, dtype=np.int64)
    t = 0
    for axis in range(3):
        value = local_vertex[axis]
        nearest = round(value)
        if abs(value - nearest) <= 1e-9:
            lo[axis] = cell[axis] + int(nearest)
        else:
            frac_axis = axis
            lo[axis] = cell[axis] + int(np.floor(value))
            t = int(round((value - np.floor(value)) * _T_QUANT))
    del glob
    return (int(lo[0]), int(lo[1]), int(lo[2]), frac_axis, t)


def build_surface_complex(corner_values, cells, h, *, marching_cubes=None):
    """Recover the per-cell MC triangle complex and weld it into one mesh.

    Returns a dict holding welded vertices (world units), triangles, the owning
    support row of every triangle, and per-cell accounting.  Every selected row
    is represented; cells that yield no triangle are recorded, never dropped.
    """
    if marching_cubes is None:
        from skimage.measure import marching_cubes

    vertex_ids: dict[tuple, int] = {}
    vertex_world: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    triangle_owner: list[int] = []
    per_cell_counts = np.zeros(len(cells), dtype=np.int64)
    per_cell_patches = np.zeros(len(cells), dtype=np.int64)
    cells_without_triangles: list[int] = []
    containment_violations = 0

    for row in range(len(cells)):
        block = cell_scalar_block(corner_values[row])
        try:
            verts, tris, _n, _v = marching_cubes(
                block, level=0.0, step_size=1, method="lewiner", allow_degenerate=False
            )
        except (ValueError, RuntimeError):
            cells_without_triangles.append(row)
            continue
        if tris.shape[0] == 0:
            cells_without_triangles.append(row)
            continue

        corners = verts[tris]
        if not np.all((corners >= -1e-9) & (corners <= 1.0 + 1e-9)):
            containment_violations += 1

        cell = cells[row].astype(np.int64)
        local_to_global = np.empty(len(verts), dtype=np.int64)
        for local_index, local_vertex in enumerate(verts):
            key = _lattice_edge_key(cell, local_vertex)
            found = vertex_ids.get(key)
            if found is None:
                found = len(vertex_world)
                vertex_ids[key] = found
                # Frozen W154 cell convention: (cell + local + 0.5) * h.
                vertex_world.append((cell.astype(np.float64) + local_vertex + 0.5) * h)
            local_to_global[local_index] = found

        for tri in tris:
            triangles.append(tuple(int(local_to_global[v]) for v in tri))
            triangle_owner.append(row)
        per_cell_counts[row] = int(tris.shape[0])
        per_cell_patches[row] = _local_patch_count(tris)

    return {
        "vertices": np.asarray(vertex_world, dtype=np.float64).reshape(-1, 3),
        "triangles": np.asarray(triangles, dtype=np.int64).reshape(-1, 3),
        "triangle_owner_row": np.asarray(triangle_owner, dtype=np.int64),
        "per_cell_triangle_count": per_cell_counts,
        "per_cell_patch_count": per_cell_patches,
        "cells_without_triangles": np.asarray(cells_without_triangles, dtype=np.int64),
        "unit_cell_containment_violations": int(containment_violations),
        "vertex_welding_rule": "exact global lattice-edge key (integer corners + quantised crossing parameter); no spatial tolerance merge",
    }


def _local_patch_count(tris):
    """Count vertex-connected triangle patches inside one cell."""
    parent = list(range(len(tris)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    seen: dict[int, int] = {}
    for index, tri in enumerate(tris):
        for vertex in tri:
            vertex = int(vertex)
            if vertex in seen:
                a, b = find(index), find(seen[vertex])
                if a != b:
                    parent[a] = b
            else:
                seen[vertex] = index
    return len({find(i) for i in range(len(tris))})


def edge_incidence(triangles):
    """Map each undirected mesh edge to the triangles using it."""
    incidence: dict[tuple[int, int], list[int]] = {}
    for index, (a, b, c) in enumerate(triangles):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            incidence.setdefault(key, []).append(index)
    return incidence


def complex_topology(complex_data):
    """Exact topology accounting for the welded triangle complex."""
    triangles = complex_data["triangles"]
    incidence = edge_incidence(triangles)
    degrees = np.array([len(v) for v in incidence.values()], dtype=np.int64)

    boundary_edges = [edge for edge, tris in incidence.items() if len(tris) == 1]
    non_manifold_edges = [edge for edge, tris in incidence.items() if len(tris) > 2]

    used_vertices = np.unique(triangles.reshape(-1)) if len(triangles) else np.empty(0, dtype=np.int64)
    vertex_count = int(used_vertices.size)
    edge_count = len(incidence)
    triangle_count = int(len(triangles))

    components = triangle_components(triangles, incidence)
    boundary_component_count = _boundary_loop_components(boundary_edges)

    return {
        "triangle_count": triangle_count,
        "welded_vertex_count": vertex_count,
        "edge_count": edge_count,
        "euler_characteristic": vertex_count - edge_count + triangle_count,
        "triangle_connected_components": int(components.max() + 1) if components.size else 0,
        "edge_degree_histogram": {int(d): int(c) for d, c in zip(*np.unique(degrees, return_counts=True))},
        "boundary_edge_count": len(boundary_edges),
        "boundary_edge_components": boundary_component_count,
        "non_manifold_edge_count": len(non_manifold_edges),
        "non_manifold_vertex_count": _non_manifold_vertex_count(triangles, incidence),
        "interpretation": "Topology accounting only; Euler characteristic and genus are not physical semantics.",
    }


def triangle_components(triangles, incidence=None):
    """Label triangles by edge-connected component (deterministic order)."""
    if len(triangles) == 0:
        return np.empty(0, dtype=np.int64)
    if incidence is None:
        incidence = edge_incidence(triangles)
    neighbours: list[list[int]] = [[] for _ in range(len(triangles))]
    for tris in incidence.values():
        for i in range(len(tris)):
            for j in range(i + 1, len(tris)):
                neighbours[tris[i]].append(tris[j])
                neighbours[tris[j]].append(tris[i])

    labels = np.full(len(triangles), -1, dtype=np.int64)
    label = 0
    for seed in range(len(triangles)):
        if labels[seed] >= 0:
            continue
        queue = deque([seed])
        labels[seed] = label
        while queue:
            current = queue.popleft()
            for neighbour in neighbours[current]:
                if labels[neighbour] < 0:
                    labels[neighbour] = label
                    queue.append(neighbour)
        label += 1
    return labels


def _boundary_loop_components(boundary_edges):
    """Connected components of the boundary-edge graph."""
    if not boundary_edges:
        return 0
    adjacency: dict[int, list[int]] = {}
    for u, v in boundary_edges:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
    seen: set[int] = set()
    count = 0
    for start in adjacency:
        if start in seen:
            continue
        count += 1
        queue = deque([start])
        seen.add(start)
        while queue:
            current = queue.popleft()
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
    return count


def _non_manifold_vertex_count(triangles, incidence):
    """Vertices whose incident triangle fan is not a single disc/fan.

    A vertex is flagged when the triangles around it do not form one
    edge-connected group -- the standard non-manifold pinch condition.
    """
    fans: dict[int, list[int]] = {}
    for index, tri in enumerate(triangles):
        for vertex in tri:
            fans.setdefault(int(vertex), []).append(index)

    edge_pairs: dict[tuple[int, int], list[int]] = {}
    for edge, tris in incidence.items():
        edge_pairs[edge] = tris

    flagged = 0
    for vertex, incident in fans.items():
        if len(incident) < 2:
            continue
        local = {t: i for i, t in enumerate(incident)}
        parent = list(range(len(incident)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for edge, tris in edge_pairs.items():
            if vertex not in edge:
                continue
            members = [local[t] for t in tris if t in local]
            for i in range(1, len(members)):
                a, b = find(members[0]), find(members[i])
                if a != b:
                    parent[a] = b
        if len({find(i) for i in range(len(incident))}) > 1:
            flagged += 1
    return flagged


def surface_shortest_path(triangles, source_triangles, target_triangles, incidence=None):
    """Deterministic BFS over edge-adjacent triangles.

    Returns the triangle-index path and its length, or None when the targets
    are not reachable.  No distance threshold participates: adjacency is shared
    mesh edges only.
    """
    if len(triangles) == 0 or not len(source_triangles) or not len(target_triangles):
        return None
    if incidence is None:
        incidence = edge_incidence(triangles)
    neighbours: list[list[int]] = [[] for _ in range(len(triangles))]
    for tris in incidence.values():
        for i in range(len(tris)):
            for j in range(i + 1, len(tris)):
                neighbours[tris[i]].append(tris[j])
                neighbours[tris[j]].append(tris[i])
    for group in neighbours:
        group.sort()

    targets = set(int(t) for t in target_triangles)
    previous = {}
    queue = deque()
    for seed in sorted(int(t) for t in source_triangles):
        if seed not in previous:
            previous[seed] = -1
            queue.append(seed)
    while queue:
        current = queue.popleft()
        if current in targets:
            path = [current]
            while previous[path[-1]] != -1:
                path.append(previous[path[-1]])
            path.reverse()
            return {"triangle_path": path, "edge_steps": len(path) - 1}
        for neighbour in neighbours[current]:
            if neighbour not in previous:
                previous[neighbour] = current
                queue.append(neighbour)
    return None
