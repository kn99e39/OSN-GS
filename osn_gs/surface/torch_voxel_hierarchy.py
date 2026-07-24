from __future__ import annotations

"""Stage 1 recursive raw-count adaptive voxel hierarchy.

This is the retained experimental ``voxel_patch_stage1`` ablation builder.
Its support-mode evidence is recorded in ``docs/worklogs/33_stage1_support_modes.md``.
It is intentionally separate from the legacy ``torch_voxel_regions`` builder
so the legacy constructor path stays byte-identical.

Policy (Stage 1, raw count only - no support mass, no merging):

- ``count < voxel_min_gaussian_count``  -> INACTIVE leaf (no patch)
- ``min <= count <= max``               -> ACTIVE leaf (one NURBS patch)
- ``count > voxel_max_gaussian_count``  -> subdivide into 8 children until
  ``voxel_max_depth`` or ``voxel_min_size`` blocks further splits
- non-planar leaf                       -> subdivide if allowed, else COMPLEX

Every node records parent/children, a stable path-based ID, its world AABB,
its raw Gaussian indices/count, and (for leaves with enough points) a local
plane descriptor. Empty children of a subdivided node are recorded explicitly.
"""

from dataclasses import dataclass, field
from typing import Any

from osn_gs.utils.torch_ops import require_torch

STATE_SUBDIVIDED = "subdivided"
STATE_ACTIVE = "active"
STATE_INACTIVE = "inactive"
STATE_COMPLEX = "complex"
STATE_EMPTY = "empty"


@dataclass
class LeafPlaneDescriptor:
    """Local PCA plane of the Gaussian centers inside one voxel."""

    centroid: Any  # (3,)
    normal: Any  # (3,)
    tangent_u: Any  # (3,)
    tangent_v: Any  # (3,)
    singular_values: Any  # (3,) descending
    local_plane_rms: float
    local_plane_max_error: float
    # smallest/mid standard-deviation ratio: ~0 for a plane, ~1 for a ball.
    thickness_ratio: float
    eigenvalue_ratio: float
    degenerate: bool


@dataclass
class VoxelNode:
    """One node of the recursive hierarchy (internal, leaf, or empty child)."""

    index: int
    node_id: str  # stable path ID: "r", "r3", "r35", ... (octant digits)
    parent: int  # index into nodes, -1 for the root
    depth: int
    aabb_min: Any  # (3,) float tensor, world space
    aabb_max: Any  # (3,) float tensor, world space
    count: int
    state: str
    children: list[int] = field(default_factory=list)  # indices into nodes
    gaussian_indices: Any | None = None  # (count,) long tensor for leaves
    plane: LeafPlaneDescriptor | None = None
    subdivision_reason: str | None = None  # "count" | "complex" for SUBDIVIDED
    subdivision_blocked: str | None = None  # "max_depth" | "min_size" when a
    # leaf wanted to split but could not


@dataclass
class TorchVoxelGaussianHierarchy:
    """Recursive raw-count voxel hierarchy over Gaussian centers."""

    nodes: list[VoxelNode]
    point_count: int
    config: dict[str, Any]

    def leaves(self) -> list[VoxelNode]:
        return [node for node in self.nodes if node.state != STATE_SUBDIVIDED]

    def leaves_in_state(self, state: str) -> list[VoxelNode]:
        return [node for node in self.nodes if node.state == state]

    def state_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node in self.nodes:
            counts[node.state] = counts.get(node.state, 0) + 1
        return counts

    def max_depth_reached(self) -> int:
        return max((node.depth for node in self.nodes), default=0)


def _fit_leaf_plane(points: Any, degenerate_eps: float = 1e-9) -> LeafPlaneDescriptor | None:
    """PCA plane of ``points`` (>= 3 required); ``None`` when too few points."""

    torch = require_torch()
    if int(points.shape[0]) < 3:
        return None
    centroid = points.mean(dim=0)
    centered = points - centroid
    # Economy SVD of the centered cloud: right singular vectors are the PCA axes.
    _, sigma, vh = torch.linalg.svd(centered, full_matrices=False)
    tangent_u, tangent_v, normal = vh[0], vh[1], vh[2]
    offsets = centered @ normal
    rms = float(offsets.square().mean().sqrt())
    max_error = float(offsets.abs().max())
    s0 = float(sigma[0])
    s1 = float(sigma[1]) if sigma.numel() > 1 else 0.0
    s2 = float(sigma[2]) if sigma.numel() > 2 else 0.0
    degenerate = s1 <= degenerate_eps
    thickness_ratio = 1.0 if degenerate else s2 / max(s1, degenerate_eps)
    eigenvalue_ratio = 1.0 if degenerate else (s2 * s2) / max(s1 * s1, degenerate_eps)
    return LeafPlaneDescriptor(
        centroid=centroid,
        normal=normal,
        tangent_u=tangent_u,
        tangent_v=tangent_v,
        singular_values=sigma,
        local_plane_rms=rms,
        local_plane_max_error=max_error,
        thickness_ratio=thickness_ratio,
        eigenvalue_ratio=eigenvalue_ratio,
        degenerate=degenerate,
    )


def build_voxel_gaussian_hierarchy(
    points: Any,
    voxel_min_gaussian_count: int = 10,
    voxel_max_gaussian_count: int = 150,
    voxel_max_depth: int = 6,
    voxel_min_size: float = 0.0,
    complex_thickness_ratio: float = 0.35,
    subdivide_complex: bool = True,
) -> TorchVoxelGaussianHierarchy:
    """Build the Stage 1 recursive raw-count hierarchy over Gaussian centers.

    Deterministic: the space partition depends only on the input point bounds
    and the config, children are always visited in octant order 0..7, and node
    IDs are the octant-digit paths, so leaf IDs are stable across runs.
    """

    torch = require_torch()
    points = torch.as_tensor(
        points, dtype=torch.float32, device=points.device if hasattr(points, "device") else None
    )
    count = int(points.shape[0])
    min_count = max(1, int(voxel_min_gaussian_count))
    max_count = max(min_count, int(voxel_max_gaussian_count))
    max_depth = max(0, int(voxel_max_depth))
    min_size = max(0.0, float(voxel_min_size))
    config = {
        "voxel_min_gaussian_count": min_count,
        "voxel_max_gaussian_count": max_count,
        "voxel_max_depth": max_depth,
        "voxel_min_size": min_size,
        "complex_thickness_ratio": float(complex_thickness_ratio),
        "subdivide_complex": bool(subdivide_complex),
    }

    nodes: list[VoxelNode] = []
    if count == 0:
        return TorchVoxelGaussianHierarchy(nodes=nodes, point_count=0, config=config)

    root_min = points.min(dim=0).values
    root_max = points.max(dim=0).values
    # A zero-thickness axis (e.g. a perfect z=0 plane) still needs a valid AABB.
    span = (root_max - root_min).clamp_min(1e-6)
    root_max = root_min + span

    all_indices = torch.arange(count, dtype=torch.long, device=points.device)
    # Explicit stack instead of recursion; children pushed in reverse octant
    # order so they are *processed* (and therefore indexed) in octant order.
    stack: list[tuple[str, int, int, Any, Any, Any]] = [
        ("r", -1, 0, root_min, root_max, all_indices)
    ]
    while stack:
        node_id, parent, depth, aabb_min, aabb_max, indices = stack.pop()
        node_count = int(indices.numel())
        index = len(nodes)
        node = VoxelNode(
            index=index,
            node_id=node_id,
            parent=parent,
            depth=depth,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            count=node_count,
            state=STATE_EMPTY,
        )
        nodes.append(node)
        if parent >= 0:
            nodes[parent].children.append(index)

        if node_count == 0:
            # Recorded empty child of a subdivided node.
            continue

        node_points = points[indices]
        plane = _fit_leaf_plane(node_points)
        is_complex = plane is not None and (
            plane.degenerate or plane.thickness_ratio > float(complex_thickness_ratio)
        )
        wants_split_count = node_count > max_count
        wants_split_complex = bool(subdivide_complex) and is_complex and node_count >= 2 * min_count
        extent = aabb_max - aabb_min
        child_edge = float(extent.max()) * 0.5
        blocked = None
        if depth >= max_depth:
            blocked = "max_depth"
        elif min_size > 0.0 and child_edge < min_size:
            blocked = "min_size"
        can_split = blocked is None and node_count >= 2

        if (wants_split_count or wants_split_complex) and can_split:
            node.state = STATE_SUBDIVIDED
            node.subdivision_reason = "count" if wants_split_count else "complex"
            node.plane = plane
            mid = (aabb_min + aabb_max) * 0.5
            octant = (
                (node_points[:, 0] >= mid[0]).long() * 4
                + (node_points[:, 1] >= mid[1]).long() * 2
                + (node_points[:, 2] >= mid[2]).long()
            )
            for child_octant in range(7, -1, -1):
                child_indices = indices[octant == child_octant]
                bits = ((child_octant >> 2) & 1, (child_octant >> 1) & 1, child_octant & 1)
                child_min = aabb_min.clone()
                child_max = mid.clone()
                for axis, bit in enumerate(bits):
                    if bit:
                        child_min[axis] = mid[axis]
                        child_max[axis] = aabb_max[axis]
                stack.append(
                    (f"{node_id}{child_octant}", index, depth + 1, child_min, child_max, child_indices)
                )
            continue

        # Leaf classification.
        node.gaussian_indices = indices
        node.plane = plane
        if wants_split_count or (wants_split_complex and blocked is not None):
            node.subdivision_blocked = blocked
        if node_count < min_count:
            node.state = STATE_INACTIVE
        elif is_complex:
            node.state = STATE_COMPLEX
        else:
            node.state = STATE_ACTIVE

    return TorchVoxelGaussianHierarchy(nodes=nodes, point_count=count, config=config)


def validate_hierarchy_conservation(hierarchy: TorchVoxelGaussianHierarchy) -> None:
    """Raise ``AssertionError`` unless Gaussian assignment is conserved.

    Checks that every subdivided node's count equals the sum of its children's
    counts, and that the leaf Gaussian indices form an exact partition of the
    input points (every point in exactly one leaf).
    """

    torch = require_torch()
    for node in hierarchy.nodes:
        if node.state == STATE_SUBDIVIDED:
            child_total = sum(hierarchy.nodes[child].count for child in node.children)
            assert child_total == node.count, (
                f"node {node.node_id}: children counts {child_total} != parent count {node.count}"
            )
            assert len(node.children) == 8, f"node {node.node_id}: expected 8 recorded children"
    leaf_indices = [
        node.gaussian_indices
        for node in hierarchy.leaves()
        if node.gaussian_indices is not None and node.count > 0
    ]
    total = sum(int(part.numel()) for part in leaf_indices)
    assert total == hierarchy.point_count, (
        f"leaf indices cover {total} points, expected {hierarchy.point_count}"
    )
    if leaf_indices:
        merged = torch.cat(leaf_indices)
        assert int(torch.unique(merged).numel()) == hierarchy.point_count, (
            "leaf Gaussian indices are not a disjoint partition"
        )


def hierarchy_payload(
    hierarchy: TorchVoxelGaussianHierarchy, include_gaussian_indices: bool = True
) -> dict[str, Any]:
    """JSON-serializable provenance of the full hierarchy (for export/viewer)."""

    def _plane_payload(plane: LeafPlaneDescriptor | None) -> dict[str, Any] | None:
        if plane is None:
            return None
        return {
            "centroid": plane.centroid.detach().cpu().tolist(),
            "normal": plane.normal.detach().cpu().tolist(),
            "tangent_u": plane.tangent_u.detach().cpu().tolist(),
            "tangent_v": plane.tangent_v.detach().cpu().tolist(),
            "singular_values": plane.singular_values.detach().cpu().tolist(),
            "local_plane_rms": plane.local_plane_rms,
            "local_plane_max_error": plane.local_plane_max_error,
            "thickness_ratio": plane.thickness_ratio,
            "eigenvalue_ratio": plane.eigenvalue_ratio,
            "degenerate": plane.degenerate,
        }

    nodes_payload = []
    for node in hierarchy.nodes:
        entry: dict[str, Any] = {
            "node_id": node.node_id,
            "parent": hierarchy.nodes[node.parent].node_id if node.parent >= 0 else None,
            "depth": node.depth,
            "state": node.state,
            "count": node.count,
            "aabb_min": node.aabb_min.detach().cpu().tolist(),
            "aabb_max": node.aabb_max.detach().cpu().tolist(),
            "children": [hierarchy.nodes[child].node_id for child in node.children],
            "subdivision_reason": node.subdivision_reason,
            "subdivision_blocked": node.subdivision_blocked,
            "local_plane": _plane_payload(node.plane),
        }
        if include_gaussian_indices and node.gaussian_indices is not None:
            entry["gaussian_indices"] = node.gaussian_indices.detach().cpu().tolist()
        nodes_payload.append(entry)
    return {
        "point_count": hierarchy.point_count,
        "config": dict(hierarchy.config),
        "state_counts": hierarchy.state_counts(),
        "max_depth_reached": hierarchy.max_depth_reached(),
        "nodes": nodes_payload,
    }


FACE_INTERIOR = "interior"
FACE_EXTERIOR = "exterior_support"
FACE_UNRESOLVED = "unresolved"

# Face index convention: axis * 2 + side, side 0 = -axis face, 1 = +axis face.
FACE_NAMES = ("-x", "+x", "-y", "+y", "-z", "+z")


def compute_leaf_face_adjacency(
    hierarchy: TorchVoxelGaussianHierarchy,
    fit_complex_leaves: bool = True,
    degenerate_axis_tolerant: bool = False,
) -> dict[str, dict[str, Any]]:
    """Face adjacency and Stage 1-F classification for every leaf voxel.

    For each of a leaf's 6 faces, records which leaf(s) touch it and classifies
    each contact:

    - ``interior``: the neighbor is ACTIVE, or COMPLEX while complex leaves are
      fitted — the surface continues across this face, so it is NOT a support
      boundary.
    - ``exterior_support``: the neighbor is INACTIVE or EMPTY, or the face lies
      on the hierarchy boundary (no neighbor) — support may genuinely end here.
    - ``unresolved``: the neighbor is COMPLEX but complex leaves are not fitted.

    A leaf is a boundary leaf iff it has at least one exterior/unresolved
    contact. The leaf partition covers the root AABB exactly, so every interior
    face is fully covered by neighbor leaves; faces on the root boundary get an
    explicit ``neighbor_id: None`` contact.

    ``degenerate_axis_tolerant`` (default ``False``, preserving the exact
    behavior every existing Stage 1 / Stage 1-F caller regression-locks on):
    the touching-face overlap test on the two non-face axes normally requires
    the overlap width to *exceed* ``eps``. For a perfectly (or near-)flat
    scene, a voxel's extent along the flat axis is itself near-zero, so a
    same-axis overlap check with a fixed global ``eps`` can spuriously reject
    two leaves that are fully touching along that thin axis, silently
    dropping the x/y face contact between them. Passing ``True`` (used by the
    Phase 1 surface-component builder, see ``torch_surface_components.py``)
    relaxes the overlap test to ``> -eps`` — treating a merely non-negative
    (including near-zero) overlap as touching — while a genuine gap still
    exceeds ``eps`` and correctly stays disjoint.
    """

    leaves = hierarchy.leaves()
    if not leaves:
        return {}
    root = hierarchy.nodes[0]
    extent = float((root.aabb_max - root.aabb_min).max())
    eps = max(extent, 1e-6) * 1e-5
    min_overlap = -eps if degenerate_axis_tolerant else eps

    def _classify(neighbor_state: str | None) -> str:
        if neighbor_state == STATE_ACTIVE:
            return FACE_INTERIOR
        if neighbor_state == STATE_COMPLEX:
            return FACE_INTERIOR if fit_complex_leaves else FACE_UNRESOLVED
        return FACE_EXTERIOR

    result: dict[str, dict[str, Any]] = {
        leaf.node_id: {"contacts": [], "is_boundary_leaf": False, "boundary_faces": []}
        for leaf in leaves
    }

    def _add_contact(leaf: VoxelNode, face: int, neighbor: VoxelNode | None) -> None:
        neighbor_state = neighbor.state if neighbor is not None else None
        classification = _classify(neighbor_state)
        entry = result[leaf.node_id]
        entry["contacts"].append(
            {
                "face": face,
                "face_name": FACE_NAMES[face],
                "neighbor_id": neighbor.node_id if neighbor is not None else None,
                "neighbor_state": neighbor_state if neighbor is not None else "outside",
                "classification": classification,
            }
        )
        if classification in (FACE_EXTERIOR, FACE_UNRESOLVED):
            entry["is_boundary_leaf"] = True
            if face not in entry["boundary_faces"]:
                entry["boundary_faces"].append(face)

    bounds = [
        (leaf, leaf.aabb_min.detach().cpu().tolist(), leaf.aabb_max.detach().cpu().tolist())
        for leaf in leaves
    ]
    root_lo = root.aabb_min.detach().cpu().tolist()
    root_hi = root.aabb_max.detach().cpu().tolist()
    for index, (leaf, lo, hi) in enumerate(bounds):
        for axis in range(3):
            if abs(lo[axis] - root_lo[axis]) <= eps:
                _add_contact(leaf, axis * 2, None)
            if abs(hi[axis] - root_hi[axis]) <= eps:
                _add_contact(leaf, axis * 2 + 1, None)
        for other, other_lo, other_hi in bounds[index + 1 :]:
            for axis in range(3):
                other_axes = [a for a in range(3) if a != axis]
                if abs(hi[axis] - other_lo[axis]) <= eps:
                    plus_face_leaf, minus_face_leaf = leaf, other
                elif abs(other_hi[axis] - lo[axis]) <= eps:
                    plus_face_leaf, minus_face_leaf = other, leaf
                else:
                    continue
                overlap = True
                for a in other_axes:
                    # Adaptive per-axis tolerance: only relax the overlap
                    # requirement on an axis where BOTH boxes are themselves
                    # near-zero-width there (a genuinely flat/degenerate
                    # axis, e.g. z on a perfectly flat scene). When either box
                    # has real extent on this axis, a zero-width intersection
                    # means the boxes only touch at a boundary plane (e.g. two
                    # z-half-space octree branches meeting at z=0) and must
                    # NOT count as overlapping, regardless of the tolerant
                    # flag -- otherwise leaves from unrelated branches that
                    # happen to share a boundary point would be spuriously
                    # linked as face-adjacent.
                    own_extent = min(hi[a] - lo[a], other_hi[a] - other_lo[a])
                    axis_min_overlap = (
                        min_overlap if degenerate_axis_tolerant and own_extent <= eps else eps
                    )
                    if min(hi[a], other_hi[a]) - max(lo[a], other_lo[a]) <= axis_min_overlap:
                        overlap = False
                        break
                if overlap:
                    _add_contact(plus_face_leaf, axis * 2 + 1, minus_face_leaf)
                    _add_contact(minus_face_leaf, axis * 2, plus_face_leaf)
    return result


def plane_aabb_intersection_polygon(
    centroid: Any, normal: Any, aabb_min: Any, aabb_max: Any, eps: float = 1e-9
) -> Any:
    """Ordered convex polygon (K, 3) where the plane slices the AABB.

    Vertices are the intersections of the plane ``n . (x - c) = 0`` with the 12
    AABB edges, deduplicated and ordered counter-clockwise around the polygon
    centroid (in the plane's own tangent frame). Returns an empty ``(0, 3)``
    tensor when the plane misses the box.
    """

    torch = require_torch()
    device, dtype = centroid.device, centroid.dtype
    lo, hi = aabb_min.to(dtype), aabb_max.to(dtype)
    corners = torch.stack(
        [
            torch.stack([lo[0] if not (i & 4) else hi[0],
                         lo[1] if not (i & 2) else hi[1],
                         lo[2] if not (i & 1) else hi[2]])
            for i in range(8)
        ]
    ).to(device)
    # Edges as corner index pairs (each differs in exactly one bit).
    edge_pairs = [
        (a, b)
        for a in range(8)
        for b in range(a + 1, 8)
        if bin(a ^ b).count("1") == 1
    ]
    distances = (corners - centroid) @ normal
    vertices = []
    for a, b in edge_pairs:
        da, db = float(distances[a]), float(distances[b])
        if abs(da) <= eps:
            vertices.append(corners[a])
        if abs(db) <= eps:
            vertices.append(corners[b])
        if (da > eps and db < -eps) or (da < -eps and db > eps):
            t = da / (da - db)
            vertices.append(corners[a] + (corners[b] - corners[a]) * t)
    if not vertices:
        return torch.empty((0, 3), dtype=dtype, device=device)
    stacked = torch.stack(vertices)
    # Deduplicate near-identical vertices (corners touched by several edges).
    scale = float((hi - lo).max().clamp_min(1e-12))
    keep: list[Any] = []
    for vertex in stacked:
        if all(float((vertex - other).norm()) > 1e-6 * scale for other in keep):
            keep.append(vertex)
    if len(keep) < 3:
        return torch.empty((0, 3), dtype=dtype, device=device)
    polygon = torch.stack(keep)
    # Order counter-clockwise in the plane's tangent frame.
    reference = polygon.mean(dim=0)
    axis_u = _any_orthogonal(normal)
    axis_v = torch.cross(normal, axis_u, dim=0)
    rel = polygon - reference
    angles = torch.atan2(rel @ axis_v, rel @ axis_u)
    return polygon[torch.argsort(angles)]


def _any_orthogonal(axis: Any) -> Any:
    torch = require_torch()
    reference = torch.tensor([0.0, 0.0, 1.0], dtype=axis.dtype, device=axis.device)
    candidate = torch.cross(axis, reference, dim=0)
    if float(torch.linalg.norm(candidate)) < 1e-5:
        reference = torch.tensor([0.0, 1.0, 0.0], dtype=axis.dtype, device=axis.device)
        candidate = torch.cross(axis, reference, dim=0)
    return torch.nn.functional.normalize(candidate, dim=0)


def rasterize_convex_polygon_uv(polygon_uv: Any, resolution: int) -> Any:
    """Boolean ``(resolution, resolution)`` mask of cells whose center lies inside.

    ``polygon_uv`` is an ordered convex polygon in the patch UV domain (values
    may extend beyond ``[0, 1]``; the raster simply clips). Cell layout matches
    ``TorchNURBSSurface.support()``: ``mask[floor(u * R), floor(v * R)]``.
    """

    torch = require_torch()
    resolution = max(1, int(resolution))
    device = polygon_uv.device if hasattr(polygon_uv, "device") else None
    mask = torch.zeros((resolution, resolution), dtype=torch.bool, device=device)
    if polygon_uv is None or int(polygon_uv.shape[0]) < 3:
        return mask
    centers = (torch.arange(resolution, dtype=polygon_uv.dtype, device=device) + 0.5) / resolution
    grid_u, grid_v = torch.meshgrid(centers, centers, indexing="ij")
    samples = torch.stack([grid_u.reshape(-1), grid_v.reshape(-1)], dim=1)
    edges_a = polygon_uv
    edges_b = torch.roll(polygon_uv, shifts=-1, dims=0)
    direction = edges_b - edges_a  # (K, 2)
    to_sample = samples[:, None, :] - edges_a[None, :, :]  # (Q, K, 2)
    cross = direction[None, :, 0] * to_sample[:, :, 1] - direction[None, :, 1] * to_sample[:, :, 0]
    # The polygon may be ordered CW or CCW in UV; accept either winding.
    tol = 1e-9
    inside = (cross >= -tol).all(dim=1) | (cross <= tol).all(dim=1)
    return inside.reshape(resolution, resolution)
