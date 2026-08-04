from __future__ import annotations

"""Explicit simple-polygon validation for an ordered closed source loop
(worklog 36 task section 9).

Total turning angle (+-2*pi) and planar z-standard-deviation alone cannot
prove a loop is a simple (non-self-intersecting) polygon -- a figure-eight
also has zero net turning under certain windings, and a bow-tie can pass a
loose planarity check. This module projects the ordered loop onto its own
best-fit local tangent plane and directly tests every non-adjacent segment
pair for proper crossing, collinear overlap, and repeated-vertex degeneracy.
"""

from dataclasses import dataclass
from typing import Any, Sequence

_EPS = 1e-9


PLANAR_ENOUGH = "planar_enough"
MILDLY_CURVED_CHART = "mildly_curved_chart"
NONPLANAR_AMBIGUOUS = "nonplanar_ambiguous"

# Worklog 37 (task section 3): thresholds for the PCA-projection precondition.
# The 2D segment-crossing test is only a valid proxy for 3D self-intersection
# when the loop is close enough to planar that the projection cannot itself
# create or hide a crossing. Conservative, not scene-tuned: box_face (a truly
# flat face) and a clean cylinder cap measure orders of magnitude inside
# these bounds; only a loop that genuinely wraps a curved patch (e.g. a
# cylinder SIDE boundary, which spans real curvature around its axis) would
# approach or exceed them.
_THICKNESS_TO_EXTENT_RATIO_LIMIT = 0.25
_MAX_POINT_TO_PLANE_DISTANCE_TO_EXTENT_RATIO_LIMIT = 0.15
_P90_POINT_TO_PLANE_DISTANCE_TO_EXTENT_RATIO_LIMIT = 0.08


@dataclass(frozen=True)
class PlanarityReport:
    pca_eigenvalues: tuple[float, float, float]  # ascending: smallest (normal) first
    normal_direction_thickness: float  # sqrt of smallest eigenvalue
    tangent_extent: float  # sqrt of largest eigenvalue
    thickness_to_tangent_extent_ratio: float
    max_point_to_plane_distance: float
    p90_point_to_plane_distance: float
    normal_dispersion: float
    planarity_class: str  # PLANAR_ENOUGH | MILDLY_CURVED_CHART | NONPLANAR_AMBIGUOUS


@dataclass(frozen=True)
class SelfIntersectionReport:
    proper_intersection_count: int
    endpoint_touch_count: int
    collinear_overlap_count: int
    near_intersection_min_distance: float
    winding_number: float
    signed_area: float
    orientation: str  # "counterclockwise" | "clockwise" | "degenerate"
    total_turning_angle: float
    repeated_vertex_count: int
    zero_area: bool
    is_simple_polygon: bool
    planarity: PlanarityReport | None
    reasons: tuple[str, ...]


def _pca_basis(points: Sequence[tuple[float, float, float]]):
    """Centroid, orthonormal (axis_u, axis_v, axis_normal) basis (largest to
    smallest variance), and the three eigenvalues (ascending, normal first)
    of the point covariance -- via power iteration, no numpy/torch
    dependency for this small CPU-only geometry utility."""
    n = len(points)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    cz = sum(p[2] for p in points) / n
    centered = [(p[0] - cx, p[1] - cy, p[2] - cz) for p in points]

    cov = [[0.0] * 3 for _ in range(3)]
    for x, y, z in centered:
        v = (x, y, z)
        for i in range(3):
            for j in range(3):
                cov[i][j] += v[i] * v[j]
    for i in range(3):
        for j in range(3):
            cov[i][j] /= n

    def matvec(vec):
        return tuple(sum(cov[i][j] * vec[j] for j in range(3)) for i in range(3))

    def normalize(vec):
        norm = sum(c * c for c in vec) ** 0.5
        if norm < _EPS:
            return (1.0, 0.0, 0.0)
        return tuple(c / norm for c in vec)

    def orthogonal_to(vec, basis):
        dot = sum(a * b for a, b in zip(vec, basis))
        return tuple(a - dot * b for a, b in zip(vec, basis))

    def rayleigh(vec):
        mv = matvec(vec)
        return sum(a * b for a, b in zip(vec, mv))

    axis_u = normalize((1.0, 0.3, 0.2))
    for _ in range(50):
        axis_u = normalize(matvec(axis_u))
    lambda_u = rayleigh(axis_u)

    axis_v0 = normalize((0.2, 1.0, 0.3))
    for _ in range(50):
        axis_v0 = normalize(matvec(orthogonal_to(axis_v0, axis_u)))
    axis_v = normalize(orthogonal_to(axis_v0, axis_u))
    lambda_v = rayleigh(axis_v)

    axis_normal = normalize((axis_u[1] * axis_v[2] - axis_u[2] * axis_v[1],
                              axis_u[2] * axis_v[0] - axis_u[0] * axis_v[2],
                              axis_u[0] * axis_v[1] - axis_u[1] * axis_v[0]))
    lambda_normal = rayleigh(axis_normal)

    return centered, axis_u, axis_v, axis_normal, (max(lambda_normal, 0.0), max(lambda_v, 0.0), max(lambda_u, 0.0))


def _project_to_local_plane(points: Sequence[tuple[float, float, float]]) -> list[tuple[float, float]]:
    """Project 3D loop points onto the best-fit plane through their centroid
    (tangent-plane coordinates only; see `_compute_planarity` for the
    eigenvalue/thickness diagnostics used to gate whether this projection is
    even a valid proxy for 3D self-intersection)."""
    centered, axis_u, axis_v, _axis_normal, _eigenvalues = _pca_basis(points)
    return [
        (sum(a * b for a, b in zip(p, axis_u)), sum(a * b for a, b in zip(p, axis_v)))
        for p in centered
    ]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[idx]


def compute_planarity(points: Sequence[tuple[float, float, float]]) -> PlanarityReport:
    """Worklog 37 (task section 3): precondition contract for the PCA-plane
    self-intersection projection. The 2D segment-crossing test is only a
    valid proxy for 3D self-intersection when the loop is close enough to
    planar that flattening it cannot itself create or hide a crossing
    (e.g. a cylinder SIDE boundary genuinely wraps around real curvature --
    projecting it to a flat chart can fold two far-apart 3D arcs onto
    overlapping 2D segments that were never actually close in 3D)."""
    n = len(points)
    if n < 3:
        return PlanarityReport((0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, NONPLANAR_AMBIGUOUS)

    centered, axis_u, axis_v, axis_normal, eigenvalues = _pca_basis(points)
    lambda_normal, lambda_v, lambda_u = eigenvalues
    normal_thickness = lambda_normal ** 0.5
    tangent_extent = lambda_u ** 0.5
    thickness_ratio = normal_thickness / max(tangent_extent, _EPS)

    plane_distances = [abs(sum(a * b for a, b in zip(p, axis_normal))) for p in centered]
    max_distance = max(plane_distances) if plane_distances else 0.0
    p90_distance = _percentile(plane_distances, 0.9)
    max_distance_ratio = max_distance / max(tangent_extent, _EPS)
    p90_distance_ratio = p90_distance / max(tangent_extent, _EPS)

    # Normal dispersion is expressed as the (extent-normalized) standard
    # deviation of point-to-plane distance (0 = perfectly flat, larger =
    # more scattered) -- simpler and more directly tied to the planarity
    # decision than estimating per-point local normals from a sparse
    # boundary polyline alone.
    mean_distance = sum(plane_distances) / n
    variance = sum((d - mean_distance) ** 2 for d in plane_distances) / n
    normal_dispersion = (variance ** 0.5) / max(tangent_extent, _EPS)

    if (
        thickness_ratio <= _THICKNESS_TO_EXTENT_RATIO_LIMIT
        and max_distance_ratio <= _MAX_POINT_TO_PLANE_DISTANCE_TO_EXTENT_RATIO_LIMIT
        and p90_distance_ratio <= _P90_POINT_TO_PLANE_DISTANCE_TO_EXTENT_RATIO_LIMIT
    ):
        planarity_class = PLANAR_ENOUGH
    elif thickness_ratio <= 2 * _THICKNESS_TO_EXTENT_RATIO_LIMIT and max_distance_ratio <= 2 * _MAX_POINT_TO_PLANE_DISTANCE_TO_EXTENT_RATIO_LIMIT:
        planarity_class = MILDLY_CURVED_CHART
    else:
        planarity_class = NONPLANAR_AMBIGUOUS

    return PlanarityReport(
        pca_eigenvalues=(lambda_normal, lambda_v, lambda_u),
        normal_direction_thickness=normal_thickness,
        tangent_extent=tangent_extent,
        thickness_to_tangent_extent_ratio=thickness_ratio,
        max_point_to_plane_distance=max_distance,
        p90_point_to_plane_distance=p90_distance,
        normal_dispersion=normal_dispersion,
        planarity_class=planarity_class,
    )


def _segments_intersect(a1, a2, b1, b2):
    """Returns ('proper', point) | ('endpoint_touch', point) |
    ('collinear_overlap', None) | (None, None)."""

    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    d1 = cross(b1, b2, a1)
    d2 = cross(b1, b2, a2)
    d3 = cross(a1, a2, b1)
    d4 = cross(a1, a2, b2)

    if ((d1 > _EPS and d2 < -_EPS) or (d1 < -_EPS and d2 > _EPS)) and \
       ((d3 > _EPS and d4 < -_EPS) or (d3 < -_EPS and d4 > _EPS)):
        # Proper crossing: solve for intersection point.
        denom = (a2[0] - a1[0]) * (b2[1] - b1[1]) - (a2[1] - a1[1]) * (b2[0] - b1[0])
        if abs(denom) < _EPS:
            return None, None
        t = ((b1[0] - a1[0]) * (b2[1] - b1[1]) - (b1[1] - a1[1]) * (b2[0] - b1[0])) / denom
        point = (a1[0] + t * (a2[0] - a1[0]), a1[1] + t * (a2[1] - a1[1]))
        return "proper", point

    def on_segment(p, q, r):
        return min(p[0], r[0]) - _EPS <= q[0] <= max(p[0], r[0]) + _EPS and \
               min(p[1], r[1]) - _EPS <= q[1] <= max(p[1], r[1]) + _EPS

    if abs(d1) < _EPS and on_segment(b1, a1, b2):
        return "endpoint_touch", a1
    if abs(d2) < _EPS and on_segment(b1, a2, b2):
        return "endpoint_touch", a2
    if abs(d3) < _EPS and on_segment(a1, b1, a2):
        return "endpoint_touch", b1
    if abs(d4) < _EPS and on_segment(a1, b2, a2):
        return "endpoint_touch", b2

    if abs(d1) < _EPS and abs(d2) < _EPS:
        # a1,a2 collinear with b1,b2 -- check for genuine overlap (not just
        # sharing an endpoint, already handled above).
        def param(p, origin, axis):
            return (p[0] - origin[0]) * axis[0] + (p[1] - origin[1]) * axis[1]
        axis = (b2[0] - b1[0], b2[1] - b1[1])
        norm = (axis[0] ** 2 + axis[1] ** 2) ** 0.5
        if norm < _EPS:
            return None, None
        axis = (axis[0] / norm, axis[1] / norm)
        ta1, ta2 = param(a1, b1, axis), param(a2, b1, axis)
        tb1, tb2 = 0.0, norm
        lo_a, hi_a = min(ta1, ta2), max(ta1, ta2)
        lo_b, hi_b = min(tb1, tb2), max(tb1, tb2)
        overlap = min(hi_a, hi_b) - max(lo_a, lo_b)
        if overlap > _EPS:
            return "collinear_overlap", None

    return None, None


def _point_distance(p, q):
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def validate_simple_closed_loop(world_points: Sequence[tuple[float, float, float]]) -> SelfIntersectionReport:
    """Direct segment-crossing validation of an ordered closed loop.

    ``world_points`` must already be in the loop's traversal order (index i
    connects to index i+1, and the last connects back to the first).
    """
    n = len(world_points)
    reasons: list[str] = []
    if n < 3:
        return SelfIntersectionReport(0, 0, 0, 0.0, 0.0, 0.0, "degenerate", 0.0, 0, True, False, None, ("fewer_than_three_vertices",))

    planarity = compute_planarity(world_points)
    if planarity.planarity_class == NONPLANAR_AMBIGUOUS:
        # Worklog 37 (task section 3): the PCA-plane projection is not a
        # valid proxy for 3D self-intersection once the loop's own
        # curvature is large relative to its tangent extent -- flattening
        # such a loop can fold two genuinely-distant 3D arcs onto
        # overlapping 2D segments (a false positive) or unfold a real
        # crossing into two segments that no longer cross in 2D (a false
        # negative). Fail closed/review rather than approve a nonplanar
        # loop on the strength of a projection whose topology-preservation
        # is not guaranteed.
        return SelfIntersectionReport(
            0, 0, 0, -1.0, 0.0, 0.0, "degenerate", 0.0, 0, False, False, planarity,
            ("self_intersection_not_checked_nonplanar",),
        )

    planar = _project_to_local_plane(world_points)

    repeated_vertex_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if _point_distance(planar[i], planar[j]) < 1e-7:
                repeated_vertex_count += 1
    if repeated_vertex_count:
        reasons.append("repeated_nonconsecutive_vertex")

    segments = [(planar[i], planar[(i + 1) % n]) for i in range(n)]

    proper_count = 0
    touch_count = 0
    collinear_count = 0
    near_min = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            # Adjacent segments (share exactly one endpoint by construction)
            # are expected to "touch" there -- not a violation.
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            kind, _point = _segments_intersect(*segments[i], *segments[j])
            if kind == "proper":
                proper_count += 1
            elif kind == "endpoint_touch":
                touch_count += 1
            elif kind == "collinear_overlap":
                collinear_count += 1
            a1, a2 = segments[i]
            b1, b2 = segments[j]
            for p in (a1, a2):
                for q in (b1, b2):
                    d = _point_distance(p, q)
                    if d < near_min:
                        near_min = d

    signed_area = 0.5 * sum(
        planar[i][0] * planar[(i + 1) % n][1] - planar[(i + 1) % n][0] * planar[i][1]
        for i in range(n)
    )
    zero_area = abs(signed_area) < 1e-10
    if signed_area > 0:
        orientation = "counterclockwise"
    elif signed_area < 0:
        orientation = "clockwise"
    else:
        orientation = "degenerate"

    total_turning = 0.0
    for i in range(n):
        prev_p, cur_p, next_p = planar[(i - 1) % n], planar[i], planar[(i + 1) % n]
        v1 = (cur_p[0] - prev_p[0], cur_p[1] - prev_p[1])
        v2 = (next_p[0] - cur_p[0], next_p[1] - cur_p[1])
        n1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
        n2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5
        if n1 < _EPS or n2 < _EPS:
            continue
        cos_a = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        sin_a = (v1[0] * v2[1] - v1[1] * v2[0]) / (n1 * n2)
        import math
        angle = math.atan2(sin_a, cos_a)
        total_turning += angle

    # Winding number of the polygon around its own centroid (a simple,
    # correctly-oriented loop with no self-intersection has winding
    # number exactly +-1).
    cx = sum(p[0] for p in planar) / n
    cy = sum(p[1] for p in planar) / n
    winding_accum = 0.0
    import math
    for i in range(n):
        p, q = planar[i], planar[(i + 1) % n]
        a1 = math.atan2(p[1] - cy, p[0] - cx)
        a2 = math.atan2(q[1] - cy, q[0] - cx)
        d = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        winding_accum += d
    winding_number = winding_accum / (2 * math.pi)

    # Orientation consistency is judged from `total_turning_angle` (the sum
    # of exterior angles at each vertex), not the centroid-based winding
    # number above: for a loop with a vertex very close to the centroid-ray
    # from another vertex, `winding_number`'s per-edge atan2 delta can
    # become numerically unstable even for a genuinely simple polygon (the
    # turning-angle sum has no such degeneracy since it only ever compares
    # ADJACENT edge directions, never a ray to a potentially-nearby
    # centroid). `winding_number` is still reported for diagnostics.
    turning_periods = total_turning / (2 * 3.141592653589793)
    if proper_count:
        reasons.append("proper_self_intersection")
    if collinear_count:
        reasons.append("non_adjacent_collinear_overlap")
    if zero_area:
        reasons.append("zero_area_cycle")
    if abs(abs(turning_periods) - 1.0) > 0.05 and not zero_area:
        reasons.append("orientation_inconsistency")

    is_simple = (
        proper_count == 0
        and collinear_count == 0
        and repeated_vertex_count == 0
        and not zero_area
        and abs(abs(turning_periods) - 1.0) <= 0.05
    )

    return SelfIntersectionReport(
        proper_intersection_count=proper_count,
        endpoint_touch_count=touch_count,
        collinear_overlap_count=collinear_count,
        near_intersection_min_distance=near_min if near_min != float("inf") else -1.0,
        winding_number=winding_number,
        signed_area=signed_area,
        orientation=orientation,
        total_turning_angle=total_turning,
        repeated_vertex_count=repeated_vertex_count,
        zero_area=zero_area,
        is_simple_polygon=is_simple,
        planarity=planarity,
        reasons=tuple(reasons),
    )
