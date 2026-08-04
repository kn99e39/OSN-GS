"""Worklog 35: rigid-rotation/translation/uniform-scale invariance of the new
deterministic one-in/one-out directed ordering, on a frozen synthetic ring
candidate set (same style as the frozen-representative test in worklog 33,
applied here to boundary candidates instead of representatives)."""

from __future__ import annotations

import math

from osn_gs.surface.torch_directed_boundary_ordering import recover_directed_boundary_components
from osn_gs.surface.torch_world_space_boundary_halfedges import WorldSpaceBoundaryHalfEdgeCandidate


def _rotate(vector, axis, angle):
    ax, ay, az = axis
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    ax, ay, az = ax / norm, ay / norm, az / norm
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    x, y, z = vector
    dot = x * ax + y * ay + z * az
    cross = (ay * z - az * y, az * x - ax * z, ax * y - ay * x)
    return (
        x * cos_a + cross[0] * sin_a + ax * dot * (1 - cos_a),
        y * cos_a + cross[1] * sin_a + ay * dot * (1 - cos_a),
        z * cos_a + cross[2] * sin_a + az * dot * (1 - cos_a),
    )


def _transform(candidates, *, rotate_angle=0.0, axis=(0.3, 0.7, 0.5), translate=(0.0, 0.0, 0.0), scale=1.0):
    from dataclasses import replace
    output = []
    for c in candidates:
        pos = tuple(v * scale for v in _rotate(c.world_position, axis, rotate_angle))
        pos = tuple(p + t for p, t in zip(pos, translate))
        normal = _rotate(c.local_normal, axis, rotate_angle)
        tangent = _rotate(c.local_tangent_direction, axis, rotate_angle)
        boundary_dir = _rotate(c.boundary_direction, axis, rotate_angle)
        output.append(replace(c, world_position=pos, local_normal=normal, local_tangent_direction=tangent, boundary_direction=boundary_dir))
    return output


def _ring_candidates(n: int, radius: float = 1.0):
    candidates = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        tangent = (-math.sin(angle), math.cos(angle), 0.0)
        candidates.append(WorldSpaceBoundaryHalfEdgeCandidate(
            half_edge_id=f"h{i}", source_region_id=0, source_gaussian_id=i, adjacent_gaussian_id=None,
            world_position=(x, y, 0.0), local_normal=(0.0, 0.0, 1.0), local_tangent_direction=tangent,
            boundary_direction=tangent, boundary_reason="observed_support_termination", source_pair_ids=None,
            confidence=0.7, ordering_state="locally_chainable", review_reasons=(),
        ))
    return candidates


def main():
    n = 14
    base = _ring_candidates(n)
    accepted = [(i, (i + 1) % n) for i in range(n)]
    _, base_components = recover_directed_boundary_components(base, accepted)
    base_closed = sorted(tuple(sorted(c.ordered_source_ids)) for c in base_components if c.ordering_state == "ordered_closed_loop")

    for label, kwargs in [
        ("rotation", dict(rotate_angle=0.77)),
        ("translation", dict(translate=(5.0, -3.0, 2.0))),
        ("uniform_scale", dict(scale=3.5)),
        ("rotation+translation+scale", dict(rotate_angle=1.1, translate=(1.0, 2.0, -1.0), scale=0.4)),
    ]:
        transformed = _transform(base, **kwargs)
        _, components = recover_directed_boundary_components(transformed, accepted)
        closed = sorted(tuple(sorted(c.ordered_source_ids)) for c in components if c.ordering_state == "ordered_closed_loop")
        match = closed == base_closed
        print(f"{label}: base_closed={base_closed} transformed_closed={closed} exact_match={match}")
        assert match, f"{label} broke exact stable-ID closed-loop invariance"

    print("ALL INVARIANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
