"""Worklog 36 (task section 10): box_face analytic boundary accuracy,
separating three distinct error sources:

1. Candidate positions themselves (do genuine termination candidates sit
   near the true analytic square boundary at all?)
2. Ordering (does the RECOVERED source polyline, after cycle-recovery, add
   error beyond what the candidate positions already had?)
3. NURBS fitting/evaluation (does the fitted surface's evaluated boundary
   add error beyond the ordered source polyline?)

box_face is a 9x9 flat grid (`_flat_grid(9, 0.12)`, normal=(0,0,1),
origin=(0,0,0)): analytic half-extent = (9-1)/2 * 0.12 = 0.48, a square
boundary at x=+-0.48 or y=+-0.48 (z=0 plane).
"""

from __future__ import annotations

import json

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians

_HALF_EXTENT = 0.48


def _distance_to_square_boundary(x: float, y: float) -> float:
    """Nearest-edge distance to the square boundary |x|<=0.48, |y|<=0.48
    (a point exactly ON an edge has distance 0; a point in the interior or
    exterior has positive distance to the nearest edge segment)."""
    # Clamp to the square, then distance to that clamped point IS the
    # distance to the boundary only if the point is outside; for interior
    # points, distance to nearest edge = half_extent - max(|x|,|y|) type calc.
    ax, ay = abs(x), abs(y)
    if ax <= _HALF_EXTENT and ay <= _HALF_EXTENT:
        # Interior: distance to nearest edge.
        return min(_HALF_EXTENT - ax, _HALF_EXTENT - ay)
    # Exterior/edge: standard point-to-box-boundary distance.
    dx = max(ax - _HALF_EXTENT, 0.0)
    dy = max(ay - _HALF_EXTENT, 0.0)
    return (dx * dx + dy * dy) ** 0.5


def _percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[idx]


def analyze(cap: int) -> dict:
    scene = make_gaussian_reliability_scene("box_face", seed=0)
    stable_ids = tuple(range(scene.positions.shape[0]))
    if cap and cap < scene.positions.shape[0]:
        opacity = torch.ones(scene.positions.shape[0])
        config = TorchPipelineConfig(canonical_construction_max_points=cap)
        pipeline = TorchOSNGSPipeline(config, device="cpu")
        bundle = pipeline._construct_canonical_with_full_evidence(scene.positions, scene.covariances, opacity, stable_ids)
        construction = bundle.construction
        id_to_pos = {sid: tuple(scene.positions[i].tolist()) for i, sid in enumerate(stable_ids)}
    else:
        construction = construct_visible_nurbs_from_gaussians(scene.positions, covariance=scene.covariances, stable_ids=stable_ids)
        id_to_pos = {sid: tuple(scene.positions[i].tolist()) for i, sid in enumerate(stable_ids)}

    # 1. Candidate positions error (genuine termination candidates only).
    genuine = [h for h in construction.boundary_halfedge_candidates if h.boundary_reason == "observed_support_termination"]
    candidate_errors = [_distance_to_square_boundary(h.world_position[0], h.world_position[1]) for h in genuine]

    # 2. Ordered source polyline error (candidates that made it into a closed loop).
    closed = [c for c in construction.ordered_boundary_components if c.ordering_state == "ordered_closed_loop"]
    ordered_errors = []
    for component in closed:
        for sid in component.ordered_source_ids:
            if sid in id_to_pos:
                x, y, _z = id_to_pos[sid]
                ordered_errors.append(_distance_to_square_boundary(x, y))

    # 3. NURBS evaluated boundary error.
    evaluated_errors = []
    for attempt in construction.materialization_attempts:
        if attempt.state != "materialized" or attempt.surface is None:
            continue
        surface = attempt.surface
        grid = torch.linspace(0.0, 1.0, 33, dtype=torch.float32)
        # Evaluate along the four parametric edges (u=0, u=1, v=0, v=1) --
        # the NURBS "evaluated boundary" is the surface boundary curve, not
        # an interior sample.
        edges_uv = []
        for u in grid.tolist():
            edges_uv.append((u, 0.0))
            edges_uv.append((u, 1.0))
        for v in grid.tolist():
            edges_uv.append((0.0, v))
            edges_uv.append((1.0, v))
        uv = torch.tensor(edges_uv, dtype=torch.float32)
        points = surface.evaluate(uv)
        for row in points.detach().cpu().tolist():
            evaluated_errors.append(_distance_to_square_boundary(row[0], row[1]))

    def stats(values):
        return {
            "median": _percentile(values, 0.5),
            "p90": _percentile(values, 0.9),
            "p95": _percentile(values, 0.95),
            "max": max(values) if values else None,
            "count": len(values),
        }

    return {
        "cap": cap if cap else "no_downsample",
        "genuine_candidate_count": len(genuine),
        "closed_component_count": len(closed),
        "candidate_position_error": stats(candidate_errors),
        "ordered_source_polyline_error": stats(ordered_errors),
        "nurbs_evaluated_boundary_error": stats(evaluated_errors),
    }


def main():
    for cap in [0, 27]:
        result = analyze(cap)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
