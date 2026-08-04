"""Worklog 36 (task section 11): trace the residual 3-node open path and
isolated node on box_face(cap=27) -- per node, determine analytic edge label,
compatibility degree, forbidden gate, assignment state, and reason for
exclusion from the main cycle."""

from __future__ import annotations

import json

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_directed_boundary_ordering import _compatible_directed_edges, _max_weight_one_in_one_out_matching

_HALF_EXTENT = 0.48


def _edge_label(x: float, y: float) -> str:
    ax, ay = abs(x), abs(y)
    if ax >= ay:
        return "x_edge(+)" if x > 0 else "x_edge(-)"
    return "y_edge(+)" if y > 0 else "y_edge(-)"


def main():
    scene = make_gaussian_reliability_scene("box_face", seed=0)
    stable_ids = tuple(range(scene.positions.shape[0]))
    opacity = torch.ones(scene.positions.shape[0])
    config = TorchPipelineConfig(canonical_construction_max_points=27)
    pipeline = TorchOSNGSPipeline(config, device="cpu")
    bundle = pipeline._construct_canonical_with_full_evidence(scene.positions, scene.covariances, opacity, stable_ids)
    construction = bundle.construction

    halfedges = [h for h in construction.boundary_halfedge_candidates if h.boundary_reason == "observed_support_termination"]
    accepted = construction.accepted_local_topology
    accepted_pairs = {frozenset(p) for p in accepted}

    def _sub(a, b): return tuple(x - y for x, y in zip(a, b))
    def _norm(a): return max(sum(x * x for x in a) ** 0.5, 1e-12)
    nearest = []
    for s in halfedges:
        for t in halfedges:
            if t.half_edge_id != s.half_edge_id:
                nearest.append(_norm(_sub(s.world_position, t.world_position)))
    local_spacing = sorted(nearest)[len(nearest) // 2]
    edges = _compatible_directed_edges(halfedges, accepted_pairs, local_spacing)
    node_ids = sorted(h.half_edge_id for h in halfedges)
    matched = _max_weight_one_in_one_out_matching(node_ids, edges)

    by_id = {h.half_edge_id: h for h in halfedges}
    from collections import Counter
    out_deg = Counter()
    in_deg = Counter()
    for (s, t) in edges:
        out_deg[s] += 1
        in_deg[t] += 1

    # Identify which nodes ended up in the closed loop vs. the fragments.
    closed = [c for c in construction.ordered_boundary_components if c.ordering_state == "ordered_closed_loop"]
    fragments = [c for c in construction.ordered_boundary_components if c.ordering_state != "ordered_closed_loop"]
    closed_ids = set()
    for c in closed:
        closed_ids.update(c.ordered_half_edge_ids)

    print("=== fragments ===")
    for component in fragments:
        print(f"state={component.ordering_state} size={len(component.ordered_source_ids)} reasons={component.unresolved_reasons}")
        for hid in component.ordered_half_edge_ids:
            h = by_id[hid]
            x, y, z = h.world_position
            label = _edge_label(x, y)
            matched_out = matched.get(hid)
            matched_in = next((s for s, t in matched.items() if t == hid), None)
            print(json.dumps({
                "gaussian_id": h.source_gaussian_id,
                "position": (round(x, 3), round(y, 3), round(z, 3)),
                "analytic_edge_label": label,
                "compat_out_degree": out_deg.get(hid, 0),
                "compat_in_degree": in_deg.get(hid, 0),
                "matched_out": matched_out,
                "matched_in": matched_in,
            }))

    print("=== nodes in closed loop, for comparison (edge label distribution) ===")
    from collections import Counter as C2
    labels = C2()
    for hid in closed_ids:
        if hid in by_id:
            x, y, _z = by_id[hid].world_position
            labels[_edge_label(x, y)] += 1
    print(dict(labels))


if __name__ == "__main__":
    main()
