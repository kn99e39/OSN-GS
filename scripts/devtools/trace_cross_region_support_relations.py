"""Worklog 40 (task section 2/3): authoritative trace of every
`observed_support_termination` candidate whose outward arc contains
out-of-region observed support, with the AFFINITY RELATION EVIDENCE for that
support.

The question this answers: when a candidate's "support-free" direction is in
fact occupied by Gaussians belonging to a different region, does the
manifold affinity graph say those Gaussians are the SAME smooth surface
(so the candidate is a nonphysical region frontier), or a crease/parallel/
competing surface (so the candidate is a genuine physical termination)?

Diagnostic only -- reads production results, changes nothing.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.surface.torch_canonical_region_tangent_frame import construct_canonical_region_tangent_frames
from osn_gs.surface.torch_gaussian_manifold_affinity import (
    CANDIDATE_STATUS_CANDIDATE,
    RELATION_CREASE,
    RELATION_PARALLEL_SEPARATE,
    RELATION_REJECTED,
    RELATION_SAME_SURFACE,
)
from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians


def _unit(v):
    return v / v.norm().clamp_min(1e-12)


def trace(scene_name: str, seed: int = 0) -> dict:
    scene = make_gaussian_reliability_scene(scene_name, seed=seed)
    ids = tuple(range(scene.positions.shape[0]))
    result = construct_visible_nurbs_from_gaussians(
        scene.positions, covariance=scene.covariances, stable_ids=ids,
    )
    canonical = construct_canonical_region_tangent_frames(
        scene.positions, result.covariance_frame, result.reliability,
        result.surface_regions, ids=ids,
    )
    regions = result.surface_regions
    index = {sid: i for i, sid in enumerate(ids)}

    # Relation lookup from the affinity graph (bounded-kNN candidate edges).
    relation_by_pair: dict[frozenset, tuple[str, float]] = {}
    for edge in result.manifold_affinity.edges:
        if edge.candidate_status != CANDIDATE_STATUS_CANDIDATE:
            continue
        relation_by_pair[frozenset((edge.source, edge.target))] = (
            edge.manifold_relation, str(getattr(edge, "relation_confidence", "")),
        )

    tangent_scales = result.covariance_frame.equivalent_tangent_scale
    rows = []
    summary = Counter()

    for candidate in result.boundary_halfedge_candidates:
        if candidate.boundary_reason != "observed_support_termination":
            continue
        source = index[candidate.source_gaussian_id]
        region_id = candidate.source_region_id
        frame = canonical[source]
        if frame is None:
            continue
        normal = frame.oriented_normal
        outward = torch.tensor(candidate.local_normal, dtype=scene.positions.dtype)
        # Reconstruct the outward direction: boundary_direction = normal x outward,
        # so outward = boundary_direction x normal.
        boundary_direction = torch.tensor(candidate.boundary_direction, dtype=scene.positions.dtype)
        outward = _unit(torch.linalg.cross(boundary_direction, normal))

        radius = float(tangent_scales[source]) * 4.0
        same_region_arc = 0
        other_region_arc = 0
        other_relations = Counter()
        other_region_ids = set()
        for target in range(len(ids)):
            if target == source:
                continue
            delta = scene.positions[target] - scene.positions[source]
            tangent = delta - normal * (delta @ normal)
            distance = float(tangent.norm())
            if not (1e-8 < distance <= radius):
                continue
            # Inside the outward half-arc we declared support-free?
            if float(_unit(tangent) @ outward) < 0.0:
                continue
            target_region = regions.node_region_id[target]
            if target_region == region_id:
                same_region_arc += 1
                continue
            if target_region < 0:
                continue
            other_region_arc += 1
            other_region_ids.add(int(target_region))
            relation = relation_by_pair.get(frozenset((source, target)))
            other_relations[relation[0] if relation else "no_candidate_edge"] += 1

        # Classify the cross-region support by the affinity graph's own verdict.
        if other_region_arc == 0:
            classification = "no_cross_region_support"
        elif other_relations.get(RELATION_CREASE, 0) > 0:
            classification = "crease_adjacent"
        elif other_relations.get(RELATION_PARALLEL_SEPARATE, 0) > 0:
            classification = "parallel_separate_neighbor"
        elif other_relations.get(RELATION_SAME_SURFACE, 0) > 0:
            classification = "smooth_cross_region_continuation"
        elif other_relations.get(RELATION_REJECTED, 0) > 0:
            classification = "competing_surface_neighbor"
        else:
            classification = "cross_region_support_without_relation_evidence"

        summary[classification] += 1
        rows.append({
            "candidate_stable_id": candidate.source_gaussian_id,
            "region_id": int(region_id),
            "same_region_support_in_arc": same_region_arc,
            "out_of_region_support_in_arc": other_region_arc,
            "other_region_ids": sorted(other_region_ids),
            "cross_region_relations": dict(other_relations),
            "classification": classification,
        })

    return {
        "scene": scene_name,
        "genuine_candidate_count": len(rows),
        "classification_counts": dict(summary),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="*", default=["sphere", "box", "cylinder", "thin_slab", "box_face"])
    parser.add_argument("--detail", action="store_true")
    args = parser.parse_args()
    for scene_name in args.scenes:
        report = trace(scene_name)
        if not args.detail:
            report.pop("rows")
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
