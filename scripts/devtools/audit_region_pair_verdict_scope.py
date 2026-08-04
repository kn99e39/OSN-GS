"""Worklog 41 (task section 2): audit the SCOPE of the worklog 40
region-pair global continuation verdict.

Answers, by measurement rather than by reading intent:
  A. can one same_surface edge suppress every candidate on the pair?
  B. can a pair be smooth in one place and crease in another?
  C. does relation evidence apply at arbitrary distance from the candidate?
  D. is precedence decided independently of candidate-local geometry?
  E. can region-pair aggregation cause false suppression?

Diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.surface.torch_boundary_support_termination import classify_cross_region_pairs
from osn_gs.surface.torch_gaussian_manifold_affinity import (
    CANDIDATE_STATUS_CANDIDATE,
    RELATION_CREASE,
    RELATION_PARALLEL_SEPARATE,
    RELATION_SAME_SURFACE,
)
from osn_gs.surface.torch_visible_surface_construction import construct_visible_nurbs_from_gaussians


def audit(positions, covariances, label: str) -> dict:
    ids = tuple(range(positions.shape[0]))
    result = construct_visible_nurbs_from_gaussians(
        positions, covariance=covariances, stable_ids=ids,
    )
    regions = result.surface_regions
    verdicts = classify_cross_region_pairs(regions, result.manifold_affinity)

    # Per region pair: where do the relation edges actually live, and how
    # spatially spread are they? A pair whose relations are homogeneous in
    # space is safe to aggregate; a pair whose smooth and crease evidence sit
    # in different places is not.
    pair_rows = []
    for key, verdict in sorted(verdicts.items()):
        relation_positions: dict[str, list] = {"same_surface": [], "crease": [], "parallel": []}
        for edge in result.manifold_affinity.edges:
            if edge.candidate_status != CANDIDATE_STATUS_CANDIDATE:
                continue
            left = regions.node_region_id[edge.source]
            right = regions.node_region_id[edge.target]
            if left < 0 or right < 0 or left == right:
                continue
            if (min(left, right), max(left, right)) != key:
                continue
            midpoint = (positions[edge.source] + positions[edge.target]) * 0.5
            if edge.manifold_relation == RELATION_SAME_SURFACE:
                relation_positions["same_surface"].append(midpoint)
            elif edge.manifold_relation == RELATION_CREASE:
                relation_positions["crease"].append(midpoint)
            elif edge.manifold_relation == RELATION_PARALLEL_SEPARATE:
                relation_positions["parallel"].append(midpoint)

        counts = {k: len(v) for k, v in relation_positions.items()}
        # How far apart are the different relation classes on this pair?
        separations = {}
        classes = [k for k, v in relation_positions.items() if v]
        for i, first in enumerate(classes):
            for second in classes[i + 1:]:
                a = torch.stack(relation_positions[first])
                b = torch.stack(relation_positions[second])
                separations[f"{first}_vs_{second}_min_distance"] = round(
                    float(torch.cdist(a, b).min()), 4
                )
        # Spatial extent of the same_surface evidence itself.
        extent = None
        if relation_positions["same_surface"]:
            pts = torch.stack(relation_positions["same_surface"])
            extent = round(float(torch.cdist(pts, pts).max()), 4) if pts.shape[0] > 1 else 0.0

        pair_rows.append({
            "region_pair": list(key),
            "verdict": verdict,
            "relation_counts": counts,
            "mixed_classes_present": len(classes) > 1,
            "same_surface_evidence_extent": extent,
            "class_separations": separations,
        })

    reasons = Counter(h.boundary_reason for h in result.boundary_halfedge_candidates)
    return {
        "label": label,
        "region_count": len(regions.regions),
        "verdicts": {str(k): v for k, v in sorted(verdicts.items())},
        "pair_detail": pair_rows,
        "candidate_reason_counts": dict(reasons),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="*", default=["sphere", "box", "cylinder", "thin_slab"])
    args = parser.parse_args()
    for scene_name in args.scenes:
        scene = make_gaussian_reliability_scene(scene_name, seed=0)
        print(json.dumps(audit(scene.positions, scene.covariances, scene_name), indent=2, default=str))


if __name__ == "__main__":
    main()
