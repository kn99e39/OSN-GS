from __future__ import annotations

"""Stage 2 scale/orientation/density diagnostics for the spatial graph.

This is a sweep artifact generator, not a constructor benchmark.  Formal
pipeline validation remains ``osn-gs benchmark --constructor boundary_first
--bf-candidate-diagnostics``.  Ground-truth labels are used only after graph
generation to categorize candidate types; the graph builder receives only the
hierarchy, raw points, and spatial configuration.
"""

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import torch

from nurbs_constructor_benchmark.scenes import make_scene
from osn_gs.surface.torch_surface_candidate_graph import (
    build_surface_cell_candidate_graph,
    candidate_graph_payload,
    classify_aabb_contact,
)
from osn_gs.surface.torch_surface_components import build_surface_components
from osn_gs.surface.torch_voxel_hierarchy import (
    STATE_ACTIVE,
    STATE_COMPLEX,
    build_voxel_gaussian_hierarchy,
)


ROOT = Path(__file__).resolve().parents[2]


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _majority_leaf_labels(leaves: dict[str, Any], point_labels: torch.Tensor | None):
    if point_labels is None:
        return {}
    result: dict[str, int] = {}
    for leaf_id, leaf in leaves.items():
        labels = point_labels[leaf.gaussian_indices].to(torch.long)
        values, counts = torch.unique(labels, return_counts=True)
        best = int(torch.argmax(counts))
        result[leaf_id] = int(values[best])
    return result


def _cross_component_touch_pairs(
    hierarchy: Any,
    components: Any,
    leaves: dict[str, Any],
    leaf_labels: dict[str, int],
) -> list[tuple[str, str]]:
    root = hierarchy.nodes[0]
    tolerance = max(float((root.aabb_max - root.aabb_min).abs().max()), 1e-6) * 1e-8
    result = []
    leaf_ids = sorted(leaves)
    for index, leaf_a_id in enumerate(leaf_ids):
        for leaf_b_id in leaf_ids[index + 1 :]:
            if components.leaf_component_id[leaf_a_id] == components.leaf_component_id[leaf_b_id]:
                continue
            if leaf_labels and leaf_labels[leaf_a_id] != leaf_labels[leaf_b_id]:
                continue
            leaf_a, leaf_b = leaves[leaf_a_id], leaves[leaf_b_id]
            relation = classify_aabb_contact(
                leaf_a.aabb_min,
                leaf_a.aabb_max,
                leaf_b.aabb_min,
                leaf_b.aabb_max,
                tolerance,
            )
            if relation != "disjoint":
                result.append((leaf_a_id, leaf_b_id))
    return result


def analyze_points(
    label: str,
    points: torch.Tensor,
    point_labels: torch.Tensor | None = None,
    *,
    radius_factor: float = 0.25,
    max_neighbors: int = 0,
    voxel_min_count: int = 10,
    voxel_max_count: int = 150,
    voxel_max_depth: int = 6,
    known_missing_pairs: list[tuple[str, str]] | None = None,
    gt_cross_candidate_type: str | None = None,
) -> dict[str, Any]:
    hierarchy = build_voxel_gaussian_hierarchy(
        points,
        voxel_min_gaussian_count=voxel_min_count,
        voxel_max_gaussian_count=voxel_max_count,
        voxel_max_depth=voxel_max_depth,
    )
    components = build_surface_components(hierarchy, points)
    leaves = {
        leaf.node_id: leaf
        for leaf in hierarchy.leaves()
        if leaf.state in {STATE_ACTIVE, STATE_COMPLEX}
    }
    leaf_labels = _majority_leaf_labels(leaves, point_labels)
    face_reason = {
        _canonical_pair(edge.leaf_a, edge.leaf_b): edge.reason
        for edge in components.edge_decisions
    }
    existing_smooth = [pair for pair, reason in face_reason.items() if reason == "merged"]
    existing_all = list(face_reason)
    same_truth_cross_component_touch = _cross_component_touch_pairs(
        hierarchy, components, leaves, leaf_labels
    )
    reference_pairs = {
        "existing_face_all": existing_all,
        "existing_face_smooth": existing_smooth,
        "same_truth_cross_component_touch": same_truth_cross_component_touch,
    }
    present_known = [
        _canonical_pair(a, b)
        for a, b in (known_missing_pairs or [])
        if a in leaves and b in leaves
    ]
    if present_known:
        reference_pairs["known_missing_smooth"] = present_known

    start = time.perf_counter()
    graph = build_surface_cell_candidate_graph(
        hierarchy,
        points,
        radius_factor=radius_factor,
        max_neighbors=max_neighbors,
    )
    elapsed = time.perf_counter() - start
    diagnostic_tags: dict[tuple[str, str], list[str]] = {}
    for pair in graph.edge_pairs():
        tags: list[str] = []
        if pair in face_reason:
            tags.append(f"legacy_face_{face_reason[pair]}")
        elif components.leaf_component_id.get(pair[0]) == components.leaf_component_id.get(pair[1]):
            tags.append("nonface_same_component")
        else:
            tags.append("nonface_cross_component")
        if leaf_labels:
            tags.append(
                "gt_same_surface"
                if leaf_labels[pair[0]] == leaf_labels[pair[1]]
                else "gt_cross_surface"
            )
        diagnostic_tags[pair] = tags

    payload = candidate_graph_payload(
        graph,
        reference_pairs=reference_pairs,
        diagnostic_tags=diagnostic_tags,
    )
    edge_count = max(payload["edge_count"], 1)
    category_counts = dict(payload["diagnostic_tag_counts"])
    gt_cross_count = category_counts.get("gt_cross_surface", 0)
    false_types = (
        {gt_cross_candidate_type: gt_cross_count}
        if gt_cross_candidate_type is not None and gt_cross_count > 0
        else {}
    )
    payload.update(
        {
            "label": label,
            "point_count": int(points.shape[0]),
            "hierarchy_state_counts": hierarchy.state_counts(),
            "component_count_unchanged": components.component_count(),
            "candidate_build_seconds": elapsed,
            "candidate_category_counts": category_counts,
            "candidate_category_ratios": {
                key: value / edge_count for key, value in category_counts.items()
            },
            "false_candidate_type_counts": false_types,
            "false_candidate_type_ratios": {
                key: value / edge_count for key, value in false_types.items()
            },
        }
    )
    return payload


def _rotation_matrix(axis: str, degrees: float) -> torch.Tensor:
    angle = math.radians(float(degrees))
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        values = ((1, 0, 0), (0, c, -s), (0, s, c))
    elif axis == "y":
        values = ((c, 0, s), (0, 1, 0), (-s, 0, c))
    elif axis == "z":
        values = ((c, -s, 0), (s, c, 0), (0, 0, 1))
    else:
        raise ValueError(axis)
    return torch.tensor(values, dtype=torch.float32)


def _density_gradient_points(count: int, seed: int, dense_fraction: float) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    dense_count = int(round(count * dense_fraction))
    sparse_count = count - dense_count
    dense = (torch.randn((dense_count, 2), generator=generator) * 0.18).clamp(-1.0, 1.0)
    sparse = torch.rand((sparse_count, 2), generator=generator) * 2.0 - 1.0
    xy = torch.cat([dense, sparse], dim=0)
    x, y = xy[:, 0], xy[:, 1]
    z = 0.20 * torch.sin(2.4 * x) * torch.cos(1.8 * y)
    return torch.stack([x, y, z], dim=1)


def _disconnected_close_points(count: int, seed: int, gap: float):
    generator = torch.Generator().manual_seed(seed)
    left_count = count // 2
    right_count = count - left_count
    left = torch.rand((left_count, 2), generator=generator)
    right = torch.rand((right_count, 2), generator=generator)
    left[:, 0] = -1.0 + left[:, 0] * (1.0 - gap * 0.5)
    right[:, 0] = gap * 0.5 + right[:, 0] * (1.0 - gap * 0.5)
    left[:, 1] = left[:, 1] * 2.0 - 1.0
    right[:, 1] = right[:, 1] * 2.0 - 1.0
    points = torch.cat(
        [
            torch.cat([left, torch.zeros((left_count, 1))], dim=1),
            torch.cat([right, torch.zeros((right_count, 1))], dim=1),
        ],
        dim=0,
    )
    labels = torch.cat(
        [torch.zeros(left_count, dtype=torch.long), torch.ones(right_count, dtype=torch.long)]
    )
    return points, labels


def build_report() -> dict[str, Any]:
    known_missing = [("r07", "r52"), ("r05", "r50"), ("r05", "r52"), ("r07", "r50")]
    curved = make_scene("curved_annulus", 600, 0)
    all_same = torch.zeros(600, dtype=torch.long)

    report: dict[str, Any] = {
        "schema_version": 1,
        "stage": "proxy_decomposition_stage2_candidate_graph",
        "production_membership_changed": False,
        "default": analyze_points(
            "curved_annulus_default",
            curved.points,
            all_same,
            known_missing_pairs=known_missing,
        ),
        "radius_factor_sweep": [],
        "rotation_sweep": [],
        "point_count_sweep": [],
        "adaptive_leaf_resolution_sweep": [],
        "density_gradient_sweep": [],
        "crease_case": None,
        "parallel_layer_distance_sweep": [],
        "disconnected_close_sweep": [],
    }
    for factor in (0.0, 0.1, 0.25, 0.5, 1.0):
        report["radius_factor_sweep"].append(
            analyze_points(
                f"curved_annulus_radius_{factor:g}",
                curved.points,
                all_same,
                radius_factor=factor,
                known_missing_pairs=known_missing,
            )
        )
    for axis in ("x", "y", "z"):
        for degrees in (30.0, 60.0, 90.0):
            rotation = _rotation_matrix(axis, degrees)
            rotated = curved.points @ rotation.T
            report["rotation_sweep"].append(
                analyze_points(
                    f"curved_annulus_rotate_{axis}_{degrees:g}",
                    rotated,
                    all_same,
                )
            )
    for count in (300, 600, 1200):
        scene = make_scene("curved_annulus", count, 0)
        report["point_count_sweep"].append(
            analyze_points(
                f"curved_annulus_points_{count}",
                scene.points,
                torch.zeros(count, dtype=torch.long),
            )
        )
    for max_count in (75, 150, 300):
        report["adaptive_leaf_resolution_sweep"].append(
            analyze_points(
                f"curved_annulus_voxel_max_count_{max_count}",
                curved.points,
                all_same,
                voxel_max_count=max_count,
            )
        )
    for fraction in (0.0, 0.5, 0.7, 0.9):
        points = _density_gradient_points(600, 0, fraction)
        report["density_gradient_sweep"].append(
            analyze_points(
                f"density_gradient_fraction_{fraction:g}",
                points,
                torch.zeros(600, dtype=torch.long),
            )
        )
    crease = make_scene("crease", 600, 0)
    report["crease_case"] = analyze_points(
        "crease_default",
        crease.points,
        crease.gt_patch_label(crease.points[:, :2]),
        gt_cross_candidate_type="crease_cross",
    )
    parallel = make_scene("close_parallel_sheets", 600, 0)
    parallel_labels = (parallel.points[:, 2] < 0).to(torch.long)
    for gap in (0.03, 0.06, 0.12, 0.24, 0.48):
        points = parallel.points.clone()
        points[:, 2] = torch.where(
            parallel_labels == 0,
            torch.full((600,), gap * 0.5),
            torch.full((600,), -gap * 0.5),
        )
        report["parallel_layer_distance_sweep"].append(
            analyze_points(
                f"parallel_layer_gap_{gap:g}",
                points,
                parallel_labels,
                gt_cross_candidate_type="parallel_layer_cross",
            )
        )
    for gap in (0.02, 0.05, 0.1, 0.2):
        points, labels = _disconnected_close_points(600, 0, gap)
        report["disconnected_close_sweep"].append(
            analyze_points(
                f"disconnected_close_gap_{gap:g}",
                points,
                labels,
                gt_cross_candidate_type="disconnected_region_cross",
            )
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "proxy_decomposition_stage2.json",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_report(), indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
