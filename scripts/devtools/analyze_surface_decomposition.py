from __future__ import annotations

"""Stage 3 diagnostics-only agglomeration feasibility and threshold sweeps.

This script evaluates the diagnostic decomposition after it has run.  Scene
names and GT labels are never passed to the runtime merge function.  Formal
production validation still uses ``osn-gs benchmark --constructor
boundary_first`` because Stage 3 is intentionally not integrated there.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from nurbs_constructor_benchmark.scenes import make_scene
from osn_gs.surface.torch_surface_candidate_graph import (
    SurfaceCandidateGraph,
    build_surface_cell_candidate_graph,
)
from osn_gs.surface.torch_surface_components import build_surface_components
from osn_gs.surface.torch_surface_decomposition import (
    ProxySurfaceDecompositionConfig,
    build_proxy_surface_components_diagnostics,
)
from osn_gs.surface.torch_voxel_hierarchy import build_voxel_gaussian_hierarchy


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ProxySurfaceDecompositionConfig()


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _majority_leaf_labels(hierarchy: Any, leaf_ids: list[str], labels: torch.Tensor):
    lookup = {node.node_id: node for node in hierarchy.leaves()}
    result: dict[str, int] = {}
    for leaf_id in leaf_ids:
        values, counts = torch.unique(
            labels[lookup[leaf_id].gaussian_indices].long(), return_counts=True
        )
        result[leaf_id] = int(values[int(torch.argmax(counts))])
    return result


def _evaluate_labels(result: Any, hierarchy: Any, labels: torch.Tensor | None):
    if labels is None:
        return None
    leaf_labels = _majority_leaf_labels(hierarchy, result.initial_leaf_ids, labels)
    mixed_regions = []
    for region in result.final_regions:
        region_labels = sorted({leaf_labels[item] for item in region.member_leaf_ids})
        if len(region_labels) > 1:
            mixed_regions.append(
                {
                    "member_leaf_ids": list(region.member_leaf_ids),
                    "labels": region_labels,
                }
            )
    false_merge_steps = []
    for merge in result.merge_history:
        merge_labels = sorted({leaf_labels[item] for item in merge["member_leaf_ids"]})
        if len(merge_labels) > 1:
            false_merge_steps.append(
                {
                    "merge_index": merge["merge_index"],
                    "member_leaf_ids": list(merge["member_leaf_ids"]),
                    "labels": merge_labels,
                }
            )
    return {
        "gt_label_count": int(torch.unique(labels.long()).numel()),
        "mixed_final_region_count": len(mixed_regions),
        "mixed_final_regions": mixed_regions,
        "false_merge_step_count": len(false_merge_steps),
        "false_merge_steps": false_merge_steps,
        "component_count_matches_gt_labels": (
            result.component_count() == int(torch.unique(labels.long()).numel())
        ),
    }


def analyze_case(
    label: str,
    points: torch.Tensor,
    point_labels: torch.Tensor | None,
    *,
    config: ProxySurfaceDecompositionConfig = DEFAULT_CONFIG,
    voxel_min_count: int = 10,
    voxel_max_count: int = 150,
    voxel_max_depth: int = 6,
    include_full_diagnostics: bool = True,
    shuffle_candidate_order: bool = False,
) -> dict[str, Any]:
    hierarchy = build_voxel_gaussian_hierarchy(
        points,
        voxel_min_gaussian_count=voxel_min_count,
        voxel_max_gaussian_count=voxel_max_count,
        voxel_max_depth=voxel_max_depth,
    )
    production = build_surface_components(hierarchy, points)
    graph = build_surface_cell_candidate_graph(
        hierarchy,
        points,
        radius_factor=config.candidate_radius_factor,
        max_neighbors=config.candidate_max_neighbors,
    )
    if shuffle_candidate_order:
        graph = SurfaceCandidateGraph(
            node_ids=list(reversed(graph.node_ids)),
            edges=list(reversed(graph.edges)),
            config=dict(graph.config),
        )
    result = build_proxy_surface_components_diagnostics(
        hierarchy, points, candidate_graph=graph, config=config
    )
    payload = result.payload()
    compact = {
        "label": label,
        "point_count": int(points.shape[0]),
        "voxel_config": {
            "minimum_count": int(voxel_min_count),
            "maximum_count": int(voxel_max_count),
            "maximum_depth": int(voxel_max_depth),
        },
        "runtime_merge_inputs": [
            "adaptive_cells",
            "points",
            "candidate_graph",
            "explicit_config",
        ],
        "runtime_uses_scene_name": False,
        "runtime_uses_gt_topology": False,
        "runtime_uses_gt_component_count": False,
        "production_component_count_unchanged": production.component_count(),
        "diagnostic_component_count": result.component_count(),
        "initial_region_count": len(result.initial_leaf_ids),
        "candidate_edge_count": result.candidate_unique_edge_count,
        "merge_count": len(result.merge_history),
        "pair_evaluation_count": len(result.pair_evaluations),
        "stale_queue_entry_count": result.stale_queue_entry_count,
        "decision_reason_counts": dict(sorted(result.decision_reason_counts.items())),
        "termination_reason": result.termination_reason,
        "artifact_hash": _payload_hash(payload),
        "evaluation_after_runtime": _evaluate_labels(result, hierarchy, point_labels),
    }
    if include_full_diagnostics:
        compact["decomposition"] = payload
    return compact


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


def _density_gradient_points(count: int, seed: int, dense_fraction: float):
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


def _high_curvature_points(resolution: int = 24):
    axis = torch.linspace(-1.0, 1.0, resolution)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack(
        [
            xx.flatten(),
            yy.flatten(),
            0.55 * xx.flatten().square() + 0.25 * yy.flatten().square(),
        ],
        dim=1,
    )
    return points, torch.zeros(points.shape[0], dtype=torch.long)


def _scene_points_and_labels(name: str, count: int, seed: int):
    scene = make_scene(name, count, seed)
    if name == "crease":
        labels = scene.gt_patch_label(scene.points[:, :2])
    elif name == "close_parallel_sheets":
        labels = scene.gt_patch_label(scene.points)
    else:
        labels = torch.zeros(count, dtype=torch.long)
    return scene.points, labels


def _compact_suite(config: ProxySurfaceDecompositionConfig):
    cases = []
    for name in (
        "curved_annulus",
        "crease",
        "close_parallel_sheets",
        "density_gradient",
    ):
        points, labels = _scene_points_and_labels(name, 600, 0)
        cases.append(
            analyze_case(
                name,
                points,
                labels,
                config=config,
                include_full_diagnostics=False,
            )
        )
    points, labels = _disconnected_close_points(600, 0, 0.1)
    cases.append(
        analyze_case(
            "disconnected_close_gap_0.1",
            points,
            labels,
            config=config,
            include_full_diagnostics=False,
        )
    )
    return cases


def _threshold_sweeps():
    specs = {
        "max_normalized_proxy_rms": (0.05, 0.075, 0.1, 0.125),
        "max_normalized_error_increase": (0.003, 0.006, 0.01, 0.02),
        "max_support_gap_over_spacing": (2.0, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0),
        "max_layer_separation": (0.1, 0.15, 0.2, 0.25, 0.3, 0.5),
        "min_layer_rms_ratio": (0.5, 1.0, 2.0, 5.0),
        "min_layer_normalized_error_increase": (0.0005, 0.001, 0.002),
        "max_layer_residual_concentration": (1.5, 2.0, 2.5, 3.0),
        "minimum_support": (6, 10, 20),
        "proxy_regularization": (1e-8, 1e-6, 1e-4),
        "candidate_radius_factor": (0.0, 0.1, 0.25, 0.5),
        "candidate_max_neighbors": (0, 4, 8),
        "support_gap_quantile": (0.01, 0.02, 0.05, 0.1),
        "max_proxy_condition_number": (1e3, 1e6, 1e10),
    }
    result = {}
    for field_name, values in specs.items():
        result[field_name] = []
        for value in values:
            config = replace(DEFAULT_CONFIG, **{field_name: value})
            result[field_name].append(
                {
                    "value": value,
                    "config": config.payload(),
                    "cases": _compact_suite(config),
                }
            )
    return {"ranges": {key: list(values) for key, values in specs.items()}, "results": result}


def build_report() -> dict[str, Any]:
    required = []
    for name in (
        "curved_annulus",
        "crease",
        "close_parallel_sheets",
        "density_gradient",
        "plane",
        "planar_hole",
        "planar_hole_offcenter",
        "planar_hole_elliptical",
        "planar_hole_density_gradient",
        "mild_curved_sheet",
    ):
        points, labels = _scene_points_and_labels(name, 600, 0)
        required.append(analyze_case(name, points, labels))
    points, labels = _disconnected_close_points(600, 0, 0.1)
    required.append(analyze_case("disconnected_close_gap_0.1", points, labels))
    points, labels = _high_curvature_points()
    required.append(analyze_case("high_curvature_smooth", points, labels))

    curved_points, curved_labels = _scene_points_and_labels("curved_annulus", 600, 0)
    deterministic_a = analyze_case(
        "curved_annulus_repeat_a",
        curved_points,
        curved_labels,
        include_full_diagnostics=False,
    )
    deterministic_b = analyze_case(
        "curved_annulus_repeat_b",
        curved_points,
        curved_labels,
        include_full_diagnostics=False,
    )
    deterministic_shuffled = analyze_case(
        "curved_annulus_shuffled_candidates",
        curved_points,
        curved_labels,
        include_full_diagnostics=False,
        shuffle_candidate_order=True,
    )

    rotation_sweep = []
    for axis in ("x", "y", "z"):
        for degrees in (30.0, 60.0, 90.0):
            rotated = curved_points @ _rotation_matrix(axis, degrees).T
            rotation_sweep.append(
                analyze_case(
                    f"curved_annulus_rotate_{axis}_{degrees:g}",
                    rotated,
                    curved_labels,
                    include_full_diagnostics=False,
                )
            )

    point_count_sweep = []
    for count in (300, 600, 1200):
        points, labels = _scene_points_and_labels("curved_annulus", count, 0)
        point_count_sweep.append(
            analyze_case(
                f"curved_annulus_points_{count}",
                points,
                labels,
                include_full_diagnostics=False,
            )
        )

    resolution_sweep = []
    for maximum_count in (75, 150, 300):
        resolution_sweep.append(
            analyze_case(
                f"curved_annulus_voxel_max_count_{maximum_count}",
                curved_points,
                curved_labels,
                voxel_max_count=maximum_count,
                include_full_diagnostics=False,
            )
        )

    density_sweep = []
    for fraction in (0.0, 0.5, 0.7, 0.9):
        points = _density_gradient_points(600, 0, fraction)
        density_sweep.append(
            analyze_case(
                f"density_gradient_fraction_{fraction:g}",
                points,
                torch.zeros(600, dtype=torch.long),
                include_full_diagnostics=False,
            )
        )

    parallel_base, parallel_labels = _scene_points_and_labels(
        "close_parallel_sheets", 600, 0
    )
    parallel_sweep = []
    for gap in (0.03, 0.06, 0.12, 0.24, 0.48):
        points = parallel_base.clone()
        points[:, 2] = torch.where(
            parallel_labels == 0,
            torch.full((600,), gap * 0.5),
            torch.full((600,), -gap * 0.5),
        )
        parallel_sweep.append(
            analyze_case(
                f"parallel_layer_gap_{gap:g}",
                points,
                parallel_labels,
                include_full_diagnostics=False,
            )
        )

    disconnected_sweep = []
    for gap in (0.02, 0.05, 0.1, 0.2):
        points, labels = _disconnected_close_points(600, 0, gap)
        disconnected_sweep.append(
            analyze_case(
                f"disconnected_close_gap_{gap:g}",
                points,
                labels,
                include_full_diagnostics=False,
            )
        )

    seed_sweep = []
    for seed in range(5):
        for name in (
            "curved_annulus",
            "crease",
            "close_parallel_sheets",
            "density_gradient",
        ):
            points, labels = _scene_points_and_labels(name, 600, seed)
            seed_sweep.append(
                analyze_case(
                    f"{name}_seed_{seed}",
                    points,
                    labels,
                    include_full_diagnostics=False,
                )
            )
        points, labels = _disconnected_close_points(600, seed, 0.1)
        seed_sweep.append(
            analyze_case(
                f"disconnected_close_gap_0.1_seed_{seed}",
                points,
                labels,
                include_full_diagnostics=False,
            )
        )

    hashes = [
        deterministic_a["artifact_hash"],
        deterministic_b["artifact_hash"],
        deterministic_shuffled["artifact_hash"],
    ]
    return {
        "schema_version": 1,
        "stage": "proxy_decomposition_stage3_merge_only_diagnostics",
        "production_integration": False,
        "default_config_is_provisional": True,
        "default_config": DEFAULT_CONFIG.payload(),
        "required_cases": required,
        "determinism": {
            "hashes": hashes,
            "repeat_hash_identical": hashes[0] == hashes[1],
            "shuffled_candidate_hash_identical": hashes[0] == hashes[2],
        },
        "rotation_sweep": rotation_sweep,
        "point_count_sweep": point_count_sweep,
        "adaptive_leaf_resolution_sweep": resolution_sweep,
        "density_gradient_sweep": density_sweep,
        "parallel_layer_distance_sweep": parallel_sweep,
        "disconnected_close_distance_sweep": disconnected_sweep,
        "seed_sweep": seed_sweep,
        "threshold_sweeps": _threshold_sweeps(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "proxy_decomposition_stage3.json",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_report(), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
