from __future__ import annotations

"""Stage 3-R Gaussian-native support-continuity investigation.

The runtime evaluator receives only region indices and Gaussian mean,
covariance, and opacity tensors.  Scene names and GT labels are used here only
afterward to select/evaluate benchmark conflict pairs.  No component merge or
production integration occurs.
"""

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from nurbs_constructor_benchmark.scenes import make_scene
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_gaussian_support_continuity import (
    GaussianSupportContinuityConfig,
    covariance_from_scale_rotation,
    evaluate_gaussian_support_continuity,
)
from osn_gs.surface.torch_surface_candidate_graph import build_surface_cell_candidate_graph
from osn_gs.surface.torch_surface_components import build_surface_components
from osn_gs.surface.torch_surface_decomposition import (
    ProxySurfaceDecompositionConfig,
    build_proxy_surface_components_diagnostics,
)
from osn_gs.surface.torch_voxel_hierarchy import build_voxel_gaussian_hierarchy


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = GaussianSupportContinuityConfig()
STAGE3_CONFIG = ProxySurfaceDecompositionConfig()


def _hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rotation_matrix(axis: str, degrees: float, dtype=torch.float64) -> torch.Tensor:
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
    return torch.tensor(values, dtype=dtype)


def _production_initial_fields(points: torch.Tensor, scale_multiplier: float = 1.0):
    config = TorchPipelineConfig(
        covariance_init="knn",
        covariance_knn_chunk_size=max(1, int(points.shape[0])),
        covariance_scale_multiplier=float(scale_multiplier),
    )
    pipeline = TorchOSNGSPipeline(config, device="cpu")
    scales = pipeline._initial_covariance_scales(points.float()).double()
    rotations = torch.zeros((points.shape[0], 4), dtype=torch.float64)
    rotations[:, 0] = 1.0
    covariance = covariance_from_scale_rotation(scales, rotations)
    opacity = torch.full((points.shape[0],), 0.12, dtype=torch.float64)
    return covariance, opacity, scales, rotations


def _local_spacing(points: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(points.double(), points.double())
    distances.fill_diagonal_(float("inf"))
    return distances.min(dim=1).values.clamp_min(1e-5)


def _frame_covariances(normals: torch.Tensor, spacing: torch.Tensor, tangent_factor=1.4, normal_factor=0.2):
    normals = torch.nn.functional.normalize(normals.double(), dim=1)
    reference = torch.zeros_like(normals)
    reference[:, 0] = 1.0
    parallel = (normals[:, 0].abs() > 0.9)
    reference[parallel] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    tangent_u = torch.nn.functional.normalize(torch.cross(normals, reference, dim=1), dim=1)
    tangent_v = torch.nn.functional.normalize(torch.cross(normals, tangent_u, dim=1), dim=1)
    basis = torch.stack([tangent_u, tangent_v, normals], dim=2)
    scales = torch.stack(
        [
            spacing * float(tangent_factor),
            spacing * float(tangent_factor),
            spacing * float(normal_factor),
        ],
        dim=1,
    )
    return basis @ torch.diag_embed(scales.square()) @ basis.transpose(1, 2)


def _case_state(points: torch.Tensor):
    hierarchy = build_voxel_gaussian_hierarchy(
        points,
        voxel_min_gaussian_count=10,
        voxel_max_gaussian_count=150,
        voxel_max_depth=6,
    )
    production = build_surface_components(hierarchy, points)
    graph = build_surface_cell_candidate_graph(
        hierarchy,
        points,
        radius_factor=STAGE3_CONFIG.candidate_radius_factor,
        max_neighbors=STAGE3_CONFIG.candidate_max_neighbors,
    )
    decomposition = build_proxy_surface_components_diagnostics(
        hierarchy, points, graph, STAGE3_CONFIG
    )
    node_lookup = {node.node_id: node for node in hierarchy.nodes}
    return hierarchy, production, graph, decomposition, node_lookup


def _leaf_indices(leaf_ids, node_lookup):
    values = torch.cat([node_lookup[leaf_id].gaussian_indices.long() for leaf_id in sorted(leaf_ids)])
    return torch.unique(values, sorted=True)


def _majority_leaf_labels(hierarchy, labels):
    result = {}
    for leaf in hierarchy.leaves():
        if leaf.gaussian_indices is None or int(leaf.gaussian_indices.numel()) == 0:
            continue
        values, counts = torch.unique(labels[leaf.gaussian_indices].long(), return_counts=True)
        result[leaf.node_id] = int(values[int(torch.argmax(counts))])
    return result


def _first_false_merge_pair(decomposition, hierarchy, labels, node_lookup):
    leaf_labels = _majority_leaf_labels(hierarchy, labels)
    for merge in decomposition.merge_history:
        if len({leaf_labels[leaf] for leaf in merge["member_leaf_ids"]}) <= 1:
            continue
        evaluation = decomposition.pair_evaluations[merge["pair_evaluation_index"]]
        return (
            _leaf_indices(evaluation.member_leaf_ids_a, node_lookup),
            _leaf_indices(evaluation.member_leaf_ids_b, node_lookup),
            evaluation,
        )
    raise RuntimeError("expected a false merge but none was found")


def _final_split_pair(decomposition):
    if decomposition.component_count() != 2:
        raise RuntimeError(f"expected two final regions, got {decomposition.component_count()}")
    region_a, region_b = decomposition.final_regions
    candidates = [
        evaluation
        for evaluation in decomposition.pair_evaluations
        if {
            frozenset(evaluation.member_leaf_ids_a),
            frozenset(evaluation.member_leaf_ids_b),
        }
        == {
            frozenset(region_a.member_leaf_ids),
            frozenset(region_b.member_leaf_ids),
        }
    ]
    evaluation = candidates[-1] if candidates else None
    return region_a.gaussian_indices, region_b.gaussian_indices, evaluation


def _record(
    label: str,
    category: str,
    expected_continuity: bool,
    points: torch.Tensor,
    region_a: torch.Tensor,
    region_b: torch.Tensor,
    covariance: torch.Tensor,
    opacity: torch.Tensor,
    covariance_source: str,
    config: GaussianSupportContinuityConfig = DEFAULT_CONFIG,
    provenance: dict[str, Any] | None = None,
):
    diagnostics = evaluate_gaussian_support_continuity(
        region_a,
        region_b,
        points,
        covariance,
        opacity,
        config,
    )
    payload = diagnostics.payload()
    return {
        "label": label,
        "category": category,
        "expected_continuity": bool(expected_continuity),
        "covariance_source": covariance_source,
        "runtime_signal_inputs": [
            "region_indices",
            "gaussian_means",
            "gaussian_covariances",
            "gaussian_opacities",
            "explicit_config",
        ],
        "runtime_uses_scene_name": False,
        "runtime_uses_gt": False,
        "provenance": provenance or {},
        "diagnostics": payload,
        "diagnostics_hash": _hash(payload),
    }


def _actual_conflict_records():
    records = []
    cached = {}

    def scene_state(name, seed):
        key = (name, seed)
        if key not in cached:
            scene = make_scene(name, 600, seed)
            cached[key] = (scene, *_case_state(scene.points))
        return cached[key]

    for name, seed in (("density_gradient", 2), ("mild_curved_sheet", 0)):
        scene, hierarchy, production, graph, decomposition, node_lookup = scene_state(name, seed)
        region_a, region_b, evaluation = _final_split_pair(decomposition)
        covariance, opacity, scales, rotations = _production_initial_fields(scene.points)
        records.append(
            _record(
                f"{name}_seed_{seed}_stage3_final_split",
                "actual_positive_conflict",
                True,
                scene.points,
                region_a,
                region_b,
                covariance,
                opacity,
                "production_knn_isotropic_identity_rotation_constant_opacity",
                provenance={
                    "stage3_pair_evaluation_index": None if evaluation is None else evaluation.evaluation_index,
                    "stage3_decision_reason": None if evaluation is None else evaluation.decision_reason,
                    "stage3_gap_over_spacing": None if evaluation is None else evaluation.diagnostics.scale_normalized_support_gap,
                    "production_component_count": production.component_count(),
                    "stage3_final_region_count": decomposition.component_count(),
                },
            )
        )

    scene, hierarchy, production, graph, decomposition, node_lookup = scene_state("curved_annulus", 0)
    covariance, opacity, scales, rotations = _production_initial_fields(scene.points)
    face_pairs = {
        tuple(sorted((edge.leaf_a, edge.leaf_b)))
        for edge in production.edge_decisions
    }
    missing_pairs = []
    graph_leaf_ids = sorted(graph.node_ids)
    for position, leaf_a_id in enumerate(graph_leaf_ids):
        leaf_a = node_lookup[leaf_a_id]
        component_a = production.leaf_component_id[leaf_a_id]
        for leaf_b_id in graph_leaf_ids[position + 1 :]:
            if production.leaf_component_id[leaf_b_id] == component_a:
                continue
            pair = tuple(sorted((leaf_a_id, leaf_b_id)))
            if pair in face_pairs:
                continue
            leaf_b = node_lookup[leaf_b_id]
            delta = torch.maximum(
                leaf_a.aabb_min - leaf_b.aabb_max,
                leaf_b.aabb_min - leaf_a.aabb_max,
            ).clamp_min(0.0)
            if float(torch.linalg.norm(delta)) > 1e-8:
                continue
            centroid_distance = float(
                torch.linalg.norm(leaf_a.plane.centroid - leaf_b.plane.centroid)
            )
            missing_pairs.append((centroid_distance, pair))
    for _, (leaf_a_id, leaf_b_id) in sorted(missing_pairs)[:4]:
        records.append(
            _record(
                f"curved_annulus_missing_{leaf_a_id}_{leaf_b_id}",
                "actual_positive_curved_missing",
                True,
                scene.points,
                node_lookup[leaf_a_id].gaussian_indices,
                node_lookup[leaf_b_id].gaussian_indices,
                covariance,
                opacity,
                "production_knn_isotropic_identity_rotation_constant_opacity",
                provenance={
                    "leaf_pair": [leaf_a_id, leaf_b_id],
                    "source": "stage1_aabb_touch_without_face_contact_nearest_four",
                },
            )
        )
    for edge in production.edge_decisions:
        if edge.reason != "merged":
            continue
        records.append(
            _record(
                f"curved_annulus_face_smooth_{edge.leaf_a}_{edge.leaf_b}",
                "actual_positive_face_smooth",
                True,
                scene.points,
                node_lookup[edge.leaf_a].gaussian_indices,
                node_lookup[edge.leaf_b].gaussian_indices,
                covariance,
                opacity,
                "production_knn_isotropic_identity_rotation_constant_opacity",
                provenance={"leaf_pair": [edge.leaf_a, edge.leaf_b], "legacy_reason": edge.reason},
            )
        )

    for gap, seeds in ((0.02, (0,)), (0.05, (0,)), (0.1, (1, 2, 4))):
        for seed in seeds:
            points, labels = _disconnected_points(600, seed, gap)
            hierarchy, production, graph, decomposition, node_lookup = _case_state(points)
            region_a, region_b, evaluation = _first_false_merge_pair(
                decomposition, hierarchy, labels, node_lookup
            )
            covariance, opacity, scales, rotations = _production_initial_fields(points)
            records.append(
                _record(
                    f"disconnected_gap_{gap:g}_seed_{seed}_first_false_merge",
                    "actual_negative_disconnected",
                    False,
                    points,
                    region_a,
                    region_b,
                    covariance,
                    opacity,
                    "production_knn_isotropic_identity_rotation_constant_opacity",
                    provenance={
                        "stage3_pair_evaluation_index": evaluation.evaluation_index,
                        "stage3_gap_over_spacing": evaluation.diagnostics.scale_normalized_support_gap,
                    },
                )
            )

    for name, rejected_reason in (("close_parallel_sheets", "offset"), ("crease", "normal")):
        scene, hierarchy, production, graph, decomposition, node_lookup = scene_state(name, 0)
        covariance, opacity, scales, rotations = _production_initial_fields(scene.points)
        for edge in production.edge_decisions:
            if edge.reason != rejected_reason:
                continue
            records.append(
                _record(
                    f"{name}_{edge.leaf_a}_{edge.leaf_b}",
                    f"actual_negative_{name}",
                    False,
                    scene.points,
                    node_lookup[edge.leaf_a].gaussian_indices,
                    node_lookup[edge.leaf_b].gaussian_indices,
                    covariance,
                    opacity,
                    "production_knn_isotropic_identity_rotation_constant_opacity",
                    provenance={"leaf_pair": [edge.leaf_a, edge.leaf_b], "legacy_reason": edge.reason},
                )
            )

    high_points, high_a, high_b, high_covariance, high_opacity = _high_curvature_fixture()
    records.append(
        _record(
            "high_curvature_smooth_analytic",
            "analytic_positive_high_curvature",
            True,
            high_points,
            high_a,
            high_b,
            high_covariance,
            high_opacity,
            "diagnostics_fixture_tangent_aligned",
        )
    )
    coplanar_points, coplanar_a, coplanar_b, coplanar_covariance, coplanar_opacity = _coplanar_disconnected_fixture()
    records.append(
        _record(
            "coplanar_disconnected_analytic",
            "analytic_negative_coplanar_disconnected",
            False,
            coplanar_points,
            coplanar_a,
            coplanar_b,
            coplanar_covariance,
            coplanar_opacity,
            "diagnostics_fixture_nonbridging_anisotropic",
        )
    )
    return records


def _disconnected_points(count: int, seed: int, gap: float):
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
    labels = torch.cat([torch.zeros(left_count, dtype=torch.long), torch.ones(right_count, dtype=torch.long)])
    return points, labels


def _high_curvature_fixture(resolution=18):
    x = torch.linspace(-1.0, 1.0, resolution, dtype=torch.float64)
    y = torch.linspace(-0.5, 0.5, 8, dtype=torch.float64)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    points = torch.stack([xx.flatten(), yy.flatten(), 0.55 * xx.flatten().square() + 0.25 * yy.flatten().square()], dim=1)
    normals = torch.stack([-1.1 * points[:, 0], -0.5 * points[:, 1], torch.ones(points.shape[0], dtype=torch.float64)], dim=1)
    covariance = _frame_covariances(normals, _local_spacing(points))
    midpoint = points[:, 0] < 0.0
    return points, torch.nonzero(midpoint).reshape(-1), torch.nonzero(~midpoint).reshape(-1), covariance, torch.ones(points.shape[0], dtype=torch.float64) * 0.8


def _coplanar_disconnected_fixture(gap=0.12):
    y = torch.linspace(-1.0, 1.0, 18, dtype=torch.float64)
    x_left = torch.linspace(-1.0, -gap * 0.5, 8, dtype=torch.float64)
    x_right = torch.linspace(gap * 0.5, 1.0, 8, dtype=torch.float64)
    left_x, left_y = torch.meshgrid(x_left, y, indexing="ij")
    right_x, right_y = torch.meshgrid(x_right, y, indexing="ij")
    left = torch.stack([left_x.flatten(), left_y.flatten(), torch.zeros(left_x.numel(), dtype=torch.float64)], dim=1)
    right = torch.stack([right_x.flatten(), right_y.flatten(), torch.zeros(right_x.numel(), dtype=torch.float64)], dim=1)
    points = torch.cat([left, right], dim=0)
    spacing = _local_spacing(points)
    normals = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).repeat(points.shape[0], 1)
    covariance = _frame_covariances(normals, spacing, tangent_factor=0.55, normal_factor=0.15)
    # Rotate elongated tangent support along y, not across the x gap.
    covariance[:, 0, 0] *= 0.15
    split = left.shape[0]
    return points, torch.arange(split), torch.arange(split, points.shape[0]), covariance, torch.ones(points.shape[0], dtype=torch.float64) * 0.8


def _fixture_records():
    records = []

    # Sparse smooth surface with tangent-aligned elongated support.
    x = torch.tensor([-1.0, -0.7, -0.4, 0.4, 0.7, 1.0], dtype=torch.float64)
    y = torch.linspace(-0.5, 0.5, 5, dtype=torch.float64)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    points = torch.stack([xx.flatten(), yy.flatten(), 0.08 * xx.flatten().square()], dim=1)
    normals = torch.stack([-0.16 * points[:, 0], torch.zeros(points.shape[0]), torch.ones(points.shape[0])], dim=1)
    covariance = _frame_covariances(normals, _local_spacing(points), tangent_factor=1.8)
    records.append(_record("fixture_sparse_smooth_tangent_elongated", "fixture_positive", True, points, torch.arange(15), torch.arange(15, 30), covariance, torch.ones(30) * 0.8, "diagnostics_fixture_tangent_aligned"))

    # Disconnected coplanar: isotropic and elongated-but-non-bridging variants.
    points, region_a, region_b, covariance, opacity = _coplanar_disconnected_fixture()
    isotropic_scale = _local_spacing(points) * 0.55
    isotropic = torch.eye(3, dtype=torch.float64)[None] * isotropic_scale[:, None, None].square()
    records.append(_record("fixture_disconnected_coplanar_isotropic", "fixture_negative", False, points, region_a, region_b, isotropic, opacity, "diagnostics_fixture_small_isotropic"))
    records.append(_record("fixture_disconnected_coplanar_nonbridging_elongated", "fixture_negative", False, points, region_a, region_b, covariance, opacity, "diagnostics_fixture_nonbridging_anisotropic"))

    # Parallel layers.
    axis = torch.linspace(-0.8, 0.8, 8, dtype=torch.float64)
    gx, gy = torch.meshgrid(axis, axis, indexing="ij")
    plane = torch.stack([gx.flatten(), gy.flatten(), torch.zeros(gx.numel())], dim=1)
    parallel = torch.cat([plane + torch.tensor([0.0, 0.0, 0.06]), plane - torch.tensor([0.0, 0.0, 0.06])], dim=0)
    normals = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).repeat(parallel.shape[0], 1)
    parallel_cov = _frame_covariances(normals, _local_spacing(parallel), tangent_factor=1.2, normal_factor=0.12)
    records.append(_record("fixture_parallel_layers", "fixture_negative", False, parallel, torch.arange(64), torch.arange(64, 128), parallel_cov, torch.ones(128) * 0.8, "diagnostics_fixture_tangent_aligned"))

    # Crease with independently rotating tangent planes.
    y = torch.linspace(-0.8, 0.8, 10, dtype=torch.float64)
    left_x = torch.linspace(-0.8, -0.02, 6, dtype=torch.float64)
    right_x = torch.linspace(0.02, 0.8, 6, dtype=torch.float64)
    lx, ly = torch.meshgrid(left_x, y, indexing="ij")
    rx, ry = torch.meshgrid(right_x, y, indexing="ij")
    left = torch.stack([lx.flatten(), ly.flatten(), -0.45 * lx.flatten()], dim=1)
    right = torch.stack([rx.flatten(), ry.flatten(), 0.45 * rx.flatten()], dim=1)
    crease = torch.cat([left, right], dim=0)
    crease_normals = torch.cat([torch.tensor([0.45, 0.0, 1.0]).repeat(60, 1), torch.tensor([-0.45, 0.0, 1.0]).repeat(60, 1)]).double()
    crease_cov = _frame_covariances(crease_normals, _local_spacing(crease), tangent_factor=1.2, normal_factor=0.15)
    records.append(_record("fixture_crease", "fixture_negative", False, crease, torch.arange(60), torch.arange(60, 120), crease_cov, torch.ones(120) * 0.8, "diagnostics_fixture_tangent_aligned"))

    high = _high_curvature_fixture()
    records.append(_record("fixture_curved_rotating_tangent_frame", "fixture_positive", True, *high, "diagnostics_fixture_tangent_aligned"))

    # Density-gradient surface with covariance scale following local density.
    scene = make_scene("density_gradient", 600, 2)
    spacing = _local_spacing(scene.points)
    residual, normals = scene.oracle(scene.points)
    density_cov = _frame_covariances(normals, spacing, tangent_factor=1.4, normal_factor=0.2)
    hierarchy, production, graph, decomposition, node_lookup = _case_state(scene.points)
    region_a, region_b, evaluation = _final_split_pair(decomposition)
    records.append(_record("fixture_density_gradient_scale_varying", "fixture_positive", True, scene.points, region_a, region_b, density_cov, torch.ones(600) * 0.8, "diagnostics_fixture_density_scaled_tangent_aligned"))
    return records


def _scalar_signals(record):
    diagnostics = record["diagnostics"]
    qkey = "q0.1"
    return {
        "point_gap_over_spacing": diagnostics["existing_point_diagnostics"]["support_gap_over_local_spacing"],
        "pooled_mahalanobis_q0.1": diagnostics["mahalanobis"]["pooled"]["quantiles"][qkey],
        "symmetric_mahalanobis_q0.1": diagnostics["mahalanobis"]["symmetric_mean"]["quantiles"][qkey],
        "ellipsoid_k2_margin_q0.1": diagnostics["ellipsoid_overlap"]["k2"]["signed_overlap_margin"]["quantiles"][qkey],
        "directional_reach_ratio_q0.1": diagnostics["projected_reach"]["center_gap_over_directional_reach"]["quantiles"][qkey],
        "tangent_reach_ratio_q0.1": diagnostics["projected_reach"]["tangent_reach_ratio"]["quantiles"][qkey],
        "normal_reach_ratio_q0.1": diagnostics["projected_reach"]["normal_reach_ratio"]["quantiles"][qkey],
        "bridge_endpoint_ratio_unweighted_q0.1": diagnostics["bridge_density"]["unweighted"]["endpoint_minimum_ratio"]["quantiles"][qkey],
        "bridge_endpoint_ratio_opacity_q0.1": diagnostics["bridge_density"]["opacity_weighted"]["endpoint_minimum_ratio"]["quantiles"][qkey],
        "facing_opacity_mass_per_gap": diagnostics["facing_support"]["opacity_mass_per_normalized_gap"],
    }


def _auc(values, labels, higher_positive):
    comparable = 0
    wins = 0.0
    for i, positive in enumerate(labels):
        if not positive:
            continue
        for j, negative in enumerate(labels):
            if negative:
                continue
            comparable += 1
            left, right = values[i], values[j]
            if math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
                wins += 0.5
            elif (left > right) == higher_positive:
                wins += 1.0
    return wins / comparable if comparable else None


def _aggregate(records, covariance_source=None):
    selected = [record for record in records if covariance_source is None or record["covariance_source"] == covariance_source]
    signal_rows = [{"label": record["label"], "positive": record["expected_continuity"], **_scalar_signals(record)} for record in selected]
    directions = {
        "point_gap_over_spacing": False,
        "pooled_mahalanobis_q0.1": False,
        "symmetric_mahalanobis_q0.1": False,
        "ellipsoid_k2_margin_q0.1": True,
        "directional_reach_ratio_q0.1": False,
        "tangent_reach_ratio_q0.1": False,
        "normal_reach_ratio_q0.1": False,
        "bridge_endpoint_ratio_unweighted_q0.1": True,
        "bridge_endpoint_ratio_opacity_q0.1": True,
        "facing_opacity_mass_per_gap": True,
    }
    summaries = {}
    labels = [row["positive"] for row in signal_rows]
    for signal, higher_positive in directions.items():
        values = [float(row[signal]) for row in signal_rows]
        positives = [value for value, positive in zip(values, labels) if positive]
        negatives = [value for value, positive in zip(values, labels) if not positive]
        if not positives or not negatives:
            continue
        positive_boundary = min(positives) if higher_positive else max(positives)
        negative_boundary = max(negatives) if higher_positive else min(negatives)
        margin = positive_boundary - negative_boundary if higher_positive else negative_boundary - positive_boundary
        summaries[signal] = {
            "higher_means_continuity": higher_positive,
            "roc_auc": _auc(values, labels, higher_positive),
            "positive_min": min(positives),
            "positive_median": sorted(positives)[len(positives) // 2],
            "positive_max": max(positives),
            "negative_min": min(negatives),
            "negative_median": sorted(negatives)[len(negatives) // 2],
            "negative_max": max(negatives),
            "minimum_separation_margin": margin,
            "strict_common_threshold_exists": margin > 0.0,
            "overlap_interval": None if margin > 0.0 else sorted([positive_boundary, negative_boundary]),
        }
    return {"record_count": len(selected), "rows": signal_rows, "signals": summaries}


def _best_two_signal_conjunction(aggregate, excluded_signals=()):
    """Evaluate a two-gate AND without adopting runtime thresholds."""

    directions = {
        "point_gap_over_spacing": False,
        "pooled_mahalanobis_q0.1": False,
        "symmetric_mahalanobis_q0.1": False,
        "ellipsoid_k2_margin_q0.1": True,
        "directional_reach_ratio_q0.1": False,
        "tangent_reach_ratio_q0.1": False,
        "normal_reach_ratio_q0.1": False,
        "bridge_endpoint_ratio_unweighted_q0.1": True,
        "bridge_endpoint_ratio_opacity_q0.1": True,
        "facing_opacity_mass_per_gap": True,
    }
    rows = aggregate["rows"]
    positives = [row for row in rows if row["positive"]]
    negatives = [row for row in rows if not row["positive"]]
    results = []
    names = sorted(set(directions) - set(excluded_signals))
    for position, signal_a in enumerate(names):
        for signal_b in names[position + 1 :]:
            thresholds = {}
            for signal in (signal_a, signal_b):
                values = [float(row[signal]) for row in positives]
                thresholds[signal] = min(values) if directions[signal] else max(values)

            def passes(row, signal):
                value = float(row[signal])
                return value >= thresholds[signal] if directions[signal] else value <= thresholds[signal]

            false_positive_labels = [
                row["label"]
                for row in negatives
                if passes(row, signal_a) and passes(row, signal_b)
            ]
            results.append(
                {
                    "signals": [signal_a, signal_b],
                    "positive_envelope_thresholds_for_analysis_only": thresholds,
                    "positive_false_negative_count": 0,
                    "negative_false_positive_count": len(false_positive_labels),
                    "negative_false_positive_labels": false_positive_labels,
                    "perfect_separation": not false_positive_labels,
                }
            )
    results.sort(key=lambda item: (item["negative_false_positive_count"], item["signals"]))
    return {
        "method": "AND of two monotonic signals at positive-enclosing boundaries",
        "thresholds_adopted": False,
        "excluded_signals": sorted(excluded_signals),
        "best": results[:10],
        "any_perfect_conjunction": any(item["perfect_separation"] for item in results),
    }


def _hardest_density_pair(points):
    hierarchy, production, graph, decomposition, node_lookup = _case_state(points)
    if decomposition.component_count() == 2:
        region_a, region_b, evaluation = _final_split_pair(decomposition)
        return region_a, region_b, "stage3_final_split"
    graph_edges = {tuple(sorted((edge.cell_a, edge.cell_b))): edge for edge in graph.edges}
    candidates = [
        edge
        for edge in production.edge_decisions
        if edge.reason == "merged"
        and tuple(sorted((edge.leaf_a, edge.leaf_b))) in graph_edges
    ]
    if not candidates:
        raise RuntimeError("density scene has no merged production edge")
    selected = max(
        candidates,
        key=lambda edge: (
            graph_edges[tuple(sorted((edge.leaf_a, edge.leaf_b)))].support_gap,
            tuple(sorted((edge.leaf_a, edge.leaf_b))),
        ),
    )
    return (
        node_lookup[selected.leaf_a].gaussian_indices,
        node_lookup[selected.leaf_b].gaussian_indices,
        "largest_support_gap_production_merged_edge",
    )


def _density_fraction_points(count, seed, fraction):
    generator = torch.Generator().manual_seed(seed)
    dense_count = int(round(count * float(fraction)))
    sparse_count = count - dense_count
    dense = (torch.randn((dense_count, 2), generator=generator) * 0.18).clamp(-1.0, 1.0)
    sparse = torch.rand((sparse_count, 2), generator=generator) * 2.0 - 1.0
    xy = torch.cat([dense, sparse], dim=0)
    x, y = xy[:, 0], xy[:, 1]
    z = 0.20 * torch.sin(2.4 * x) * torch.cos(1.8 * y)
    return torch.stack([x, y, z], dim=1)

def _stability_sweeps(base_records):
    representatives = [
        next(record for record in base_records if record["label"].startswith("density_gradient_seed_2")),
        next(record for record in base_records if record["label"].startswith("disconnected_gap_0.1_seed_1")),
    ]
    rotation = []
    covariance_noise = []
    scale = []
    bridge = []
    for record in representatives:
        diagnostics = record["diagnostics"]
        points = torch.tensor(
            [diagnostics["boundary_pair_metrics"][0]["gaussian_a"]]
        )  # marker only; source tensors are reconstructed below
        label = record["label"]
        if label.startswith("density"):
            scene = make_scene("density_gradient", 600, 2)
            hierarchy, production, graph, decomposition, node_lookup = _case_state(scene.points)
            region_a, region_b, evaluation = _final_split_pair(decomposition)
            means = scene.points
        else:
            means, labels = _disconnected_points(600, 1, 0.1)
            hierarchy, production, graph, decomposition, node_lookup = _case_state(means)
            region_a, region_b, evaluation = _first_false_merge_pair(decomposition, hierarchy, labels, node_lookup)
        covariance, opacity, scales0, rotations0 = _production_initial_fields(means)
        for axis, degrees in (("x", 37.0), ("y", 61.0), ("z", 83.0)):
            matrix = _rotation_matrix(axis, degrees)
            rotated_means = means.double() @ matrix.T
            rotated_covariance = matrix[None] @ covariance @ matrix.T[None]
            rotation.append(_record(f"{label}_rotate_{axis}_{degrees:g}", "stability", record["expected_continuity"], rotated_means, region_a, region_b, rotated_covariance, opacity, record["covariance_source"]))
        generator = torch.Generator().manual_seed(1234)
        for noise in (0.05, 0.15, 0.3):
            factors = torch.exp(torch.randn(scales0.shape, generator=generator, dtype=torch.float64) * noise)
            noisy_covariance = covariance_from_scale_rotation(scales0 * factors, rotations0)
            covariance_noise.append(_record(f"{label}_covariance_noise_{noise:g}", "stability", record["expected_continuity"], means, region_a, region_b, noisy_covariance, opacity, "production_init_scale_noise"))
        for multiplier in (0.5, 1.0, 2.0):
            scaled_covariance, scaled_opacity, _, _ = _production_initial_fields(means, multiplier)
            scale.append(_record(f"{label}_scale_multiplier_{multiplier:g}", "stability", record["expected_continuity"], means, region_a, region_b, scaled_covariance, scaled_opacity, "production_knn_isotropic_scale_sweep"))
        for samples in (17, 33, 65):
            for truncation in (3.0, 4.0, 6.0):
                config = replace(DEFAULT_CONFIG, bridge_sample_count=samples, kernel_truncation_radius=truncation)
                bridge.append(_record(f"{label}_bridge_samples_{samples}_trunc_{truncation:g}", "stability", record["expected_continuity"], means, region_a, region_b, covariance, opacity, record["covariance_source"], config=config))
    density_seed = []
    for seed in range(5):
        scene = make_scene("density_gradient", 600, seed)
        region_a, region_b, source = _hardest_density_pair(scene.points)
        covariance, opacity, _, _ = _production_initial_fields(scene.points)
        density_seed.append(
            _record(
                f"density_gradient_seed_{seed}_hardest_connected_pair",
                "stability_density_seed",
                True,
                scene.points,
                region_a,
                region_b,
                covariance,
                opacity,
                "production_knn_isotropic_identity_rotation_constant_opacity",
                provenance={"pair_source": source},
            )
        )
    density_fraction = []
    for fraction in (0.0, 0.5, 0.7, 0.9):
        means = _density_fraction_points(600, 0, fraction)
        region_a, region_b, source = _hardest_density_pair(means)
        covariance, opacity, _, _ = _production_initial_fields(means)
        density_fraction.append(
            _record(
                f"density_fraction_{fraction:g}_hardest_connected_pair",
                "stability_density_fraction",
                True,
                means,
                region_a,
                region_b,
                covariance,
                opacity,
                "production_knn_isotropic_identity_rotation_constant_opacity",
                provenance={"pair_source": source},
            )
        )
    return {
        "rotation": rotation,
        "covariance_noise": covariance_noise,
        "scale": scale,
        "bridge_bandwidth_samples": bridge,
        "density_seed": density_seed,
        "density_fraction": density_fraction,
    }


def build_report():
    actual = _actual_conflict_records()
    fixtures = _fixture_records()
    stability = _stability_sweeps(actual)
    production_source = "production_knn_isotropic_identity_rotation_constant_opacity"
    pair_payload = {
        "schema_version": 1,
        "stage": "gaussian_native_support_continuity_stage3r",
        "production_integration": False,
        "config": DEFAULT_CONFIG.payload(),
        "field_audit": {
            "torch_gaussian_model": {
                "center": "get_xyz / _xyz",
                "anisotropic_scale": "get_scaling=exp(_scaling), shape (N,3)",
                "rotation": "get_rotation=normalized WXYZ quaternion, shape (N,4)",
                "covariance": "renderer convention Sigma=(S*R)^T(S*R)",
                "opacity": "get_opacity=sigmoid(_opacity)",
                "principal_axis_normal": "smallest covariance eigenvector only when anisotropy is non-ambiguous",
                "color_used": False,
            },
            "synthetic_scene": {
                "available_fields": ["points", "colors", "analytic oracle/GT for evaluation"],
                "missing_fields": ["scale", "rotation", "covariance", "opacity"],
            },
            "boundary_first": {
                "runtime_model": "raw points only; no covariance/opacity fitting",
                "renderer_export_placeholder": "scale exp(-4.6) isotropic, identity rotation, opacity sigmoid(10)",
            },
            "production_initialization": {
                "scale": "nearest-neighbor distance repeated identically on xyz",
                "rotation": "identity quaternion",
                "opacity": 0.12,
                "meaningful_surface_anisotropy": False,
                "post_training_note": "scale/rotation are trainable, but no trained checkpoint is part of this synthetic benchmark",
            },
        },
        "actual_conflict_pairs": actual,
        "diagnostics_covariance_fixtures": fixtures,
        "stability_sweeps": stability,
    }
    actual_production = [
        record for record in actual if record["covariance_source"] == production_source
    ]
    core_conflicts = [
        record
        for record in actual_production
        if record["category"] in {"actual_positive_conflict", "actual_negative_disconnected"}
    ]
    core_distribution = _aggregate(core_conflicts)
    opacity_differences = []
    for record in actual_production:
        bridge_density = record["diagnostics"]["bridge_density"]
        opacity_differences.append(
            abs(
                bridge_density["unweighted"]["endpoint_minimum_ratio"]["median"]
                - bridge_density["opacity_weighted"]["endpoint_minimum_ratio"]["median"]
            )
        )
    stability_signal_rows = {
        key: [
            {
                "label": record["label"],
                "expected_continuity": record["expected_continuity"],
                **_scalar_signals(record),
            }
            for record in values
        ]
        for key, values in stability.items()
    }
    bridge_sensitivity = {}
    for prefix in ("density_gradient", "disconnected"):
        rows = [
            row
            for row in stability_signal_rows["bridge_bandwidth_samples"]
            if row["label"].startswith(prefix)
        ]
        values = [float(row["bridge_endpoint_ratio_unweighted_q0.1"]) for row in rows]
        bridge_sensitivity[prefix] = {
            "minimum": min(values),
            "maximum": max(values),
            "range": max(values) - min(values),
        }
    rotation_normal_reach = {}
    for prefix in ("density_gradient", "disconnected"):
        values = [
            float(row["normal_reach_ratio_q0.1"])
            for row in stability_signal_rows["rotation"]
            if row["label"].startswith(prefix)
        ]
        rotation_normal_reach[prefix] = {
            "minimum": min(values),
            "maximum": max(values),
        }
    summary = {
        "schema_version": 1,
        "stage": "gaussian_native_support_continuity_stage3r_summary",
        "production_integration": False,
        "pair_artifact_hash": _hash(pair_payload),
        "actual_production_init_distribution": _aggregate(actual, production_source),
        "stage3_core_conflict_distribution": core_distribution,
        "two_signal_conjunction_diagnostic": _best_two_signal_conjunction(
            core_distribution,
            excluded_signals=("normal_reach_ratio_q0.1", "tangent_reach_ratio_q0.1"),
        ),
        "actual_field_validity": {
            "record_count": len(actual_production),
            "principal_axis_meaningful_record_count": sum(
                bool(record["diagnostics"]["validity_flags"]["principal_axis_meaningful_for_all"])
                for record in actual_production
            ),
            "all_actual_covariances_isotropic_and_axis_ambiguous": all(
                not bool(record["diagnostics"]["validity_flags"]["principal_axis_meaningful_for_all"])
                for record in actual_production
            ),
            "opacity_weighting_max_endpoint_ratio_difference": max(opacity_differences, default=0.0),
        },
        "all_actual_and_analytic_distribution": _aggregate(actual),
        "fixture_distribution": _aggregate(fixtures),
        "stability_counts": {key: len(value) for key, value in stability.items()},
        "stability_signal_rows": stability_signal_rows,
        "stability_interpretation": {
            "bridge_bandwidth_sample_ranges": bridge_sensitivity,
            "principal_axis_normal_reach_rotation_ranges": rotation_normal_reach,
            "principal_axis_signal_orientation_stable": False,
            "bridge_scale_sensitive": True,
        },
        "deterministic_cost_totals": {
            key: sum(
                record["diagnostics"]["computational_cost"][key]
                for record in actual
            )
            for key in (
                "cross_distance_evaluations",
                "selected_boundary_pair_count",
                "bridge_kernel_evaluations",
            )
        },
    }
    return pair_payload, summary


def _write_csv(path: Path, records):
    rows = []
    for record in records:
        rows.append(
            {
                "label": record["label"],
                "category": record["category"],
                "expected_continuity": record["expected_continuity"],
                "covariance_source": record["covariance_source"],
                **_scalar_signals(record),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-output", type=Path, default=ROOT / "artifacts" / "gaussian_support_continuity_stage3r_pairs.json")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "artifacts" / "gaussian_support_continuity_stage3r_summary.json")
    parser.add_argument("--csv-output", type=Path, default=ROOT / "artifacts" / "gaussian_support_continuity_stage3r_signals.csv")
    args = parser.parse_args()
    pairs, summary = build_report()
    for path in (args.pairs_output, args.summary_output, args.csv_output):
        path.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.pairs_output.resolve().write_text(json.dumps(pairs, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    args.summary_output.resolve().write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    _write_csv(args.csv_output.resolve(), pairs["actual_conflict_pairs"] + pairs["diagnostics_covariance_fixtures"])
    print(args.pairs_output.resolve())
    print(args.summary_output.resolve())
    print(args.csv_output.resolve())


if __name__ == "__main__":
    main()
