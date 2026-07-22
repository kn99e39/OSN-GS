from __future__ import annotations

"""Generate Stage 1 proxy-decomposition diagnostics without changing production.

The output compares independent proxy/support/layer signals on current
voxel-leaf pairs.  It deliberately does not define a merge threshold or call
the production component builder with proxy-based membership.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from nurbs_constructor_benchmark.scenes import make_scene
from osn_gs.surface.torch_surface_components import build_surface_components
from osn_gs.surface.torch_surface_proxy import merge_proxy_diagnostics
from osn_gs.surface.torch_voxel_hierarchy import (
    STATE_ACTIVE,
    STATE_COMPLEX,
    build_voxel_gaussian_hierarchy,
)


ROOT = Path(__file__).resolve().parents[2]


def _hierarchy_and_components(scene_name: str, count: int, seed: int):
    scene = make_scene(scene_name, count=count, seed=seed)
    hierarchy = build_voxel_gaussian_hierarchy(
        scene.points,
        voxel_min_gaussian_count=10,
        voxel_max_gaussian_count=150,
        voxel_max_depth=6,
    )
    components = build_surface_components(hierarchy, scene.points)
    leaves = {
        leaf.node_id: leaf
        for leaf in hierarchy.leaves()
        if leaf.state in {STATE_ACTIVE, STATE_COMPLEX}
    }
    return scene, hierarchy, components, leaves


def _leaf_pair_payload(
    category: str,
    scene_name: str,
    scene: Any,
    leaves: dict[str, Any],
    leaf_a_id: str,
    leaf_b_id: str,
    source: str,
) -> dict[str, Any]:
    leaf_a, leaf_b = leaves[leaf_a_id], leaves[leaf_b_id]
    diagnostics = merge_proxy_diagnostics(
        scene.points[leaf_a.gaussian_indices],
        scene.points[leaf_b.gaussian_indices],
    )
    payload = diagnostics.payload()
    payload.update(
        {
            "category": category,
            "scene": scene_name,
            "source": source,
            "leaf_a": leaf_a_id,
            "leaf_b": leaf_b_id,
            "leaf_a_count": int(leaf_a.count),
            "leaf_b_count": int(leaf_b.count),
            "merged_normalized_rms": diagnostics.merged_proxy.normalized_rms_residual,
            "merged_plane_normalized_rms": diagnostics.merged_proxy.plane_normalized_rms_residual,
            "quadratic_to_plane_rms_ratio": diagnostics.merged_proxy.normalized_rms_residual
            / max(diagnostics.merged_proxy.plane_normalized_rms_residual, 1e-15),
            "merged_residual_concentration": diagnostics.merged_proxy.residual_concentration,
            "merged_condition_number": diagnostics.merged_proxy.condition_number,
        }
    )
    return payload


def _aabb_distance(leaf_a: Any, leaf_b: Any) -> float:
    delta = torch.maximum(
        leaf_a.aabb_min - leaf_b.aabb_max,
        leaf_b.aabb_min - leaf_a.aabb_max,
    ).clamp_min(0)
    return float(torch.linalg.norm(delta))


def _current_leaf_pairs(count: int, seed: int) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    selections = (
        ("mild_curved_merged", "mild_curved_sheet", "merged"),
        ("curved_annulus_existing_merged", "curved_annulus", "merged"),
        ("crease_rejected", "crease", "normal"),
        ("parallel_layers_rejected", "close_parallel_sheets", "offset"),
        ("density_gradient_merged", "density_gradient", "merged"),
        ("sparse_boundary_merged", "planar_hole_density_gradient", "merged"),
    )
    cached: dict[str, tuple[Any, Any, Any, dict[str, Any]]] = {}
    for category, scene_name, reason in selections:
        current = _hierarchy_and_components(scene_name, count, seed)
        cached[scene_name] = current
        scene, _, components, leaves = current
        decisions = [edge for edge in components.edge_decisions if edge.reason == reason]
        for edge in decisions:
            pairs.append(
                _leaf_pair_payload(
                    category,
                    scene_name,
                    scene,
                    leaves,
                    edge.leaf_a,
                    edge.leaf_b,
                    f"face_edge:{reason}",
                )
            )

    scene, _, components, leaves = cached["curved_annulus"]
    face_pairs = {
        tuple(sorted((edge.leaf_a, edge.leaf_b))) for edge in components.edge_decisions
    }
    cross_component = []
    leaf_ids = sorted(leaves)
    for position, leaf_a_id in enumerate(leaf_ids):
        component_a = components.leaf_component_id[leaf_a_id]
        leaf_a = leaves[leaf_a_id]
        for leaf_b_id in leaf_ids[position + 1 :]:
            if components.leaf_component_id[leaf_b_id] == component_a:
                continue
            pair_id = tuple(sorted((leaf_a_id, leaf_b_id)))
            if pair_id in face_pairs:
                continue
            leaf_b = leaves[leaf_b_id]
            aabb_distance = _aabb_distance(leaf_a, leaf_b)
            if aabb_distance > 1e-8:
                continue
            centroid_distance = float(
                torch.linalg.norm(leaf_a.plane.centroid - leaf_b.plane.centroid)
            )
            cross_component.append((centroid_distance, pair_id))
    for _, (leaf_a_id, leaf_b_id) in sorted(cross_component)[:4]:
        pairs.append(
            _leaf_pair_payload(
                "curved_annulus_missing_cross_component",
                "curved_annulus",
                scene,
                leaves,
                leaf_a_id,
                leaf_b_id,
                "aabb_touch_without_face_contact",
            )
        )
    return pairs


def _grid(x0: float, x1: float, nx: int = 12, ny: int = 10):
    x = torch.linspace(x0, x1, nx)
    y = torch.linspace(-0.6, 0.6, ny)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return xx.reshape(-1), yy.reshape(-1)


def _synthetic_pair_payload(
    category: str, points_a: torch.Tensor, points_b: torch.Tensor
) -> dict[str, Any]:
    payload = merge_proxy_diagnostics(points_a, points_b).payload()
    payload.update(
        {
            "category": category,
            "scene": "diagnostic_control",
            "source": "analytic_pair",
            "leaf_a": "analytic_a",
            "leaf_b": "analytic_b",
            "leaf_a_count": int(points_a.shape[0]),
            "leaf_b_count": int(points_b.shape[0]),
            "merged_normalized_rms": payload["merged_proxy"]["normalized_rms_residual"],
            "merged_plane_normalized_rms": payload["merged_proxy"]["plane_normalized_rms_residual"],
            "quadratic_to_plane_rms_ratio": payload["merged_proxy"]["normalized_rms_residual"]
            / max(payload["merged_proxy"]["plane_normalized_rms_residual"], 1e-15),
            "merged_residual_concentration": payload["merged_proxy"]["residual_concentration"],
            "merged_condition_number": payload["merged_proxy"]["condition_number"],
        }
    )
    return payload


def _analytic_controls() -> list[dict[str, Any]]:
    xa, ya = _grid(-1.0, -0.45)
    xb, yb = _grid(0.45, 1.0)
    disconnected_a = torch.stack([xa, ya, torch.zeros_like(xa)], dim=1)
    disconnected_b = torch.stack([xb, yb, torch.zeros_like(xb)], dim=1)

    xa, ya = _grid(-1.0, 0.0)
    xb, yb = _grid(0.0, 1.0)
    high_curvature_a = torch.stack(
        [xa, ya, 0.55 * xa.square() + 0.25 * ya.square()], dim=1
    )
    high_curvature_b = torch.stack(
        [xb, yb, 0.55 * xb.square() + 0.25 * yb.square()], dim=1
    )
    return [
        _synthetic_pair_payload(
            "disconnected_coplanar_control", disconnected_a, disconnected_b
        ),
        _synthetic_pair_payload(
            "high_curvature_smooth_control", high_curvature_a, high_curvature_b
        ),
    ]


SUMMARY_FIELDS = (
    "normalized_error_increase",
    "merged_to_child_rms_ratio",
    "merged_normalized_rms",
    "quadratic_to_plane_rms_ratio",
    "merged_residual_concentration",
    "merged_condition_number",
    "scale_normalized_support_gap",
    "normal_angle_degrees",
    "normal_change_rate",
    "layer_separation_score",
)


def _summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        by_category.setdefault(pair["category"], []).append(pair)
    result: dict[str, Any] = {}
    for category, items in sorted(by_category.items()):
        category_summary: dict[str, Any] = {"pair_count": len(items)}
        for field in SUMMARY_FIELDS:
            values = sorted(float(item[field]) for item in items if math.isfinite(float(item[field])))
            if not values:
                category_summary[field] = None
                continue
            category_summary[field] = {
                "min": values[0],
                "median": values[len(values) // 2],
                "max": values[-1],
            }
        result[category] = category_summary
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "proxy_decomposition_stage1.json",
    )
    args = parser.parse_args()

    pairs = _current_leaf_pairs(args.count, args.seed) + _analytic_controls()
    report = {
        "schema_version": 1,
        "stage": "proxy_decomposition_stage1_diagnostics",
        "production_changed": False,
        "config": {
            "count": args.count,
            "seed": args.seed,
            "proxy_regularization": 1e-6,
            "support_gap_quantile": 0.02,
        },
        "summary": _summary(pairs),
        "pairs": pairs,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
