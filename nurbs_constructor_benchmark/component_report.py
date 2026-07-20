"""Phase 1 Surface-Cell Component Builder report.

Runs the Stage 1 raw-count voxel hierarchy + the new Phase 1 component builder
(``osn_gs/surface/torch_surface_components.py``) against synthetic scenes and
reports the values required by
``OSN_GS_Final_Boundary_First_NURBS_Direction.md`` §15/§14 (Phase 1 gate):
voxel leaf count, component count, per-component Gaussian/member-leaf counts,
GT component count, component-assignment ARI, merge/split error counts,
runtime, and failure cases.

This is analysis-only: it does not fit NURBS geometry, does not touch the
``legacy``/``voxel_patch_stage1`` constructors, and writes nothing the
trainer/ADC path consumes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from osn_gs.surface.torch_surface_components import (
    build_surface_components,
    surface_component_set_payload,
)
from osn_gs.surface.torch_voxel_hierarchy import (
    build_voxel_gaussian_hierarchy,
    hierarchy_payload,
    validate_hierarchy_conservation,
)

from .metrics import adjusted_rand_index
from .scenes import SyntheticGaussianScene, make_scene

REQUIRED_SCENES = ("plane", "sine", "planar_hole", "crease", "close_parallel_sheets", "density_gradient")


def evaluate_scene_components(
    scene: SyntheticGaussianScene,
    voxel_min_gaussian_count: int = 10,
    voxel_max_gaussian_count: int = 150,
    voxel_max_depth: int = 6,
    voxel_min_size: float = 0.0,
    fit_complex_leaves: bool = True,
    normal_threshold_degrees: float = 40.0,
    offset_threshold_ratio: float = 0.5,
    export_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the hierarchy + components for one scene and report Phase 1 stats."""

    hierarchy_start = time.perf_counter()
    hierarchy = build_voxel_gaussian_hierarchy(
        scene.points,
        voxel_min_gaussian_count=voxel_min_gaussian_count,
        voxel_max_gaussian_count=voxel_max_gaussian_count,
        voxel_max_depth=voxel_max_depth,
        voxel_min_size=voxel_min_size,
    )
    validate_hierarchy_conservation(hierarchy)
    hierarchy_seconds = time.perf_counter() - hierarchy_start

    component_start = time.perf_counter()
    component_set = build_surface_components(
        hierarchy,
        scene.points,
        fit_complex_leaves=fit_complex_leaves,
        normal_threshold_degrees=normal_threshold_degrees,
        offset_threshold_ratio=offset_threshold_ratio,
    )
    component_seconds = time.perf_counter() - component_start

    count = int(scene.points.shape[0])
    predicted = torch.full((count,), -1, dtype=torch.long)
    for leaf in hierarchy.leaves():
        component_id = component_set.leaf_component_id.get(leaf.node_id)
        if component_id is not None and leaf.gaussian_indices is not None:
            predicted[leaf.gaussian_indices] = component_id
    unassigned = int((predicted < 0).sum())
    # Matches the existing topology_label_ari convention (ground_truth_metrics
    # in metrics.py): unassigned Gaussians are folded into cluster 0 rather
    # than excluded, so the fraction above must be read alongside the ARI.
    ari = adjusted_rand_index(predicted.clamp_min(0), scene.gt_patch_label(scene.points))

    reasons = component_set.edge_reason_counts()
    leaf_state_counts = hierarchy.state_counts()
    component_sizes = [int(c.gaussian_indices.numel()) for c in component_set.components]
    member_leaf_counts = [len(c.member_leaf_ids) for c in component_set.components]

    failure_cases = []
    if reasons.get("missing_plane", 0) > 0:
        failure_cases.append(
            f"{reasons['missing_plane']} candidate edges skipped: leaf plane unavailable (<3 points)"
        )
    singleton_components = sum(1 for size in member_leaf_counts if size == 1)
    if singleton_components:
        failure_cases.append(f"{singleton_components} component(s) made of a single leaf (no merge partner found)")
    if component_set.component_count() == 0 and leaf_state_counts.get("active", 0) + leaf_state_counts.get("complex", 0) > 0:
        failure_cases.append("mergeable leaves exist but produced zero components (unexpected)")

    result = {
        "scene": scene.name,
        "input_gaussians": count,
        "voxel_leaf_count": len(hierarchy.leaves()),
        "leaf_state_counts": leaf_state_counts,
        "component_count": component_set.component_count(),
        "gt_component_count": int(scene.gt_patch_count),
        "component_count_delta": component_set.component_count() - int(scene.gt_patch_count),
        "component_gaussian_counts": component_sizes,
        "component_member_leaf_counts": member_leaf_counts,
        "unassigned_gaussians": unassigned,
        "unassigned_fraction": unassigned / count if count else 0.0,
        "component_assignment_ari": ari,
        "edge_reason_counts": reasons,
        "merge_edge_count": reasons.get("merged", 0),
        "split_edge_count": reasons.get("normal", 0) + reasons.get("offset", 0),
        "missing_plane_edge_count": reasons.get("missing_plane", 0),
        "hierarchy_construction_seconds": hierarchy_seconds,
        "component_construction_seconds": component_seconds,
        "failure_cases": failure_cases,
    }

    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "hierarchy.json").write_text(
            json.dumps(hierarchy_payload(hierarchy), indent=2), encoding="utf-8"
        )
        (export_dir / "components.json").write_text(
            json.dumps(surface_component_set_payload(component_set), indent=2), encoding="utf-8"
        )
        result["hierarchy_export"] = str(export_dir / "hierarchy.json")
        result["components_export"] = str(export_dir / "components.json")

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", choices=(*REQUIRED_SCENES, "all"), default=["all"])
    parser.add_argument("--output", type=Path, default=Path("nurbs_constructor_benchmark/results/phase1_components"))
    parser.add_argument("--points", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--voxel-min-count", type=int, default=10)
    parser.add_argument("--voxel-max-count", type=int, default=150)
    parser.add_argument("--voxel-max-depth", type=int, default=6)
    parser.add_argument("--normal-threshold-degrees", type=float, default=40.0)
    parser.add_argument("--offset-threshold-ratio", type=float, default=0.5)
    parser.add_argument("--skip-export", action="store_true", help="Skip writing per-scene hierarchy/component JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    names = list(REQUIRED_SCENES) if "all" in args.scenes else args.scenes
    results = []
    for name in names:
        scene = make_scene(name, args.points, args.seed)
        export_dir = None if args.skip_export else args.output / name
        result = evaluate_scene_components(
            scene,
            voxel_min_gaussian_count=args.voxel_min_count,
            voxel_max_gaussian_count=args.voxel_max_count,
            voxel_max_depth=args.voxel_max_depth,
            normal_threshold_degrees=args.normal_threshold_degrees,
            offset_threshold_ratio=args.offset_threshold_ratio,
            export_dir=export_dir,
        )
        results.append(result)
        print(
            f"{result['scene']}: leaves={result['voxel_leaf_count']} "
            f"components={result['component_count']}(gt {result['gt_component_count']}) "
            f"ari={result['component_assignment_ari']:.3f} "
            f"unassigned={result['unassigned_fraction']:.3f} "
            f"merge={result['merge_edge_count']} split={result['split_edge_count']} "
            f"sizes={result['component_gaussian_counts']} "
            f"time={result['hierarchy_construction_seconds'] + result['component_construction_seconds']:.3f}s"
        )
        for failure in result["failure_cases"]:
            print(f"  failure: {failure}")

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps({"run": vars(args) | {"output": str(args.output)}, "results": results}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
