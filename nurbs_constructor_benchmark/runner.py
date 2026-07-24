"""Run the real OSN-GS constructor against synthetic Gaussian scenes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig, nurbs_intermediate_payload

from .boundary_first import construct_boundary_first
from .boundary_first import renderer_payload as boundary_first_renderer_payload
from .diagnostics import export as export_construction_diagnostics
from .ground_truth import gt_nurbs_payload
from .metrics import (
    contour_vs_gt_boundary,
    ground_truth_metrics,
    patch_union_metrics,
    support_boundary_conformality,
)
from .scenes import SCENE_NAMES, SyntheticGaussianScene, make_scene


def _surface_grid(patch: Any, resolution: int = 24) -> torch.Tensor:
    lin = torch.linspace(0.0, 1.0, resolution, device=patch.control_grid.device)
    u, v = torch.meshgrid(lin, lin, indexing="ij")
    return torch.stack([u.reshape(-1), v.reshape(-1)], dim=1)


def export_renderer_output(scene: SyntheticGaussianScene, state: Any, output_dir: Path) -> None:
    """Save the synthetic Gaussian set + constructed NURBS in the renderer's
    expected format (see ``RENDERER_INPUT_FORMAT.md``): a Graphdeco-style
    ``point_cloud.ply`` plus a ``nurbs_surface.json`` sibling, matching the
    layout of a real training run's ``final`` output directory.

    Also writes ``nurbs_surface_gt.json`` -- the ground-truth NURBS in the same
    format -- so the true and reconstructed surfaces can be overlaid in the
    renderer for a direct visual comparison.

    ``<scene>_gt/`` gets the same ``point_cloud.ply`` as ``<scene>/`` so it is
    a self-contained renderer input directory (point cloud + one NURBS JSON),
    loadable on its own instead of only alongside the reconstruction.
    """

    scene_dir = output_dir / scene.name
    gt_dir = output_dir / f"{scene.name}_gt"
    scene_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    state.model.save_ply(scene_dir / "point_cloud.ply")
    state.model.save_ply(gt_dir / "point_cloud.ply")
    payload = nurbs_intermediate_payload(state)
    (scene_dir / "nurbs_surface.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Keep GT in its own renderer-loadable folder so one snapshot has only
    # one NURBS JSON payload.
    (gt_dir / "nurbs_surface.json").write_text(
        json.dumps(gt_nurbs_payload(scene), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _export_support_artifact(raster: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "support_domain.json"
    json_path.write_text(json.dumps(raster, indent=2), encoding="utf-8")
    resolution = int(raster["resolution"])
    panels = []
    for index, key in enumerate(("gt_mask", "generated_mask", "intersection_mask")):
        mask = raster[key]
        pixels = "".join(
            f'<rect x="{index * (resolution + 8) + row}" y="{col}" width="1" height="1"/>'
            for row, values in enumerate(mask) for col, enabled in enumerate(values) if enabled
        )
        panels.append(f'<g fill="#20a4d8">{pixels}</g><text x="{index * (resolution + 8)}" y="{resolution + 12}" fill="#222">{key.replace("_mask", "")}</text>')
    svg_path = output_dir / "support_domain.svg"
    width = 3 * (resolution + 8)
    svg_path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{resolution + 20}" viewBox="0 0 {width} {resolution + 20}"><rect width="100%" height="100%" fill="white"/>' + "".join(panels) + "</svg>", encoding="utf-8")
    return {"support_domain_json": str(json_path), "support_domain_svg": str(svg_path)}


def _export_union_artifact(raster: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Global patch-union / hole diagnostic view (Stage 1-E minimal viewer aid)."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "support_union.json"
    json_path.write_text(json.dumps(raster, indent=2), encoding="utf-8")
    resolution = int(raster["resolution"])
    panels = []
    layers = [
        ("gt_mask", "#8a8f98", "gt"),
        ("union_mask", "#20a4d8", "union"),
        ("gt_hole_mask", "#e0b13c", "gt holes"),
        ("union_hole_mask", "#d84a3c", "union holes"),
        ("overlap_mask", "#7a3cd8", "overlap>=2"),
        ("seam_mask", "#1fbf6b", "seams"),
    ]
    for index, (key, color, label) in enumerate(layers):
        mask = raster[key]
        pixels = "".join(
            f'<rect x="{index * (resolution + 8) + row}" y="{col}" width="1" height="1"/>'
            for row, values in enumerate(mask) for col, enabled in enumerate(values) if enabled
        )
        panels.append(
            f'<g fill="{color}">{pixels}</g>'
            f'<text x="{index * (resolution + 8)}" y="{resolution + 12}" fill="#222">{label}</text>'
        )
    svg_path = output_dir / "support_union.svg"
    width = len(layers) * (resolution + 8)
    svg_path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{resolution + 20}" '
        f'viewBox="0 0 {width} {resolution + 20}"><rect width="100%" height="100%" fill="white"/>'
        + "".join(panels)
        + "</svg>",
        encoding="utf-8",
    )
    return {"support_union_json": str(json_path), "support_union_svg": str(svg_path)}


def _stage1_scene_metrics(scene: SyntheticGaussianScene, state: Any, union_raster: dict[str, Any]) -> dict[str, Any]:
    """Stage 1 hierarchy/leaf/per-patch summary for one scene result."""

    hierarchy = getattr(state, "voxel_hierarchy", None)
    if hierarchy is None:
        return {}
    import torch as _torch

    resolution = int(union_raster["resolution"])
    gt_holes = _torch.tensor(union_raster["gt_hole_mask"], dtype=_torch.bool)
    inactive_enclosed = 0
    for node in hierarchy.leaves_in_state("inactive"):
        center = ((node.aabb_min + node.aabb_max) * 0.5).detach().cpu()
        cell = (((center[:2] + 1.0) * 0.5) * resolution).long().clamp(0, resolution - 1)
        if bool(gt_holes[int(cell[0]), int(cell[1])]):
            inactive_enclosed += 1
    provenance = getattr(state, "stage1_patch_provenance", [])
    refinements = [p["density_refinement"] for p in provenance if p.get("density_refinement")]
    contour_points = [
        endpoint[:2]
        for refinement in refinements
        for segment in refinement["contour_world"]
        for endpoint in segment
    ]
    contour_chamfer, contour_hausdorff = contour_vs_gt_boundary(
        scene,
        _torch.tensor(contour_points) if contour_points else _torch.empty((0, 2)),
        resolution,
    )
    return {
        "leaf_state_counts": hierarchy.state_counts(),
        "hierarchy_max_depth": hierarchy.max_depth_reached(),
        "inactive_enclosed_leaf_count": inactive_enclosed,
        "underdetermined_patch_count": sum(1 for p in provenance if p["underdetermined"]),
        "empty_support_mask_count": sum(1 for p in provenance if p["support_mask_empty"]),
        "observations_per_control_min": min((p["observations_per_control"] for p in provenance), default=None),
        "observations_per_control_median": sorted(p["observations_per_control"] for p in provenance)[len(provenance) // 2] if provenance else None,
        # Stage 1-F boundary refinement summary.
        "boundary_leaf_count": sum(1 for p in provenance if p.get("is_boundary_leaf")),
        "refined_patch_count": len(refinements),
        "refined_boundary_length_world": sum(r["refined_boundary_length_world"] for r in refinements),
        "density_contour_gt_chamfer": contour_chamfer,
        "density_contour_gt_hausdorff": contour_hausdorff,
        "patches": provenance,
    }


def _export_boundary_refinement_artifact(state: Any, output_dir: Path, max_panels: int = 16) -> dict[str, str]:
    """Per-boundary-leaf SVG: density heatmap + threshold contour + coarse polygon
    + refined mask, so the refinement is verifiable without a browser viewer."""

    provenance = [p for p in getattr(state, "stage1_patch_provenance", []) if p.get("density_refinement")]
    if not provenance:
        return {}
    output_dir.mkdir(parents=True, exist_ok=True)
    size, pad = 140, 26
    panels = []
    for slot, patch_provenance in enumerate(provenance[:max_panels]):
        refinement = patch_provenance["density_refinement"]
        origin_x = slot * (size + pad) + pad
        grid = refinement["density_grid"]
        grid_resolution = len(grid)
        level = refinement["threshold_level"]
        peak = max((max(row) for row in grid), default=1.0) or 1.0
        cell = size / grid_resolution
        rects = []
        for i, row in enumerate(grid):
            for j, value in enumerate(row):
                shade = int(235 - 195 * min(1.0, value / peak))
                rects.append(
                    f'<rect x="{origin_x + i * cell:.1f}" y="{pad + (grid_resolution - 1 - j) * cell:.1f}" '
                    f'width="{cell:.2f}" height="{cell:.2f}" fill="rgb({shade},{shade},{shade})"/>'
                )
        contour_lines = "".join(
            f'<line x1="{origin_x + a[0] * size:.1f}" y1="{pad + (1 - a[1]) * size:.1f}" '
            f'x2="{origin_x + b[0] * size:.1f}" y2="{pad + (1 - b[1]) * size:.1f}" '
            'stroke="#d8402f" stroke-width="1.2"/>'
            for a, b in refinement["contour_uv"]
        )
        polygon_uv = patch_provenance.get("support_polygon_uv") or []
        polygon_points = " ".join(
            f"{origin_x + p[0] * size:.1f},{pad + (1 - p[1]) * size:.1f}" for p in polygon_uv
        )
        polygon_svg = (
            f'<polygon points="{polygon_points}" fill="none" stroke="#2b7bd8" stroke-width="1.2"/>'
            if polygon_uv else ""
        )
        label = (
            f'{patch_provenance["source_leaf_voxel_id"]}: '
            f'{refinement["coarse_cells"]}→{refinement["refined_cells"]} cells, '
            f'protected={refinement["protected_cells"]}, borrowed={refinement["borrowed_points"]}'
        )
        panels.append(
            "".join(rects) + contour_lines + polygon_svg
            + f'<text x="{origin_x}" y="{pad + size + 14}" fill="#222" font-size="9">{label}</text>'
        )
    width = min(len(provenance), max_panels) * (size + pad) + pad
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{size + 2 * pad + 20}" '
        f'viewBox="0 0 {width} {size + 2 * pad + 20}"><rect width="100%" height="100%" fill="white"/>'
        '<text x="8" y="14" fill="#222" font-size="10">density heatmap (gray) + threshold contour (red) + coarse voxel polygon (blue)</text>'
        + "".join(panels)
        + "</svg>"
    )
    svg_path = output_dir / "boundary_refinement.svg"
    svg_path.write_text(svg, encoding="utf-8")
    json_path = output_dir / "boundary_refinement.json"
    json_path.write_text(
        json.dumps(
            {
                "patches": [
                    {
                        "patch_id": p["patch_id"],
                        "source_leaf_voxel_id": p["source_leaf_voxel_id"],
                        "boundary_faces": p["boundary_faces"],
                        "density_refinement": p["density_refinement"],
                        "support_polygon_uv": p["support_polygon_uv"],
                    }
                    for p in provenance
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "boundary_refinement_svg": str(svg_path),
        "boundary_refinement_json": str(json_path),
    }


def evaluate_scene(
    scene: SyntheticGaussianScene,
    config: TorchPipelineConfig,
    device: str,
    export_dir: Path | None = None,
) -> dict[str, Any]:
    """Construct with the production pipeline (``legacy``/``voxel_patch_stage1``) and score against scene truth."""

    import time

    pipeline = TorchOSNGSPipeline(config, device=device)
    construct_start = time.perf_counter()
    state = pipeline.initialize(scene.points, scene.colors)
    construct_seconds = time.perf_counter() - construct_start
    if export_dir is not None:
        export_renderer_output(scene, state, export_dir)
    return score_state(scene, state, construct_seconds, export_dir)


def evaluate_scene_boundary_first(
    scene: SyntheticGaussianScene,
    args: argparse.Namespace,
    export_dir: Path | None = None,
) -> dict[str, Any]:
    """Construct with the boundary-first Phase 1-4 pipeline and score against scene truth.

    Reuses ``score_state`` -- the exact same scoring body as
    ``legacy``/``voxel_patch_stage1`` -- so all three constructors land in one
    ``report.json`` with directly comparable numbers, instead of separate
    per-phase report scripts.
    """

    import time

    construct_start = time.perf_counter()
    state, payload_patches = construct_boundary_first(
        scene,
        normal_threshold_degrees=args.bf_normal_threshold_degrees,
        offset_threshold_ratio=args.bf_offset_threshold_ratio,
        boundary_resolution=args.bf_boundary_resolution,
        density_threshold=args.bf_density_threshold,
        coarse_gap_closing_cells=args.bf_coarse_gap_closing_cells,
        annulus_segments=args.bf_annulus_segments,
        annulus_segment_placement=args.bf_annulus_segment_placement,
        annulus_seam_phase_offset=args.bf_seam_phase_offset,
        annulus_hermite_boundary_seed=args.bf_hermite_boundary_seed,
        annulus_coupled_boundary_fit=not args.bf_disable_coupled_boundary_fit,
        candidate_graph_diagnostics=args.bf_candidate_diagnostics,
        candidate_radius_factor=args.bf_candidate_radius_factor,
        candidate_max_neighbors=args.bf_candidate_max_neighbors,
        export_dir=export_dir,
    )
    construct_seconds = time.perf_counter() - construct_start
    if export_dir is not None:
        from .boundary_first import write_point_cloud_ply

        scene_dir = export_dir / scene.name
        gt_dir = export_dir / f"{scene.name}_gt"
        scene_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        write_point_cloud_ply(scene, scene_dir / "point_cloud.ply")
        write_point_cloud_ply(scene, gt_dir / "point_cloud.ply")
        (scene_dir / "nurbs_surface.json").write_text(
            json.dumps(
                boundary_first_renderer_payload(
                    scene.name, payload_patches, [boundary.payload() for boundary in state.patch_boundaries]
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        (gt_dir / "nurbs_surface.json").write_text(
            json.dumps(gt_nurbs_payload(scene), indent=2), encoding="utf-8"
        )
    result = score_state(scene, state, construct_seconds, export_dir=None)
    result["boundary_first"] = {
        "component_count": state.component_count,
        "per_component": state.per_component,
        "patch_boundary_count": len(state.patch_boundaries),
        "patch_boundary_states": {
            boundary_state: sum(1 for boundary in state.patch_boundaries if boundary.state == boundary_state)
            for boundary_state in sorted({boundary.state for boundary in state.patch_boundaries})
        },
        "candidate_graph": state.candidate_graph,
    }
    return result


def score_state(
    scene: SyntheticGaussianScene,
    state: Any,
    construct_seconds: float,
    export_dir: Path | None = None,
) -> dict[str, Any]:
    """Score any duck-typed state (``surface_patches`` + ``model.cluster_ids``/``.surface_uv``/``.get_xyz``)
    against scene truth. Shared by every constructor so numbers are directly comparable."""

    diagnostics_dir = export_dir.parent / "NURBS_diagnostics" / scene.name if export_dir is not None else None
    construction_diagnostics = (
        export_construction_diagnostics(state, diagnostics_dir / "construction_diagnostics.json")
        if diagnostics_dir is not None and getattr(state, "surface_fit_diagnostics", None) is not None
        else []
    )
    points = state.model.get_xyz.detach()
    anchors = torch.empty_like(points)
    for patch_id, patch in enumerate(state.surface_patches):
        mask = state.model.cluster_ids == patch_id
        if bool(mask.any()):
            anchors[mask] = patch.evaluate(state.model.surface_uv[mask]).detach()
    invalid = (state.model.cluster_ids < 0) | (state.model.cluster_ids >= len(state.surface_patches))
    if bool(invalid.any()):
        # Constructors with a single coarse fallback surface (legacy/Stage 1)
        # score unassigned points against it. Boundary-first has no such
        # surface (unassigned points are components the hierarchy simply
        # never covered) -- score them as zero residual so they only show up
        # via `unassigned_gaussians`/`fit_rms_assigned`, not a fabricated
        # fallback-surface distance.
        if getattr(state, "surface", None) is not None:
            anchors[invalid] = state.surface.evaluate(state.model.surface_uv[invalid]).detach()
        else:
            anchors[invalid] = points[invalid]
    fit_distances = (points - anchors).norm(dim=1)

    surface_residuals, normal_errors = [], []
    for patch in state.surface_patches:
        uv = _surface_grid(patch)
        samples = patch.evaluate(uv).detach()
        residual, expected_normals = scene.oracle(samples)
        predicted_normals = patch.normals(uv).detach()
        cosine = (predicted_normals * expected_normals).sum(dim=1).abs().clamp(0.0, 1.0)
        surface_residuals.append(residual.abs())
        normal_errors.append(torch.rad2deg(torch.acos(cosine)))
    residuals = torch.cat(surface_residuals)
    normal_degrees = torch.cat(normal_errors)
    controls = sum(int(patch.control_grid.shape[0] * patch.control_grid.shape[1]) for patch in state.surface_patches)
    gt, support_raster = ground_truth_metrics(scene, state)
    union_metrics, union_raster = patch_union_metrics(scene, state)
    # Stage 1-F: score the coarse polygon-only support with the same union
    # machinery so the density refinement's effect is directly comparable.
    coarse_union = None
    coarse_vs_refined_iou = None
    coarse_masks = getattr(state, "stage1_coarse_masks", [])
    if any(mask is not None for mask in coarse_masks):
        coarse_union, coarse_raster = patch_union_metrics(scene, state, mask_override=coarse_masks)
        refined_mask = torch.tensor(union_raster["union_mask"], dtype=torch.bool)
        coarse_mask = torch.tensor(coarse_raster["union_mask"], dtype=torch.bool)
        union_cells = int((refined_mask | coarse_mask).sum())
        coarse_vs_refined_iou = (
            int((refined_mask & coarse_mask).sum()) / union_cells if union_cells else 1.0
        )
    support_artifact_dir = export_dir.parent / "NURBS_diagnostics" / scene.name if export_dir is not None else None
    support_artifacts = _export_support_artifact(support_raster, support_artifact_dir) if support_artifact_dir is not None else {}
    if support_artifact_dir is not None:
        support_artifacts.update(_export_union_artifact(union_raster, support_artifact_dir))
        support_artifacts.update(_export_boundary_refinement_artifact(state, support_artifact_dir))
    if diagnostics_dir is not None:
        support_artifacts.update({"uv_occupancy_json": str(diagnostics_dir / "uv_support.json"), "uv_occupancy_svg": str(diagnostics_dir / "uv_support.svg")})
    assigned = state.model.cluster_ids >= 0
    assigned_rms = (
        float(fit_distances[assigned].square().mean().sqrt().cpu()) if bool(assigned.any()) else float("nan")
    )
    return {
        "scene": scene.name,
        "description": scene.description,
        "input_gaussians": len(state.model),
        "patches": len(state.surface_patches),
        "control_points": controls,
        "construction_seconds": construct_seconds,
        "unassigned_gaussians": int((~assigned).sum()),
        "fit_rms_assigned": assigned_rms,
        "fit_rms": float(fit_distances.square().mean().sqrt().cpu()),
        "fit_max": float(fit_distances.max().cpu()),
        "surface_chart_rms": float(residuals.square().mean().sqrt().cpu()),
        "surface_chart_max": float(residuals.max().cpu()),
        "normal_mean_degrees": float(normal_degrees.mean().cpu()),
        "normal_p95_degrees": float(torch.quantile(normal_degrees, 0.95).cpu()),
        # Ground-truth NURBS surface metrics, split by construction concern.
        "ground_truth": gt,
        # Boundary-conformal topology ideal: how much of the support boundary
        # is realized as chart edges (GT conformal charts score 1.0).
        "support_conformality": support_boundary_conformality(state),
        # Stage 1-D: support topology on the world-space union of all patches.
        "patch_union": union_metrics,
        # Stage 1-F: same union metrics for the coarse polygon-only masks.
        "patch_union_coarse": coarse_union,
        "coarse_vs_refined_support_iou": coarse_vs_refined_iou,
        "stage1": _stage1_scene_metrics(scene, state, union_raster),
        "finite": bool(
            torch.isfinite(anchors).all()
            and torch.isfinite(residuals).all()
            and torch.isfinite(normal_degrees).all()
            and torch.isfinite(torch.tensor([gt["chamfer_rms"], gt["accuracy_rms"], gt["completeness_rms"]])).all()
        ),
        "patch_diagnostics": construction_diagnostics,
        "construction_diagnostics": str(diagnostics_dir / "construction_diagnostics.json") if diagnostics_dir is not None else None,
        "support_artifacts": support_artifacts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic validation for the production OSN-GS NURBS constructor.")
    parser.add_argument("--scenes", nargs="+", choices=(*SCENE_NAMES, "all"), default=["all"])
    parser.add_argument("--output", type=Path, default=Path("nurbs_constructor_benchmark/results"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--points", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--disable-voxel", action="store_true")
    parser.add_argument("--adaptive-voxel", action="store_true")
    parser.add_argument("--voxel-grid", type=int, default=6)
    parser.add_argument(
        "--constructor",
        choices=("voxel_patch_stage1", "boundary_first"),
        default="boundary_first",
        help=(
            "NURBS constructor architecture. 'boundary_first' (default) runs the Phase 1-4 "
            "component/boundary/topology-routed-chart pipeline "
            "(docs/Urgent_Work/OSN_GS_Final_Boundary_First_NURBS_Direction.md); see the --bf-* options. "
            "'voxel_patch_stage1' is the Stage 1 ablation baseline, kept for comparison only. "
            "The legacy pipeline is retired from this benchmark and cannot be selected here."
        ),
    )
    parser.add_argument("--voxel-min-count", type=int, default=10, help="[stage1] Minimum raw Gaussian count for an active leaf voxel.")
    parser.add_argument("--voxel-max-count", type=int, default=150, help="[stage1] Raw count above which a voxel subdivides.")
    parser.add_argument("--voxel-max-depth", type=int, default=6, help="[stage1] Maximum subdivision depth.")
    parser.add_argument("--voxel-min-size", type=float, default=0.0, help="[stage1] Minimum voxel edge length (0 disables).")
    parser.add_argument(
        "--stage1-support",
        choices=("voxel_density", "voxel", "none"),
        default="voxel_density",
        help="[stage1] Patch support mask: plane-AABB polygon + density-refined boundary ('voxel_density'), polygon only ('voxel'), or untrimmed ('none').",
    )
    parser.add_argument("--stage1-obs-per-control", type=float, default=2.0, help="[stage1] Target observations per control point when sizing patch grids.")
    parser.add_argument("--stage1-density-resolution", type=int, default=32, help="[stage1-F] Boundary-leaf density grid resolution.")
    parser.add_argument("--stage1-density-bandwidth", type=float, default=2.0, help="[stage1-F] Adaptive KDE bandwidth as a multiple of each sample's own UV NN spacing.")
    parser.add_argument("--stage1-density-threshold", type=float, default=2.0, help="[stage1-F] Absolute support level in effective-neighbor units.")
    parser.add_argument("--no-stage1-boundary-refinement", action="store_true", help="[stage1-F] Disable density boundary refinement (voxel_density falls back to polygon-only).")
    parser.add_argument("--bf-normal-threshold-degrees", type=float, default=40.0, help="[boundary_first] Phase 1 leaf-merge normal-angle tolerance.")
    parser.add_argument("--bf-offset-threshold-ratio", type=float, default=0.5, help="[boundary_first] Phase 1 leaf-merge coplanarity tolerance.")
    parser.add_argument("--bf-boundary-resolution", type=int, default=64, help="[boundary_first] Phase 2 per-component UV raster resolution.")
    parser.add_argument("--bf-density-threshold", type=float, default=3.0, help="[boundary_first] Phase 2/3 KDE support threshold, tuned against the rendered/trimmed surface (see docs/worklogs/30).")
    parser.add_argument("--bf-coarse-gap-closing-cells", type=int, default=2, help="[boundary_first] Phase 2 curved-component polygon-reprojection seam-closing dilation.")
    parser.add_argument("--bf-annulus-segments", type=int, default=8, help="[boundary_first] Phase 4 O-grid wedge count for annulus-topology components.")
    parser.add_argument(
        "--bf-annulus-segment-placement",
        choices=("uniform_angle", "outer_radius_weighted_segment_placement", "worst_wedge_optimized", "profile_constrained"),
        default="uniform_angle",
        help="[boundary_first] Phase 4 hardening Step 4: 'uniform_angle' (original), 'outer_radius_weighted_segment_placement' (equal arc length along the outer boundary), 'worst_wedge_optimized' (Step 4-D local coordinate-descent refinement, not adopted -- see docs/worklogs/52), or 'profile_constrained' (Step 4-D re-evaluation with a robust profile-based objective -- see docs/worklogs/53) -- see OSN_GS_Phase4_Hardening_Plan.md.",
    )
    parser.add_argument(
        "--bf-seam-phase-offset", type=float, default=0.0,
        help="[boundary_first] Phase 4 hardening Step 4-B: rotates all uniform_angle seam angles by this many radians, wedge width unchanged (see OSN_GS_Phase4_Hardening_Plan.md).",
    )
    parser.add_argument(
        "--bf-hermite-boundary-seed", action="store_true",
        help="[boundary_first] Phase 4 hardening Step 4-C: cubic-Hermite (shared-slope) Coons boundary seed instead of pure linear -- targets seam continuity only (see docs/worklogs/42).",
    )
    parser.add_argument(
        "--bf-disable-coupled-boundary-fit", action="store_true",
        help="[boundary_first] Phase 5 Step 5-A (production default since 2026-07-22, docs/worklogs/55): pass this to fall back to the pre-Step-5-A independent per-wedge fit instead of the joint shared-seam-boundary solve -- see docs/Urgent_Work/OSN_GS_Phase5_Boundary_Aligned_Extension_Plan.md.",
    )
    parser.add_argument(
        "--bf-candidate-diagnostics",
        action="store_true",
        help="[boundary_first] Emit diagnostics-only Stage 2 spatial candidate graph; never changes component membership.",
    )
    parser.add_argument(
        "--bf-candidate-radius-factor",
        type=float,
        default=0.25,
        help="[boundary_first diagnostics] Scale-aware AABB radius as a multiple of max adaptive leaf diagonal.",
    )
    parser.add_argument(
        "--bf-candidate-max-neighbors",
        type=int,
        default=0,
        help="[boundary_first diagnostics] Deterministic per-node degree cap; 0 keeps all candidates for recall analysis.",
    )
    parser.add_argument("--resolution-u", type=int, default=8)
    parser.add_argument("--resolution-v", type=int, default=4)
    parser.add_argument("--fit-mode", choices=("lsq", "idw"), default="lsq")
    parser.add_argument("--trim-resolution", type=int, default=None, help="UV trim mask resolution per patch. 0 disables trimming; omit to use the pipeline default.")
    parser.add_argument("--trim-dilation", type=int, default=None, help="UV trim mask dilation (cells) to close gaps; omit to use the pipeline default.")
    parser.add_argument("--max-fit-rms", type=float, default=None, help="Fail if a scene's input-point RMS exceeds this value.")
    parser.add_argument("--max-chart-rms", type=float, default=None, help="Fail if a scene's sampled chart RMS exceeds this value.")
    # Ground-truth NURBS gates, one per construction concern.
    parser.add_argument("--max-chamfer-rms", type=float, default=None, help="[accuracy] Fail if a scene's GT chamfer RMS exceeds this value.")
    parser.add_argument("--max-extrapolation", type=float, default=None, help="[support] Fail if the generated-surface fraction beyond the data exceeds this value.")
    parser.add_argument("--min-topology-ari", type=float, default=None, help="[topology] Fail if the patch-label ARI against GT falls below this value.")
    parser.add_argument(
        "--skip-renderer-export",
        action="store_true",
        help="Skip writing NURBS_output/<scene>/{point_cloud.ply,nurbs_surface.json} for the 3DGS_Renderer.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested but CUDA is unavailable.")
    names = list(SCENE_NAMES) if "all" in args.scenes else args.scenes
    export_dir = None if args.skip_renderer_export else args.output / "NURBS_output"
    if args.constructor == "boundary_first":
        results = [
            evaluate_scene_boundary_first(make_scene(name, args.points, args.seed, args.noise_std), args, export_dir)
            for name in names
        ]
    else:
        config = TorchPipelineConfig(
            base_curve_count=4,
            visible_surface_resolution_u=max(2, args.resolution_u),
            visible_surface_resolution_v=max(2, args.resolution_v),
            surface_fit_mode=args.fit_mode,
            use_voxel_surface_regions=not args.disable_voxel,
            voxel_grid_resolution=max(2, args.voxel_grid),
            adaptive_voxel_density=args.adaptive_voxel,
            voxel_max_subdivision_depth=1 if args.adaptive_voxel else 0,
            max_surface_control_points=4096,
            nurbs_constructor_mode=args.constructor,
            voxel_min_gaussian_count=max(1, args.voxel_min_count),
            voxel_max_gaussian_count=max(1, args.voxel_max_count),
            voxel_max_depth=max(0, args.voxel_max_depth),
            voxel_min_size=max(0.0, args.voxel_min_size),
            stage1_support_mode=args.stage1_support,
            stage1_observations_per_control=max(0.1, args.stage1_obs_per_control),
            stage1_boundary_refinement_enabled=not args.no_stage1_boundary_refinement,
            stage1_boundary_density_resolution=max(4, args.stage1_density_resolution),
            stage1_boundary_density_bandwidth=max(0.1, args.stage1_density_bandwidth),
            stage1_boundary_density_threshold=max(0.0, args.stage1_density_threshold),
            **({"surface_trim_resolution": args.trim_resolution} if args.trim_resolution is not None else {}),
            **({"surface_trim_dilation": args.trim_dilation} if args.trim_dilation is not None else {}),
        )
        results = [
            evaluate_scene(make_scene(name, args.points, args.seed, args.noise_std), config, args.device, export_dir)
            for name in names
        ]
    failures = [
        result["scene"] for result in results
        if not result["finite"]
        or (args.max_fit_rms is not None and result["fit_rms"] > args.max_fit_rms)
        or (args.max_chart_rms is not None and result["surface_chart_rms"] > args.max_chart_rms)
        or (args.max_chamfer_rms is not None and result["ground_truth"]["chamfer_rms"] > args.max_chamfer_rms)
        or (args.max_extrapolation is not None and result["ground_truth"]["support_extrapolation_fraction"] > args.max_extrapolation)
        or (args.min_topology_ari is not None and result["ground_truth"]["topology_label_ari"] < args.min_topology_ari)
    ]
    report = {
        "config": asdict(config) if args.constructor != "boundary_first" else None,
        "run": vars(args) | {"output": str(args.output)},
        "results": results,
        "failures": failures,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for result in results:
        gt = result["ground_truth"]
        union = result["patch_union"]
        print(
            f"{result['scene']}: patches={result['patches']}(gt {gt['topology_gt_patch_count']}) "
            f"controls={result['control_points']} "
            f"| [accuracy] chamfer_rms={gt['chamfer_rms']:.6f} acc_rms={gt['accuracy_rms']:.6f} "
            f"| [support] uncovered={gt['support_coverage_uncovered_fraction']:.3f} "
            f"extrapolation_global={gt['support_extrapolation_fraction']:.3f} "
            f"extrapolation_local={gt['support_extrapolation_fraction_local']:.3f} iou={gt['support_iou']:.3f} "
            f"| [topology] ari={gt['topology_label_ari']:.3f} "
            f"| [union] iou={union['union_iou']:.3f} holes={union['union_hole_count']}"
            f"(gt {union['union_gt_hole_count']}) hole_iou={union['union_hole_iou']:.3f} "
            f"false_fill={union['union_false_fill_ratio']:.3f} "
            f"overlap={union['union_patch_overlap_ratio']:.3f} gap={union['union_interpatch_gap_ratio']:.3f}"
        )
        if result.get("stage1"):
            stage1 = result["stage1"]
            print(
                f"  stage1: leaves={stage1['leaf_state_counts']} "
                f"inactive_enclosed={stage1['inactive_enclosed_leaf_count']} "
                f"underdetermined={stage1['underdetermined_patch_count']} "
                f"empty_masks={stage1['empty_support_mask_count']} "
                f"construction={result['construction_seconds']:.2f}s"
            )
        if result.get("boundary_first"):
            bf = result["boundary_first"]
            topologies = [c["topology"] for c in bf["per_component"]]
            print(f"  boundary_first: components={bf['component_count']} topologies={topologies}")
            if bf.get("candidate_graph"):
                graph = bf["candidate_graph"]
                recall = graph["reference_recall"]["existing_face_smooth"]["recall"]
                degree = graph["degree"]
                print(
                    f"  candidate_graph: nodes={graph['node_count']} edges={graph['edge_count']} "
                    f"degree_mean={degree['mean']:.2f} p95={degree['p95']} max={degree['max']} "
                    f"face_smooth_recall={recall:.3f} sources={graph['candidate_source_counts']}"
                )
            for c in bf["per_component"]:
                if c["chart"] == "o_grid":
                    cq = c["chart_quality"]
                    print(
                        f"    c{c['component_id']} [o_grid x{c['segments']}]: "
                        f"mean_seam_gap={c['mean_seam_gap']:.5f} max_seam_gap={c['max_seam_gap']:.5f} "
                        f"near_degenerate={c['topology_checks']['near_degenerate_slice_count']} "
                        f"seam_tangent_deg={cq['seams']['tangent_angle_deg_mean']:.2f} "
                        f"seam_normal_deg={cq['seams']['normal_angle_deg_mean']:.2f} "
                        f"jacobian_cond_p95~{cq['jacobian']['jacobian_condition_max_of_slice_p95']:.2f} "
                        f"orientation_flips={cq['jacobian']['total_orientation_flip_samples']}"
                    )
                else:
                    fm = c["fit_metrics"]
                    print(f"    c{c['component_id']} [{c['chart']}]: rms={fm['point_to_surface_rms']:.6f} jacobian_min={fm['jacobian_min']:.4f}")
    print(f"report={path}")
    if export_dir is not None:
        print(f"renderer output={export_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
