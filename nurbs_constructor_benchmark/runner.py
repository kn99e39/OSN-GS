"""Run the real OSN-GS constructor against synthetic Gaussian scenes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig, nurbs_intermediate_payload

from .diagnostics import export as export_construction_diagnostics
from .ground_truth import gt_nurbs_payload
from .metrics import ground_truth_metrics, patch_union_metrics
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
    """

    scene_dir = output_dir / scene.name
    gt_dir = output_dir / f"{scene.name}_gt"
    scene_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    state.model.save_ply(scene_dir / "point_cloud.ply")
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
    return {
        "leaf_state_counts": hierarchy.state_counts(),
        "hierarchy_max_depth": hierarchy.max_depth_reached(),
        "inactive_enclosed_leaf_count": inactive_enclosed,
        "underdetermined_patch_count": sum(1 for p in provenance if p["underdetermined"]),
        "empty_support_mask_count": sum(1 for p in provenance if p["support_mask_empty"]),
        "observations_per_control_min": min((p["observations_per_control"] for p in provenance), default=None),
        "observations_per_control_median": sorted(p["observations_per_control"] for p in provenance)[len(provenance) // 2] if provenance else None,
        "patches": provenance,
    }


def evaluate_scene(
    scene: SyntheticGaussianScene,
    config: TorchPipelineConfig,
    device: str,
    export_dir: Path | None = None,
) -> dict[str, Any]:
    """Construct with the production pipeline and evaluate against scene truth."""

    import time

    pipeline = TorchOSNGSPipeline(config, device=device)
    construct_start = time.perf_counter()
    state = pipeline.initialize(scene.points, scene.colors)
    construct_seconds = time.perf_counter() - construct_start
    diagnostics_dir = export_dir.parent / "NURBS_diagnostics" / scene.name if export_dir is not None else None
    construction_diagnostics = export_construction_diagnostics(state, diagnostics_dir / "construction_diagnostics.json") if diagnostics_dir is not None else []
    if export_dir is not None:
        export_renderer_output(scene, state, export_dir)
    points = state.model.get_xyz.detach()
    anchors = torch.empty_like(points)
    for patch_id, patch in enumerate(state.surface_patches):
        mask = state.model.cluster_ids == patch_id
        if bool(mask.any()):
            anchors[mask] = patch.evaluate(state.model.surface_uv[mask]).detach()
    invalid = (state.model.cluster_ids < 0) | (state.model.cluster_ids >= len(state.surface_patches))
    if bool(invalid.any()):
        anchors[invalid] = state.surface.evaluate(state.model.surface_uv[invalid]).detach()
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
    support_artifact_dir = export_dir.parent / "NURBS_diagnostics" / scene.name if export_dir is not None else None
    support_artifacts = _export_support_artifact(support_raster, support_artifact_dir) if support_artifact_dir is not None else {}
    if support_artifact_dir is not None:
        support_artifacts.update(_export_union_artifact(union_raster, support_artifact_dir))
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
        # Stage 1-D: support topology on the world-space union of all patches.
        "patch_union": union_metrics,
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
        choices=("legacy", "voxel_patch_stage1"),
        default="legacy",
        help="NURBS constructor architecture. 'legacy' is the unchanged production path.",
    )
    parser.add_argument("--voxel-min-count", type=int, default=10, help="[stage1] Minimum raw Gaussian count for an active leaf voxel.")
    parser.add_argument("--voxel-max-count", type=int, default=150, help="[stage1] Raw count above which a voxel subdivides.")
    parser.add_argument("--voxel-max-depth", type=int, default=6, help="[stage1] Maximum subdivision depth.")
    parser.add_argument("--voxel-min-size", type=float, default=0.0, help="[stage1] Minimum voxel edge length (0 disables).")
    parser.add_argument(
        "--stage1-support",
        choices=("voxel", "none"),
        default="voxel",
        help="[stage1] Patch support mask: exact plane-AABB polygon ('voxel') or untrimmed ('none').",
    )
    parser.add_argument("--stage1-obs-per-control", type=float, default=2.0, help="[stage1] Target observations per control point when sizing patch grids.")
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
        **({"surface_trim_resolution": args.trim_resolution} if args.trim_resolution is not None else {}),
        **({"surface_trim_dilation": args.trim_dilation} if args.trim_dilation is not None else {}),
    )
    export_dir = None if args.skip_renderer_export else args.output / "NURBS_output"
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
    report = {"config": asdict(config), "run": vars(args) | {"output": str(args.output)}, "results": results, "failures": failures}
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
    print(f"report={path}")
    if export_dir is not None:
        print(f"renderer output={export_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
