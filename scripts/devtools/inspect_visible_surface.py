from __future__ import annotations

"""Inspect the Stage 1 visible NURBS surface built from a dataset's initial Gaussians.

This tool skips image loading entirely -- it only reads `points3D` from the COLMAP
sparse reconstruction, places initial (certain) Gaussians at those points, runs the
voxel bootstrap + NURBS fit exactly like `train.py` does at iteration 0, and then
writes:

  <output>/renderer_snapshot.json       -- same schema TorchOSNGSTrainer streams,
                                            for renderer inspection without any
                                            training.
  <output>/surface_quality.json         -- per-patch normalized point-to-surface
                                            residual (foot-point projection), for
                                            judging fit quality without a renderer.
  <output>/surface_quality.txt          -- human-readable summary of the above.

Usage:
    osn-gs inspect-surface

The default dataset and output paths, and all fitting defaults, intentionally
match the local OSN-GS branch of ``colab_train_3dgs.ipynb``:
``DATASET`` and ``output/osn_gs_scene/inspect-surface``.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.data.colmap_scene import read_colmap_points3d
from osn_gs.interop.colab_args import (
    add_stage1_constructor_arguments,
    add_surface_fit_arguments,
    stage1_constructor_config_kwargs,
    surface_fit_config_kwargs,
)
from osn_gs.utils.torch_ops import default_device, require_torch, sh_dc_to_rgb


_ROOT = Path(__file__).resolve().parents[2]
_NOTEBOOK_DATA_ROOT = _ROOT / "DATASET"
_NOTEBOOK_MODEL_ROOT = _ROOT / "output" / "osn_gs_scene"
_DEFAULT_OUTPUT_DIR = _NOTEBOOK_MODEL_ROOT / "inspect-surface"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and inspect the Stage 1 visible NURBS surface from a dataset's sparse points only."
    )
    parser.add_argument(
        "-s",
        "--source_path",
        default=str(_NOTEBOOK_DATA_ROOT),
        help="COLMAP scene root with sparse/0/points3D.* (default: notebook DATA_ROOT local path).",
    )
    parser.add_argument("--sparse_dir", type=str, default="sparse/0", help="Sparse COLMAP folder under --source_path.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(_DEFAULT_OUTPUT_DIR),
        help="Inspection artifact directory (default: notebook MODEL_ROOT/inspect-surface).",
    )
    parser.add_argument("--device", type=str, default="", help="cuda, cpu, or empty for auto.")
    parser.add_argument("--max_points", type=int, default=0, help="Optional cap on sparse points used. 0 uses all.")

    parser.add_argument("--base_curve_count", type=int, default=8)
    parser.add_argument("--visible_surface_resolution_u", type=int, default=8)
    parser.add_argument("--visible_surface_resolution_v", type=int, default=4)
    parser.add_argument("--visible_surface_resolution_scale", type=float, default=4.0, help="Notebook Train default: 4.0.")
    parser.add_argument("--max_surface_control_points", type=int, default=65536)
    parser.add_argument("--visible_surface_fit_device", type=str, default=default_device(prefer_cuda=True), choices=("cpu", "cuda", "auto"), help="Notebook Train default: active training device.")
    parser.add_argument("--visible_surface_fit_chunk_size", type=int, default=0)
    add_surface_fit_arguments(parser)

    parser.add_argument("--disable_voxel_surface_regions", action="store_true")
    parser.add_argument("--voxel_grid_resolution", type=int, default=16)
    parser.add_argument("--disable_adaptive_voxel_density", action="store_true")
    parser.add_argument("--voxel_max_subdivision_depth", type=int, default=1)
    parser.add_argument("--voxel_density_quantile", type=float, default=0.75)
    parser.add_argument("--voxel_density_covariance_weight_cap", type=float, default=10.0)
    parser.add_argument("--voxel_normal_knn", type=int, default=16)
    parser.add_argument("--voxel_boundary_angle_degrees", type=float, default=35.0)
    parser.add_argument("--voxel_min_points_per_region", type=int, default=1)
    parser.add_argument("--voxel_normal_chunk_size", type=int, default=4096)
    add_stage1_constructor_arguments(parser)

    parser.add_argument("--covariance_init", type=str, default="knn", choices=("knn", "constant"))
    parser.add_argument("--covariance_knn_chunk_size", type=int, default=0)
    parser.add_argument("--covariance_min_scale", type=float, default=1e-4)
    parser.add_argument("--covariance_max_scale_ratio", type=float, default=0.05)
    parser.add_argument("--covariance_scale_multiplier", type=float, default=1.0)
    return parser


def build_pipeline_config(args: argparse.Namespace) -> TorchPipelineConfig:
    return TorchPipelineConfig(
        base_curve_count=args.base_curve_count,
        visible_surface_resolution_u=args.visible_surface_resolution_u,
        visible_surface_resolution_v=args.visible_surface_resolution_v,
        visible_surface_resolution_scale=args.visible_surface_resolution_scale,
        max_surface_control_points=max(4, int(args.max_surface_control_points)),
        visible_surface_fit_device=args.visible_surface_fit_device,
        visible_surface_fit_chunk_size=args.visible_surface_fit_chunk_size,
        use_voxel_surface_regions=not args.disable_voxel_surface_regions,
        voxel_grid_resolution=args.voxel_grid_resolution,
        adaptive_voxel_density=not args.disable_adaptive_voxel_density,
        voxel_max_subdivision_depth=max(0, int(args.voxel_max_subdivision_depth)),
        voxel_density_quantile=min(1.0, max(0.0, float(args.voxel_density_quantile))),
        voxel_density_covariance_weight_cap=max(0.1, float(args.voxel_density_covariance_weight_cap)),
        voxel_normal_knn=args.voxel_normal_knn,
        voxel_boundary_angle_degrees=args.voxel_boundary_angle_degrees,
        voxel_min_points_per_region=args.voxel_min_points_per_region,
        voxel_normal_chunk_size=args.voxel_normal_chunk_size,
        covariance_init=args.covariance_init,
        covariance_knn_chunk_size=args.covariance_knn_chunk_size,
        covariance_min_scale=args.covariance_min_scale,
        covariance_max_scale_ratio=args.covariance_max_scale_ratio,
        covariance_scale_multiplier=args.covariance_scale_multiplier,
        **surface_fit_config_kwargs(args),
        **stage1_constructor_config_kwargs(args),
    )


def voxel_regions_payload(regions) -> dict | None:
    if regions is None:
        return None
    centers = regions.region_centers.detach().cpu()
    normals = regions.region_normals.detach().cpu()
    boundary = regions.boundary_mask.detach().cpu()
    voxel_indices = regions.voxel_indices.detach().cpu()
    return {
        "type": "voxel_surface_regions",
        "count": int(centers.shape[0]),
        "boundary_count": int(boundary.sum().item()),
        "centers": centers.reshape(-1).tolist(),
        "normals": normals.reshape(-1).tolist(),
        "boundary_mask": boundary.tolist(),
        "voxel_indices": voxel_indices.reshape(-1).tolist(),
        "region_patch_ids": regions.region_patch_ids.detach().cpu().tolist(),
        "region_levels": regions.region_levels.detach().cpu().tolist(),
        "region_density": regions.region_density.detach().cpu().tolist(),
        "region_bounds": regions.region_bounds.detach().cpu().reshape(-1).tolist(),
        "flattened": True,
    }


def build_stream_payload(state, quality_report: dict) -> dict:
    torch = require_torch()
    model = state.model
    with torch.no_grad():
        xyz = model.get_xyz.detach().float().cpu()
        scaling = model.get_scaling.detach().float().cpu()
        rotation = model.get_rotation.detach().float().cpu()
        opacity = model.get_opacity.detach().float().reshape(-1).cpu()
        color = torch.clamp(sh_dc_to_rgb(model.get_features_dc[:, 0, :].detach().float()), 0.0, 1.0).cpu()

    count = int(xyz.shape[0])
    payload: dict = {
        "type": "snapshot",
        "iteration": 0,
        "parameterSpace": "render",
        "count": count,
        "positions": xyz.reshape(-1).tolist(),
        "scales": scaling.reshape(-1).tolist(),
        "colors": color.reshape(-1).tolist(),
        "opacities": opacity.reshape(-1).tolist(),
        "rotations": rotation.reshape(-1).tolist(),
        "metadata": {
            "source": "osn-gs-visible-surface-inspect",
            "totalCount": count,
            "sentCount": count,
            "capped": False,
            "loss": 0.0,
            "psnr": 0.0,
        },
    }

    grid = state.surface.control_grid.detach().cpu()
    weights = state.surface.weights.detach().cpu()
    nurbs_payload = {
        "type": "visible_nurbs_intermediate",
        "iteration": 0,
        "parameter_domain": {"u": [0.0, 1.0], "v": [0.0, 1.0]},
        "degree_u": int(state.surface.degree_u),
        "degree_v": int(state.surface.degree_v),
        "observed_v_max": float(state.surface.observed_v_max),
        "control_grid_shape": [int(value) for value in grid.shape],
        "control_grid": grid.reshape(-1, 3).tolist(),
        "weights": weights.reshape(-1).tolist(),
        "patches": [
            {
                "patch_id": patch_id,
                "control_grid_shape": [int(value) for value in patch.control_grid.shape],
                "control_grid": patch.control_grid.detach().cpu().reshape(-1, 3).tolist(),
                "weights": patch.weights.detach().cpu().reshape(-1).tolist(),
                "degree_u": int(patch.degree_u),
                "degree_v": int(patch.degree_v),
            }
            for patch_id, patch in enumerate(state.surface_patches)
        ],
        "metadata": {
            "source": "osn_gs_visible_surface_inspect",
            "gaussian_count": len(state.model),
            "uncertain_count": 0,
            "voxel_role": "initial_bootstrap",
            "surface_topology_version": int(state.surface_topology_version),
            "patch_residual_ratios": quality_report["patch_residual_ratios"],
            "flattened": True,
        },
    }
    voxel_payload = voxel_regions_payload(state.voxel_regions)
    if voxel_payload is not None:
        nurbs_payload["voxel_regions"] = voxel_payload
    payload["nurbs_surface"] = nurbs_payload
    return payload


def evaluate_surface_quality(pipeline: TorchOSNGSPipeline, state) -> dict:
    """Run one honest quality pass: foot-point UV refresh + per-patch residuals.

    Reuses `maintain_surface_from_certain` with local correction disabled so this
    is a read-only inspection: it refreshes UV bindings to their true closest
    surface point and reports the resulting normalized residual, but does not
    split or modify any patch.
    """

    report = pipeline.maintain_surface_from_certain(
        state,
        residual_ratio_threshold=0.0,
        residual_patience=1,
        enable_local_correction=False,
        refresh_uv=True,
    )
    return {
        "patches": report["patches"],
        "checked_gaussians": report["checked"],
        "max_residual_ratio": report["max_residual_ratio"],
        "uv_refreshed": report["uv_refreshed"],
        "patch_residual_ratios": {str(key): value for key, value in state.surface_patch_residuals.items()},
    }


def main() -> None:
    args = build_parser().parse_args()
    torch = require_torch()
    device = args.device or default_device(prefer_cuda=True)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sparse_root = Path(args.source_path) / args.sparse_dir
    if not sparse_root.exists():
        raise FileNotFoundError(f"Missing COLMAP sparse directory: {sparse_root}")
    point_cloud = read_colmap_points3d(sparse_root)
    if point_cloud.xyz.shape[0] == 0:
        raise ValueError(f"No sparse points found in {sparse_root}")

    points = torch.as_tensor(point_cloud.xyz, dtype=torch.float32, device=device)
    colors = torch.as_tensor(point_cloud.rgb, dtype=torch.float32, device=device)
    if args.max_points > 0 and points.shape[0] > args.max_points:
        indices = torch.linspace(0, points.shape[0] - 1, steps=args.max_points, device=device).round().long()
        points, colors = points[indices], colors[indices]

    print(f"OSN-GS inspect: device={device} points={int(points.shape[0])} source={sparse_root}", flush=True)

    pipeline = TorchOSNGSPipeline(build_pipeline_config(args), device=device)
    state = pipeline.initialize(points, colors)
    quality_report = evaluate_surface_quality(pipeline, state)

    payload = build_stream_payload(state, quality_report)
    snapshot_path = output_dir / "renderer_snapshot.json"
    snapshot_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    quality_path = output_dir / "surface_quality.json"
    quality_path.write_text(json.dumps(quality_report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_lines = [
        f"source={args.source_path}",
        f"gaussians={len(state.model)}",
        f"voxel_regions={0 if state.voxel_regions is None else int(state.voxel_regions.region_centers.shape[0])}",
        f"patches={quality_report['patches']}",
        f"checked_gaussians={quality_report['checked_gaussians']}",
        f"max_residual_ratio={quality_report['max_residual_ratio']:.6g}",
        "  (normalized point-to-surface distance / certain-Gaussian bbox diagonal; lower is a tighter fit)",
        "per_patch_residual_ratio:",
    ]
    for patch_id, ratio in sorted(quality_report["patch_residual_ratios"].items(), key=lambda item: int(item[0])):
        control_shape = state.surface_patches[int(patch_id)].control_grid.shape
        summary_lines.append(f"  patch {patch_id}: ratio={ratio:.6g} control_grid={tuple(control_shape[:2])}")
    summary_text = "\n".join(summary_lines) + "\n"
    (output_dir / "surface_quality.txt").write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print(f"Renderer snapshot: {snapshot_path}", flush=True)
    print(
        "Load renderer_snapshot.json in a renderer-compatible snapshot path to view the surface, "
        "or inspect surface_quality.txt/.json directly.",
        flush=True,
    )


if __name__ == "__main__":
    main()
