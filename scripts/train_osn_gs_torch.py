from __future__ import annotations

"""OSN-GS Torch training CLI."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osn_gs.core.torch_pipeline import TorchPipelineConfig
from osn_gs.core.torch_trainer import TorchOSNGSTrainer, TorchTrainingConfig
from osn_gs.data.colmap_scene import load_colmap_scene
from osn_gs.gaussian.torch_density_control import TorchDensityControlConfig
from osn_gs.interop.colab_args import add_surface_fit_arguments, surface_fit_config_kwargs
from osn_gs.render.diff_gaussian_loader import validate_diff_gaussian_build_environment
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig
from osn_gs.utils.torch_ops import default_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the OSN-GS torch framework.")
    parser.add_argument("-s", "--source_path", type=str, default="", help="COLMAP scene root with images/ and sparse/.")
    parser.add_argument("--images", type=str, default="images", help="Image folder name under --source_path.")
    parser.add_argument("--sparse_dir", type=str, default="sparse/0", help="Sparse COLMAP folder under --source_path.")
    parser.add_argument("--image_downscale", type=int, default=1, help="Integer image downscale for COLMAP loading.")
    parser.add_argument("--max_images", type=int, default=0, help="Limit loaded COLMAP images; 0 means all.")
    parser.add_argument("--output", type=str, default="outputs/osn_gs_torch", help="Output directory.")
    parser.add_argument("--device", type=str, default="", help="cuda, cpu, or empty for auto.")
    parser.add_argument(
        "--image_device",
        type=str,
        default="",
        help="Device for storing training images. Images stay CPU-staged and only sampled views are transferred to the training device.",
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--train_resolution_scale", type=int, default=1, help="Additional training-time render downscale.")
    parser.add_argument("--base_curve_count", type=int, default=8)
    parser.add_argument("--visible_surface_resolution_u", type=int, default=8)
    parser.add_argument("--visible_surface_resolution_v", type=int, default=4)
    parser.add_argument("--visible_surface_resolution_scale", type=float, default=1.0)
    parser.add_argument("--max_surface_control_points", type=int, default=65536)
    parser.add_argument("--covariance_init", type=str, default="knn", choices=("knn", "constant"))
    parser.add_argument("--covariance_knn_chunk_size", type=int, default=0)
    parser.add_argument("--covariance_min_scale", type=float, default=1e-4)
    parser.add_argument("--covariance_max_scale_ratio", type=float, default=0.05)
    parser.add_argument("--covariance_scale_multiplier", type=float, default=1.0)
    parser.add_argument(
        "--visible_surface_fit_device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda", "auto"),
        help="Workspace device for visible NURBS fitting. cpu lowers VRAM use; cuda can be faster.",
    )
    parser.add_argument(
        "--visible_surface_fit_chunk_size",
        type=int,
        default=0,
        help="NURBS grid samples per fitting chunk. 0 auto-selects once from available VRAM at startup.",
    )
    add_surface_fit_arguments(parser)
    parser.add_argument("--disable_voxel_surface_regions", action="store_true", help="Bypass pre-NURBS voxel surface-region placement.")
    parser.add_argument("--voxel_grid_resolution", type=int, default=16, help="Coarse voxel grid resolution per axis before NURBS fitting.")
    parser.add_argument("--disable_adaptive_voxel_density", action="store_true", help="Keep all occupied voxel cells at the coarse resolution.")
    parser.add_argument("--voxel_max_subdivision_depth", type=int, default=1, help="Density-driven voxel subdivision depth.")
    parser.add_argument("--voxel_density_quantile", type=float, default=0.75, help="Occupied-cell density quantile selected for subdivision.")
    parser.add_argument("--voxel_density_covariance_weight_cap", type=float, default=10.0, help="Maximum inverse-covariance contribution to voxel density.")
    parser.add_argument("--voxel_normal_knn", type=int, default=16, help="Neighbor count for local PCA normal estimation.")
    parser.add_argument("--voxel_boundary_angle_degrees", type=float, default=35.0, help="Normal-angle change used to mark voxel boundaries.")
    parser.add_argument("--voxel_min_points_per_region", type=int, default=1, help="Minimum Gaussians required to keep a voxel region.")
    parser.add_argument("--voxel_normal_chunk_size", type=int, default=4096, help="Regions processed per batched SVD/cdist chunk during voxel normal estimation.")
    parser.add_argument("--uncertain_samples_u", type=int, default=16)
    parser.add_argument("--uncertain_samples_v", type=int, default=3)
    parser.add_argument("--max_uncertain_gaussians", type=int, default=0, help="Cap the number of uncertain Gaussians.")
    parser.add_argument("--densify_from_iter", type=int, default=500)
    parser.add_argument("--densify_until_iter", type=int, default=0, help="Run 3DGS-style ADC until this iteration. 0 disables ADC.")
    parser.add_argument("--densification_interval", type=int, default=0, help="Run 3DGS-style ADC every N iterations. 0 disables ADC.")
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002, help="Screen-space gradient threshold for ADC clone/split.")
    parser.add_argument("--adc_max_gaussians", type=int, default=0, help="Optional hard cap for Gaussian count during ADC. 0 means uncapped.")
    parser.add_argument("--adc_percent_dense", type=float, default=0.01)
    parser.add_argument("--adc_prune_opacity_threshold", type=float, default=0.005)
    parser.add_argument("--adc_split_samples", type=int, default=2)
    parser.add_argument("--adc_max_screen_size", type=float, default=20.0)
    parser.add_argument("--adc_max_scale_ratio", type=float, default=0.1)
    parser.add_argument("--opacity_reset_interval", type=int, default=3000)
    parser.add_argument("--screen_size_prune_from_iter", type=int, default=3000)
    parser.add_argument(
        "--surface_update_interval",
        "--surface_rebuild_interval",
        dest="surface_update_interval",
        type=int,
        default=1000,
        help="Inspect persistent NURBS patch quality every N iterations; this does not globally rebuild voxels.",
    )
    parser.add_argument("--surface_loss_patch_budget", type=int, default=16, help="NURBS patches evaluated per iteration. 0 evaluates all patches.")
    parser.add_argument("--surface_residual_ratio_threshold", type=float, default=0.03)
    parser.add_argument("--surface_residual_patience", type=int, default=3)
    parser.add_argument("--surface_local_min_gaussians", type=int, default=64)
    parser.add_argument("--surface_local_min_component", type=int, default=16)
    parser.add_argument("--disable_local_surface_correction", action="store_true")
    parser.add_argument("--density_control_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--progress_log_interval", type=int, default=100, help="Print training progress every N iterations. 0 disables periodic progress logs.")
    parser.add_argument("--timing_log_interval", type=int, default=100, help="Print per-stage training timing every N iterations. 0 disables periodic timing logs.")
    parser.add_argument("--skip_cuda_build_preflight", action="store_true", help="Skip the early MSVC/CUDA/Ninja readiness check before CUDA rasterizer loading.")
    parser.add_argument("--disable_cuda_rasterizer", action="store_true")
    parser.add_argument("--stream_url", type=str, default="")
    parser.add_argument("--stream_every", type=int, default=0)
    parser.add_argument("--stream_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--stream_max_gaussians", type=int, default=0)
    parser.add_argument("--stream_cache_dir", type=str, default="", help="Directory for cached stream snapshot JSON files.")
    parser.add_argument("--disable_stream_nurbs", action="store_true")
    parser.add_argument("--disable_output_files", action="store_true")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument(
        "--low_vram",
        action="store_true",
        help="Apply a conservative preset for 16GB-class GPUs: keep images on CPU, halve train resolution, and cap uncertain Gaussians.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = args.device or default_device(prefer_cuda=True)
    image_device = args.image_device or ("auto" if device == "cuda" else device)
    if args.low_vram and not args.image_device:
        image_device = "cpu"
    print(f"OSN-GS device: train={device}, images={image_device}", flush=True)
    if not args.disable_cuda_rasterizer and not args.skip_cuda_build_preflight:
        preflight = validate_diff_gaussian_build_environment()
        print(
            "OSN-GS CUDA build preflight: "
            f"cl={preflight['compiler']} nvcc={preflight['nvcc']}",
            flush=True,
        )
    train_resolution_scale = max(1, int(args.train_resolution_scale))
    uncertain_samples_u = args.uncertain_samples_u
    uncertain_samples_v = args.uncertain_samples_v
    max_uncertain_gaussians = max(0, int(args.max_uncertain_gaussians))
    if args.low_vram:
        train_resolution_scale = max(train_resolution_scale, 2)
        uncertain_samples_u = min(uncertain_samples_u, 8)
        uncertain_samples_v = min(uncertain_samples_v, 2)
        if max_uncertain_gaussians == 0:
            max_uncertain_gaussians = 128

    densify_grad_threshold = float(args.densify_grad_threshold)
    if densify_grad_threshold <= 0.0:
        densify_grad_threshold = TorchDensityControlConfig().densify_grad_threshold
    density_control_config = TorchDensityControlConfig(
        densify_from_iter=max(0, int(args.densify_from_iter)),
        densify_until_iter=max(0, int(args.densify_until_iter)),
        densification_interval=max(0, int(args.densification_interval)),
        densify_grad_threshold=densify_grad_threshold,
        max_gaussians=max(0, int(getattr(args, "adc_max_gaussians", 0))),
        percent_dense=max(0.0, float(args.adc_percent_dense)),
        prune_opacity_threshold=max(0.0, float(args.adc_prune_opacity_threshold)),
        split_samples=max(1, int(args.adc_split_samples)),
        max_screen_size=max(0.0, float(args.adc_max_screen_size)),
        max_scale_ratio=max(0.0, float(args.adc_max_scale_ratio)),
        opacity_reset_interval=max(0, int(args.opacity_reset_interval)),
        screen_size_prune_from_iter=max(0, int(args.screen_size_prune_from_iter)),
    )

    pipeline_config = TorchPipelineConfig(
        base_curve_count=args.base_curve_count,
        visible_surface_resolution_u=args.visible_surface_resolution_u,
        visible_surface_resolution_v=args.visible_surface_resolution_v,
        visible_surface_resolution_scale=args.visible_surface_resolution_scale,
        max_surface_control_points=max(4, int(args.max_surface_control_points)),
        covariance_init=args.covariance_init,
        covariance_knn_chunk_size=args.covariance_knn_chunk_size,
        covariance_min_scale=args.covariance_min_scale,
        covariance_max_scale_ratio=args.covariance_max_scale_ratio,
        covariance_scale_multiplier=args.covariance_scale_multiplier,
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
        uncertain_samples_u=uncertain_samples_u,
        uncertain_samples_v=uncertain_samples_v,
        max_uncertain_gaussians=max_uncertain_gaussians,
        **surface_fit_config_kwargs(args),
    )
    training_config = TorchTrainingConfig(
        iterations=args.iterations,
        surface_rebuild_interval=max(0, int(args.surface_update_interval)),
        surface_loss_patch_budget=max(0, int(args.surface_loss_patch_budget)),
        surface_residual_ratio_threshold=max(0.0, float(args.surface_residual_ratio_threshold)),
        surface_residual_patience=max(1, int(args.surface_residual_patience)),
        surface_local_min_gaussians=max(4, int(args.surface_local_min_gaussians)),
        surface_local_min_component=max(4, int(args.surface_local_min_component)),
        enable_local_surface_correction=not args.disable_local_surface_correction,
        density_control_interval=args.density_control_interval,
        save_interval=args.save_interval,
        save_iterations=(),
        progress_log_interval=args.progress_log_interval,
        timing_log_interval=args.timing_log_interval,
        stream_url=args.stream_url,
        stream_every=max(0, int(args.stream_every)),
        stream_iterations=tuple(sorted({int(value) for value in args.stream_iterations if int(value) > 0})),
        stream_max_gaussians=max(0, int(args.stream_max_gaussians)),
        stream_cache_dir=args.stream_cache_dir,
        stream_nurbs=not args.disable_stream_nurbs,
        write_output_files=not args.disable_output_files,
        resume_checkpoint=args.resume_checkpoint,
        prefer_cuda=device == "cuda",
        train_resolution_scale=train_resolution_scale,
        density_control=density_control_config,
    )

    print(
        "OSN-GS surface loss: "
        f"patch_budget={training_config.surface_loss_patch_budget} (0=all patches)",
        flush=True,
    )
    rasterizer_config = GaussianRasterizerConfig(prefer_cuda=not args.disable_cuda_rasterizer)
    trainer = TorchOSNGSTrainer(
        pipeline_config=pipeline_config,
        training_config=training_config,
        rasterizer_config=rasterizer_config,
        device=device,
    )

    if not args.source_path:
        raise ValueError("OSN-GS requires --source_path/-s pointing to a COLMAP dataset root.")

    scene = load_colmap_scene(
        args.source_path,
        device=device,
        image_device=image_device,
        image_dir_name=args.images,
        sparse_dir_name=args.sparse_dir,
        image_downscale=args.image_downscale,
        max_images=args.max_images,
    )

    result = trainer.train(scene, args.output)
    print(
        "OSN-GS torch training complete: "
        f"iteration={result.state.iteration}, "
        f"loss={result.state.last_loss:.6f}, "
        f"psnr={result.state.last_psnr:.3f}, "
        f"gaussians={len(result.state.model)}, "
        f"uncertain={int(result.state.model.is_uncertain.sum().item())}, "
        f"output={result.output_dir}"
    )


if __name__ == "__main__":
    main()




