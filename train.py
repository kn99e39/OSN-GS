from __future__ import annotations

"""Notebook-compatible OSN-GS training entrypoint.

The renderer notebook discovers a project by searching for `train.py`.
This wrapper keeps that workflow intact while delegating the real work to
`TorchOSNGSTrainer`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from osn_gs.core.torch_pipeline import TorchPipelineConfig
from osn_gs.core.torch_trainer import TorchOSNGSTrainer, TorchTrainingConfig
from osn_gs.data.colmap_scene import load_colmap_scene, load_colmap_scene_with_eval_split
from osn_gs.eval.held_out_metrics import evaluate_held_out_cameras
from osn_gs.gaussian.torch_density_control import TorchDensityControlConfig
from osn_gs.interop.colab_args import (
    build_osn_gs_train_parser,
    output_dir_from_args,
    save_interval_from_args,
    save_iterations_from_args,
    stage1_constructor_config_kwargs,
    surface_fit_config_kwargs,
)
from osn_gs.render.diff_gaussian_loader import validate_diff_gaussian_build_environment
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig
from osn_gs.utils.torch_ops import default_device


def main() -> None:
    args = build_osn_gs_train_parser().parse_args()
    device = args.device or default_device(prefer_cuda=True)
    image_device = args.image_device or ("auto" if device == "cuda" else device)
    if args.low_vram and not args.image_device:
        image_device = "cpu"
    output_dir = output_dir_from_args(args)
    print(f"OSN-GS device: train={device}, images={image_device}", flush=True)
    if not args.disable_cuda_rasterizer and not args.skip_cuda_build_preflight:
        preflight = validate_diff_gaussian_build_environment()
        print(
            "OSN-GS CUDA build preflight: "
            f"cl={preflight['compiler']} nvcc={preflight['nvcc']}",
            flush=True,
        )
    save_interval = save_interval_from_args(args)
    save_iterations = save_iterations_from_args(args)
    stream_iterations = tuple(sorted({int(value) for value in args.stream_iterations if int(value) > 0}))
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
    if args.eval and train_resolution_scale > 1:
        print(
            f"OSN-GS --eval: ignoring train_resolution_scale={train_resolution_scale} "
            "(--low_vram or --train_resolution_scale would otherwise apply an EXTRA "
            "training-time downscale on top of the loader's own baseline-matched "
            "resolution, breaking the same-resolution A/B).",
            flush=True,
        )
        train_resolution_scale = 1

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
        **stage1_constructor_config_kwargs(args),
    )
    training_config = TorchTrainingConfig(
        iterations=args.iterations,
        surface_rebuild_interval=max(0, int(args.surface_update_interval)),
        surface_loss_patch_budget=max(0, int(args.surface_loss_patch_budget)),
        surface_maintenance_patch_budget=max(0, int(args.surface_maintenance_patch_budget)),
        surface_residual_ratio_threshold=max(0.0, float(args.surface_residual_ratio_threshold)),
        surface_residual_patience=max(1, int(args.surface_residual_patience)),
        surface_local_min_gaussians=max(4, int(args.surface_local_min_gaussians)),
        surface_local_min_component=max(4, int(args.surface_local_min_component)),
        enable_local_surface_correction=not args.disable_local_surface_correction,
        density_control_interval=args.density_control_interval,
        save_interval=save_interval,
        save_iterations=save_iterations,
        progress_log_interval=args.progress_log_interval,
        timing_log_interval=args.timing_log_interval,
        stream_url=args.stream_url,
        stream_every=max(0, int(args.stream_every)),
        stream_iterations=stream_iterations,
        stream_max_gaussians=max(0, int(args.stream_max_gaussians)),
        stream_cache_dir=args.stream_cache_dir,
        stream_queue_size=max(1, int(args.stream_queue_size)),
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

    eval_split = None
    if args.eval:
        eval_split = load_colmap_scene_with_eval_split(
            args.source_path,
            device=device,
            image_dir_name=args.images,
            sparse_dir_name=args.sparse_dir,
            resolution=args.resolution,
            resolution_scale=args.resolution_scale,
            eval=True,
            llffhold=args.llffhold,
        )
        scene = eval_split.train_scene
    else:
        scene = load_colmap_scene(
            args.source_path,
            device=device,
            image_device=image_device,
            image_dir_name=args.images,
            sparse_dir_name=args.sparse_dir,
            image_downscale=args.image_downscale,
            max_images=args.max_images,
        )

    result = trainer.train(scene, output_dir)
    print(
        "OSN-GS train.py complete: "
        f"iteration={result.state.iteration}, "
        f"loss={result.state.last_loss:.6f}, "
        f"psnr={result.state.last_psnr:.3f}, "
        f"gaussians={len(result.state.model)}, "
        f"uncertain={int(result.state.model.is_uncertain.sum().item())}, "
        f"output={result.output_dir}"
    )

    if eval_split is not None:
        held_out = evaluate_held_out_cameras(
            trainer.rasterizer, result.state.model, eval_split.test_cameras, eval_split.test_images, device=device,
        )
        print(
            "OSN-GS held-out eval "
            f"(cameras={held_out['camera_count']}, resolution={eval_split.resolution}, "
            f"llffhold={args.llffhold}): "
            f"psnr_mean={held_out['psnr_mean']:.3f} ssim_mean={held_out['ssim_mean']:.4f}",
            flush=True,
        )
        if not args.disable_output_files:
            import json

            report_path = Path(result.output_dir) / "held_out_eval.json"
            report_path.write_text(
                json.dumps(
                    {
                        "iteration": result.state.iteration,
                        "resolution": list(eval_split.resolution),
                        "downscale_factor": eval_split.downscale_factor,
                        "llffhold": args.llffhold,
                        **held_out,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"OSN-GS held-out eval report: {report_path}", flush=True)


if __name__ == "__main__":
    main()



