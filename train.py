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
from osn_gs.data.colmap_scene import load_colmap_scene
from osn_gs.interop.colab_args import build_osn_gs_train_parser, output_dir_from_args, save_interval_from_args
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig
from osn_gs.utils.torch_ops import default_device


def main() -> None:
    args = build_osn_gs_train_parser().parse_args()
    device = args.device or default_device(prefer_cuda=True)
    image_device = args.image_device or ("cpu" if device == "cuda" else device)
    if args.low_vram and not args.image_device:
        image_device = "cpu"
    output_dir = output_dir_from_args(args)
    save_interval = save_interval_from_args(args)
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

    pipeline_config = TorchPipelineConfig(
        base_curve_count=args.base_curve_count,
        visible_surface_resolution_u=args.visible_surface_resolution_u,
        visible_surface_resolution_v=args.visible_surface_resolution_v,
        visible_surface_resolution_scale=args.visible_surface_resolution_scale,
        uncertain_samples_u=uncertain_samples_u,
        uncertain_samples_v=uncertain_samples_v,
        max_uncertain_gaussians=max_uncertain_gaussians,
    )
    training_config = TorchTrainingConfig(
        iterations=args.iterations,
        surface_rebuild_interval=args.surface_rebuild_interval,
        density_control_interval=args.density_control_interval,
        save_interval=save_interval,
        prefer_cuda=device == "cuda",
        train_resolution_scale=train_resolution_scale,
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


if __name__ == "__main__":
    main()


