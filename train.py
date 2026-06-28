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
from osn_gs.data.torch_scene import load_npz_scene, make_torch_synthetic_scene
from osn_gs.interop.colab_args import build_osn_gs_train_parser, output_dir_from_args, save_interval_from_args
from osn_gs.render.cuda_rasterizer_adapter import RasterizerPipelineOptions
from osn_gs.utils.torch_ops import default_device


def main() -> None:
    args = build_osn_gs_train_parser().parse_args()
    device = args.device or default_device(prefer_cuda=True)
    output_dir = output_dir_from_args(args)
    save_interval = save_interval_from_args(args)

    pipeline_config = TorchPipelineConfig(
        base_curve_count=args.base_curve_count,
        uncertain_samples_u=args.uncertain_samples_u,
        uncertain_samples_v=args.uncertain_samples_v,
    )
    training_config = TorchTrainingConfig(
        iterations=args.iterations,
        surface_rebuild_interval=args.surface_rebuild_interval,
        density_control_interval=args.density_control_interval,
        save_interval=save_interval,
        prefer_cuda=device == "cuda",
    )
    rasterizer_options = RasterizerPipelineOptions(prefer_cuda_rasterizer=not args.disable_cuda_rasterizer)
    trainer = TorchOSNGSTrainer(
        pipeline_config=pipeline_config,
        training_config=training_config,
        rasterizer_options=rasterizer_options,
        device=device,
    )

    if args.scene_npz:
        scene = load_npz_scene(args.scene_npz, device=device)
    elif args.source_path:
        scene = load_colmap_scene(
            args.source_path,
            device=device,
            image_dir_name=args.images,
            sparse_dir_name=args.sparse_dir,
            image_downscale=args.image_downscale,
            max_images=args.max_images,
        )
    else:
        scene = make_torch_synthetic_scene(
            point_count=args.point_count,
            image_size=args.image_size,
            device=device,
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
