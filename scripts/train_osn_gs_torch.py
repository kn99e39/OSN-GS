from __future__ import annotations

"""OSN-GS Torch 학습 CLI.

CUDA 서버에서 가장 먼저 실행할 entrypoint다. synthetic scene으로 smoke run을
돌릴 수도 있고, 최소 NPZ scene을 넘겨 실제 데이터 연결 전 단계 실험도 가능하다.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osn_gs.core.torch_pipeline import TorchPipelineConfig
from osn_gs.core.torch_trainer import TorchOSNGSTrainer, TorchTrainingConfig
from osn_gs.data.colmap_scene import load_colmap_scene
from osn_gs.data.torch_scene import load_npz_scene, make_torch_synthetic_scene
from osn_gs.render.cuda_rasterizer_adapter import RasterizerPipelineOptions
from osn_gs.utils.torch_ops import default_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the OSN-GS torch framework.")
    # scene_npz가 비어 있으면 synthetic scene을 생성한다.
    parser.add_argument("--scene_npz", type=str, default="", help="Optional NPZ scene with points, colors, images.")
    parser.add_argument("-s", "--source_path", type=str, default="", help="COLMAP scene root with images/ and sparse/.")
    parser.add_argument("--images", type=str, default="images", help="Image folder name under --source_path.")
    parser.add_argument("--sparse_dir", type=str, default="sparse/0", help="Sparse COLMAP folder under --source_path.")
    parser.add_argument("--image_downscale", type=int, default=1, help="Integer image downscale for COLMAP loading.")
    parser.add_argument("--max_images", type=int, default=0, help="Limit loaded COLMAP images; 0 means all.")
    # output 아래에 final/iteration_xxxxxx 폴더가 생긴다.
    parser.add_argument("--output", type=str, default="outputs/osn_gs_torch", help="Output directory.")
    # 빈 문자열이면 CUDA 가능 여부를 보고 자동 선택한다.
    parser.add_argument("--device", type=str, default="", help="cuda, cpu, or empty for auto.")
    # 기본 학습 제어 인자들.
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--point_count", type=int, default=48)
    # surface/uncertain Gaussian 초기화 해상도.
    parser.add_argument("--base_curve_count", type=int, default=8)
    parser.add_argument("--uncertain_samples_u", type=int, default=16)
    parser.add_argument("--uncertain_samples_v", type=int, default=3)
    # OSN-GS 특유의 update 주기.
    parser.add_argument("--surface_rebuild_interval", type=int, default=1000)
    parser.add_argument("--density_control_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    # CUDA extension이 설치되어 있어도 fallback renderer를 강제로 쓰고 싶을 때 사용한다.
    parser.add_argument("--disable_cuda_rasterizer", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = args.device or default_device(prefer_cuda=True)

    # CLI 인자를 dataclass config로 옮긴다.
    pipeline_config = TorchPipelineConfig(
        base_curve_count=args.base_curve_count,
        uncertain_samples_u=args.uncertain_samples_u,
        uncertain_samples_v=args.uncertain_samples_v,
    )
    training_config = TorchTrainingConfig(
        iterations=args.iterations,
        surface_rebuild_interval=args.surface_rebuild_interval,
        density_control_interval=args.density_control_interval,
        save_interval=args.save_interval,
        prefer_cuda=device == "cuda",
    )

    # diff_gaussian_rasterization 사용 여부는 adapter가 import 가능성으로 자동 판단한다.
    rasterizer_options = RasterizerPipelineOptions(prefer_cuda_rasterizer=not args.disable_cuda_rasterizer)
    trainer = TorchOSNGSTrainer(
        pipeline_config=pipeline_config,
        training_config=training_config,
        rasterizer_options=rasterizer_options,
        device=device,
    )

    # 최소 scene abstraction만 맞추면 trainer는 synthetic/NPZ/향후 COLMAP loader를 구분하지 않는다.
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
