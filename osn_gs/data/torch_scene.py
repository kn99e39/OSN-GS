from __future__ import annotations

"""Torch training scene helpers.

정식 COLMAP/3DGS scene loader를 붙이기 전까지 사용하는 최소 scene abstraction이다.
학습 루프는 `TorchScene.sample_views()`만 알면 되므로, 나중에 실제 camera/image
loader로 바꾸더라도 trainer 쪽 변경을 줄일 수 있다.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osn_gs.render.torch_fallback import TorchCamera
from osn_gs.utils.torch_ops import require_torch


@dataclass
class TorchImageBatch:
    """trainer에 전달되는 camera/image batch."""

    cameras: list[TorchCamera]
    # Shape: (B, 3, H, W).
    images: Any


@dataclass
class TorchScene:
    """OSN-GS Torch trainer가 요구하는 최소 scene protocol."""

    # 초기 certain Gaussian center 후보. 보통 SfM point나 기존 3DGS Gaussian center에서 온다.
    initial_points: Any
    # 초기 certain Gaussian 색상 prior.
    initial_colors: Any
    # render target camera 목록.
    cameras: list[TorchCamera]
    # target image tensor. Shape은 (V, 3, H, W)로 맞춘다.
    images: Any
    # scene scale. 추후 3DGS ADC threshold와 연결할 값이다.
    extent: float = 1.0

    def sample_views(self, iteration: int, batch_size: int = 1) -> TorchImageBatch:
        """iteration 기반 deterministic camera sampling.

        실제 학습에서는 random sampling을 써도 되지만, 초기 프레임워크에서는
        재현성을 위해 순차 index를 사용한다.
        """

        torch = require_torch()
        count = len(self.cameras)
        if count == 0:
            raise ValueError("TorchScene requires at least one camera.")
        indices = [(iteration + offset) % count for offset in range(batch_size)]
        image_indices = torch.as_tensor(indices, dtype=torch.long, device=self.images.device)
        return TorchImageBatch(cameras=[self.cameras[idx] for idx in indices], images=self.images[image_indices])


def make_torch_synthetic_scene(
    point_count: int = 48,
    image_size: int = 96,
    device: str = "cuda",
) -> TorchScene:
    """파이프라인 smoke run용 synthetic scene을 만든다."""

    torch = require_torch()

    # 관측 Gaussian center 역할을 하는 작은 3D 곡선.
    x = torch.linspace(-0.8, 0.8, point_count, device=device)
    y = 0.18 * torch.sin(2.0 * torch.pi * x)
    z = 0.15 * torch.cos(torch.pi * x)
    points = torch.stack([x, y, z], dim=-1)

    # x/y에 따라 부드럽게 변하는 색상. uncertain color prior 검증용이다.
    colors = torch.stack([(x + 0.8) / 1.6, 0.55 + y, 1.0 - (x + 0.8) / 1.6], dim=-1).clamp(0.0, 1.0)
    camera = TorchCamera(image_height=image_size, image_width=image_size, image_name="synthetic")

    # fallback renderer용 단순 target. 실제 품질 목적 데이터가 아니라 loop 검증용이다.
    target = colors.mean(dim=0).view(1, 3, 1, 1).expand(1, 3, image_size, image_size).contiguous()
    return TorchScene(initial_points=points, initial_colors=colors, cameras=[camera], images=target, extent=1.0)


def load_npz_scene(path: str | Path, device: str = "cuda") -> TorchScene:
    """NPZ 파일에서 최소 scene을 로드한다.

    필요한 key:
    - points: (N, 3)
    - colors: (N, 3)
    - images: (V, 3, H, W) 또는 (V, H, W, 3)
    - extent: optional scalar
    """

    torch = require_torch()
    import numpy as np

    data = np.load(Path(path), allow_pickle=False)
    points = torch.as_tensor(data["points"], dtype=torch.float32, device=device)
    colors = torch.as_tensor(data["colors"], dtype=torch.float32, device=device)
    images = torch.as_tensor(data["images"], dtype=torch.float32, device=device)

    # channel-last 이미지도 trainer가 기대하는 channel-first로 바꾼다.
    if images.ndim == 4 and images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2).contiguous()

    # 아직 실제 camera pose를 읽지 않으므로 image마다 placeholder camera를 만든다.
    cameras = [
        TorchCamera(
            image_height=int(images.shape[-2]),
            image_width=int(images.shape[-1]),
            image_name=f"npz_{idx:04d}",
        )
        for idx in range(images.shape[0])
    ]
    extent = float(data["extent"]) if "extent" in data else 1.0
    return TorchScene(initial_points=points, initial_colors=colors, cameras=cameras, images=images, extent=extent)
