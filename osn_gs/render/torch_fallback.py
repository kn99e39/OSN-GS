from __future__ import annotations

"""개발용 differentiable fallback renderer.

이 renderer는 진짜 3DGS rasterizer를 대체하기 위한 품질 목적 구현이 아니다.
CUDA extension이 없는 환경에서도 OSN-GS의 tensor 흐름, loss, 저장 경로를
검증할 수 있게 하는 부드러운 2D Gaussian splat approximation이다.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.utils.torch_ops import require_torch


@dataclass
class TorchCamera:
    """CUDA rasterizer와 fallback renderer가 공유하는 최소 camera protocol."""

    image_height: int
    image_width: int
    # CUDA rasterizer 사용 시 필요한 행렬들. synthetic/fallback에서는 None이어도 된다.
    world_view_transform: Any | None = None
    full_proj_transform: Any | None = None
    camera_center: Any | None = None
    FoVx: float = 0.7
    FoVy: float = 0.7
    image_name: str = "camera"


def fallback_render(camera: TorchCamera, model: TorchGaussianModel, background: Any | None = None) -> dict[str, Any]:
    """XY 평면에 Gaussian을 splat하는 단순 differentiable renderer."""

    torch = require_torch()
    device = model.device
    height = int(camera.image_height)
    width = int(camera.image_width)

    # normalized image plane [-1, 1]^2를 만든다.
    ys = torch.linspace(-1.0, 1.0, height, device=device)
    xs = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack([xx, yy], dim=-1)

    xyz = model.get_xyz
    # 임시 projection: x/y만 image plane 좌표로 사용한다.
    xy = xyz[:, :2].clamp(-1.2, 1.2)
    depth = xyz[:, 2]
    # scale은 screen-space Gaussian radius 역할을 한다.
    scales = model.get_scaling[:, :2].mean(dim=-1).clamp(min=1e-3, max=0.5)
    opacity = model.get_opacity[:, 0].clamp(0.0, 1.0)
    colors = model.rgb.clamp(0.0, 1.0)

    # 각 Gaussian이 각 pixel에 주는 weight를 계산한다.
    dist2 = (pixels[None, :, :, :] - xy[:, None, None, :]).square().sum(dim=-1)
    weights = torch.exp(-dist2 / (2.0 * scales[:, None, None].square())) * opacity[:, None, None]
    alpha = weights.sum(dim=0).clamp(0.0, 1.0)

    # weighted color average 후 alpha compositing.
    color_accum = (weights[:, None, :, :] * colors[:, :, None, None]).sum(dim=0)
    denom = weights.sum(dim=0, keepdim=True).clamp(min=1e-6)
    image = color_accum / denom
    if background is None:
        background = torch.zeros((3,), dtype=torch.float32, device=device)
    image = image * alpha[None, :, :] + background[:, None, None] * (1.0 - alpha[None, :, :])

    # depth/radii는 train loop 호환을 위한 placeholder 성격이다.
    depth_image = (weights * depth[:, None, None]).sum(dim=0, keepdim=True) / denom
    radii = scales * max(height, width)
    return {
        "render": image.clamp(0.0, 1.0),
        "viewspace_points": model.get_xyz,
        "visibility_filter": torch.nonzero(radii > 0, as_tuple=False).reshape(-1),
        "radii": radii,
        "depth": depth_image,
    }
