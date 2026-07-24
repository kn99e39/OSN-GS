from __future__ import annotations

"""Differentiable torch fallback renderer for OSN-GS.

This renderer is intentionally slower than the CUDA rasterizer, but it keeps
memory bounded by processing Gaussians in chunks instead of materializing the
full [gaussians, height, width] interaction tensor at once.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.utils.torch_ops import require_torch


@dataclass
class TorchCamera:
    """Minimal camera protocol shared by the CUDA and fallback renderers."""

    image_height: int
    image_width: int
    world_view_transform: Any | None = None
    full_proj_transform: Any | None = None
    camera_center: Any | None = None
    FoVx: float = 0.7
    FoVy: float = 0.7
    image_name: str = "camera"


def _auto_chunk_size(height: int, width: int, gaussian_count: int) -> int:
    pixel_count = max(height * width, 1)
    # Keep each temporary dist/weight tensor small enough for 16GB-class GPUs.
    max_interactions = 16_000_000
    return max(1, min(128, gaussian_count, max_interactions // pixel_count))


def fallback_render(camera: TorchCamera, model: TorchGaussianModel, background: Any | None = None) -> dict[str, Any]:
    """Render a simple 2D Gaussian splat approximation with bounded memory."""

    torch = require_torch()
    device = model.device
    height = int(camera.image_height)
    width = int(camera.image_width)

    ys = torch.linspace(-1.0, 1.0, height, device=device)
    xs = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack([xx, yy], dim=-1)

    xyz = model.get_xyz
    xy = xyz[:, :2].clamp(-1.2, 1.2)
    depth = xyz[:, 2]
    scales = model.get_scaling[:, :2].mean(dim=-1).clamp(min=1e-3, max=0.5)
    opacity = model.get_opacity[:, 0].clamp(0.0, 1.0)
    colors = model.rgb.clamp(0.0, 1.0)

    gaussian_count = int(xy.shape[0])
    chunk_size = _auto_chunk_size(height, width, gaussian_count)

    weight_sum = torch.zeros((1, height, width), dtype=colors.dtype, device=device)
    color_accum = torch.zeros((3, height, width), dtype=colors.dtype, device=device)
    depth_accum = torch.zeros((1, height, width), dtype=colors.dtype, device=device)

    for start in range(0, gaussian_count, chunk_size):
        end = min(start + chunk_size, gaussian_count)
        chunk_xy = xy[start:end]
        chunk_scales = scales[start:end]
        chunk_opacity = opacity[start:end]
        chunk_colors = colors[start:end]
        chunk_depth = depth[start:end]

        dist2 = (pixels[None, :, :, :] - chunk_xy[:, None, None, :]).square().sum(dim=-1)
        weights = torch.exp(-dist2 / (2.0 * chunk_scales[:, None, None].square())) * chunk_opacity[:, None, None]

        weight_sum = weight_sum + weights.sum(dim=0, keepdim=True)
        color_accum = color_accum + (weights[:, None, :, :] * chunk_colors[:, :, None, None]).sum(dim=0)
        depth_accum = depth_accum + (weights * chunk_depth[:, None, None]).sum(dim=0, keepdim=True)

    alpha = weight_sum.squeeze(0).clamp(0.0, 1.0)
    denom = weight_sum.clamp(min=1e-6)
    image = color_accum / denom
    if background is None:
        background = torch.zeros((3,), dtype=torch.float32, device=device)
    image = image * alpha[None, :, :] + background[:, None, None] * (1.0 - alpha[None, :, :])

    depth_image = depth_accum / denom
    radii = scales * max(height, width)
    return {
        "render": image.clamp(0.0, 1.0),
        "viewspace_points": model.get_xyz,
        "visibility_filter": torch.nonzero(radii > 0, as_tuple=False).reshape(-1),
        "radii": radii,
        "depth": depth_image,
        "alpha": alpha,
        "valid_depth_mask": alpha > 1e-3,
    }
