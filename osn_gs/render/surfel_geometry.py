from __future__ import annotations

"""Depth-derived surface geometry for the 2DGS branch.

Direct port of `utils/point_utils.py` from the official 2DGS implementation
(`hbb1/2d-gaussian-splatting` @ 335ad61). These two functions produce the
depth-derived surface normal `N` of the paper's eq. 15,

    N(x, y) = (grad_x p_s  x  grad_y p_s) / |grad_x p_s  x  grad_y p_s|

by unprojecting the rendered surface depth map into camera-ray points `p_s`
and taking central finite differences in image space.

Fidelity: OFFICIAL_CODE_FAITHFUL. The only changes are (a) tensors follow the
input depth map's device instead of the upstream hard-coded ``device='cuda'``,
so the CPU-only focused tests can exercise this code, and (b) `torch` is
resolved through OSN-GS's `require_torch()`. Neither changes any produced
value.

Note the upstream `ndc2pix` here uses `W/2, H/2` offsets while the CUDA
rasterizer's own `compute_transmat` uses `(W-1)/2, (H-1)/2`. That half-pixel
inconsistency is upstream's and is preserved deliberately: the depth-derived
normal `N` and the rasterized normal are compared to each other in the
normal-consistency loss exactly as the official code compares them.
"""

from typing import Any

from osn_gs.utils.torch_ops import require_torch


def depths_to_points(view: Any, depthmap: Any) -> Any:
    """Unproject a (1, H, W) depth map into world-space points, (H*W, 3)."""

    torch = require_torch()
    device = depthmap.device
    c2w = (view.world_view_transform.T).inverse()
    W, H = int(view.image_width), int(view.image_height)
    ndc2pix = torch.tensor(
        [
            [W / 2, 0, 0, (W) / 2],
            [0, H / 2, 0, (H) / 2],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    ).T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3, :3].T

    grid_x, grid_y = torch.meshgrid(
        torch.arange(W, device=device).float(),
        torch.arange(H, device=device).float(),
        indexing="xy",
    )
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
    rays_o = c2w[:3, 3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points


def depth_to_normal(view: Any, depth: Any) -> Any:
    """Depth-derived surface normal map, (H, W, 3). Border pixels stay zero."""

    torch = require_torch()
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output
