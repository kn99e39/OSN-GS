from __future__ import annotations

"""OSN-GS 2DGS surfel rasterization entrypoint.

Direct port of `gaussian_renderer/__init__.py` from the official 2DGS
implementation (`hbb1/2d-gaussian-splatting` @ 335ad61) onto OSN-GS's
`TorchCamera` / `TorchGaussianSurfelModel` types.

The actual rasterization is the UNMODIFIED official CUDA kernel vendored at
`osn_gs/render/vendor/diff_surfel_rasterization` -- the perspective-correct
ray-splat intersection of paper eqs. 8-10, the object-space low-pass filter of
eq. 11, front-to-back alpha compositing, and the CUDA-side depth-distortion
accumulation. Nothing here re-derives that math, and there is deliberately no
torch fallback: an approximation of the ray-splat intersection would silently
invalidate the branch's 2DGS claim.

Outputs (all required by the paper's regularizers and by OSN-GS structural
evidence analysis):

| key | meaning |
|---|---|
| `render` | RGB, (3, H, W) |
| `rend_alpha` | accumulated alpha `A = 1 - T`, (1, H, W) |
| `rend_normal` | alpha-weighted rasterized surfel normal `sum_i w_i n_i`, world space, (3, H, W) |
| `rend_dist` | per-pixel depth distortion (paper eq. 13, CUDA form), (1, H, W) |
| `surf_depth` | surface depth, mean/median mix by `depth_ratio`, (1, H, W) |
| `surf_normal` | depth-derived surface normal `N` (paper eq. 15) scaled by detached alpha, (3, H, W) |
| `depth_expected` | alpha-normalized perspective-correct intersection depth, (1, H, W) |
| `depth_median` | median (T = 0.5 crossing) intersection depth, (1, H, W) |
| `radii` | per-surfel screen-space radius |
| `visibility_filter` | index tensor of surfels with `radii > 0` |
| `viewspace_points` | `means2D` proxy carrying the screen-projected 3D-center gradient |

`visibility_filter` is exposed as an INDEX tensor (not the official boolean
mask) because that is what OSN-GS's existing `add_densification_stats` /
`update_max_radii` already consume; `visibility_mask` carries the official
boolean form for anything that wants it.
"""

from dataclasses import dataclass
import math
from typing import Any

from osn_gs.render.diff_surfel_loader import diff_surfel_load_error, get_diff_surfel_backend
from osn_gs.render.surfel_geometry import depth_to_normal
from osn_gs.render.torch_fallback import TorchCamera
from osn_gs.utils.torch_ops import require_torch


@dataclass
class SurfelRasterizerConfig:
    """2DGS rasterization options.

    `depth_ratio` matches the official `PipelineParams.depth_ratio`: 0 selects
    the mean (expected) depth, 1 the median depth. Upstream default is 0
    ("0 works for most cases", unbounded/large scenes); the official DTU and
    TnT evaluation scripts pass `--depth_ratio 1.0` for bounded scenes.
    """

    depth_ratio: float = 0.0
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False
    # OSN_GS_ADAPTATION. The official 2DGS renderer returns the raw rasterized
    # image and feeds it unclamped to L1/D-SSIM. OSN-GS's 3DGS path clamps to
    # [0, 1] inside `OSNGaussianRasterizer._render_cuda`, so the vanilla arm of
    # this comparison trains against a clamped image. Clamping here too keeps
    # the photometric component identical across the two arms, which the fair-
    # comparison contract requires; it is NOT what upstream 2DGS does. Set
    # False to recover the exact official photometric path.
    clamp_render: bool = True


class OSNSurfelRasterizer:
    """Render `TorchGaussianSurfelModel` views through the official 2DGS kernel."""

    def __init__(self, config: SurfelRasterizerConfig | None = None) -> None:
        self.config = config or SurfelRasterizerConfig()
        backend = get_diff_surfel_backend()
        if backend is None:
            raise RuntimeError(
                "The 2DGS diff-surfel rasterizer is unavailable and this branch has no "
                "fallback: a torch approximation of the ray-splat intersection would not "
                f"be 2DGS. Loader error: {diff_surfel_load_error()}"
            )
        self._backend = backend
        self.backend_source = backend.source

    @property
    def has_cuda_backend(self) -> bool:
        return True

    def render(
        self,
        camera: TorchCamera,
        model: Any,
        background: Any | None = None,
        scaling_modifier: float = 1.0,
    ) -> dict[str, Any]:
        torch = require_torch()
        if background is None:
            background = torch.zeros((3,), dtype=torch.float32, device=model.device)

        settings_cls, rasterizer_cls = self._backend.settings_cls, self._backend.rasterizer_cls

        # `means2D` never feeds the forward pass. The 2DGS backward writes the
        # 3D center's gradient projected to screen space into it (paper sec.
        # 6.1: "we project the gradient of 3D center p_k onto the screen space
        # as an approximation"), which is what drives densification.
        screenspace_points = torch.zeros_like(model.get_xyz, requires_grad=True, device=model.device) + 0
        try:
            screenspace_points.retain_grad()
        except RuntimeError:
            pass

        raster_settings = settings_cls(
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            tanfovx=math.tan(float(camera.FoVx) * 0.5),
            tanfovy=math.tan(float(camera.FoVy) * 0.5),
            bg=background,
            scale_modifier=float(scaling_modifier),
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=model.active_sh_degree,
            campos=camera.camera_center,
            prefiltered=False,
            debug=self.config.debug,
        )
        rasterizer = rasterizer_cls(raster_settings=raster_settings)

        rendered_image, radii, allmap = rasterizer(
            means3D=model.get_xyz,
            means2D=screenspace_points,
            shs=model.get_features,
            colors_precomp=None,
            opacities=model.get_opacity,
            scales=model.get_scaling,
            rotations=model.get_rotation,
            cov3D_precomp=None,
        )

        package: dict[str, Any] = {
            "render": rendered_image.clamp(0.0, 1.0) if self.config.clamp_render else rendered_image,
            "render_unclamped": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_mask": radii > 0,
            "visibility_filter": torch.nonzero(radii > 0, as_tuple=False).reshape(-1),
            "radii": radii,
        }

        render_alpha = allmap[1:2]

        # Rasterized surfel normal, view space -> world space.
        render_normal = allmap[2:5]
        render_normal = (
            render_normal.permute(1, 2, 0) @ (camera.world_view_transform[:3, :3].T)
        ).permute(2, 0, 1)

        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        render_depth_expected = allmap[0:1] / render_alpha
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

        render_dist = allmap[6:7]

        depth_ratio = float(self.config.depth_ratio)
        surf_depth = render_depth_expected * (1 - depth_ratio) + depth_ratio * render_depth_median

        surf_normal = depth_to_normal(camera, surf_depth).permute(2, 0, 1)
        # `render_normal` is un-normalized (it is alpha-weighted), so the
        # depth-derived normal is scaled by the same accumulated alpha before
        # the two are compared. Detached upstream, kept detached here.
        surf_normal = surf_normal * render_alpha.detach()

        package.update(
            {
                "rend_alpha": render_alpha,
                "rend_normal": render_normal,
                "rend_dist": render_dist,
                "surf_depth": surf_depth,
                "surf_normal": surf_normal,
                "depth_expected": render_depth_expected,
                "depth_median": render_depth_median,
                # Names the rest of OSN-GS already expects from the 3DGS path.
                "depth": surf_depth,
                "valid_depth_mask": surf_depth.squeeze(0).abs() > 1e-8,
            }
        )
        return package
