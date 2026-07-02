from __future__ import annotations

"""OSN-GS Gaussian rasterizer.

This is the first-class rendering entrypoint for the torch training path. It
owns backend selection, translates OSN-GS model/camera tensors into the CUDA
rasterizer call, and only falls back to the slow torch renderer for small debug
cases.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.render.diff_gaussian_loader import diff_gaussian_load_error, get_diff_gaussian_backend
from osn_gs.render.torch_fallback import TorchCamera, fallback_render
from osn_gs.utils.torch_ops import require_torch


@dataclass
class GaussianRasterizerConfig:
    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False
    antialiasing: bool = False
    prefer_cuda: bool = True
    allow_fallback: bool = False


class OSNGaussianRasterizer:
    """Render ``TorchGaussianModel`` views through the OSN-GS rasterization API."""

    def __init__(self, config: GaussianRasterizerConfig | None = None) -> None:
        self.config = config or GaussianRasterizerConfig()
        self._cuda_backend: Any | None = None
        self.backend_source = "fallback"
        if self.config.prefer_cuda:
            backend = get_diff_gaussian_backend()
            if backend is not None:
                self._cuda_backend = (backend.settings_cls, backend.rasterizer_cls)
                self.backend_source = backend.source

    @property
    def has_cuda_backend(self) -> bool:
        return self._cuda_backend is not None

    def render(self, camera: TorchCamera, model: TorchGaussianModel, background: Any | None = None) -> dict[str, Any]:
        torch = require_torch()
        if background is None:
            background = torch.zeros((3,), dtype=torch.float32, device=model.device)
        if self._cuda_backend is None:
            if self.config.allow_fallback or not self.config.prefer_cuda:
                return fallback_render(camera, model, background)
            backend_error = diff_gaussian_load_error()
            raise RuntimeError(
                "Diff Gaussian rasterizer is unavailable, and training fallback is disabled to avoid excessive VRAM use."
                f" Loader error: {backend_error}"
            )
        try:
            return self._render_cuda(camera, model, background)
        except Exception as exc:
            if self.config.allow_fallback or not self.config.prefer_cuda:
                self._cuda_backend = None
                self.backend_source = "fallback"
                print(f"[OSN-GS] CUDA rasterizer failed; using chunked torch fallback: {exc}", flush=True)
                return fallback_render(camera, model, background)
            raise RuntimeError("Diff Gaussian rasterizer failed during rendering.") from exc

    def _render_cuda(self, camera: TorchCamera, model: TorchGaussianModel, background: Any) -> dict[str, Any]:
        torch = require_torch()
        settings_cls, rasterizer_cls = self._cuda_backend
        screenspace_points = torch.zeros_like(model.get_xyz, requires_grad=True, device=model.device)
        try:
            screenspace_points.retain_grad()
        except RuntimeError:
            pass

        raster_settings = settings_cls(
            image_height=int(camera.image_height),
            image_width=int(camera.image_width),
            tanfovx=float(torch.tan(torch.tensor(camera.FoVx * 0.5)).item()),
            tanfovy=float(torch.tan(torch.tensor(camera.FoVy * 0.5)).item()),
            bg=background,
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=model.active_sh_degree,
            campos=camera.camera_center,
            prefiltered=False,
            debug=self.config.debug,
            antialiasing=self.config.antialiasing,
        )
        rasterizer = rasterizer_cls(raster_settings=raster_settings)
        rendered_image, radii, depth_image = rasterizer(
            means3D=model.get_xyz,
            means2D=screenspace_points,
            shs=model.get_features,
            colors_precomp=None,
            opacities=model.get_opacity,
            scales=model.get_scaling,
            rotations=model.get_rotation,
            cov3D_precomp=None,
        )
        return {
            "render": rendered_image.clamp(0.0, 1.0),
            "viewspace_points": screenspace_points,
            "visibility_filter": torch.nonzero(radii > 0, as_tuple=False).reshape(-1),
            "radii": radii,
            "depth": depth_image,
        }

    def _raise_if_fallback_is_too_large(self, camera: TorchCamera, model: TorchGaussianModel) -> None:
        gaussian_count = max(len(model), 1)
        pixel_count = max(int(camera.image_height) * int(camera.image_width), 1)
        interaction_count = gaussian_count * pixel_count
        if interaction_count < 150_000_000:
            return
        backend_error = diff_gaussian_load_error()
        details = f"Fallback renderer disabled for large scenes ({gaussian_count} gaussians x {pixel_count} pixels)."
        if backend_error is not None:
            details += f" Rasterizer load/build error: {backend_error}"
        raise RuntimeError(
            details
            + " Use the vendored CUDA rasterizer, lower --image_downscale / --train_resolution_scale, or reduce Gaussian count."
        )
