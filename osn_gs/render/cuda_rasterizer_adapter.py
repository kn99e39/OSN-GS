from __future__ import annotations

"""CUDA Gaussian rasterizer bridge.

OSN-GS의 `TorchGaussianModel`을 3DGS CUDA rasterizer가 기대하는 인자 형태로
넘기는 adapter다. target 환경에 `diff_gaussian_rasterization`이 설치되어 있으면
그 backend를 사용하고, 없으면 Torch fallback renderer로 내려간다.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.render.torch_fallback import TorchCamera, fallback_render
from osn_gs.utils.torch_ops import require_torch


@dataclass
class RasterizerPipelineOptions:
    """3DGS renderer option과 fallback 선택을 제어한다."""

    # 기존 3DGS 옵션과 이름을 맞춰둔 placeholder. 추후 SH python conversion 구현에 사용.
    convert_SHs_python: bool = False
    # covariance를 Python에서 미리 계산할지 여부. 현재 adapter는 scales/rotations를 넘긴다.
    compute_cov3D_python: bool = False
    # CUDA rasterizer debug flag.
    debug: bool = False
    # diff_gaussian_rasterization의 antialiasing option.
    antialiasing: bool = False
    # False면 CUDA backend가 있어도 fallback renderer를 강제로 사용한다.
    prefer_cuda_rasterizer: bool = True


class TorchRasterizerAdapter:
    """CUDA backend 유무를 감추고 동일한 render API를 제공한다."""

    def __init__(self, options: RasterizerPipelineOptions | None = None) -> None:
        self.options = options or RasterizerPipelineOptions()
        self._cuda_backend: Any | None = None
        if self.options.prefer_cuda_rasterizer:
            try:
                # import가 성공한다는 것은 target Python env에 3DGS CUDA extension이 설치됐다는 뜻이다.
                from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

                self._cuda_backend = (GaussianRasterizationSettings, GaussianRasterizer)
            except ImportError:
                # 개발 환경이나 macOS에서는 여기로 오며 fallback renderer를 사용한다.
                self._cuda_backend = None

    @property
    def has_cuda_backend(self) -> bool:
        # metrics.txt에 기록해 실제 결과가 CUDA rasterizer 기반인지 확인할 수 있게 한다.
        return self._cuda_backend is not None

    def render(self, camera: TorchCamera, model: TorchGaussianModel, background: Any | None = None) -> dict[str, Any]:
        """camera/model을 렌더링하고 3DGS train loop와 비슷한 dict를 반환한다."""

        torch = require_torch()
        if background is None:
            background = torch.zeros((3,), dtype=torch.float32, device=model.device)
        if self._cuda_backend is None:
            return fallback_render(camera, model, background)
        return self._render_cuda(camera, model, background)

    def _render_cuda(self, camera: TorchCamera, model: TorchGaussianModel, background: Any) -> dict[str, Any]:
        """diff_gaussian_rasterization 호출부."""

        torch = require_torch()
        settings_cls, rasterizer_cls = self._cuda_backend

        # screenspace_points는 3DGS densification 통계를 위한 gradient carrier다.
        screenspace_points = torch.zeros_like(model.get_xyz, requires_grad=True, device=model.device)
        try:
            screenspace_points.retain_grad()
        except RuntimeError:
            pass

        # camera는 우선 TorchCamera protocol만 맞추고, 실제 COLMAP camera 연결 시
        # world_view_transform/full_proj_transform/camera_center를 채워 넣으면 된다.
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
            debug=self.options.debug,
            antialiasing=self.options.antialiasing,
        )
        rasterizer = rasterizer_cls(raster_settings=raster_settings)

        # model property 이름은 기존 3DGS GaussianModel과 최대한 맞춰두었다.
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
        # 반환 key도 기존 3DGS train.py와 같은 이름을 사용한다.
        return {
            "render": rendered_image.clamp(0.0, 1.0),
            "viewspace_points": screenspace_points,
            "visibility_filter": torch.nonzero(radii > 0, as_tuple=False).reshape(-1),
            "radii": radii,
            "depth": depth_image,
        }
