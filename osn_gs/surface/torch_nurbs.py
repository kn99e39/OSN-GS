from __future__ import annotations

"""Torch 기반 NURBS-like surface utilities.

현재 구현은 "완전한 NURBS evaluator"라기보다는, OSN-GS 학습 루프를
구성하기 위한 differentiable parametric surface placeholder다.
control grid, uv parameter, smoothness loss의 인터페이스를 먼저 고정해두고,
추후 Cox-de Boor basis 기반의 정식 NURBS evaluator로 교체할 수 있게 했다.
"""

from dataclasses import dataclass
from typing import Any

from osn_gs.utils.torch_ops import require_torch


@dataclass
class TorchCurveSet:
    """여러 curve의 control point를 batch tensor로 보관한다."""

    # Shape: (C, K, 3). C는 curve 개수, K는 curve별 control point 개수.
    control_points: Any
    # 관측 curve인지, occlusion prediction으로 생긴 curve인지 표시한다.
    observed: Any


@dataclass
class TorchNURBSSurface:
    """OSN-GS가 uncertain Gaussian anchor로 사용하는 parametric surface."""

    # Shape: (U, V, 3). 현재 V=2로 base/occlusion edge를 의미한다.
    control_grid: Any
    # 정식 NURBS 확장을 위한 weight placeholder.
    weights: Any
    # degree 값은 저장하지만 현재 evaluator는 bilinear interpolation을 사용한다.
    degree_u: int = 2
    degree_v: int = 1
    # v <= observed_v_max는 관측 쪽, v > observed_v_max는 occluded 쪽으로 해석한다.
    observed_v_max: float = 0.5

    def evaluate(self, uv: Any) -> Any:
        """uv parameter를 3D surface point로 변환한다.

        현재는 u 방향으로 인접 control row를 선형 보간하고, v 방향으로
        base edge와 occlusion edge를 선형 보간한다.
        """

        torch = require_torch()
        uv = torch.as_tensor(uv, dtype=self.control_grid.dtype, device=self.control_grid.device)
        if uv.ndim == 1:
            uv = uv[None, :]

        # parameter domain은 [0, 1]^2로 고정한다.
        u = torch.clamp(uv[:, 0], 0.0, 1.0)
        v = torch.clamp(uv[:, 1], 0.0, 1.0)

        # u를 control_grid row index로 변환한다.
        pos = u * (self.control_grid.shape[0] - 1)
        lo = torch.floor(pos).long()
        hi = torch.clamp(lo + 1, max=self.control_grid.shape[0] - 1)
        t = (pos - lo.float())[:, None]

        # low는 observed/base side, high는 occluded side에 해당한다.
        low = (1.0 - t) * self.control_grid[lo, 0] + t * self.control_grid[hi, 0]
        high = (1.0 - t) * self.control_grid[lo, 1] + t * self.control_grid[hi, 1]
        return (1.0 - v[:, None]) * low + v[:, None] * high

    def smoothness(self) -> Any:
        """u 방향 second derivative penalty.

        control row가 급격히 꺾이는 surface hypothesis를 억제한다.
        """

        torch = require_torch()
        if self.control_grid.shape[0] < 3:
            return torch.zeros((), dtype=self.control_grid.dtype, device=self.control_grid.device)
        second = self.control_grid[:-2] - 2.0 * self.control_grid[1:-1] + self.control_grid[2:]
        return second.square().mean()


def fit_torch_base_curves(points: Any, curve_count: int = 4) -> TorchCurveSet:
    """관측 Gaussian center에서 base curve set을 추정한다.

    첫 구현은 PCA 주축으로 point를 정렬한 뒤 chunk별 3개 control point
    `[start, mean, end]`를 만드는 단순한 방법이다.
    """

    torch = require_torch()
    points = torch.as_tensor(points, dtype=torch.float32, device=points.device if hasattr(points, "device") else None)
    if points.numel() == 0:
        control = torch.zeros((0, 3, 3), dtype=torch.float32, device=points.device)
        return TorchCurveSet(control_points=control, observed=torch.zeros((0,), dtype=torch.bool, device=points.device))

    # PCA/SVD로 관측 point cloud의 주 방향을 잡는다.
    centered = points - points.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    axis = vh[0]

    # 주 방향 projection 값으로 정렬하면 curve 방향 chunking이 가능해진다.
    order = torch.argsort(points @ axis)
    sorted_points = points[order]
    curve_count = max(1, min(curve_count, sorted_points.shape[0]))
    chunks = torch.tensor_split(sorted_points, curve_count, dim=0)

    # 각 chunk를 작은 quadratic-like curve로 본다.
    controls = []
    for chunk in chunks:
        if chunk.shape[0] == 1:
            controls.append(chunk.repeat(3, 1))
        else:
            controls.append(torch.stack([chunk[0], chunk.mean(dim=0), chunk[-1]], dim=0))
    return TorchCurveSet(
        control_points=torch.stack(controls, dim=0),
        observed=torch.ones((len(controls),), dtype=torch.bool, device=points.device),
    )


def predict_torch_occlusion_curves(base_curves: TorchCurveSet, offset_scale: float = 0.25) -> TorchCurveSet:
    """base curve를 비관측 방향으로 평행 이동해 occlusion curve hypothesis를 만든다."""

    torch = require_torch()
    control = base_curves.control_points
    if control.shape[0] == 0:
        return TorchCurveSet(control_points=control, observed=torch.zeros_like(base_curves.observed))

    # curve 시작-끝 방향을 모아 전체 surface의 평균 tangent를 추정한다.
    directions = torch.nn.functional.normalize(control[:, -1] - control[:, 0], dim=-1)
    mean_dir = torch.nn.functional.normalize(directions.mean(dim=0), dim=0)

    # 임시 prior: world z축과 tangent의 cross product를 occlusion 방향으로 사용한다.
    reference = torch.tensor([0.0, 0.0, 1.0], dtype=control.dtype, device=control.device)
    normal = torch.cross(mean_dir, reference, dim=0)
    if torch.linalg.norm(normal) < 1e-5:
        # tangent가 z축과 거의 평행하면 cross product가 불안정하므로 z축 자체를 fallback으로 둔다.
        normal = reference
    normal = torch.nn.functional.normalize(normal, dim=0)
    return TorchCurveSet(
        control_points=control + normal.view(1, 1, 3) * offset_scale,
        observed=torch.zeros_like(base_curves.observed),
    )


def build_torch_surface(base_curves: TorchCurveSet, occlusion_curves: TorchCurveSet) -> TorchNURBSSurface:
    """base/occlusion curve를 surface control grid로 묶는다."""

    torch = require_torch()
    count = min(base_curves.control_points.shape[0], occlusion_curves.control_points.shape[0])
    if count == 0:
        raise ValueError("Cannot build NURBS surface without curves.")

    # 현재는 curve별 mean point를 surface row의 양 끝점으로 사용한다.
    # 추후에는 curve 전체 control point를 tensor-product surface basis로 연결할 예정이다.
    base = base_curves.control_points[:count].mean(dim=1)
    occ = occlusion_curves.control_points[:count].mean(dim=1)
    grid = torch.stack([base, occ], dim=1)
    weights = torch.ones(grid.shape[:2], dtype=grid.dtype, device=grid.device)
    return TorchNURBSSurface(control_grid=grid, weights=weights)


def sample_torch_occluded_surface(surface: TorchNURBSSurface, samples_u: int, samples_v: int) -> tuple[Any, Any]:
    """surface의 occluded domain에서 uncertain Gaussian 위치를 샘플링한다."""

    torch = require_torch()
    device = surface.control_grid.device

    # u는 curve 진행 방향, v는 observed -> occluded 방향이다.
    u = torch.linspace(0.0, 1.0, max(samples_u, 1), device=device)
    v = torch.linspace(surface.observed_v_max, 1.0, max(samples_v, 1), device=device)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)
    return surface.evaluate(uv), uv
