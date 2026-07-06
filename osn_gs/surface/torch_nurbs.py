from __future__ import annotations

"""Torch 기반 NURBS-like surface utilities.

현재 구현은 "완전한 NURBS evaluator"라기보다는, OSN-GS 학습 루프를
구성하기 위한 differentiable parametric surface placeholder다.
Stage 1은 visible Gaussian geometry만 parameterize하고, occluded surface
생성은 별도 stage로 분리한다.
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
    """OSN-GS가 visible Gaussian geometry에 맞추는 parametric surface."""

    # Shape: (U, V, 3). Stage 1에서는 visible surface control grid를 의미한다.
    control_grid: Any
    # 정식 NURBS 확장을 위한 weight placeholder.
    weights: Any
    # degree 값은 저장하지만 현재 evaluator는 bilinear interpolation을 사용한다.
    degree_u: int = 2
    degree_v: int = 1
    # Stage 1 visible-only surface는 전체 v domain이 관측 surface다.
    observed_v_max: float = 1.0

    def evaluate(self, uv: Any) -> Any:
        """uv parameter를 3D surface point로 변환한다.

        현재는 정식 NURBS basis 대신 control grid 위 bilinear interpolation을
        사용한다. 인터페이스를 먼저 고정해두고 이후 NURBS evaluator로
        교체할 수 있게 둔다.
        """

        torch = require_torch()
        uv = torch.as_tensor(uv, dtype=self.control_grid.dtype, device=self.control_grid.device)
        if uv.ndim == 1:
            uv = uv[None, :]

        u = torch.clamp(uv[:, 0], 0.0, 1.0)
        v = torch.clamp(uv[:, 1], 0.0, 1.0)

        u_pos = u * (self.control_grid.shape[0] - 1)
        v_pos = v * (self.control_grid.shape[1] - 1)
        u0 = torch.floor(u_pos).long()
        v0 = torch.floor(v_pos).long()
        u1 = torch.clamp(u0 + 1, max=self.control_grid.shape[0] - 1)
        v1 = torch.clamp(v0 + 1, max=self.control_grid.shape[1] - 1)
        tu = (u_pos - u0.float())[:, None]
        tv = (v_pos - v0.float())[:, None]

        p00 = self.control_grid[u0, v0]
        p10 = self.control_grid[u1, v0]
        p01 = self.control_grid[u0, v1]
        p11 = self.control_grid[u1, v1]
        low = (1.0 - tu) * p00 + tu * p10
        high = (1.0 - tu) * p01 + tu * p11
        return (1.0 - tv) * low + tv * high

    def smoothness(self) -> Any:
        """control grid second derivative penalty."""

        torch = require_torch()
        terms = []
        if self.control_grid.shape[0] >= 3:
            second_u = self.control_grid[:-2] - 2.0 * self.control_grid[1:-1] + self.control_grid[2:]
            terms.append(second_u.square().mean())
        if self.control_grid.shape[1] >= 3:
            second_v = self.control_grid[:, :-2] - 2.0 * self.control_grid[:, 1:-1] + self.control_grid[:, 2:]
            terms.append(second_v.square().mean())
        if not terms:
            return torch.zeros((), dtype=self.control_grid.dtype, device=self.control_grid.device)
        return torch.stack(terms).mean()


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

    centered = points - points.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    axis = vh[0]

    order = torch.argsort(points @ axis)
    sorted_points = points[order]
    curve_count = max(1, min(curve_count, sorted_points.shape[0]))
    chunks = torch.tensor_split(sorted_points, curve_count, dim=0)

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


def fit_torch_visible_surface(
    points: Any,
    resolution_u: int = 8,
    resolution_v: int = 4,
    chunk_size: int = 4096,
) -> TorchNURBSSurface:
    """관측 Gaussian center만 사용해 visible surface parameter grid를 만든다.

    Stage 1은 occluded surface를 만들지 않는다. 대신 point cloud를 PCA 기반
    2D parameter domain으로 펼친 뒤, regular uv grid의 각 control point를
    주변 observed point의 weighted average로 채운다.
    """

    torch = require_torch()
    points = torch.as_tensor(points, dtype=torch.float32, device=points.device if hasattr(points, "device") else None)
    if points.numel() == 0:
        raise ValueError("Cannot fit a visible NURBS surface without observed points.")

    resolution_u = max(2, int(resolution_u))
    resolution_v = max(2, int(resolution_v))
    device = points.device
    dtype = points.dtype

    if points.shape[0] == 1:
        grid = points[0].view(1, 1, 3).repeat(resolution_u, resolution_v, 1)
        weights = torch.ones((resolution_u, resolution_v), dtype=dtype, device=device)
        return TorchNURBSSurface(control_grid=grid, weights=weights, observed_v_max=1.0)

    centered = points - points.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    axis_u = vh[0]
    axis_v = vh[1] if vh.shape[0] > 1 else _orthogonal_axis(axis_u)
    axes = torch.stack([axis_u, axis_v], dim=1)
    coords = centered @ axes

    coord_min = coords.min(dim=0).values
    coord_max = coords.max(dim=0).values
    span = torch.clamp(coord_max - coord_min, min=1e-6)
    uv_points = (coords - coord_min) / span

    u = torch.linspace(0.0, 1.0, resolution_u, dtype=dtype, device=device)
    v = torch.linspace(0.0, 1.0, resolution_v, dtype=dtype, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    grid_uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)

    neighbor_count = min(points.shape[0], max(4, min(16, points.shape[0])))
    chunk_size = max(1, int(chunk_size))
    controls = []
    for uv_chunk in torch.split(grid_uv, chunk_size, dim=0):
        distances = torch.cdist(uv_chunk, uv_points)
        nearest_dist, nearest_idx = torch.topk(distances, k=neighbor_count, largest=False, dim=1)
        neighbor_points = points[nearest_idx]
        # Inverse-distance weights keep the control grid on the visible point cloud
        # while still smoothing sparse COLMAP samples.
        blend_weights = 1.0 / torch.clamp(nearest_dist, min=1e-4)
        blend_weights = blend_weights / blend_weights.sum(dim=1, keepdim=True)
        controls.append((neighbor_points * blend_weights[..., None]).sum(dim=1))
    control = torch.cat(controls, dim=0)
    control_grid = control.reshape(resolution_u, resolution_v, 3)
    weights = torch.ones((resolution_u, resolution_v), dtype=dtype, device=device)
    return TorchNURBSSurface(control_grid=control_grid, weights=weights, observed_v_max=1.0)


def _orthogonal_axis(axis: Any) -> Any:
    """Return a stable unit vector orthogonal to ``axis``."""

    torch = require_torch()
    reference = torch.tensor([0.0, 0.0, 1.0], dtype=axis.dtype, device=axis.device)
    candidate = torch.cross(axis, reference, dim=0)
    if torch.linalg.norm(candidate) < 1e-5:
        reference = torch.tensor([0.0, 1.0, 0.0], dtype=axis.dtype, device=axis.device)
        candidate = torch.cross(axis, reference, dim=0)
    return torch.nn.functional.normalize(candidate, dim=0)


def predict_torch_occlusion_curves(base_curves: TorchCurveSet, offset_scale: float = 0.25) -> TorchCurveSet:
    """base curve를 비관측 방향으로 평행 이동해 occlusion curve hypothesis를 만든다.

    Stage 2용 legacy helper다. Stage 1 visible reconstruction path에서는 호출하지 않는다.
    """

    torch = require_torch()
    control = base_curves.control_points
    if control.shape[0] == 0:
        return TorchCurveSet(control_points=control, observed=torch.zeros_like(base_curves.observed))

    directions = torch.nn.functional.normalize(control[:, -1] - control[:, 0], dim=-1)
    mean_dir = torch.nn.functional.normalize(directions.mean(dim=0), dim=0)

    reference = torch.tensor([0.0, 0.0, 1.0], dtype=control.dtype, device=control.device)
    normal = torch.cross(mean_dir, reference, dim=0)
    if torch.linalg.norm(normal) < 1e-5:
        normal = reference
    normal = torch.nn.functional.normalize(normal, dim=0)
    return TorchCurveSet(
        control_points=control + normal.view(1, 1, 3) * offset_scale,
        observed=torch.zeros_like(base_curves.observed),
    )


def build_torch_surface(base_curves: TorchCurveSet, occlusion_curves: TorchCurveSet) -> TorchNURBSSurface:
    """base/occlusion curve를 surface control grid로 묶는다.

    Stage 2용 legacy helper다. Visible-only reconstruction은
    `fit_torch_visible_surface`를 사용한다.
    """

    torch = require_torch()
    count = min(base_curves.control_points.shape[0], occlusion_curves.control_points.shape[0])
    if count == 0:
        raise ValueError("Cannot build NURBS surface without curves.")

    base = base_curves.control_points[:count].mean(dim=1)
    occ = occlusion_curves.control_points[:count].mean(dim=1)
    grid = torch.stack([base, occ], dim=1)
    weights = torch.ones(grid.shape[:2], dtype=grid.dtype, device=grid.device)
    return TorchNURBSSurface(control_grid=grid, weights=weights, observed_v_max=0.5)


def sample_torch_occluded_surface(surface: TorchNURBSSurface, samples_u: int, samples_v: int) -> tuple[Any, Any]:
    """surface의 occluded domain에서 uncertain Gaussian 위치를 샘플링한다.

    Stage 2용 legacy helper다. Stage 1 visible reconstruction path에서는 호출하지 않는다.
    """

    torch = require_torch()
    device = surface.control_grid.device

    u = torch.linspace(0.0, 1.0, max(samples_u, 1), device=device)
    v = torch.linspace(surface.observed_v_max, 1.0, max(samples_v, 1), device=device)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)
    return surface.evaluate(uv), uv
