from __future__ import annotations

"""Torch 기반 NURBS surface utilities.

`TorchNURBSSurface.evaluate()`는 Cox-de Boor recursion으로 계산한 clamped
uniform B-spline basis function을 이용해 rational tensor-product NURBS를
평가한다 (`weights`가 모두 1이면 non-rational B-spline으로 축소된다).
control grid는 초기 voxel bootstrap 이후 persistent trainable parameter로
유지되며 매 iteration surface loss의 backprop으로 갱신된다. 주기적인
maintenance는 전체 voxel/NURBS를 재생성하지 않고 품질을 검사하며, 지속적으로
실패한 patch 내부에서만 local voxel correction을 허용한다. Stage 1은 visible
Gaussian geometry만 parameterize하고, occluded surface 생성은 별도 stage로 분리한다.
"""

from dataclasses import dataclass, field
from typing import Any

from osn_gs.utils.torch_ops import require_torch


def _effective_degree(control_point_count: int, degree: int) -> int:
    """Clamp a requested NURBS degree to what the control point count supports."""

    return max(0, min(int(degree), max(int(control_point_count) - 1, 0)))


def _clamped_knot_vector(control_point_count: int, degree: int, dtype: Any, device: Any) -> Any:
    """Return an open/clamped uniform knot vector for a tensor-product NURBS axis.

    Length is ``control_point_count + degree + 1``. The first and last
    ``degree + 1`` knots are pinned to 0 and 1 so the surface interpolates
    the first/last control point row, matching standard clamped NURBS.
    """

    torch = require_torch()
    n_ctrl = max(int(control_point_count), 1)
    degree = _effective_degree(n_ctrl, degree)
    n_interior = n_ctrl - degree - 1

    knots = torch.empty((n_ctrl + degree + 1,), dtype=dtype, device=device)
    knots[: degree + 1] = 0.0
    if n_interior > 0:
        knots[degree + 1 : degree + 1 + n_interior] = torch.linspace(
            0.0, 1.0, n_interior + 2, dtype=dtype, device=device
        )[1:-1]
    knots[degree + 1 + n_interior :] = 1.0
    return knots


def _bspline_basis_pair(u: Any, degree: int, knots: Any, control_point_count: int) -> tuple[Any, Any | None]:
    """Vectorized Cox-de Boor recursion returning degree-p and degree-(p-1) bases.

    Returns ``(basis, lower)`` where ``basis`` is the ``(Q, control_point_count)``
    degree-``degree`` basis matrix and ``lower`` is the degree-``degree - 1``
    basis (or ``None`` when ``degree == 0``). ``lower`` feeds the standard
    B-spline derivative formula. ``knots``/``degree`` are not trainable, so the
    recursion only needs to be differentiable with respect to ``u``.
    """

    torch = require_torch()
    eps = 1e-7
    knot_min = float(knots[0])
    knot_max = float(knots[-1])
    u = torch.clamp(u, knot_min, knot_max - eps)
    span_count = int(knots.shape[0]) - 1
    u_col = u.view(-1, 1)

    # `u` is clamped strictly below `knot_max`, so the half-open interval test
    # below already resolves it into the last non-degenerate span even though
    # clamped/open knot vectors repeat `knot_max` `degree + 1` times.
    left = knots[:-1].view(1, -1)
    right = knots[1:].view(1, -1)
    basis = ((u_col >= left) & (u_col < right)).to(u.dtype)

    lower = None
    for d in range(1, int(degree) + 1):
        if d == int(degree):
            lower = basis
        width = span_count - d
        left_den = (knots[d : d + width] - knots[:width]).view(1, -1)
        left_term = torch.where(
            left_den > eps,
            (u_col - knots[:width].view(1, -1)) / torch.clamp(left_den, min=eps) * basis[:, :width],
            torch.zeros_like(basis[:, :width]),
        )
        right_den = (knots[d + 1 : d + 1 + width] - knots[1 : 1 + width]).view(1, -1)
        right_term = torch.where(
            right_den > eps,
            (knots[d + 1 : d + 1 + width].view(1, -1) - u_col) / torch.clamp(right_den, min=eps) * basis[:, 1 : 1 + width],
            torch.zeros_like(basis[:, 1 : 1 + width]),
        )
        basis = left_term + right_term

    return basis[:, :control_point_count], lower


def _bspline_basis(u: Any, degree: int, knots: Any, control_point_count: int) -> Any:
    """Degree-``degree`` Cox-de Boor basis matrix of shape ``(Q, control_point_count)``."""

    return _bspline_basis_pair(u, degree, knots, control_point_count)[0]


def _bspline_basis_derivative(lower: Any | None, degree: int, knots: Any, control_point_count: int) -> Any:
    """First derivative of the degree-``degree`` basis from the degree-``degree - 1`` basis.

    Uses ``N'_{i,p} = p * (N_{i,p-1} / (t_{i+p} - t_i) - N_{i+1,p-1} / (t_{i+p+1} - t_{i+1}))``.
    Zero-width denominators contribute zero, matching the clamped-knot convention.
    """

    torch = require_torch()
    if lower is None or int(degree) == 0:
        raise ValueError("Degree-0 basis has no derivative table; handle degree 0 at the caller.")
    eps = 1e-7
    p = int(degree)
    n = int(control_point_count)
    den_left = (knots[p : p + n] - knots[:n]).view(1, -1)
    den_right = (knots[p + 1 : p + 1 + n] - knots[1 : 1 + n]).view(1, -1)
    left = torch.where(
        den_left > eps,
        lower[:, :n] / torch.clamp(den_left, min=eps),
        torch.zeros_like(lower[:, :n]),
    )
    right = torch.where(
        den_right > eps,
        lower[:, 1 : 1 + n] / torch.clamp(den_right, min=eps),
        torch.zeros_like(lower[:, 1 : 1 + n]),
    )
    return float(p) * (left - right)


@dataclass
class TorchCurveSet:
    """여러 curve의 control point를 batch tensor로 보관한다."""

    # Shape: (C, K, 3). C는 curve 개수, K는 curve별 control point 개수.
    control_points: Any
    # 관측 curve인지, occlusion prediction으로 생긴 curve인지 표시한다.
    observed: Any


@dataclass
class NURBSFitRoundDiagnostics:
    control_grid_after_lsq: Any
    uv_after_footpoint: Any


@dataclass
class NURBSFitDiagnostics:
    fitting_points: Any
    point_weights: Any | None
    initial_uv: Any
    idw_seed_control_grid: Any
    rounds: list[NURBSFitRoundDiagnostics]
    final_control_grid: Any
    final_weights: Any
    final_gaussian_indices: Any | None = None
    final_gaussian_uv: Any | None = None


@dataclass
class TorchNURBSSurface:
    """OSN-GS가 visible Gaussian geometry에 맞추는 parametric surface."""

    # Shape: (U, V, 3). Stage 1에서는 visible surface control grid를 의미한다.
    control_grid: Any
    # Rational NURBS weight. 모두 1이면 non-rational B-spline과 동일하다.
    weights: Any
    # Clamped uniform knot vector 생성에 사용하는 NURBS degree.
    # control point 수가 부족한 축은 평가 시점에 degree가 자동으로 내려간다.
    degree_u: int = 2
    degree_v: int = 2
    # Stage 1 visible-only surface는 전체 v domain이 관측 surface다.
    observed_v_max: float = 1.0
    # Optional UV trimming mask (R, R) over [0, 1]^2: True where the patch is
    # supported by observed data. ``None`` means the whole domain is valid.
    # Consumers (renderer export, support metrics) restrict the surface to the
    # supported cells so it is not drawn/measured where there is no data.
    uv_support_mask: Any | None = None
    # Knot vectors depend only on the surface structure, not on trainable
    # control-point coordinates or rational weights. Keep them out of the
    # public constructor/checkpoint contract and rebuild lazily if that
    # structure (or tensor placement) changes.
    _knot_cache_key: tuple[Any, ...] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _cached_knots_u: Any | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _cached_knots_v: Any | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def support(self, uv: Any) -> Any:
        """Boolean mask of whether each ``uv`` lies in the trimmed (supported) domain."""

        torch = require_torch()
        uv = torch.as_tensor(uv, dtype=self.control_grid.dtype, device=self.control_grid.device)
        if uv.ndim == 1:
            uv = uv[None, :]
        if self.uv_support_mask is None:
            return torch.ones((uv.shape[0],), dtype=torch.bool, device=uv.device)
        mask = torch.as_tensor(self.uv_support_mask, dtype=torch.bool, device=uv.device)
        res_u, res_v = int(mask.shape[0]), int(mask.shape[1])
        cell_u = torch.clamp((uv[:, 0] * res_u).long(), 0, res_u - 1)
        cell_v = torch.clamp((uv[:, 1] * res_v).long(), 0, res_v - 1)
        return mask[cell_u, cell_v]

    def _basis_tables(self, uv: Any) -> tuple[Any, Any, Any | None, Any | None]:
        """Return ``(basis_u, basis_v, dbasis_u, dbasis_v)`` for query ``uv``.

        Derivative tables are ``None`` when the effective degree on that axis is 0
        (a single control point row cannot vary along that axis).
        """

        torch = require_torch()
        uv = torch.as_tensor(uv, dtype=self.control_grid.dtype, device=self.control_grid.device)
        if uv.ndim == 1:
            uv = uv[None, :]
        n_u = int(self.control_grid.shape[0])
        n_v = int(self.control_grid.shape[1])
        dtype, device = self.control_grid.dtype, self.control_grid.device
        degree_u = _effective_degree(n_u, self.degree_u)
        degree_v = _effective_degree(n_v, self.degree_v)
        cache_key = (n_u, n_v, degree_u, degree_v, dtype, device)
        if (
            self._knot_cache_key != cache_key
            or self._cached_knots_u is None
            or self._cached_knots_v is None
        ):
            self._cached_knots_u = _clamped_knot_vector(n_u, degree_u, dtype, device)
            self._cached_knots_v = _clamped_knot_vector(n_v, degree_v, dtype, device)
            self._knot_cache_key = cache_key
        knots_u = self._cached_knots_u
        knots_v = self._cached_knots_v

        u = torch.clamp(uv[:, 0], 0.0, 1.0)
        v = torch.clamp(uv[:, 1], 0.0, 1.0)
        basis_u, lower_u = _bspline_basis_pair(u, degree_u, knots_u, n_u)
        basis_v, lower_v = _bspline_basis_pair(v, degree_v, knots_v, n_v)
        dbasis_u = (
            _bspline_basis_derivative(lower_u, degree_u, knots_u, n_u) if degree_u > 0 else None
        )
        dbasis_v = (
            _bspline_basis_derivative(lower_v, degree_v, knots_v, n_v) if degree_v > 0 else None
        )
        return basis_u, basis_v, dbasis_u, dbasis_v

    def _rational_point(self, basis_u: Any, basis_v: Any) -> tuple[Any, Any]:
        """Evaluate the rational surface point and its weight denominator."""

        torch = require_torch()
        weighted_control = self.control_grid * self.weights[..., None]
        numerator = torch.einsum("qi,qj,ijc->qc", basis_u, basis_v, weighted_control)
        denominator = torch.einsum("qi,qj,ij->q", basis_u, basis_v, self.weights)
        denominator = torch.clamp(denominator, min=1e-8)
        return numerator / denominator[:, None], denominator

    def evaluate(self, uv: Any) -> Any:
        """uv parameter를 3D surface point로 변환한다.

        Clamped uniform knot vector 위에서 Cox-de Boor basis function을
        계산하고, `weights`로 가중한 rational tensor-product NURBS
        (`sum_ij N_i(u) N_j(v) w_ij P_ij / sum_ij N_i(u) N_j(v) w_ij`)를
        평가한다. control point 개수가 degree보다 작으면 degree를 자동으로
        낮춰 안전하게 평가한다.
        """

        basis_u, basis_v, _, _ = self._basis_tables(uv)
        return self._rational_point(basis_u, basis_v)[0]

    def evaluate_with_derivatives(self, uv: Any) -> tuple[Any, Any, Any]:
        """Return ``(S, dS/du, dS/dv)`` of the rational surface at ``uv``.

        Rational derivative via the quotient rule: with ``A = sum N w P`` and
        ``W = sum N w``, ``S_u = (A_u - W_u * S) / W``. Axes whose effective
        degree is 0 return a zero derivative.
        """

        torch = require_torch()
        basis_u, basis_v, dbasis_u, dbasis_v = self._basis_tables(uv)
        weighted_control = self.control_grid * self.weights[..., None]
        point, denominator = self._rational_point(basis_u, basis_v)

        def _partial(table_u: Any, table_v: Any) -> Any:
            numerator_d = torch.einsum("qi,qj,ijc->qc", table_u, table_v, weighted_control)
            denominator_d = torch.einsum("qi,qj,ij->q", table_u, table_v, self.weights)
            return (numerator_d - denominator_d[:, None] * point) / denominator[:, None]

        zeros = torch.zeros_like(point)
        deriv_u = _partial(dbasis_u, basis_v) if dbasis_u is not None else zeros
        deriv_v = _partial(basis_u, dbasis_v) if dbasis_v is not None else zeros
        return point, deriv_u, deriv_v

    def normals(self, uv: Any) -> Any:
        """Unit surface normal ``normalize(S_u x S_v)`` at each ``uv`` query."""

        torch = require_torch()
        _, deriv_u, deriv_v = self.evaluate_with_derivatives(uv)
        return torch.nn.functional.normalize(
            torch.cross(deriv_u, deriv_v, dim=-1), dim=-1, eps=1e-12
        )

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


def fit_torch_base_curves(points: Any, curve_count: int = 4, patch_ids: Any | None = None) -> TorchCurveSet:
    """관측 Gaussian center에서 base curve set을 추정한다.

    첫 구현은 PCA 주축으로 point를 정렬한 뒤 chunk별 3개 control point
    `[start, mean, end]`를 만드는 단순한 방법이다.
    """

    torch = require_torch()
    points = torch.as_tensor(points, dtype=torch.float32, device=points.device if hasattr(points, "device") else None)
    if points.numel() == 0:
        control = torch.zeros((0, 3, 3), dtype=torch.float32, device=points.device)
        return TorchCurveSet(control_points=control, observed=torch.zeros((0,), dtype=torch.bool, device=points.device))

    if patch_ids is not None:
        patch_ids = torch.as_tensor(patch_ids, dtype=torch.long, device=points.device).reshape(-1)
        if int(patch_ids.numel()) == int(points.shape[0]) and int(torch.unique(patch_ids).numel()) > 1:
            curves = []
            for patch_id in torch.unique(patch_ids, sorted=True):
                patch_points = points[patch_ids == patch_id]
                local_count = max(1, min(int(curve_count), int(patch_points.shape[0])))
                curves.append(fit_torch_base_curves(patch_points, local_count).control_points)
            control = torch.cat(curves, dim=0)
            return TorchCurveSet(
                control_points=control,
                observed=torch.ones((control.shape[0],), dtype=torch.bool, device=points.device),
            )

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


@dataclass
class UVFrame:
    """Affine 3D->UV chart: project onto two axes, then min-max normalize.

    Matches the parameterization convention of :func:`pca_parameterize_points`
    so a polygon mapped through the same frame lands in the same UV domain the
    fitted chart uses. Stage 1 builds this from each voxel's local plane
    tangents; the origin is the point set's mean/centroid.
    """

    origin: Any  # (3,)
    axis_u: Any  # (3,)
    axis_v: Any  # (3,)
    coord_min: Any  # (2,)
    span: Any  # (2,)

    def apply(self, points: Any, clamp: bool = True) -> Any:
        torch = require_torch()
        points = torch.as_tensor(points, dtype=self.origin.dtype, device=self.origin.device)
        coords = (points - self.origin) @ torch.stack([self.axis_u, self.axis_v], dim=1)
        uv = (coords - self.coord_min) / self.span
        return torch.clamp(uv, 0.0, 1.0) if clamp else uv

    def to_world(self, uv: Any) -> Any:
        """Map ``(Q, 2)`` UV back to 3D points on the frame's tangent plane."""

        torch = require_torch()
        uv = torch.as_tensor(uv, dtype=self.origin.dtype, device=self.origin.device)
        coords = uv * self.span + self.coord_min
        return (
            self.origin
            + coords[:, 0:1] * self.axis_u[None, :]
            + coords[:, 1:2] * self.axis_v[None, :]
        )


def uv_frame_from_axes(points: Any, origin: Any, axis_u: Any, axis_v: Any) -> UVFrame:
    """UV frame over the given tangent axes, normalized to the points' extent."""

    torch = require_torch()
    points = torch.as_tensor(points, dtype=torch.float32, device=points.device if hasattr(points, "device") else None)
    axes = torch.stack([axis_u, axis_v], dim=1).to(points.dtype)
    coords = (points - origin) @ axes
    coord_min = coords.min(dim=0).values
    span = torch.clamp(coords.max(dim=0).values - coord_min, min=1e-6)
    return UVFrame(origin=origin, axis_u=axes[:, 0], axis_v=axes[:, 1], coord_min=coord_min, span=span)


def pca_parameterize_points(points: Any) -> Any:
    """Unroll points into a normalized ``[0, 1]^2`` PCA parameter domain.

    This is the shared initial parameterization used by both the IDW seed fit
    and the least-squares fit before foot-point reprojection replaces it.
    """

    torch = require_torch()
    points = torch.as_tensor(points, dtype=torch.float32, device=points.device if hasattr(points, "device") else None)
    if points.numel() == 0:
        return torch.zeros((0, 2), dtype=torch.float32, device=points.device)
    if int(points.shape[0]) == 1:
        return torch.full((1, 2), 0.5, dtype=torch.float32, device=points.device)
    centered = points - points.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    axis_u = vh[0]
    axis_v = vh[1] if vh.shape[0] > 1 else _orthogonal_axis(axis_u)
    coords = centered @ torch.stack([axis_u, axis_v], dim=1)
    coord_min = coords.min(dim=0).values
    span = torch.clamp(coords.max(dim=0).values - coord_min, min=1e-6)
    return torch.clamp((coords - coord_min) / span, 0.0, 1.0)


def pca_extent_aspect_ratio(points: Any, min_ratio: float = 0.1, max_ratio: float = 10.0) -> float:
    """Return the PCA in-plane extent[0]/extent[1] ratio of ``points``, clamped.

    Uses the same PCA axes as :func:`pca_parameterize_points` so the reported
    aspect ratio matches what that function normalizes away.
    """

    torch = require_torch()
    points = torch.as_tensor(points, dtype=torch.float32, device=points.device if hasattr(points, "device") else None)
    if int(points.shape[0]) < 2:
        return 1.0
    centered = points - points.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    axis_u = vh[0]
    axis_v = vh[1] if vh.shape[0] > 1 else _orthogonal_axis(axis_u)
    coords = centered @ torch.stack([axis_u, axis_v], dim=1)
    extent = coords.max(dim=0).values - coords.min(dim=0).values
    ratio = float(extent[0] / extent[1].clamp_min(1e-6))
    return max(min_ratio, min(max_ratio, ratio))


def fit_torch_visible_surface(
    points: Any,
    resolution_u: int = 8,
    resolution_v: int = 4,
    chunk_size: int = 4096,
    degree_u: int = 2,
    degree_v: int = 2,
    initial_uv: Any | None = None,
) -> TorchNURBSSurface:
    """관측 Gaussian center만 사용해 visible surface parameter grid를 만든다.

    Stage 1은 occluded surface를 만들지 않는다. 대신 point cloud를 PCA 기반
    2D parameter domain으로 펼친 뒤, regular uv grid의 각 control point를
    주변 observed point의 weighted average로 채운다. 이 결과는 초기값이며,
    실제 fitting 품질은 `fit_torch_visible_surface_lsq`가 담당한다.
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
        return TorchNURBSSurface(
            control_grid=grid, weights=weights, degree_u=degree_u, degree_v=degree_v, observed_v_max=1.0
        )

    uv_points = pca_parameterize_points(points) if initial_uv is None else torch.as_tensor(
        initial_uv, dtype=dtype, device=device
    )

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
    return TorchNURBSSurface(
        control_grid=control_grid, weights=weights, degree_u=degree_u, degree_v=degree_v, observed_v_max=1.0
    )


def _second_difference_penalty(n_u: int, n_v: int, dtype: Any, device: Any) -> Any:
    """Discrete thin-plate style penalty over the flattened ``(n_u * n_v)`` control grid.

    ``L = (D_u^T D_u) ⊗ I_v + I_u ⊗ (D_v^T D_v)`` with second-difference operators
    per axis, matching the second-derivative smoothness used by ``smoothness()``.
    """

    torch = require_torch()
    n = n_u * n_v
    penalty = torch.zeros((n, n), dtype=dtype, device=device)
    if n_u >= 3:
        diff_u = torch.zeros((n_u - 2, n_u), dtype=dtype, device=device)
        rows = torch.arange(n_u - 2, device=device)
        diff_u[rows, rows] = 1.0
        diff_u[rows, rows + 1] = -2.0
        diff_u[rows, rows + 2] = 1.0
        penalty = penalty + torch.kron(diff_u.T @ diff_u, torch.eye(n_v, dtype=dtype, device=device))
    if n_v >= 3:
        diff_v = torch.zeros((n_v - 2, n_v), dtype=dtype, device=device)
        rows = torch.arange(n_v - 2, device=device)
        diff_v[rows, rows] = 1.0
        diff_v[rows, rows + 1] = -2.0
        diff_v[rows, rows + 2] = 1.0
        penalty = penalty + torch.kron(torch.eye(n_u, dtype=dtype, device=device), diff_v.T @ diff_v)
    return penalty


def _lsq_normal_system(
    points: Any,
    uv: Any,
    surface: TorchNURBSSurface,
    smoothness_lambda: float,
    tikhonov_lambda: float,
    chunk_size: int,
    point_weights: Any | None,
) -> tuple[Any, Any, float]:
    """Assemble the regularized normal-equations system (matrix, rhs) for one
    surface's control grid at fixed UVs, WITHOUT solving it.

    Factored out of ``_solve_control_grid_lsq`` (Phase 5 Step 5-A,
    ``docs/worklogs/55``) so a coupled multi-surface fit
    (``fit_coupled_wedge_ring_lsq``) can accumulate several surfaces' own
    local systems into one shared global system before a single joint solve,
    instead of each surface solving independently. Returns ``(matrix, rhs,
    total_weight)`` at the SAME per-surface scale ``_solve_control_grid_lsq``
    used (matrix/rhs still need dividing by ``total_weight`` and combining
    with the smoothness/Tikhonov terms -- callers that solve a single surface
    should keep using ``_solve_control_grid_lsq``, which now calls this).
    """

    torch = require_torch()
    n_u = int(surface.control_grid.shape[0])
    n_v = int(surface.control_grid.shape[1])
    n = n_u * n_v
    dtype, device = surface.control_grid.dtype, surface.control_grid.device

    normal_matrix = torch.zeros((n, n), dtype=dtype, device=device)
    normal_rhs = torch.zeros((n, 3), dtype=dtype, device=device)
    total_weight = 0.0
    chunk_size = max(1, int(chunk_size))
    for start in range(0, int(points.shape[0]), chunk_size):
        end = min(start + chunk_size, int(points.shape[0]))
        basis_u, basis_v, _, _ = surface._basis_tables(uv[start:end])
        rows = torch.einsum("qi,qj->qij", basis_u, basis_v).reshape(end - start, n)
        chunk_points = points[start:end]
        if point_weights is not None:
            w = point_weights[start:end].reshape(-1, 1)
            normal_matrix = normal_matrix + rows.T @ (rows * w)
            normal_rhs = normal_rhs + rows.T @ (chunk_points * w)
            total_weight += float(w.sum())
        else:
            normal_matrix = normal_matrix + rows.T @ rows
            normal_rhs = normal_rhs + rows.T @ chunk_points
            total_weight += float(end - start)
    return normal_matrix, normal_rhs, total_weight


def _solve_control_grid_lsq(
    points: Any,
    uv: Any,
    surface: TorchNURBSSurface,
    smoothness_lambda: float,
    tikhonov_lambda: float,
    chunk_size: int,
    point_weights: Any | None,
) -> Any:
    """Solve the regularized linear system for the control grid at fixed UVs.

    Valid while rational weights are all 1 (true at fitting time): the surface is
    then exactly linear in the control points, ``S(u, v) = B(u, v) · P``. The
    Tikhonov term anchors to the current (seed) grid instead of zero so sparsely
    covered control points follow the smooth seed rather than collapsing to origin.
    """

    torch = require_torch()
    n_u = int(surface.control_grid.shape[0])
    n_v = int(surface.control_grid.shape[1])
    n = n_u * n_v
    dtype, device = surface.control_grid.dtype, surface.control_grid.device

    normal_matrix, normal_rhs, total_weight = _lsq_normal_system(
        points, uv, surface, smoothness_lambda, tikhonov_lambda, chunk_size, point_weights
    )
    scale = max(total_weight, 1e-8)
    penalty = _second_difference_penalty(n_u, n_v, dtype, device)
    seed = surface.control_grid.detach().reshape(n, 3)
    system = (
        normal_matrix / scale
        + float(smoothness_lambda) * penalty
        + float(tikhonov_lambda) * torch.eye(n, dtype=dtype, device=device)
    )
    rhs = normal_rhs / scale + float(tikhonov_lambda) * seed
    try:
        solution = torch.linalg.solve(system, rhs)
    except Exception:
        solution = torch.linalg.lstsq(system, rhs).solution
    return solution.reshape(n_u, n_v, 3)


def fit_torch_visible_surface_lsq(
    points: Any,
    resolution_u: int = 8,
    resolution_v: int = 4,
    degree_u: int = 2,
    degree_v: int = 2,
    smoothness_lambda: float = 1e-4,
    tikhonov_lambda: float = 1e-4,
    correction_rounds: int = 2,
    chunk_size: int = 4096,
    point_weights: Any | None = None,
    projection_iterations: int = 4,
    collect_diagnostics: bool = False,
    initial_uv: Any | None = None,
) -> tuple[TorchNURBSSurface, Any] | tuple[TorchNURBSSurface, Any, NURBSFitDiagnostics]:
    """Least-squares visible NURBS fit with foot-point parameter correction.

    Seeds control points with the IDW heuristic, then alternates a regularized
    linear solve for the control grid with foot-point UV reprojection (standard
    surface-fitting parameter correction). Returns the fitted surface and the
    final foot-point UVs of the input points on that surface.
    """

    torch = require_torch()
    points = torch.as_tensor(points, dtype=torch.float32, device=points.device if hasattr(points, "device") else None)
    if initial_uv is not None:
        initial_uv = torch.as_tensor(initial_uv, dtype=torch.float32, device=points.device)
    surface = fit_torch_visible_surface(
        points,
        resolution_u=resolution_u,
        resolution_v=resolution_v,
        chunk_size=chunk_size,
        degree_u=degree_u,
        degree_v=degree_v,
        initial_uv=initial_uv,
    )
    if initial_uv is None:
        initial_uv = pca_parameterize_points(points)
    diagnostics = NURBSFitDiagnostics(points.detach().clone(), None, initial_uv.detach().clone(), surface.control_grid.detach().clone(), [], surface.control_grid.detach().clone(), surface.weights.detach().clone()) if collect_diagnostics else None
    if int(points.shape[0]) <= 1:
        return (surface, initial_uv, diagnostics) if diagnostics is not None else (surface, initial_uv)
    if point_weights is not None:
        point_weights = torch.as_tensor(point_weights, dtype=points.dtype, device=points.device).reshape(-1)
        point_weights = torch.nan_to_num(point_weights, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        if not bool((point_weights > 0).any()):
            point_weights = None
    if diagnostics is not None and point_weights is not None:
        diagnostics.point_weights = point_weights.detach().clone()

    uv = initial_uv
    with torch.no_grad():
        for _ in range(max(1, int(correction_rounds))):
            surface.control_grid = _solve_control_grid_lsq(
                points, uv, surface, smoothness_lambda, tikhonov_lambda, chunk_size, point_weights
            )
            control_grid_after_lsq = surface.control_grid.detach().clone()
            uv = project_torch_points_to_nurbs(
                points,
                surface,
                iterations=int(projection_iterations),
                chunk_size=chunk_size,
            )
            if diagnostics is not None:
                diagnostics.rounds.append(NURBSFitRoundDiagnostics(control_grid_after_lsq, uv.detach().clone()))
    if diagnostics is not None:
        diagnostics.final_control_grid = surface.control_grid.detach().clone()
        diagnostics.final_weights = surface.weights.detach().clone()
        return surface, uv, diagnostics
    return surface, uv


def fit_coupled_wedge_ring_lsq(
    wedge_points: list[Any],
    wedge_initial_uv: list[Any],
    resolution_u: int = 8,
    resolution_v: int = 4,
    degree_u: int = 2,
    degree_v: int = 2,
    smoothness_lambda: float = 1e-4,
    tikhonov_lambda: float = 1e-4,
    correction_rounds: int = 2,
    chunk_size: int = 4096,
    projection_iterations: int = 4,
    collect_diagnostics: bool = False,
) -> list[tuple[TorchNURBSSurface, Any]] | list[tuple[TorchNURBSSurface, Any, NURBSFitDiagnostics]]:
    """Phase 5 Step 5-A (``docs/worklogs/55``): jointly fit ``len(wedge_points)``
    surfaces arranged in a cyclic ring -- matching the annulus O-grid's own
    convention that wedge ``k``'s ``u=1`` edge is the SAME physical boundary
    as wedge ``(k+1) % segments``'s ``u=0`` edge -- with each shared boundary
    column solved as ONE joint variable instead of two independently-fit
    columns later averaged.

    This is NOT the previously-rejected hard-C0 pattern (see
    ``build_annulus_chart``'s "NOT hard-enforced" docstring block): that
    pattern fit each wedge fully independently to its own local optimum, then
    overwrote the shared boundary columns post-hoc, so the interior was
    optimized against a boundary it never actually saw. Here the shared
    boundary control points are unknowns in ONE linear system from the first
    solve onward, so both wedges' interiors are fit consistently against the
    boundary they will actually share.

    Only the boundary COLUMNS (``u=0``/``u=1``, i.e. ``resolution_v`` control
    points each) are shared variables; each wedge's own interior columns stay
    private, independently-regularized unknowns (this function's own
    smoothness/Tikhonov terms never span across a seam) -- deliberately
    narrower coupling than a cross-seam smoothness stencil would give, since
    that (soft G1/tangent continuity across the seam) is explicitly scoped as
    a separate, only-if-needed Step 5-B, not bundled into this function.

    Each wedge is seeded independently exactly as ``fit_torch_visible_surface_lsq``
    seeds a single surface (IDW off its own ``wedge_initial_uv``); by
    construction (``build_annulus_chart``'s shared Coons boundary radii) two
    adjacent wedges' seeded boundary columns already agree before any solve,
    so accumulating both wedges' Tikhonov anchors into the one shared
    variable is consistent from round 0 onward, not just after the first
    joint solve.
    """

    torch = require_torch()
    segments = len(wedge_points)
    if segments < 2:
        raise ValueError("fit_coupled_wedge_ring_lsq needs at least 2 wedges to form a ring.")
    wedge_points = [
        torch.as_tensor(p, dtype=torch.float32, device=p.device if hasattr(p, "device") else None)
        for p in wedge_points
    ]
    wedge_initial_uv = [
        torch.as_tensor(uv, dtype=torch.float32, device=wedge_points[k].device)
        for k, uv in enumerate(wedge_initial_uv)
    ]

    surfaces = [
        fit_torch_visible_surface(
            wedge_points[k],
            resolution_u=resolution_u,
            resolution_v=resolution_v,
            chunk_size=chunk_size,
            degree_u=degree_u,
            degree_v=degree_v,
            initial_uv=wedge_initial_uv[k],
        )
        for k in range(segments)
    ]
    n_u = int(surfaces[0].control_grid.shape[0])
    n_v = int(surfaces[0].control_grid.shape[1])
    dtype, device = surfaces[0].control_grid.dtype, surfaces[0].control_grid.device
    n_interior_cols = max(n_u - 2, 0)
    total_boundary = segments * n_v
    total_interior = segments * n_interior_cols * n_v
    total = total_boundary + total_interior

    # local_to_global[k] maps wedge k's flattened (n_u*n_v,) local grid index
    # to its unique global variable index -- column 0 -> the boundary BEFORE
    # wedge k (shared with wedge k-1's column n_u-1), column n_u-1 -> the
    # boundary AFTER wedge k (shared with wedge k+1's column 0), interior
    # columns 1..n_u-2 -> wedge-private variables.
    local_to_global = []
    for k in range(segments):
        idx = torch.empty((n_u, n_v), dtype=torch.long, device=device)
        idx[0, :] = k * n_v + torch.arange(n_v, device=device)
        idx[n_u - 1, :] = ((k + 1) % segments) * n_v + torch.arange(n_v, device=device)
        for i in range(1, n_u - 1):
            idx[i, :] = total_boundary + (k * n_interior_cols + (i - 1)) * n_v + torch.arange(n_v, device=device)
        local_to_global.append(idx.reshape(-1))

    diagnostics_list: list[NURBSFitDiagnostics | None] = [
        NURBSFitDiagnostics(
            wedge_points[k].detach().clone(), None, wedge_initial_uv[k].detach().clone(),
            surfaces[k].control_grid.detach().clone(), [],
            surfaces[k].control_grid.detach().clone(), surfaces[k].weights.detach().clone(),
        )
        if collect_diagnostics else None
        for k in range(segments)
    ]

    uv = list(wedge_initial_uv)
    with torch.no_grad():
        for _ in range(max(1, int(correction_rounds))):
            global_matrix = torch.zeros((total, total), dtype=dtype, device=device)
            global_rhs = torch.zeros((total, 3), dtype=dtype, device=device)
            for k in range(segments):
                # Unlike the single-surface path, a too-sparse wedge is not
                # skipped entirely here: its interior variables must still
                # receive the smoothness/Tikhonov regularization below or the
                # global system becomes singular in that wedge's block.
                matrix, rhs, weight = _lsq_normal_system(
                    wedge_points[k], uv[k], surfaces[k], smoothness_lambda, tikhonov_lambda, chunk_size, None
                )
                scale = max(weight, 1e-8)
                penalty = _second_difference_penalty(n_u, n_v, dtype, device)
                seed = surfaces[k].control_grid.detach().reshape(n_u * n_v, 3)
                local_system = (
                    matrix / scale
                    + float(smoothness_lambda) * penalty
                    + float(tikhonov_lambda) * torch.eye(n_u * n_v, dtype=dtype, device=device)
                )
                local_rhs = rhs / scale + float(tikhonov_lambda) * seed
                m = local_to_global[k]
                flat_index = (m.unsqueeze(1) * total + m.unsqueeze(0)).reshape(-1)
                global_matrix.view(-1).index_add_(0, flat_index, local_system.reshape(-1))
                global_rhs.index_add_(0, m, local_rhs)
            try:
                solution = torch.linalg.solve(global_matrix, global_rhs)
            except Exception:
                solution = torch.linalg.lstsq(global_matrix, global_rhs).solution

            for k in range(segments):
                surfaces[k].control_grid = solution[local_to_global[k]].reshape(n_u, n_v, 3)
                control_grid_after_lsq = surfaces[k].control_grid.detach().clone()
                uv[k] = project_torch_points_to_nurbs(
                    wedge_points[k], surfaces[k], iterations=int(projection_iterations), chunk_size=chunk_size
                )
                if diagnostics_list[k] is not None:
                    diagnostics_list[k].rounds.append(
                        NURBSFitRoundDiagnostics(control_grid_after_lsq, uv[k].detach().clone())
                    )

    results = []
    for k in range(segments):
        if diagnostics_list[k] is not None:
            diagnostics_list[k].final_control_grid = surfaces[k].control_grid.detach().clone()
            diagnostics_list[k].final_weights = surfaces[k].weights.detach().clone()
            results.append((surfaces[k], uv[k], diagnostics_list[k]))
        else:
            results.append((surfaces[k], uv[k]))
    return results


def project_torch_points_to_nurbs(
    points: Any,
    surface: TorchNURBSSurface,
    grid_u: int = 0,
    grid_v: int = 0,
    iterations: int = 4,
    chunk_size: int = 65536,
) -> Any:
    """Foot-point projection: closest ``(u, v)`` on the surface per query point.

    Initializes each point from the nearest sample of a dense UV evaluation grid,
    then runs damped Gauss-Newton on the point-to-surface residual using analytic
    surface derivatives. The refined UV is only kept when it does not increase the
    residual, so the result is never worse than the grid initialization. Runs
    detached: UV bindings are data, not part of the autodiff graph.
    """

    torch = require_torch()
    with torch.no_grad():
        points = torch.as_tensor(
            points, dtype=surface.control_grid.dtype, device=surface.control_grid.device
        )
        if points.numel() == 0:
            return torch.zeros((0, 2), dtype=surface.control_grid.dtype, device=points.device)

        n_u = int(surface.control_grid.shape[0])
        n_v = int(surface.control_grid.shape[1])
        samples_u = int(grid_u) if int(grid_u) > 1 else min(max(2 * n_u, 8), 64)
        samples_v = int(grid_v) if int(grid_v) > 1 else min(max(2 * n_v, 8), 64)
        lin_u = torch.linspace(0.0, 1.0, samples_u, dtype=points.dtype, device=points.device)
        lin_v = torch.linspace(0.0, 1.0, samples_v, dtype=points.dtype, device=points.device)
        grid_uu, grid_vv = torch.meshgrid(lin_u, lin_v, indexing="ij")
        grid_uv = torch.stack([grid_uu.reshape(-1), grid_vv.reshape(-1)], dim=-1)
        grid_points = surface.evaluate(grid_uv)

        chunk_size = max(1, int(chunk_size))
        iterations = max(0, int(iterations))
        results = []
        for chunk in torch.split(points, chunk_size, dim=0):
            nearest = torch.cdist(chunk, grid_points).argmin(dim=1)
            uv = grid_uv[nearest].clone()
            best_uv = uv.clone()
            best_dist = (surface.evaluate(uv) - chunk).norm(dim=1)
            for _ in range(iterations):
                point, deriv_u, deriv_v = surface.evaluate_with_derivatives(uv)
                residual = point - chunk
                jacobian = torch.stack([deriv_u, deriv_v], dim=-1)
                jtj = jacobian.transpose(1, 2) @ jacobian
                damping = 1e-6 * jtj.diagonal(dim1=1, dim2=2).mean(dim=1).clamp_min(1e-12)
                jtj = jtj + damping[:, None, None] * torch.eye(
                    2, dtype=jtj.dtype, device=jtj.device
                )
                jtr = (jacobian.transpose(1, 2) @ residual[..., None]).squeeze(-1)
                step = torch.linalg.solve(jtj, -jtr)
                # One grid cell per step keeps far-off linearizations from jumping charts.
                step = step.clamp(min=-0.25, max=0.25)
                uv = torch.clamp(uv + step, 0.0, 1.0)
                dist = (surface.evaluate(uv) - chunk).norm(dim=1)
                improved = dist < best_dist
                best_uv[improved] = uv[improved]
                best_dist = torch.where(improved, dist, best_dist)
            results.append(best_uv)
        return torch.cat(results, dim=0)


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
