"""Deterministic Performance Track batching for independent visible charts.

This module is deliberately separate from :mod:`torch_nurbs`: the established
single-chart functions remain the immutable scientific/reference path.  The
functions here preserve each chart's own control grid, UV sequence, solve
count, projection count, and point ordering while submitting equal-topology
charts as fixed, recorded batches.

No runtime OOM splitting, backend fallback, CUDA graphs, custom kernels, or
multi-stream execution is performed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Sequence

from osn_gs.surface.torch_nurbs import (
    TorchNURBSSurface,
    _identity_matrix,
    _lsq_normal_system,
    _regular_uv_grid,
    _solve_control_grid_lsq,
    fit_torch_visible_surface,
    project_torch_points_to_nurbs,
)
from osn_gs.utils.torch_ops import require_torch


@dataclass(frozen=True)
class DeterministicChartBatchConfig:
    """Recorded, data-independent limits used to derive an official plan."""

    bucket_upper_bounds: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048, 4096)
    max_batch_charts: int = 256
    max_padded_points: int = 65536
    oversize_execution: str = "serial-reference"

    def __post_init__(self) -> None:
        bounds = tuple(int(value) for value in self.bucket_upper_bounds)
        if not bounds or bounds != tuple(sorted(set(bounds))) or bounds[0] < 1:
            raise ValueError("bucket_upper_bounds must be strictly increasing positive integers")
        if int(self.max_batch_charts) < 1 or int(self.max_padded_points) < bounds[0]:
            raise ValueError("batch limits must admit at least one smallest-bucket chart")
        if self.oversize_execution != "serial-reference":
            raise ValueError("Only the immutable serial-reference oversize policy is approved")


@dataclass(frozen=True)
class PlannedChartBatch:
    batch_id: int
    bucket_upper: int | None
    chart_indices: tuple[int, ...]
    chart_lengths: tuple[int, ...]
    padded_points: int
    execution: str


@dataclass(frozen=True)
class DeterministicChartBatchPlan:
    config: DeterministicChartBatchConfig
    chart_count: int
    point_count: int
    batches: tuple[PlannedChartBatch, ...]
    pathological_chart_indices: tuple[int, ...]
    forced_reference_chart_indices: tuple[int, ...]
    digest_sha256: str

    def to_record(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "chart_count": self.chart_count,
            "point_count": self.point_count,
            "batch_count": len(self.batches),
            "pathological_chart_indices": list(self.pathological_chart_indices),
            "forced_reference_chart_indices": list(self.forced_reference_chart_indices),
            "digest_sha256": self.digest_sha256,
            "batches": [asdict(batch) for batch in self.batches],
            "padding": chart_plan_padding_summary(self),
        }


def _pathological_chart_indices(lengths: Sequence[int], bounds: Sequence[int]) -> tuple[int, ...]:
    """Deterministically retain minimum, boundaries, and long-tail charts."""

    if not lengths:
        return ()
    selected = {0, len(lengths) - 1}
    selected.add(min(range(len(lengths)), key=lambda index: (int(lengths[index]), index)))
    selected.update(
        sorted(range(len(lengths)), key=lambda index: (-int(lengths[index]), index))[:10]
    )
    targets = set(int(bound) for bound in bounds)
    targets.update(int(bound) + 1 for bound in bounds)
    for target in targets:
        exact = [index for index, length in enumerate(lengths) if int(length) == target]
        if exact:
            selected.add(exact[0])
        else:
            selected.add(min(range(len(lengths)), key=lambda index: (abs(int(lengths[index]) - target), index)))
    return tuple(sorted(selected))


def plan_chart_batches(
    chart_lengths: Sequence[int],
    config: DeterministicChartBatchConfig = DeterministicChartBatchConfig(),
    forced_reference_chart_indices: Sequence[int] = (),
) -> DeterministicChartBatchPlan:
    """Create a fixed plan solely from ordered chart lengths and explicit limits."""

    lengths = tuple(int(length) for length in chart_lengths)
    if any(length < 1 for length in lengths):
        raise ValueError("Every chart length must be positive")
    forced = frozenset(int(index) for index in forced_reference_chart_indices)
    if any(index < 0 or index >= len(lengths) for index in forced):
        raise ValueError("forced reference chart index is outside the corpus")
    grouped: dict[int | None, list[int]] = {bound: [] for bound in config.bucket_upper_bounds}
    grouped[None] = []
    for index, length in enumerate(lengths):
        upper = None if index in forced else next(
            (bound for bound in config.bucket_upper_bounds if length <= bound), None
        )
        grouped[upper].append(index)

    pending: list[tuple[int | None, tuple[int, ...], str]] = []
    for upper in config.bucket_upper_bounds:
        indices = grouped[upper]
        charts_per_batch = min(
            int(config.max_batch_charts), max(1, int(config.max_padded_points) // int(upper))
        )
        for start in range(0, len(indices), charts_per_batch):
            pending.append((upper, tuple(indices[start : start + charts_per_batch]), "deterministic-batched"))
    for index in grouped[None]:
        pending.append((None, (index,), config.oversize_execution))

    batches = tuple(
        PlannedChartBatch(
            batch_id=batch_id,
            bucket_upper=upper,
            chart_indices=indices,
            chart_lengths=tuple(lengths[index] for index in indices),
            padded_points=(int(upper) * len(indices) if upper is not None else lengths[indices[0]]),
            execution=execution,
        )
        for batch_id, (upper, indices, execution) in enumerate(pending)
    )
    payload = {
        "config": asdict(config),
        "chart_lengths": lengths,
        "forced_reference_chart_indices": tuple(sorted(forced)),
        "batches": [asdict(batch) for batch in batches],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return DeterministicChartBatchPlan(
        config=config,
        chart_count=len(lengths),
        point_count=sum(lengths),
        batches=batches,
        pathological_chart_indices=tuple(sorted(set(
            _pathological_chart_indices(lengths, config.bucket_upper_bounds)
        ) | forced)),
        forced_reference_chart_indices=tuple(sorted(forced)),
        digest_sha256=digest,
    )


def chart_plan_padding_summary(plan: DeterministicChartBatchPlan) -> dict[str, Any]:
    batched = [batch for batch in plan.batches if batch.execution == "deterministic-batched"]
    actual = sum(sum(batch.chart_lengths) for batch in batched)
    padded = sum(batch.padded_points for batch in batched)
    return {
        "batched_chart_count": sum(len(batch.chart_indices) for batch in batched),
        "oversize_serial_chart_count": sum(
            len(batch.chart_indices) for batch in plan.batches if batch.execution == "serial-reference"
        ),
        "actual_points": actual,
        "padded_points": padded,
        "wasted_points": padded - actual,
        "waste_fraction": ((padded - actual) / padded) if padded else 0.0,
    }


@dataclass(frozen=True)
class ChartBatchEligibility:
    chart_index: int
    eligible: bool
    reasons: tuple[str, ...]
    uv_rank_ratio: float
    coordinate_offset_ratio: float
    duplicate_uv_count: int


def classify_chart_batch_eligibility(chart_index: int, points: Any, uv: Any) -> ChartBatchEligibility:
    """Deterministic preflight; rejected charts are recorded before execution."""

    torch = require_torch()
    points = torch.as_tensor(points)
    uv = torch.as_tensor(uv, dtype=points.dtype, device=points.device)
    reasons: list[str] = []
    if str(points.device.type) != "cuda":
        reasons.append("non-cuda-device")
    finite = bool(torch.isfinite(points).all() and torch.isfinite(uv).all())
    if not finite:
        reasons.append("non-finite-input")
    centered_uv = uv - uv.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered_uv.T @ centered_uv / max(int(uv.shape[0]), 1))
    uv_rank_ratio = float(singular[-1] / singular[0].clamp_min(1e-30))
    if uv_rank_ratio < 1e-6:
        reasons.append("near-rank-one-uv")
    duplicate_count = int(uv.shape[0]) - int(torch.unique(uv, dim=0).shape[0])
    if duplicate_count:
        reasons.append("duplicate-camera-uv")
    extent = (points.amax(dim=0) - points.amin(dim=0)).norm().clamp_min(1e-12)
    offset_ratio = float(points.mean(dim=0).norm() / extent)
    if offset_ratio > 100.0:
        reasons.append("large-coordinate-offset")
    return ChartBatchEligibility(
        chart_index=int(chart_index), eligible=not reasons, reasons=tuple(reasons),
        uv_rank_ratio=uv_rank_ratio, coordinate_offset_ratio=offset_ratio,
        duplicate_uv_count=duplicate_count,
    )


def plan_chart_corpus(
    chart_points: Sequence[Any],
    chart_uv: Sequence[Any],
    config: DeterministicChartBatchConfig = DeterministicChartBatchConfig(),
) -> tuple[DeterministicChartBatchPlan, tuple[ChartBatchEligibility, ...]]:
    if len(chart_points) != len(chart_uv):
        raise ValueError("chart point and UV corpora must align")
    eligibility = tuple(
        classify_chart_batch_eligibility(index, points, uv)
        for index, (points, uv) in enumerate(zip(chart_points, chart_uv))
    )
    forced = [record.chart_index for record in eligibility if not record.eligible]
    plan = plan_chart_batches(
        [int(points.shape[0]) for points in chart_points], config, forced
    )
    return plan, eligibility


@dataclass
class BatchedChartFitResult:
    chart_index: int
    surface_a: TorchNURBSSurface
    surface_b: TorchNURBSSurface
    uv_footpoint: Any
    uv_geo_b: Any
    fitted_a_at_footpoint: Any
    normals_a: Any
    residual_g_a: Any
    residual_g_b: Any
    residual_c_a: Any
    residual_c_b: Any
    solve_fallbacks_a: int = 0
    solve_fallbacks_b: int = 0


def _template_surface(batch_control: Any, degree_u: int, degree_v: int) -> TorchNURBSSurface:
    torch = require_torch()
    n_u, n_v = int(batch_control.shape[1]), int(batch_control.shape[2])
    return TorchNURBSSurface(
        control_grid=torch.zeros((n_u, n_v, 3), dtype=batch_control.dtype, device=batch_control.device),
        weights=torch.ones((n_u, n_v), dtype=batch_control.dtype, device=batch_control.device),
        degree_u=degree_u,
        degree_v=degree_v,
    )


def _basis_rows(template: TorchNURBSSurface, uv: Any, derivatives: bool) -> tuple[Any, Any | None, Any | None]:
    batch, count = int(uv.shape[0]), int(uv.shape[1])
    flat = uv.reshape(-1, 2)
    if derivatives:
        basis_u, basis_v, deriv_u, deriv_v = template._basis_tables(flat)
    else:
        basis_u, basis_v = template._basis_values(flat)
        deriv_u = deriv_v = None
    n_u, n_v = int(template.control_grid.shape[0]), int(template.control_grid.shape[1])

    def rows(left: Any, right: Any) -> Any:
        return (left[:, :, None] * right[:, None, :]).reshape(batch, count, n_u * n_v)

    value_rows = rows(basis_u, basis_v)
    du_rows = rows(deriv_u, basis_v) if deriv_u is not None else None
    dv_rows = rows(basis_u, deriv_v) if deriv_v is not None else None
    return value_rows, du_rows, dv_rows


def _evaluate_batch(control: Any, uv: Any, derivatives: bool = False) -> tuple[Any, Any | None, Any | None]:
    torch = require_torch()
    template = _template_surface(control, degree_u=2, degree_v=2)
    rows, rows_u, rows_v = _basis_rows(template, uv, derivatives)
    flat_control = control.reshape(control.shape[0], -1, 3)
    numerator = torch.bmm(rows, flat_control)
    denominator = rows.sum(dim=-1).clamp_min(1e-8)
    point = numerator / denominator[..., None]
    if not derivatives:
        return point, None, None

    def partial(partial_rows: Any | None) -> Any:
        if partial_rows is None:
            return torch.zeros_like(point)
        numerator_d = torch.bmm(partial_rows, flat_control)
        denominator_d = partial_rows.sum(dim=-1)
        return (numerator_d - denominator_d[..., None] * point) / denominator[..., None]

    return point, partial(rows_u), partial(rows_v)


def _pad_chart_tensors(points: Sequence[Any], uv: Sequence[Any], padded_count: int) -> tuple[Any, Any, Any, Any]:
    torch = require_torch()
    batch = len(points)
    dtype, device = points[0].dtype, points[0].device
    padded_points = torch.zeros((batch, padded_count, 3), dtype=dtype, device=device)
    padded_uv = torch.zeros((batch, padded_count, 2), dtype=dtype, device=device)
    mask = torch.zeros((batch, padded_count), dtype=torch.bool, device=device)
    lengths = torch.tensor([int(value.shape[0]) for value in points], dtype=torch.int64, device=device)
    for row, (chart_points, chart_uv) in enumerate(zip(points, uv)):
        count = int(chart_points.shape[0])
        padded_points[row, :count] = chart_points
        padded_uv[row, :count] = chart_uv
        mask[row, :count] = True
    return padded_points, padded_uv, mask, lengths


def _project_batch(points: Any, mask: Any, control: Any, iterations: int) -> Any:
    torch = require_torch()
    batch, padded = int(points.shape[0]), int(points.shape[1])
    n_u, n_v = int(control.shape[1]), int(control.shape[2])
    samples_u, samples_v = min(max(2 * n_u, 8), 64), min(max(2 * n_v, 8), 64)
    grid_uv = _regular_uv_grid(samples_u, samples_v, points.dtype, points.device)
    # Foot-point basins are part of the scientific semantics.  A batched
    # cdist/grid evaluation can perturb nearest-grid ties enough to enter a
    # different Gauss-Newton basin, so initialization stays bit-for-bit on
    # the immutable per-chart evaluator.  Only the terminal GN updates below
    # are submitted as one batch.
    nearest = torch.zeros((batch, padded), dtype=torch.int64, device=points.device)
    for row in range(batch):
        count = int(mask[row].sum())
        surface = TorchNURBSSurface(
            control_grid=control[row],
            weights=torch.ones((n_u, n_v), dtype=points.dtype, device=points.device),
            degree_u=2, degree_v=2,
        )
        grid_points = surface.evaluate(grid_uv)
        nearest[row, :count] = torch.cdist(
            points[row, :count], grid_points
        ).argmin(dim=1)
    uv = grid_uv[nearest].clone()
    best_uv = uv.clone()
    iterations = max(0, int(iterations))
    if iterations == 0:
        point, _, _ = _evaluate_batch(control, uv)
        best_dist = (point - points).norm(dim=2)
    else:
        point, deriv_u, deriv_v = _evaluate_batch(control, uv, derivatives=True)
        best_dist = (point - points).norm(dim=2)
    for iteration in range(iterations):
        residual = point - points
        jacobian = torch.stack((deriv_u, deriv_v), dim=-1)
        jtj = jacobian.transpose(-1, -2) @ jacobian
        damping = 1e-6 * jtj.diagonal(dim1=-2, dim2=-1).mean(dim=-1).clamp_min(1e-12)
        jtj = jtj + damping[..., None, None] * _identity_matrix(2, points.dtype, points.device)
        jtr = (jacobian.transpose(-1, -2) @ residual[..., None]).squeeze(-1)
        step = torch.linalg.solve(jtj, -jtr).clamp(min=-0.25, max=0.25)
        uv = torch.clamp(uv + step, 0.0, 1.0)
        if iteration + 1 < iterations:
            point, deriv_u, deriv_v = _evaluate_batch(control, uv, derivatives=True)
            dist = (point - points).norm(dim=2)
        else:
            point, _, _ = _evaluate_batch(control, uv)
            dist = (point - points).norm(dim=2)
        improved = (dist < best_dist) & mask
        best_uv[improved] = uv[improved]
        best_dist = torch.where(improved, dist, best_dist)
    return best_uv


def fit_serial_chart_reference(
    chart_index: int,
    points: Any,
    camera_uv: Any,
    *,
    correction_rounds: int = 2,
    projection_iterations: int = 3,
) -> BatchedChartFitResult:
    """Call the immutable WL119 mathematics for one chart without batching."""

    torch = require_torch()
    with torch.no_grad():
        surface_a = fit_torch_visible_surface(
            points, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2,
            initial_uv=camera_uv,
        )
        uv_a = camera_uv
        for _ in range(max(1, int(correction_rounds))):
            surface_a.control_grid = _solve_control_grid_lsq(
                points, uv_a, surface_a, 1e-4, 1e-4, 4096, None
            )
            uv_a = project_torch_points_to_nurbs(
                points, surface_a, iterations=projection_iterations, chunk_size=4096
            )
        fitted_a, normals_a = surface_a.evaluate_with_normals(uv_a)

        surface_b = fit_torch_visible_surface(
            points, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2,
            initial_uv=camera_uv,
        )
        fixed = _lsq_normal_system(points, camera_uv, surface_b, 1e-4, 1e-4, 4096, None)
        for _ in range(max(1, int(correction_rounds))):
            surface_b.control_grid = _solve_control_grid_lsq(
                points, camera_uv, surface_b, 1e-4, 1e-4, 4096, None,
                preassembled_normal_system=fixed,
            )
        uv_b = project_torch_points_to_nurbs(
            points, surface_b, iterations=projection_iterations, chunk_size=4096
        )
        residual_g_a = (surface_a.evaluate(uv_a) - points).norm(dim=-1)
        residual_g_b = (surface_b.evaluate(uv_b) - points).norm(dim=-1)
        residual_c_a = (surface_a.evaluate(camera_uv) - points).norm(dim=-1)
        residual_c_b = (surface_b.evaluate(camera_uv) - points).norm(dim=-1)
    return BatchedChartFitResult(
        chart_index=int(chart_index), surface_a=surface_a, surface_b=surface_b,
        uv_footpoint=uv_a, uv_geo_b=uv_b,
        fitted_a_at_footpoint=fitted_a, normals_a=normals_a,
        residual_g_a=residual_g_a, residual_g_b=residual_g_b,
        residual_c_a=residual_c_a, residual_c_b=residual_c_b,
    )


def fit_batched_charts(
    chart_indices: Sequence[int],
    chart_points: Sequence[Any],
    chart_camera_uv: Sequence[Any],
    *,
    padded_count: int,
    resolution_u: int = 8,
    resolution_v: int = 4,
    degree_u: int = 2,
    degree_v: int = 2,
    smoothness_lambda: float = 1e-4,
    tikhonov_lambda: float = 1e-4,
    correction_rounds: int = 2,
    projection_iterations: int = 3,
) -> list[BatchedChartFitResult]:
    """Fit WL119 charts with only dependency-terminal work batched.

    Seeds, every LSQ solve, and ARM A's first (solve-feeding) projection use
    the immutable reference functions in original chart order.  The final ARM
    A projection and ARM B's evaluation-only projection cannot affect another
    solve, so those two independent terminal stages are submitted together.
    """

    torch = require_torch()
    if not chart_points or len(chart_points) != len(chart_camera_uv):
        raise ValueError("A non-empty, aligned chart batch is required")
    if (resolution_u, resolution_v, degree_u, degree_v) != (8, 4, 2, 2):
        raise ValueError("Performance Track batching is approved only for frozen WL119 8x4 degree-2 charts")
    if min(int(points.shape[0]) for points in chart_points) < 16:
        raise ValueError("The frozen WL119 minimum chart population is required")
    if max(int(points.shape[0]) for points in chart_points) > int(padded_count):
        raise ValueError("padded_count does not cover this recorded batch")

    points, camera_uv, mask, lengths = _pad_chart_tensors(
        chart_points, chart_camera_uv, int(padded_count)
    )
    control_a_rows = []
    control_b_rows = []
    with torch.no_grad():
        for chart_point, chart_uv in zip(chart_points, chart_camera_uv):
            surface_a = fit_torch_visible_surface(
                chart_point, resolution_u=resolution_u, resolution_v=resolution_v,
                degree_u=degree_u, degree_v=degree_v, initial_uv=chart_uv,
            )
            uv_for_solve = chart_uv
            for round_index in range(max(1, int(correction_rounds))):
                surface_a.control_grid = _solve_control_grid_lsq(
                    chart_point, uv_for_solve, surface_a,
                    smoothness_lambda, tikhonov_lambda, 4096, None,
                )
                # The final projection is terminal and is executed below in
                # the deterministic batch.  Earlier projections remain exact
                # because they feed the next chart-local solve.
                if round_index + 1 < max(1, int(correction_rounds)):
                    uv_for_solve = project_torch_points_to_nurbs(
                        chart_point, surface_a,
                        iterations=projection_iterations, chunk_size=4096,
                    )
            control_a_rows.append(surface_a.control_grid)

            surface_b = fit_torch_visible_surface(
                chart_point, resolution_u=resolution_u, resolution_v=resolution_v,
                degree_u=degree_u, degree_v=degree_v, initial_uv=chart_uv,
            )
            fixed_system = _lsq_normal_system(
                chart_point, chart_uv, surface_b,
                smoothness_lambda, tikhonov_lambda, 4096, None,
            )
            for _ in range(max(1, int(correction_rounds))):
                surface_b.control_grid = _solve_control_grid_lsq(
                    chart_point, chart_uv, surface_b,
                    smoothness_lambda, tikhonov_lambda, 4096, None,
                    preassembled_normal_system=fixed_system,
                )
            control_b_rows.append(surface_b.control_grid)

        control_a = torch.stack(control_a_rows)
        control_b = torch.stack(control_b_rows)
        doubled_points = torch.cat((points, points), dim=0)
        doubled_mask = torch.cat((mask, mask), dim=0)
        doubled_control = torch.cat((control_a, control_b), dim=0)
        doubled_uv = _project_batch(
            doubled_points, doubled_mask, doubled_control, projection_iterations
        )
        batch_size = len(chart_points)
        uv_footpoint = doubled_uv[:batch_size]
        uv_geo_b = doubled_uv[batch_size:]

        fitted_a, deriv_u, deriv_v = _evaluate_batch(
            control_a, uv_footpoint, derivatives=True
        )
        normals_a = torch.nn.functional.normalize(
            torch.cross(deriv_u, deriv_v, dim=-1), dim=-1, eps=1e-12
        )
        residual_g_a = (fitted_a - points).norm(dim=2)
        fitted_b_geo, _, _ = _evaluate_batch(control_b, uv_geo_b)
        residual_g_b = (fitted_b_geo - points).norm(dim=2)
        fitted_a_camera, _, _ = _evaluate_batch(control_a, camera_uv)
        fitted_b_camera, _, _ = _evaluate_batch(control_b, camera_uv)
        residual_c_a = (fitted_a_camera - points).norm(dim=2)
        residual_c_b = (fitted_b_camera - points).norm(dim=2)

    results: list[BatchedChartFitResult] = []
    for row, chart_index in enumerate(chart_indices):
        count = int(lengths[row])
        surface_a = TorchNURBSSurface(
            control_grid=control_a[row],
            weights=torch.ones((resolution_u, resolution_v), dtype=points.dtype, device=points.device),
            degree_u=degree_u, degree_v=degree_v,
        )
        surface_b = TorchNURBSSurface(
            control_grid=control_b[row],
            weights=torch.ones((resolution_u, resolution_v), dtype=points.dtype, device=points.device),
            degree_u=degree_u, degree_v=degree_v,
        )
        results.append(BatchedChartFitResult(
            chart_index=int(chart_index), surface_a=surface_a, surface_b=surface_b,
            uv_footpoint=uv_footpoint[row, :count], uv_geo_b=uv_geo_b[row, :count],
            fitted_a_at_footpoint=fitted_a[row, :count], normals_a=normals_a[row, :count],
            residual_g_a=residual_g_a[row, :count], residual_g_b=residual_g_b[row, :count],
            residual_c_a=residual_c_a[row, :count], residual_c_b=residual_c_b[row, :count],
            solve_fallbacks_a=0, solve_fallbacks_b=0,
        ))
    return results


def execute_chart_plan(
    plan: DeterministicChartBatchPlan,
    chart_points: Sequence[Any],
    chart_uv: Sequence[Any],
) -> list[BatchedChartFitResult]:
    """Execute exactly the recorded plan; no splitting or fallback is allowed."""

    if len(chart_points) != plan.chart_count or len(chart_uv) != plan.chart_count:
        raise ValueError("The corpus does not match the recorded chart plan")
    by_index: list[BatchedChartFitResult | None] = [None] * plan.chart_count
    for batch in plan.batches:
        if batch.execution == "deterministic-batched":
            results = fit_batched_charts(
                batch.chart_indices,
                [chart_points[index] for index in batch.chart_indices],
                [chart_uv[index] for index in batch.chart_indices],
                padded_count=int(batch.bucket_upper),
            )
        elif batch.execution == "serial-reference":
            results = [
                fit_serial_chart_reference(index, chart_points[index], chart_uv[index])
                for index in batch.chart_indices
            ]
        else:
            raise ValueError(f"Unapproved chart execution mode: {batch.execution}")
        for result in results:
            if by_index[result.chart_index] is not None:
                raise RuntimeError("A recorded plan assigned one chart more than once")
            by_index[result.chart_index] = result
    if any(result is None for result in by_index):
        raise RuntimeError("A recorded plan omitted one or more charts")
    return [result for result in by_index if result is not None]


def batched_result_scalar_metrics(result: BatchedChartFitResult) -> dict[str, float]:
    """The exact WL119 chart-level scalar semantics, retained for comparison."""

    torch = require_torch()
    values = torch.stack([
        result.residual_g_a.median(), result.residual_g_a.quantile(0.95), result.residual_g_a.max(),
        result.residual_g_b.median(), result.residual_g_b.quantile(0.95), result.residual_g_b.max(),
        result.residual_c_a.median(), result.residual_c_a.quantile(0.95), result.residual_c_a.max(),
        result.residual_c_b.median(), result.residual_c_b.quantile(0.95), result.residual_c_b.max(),
        (result.surface_a.control_grid - result.surface_b.control_grid).norm(dim=-1).mean(),
        result.surface_a.smoothness(), result.surface_b.smoothness(),
    ]).detach().cpu().tolist()
    names = (
        "residual_g_arm_a_median", "residual_g_arm_a_p95", "residual_g_arm_a_max",
        "residual_g_arm_b_median", "residual_g_arm_b_p95", "residual_g_arm_b_max",
        "residual_c_arm_a_median", "residual_c_arm_a_p95", "residual_c_arm_a_max",
        "residual_c_arm_b_median", "residual_c_arm_b_p95", "residual_c_arm_b_max",
        "control_grid_diff_mean", "smoothness_arm_a", "smoothness_arm_b",
    )
    return {name: float(value) for name, value in zip(names, values)}
