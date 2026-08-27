import math

import pytest
import torch

from osn_gs.surface.torch_nurbs import (
    _lsq_normal_system,
    _solve_control_grid_lsq,
    fit_torch_visible_surface,
    fit_torch_visible_surface_lsq,
    project_torch_points_to_nurbs,
)
from osn_gs.surface.torch_nurbs_performance_batch import (
    DeterministicChartBatchConfig,
    batched_result_scalar_metrics,
    execute_chart_plan,
    fit_batched_charts,
    plan_chart_batches,
    plan_chart_corpus,
)


def _chart(count: int, seed: int, device: str = "cpu", pathology: str = "curved"):
    generator = torch.Generator(device=device).manual_seed(seed)
    uv = torch.rand((count, 2), generator=generator, device=device)
    if pathology == "near-line":
        uv[:, 1] = 0.5 + 1e-4 * (uv[:, 1] - 0.5)
    if pathology == "duplicate-tail":
        uv[-4:] = uv[0]
    points = torch.stack([
        uv[:, 0], uv[:, 1],
        0.08 * torch.sin(3.0 * uv[:, 0]) * torch.cos(2.0 * uv[:, 1]),
    ], dim=1)
    if pathology == "large-offset":
        points = points + torch.tensor([1000.0, -700.0, 300.0], device=device)
    return points.to(torch.float32), uv.to(torch.float32)


def _serial_result(points, uv):
    surface_a, uv_a = fit_torch_visible_surface_lsq(
        points, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2,
        initial_uv=uv, correction_rounds=2, projection_iterations=3,
    )
    surface_b = fit_torch_visible_surface(
        points, resolution_u=8, resolution_v=4, degree_u=2, degree_v=2,
        initial_uv=uv,
    )
    normal = _lsq_normal_system(points, uv, surface_b, 1e-4, 1e-4, 4096, None)
    for _ in range(2):
        surface_b.control_grid = _solve_control_grid_lsq(
            points, uv, surface_b, 1e-4, 1e-4, 4096, None,
            preassembled_normal_system=normal,
        )
    fitted_a, normals_a = surface_a.evaluate_with_normals(uv_a)
    residual_g_a = (fitted_a - points).norm(dim=-1)
    uv_b = project_torch_points_to_nurbs(points, surface_b, iterations=3, chunk_size=4096)
    residual_g_b = (surface_b.evaluate(uv_b) - points).norm(dim=-1)
    residual_c_a = (surface_a.evaluate(uv) - points).norm(dim=-1)
    residual_c_b = (surface_b.evaluate(uv) - points).norm(dim=-1)
    return surface_a, surface_b, uv_a, uv_b, fitted_a, normals_a, residual_g_a, residual_g_b, residual_c_a, residual_c_b


def test_batch_plan_is_deterministic_recorded_and_never_runtime_splits():
    lengths = [32, 64, 65, 127, 128, 129, 4096, 4097, 217000]
    config = DeterministicChartBatchConfig(max_batch_charts=3, max_padded_points=8192)
    first = plan_chart_batches(lengths, config)
    second = plan_chart_batches(lengths, config)
    assert first == second
    assert first.digest_sha256 == second.digest_sha256
    assigned = [index for batch in first.batches for index in batch.chart_indices]
    assert sorted(assigned) == list(range(len(lengths)))
    assert len(assigned) == len(set(assigned))
    oversize = [batch for batch in first.batches if batch.execution == "serial-reference"]
    assert [batch.chart_indices for batch in oversize] == [(7,), (8,)]
    assert set((0, 6, 7, 8)).issubset(first.pathological_chart_indices)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_batched_chart_math_matches_immutable_serial_reference(device):
    specifications = [
        (32, 11, "curved"),
        (63, 12, "near-line"),
        (64, 13, "duplicate-tail"),
        (57, 14, "large-offset"),
    ]
    inputs = [_chart(count, seed, device, pathology) for count, seed, pathology in specifications]
    points = [item[0] for item in inputs]
    uv = [item[1] for item in inputs]
    plan, eligibility = plan_chart_corpus(
        points, uv, DeterministicChartBatchConfig(max_batch_charts=8, max_padded_points=512)
    )
    expected_eligibility = [True, False, False, False] if device == "cuda" else [False] * 4
    expected_forced = (1, 2, 3) if device == "cuda" else (0, 1, 2, 3)
    assert [record.eligible for record in eligibility] == expected_eligibility
    assert plan.forced_reference_chart_indices == expected_forced
    candidate = execute_chart_plan(plan, points, uv)

    for index, (chart_points, chart_uv) in enumerate(inputs):
        reference = _serial_result(chart_points, chart_uv)
        actual = candidate[index]
        pairs = (
            (actual.surface_a.control_grid, reference[0].control_grid),
            (actual.surface_b.control_grid, reference[1].control_grid),
            (actual.uv_footpoint, reference[2]),
            (actual.uv_geo_b, reference[3]),
            (actual.fitted_a_at_footpoint, reference[4]),
            (actual.normals_a, reference[5]),
            (actual.residual_g_a, reference[6]),
            (actual.residual_g_b, reference[7]),
            (actual.residual_c_a, reference[8]),
            (actual.residual_c_b, reference[9]),
        )
        for observed, expected in pairs:
            assert torch.equal(torch.isnan(observed), torch.isnan(expected))
            assert torch.equal(torch.isinf(observed), torch.isinf(expected))
            torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-5)
        assert torch.isfinite(actual.surface_a.control_grid).all()
        assert torch.isfinite(actual.surface_b.control_grid).all()


def test_scalar_metric_names_and_semantics_match_reference():
    points, uv = _chart(48, 21)
    result = fit_batched_charts([7], [points], [uv], padded_count=64)[0]
    metrics = batched_result_scalar_metrics(result)
    assert len(metrics) == 15
    assert metrics["residual_g_arm_a_median"] == float(result.residual_g_a.median())
    assert metrics["residual_c_arm_b_max"] == float(result.residual_c_b.max())
    assert all(math.isfinite(value) for value in metrics.values())
