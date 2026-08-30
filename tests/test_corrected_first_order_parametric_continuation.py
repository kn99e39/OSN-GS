"""Focused contracts for the corrected first-order Taylor arm."""

from __future__ import annotations

import numpy as np
import pytest

from devtools.demo.corrected_first_order_parametric_continuation import (
    EVENT_PLY,
    PRIMARY_ROI,
    _canonical_coverage,
    _event_point_source,
    _surface_from_grid,
    derivative_magnitude_ratio,
    evaluate_first_order_taylor,
    invoke_observed_only_fitter,
)
from devtools.demo.parametric_surface_continuation import build_continuation_control_grid


def _surface():
    import torch

    u = torch.linspace(0.0, 1.0, 6)
    v = torch.linspace(0.0, 1.0, 4)
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    grid = torch.stack([uu, vv, 0.25 * uu.square() + 0.1 * uu * vv], dim=-1)
    return _surface_from_grid(grid.numpy())


def test_corrected_taylor_matches_boundary_and_rescaled_derivative():
    import torch

    surface = _surface()
    c = 0.58
    r = (1.0 - c) / c
    v = torch.tensor([0.0, 0.37, 1.0])
    t = torch.zeros_like(v)
    points, _normals, derivative_t, _derivative_v = evaluate_first_order_taylor(surface, t, v, r)
    uv = torch.stack([torch.ones_like(v), v], dim=1)
    boundary, derivative_u, _derivative_v_observed = surface.evaluate_with_derivatives(uv)
    assert torch.allclose(points, boundary, atol=1e-6, rtol=1e-6)
    assert torch.allclose(derivative_t, r * derivative_u, atol=1e-6, rtol=1e-6)
    ratio = derivative_magnitude_ratio(surface, "corrected_taylor", None, c)
    assert ratio["median"] == pytest.approx(1.0, abs=1e-6)
    assert ratio["p95"] == pytest.approx(1.0, abs=1e-6)


def test_historical_control_column_rule_is_reproducible_and_distinct():
    import torch

    grid = torch.arange(8 * 4 * 3, dtype=torch.float32).reshape(8, 4, 3)
    first = build_continuation_control_grid(grid, 0.58)
    second = build_continuation_control_grid(grid, 0.58)
    assert torch.equal(first, second)
    ratio = (1.0 - 0.58) / 0.58
    expected = grid[-1][None] + torch.linspace(0.0, 1.0, 8)[:, None, None] * ratio * (grid[-1] - grid[-2])[None]
    assert torch.equal(first, expected)


def test_observed_only_case_builder_blocks_withheld_rows_end_to_end():
    # Use the mandatory primary ROI's real fixed axes/bounds, but a tiny
    # deterministic graph so the test never loads the 28.7M-vertex mesh.
    xs = np.linspace(PRIMARY_ROI.u_bounds[0], PRIMARY_ROI.u_bounds[1], 11)
    zs = np.linspace(PRIMARY_ROI.v_bounds[0], PRIMARY_ROI.v_bounds[1], 7)
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    yy = 1.25 + 0.08 * (xx - xs.min()) + 0.03 * (zz - zs.min())
    reference = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    observed = reference[
        ((reference[:, 0] - PRIMARY_ROI.u_bounds[0]) / (PRIMARY_ROI.u_bounds[1] - PRIMARY_ROI.u_bounds[0])) <= PRIMARY_ROI.holdout_u_cut
    ]
    withheld = reference[
        ((reference[:, 0] - PRIMARY_ROI.u_bounds[0]) / (PRIMARY_ROI.u_bounds[1] - PRIMARY_ROI.u_bounds[0])) > PRIMARY_ROI.holdout_u_cut
    ]
    seen = {}

    def spy(points, *, initial_uv):
        seen["points"] = points.detach().cpu().numpy().copy()
        seen["initial_uv"] = initial_uv.detach().cpu().numpy().copy()
        return "fitted-observed-only"

    result, payload = invoke_observed_only_fitter(reference, PRIMARY_ROI, spy, max_fit_points=32)
    assert result == "fitted-observed-only"
    assert np.all([any(np.allclose(row, candidate, atol=1e-6, rtol=0.0) for candidate in observed) for row in seen["points"]])
    assert not any(np.allclose(row, candidate, atol=1e-6, rtol=0.0) for row in seen["points"] for candidate in withheld)
    assert np.all(seen["initial_uv"][:, 0] <= 1.0 + 1e-6)
    assert np.array_equal(np.sort(payload.observed_global_indices), np.sort(np.flatnonzero(np.isin(reference[:, 0], observed[:, 0]))))
    assert payload.withheld_global_indices.size > 0


def test_hero_coverage_keeps_all_event_population_and_names_finite_only(tmp_path):
    evidence = tmp_path / "evidence.npz"
    raycast = tmp_path / "raycast.npz"
    np.savez(evidence, distance=np.array([0.1, np.nan, 0.3], dtype=np.float32))
    np.savez(raycast, counted=np.array([[3, 2]], dtype=np.int64))
    coverage = _canonical_coverage(evidence, raycast, 0.2)
    assert coverage["total_event_count"] == 3
    assert coverage["canonical_all_event_coverage_le_h"] == pytest.approx(1.0 / 3.0)
    assert coverage["finite_only_coverage_le_h"] == pytest.approx(1.0 / 2.0)
    assert coverage["ray_hit_coverage"] == pytest.approx(2.0 / 3.0)


def test_event_panel_uses_actual_renderer_median_samples_when_available():
    if not EVENT_PLY.exists():
        pytest.skip("WL127 presentation PLY is not available in this checkout")
    points, metadata = _event_point_source()
    assert points is not None and len(points) == 21896
    assert metadata["label"] == "Renderer median surface event samples"
    assert metadata["marker_contract"].startswith("actual renderer median surface event samples")
