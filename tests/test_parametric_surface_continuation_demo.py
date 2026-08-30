"""Focused contracts for the isolated meeting-demo continuation module."""

from __future__ import annotations

import numpy as np

from devtools.demo.parametric_surface_continuation import (
    PRIMARY_ROI,
    build_continuation_control_grid,
    build_holdout_partition,
    deterministic_indices,
    evaluate_withheld_geometry,
)


def _synthetic_grid() -> np.ndarray:
    u = np.linspace(0.0, 1.0, 9)
    v = np.linspace(0.0, 1.0, 5)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return np.stack([uu, vv, 0.2 * np.sin(uu * 2.0) + 0.1 * vv], axis=-1).reshape(-1, 3)


def test_boundary_holdout_is_one_sided_and_deterministic() -> None:
    points = _synthetic_grid()
    config = PRIMARY_ROI.__class__(
        name="synthetic",
        semantic_label="synthetic",
        origin=(0.0, 0.0, 0.0),
        axis_u=(1.0, 0.0, 0.0),
        axis_v=(0.0, 1.0, 0.0),
        axis_n=(0.0, 0.0, 1.0),
        u_bounds=(0.0, 1.0),
        v_bounds=(0.0, 1.0),
        n_bounds=(-1.0, 1.0),
        holdout_u_cut=0.62,
    )
    first = build_holdout_partition(points, config)
    second = build_holdout_partition(points, config)
    assert np.array_equal(first.observed_mask, second.observed_mask)
    assert np.array_equal(first.withheld_mask, second.withheld_mask)
    assert first.observed_mask.any() and first.withheld_mask.any()
    assert np.max(first.u_norm[first.observed_mask]) <= 0.62
    assert np.min(first.u_norm[first.withheld_mask]) > 0.62


def test_continuation_is_deterministic_and_boundary_attached() -> None:
    import torch

    grid = torch.arange(8 * 3 * 3, dtype=torch.float32).reshape(8, 3, 3)
    first = build_continuation_control_grid(grid, 0.5)
    second = build_continuation_control_grid(grid, 0.5)
    assert torch.equal(first, second)
    assert torch.equal(first[0], grid[-1])
    expected = grid[-1] + (grid[-1] - grid[-2])
    assert torch.allclose(first[-1], expected)


def test_fit_input_indices_cannot_include_withheld_rows() -> None:
    observed_count = 17
    withheld_count = 9
    fit_indices = deterministic_indices(observed_count, 8)
    fit_global = fit_indices
    withheld_global = np.arange(observed_count, observed_count + withheld_count)
    assert np.intersect1d(fit_global, withheld_global).size == 0


def test_metrics_evaluate_only_withheld_reference_population() -> None:
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    predicted = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    metrics = evaluate_withheld_geometry(reference, predicted, 0.5)
    assert metrics["evaluation_population"] == "withheld_reference_only"
    assert metrics["samples"] == 2
    assert metrics["withheld_reference_coverage"]["fraction_le_h"] == 0.5
