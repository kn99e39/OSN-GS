from __future__ import annotations

from osn_gs.surface.torch_adaptive_nurbs_capacity import (
    MAX_RESOLUTION,
    MIN_RESOLUTION,
    select_adaptive_control_grid_capacity,
)


def test_capacity_is_deterministic_given_same_inputs():
    first = select_adaptive_control_grid_capacity(4, 4, 400, 1.0, 0.72)
    second = select_adaptive_control_grid_capacity(4, 4, 400, 1.0, 0.72)
    assert first == second


def test_capacity_is_bounded():
    for sample_count in (1, 10, 100, 1000, 100000):
        capacity = select_adaptive_control_grid_capacity(2, 2, sample_count, 1.0, 1.0)
        assert MIN_RESOLUTION <= capacity.resolution_u <= MAX_RESOLUTION
        assert MIN_RESOLUTION <= capacity.resolution_v <= MAX_RESOLUTION


def test_capacity_never_reduced_below_observed_curve_counts_within_bounds():
    capacity = select_adaptive_control_grid_capacity(8, 3, 20, 1.0, 1.0)
    assert capacity.resolution_u >= min(8, MAX_RESOLUTION)


def test_capacity_reflects_aspect_ratio():
    wide = select_adaptive_control_grid_capacity(2, 2, 100, 4.0, 1.0)
    narrow = select_adaptive_control_grid_capacity(2, 2, 100, 1.0, 4.0)
    assert wide.resolution_u >= narrow.resolution_u
    assert wide.resolution_v <= narrow.resolution_v


def test_never_depends_on_fit_or_held_out_result():
    # The selection function's signature accepts only structural
    # quantities (curve counts, sample count, uv extent) -- no fit error,
    # no held-out error, no NURBS import at all.
    import ast
    import inspect

    from osn_gs.surface import torch_adaptive_nurbs_capacity as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("nurbs" in name.lower() for name in imported_names)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "select_adaptive_control_grid_capacity":
            arg_names = {arg.arg for arg in node.args.args}
            assert not any("error" in name.lower() or "residual" in name.lower() for name in arg_names)
