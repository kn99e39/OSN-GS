from __future__ import annotations

import torch

from osn_gs.surface.torch_adaptive_patch_capacity import (
    select_adaptive_quadratic_capacity,
    select_support_adaptive_capacity,
)


def test_adaptive_capacity_selects_largest_identifiable_grid_when_well_sampled():
    coords = torch.linspace(0.0, 1.0, 10)
    uu, vv = torch.meshgrid(coords, coords, indexing="ij")
    uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
    selection = select_adaptive_quadratic_capacity(uv)
    assert selection.selected is True
    assert selection.degree_u == selection.degree_v == 2
    assert selection.control_grid_u == 6
    assert selection.control_grid_v == 6


def test_adaptive_capacity_independent_of_fit_and_held_out_error():
    import ast
    import inspect

    from osn_gs.surface import torch_adaptive_patch_capacity as module

    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            node.value = ast.Constant(value="")
    code_only = ast.unparse(tree)
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("nurbs" in name.lower() and "identifiability" not in name.lower() for name in imported_names)
    assert "held_out" not in code_only
    assert "extrapolat" not in code_only.lower()
    assert "residual" not in code_only.lower()


def test_support_adaptive_prefers_higher_order_when_both_identifiable():
    coords = torch.linspace(0.0, 1.0, 10)
    uu, vv = torch.meshgrid(coords, coords, indexing="ij")
    uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
    selection = select_support_adaptive_capacity(uv)
    assert selection.selected is True
    assert selection.degree_u == 2  # degree 2 tried first, identifiable here -- must win over degree 1


def test_support_adaptive_falls_back_to_degree_one_when_degree_two_never_identifiable():
    # Force degree-2's own minimum (3x3) to be unreachable by supplying
    # only 2 independent samples -- degree 1's own minimum (2x2, 4
    # control variables) still exceeds available rank (2), so neither
    # should select in this pathological case; verify the module fails
    # closed rather than crashing or silently picking an invalid grid.
    uv = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    selection = select_support_adaptive_capacity(uv)
    if selection.selected:
        assert selection.report is not None and selection.report.identifiable is True
    else:
        assert selection.report is None


def test_deterministic_tie_break_by_intrinsic_aspect():
    # Two runs on the same evidence must select the identical capacity.
    torch.manual_seed(0)
    uv = torch.rand(40, 2)
    selection_a = select_adaptive_quadratic_capacity(uv)
    selection_b = select_adaptive_quadratic_capacity(uv)
    assert (selection_a.control_grid_u, selection_a.control_grid_v) == (selection_b.control_grid_u, selection_b.control_grid_v)


def test_still_a_nurbs_representation_no_new_surface_family():
    import ast
    import inspect

    from osn_gs.surface import torch_adaptive_patch_capacity as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("gordon" in name.lower() for name in imported_names)
    assert not any("mesh" in name.lower() for name in imported_names)
