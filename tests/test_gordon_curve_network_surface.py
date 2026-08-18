from __future__ import annotations

import torch

from osn_gs.surface.torch_gordon_curve_network_surface import construct_gordon_surface


def _clean_synthetic_patch(n: int = 400, seed: int = 0):
    torch.manual_seed(seed)
    uu = torch.rand(n)
    vv = torch.rand(n)
    zz = 0.1 * torch.sin(uu * 3.14) * torch.cos(vv * 3.14) + torch.randn(n) * 0.005
    points = torch.stack([uu * 3, vv * 3, zz], dim=1)
    uv = torch.stack([uu, vv], dim=1)
    return points, uv


def test_gordon_construction_on_consistent_uv_network_succeeds():
    points, uv = _clean_synthetic_patch()
    result = construct_gordon_surface(points, uv, resolution_u=6, resolution_v=6, u_curve_count=4, v_curve_count=4)
    assert result.valid is True
    assert result.surface is not None
    assert result.surface.control_grid.shape == (6, 6, 3)


def test_gordon_reproduces_crossing_curves_within_tolerance():
    points, uv = _clean_synthetic_patch()
    result = construct_gordon_surface(points, uv, resolution_u=6, resolution_v=6, u_curve_count=4, v_curve_count=4)
    assert result.valid is True
    predicted = result.surface.evaluate(uv)
    error = (predicted - points).norm(dim=1)
    assert float(error.mean()) < 0.3


def test_gordon_fails_on_incompatible_curve_correspondence():
    # Only 2 points total -- nowhere near enough to populate even the
    # minimum 2x2 level grid at the required per-level population.
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    uv = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    result = construct_gordon_surface(points, uv, resolution_u=6, resolution_v=6, u_curve_count=4, v_curve_count=4)
    assert result.valid is False
    assert result.invalid_reason == "insufficient_populated_curve_levels"
    assert result.surface is None


def test_gordon_never_falls_back_to_pca():
    import ast
    import inspect

    from osn_gs.surface import torch_gordon_curve_network_surface as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("pca" in name.lower() for name in imported_names)


def test_level_assignment_is_deterministic_not_id_dependent():
    from osn_gs.surface.torch_gordon_curve_network_surface import _assign_levels

    values = torch.tensor([0.1, 0.9, 0.5, 0.3, 0.7])
    first = _assign_levels(values, 4)
    second = _assign_levels(values, 4)
    assert torch.equal(first, second)
    # Reordering the SAME values (simulating different input/stable-ID
    # order) must not change each point's own assigned level.
    permutation = torch.tensor([3, 1, 4, 0, 2])
    permuted = _assign_levels(values[permutation], 4)
    assert torch.equal(permuted, first[permutation])


def test_intersection_residual_reported():
    points, uv = _clean_synthetic_patch()
    result = construct_gordon_surface(points, uv, resolution_u=6, resolution_v=6, u_curve_count=4, v_curve_count=4)
    assert result.intersection_grid_residual is not None
    assert result.intersection_grid_residual >= 0.0
