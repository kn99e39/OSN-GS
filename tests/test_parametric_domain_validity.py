from __future__ import annotations

import torch

from osn_gs.surface.torch_parametric_domain_validity import (
    assess_parametric_domain_validity,
    cycle_position_drift_p95,
)


def test_clean_planar_uv_map_is_valid():
    # A flat sheet where (u, v) is literally the actual (x, y) is a
    # perfectly conditioned, fold-free parameter domain.
    coords = torch.linspace(-2.0, 2.0, 12)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    positions = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(144)], dim=1)
    uv = torch.stack([(xx.reshape(-1) + 2.0) / 4.0, (yy.reshape(-1) + 2.0) / 4.0], dim=1)
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(144, 1)
    report = assess_parametric_domain_validity(positions, uv, normals, median_spacing=coords[1] - coords[0])
    assert report.valid is True
    assert report.fold_fraction == 0.0
    assert report.duplicate_incompatible_count == 0


def test_folded_uv_map_is_detected():
    # Fold the parameter domain onto itself: the SAME uv value is assigned
    # to two rows of points mirrored across x=0, so a local Jacobian
    # estimate must flip orientation somewhere.
    coords = torch.linspace(-2.0, 2.0, 12)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    x_flat, y_flat = xx.reshape(-1), yy.reshape(-1)
    positions = torch.stack([x_flat, y_flat, torch.zeros(144)], dim=1)
    # UV folds back on itself in u once x crosses 0 -- a classic foldover.
    folded_u = torch.where(x_flat >= 0, x_flat, -x_flat)
    uv = torch.stack([(folded_u) / 2.0, (y_flat + 2.0) / 4.0], dim=1)
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(144, 1)
    report = assess_parametric_domain_validity(positions, uv, normals, median_spacing=coords[1] - coords[0])
    assert report.valid is False
    assert "uv_orientation_reversal_or_foldover" in report.invalid_reasons
    assert report.fold_fraction > 0.0


def test_degenerate_extent_is_detected():
    positions = torch.randn(10, 3)
    uv = torch.zeros(10, 2)  # every point maps to the same (u, v) -- zero extent
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(10, 1)
    report = assess_parametric_domain_validity(positions, uv, normals, median_spacing=1.0)
    assert report.valid is False
    assert "degenerate_uv_extent" in report.invalid_reasons


def test_extreme_stretch_reported_relative_to_local_spacing():
    # Compress a wide 3D extent into a tiny UV range -- huge local stretch
    # ratio relative to the actual 3D spacing.
    coords = torch.linspace(-10.0, 10.0, 12)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    positions = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(144)], dim=1)
    uv = torch.stack([(xx.reshape(-1) + 10.0) / 20000.0, (yy.reshape(-1) + 10.0) / 20.0], dim=1)
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(144, 1)
    report = assess_parametric_domain_validity(positions, uv, normals, median_spacing=coords[1] - coords[0])
    assert report.stretch_ratio_p95 is not None
    assert report.stretch_ratio_p95 > 100


def test_cycle_position_drift_zero_on_perfectly_flat_field():
    from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support
    from osn_gs.surface.torch_latent_surface_tangent_frame_field import build_tangent_frame_field

    coords = torch.linspace(-2.0, 2.0, 10)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(100)], dim=1)
    support = build_latent_surface_support(points)
    field = build_tangent_frame_field(points, support)
    component = field.components[0]
    drift = cycle_position_drift_p95(component, support.median_spacing)
    assert drift is not None
    assert drift < 1e-2


def test_never_uses_pca():
    import ast
    import inspect

    from osn_gs.surface import torch_parametric_domain_validity as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("pca" in name.lower() for name in imported_names)


def test_no_fit_dependency():
    import ast
    import inspect

    from osn_gs.surface import torch_parametric_domain_validity as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("nurbs" in name.lower() for name in imported_names)
