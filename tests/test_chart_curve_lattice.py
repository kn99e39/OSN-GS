from __future__ import annotations

import torch

from osn_gs.surface.torch_chart_curve_lattice import build_chart_curve_lattice
from osn_gs.surface.torch_intrinsic_chart_atlas import build_local_chart_atlas
from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support
from osn_gs.surface.torch_latent_surface_tangent_frame_field import build_tangent_frame_field


def _v_fold_component(size: int = 14):
    coords = torch.linspace(-2.0, 2.0, size)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    x_flat, y_flat = xx.reshape(-1), yy.reshape(-1)
    zz = torch.where(x_flat >= 0, 0.9 * x_flat, torch.zeros_like(x_flat))
    points = torch.stack([x_flat, y_flat, zz], dim=1)
    support = build_latent_surface_support(points)
    field = build_tangent_frame_field(points, support)
    for component in field.components:
        if component.coherent:
            return component, support
    raise AssertionError("expected at least one coherent component")


def test_chart_curve_lattice_valid_for_each_chart():
    component, support = _v_fold_component()
    atlas = build_local_chart_atlas(component, support.median_spacing)
    assert atlas.charts
    for chart in atlas.charts:
        result = build_chart_curve_lattice(chart.component, component, chart.node_indices, support)
        assert result.lattice.valid is True
        assert len(result.lattice.u_curves) > 0
        assert len(result.lattice.v_curves) > 0


def test_chart_curve_lattice_uses_existing_builder_not_new_seeding():
    import ast
    import inspect

    from osn_gs.surface import torch_chart_curve_lattice as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "build_curve_lattice" in imported_names
    assert not any("seed_curve" in name.lower() for name in imported_names)


def test_truncated_curve_points_all_belong_to_chart():
    component, support = _v_fold_component()
    atlas = build_local_chart_atlas(component, support.median_spacing)
    for chart in atlas.charts:
        result = build_chart_curve_lattice(chart.component, component, chart.node_indices, support)
        if not result.lattice.valid:
            continue
        node_set = frozenset(chart.node_indices)
        for curve in result.lattice.u_curves + result.lattice.v_curves:
            for row in range(int(curve.points.shape[0])):
                point = curve.points[row]
                nearest = int(torch.cdist(point.reshape(1, 3), component.positions).reshape(-1).argmin().item())
                assert nearest in node_set
