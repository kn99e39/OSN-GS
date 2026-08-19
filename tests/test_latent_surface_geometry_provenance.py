from __future__ import annotations

import torch

from osn_gs.surface.torch_intrinsic_chart_atlas import build_local_chart_atlas
from osn_gs.surface.torch_latent_surface_coverage_audit import audit_region_latent_coverage
from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support
from osn_gs.surface.torch_latent_surface_tangent_frame_field import build_tangent_frame_field
from osn_gs.surface.torch_latent_surface_visualization_nurbs import (
    REPRESENTATION_KIND,
    fit_visualization_nurbs,
)


def _noisy_curved_points():
    torch.manual_seed(0)
    coords = torch.linspace(-2.0, 2.0, 12)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    zz = 0.1 * torch.sin(xx) * torch.cos(yy) + 0.03 * torch.randn(144).reshape(12, 12)
    return torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)


def test_component_positions_equal_supported_query_positions():
    points = _noisy_curved_points()
    support = build_latent_surface_support(points)
    field = build_tangent_frame_field(points, support)
    component = field.components[0]
    query = support.query_batch(component.raw_positions)
    assert torch.allclose(component.positions, query.positions, atol=1e-5)
    # Genuinely different from raw for noisy input -- the fix is not a no-op.
    assert not torch.allclose(component.positions, component.raw_positions, atol=1e-6)


def test_raw_gaussian_positions_unchanged_by_query():
    points = _noisy_curved_points()
    original = points.clone()
    support = build_latent_surface_support(points)
    build_tangent_frame_field(points, support)
    assert torch.equal(points, original)


def test_normals_still_come_from_latent_support_query():
    points = _noisy_curved_points()
    support = build_latent_surface_support(points)
    field = build_tangent_frame_field(points, support)
    component = field.components[0]
    query = support.query_batch(component.raw_positions)
    assert torch.allclose(component.normals, query.normals, atol=1e-5)


def test_projection_provenance_survives_chart_restriction():
    points = _noisy_curved_points()
    support = build_latent_surface_support(points)
    field = build_tangent_frame_field(points, support)
    component = field.components[0]
    atlas = build_local_chart_atlas(component, support.median_spacing)
    for chart in atlas.charts:
        assert chart.component.raw_positions is not None
        assert chart.component.projection_displacement is not None
        assert chart.component.latent_supported is not None
        assert chart.component.raw_positions.shape == chart.component.positions.shape
        # Displacement must equal positions - raw_positions exactly (identity,
        # not re-derived independently).
        assert torch.allclose(
            chart.component.projection_displacement,
            chart.component.positions - chart.component.raw_positions,
            atol=1e-6,
        )


def test_unsupported_observations_never_become_projected_coverage():
    points = _noisy_curved_points()
    support = build_latent_surface_support(points)
    audit = audit_region_latent_coverage(0, points, support)
    assert audit.unsupported_raw_positions.shape[0] == audit.latent_unsupported_count
    total_unit_nodes = sum(len(unit.node_indices) for unit in audit.units)
    assert total_unit_nodes <= audit.latent_supported_count
    for unit in audit.units:
        query = support.query_batch(unit.raw_positions)
        assert bool(query.supported.all().item())


def test_visualization_nurbs_not_hidden_by_downstream_safety_labels():
    # A tiny, geometrically poor unit (a straight line -- guaranteed to
    # produce a degenerate/unsafe-looking patch) must still materialize
    # and be reported, never silently dropped.
    line = torch.stack([torch.linspace(0, 1, 6), torch.linspace(0, 1, 6), torch.zeros(6)], dim=1)
    result = fit_visualization_nurbs(0, line)
    assert result.materialized is True
    assert result.surface is not None


def test_raw_and_visualization_nurbs_use_distinct_representation_kinds():
    assert REPRESENTATION_KIND == "latent_surface_coverage_visualization_nurbs"
    assert REPRESENTATION_KIND != "worklog102_existing_nurbs"


def test_coverage_audit_does_not_require_frame_coherence():
    import ast
    import inspect

    from osn_gs.surface import torch_latent_surface_coverage_audit as module

    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            node.value = ast.Constant(value="")
    code_only = ast.unparse(tree)
    assert ".coherent" not in code_only
    assert "chart" not in code_only.lower()


def test_visualization_nurbs_never_uses_identifiability_gate():
    import ast
    import inspect

    from osn_gs.surface import torch_latent_surface_visualization_nurbs as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("identifiability" in name.lower() for name in imported_names)
    assert not any("adaptive_patch_capacity" in name.lower() for name in imported_names)
