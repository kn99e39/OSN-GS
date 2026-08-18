from __future__ import annotations

import torch

from osn_gs.surface.torch_chart_overlap_consistency import evaluate_overlap_consistency
from osn_gs.surface.torch_intrinsic_chart_atlas import build_local_chart_atlas
from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support
from osn_gs.surface.torch_latent_surface_tangent_frame_field import build_tangent_frame_field
from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_from_uv


def _v_fold_atlas():
    coords = torch.linspace(-2.0, 2.0, 14)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    x_flat, y_flat = xx.reshape(-1), yy.reshape(-1)
    zz = torch.where(x_flat >= 0, 0.9 * x_flat, torch.zeros_like(x_flat))
    points = torch.stack([x_flat, y_flat, zz], dim=1)
    support = build_latent_surface_support(points)
    field = build_tangent_frame_field(points, support)
    for component in field.components:
        if component.coherent:
            atlas = build_local_chart_atlas(component, support.median_spacing)
            return atlas, support
    raise AssertionError("expected a coherent component")


def test_overlap_consistency_reports_pairs_with_both_fitted():
    atlas, support = _v_fold_atlas()
    surfaces = {}
    uvs = {}
    for chart in atlas.charts:
        uvs[chart.chart_id] = chart.integration.uv
        if int(chart.component.positions.shape[0]) >= 9:
            surfaces[chart.chart_id] = fit_torch_visible_surface_from_uv(
                chart.component.positions, chart.integration.uv, resolution_u=3, resolution_v=3, degree_u=2, degree_v=2,
            )
        else:
            surfaces[chart.chart_id] = None
    results = evaluate_overlap_consistency(list(atlas.charts), surfaces, uvs, support.median_spacing)
    assert isinstance(results, tuple)
    for pair in results:
        if pair.both_fitted:
            assert pair.position_disagreement_p50 is not None
            assert pair.normal_disagreement_degrees_p50 is not None
        else:
            assert pair.position_disagreement_p50 is None


def test_overlap_consistency_never_modifies_charts():
    atlas, support = _v_fold_atlas()
    node_indices_before = [chart.node_indices for chart in atlas.charts]
    surfaces = {chart.chart_id: None for chart in atlas.charts}
    uvs = {chart.chart_id: chart.integration.uv for chart in atlas.charts}
    evaluate_overlap_consistency(list(atlas.charts), surfaces, uvs, support.median_spacing)
    node_indices_after = [chart.node_indices for chart in atlas.charts]
    assert node_indices_before == node_indices_after


def test_overlap_consistency_is_reporting_only_no_capacity_tuning():
    import ast
    import inspect

    from osn_gs.surface import torch_chart_overlap_consistency as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("adaptive_patch_capacity" in name.lower() for name in imported_names)
    assert not any("identifiability" in name.lower() for name in imported_names)
