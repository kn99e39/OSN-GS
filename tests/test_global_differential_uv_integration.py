from __future__ import annotations

import torch

from osn_gs.surface.torch_global_differential_uv_integration import integrate_global_differential_uv
from osn_gs.surface.torch_latent_surface_edge_differential import build_edge_differentials
from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support
from osn_gs.surface.torch_latent_surface_tangent_frame_field import build_tangent_frame_field
from osn_gs.surface.torch_orientation_preserving_uv_integration import integrate_orientation_preserving_uv


def _flat_component(size: int = 10):
    coords = torch.linspace(-2.0, 2.0, size)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(size * size)], dim=1)
    support = build_latent_surface_support(points)
    field = build_tangent_frame_field(points, support)
    return field.components[0], support


def test_exact_integration_of_planar_constant_frame_field():
    # A flat sheet has a genuinely path-independent differential field --
    # global integration must recover (u, v) matching the tree-integrated
    # potential up to the fixed gauge (translation), with near-zero
    # residual.
    component, support = _flat_component()
    edges = build_edge_differentials(component, support.median_spacing)
    result = integrate_global_differential_uv(component, edges)
    assert result.valid is True
    assert result.overall_residual_rms < 1e-4
    assert result.per_edge_residual_p95 < 1e-3


def test_global_least_squares_recovers_multi_path_cycles():
    # The flat-sheet grid has many independent paths/cycles between any two
    # nodes (not just one spanning tree) -- global integration must use all
    # of them and still recover a consistent, low-residual (u, v).
    component, support = _flat_component(size=8)
    edges = build_edge_differentials(component, support.median_spacing)
    assert len(edges) > len(component.tree_edges)  # genuine cycles present
    result = integrate_global_differential_uv(component, edges)
    assert result.valid is True
    assert result.cycle_edge_residual_rms is not None
    assert result.cycle_edge_residual_rms < 1e-3


def test_curved_but_integrable_synthetic_field_has_low_residual():
    # A gently curved (low-curvature) surface is still well approximated
    # by locally-flat differentials -- residual should stay small relative
    # to the domain extent.
    coords = torch.linspace(-1.0, 1.0, 10)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    zz = 0.02 * (xx.reshape(-1) ** 2 + yy.reshape(-1) ** 2)
    points = torch.stack([xx.reshape(-1), yy.reshape(-1), zz], dim=1)
    support = build_latent_surface_support(points)
    field = build_tangent_frame_field(points, support)
    coherent = [c for c in field.components if c.coherent]
    assert coherent
    component = coherent[0]
    edges = build_edge_differentials(component, support.median_spacing)
    result = integrate_global_differential_uv(component, edges)
    assert result.valid is True
    assert result.overall_residual_rms < 0.1


def test_no_edges_fails_closed():
    result = integrate_global_differential_uv(object(), ())
    assert result.valid is False
    assert result.invalid_reason == "no_supported_edges"


def test_orientation_preserving_matches_base_on_clean_flat_field():
    # No fold present -- candidate C should converge in zero refinement
    # iterations and match candidate B's own (u, v) exactly.
    component, support = _flat_component()
    edges = build_edge_differentials(component, support.median_spacing)
    result = integrate_orientation_preserving_uv(component, edges, support.median_spacing)
    assert result.valid is True
    assert result.refinement_iterations_used == 0
    assert torch.allclose(result.uv, result.base_result.uv, atol=1e-5)


def test_orientation_preserving_initialized_strictly_from_candidate_b():
    component, support = _flat_component()
    edges = build_edge_differentials(component, support.median_spacing)
    base = integrate_global_differential_uv(component, edges)
    refined = integrate_orientation_preserving_uv(component, edges, support.median_spacing)
    assert refined.base_result.uv is not None
    assert torch.allclose(refined.base_result.uv, base.uv, atol=1e-6)


def test_fail_closed_when_base_integration_fails():
    result = integrate_orientation_preserving_uv(object(), (), 1.0)
    assert result.valid is False
    assert result.invalid_reason is not None
    assert "base_integration_failed" in result.invalid_reason


def test_no_pca_fallback_in_integration_modules():
    import ast
    import inspect

    from osn_gs.surface import (
        torch_global_differential_uv_integration as global_module,
        torch_orientation_preserving_uv_integration as orient_module,
    )

    for module in (global_module, orient_module):
        tree = ast.parse(inspect.getsource(module))
        imported_names = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not any("pca" in name.lower() for name in imported_names)
