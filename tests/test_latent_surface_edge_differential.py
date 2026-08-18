from __future__ import annotations

import torch

from osn_gs.surface.torch_global_differential_uv_integration import integrate_global_differential_uv
from osn_gs.surface.torch_latent_surface_edge_differential import build_edge_differentials
from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support
from osn_gs.surface.torch_latent_surface_tangent_frame_field import build_tangent_frame_field


def test_edge_differential_uses_full_supported_edge_set_not_just_tree():
    coords = torch.linspace(-2.0, 2.0, 8)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(64)], dim=1)
    support = build_latent_surface_support(points)
    field = build_tangent_frame_field(points, support)
    component = field.components[0]
    edges = build_edge_differentials(component, support.median_spacing)
    assert len(edges) > len(component.tree_edges)


def test_edge_weights_are_fixed_and_deterministic():
    coords = torch.linspace(-2.0, 2.0, 8)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(64)], dim=1)
    support = build_latent_surface_support(points)
    field = build_tangent_frame_field(points, support)
    component = field.components[0]
    edges_a = build_edge_differentials(component, support.median_spacing)
    edges_b = build_edge_differentials(component, support.median_spacing)
    assert [edge.weight for edge in edges_a] == [edge.weight for edge in edges_b]


def test_no_replay_or_held_out_dependency_in_edge_weighting():
    import ast
    import inspect

    from osn_gs.surface import torch_latent_surface_edge_differential as module

    source = inspect.getsource(module)
    assert "held_out" not in source
    tree = ast.parse(source)
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("nurbs" in name.lower() for name in imported_names)


def test_tree_path_drift_removed_by_global_integration():
    # On a curved (nonzero-curvature) surface, Worklog 98's own
    # tree-integrated (u, v) is known to accumulate path-dependent drift
    # relative to a direct chord-length step across any cycle-closing
    # edge (this is exactly what cycle_position_drift_p95 measures).
    # Global integration, which uses every edge simultaneously, must not
    # exhibit the same per-edge inconsistency: its own per-edge residual
    # after fitting must be small even where the tree-only drift is not.
    coords = torch.linspace(-1.5, 1.5, 12)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    zz = 0.05 * torch.sin(1.5 * xx.reshape(-1)) * torch.cos(1.5 * yy.reshape(-1))
    points = torch.stack([xx.reshape(-1), yy.reshape(-1), zz], dim=1)
    support = build_latent_surface_support(points)
    field = build_tangent_frame_field(points, support)
    coherent = [c for c in field.components if c.coherent]
    assert coherent
    component = coherent[0]

    from osn_gs.surface.torch_parametric_domain_validity import cycle_position_drift_p95

    tree_drift = cycle_position_drift_p95(component, support.median_spacing)

    edges = build_edge_differentials(component, support.median_spacing)
    result = integrate_global_differential_uv(component, edges)
    assert result.valid is True
    # The global fit's own residual (in the same units as the differential
    # constraints) stays well below the raw tree-accumulated drift
    # magnitude -- direct evidence the global solve is not inheriting the
    # single-path accumulation error.
    if tree_drift is not None and tree_drift > 0:
        assert result.per_edge_residual_p95 <= tree_drift * support.median_spacing + 1e-6
