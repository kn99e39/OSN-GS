from __future__ import annotations

import torch

from osn_gs.surface.torch_latent_surface_curve_lattice import build_curve_lattice
from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support
from osn_gs.surface.torch_latent_surface_tangent_frame_field import build_tangent_frame_field


def _grid(n: int, extent: float = 3.0) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.linspace(-extent, extent, n)
    uu, vv = torch.meshgrid(coords, coords, indexing="ij")
    return uu.reshape(-1), vv.reshape(-1)


def _bowl_component(n: int = 20, extent: float = 3.0, noise: float = 0.01):
    torch.manual_seed(0)
    uu, vv = _grid(n, extent)
    zz = 0.05 * (uu.square() + vv.square()) + torch.randn_like(uu) * noise
    points = torch.stack([uu, vv, zz], dim=1)
    support = build_latent_surface_support(points)
    field = build_tangent_frame_field(points, support)
    return field.components[0], support


def test_intrinsic_uv_valid_from_coherent_field():
    component, support = _bowl_component()
    lattice = build_curve_lattice(component, support)
    assert lattice.valid is True
    assert lattice.uv.min() >= -1e-5
    assert lattice.uv.max() <= 1.0 + 1e-5


def test_uv_matches_point_count():
    component, support = _bowl_component()
    lattice = build_curve_lattice(component, support)
    assert lattice.points.shape[0] == lattice.uv.shape[0]
    assert lattice.points.shape[0] == len(component.node_indices)


def test_u_and_v_curves_are_generated():
    component, support = _bowl_component()
    lattice = build_curve_lattice(component, support)
    assert len(lattice.u_curves) > 0
    assert len(lattice.v_curves) > 0
    for curve in lattice.u_curves + lattice.v_curves:
        assert curve.points.shape[0] >= 1
        assert curve.uv.shape[0] == curve.points.shape[0]


def test_incoherent_component_is_rejected_without_pca_repair():
    component, support = _bowl_component()
    incoherent = component.__class__(
        node_indices=component.node_indices, positions=component.positions, normals=component.normals,
        e_u=component.e_u, e_v=component.e_v, u=component.u, v=component.v,
        tree_edges=component.tree_edges, holonomy_edges=component.holonomy_edges,
        singularities=component.singularities, coherent=False, incoherence_reason="holonomy_inconsistency",
        anchor_seed_type=None,
    )
    lattice = build_curve_lattice(incoherent, support)
    assert lattice.valid is False
    assert lattice.invalid_reason == "holonomy_inconsistency"
    assert lattice.points is None


def test_never_uses_pca():
    import ast
    import inspect

    from osn_gs.surface import torch_latent_surface_curve_lattice as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("pca" in name.lower() for name in imported_names)


def test_no_fit_driven_modification():
    # build_curve_lattice must never call the NURBS fitter -- lattice
    # construction is decided purely from the field, before any fit.
    import ast
    import inspect

    from osn_gs.surface import torch_latent_surface_curve_lattice as module

    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "build_curve_lattice":
            calls = {
                sub.func.id for sub in ast.walk(node)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
            }
            assert not any("fit" in name.lower() for name in calls)
