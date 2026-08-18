from __future__ import annotations

import torch

from osn_gs.surface.torch_curve_lattice_native_fit import fit_curve_lattice_native
from osn_gs.surface.torch_latent_surface_curve_lattice import build_curve_lattice
from osn_gs.surface.torch_latent_surface_support import build_latent_surface_support
from osn_gs.surface.torch_latent_surface_tangent_frame_field import build_tangent_frame_field


def _grid(n: int, extent: float = 3.0) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.linspace(-extent, extent, n)
    uu, vv = torch.meshgrid(coords, coords, indexing="ij")
    return uu.reshape(-1), vv.reshape(-1)


def _bowl_lattice(n: int = 20, extent: float = 3.0, noise: float = 0.01):
    torch.manual_seed(0)
    uu, vv = _grid(n, extent)
    zz = 0.05 * (uu.square() + vv.square()) + torch.randn_like(uu) * noise
    points = torch.stack([uu, vv, zz], dim=1)
    support = build_latent_surface_support(points)
    field = build_tangent_frame_field(points, support)
    component = field.components[0]
    return build_curve_lattice(component, support)


def test_native_fit_from_synchronized_lattice_succeeds():
    lattice = _bowl_lattice()
    result = fit_curve_lattice_native(lattice)
    assert result.valid_lattice is True
    assert result.surface is not None
    assert result.overall_residual.count > 0


def test_capacity_is_fixed_at_6x6_degree_2():
    lattice = _bowl_lattice()
    result = fit_curve_lattice_native(lattice)
    assert result.surface.control_grid.shape == (6, 6, 3)
    assert result.surface.degree_u == 2
    assert result.surface.degree_v == 2


def test_invalid_lattice_never_falls_back_to_pca():
    from osn_gs.surface.torch_latent_surface_curve_lattice import CurveLatticeUV

    invalid_lattice = CurveLatticeUV(False, "insufficient_component_size")
    result = fit_curve_lattice_native(invalid_lattice)
    assert result.valid_lattice is False
    assert result.surface is None


def test_module_never_calls_pca_fit_function():
    import ast
    import inspect

    from osn_gs.surface import torch_curve_lattice_native_fit as module

    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "fit_curve_lattice_native":
            calls = {
                sub.func.id for sub in ast.walk(node)
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
            }
            assert "fit_torch_visible_surface_lsq" not in calls


def test_reports_separate_u_and_v_curve_residuals():
    lattice = _bowl_lattice()
    result = fit_curve_lattice_native(lattice)
    assert result.u_curve_residual.count > 0
    assert result.v_curve_residual.count > 0
