from __future__ import annotations

import torch

from osn_gs.surface.torch_nurbs import fit_torch_visible_surface_from_uv


def _sample_surface(n: int = 200, seed: int = 0):
    torch.manual_seed(seed)
    uu = torch.rand(n)
    vv = torch.rand(n)
    zz = 0.1 * torch.sin(uu * 3.14) * torch.cos(vv * 3.14) + torch.randn(n) * 0.01
    points = torch.stack([uu * 3, vv * 3, zz], dim=1)
    uv = torch.stack([uu, vv], dim=1)
    return points, uv


def test_fit_from_externally_supplied_uv_reaches_low_residual():
    points, uv = _sample_surface()
    surface = fit_torch_visible_surface_from_uv(points, uv, resolution_u=6, resolution_v=6, degree_u=2, degree_v=2)
    predicted = surface.evaluate(uv)
    error = (predicted - points).norm(dim=1)
    assert float(error.mean()) < 0.05


def test_uv_is_not_recomputed_from_pca():
    # A UV that is deliberately inconsistent with the PCA-derived
    # parameterization must still be honored -- if the fitter silently
    # routed back through PCA-UV, evaluating at OUR uv would not track the
    # actual point positions the way it does here.
    points, _true_uv = _sample_surface()
    # Swap U and V so the supplied uv differs from whatever PCA would pick.
    swapped_uv = torch.stack([_true_uv[:, 1], _true_uv[:, 0]], dim=1)
    surface = fit_torch_visible_surface_from_uv(points, swapped_uv, resolution_u=6, resolution_v=6)
    predicted = surface.evaluate(swapped_uv)
    error = (predicted - points).norm(dim=1)
    assert float(error.mean()) < 0.05


def test_single_point_does_not_crash():
    points = torch.tensor([[0.0, 0.0, 0.0]])
    uv = torch.tensor([[0.5, 0.5]])
    surface = fit_torch_visible_surface_from_uv(points, uv, resolution_u=6, resolution_v=6)
    assert surface.control_grid.shape == (6, 6, 3)


def test_control_grid_capacity_is_unchanged_from_default():
    points, uv = _sample_surface()
    surface = fit_torch_visible_surface_from_uv(points, uv, resolution_u=6, resolution_v=6, degree_u=2, degree_v=2)
    assert surface.control_grid.shape[0] == 6
    assert surface.control_grid.shape[1] == 6
    assert surface.degree_u == 2
    assert surface.degree_v == 2
