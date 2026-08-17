from __future__ import annotations

import torch

from osn_gs.surface.torch_chart_unit_latent_midsurface_attribution import (
    attribute_latent_midsurface_recoverability,
)


def _grid(n: int, extent: float = 3.0) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.linspace(-extent, extent, n)
    uu, vv = torch.meshgrid(coords, coords, indexing="ij")
    return uu.reshape(-1), vv.reshape(-1)


def test_flat_sheet_has_zero_thickness_and_full_recoverability():
    uu, vv = _grid(8)
    points = torch.stack([uu, vv, torch.zeros_like(uu)], dim=1)
    result = attribute_latent_midsurface_recoverability(points, list(range(points.shape[0])))
    assert result.local_thickness_over_spacing_evidence_weighted == 0.0
    assert result.raw_open_or_nonmanifold_fraction == 0.0
    assert result.diagnostic_open_or_nonmanifold_fraction == 0.0
    assert result.curvature_preserved is True
    assert result.observed_support_band_fidelity_fraction == 1.0


def test_thick_unimodal_sheet_collapses_to_better_manifold_topology():
    # A thick single sheet (Gaussian-noise normal-direction spread) should
    # show real thickness, and the diagnostic thickness-collapsed
    # projection should show equal-or-better valid face incidence than the
    # raw center geometry -- the core TOPOLOGY_RECOVERABILITY claim.
    torch.manual_seed(0)
    uu, vv = _grid(10)
    zz = torch.randn_like(uu) * 0.3
    points = torch.stack([uu, vv, zz], dim=1)
    result = attribute_latent_midsurface_recoverability(points, list(range(points.shape[0])))
    assert result.local_thickness_over_spacing_evidence_weighted > 0.05
    assert result.valid_local_face_incidence_fraction >= result.raw_valid_local_face_incidence_fraction
    assert result.diagnostic_open_or_nonmanifold_fraction <= result.raw_open_or_nonmanifold_fraction
    # Curvature was genuinely near zero before collapse; must not falsely
    # trip the flattening guard.
    assert result.curvature_preserved is True


def test_curved_bowl_preserves_curvature_after_collapse():
    # A genuinely curved (not thick) single sheet must not be flattened by
    # the diagnostic collapse: curvature after should stay close to
    # curvature before, not drop toward zero.
    uu, vv = _grid(10, extent=4.0)
    zz = 0.15 * (uu**2 + vv**2)
    points = torch.stack([uu, vv, zz], dim=1)
    result = attribute_latent_midsurface_recoverability(points, list(range(points.shape[0])))
    assert result.mean_curvature_before is not None
    assert result.mean_curvature_before > 0.0
    assert result.curvature_preserved is True
    ratio = result.mean_curvature_after / result.mean_curvature_before
    assert ratio > 0.5  # curvature substantially retained, not just above the floor


def test_projection_never_exceeds_observed_support_band_for_bounded_thickness():
    torch.manual_seed(1)
    uu, vv = _grid(10)
    zz = torch.randn_like(uu) * 0.2
    points = torch.stack([uu, vv, zz], dim=1)
    result = attribute_latent_midsurface_recoverability(points, list(range(points.shape[0])))
    assert result.observed_support_band_fidelity_fraction is not None
    assert result.observed_support_band_fidelity_fraction >= 0.75


def test_tiny_member_count_reports_non_recoverable_defaults():
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = attribute_latent_midsurface_recoverability(points, [0, 1, 2])
    assert result.member_count == 3
    assert result.raw_open_or_nonmanifold_fraction == 1.0
    assert result.diagnostic_open_or_nonmanifold_fraction == 1.0
    assert result.curvature_preserved is False


def test_never_mutates_input_positions():
    torch.manual_seed(2)
    uu, vv = _grid(8)
    zz = torch.randn_like(uu) * 0.3
    points = torch.stack([uu, vv, zz], dim=1)
    original = points.clone()
    attribute_latent_midsurface_recoverability(points, list(range(points.shape[0])))
    assert torch.equal(points, original)


def test_never_reads_covariance_signature():
    import ast
    import inspect

    from osn_gs.surface import torch_chart_unit_latent_midsurface_attribution as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("covariance" in name.lower() for name in imported_names)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            arg_names = [arg.arg for arg in node.args.args]
            assert not any("covariance" in name.lower() for name in arg_names)
