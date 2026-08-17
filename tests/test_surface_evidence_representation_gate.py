from __future__ import annotations

import torch

from osn_gs.surface.torch_gaussian_covariance_frame import (
    covariance_from_scale_rotation,
    extract_covariance_frame,
)
from osn_gs.surface.torch_surface_evidence_representation_gate import (
    REPRESENTATION_CENTER_LATENT_SURFACE,
    REPRESENTATION_COVARIANCE_SURFEL_SUPPORT,
    REPRESENTATION_HYBRID_LATENT_PLUS_SUPPORT,
    REPRESENTATION_RAW_CENTER_BASELINE,
    REPRESENTATIONS,
    build_representation_evidence,
)


def _grid(n: int, extent: float = 3.0) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.linspace(-extent, extent, n)
    uu, vv = torch.meshgrid(coords, coords, indexing="ij")
    return uu.reshape(-1), vv.reshape(-1)


def _thick_sheet_fixture(seed: int = 0):
    torch.manual_seed(seed)
    uu, vv = _grid(10)
    zz = torch.randn_like(uu) * 0.3
    points = torch.stack([uu, vv, zz], dim=1)
    scale = torch.tensor([[0.05, 0.05, 0.01]]).repeat(points.shape[0], 1)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(points.shape[0], 1)
    covariance = covariance_from_scale_rotation(scale, quat)
    return points, covariance


def test_raw_center_baseline_is_a_pure_passthrough():
    points, covariance = _thick_sheet_fixture()
    result = build_representation_evidence(REPRESENTATION_RAW_CENTER_BASELINE, points, covariance)
    assert torch.equal(result.positions, points)
    assert torch.equal(result.covariance, covariance)
    assert result.displacement_over_spacing == 0.0


def test_covariance_surfel_support_keeps_raw_positions_and_orientation():
    # Representation C never asserts the covariance normal is ground truth;
    # it must not move centers or reorient covariance, only disclose
    # support extent.
    points, covariance = _thick_sheet_fixture()
    result = build_representation_evidence(REPRESENTATION_COVARIANCE_SURFEL_SUPPORT, points, covariance)
    assert torch.equal(result.positions, points)
    assert torch.equal(result.covariance, covariance)
    assert result.support_radius_scale is not None
    assert result.support_radius_scale > 0.0


def test_center_latent_surface_moves_points_and_reorients_covariance():
    points, covariance = _thick_sheet_fixture()
    result = build_representation_evidence(REPRESENTATION_CENTER_LATENT_SURFACE, points, covariance)
    assert not torch.equal(result.positions, points)
    assert result.displacement_over_spacing is not None
    assert result.displacement_over_spacing > 0.0
    # Covariance orientation must reflect the latent tangent frame, not the
    # raw per-Gaussian orientation (which was identical for every point in
    # this fixture, so the adapted frame should differ per point once
    # local plane fits diverge across the noisy thick sheet).
    raw_frame = extract_covariance_frame(covariance)
    adapted_frame = extract_covariance_frame(result.covariance)
    assert not torch.allclose(adapted_frame.normal_candidate, raw_frame.normal_candidate)


def test_center_latent_surface_is_not_independent_per_point_projection():
    # Cross-neighborhood consistency enforcement (Jacobi-style consensus
    # averaging) must differ from a naive independent per-point plane
    # projection -- verified by checking the module's own internal
    # consensus step actually blends neighbor consensus, not just each
    # point's own local projection.
    from osn_gs.surface.torch_surface_evidence_representation_gate import (
        _center_latent_surface_positions,
    )

    points, _covariance = _thick_sheet_fixture()
    consensus_positions, _displacement = _center_latent_surface_positions(points)

    # An independent one-shot per-point projection (no neighbor blending)
    # would move every point fully onto ITS OWN local plane. The consensus
    # result must generally differ from that because it blends toward the
    # neighbor average instead.
    from osn_gs.surface.torch_chart_unit_local_center_geometry_attribution import (
        _knn_indices,
        _local_plane_normal,
    )

    k = min(8, points.shape[0] - 1)
    neighbor_indices = _knn_indices(points, k)
    independent_projection = points.clone()
    for node in range(points.shape[0]):
        neighborhood = torch.cat([torch.tensor([node]), neighbor_indices[node]])
        local_points = points[neighborhood]
        normal = _local_plane_normal(local_points)
        plane_point = local_points.mean(dim=0)
        offset = (points[node] - plane_point) @ normal
        independent_projection[node] = points[node] - offset * normal

    assert not torch.allclose(consensus_positions, independent_projection)


def test_hybrid_uses_latent_positions_with_raw_footprint_support_extent():
    points, covariance = _thick_sheet_fixture()
    latent = build_representation_evidence(REPRESENTATION_CENTER_LATENT_SURFACE, points, covariance)
    hybrid = build_representation_evidence(REPRESENTATION_HYBRID_LATENT_PLUS_SUPPORT, points, covariance)
    assert torch.allclose(hybrid.positions, latent.positions)
    assert hybrid.support_radius_scale is not None
    assert hybrid.support_radius_scale > 0.0


def test_all_four_representations_produce_valid_shapes_for_every_class():
    points, covariance = _thick_sheet_fixture()
    for representation in REPRESENTATIONS:
        result = build_representation_evidence(representation, points, covariance)
        assert result.positions.shape == points.shape
        assert result.covariance.shape == covariance.shape


def test_never_mutates_raw_inputs():
    points, covariance = _thick_sheet_fixture()
    points_copy = points.clone()
    covariance_copy = covariance.clone()
    for representation in REPRESENTATIONS:
        build_representation_evidence(representation, points, covariance)
    assert torch.equal(points, points_copy)
    assert torch.equal(covariance, covariance_copy)
