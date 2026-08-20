from __future__ import annotations

import torch

from osn_gs.surface.torch_gaussian_covariance_frame import covariance_from_scale_rotation
from osn_gs.surface.torch_gaussian_surface_orientation import (
    SEPARABILITY_CODES,
    SEPARABILITY_ISOTROPIC,
    SEPARABILITY_NON_FINITE,
    SEPARABILITY_NORMAL_AXIS_DEGENERATE,
    derive_surface_orientation_from_covariance,
    derive_surface_orientation_from_scale_rotation,
    unsigned_normal_alignment,
)


def test_normal_is_thinnest_axis_regardless_of_stored_scale_order():
    """The stored `scale_0/1/2` order carries no meaning -- the derived normal
    must follow the SMALLEST scale wherever it happens to sit."""

    positions = torch.zeros((3, 3))
    identity_rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)
    # Same physical surfel, thin axis stored in slot 2, 0 and 1 respectively.
    scaling = torch.tensor([
        [0.30, 0.20, 0.01],
        [0.01, 0.30, 0.20],
        [0.20, 0.01, 0.30],
    ])
    orientation = derive_surface_orientation_from_scale_rotation(positions, scaling, identity_rotation)
    expected = torch.tensor([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    assert torch.allclose(unsigned_normal_alignment(orientation.surface_normal, expected), torch.ones(3), atol=1e-6)
    # Eigenvalues always come back descending, whatever the storage order was.
    assert torch.all(orientation.eigenvalues[:, 0] >= orientation.eigenvalues[:, 1])
    assert torch.all(orientation.eigenvalues[:, 1] >= orientation.eigenvalues[:, 2])


def test_scale_rotation_and_covariance_paths_share_one_normal_definition():
    """Both entry points must produce the same frame -- no module may quietly
    use a different notion of 'the normal'."""

    torch.manual_seed(11)
    count = 64
    positions = torch.randn(count, 3)
    scaling = torch.rand(count, 3) * 0.3 + 0.01
    scaling[:, 2] *= 0.02  # force a clearly thinnest axis so ordering is well posed
    rotation = torch.nn.functional.normalize(torch.randn(count, 4), dim=-1)

    from_parameters = derive_surface_orientation_from_scale_rotation(positions, scaling, rotation)
    from_covariance = derive_surface_orientation_from_covariance(
        positions, covariance_from_scale_rotation(scaling, rotation)
    )
    assert from_parameters.source == "scale_rotation"
    assert from_covariance.source == "covariance"
    assert torch.allclose(from_parameters.surface_normal, from_covariance.surface_normal, atol=1e-4)
    assert torch.allclose(from_parameters.tangent_axis_u, from_covariance.tangent_axis_u, atol=1e-4)
    assert torch.allclose(from_parameters.tangent_axis_v, from_covariance.tangent_axis_v, atol=1e-4)
    assert torch.allclose(from_parameters.eigenvalues, from_covariance.eigenvalues, atol=1e-5)


def test_frame_is_orthonormal_and_right_handed():
    torch.manual_seed(3)
    count = 128
    positions = torch.randn(count, 3)
    scaling = torch.rand(count, 3) * 0.5 + 0.001
    rotation = torch.nn.functional.normalize(torch.randn(count, 4), dim=-1)
    orientation = derive_surface_orientation_from_scale_rotation(positions, scaling, rotation)

    u, v, n = orientation.tangent_axis_u, orientation.tangent_axis_v, orientation.surface_normal
    assert torch.allclose(u.norm(dim=-1), torch.ones(count), atol=1e-5)
    assert torch.allclose(v.norm(dim=-1), torch.ones(count), atol=1e-5)
    assert torch.allclose(n.norm(dim=-1), torch.ones(count), atol=1e-5)
    assert torch.allclose((u * v).sum(-1), torch.zeros(count), atol=1e-5)
    assert torch.allclose((u * n).sum(-1), torch.zeros(count), atol=1e-5)
    assert torch.allclose((v * n).sum(-1), torch.zeros(count), atol=1e-5)
    determinant = torch.linalg.det(torch.stack([u, v, n], dim=-1))
    assert torch.allclose(determinant, torch.ones(count), atol=1e-5)


def test_sign_gauge_is_deterministic_and_quaternion_double_cover_agrees():
    """q and -q are the same rotation; the canonical sign gauge must not make
    them look like two different surfels."""

    torch.manual_seed(5)
    count = 40
    positions = torch.randn(count, 3)
    scaling = torch.rand(count, 3) * 0.4 + 0.02
    rotation = torch.nn.functional.normalize(torch.randn(count, 4), dim=-1)

    first = derive_surface_orientation_from_scale_rotation(positions, scaling, rotation)
    repeat = derive_surface_orientation_from_scale_rotation(positions, scaling, rotation)
    negated = derive_surface_orientation_from_scale_rotation(positions, scaling, -rotation)

    assert torch.equal(first.surface_normal, repeat.surface_normal)
    assert torch.equal(first.tangent_axis_u, repeat.tangent_axis_u)
    assert torch.allclose(first.surface_normal, negated.surface_normal, atol=1e-5)
    assert torch.allclose(first.tangent_axis_u, negated.tangent_axis_u, atol=1e-5)


def test_unsigned_alignment_treats_opposite_normals_as_identical():
    normals = torch.nn.functional.normalize(torch.randn(32, 3), dim=-1)
    assert torch.allclose(
        unsigned_normal_alignment(normals, -normals), torch.ones(32), atol=1e-6
    )
    assert torch.allclose(
        unsigned_normal_alignment(normals, normals), unsigned_normal_alignment(normals, -normals), atol=1e-6
    )


def test_degenerate_and_non_finite_rows_still_produce_exactly_one_row_each():
    """Coverage contract at the representation stage: a Gaussian with no usable
    orientation evidence is LABELLED, never removed."""

    positions = torch.zeros((4, 3))
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1)
    scaling = torch.tensor([
        [0.30, 0.20, 0.01],  # well-posed surfel
        [0.20, 0.20, 0.20],  # perfectly isotropic -- no orientation evidence
        [0.30, 0.10, 0.09],  # lambda2 ~ lambda3 -- normal direction unresolved
        [float("nan"), 0.1, 0.1],  # non-finite input
    ])
    orientation = derive_surface_orientation_from_scale_rotation(positions, scaling, rotation)

    assert len(orientation) == 4
    assert orientation.surface_normal.shape == (4, 3)
    assert torch.isfinite(orientation.surface_normal).all()
    assert torch.isfinite(orientation.tangent_axis_u).all()
    assert torch.isfinite(orientation.tangent_axis_v).all()
    labels = [SEPARABILITY_CODES[int(code)] for code in orientation.axis_separability]
    assert labels[1] == SEPARABILITY_ISOTROPIC
    assert labels[2] == SEPARABILITY_NORMAL_AXIS_DEGENERATE
    assert labels[3] == SEPARABILITY_NON_FINITE
    assert sum(orientation.separability_counts().values()) == 4


def test_non_finite_covariance_row_survives_the_covariance_path():
    covariance = torch.eye(3).reshape(1, 3, 3).repeat(3, 1, 1)
    covariance[0] = torch.diag(torch.tensor([0.09, 0.04, 0.0001]))
    covariance[2] = float("inf")
    positions = torch.zeros((3, 3))
    orientation = derive_surface_orientation_from_covariance(positions, covariance)
    assert len(orientation) == 3
    assert torch.isfinite(orientation.surface_normal).all()
    assert SEPARABILITY_CODES[int(orientation.axis_separability[2])] == SEPARABILITY_NON_FINITE


def test_provenance_ids_and_positions_are_carried_through_unchanged():
    positions = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    original = positions.clone()
    scaling = torch.tensor([[0.3, 0.2, 0.01], [0.3, 0.2, 0.01]])
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    ids = torch.tensor([77, 4242])

    orientation = derive_surface_orientation_from_scale_rotation(positions, scaling, rotation, ids)
    assert torch.equal(orientation.gaussian_ids, ids)
    assert torch.equal(orientation.positions, original)
    assert torch.equal(positions, original)
    # Default provenance is the positional index.
    default_ids = derive_surface_orientation_from_scale_rotation(positions, scaling, rotation).gaussian_ids
    assert torch.equal(default_ids, torch.arange(2))
