from __future__ import annotations

import ast
import inspect

import pytest
import torch

from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.gaussian.torch_surfel_model import TorchGaussianSurfelModel
from osn_gs.surface.torch_coverage_first_subset_partition import partition_gaussian_subsets, partition_accounting
from osn_gs.surface.torch_gaussian_surface_orientation import unsigned_normal_alignment
from osn_gs.surface.torch_surfel_surface_orientation import derive_surface_orientation_from_surfel


def _identity_quaternion(count: int) -> torch.Tensor:
    return torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1)


def _build_surfel_model(
    positions: torch.Tensor, rotation: torch.Tensor, scale_u: float = 0.05, scale_v: float = 0.04
) -> TorchGaussianSurfelModel:
    count = int(positions.shape[0])
    model = TorchGaussianSurfelModel(sh_degree=0, device="cpu")
    model.replace_tensors(
        xyz=positions,
        features_dc=torch.zeros(count, 1, 3),
        features_rest=torch.zeros(count, 0, 3),
        opacity=torch.zeros(count, 1),
        scaling=torch.log(torch.tensor([[scale_u, scale_v]])).repeat(count, 1),
        rotation=rotation,
        uncertain_confidence=torch.zeros(count, 1),
        uncertain_mask=torch.zeros(count, dtype=torch.bool),
        surface_uv=torch.zeros(count, 2),
        cluster_ids=torch.full((count,), -1, dtype=torch.long),
        stable_gaussian_ids=torch.arange(1000, 1000 + count, dtype=torch.long),
    )
    return model


# --------------------------------------------------------------------------
# scale_dim / no third scale
# --------------------------------------------------------------------------


def test_surfel_model_scale_dim_is_two_with_no_trainable_normal_scale():
    model = _build_surfel_model(torch.zeros(3, 3), _identity_quaternion(3))
    assert TorchGaussianSurfelModel.scale_dim == 2
    assert model.scale_dim == 2
    assert model.get_scaling.shape == (3, 2)
    assert model._scaling.shape[1] == 2
    # No third column exists anywhere to introduce normal-direction extent.
    with pytest.raises(IndexError):
        model.get_scaling[:, 2]


def test_base_volumetric_model_is_unaffected_by_the_surfel_subclass():
    assert TorchGaussianModel.scale_dim == 3
    volumetric = TorchGaussianModel(sh_degree=0, device="cpu")
    volumetric.replace_tensors(
        xyz=torch.zeros(2, 3),
        features_dc=torch.zeros(2, 1, 3),
        features_rest=torch.zeros(2, 0, 3),
        opacity=torch.zeros(2, 1),
        scaling=torch.zeros(2, 3),
        rotation=_identity_quaternion(2),
        uncertain_confidence=torch.zeros(2, 1),
        uncertain_mask=torch.zeros(2, dtype=torch.bool),
        surface_uv=torch.zeros(2, 2),
        cluster_ids=torch.full((2,), -1, dtype=torch.long),
    )
    assert volumetric.get_scaling.shape == (2, 3)


# --------------------------------------------------------------------------
# intrinsic normal is read, never recomputed
# --------------------------------------------------------------------------


def test_derived_orientation_matches_model_intrinsic_properties_exactly():
    torch.manual_seed(0)
    count = 20
    positions = torch.randn(count, 3)
    rotation = torch.nn.functional.normalize(torch.randn(count, 4), dim=-1)
    model = _build_surfel_model(positions, rotation)

    orientation = derive_surface_orientation_from_surfel(model)

    assert torch.equal(orientation.tangent_axis_u, model.get_tangent_u)
    assert torch.equal(orientation.tangent_axis_v, model.get_tangent_v)
    assert torch.equal(orientation.surface_normal, model.get_normal)
    assert orientation.source == "surfel_intrinsic"


def test_normal_equals_cross_of_trained_tangents_not_a_recomputation():
    """t_w must equal t_u x t_v of THIS model's own trained rotation -- the
    module reads it, it does not run any decomposition to rediscover it."""

    torch.manual_seed(1)
    count = 16
    positions = torch.randn(count, 3)
    rotation = torch.nn.functional.normalize(torch.randn(count, 4), dim=-1)
    model = _build_surfel_model(positions, rotation)
    orientation = derive_surface_orientation_from_surfel(model)

    cross = torch.cross(orientation.tangent_axis_u, orientation.tangent_axis_v, dim=-1)
    assert torch.allclose(orientation.surface_normal, cross, atol=1e-5)


def test_orientation_is_independent_of_tangent_scale_magnitude():
    """Unlike the volumetric path, axis assignment must not depend on s_u vs s_v
    -- there is no eigenvalue-driven reordering for a surfel."""

    torch.manual_seed(2)
    count = 10
    positions = torch.randn(count, 3)
    rotation = torch.nn.functional.normalize(torch.randn(count, 4), dim=-1)

    small_v = _build_surfel_model(positions, rotation, scale_u=0.05, scale_v=0.001)
    large_v = _build_surfel_model(positions, rotation, scale_u=0.05, scale_v=0.20)

    orientation_small = derive_surface_orientation_from_surfel(small_v)
    orientation_large = derive_surface_orientation_from_surfel(large_v)

    assert torch.equal(orientation_small.tangent_axis_u, orientation_large.tangent_axis_u)
    assert torch.equal(orientation_small.tangent_axis_v, orientation_large.tangent_axis_v)
    assert torch.equal(orientation_small.surface_normal, orientation_large.surface_normal)


def test_module_contains_no_eigendecomposition_or_covariance_construction():
    """Static proof: this module must never call eigh or build a covariance
    to recover a normal the surfel already stores intrinsically. Every string
    literal (docstrings AND error-message text, which legitimately mentions
    the forbidden words while explaining their absence) is stripped before
    scanning, leaving only actual code tokens -- names, calls, imports."""

    from osn_gs.surface import torch_surfel_surface_orientation

    tree = ast.parse(inspect.getsource(torch_surfel_surface_orientation))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
        elif isinstance(node, ast.JoinedStr):
            node.values = []
    code_only = ast.unparse(tree)

    forbidden_tokens = ("eigh", "eigenvalue", "covariance", "Sigma", "_batched_eigh")
    for token in forbidden_tokens:
        assert token not in code_only, f"found forbidden token {token!r} in torch_surfel_surface_orientation.py code"

    imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
    for name in imported:
        assert "covariance_frame" not in name, f"must not import {name}"


def test_fails_closed_on_a_volumetric_model():
    """No silent fallback to the covariance-minor-axis normal definition."""

    volumetric = TorchGaussianModel(sh_degree=0, device="cpu")
    volumetric.replace_tensors(
        xyz=torch.zeros(1, 3),
        features_dc=torch.zeros(1, 1, 3),
        features_rest=torch.zeros(1, 0, 3),
        opacity=torch.zeros(1, 1),
        scaling=torch.zeros(1, 3),
        rotation=_identity_quaternion(1),
        uncertain_confidence=torch.zeros(1, 1),
        uncertain_mask=torch.zeros(1, dtype=torch.bool),
        surface_uv=torch.zeros(1, 2),
        cluster_ids=torch.full((1,), -1, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="scale_dim"):
        derive_surface_orientation_from_surfel(volumetric)


# --------------------------------------------------------------------------
# stable ID provenance
# --------------------------------------------------------------------------


def test_gaussian_ids_default_to_the_model_own_stable_ids():
    positions = torch.zeros(3, 3)
    model = _build_surfel_model(positions, _identity_quaternion(3))
    orientation = derive_surface_orientation_from_surfel(model)
    assert torch.equal(orientation.gaussian_ids, torch.tensor([1000, 1001, 1002]))


def test_explicit_gaussian_ids_override_the_model_default():
    positions = torch.zeros(2, 3)
    model = _build_surfel_model(positions, _identity_quaternion(2))
    orientation = derive_surface_orientation_from_surfel(model, gaussian_ids=torch.tensor([7, 8]))
    assert torch.equal(orientation.gaussian_ids, torch.tensor([7, 8]))


def test_source_tensors_are_unchanged_after_deriving_orientation():
    torch.manual_seed(3)
    count = 12
    positions = torch.randn(count, 3)
    rotation = torch.nn.functional.normalize(torch.randn(count, 4), dim=-1)
    model = _build_surfel_model(positions, rotation)
    xyz_before = model._xyz.detach().clone()
    rotation_before = model._rotation.detach().clone()
    scaling_before = model._scaling.detach().clone()

    derive_surface_orientation_from_surfel(model)

    assert torch.equal(model._xyz.detach(), xyz_before)
    assert torch.equal(model._rotation.detach(), rotation_before)
    assert torch.equal(model._scaling.detach(), scaling_before)


# --------------------------------------------------------------------------
# sign contract carries over unchanged
# --------------------------------------------------------------------------


def test_sign_flipped_normals_remain_compatible_through_unsigned_alignment():
    torch.manual_seed(4)
    count = 8
    positions = torch.randn(count, 3)
    rotation = torch.nn.functional.normalize(torch.randn(count, 4), dim=-1)
    model = _build_surfel_model(positions, rotation)
    orientation = derive_surface_orientation_from_surfel(model)

    flipped = -orientation.surface_normal
    assert torch.allclose(
        unsigned_normal_alignment(orientation.surface_normal, flipped), torch.ones(count), atol=1e-6
    )


# --------------------------------------------------------------------------
# end-to-end into the coverage-first partition
# --------------------------------------------------------------------------


def _flat_surfel_sheet(rows: int, columns: int, pitch: float, z: float = 0.0) -> torch.Tensor:
    u = torch.arange(rows, dtype=torch.float32) * pitch
    v = torch.arange(columns, dtype=torch.float32) * pitch
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    return torch.stack([uu.reshape(-1), vv.reshape(-1), torch.full_like(uu.reshape(-1), z)], dim=1)


def test_surfel_orientation_drives_the_coverage_first_partition_end_to_end():
    positions = torch.cat([_flat_surfel_sheet(12, 12, 0.1), _flat_surfel_sheet(12, 12, 0.1, z=5.0)], dim=0)
    count = int(positions.shape[0])
    model = _build_surfel_model(positions, _identity_quaternion(count))

    orientation = derive_surface_orientation_from_surfel(model)
    partition = partition_gaussian_subsets(orientation)
    accounting = partition_accounting(partition)

    assert accounting["subset_count"] == 2
    assert accounting["coverage_identity_holds"] is True
    assert accounting["spatially_disconnected_subset_count"] == 0
    assert accounting["multiply_owned_gaussian_count"] == 0


def test_partition_does_not_import_eigendecomposition_when_fed_surfel_orientation():
    """Static proof at the integration seam: nothing about running the
    coverage-first partition on surfel evidence pulls in the covariance
    eigen-decomposition path."""

    from osn_gs.surface import torch_surfel_surface_orientation

    source = inspect.getsource(torch_surfel_surface_orientation)
    assert "torch_gaussian_covariance_frame" not in source
