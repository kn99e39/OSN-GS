from types import SimpleNamespace

import numpy as np
import torch

from osn_gs.surface.torch_gaussian_region_owned_tsdf import (
    ABSTAIN_REPRESENTATIVE,
    MEMBERSHIP_CORE,
    MATERIALIZED_REPRESENTATIVE,
    EvidenceBoundedTSDFField,
    GaussianRegionMembership,
    RegionOwnedTSDFSupport,
    TSDFVisibleSurfaceSamples,
    associate_tsdf_samples_to_gaussians,
    assign_region_owned_tsdf_support,
    build_native_tsdf_support_components,
    derive_native_support_boundary,
    encode_cell_keys,
    extract_tsdf_zero_surface_samples,
    fit_boundary_first_region_representative,
)


def _partition(region_ids, roles=None, ambiguous=None):
    count = len(region_ids)
    return SimpleNamespace(
        subset_ids=torch.tensor(region_ids, dtype=torch.int64),
        partition_role=torch.tensor(roles or [0] * count, dtype=torch.int8),
        ambiguous_multi_region=torch.tensor(ambiguous or [False] * count, dtype=torch.bool),
        rejected_merge_mask=torch.zeros((0,), dtype=torch.bool),
    )


def _samples(cells, *, h=1.0, region_ids=None, statuses=None):
    cells = torch.tensor(sorted(cells, key=lambda cell: (cell[0], cell[1], cell[2])), dtype=torch.int64)
    keys = encode_cell_keys(cells)
    xyz = (cells.to(torch.float32) + 0.5) * h
    normals = torch.zeros_like(xyz)
    normals[:, 2] = 1.0
    n = len(cells)
    return TSDFVisibleSurfaceSamples(
        source_cell_keys=keys,
        cell_indices=cells,
        world_xyz=xyz,
        normals=normals,
        corner_values=torch.zeros((n, 8)),
        corner_support_count=torch.ones((n, 8), dtype=torch.int32),
        h=h,
    )


def test_zero_samples_are_directly_cell_owned_without_mesh():
    cells = torch.tensor([[0, 0, 0]], dtype=torch.int64)
    keys = encode_cell_keys(cells)
    values = torch.tensor([[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]])
    field = EvidenceBoundedTSDFField(
        keys=encode_cell_keys(torch.tensor([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])),
        value=torch.tensor([-1.0 if x == 0 else 1.0 for x in (0, 1) for y in (0, 1) for z in (0, 1)]),
        support_count=torch.ones((8,), dtype=torch.int32),
        h=1.0,
        mu=3.0,
    )
    samples = extract_tsdf_zero_surface_samples(field)
    assert samples.stats["mesh_intermediate"] is False
    assert samples.source_cell_keys.numel() == 1
    assert int(samples.source_cell_keys[0]) == int(keys[0])
    assert samples.corner_values.shape == (1, 8)
    assert torch.isfinite(samples.world_xyz).all()


def test_nearest_association_tie_is_stable_id_ordered_and_has_no_rejection():
    result = associate_tsdf_samples_to_gaussians(
        torch.tensor([[0.0, 0.0, 0.0], [-100.0, 0.0, 0.0]]),
        torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        torch.tensor([10, 2]),
    )
    assert result.nearest_gaussian_id.tolist() == [2, 10]
    assert result.stats["rejection_radius"] is None
    tree_result = associate_tsdf_samples_to_gaussians(
        torch.tensor([[0.0, 0.0, 0.0]]),
        torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        torch.tensor([10, 2]),
        torch_pair_limit=0,
    )
    assert tree_result.nearest_gaussian_id.tolist() == [2]


def test_region_owned_support_preserves_ambiguous_and_unassigned_states():
    association = associate_tsdf_samples_to_gaussians(
        torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        torch.tensor([100, 101, 102]),
    )
    membership = GaussianRegionMembership(
        region_ids=torch.tensor([3, 4, 5]),
        status=(MEMBERSHIP_CORE, "ambiguous", "unassigned"),
        accepted_mask=torch.tensor([True, False, False]),
        accounting={},
    )
    support = assign_region_owned_tsdf_support(association, membership)
    assert support.owned_region_id.tolist() == [3, -1, -1]
    assert support.membership_status == (MEMBERSHIP_CORE, "UNOWNED_TSDF_SUPPORT", "UNOWNED_TSDF_SUPPORT")


def test_native_cell_adjacency_keeps_disconnected_islands_separate():
    samples = _samples([[0, 0, 0], [1, 0, 0], [10, 0, 0], [11, 0, 0]])
    support = RegionOwnedTSDFSupport(
        nearest_region_id=torch.zeros((4,), dtype=torch.int64),
        nearest_membership_status=(MEMBERSHIP_CORE,) * 4,
        owned_region_id=torch.zeros((4,), dtype=torch.int64),
        membership_status=(MEMBERSHIP_CORE,) * 4,
        accepted_mask=torch.ones((4,), dtype=torch.bool),
        accounting={},
    )
    components, component_ids = build_native_tsdf_support_components(samples, support)
    assert len(components) == 2
    assert component_ids.tolist() == [0, 0, 1, 1]
    assert components[0].max_cell == (1, 0, 0)
    assert components[1].min_cell == (10, 0, 0)


def test_boundary_first_fit_materializes_valid_component_and_abstains_small_component():
    cells = [[x, y, 0] for y in range(9) for x in range(9)]
    samples = _samples(cells)
    support = RegionOwnedTSDFSupport(
        nearest_region_id=torch.zeros((len(cells),), dtype=torch.int64),
        nearest_membership_status=(MEMBERSHIP_CORE,) * len(cells),
        owned_region_id=torch.zeros((len(cells),), dtype=torch.int64),
        membership_status=(MEMBERSHIP_CORE,) * len(cells),
        accepted_mask=torch.ones((len(cells),), dtype=torch.bool),
        accounting={},
    )
    components, _ = build_native_tsdf_support_components(samples, support)
    assert len(components) == 1
    boundary = derive_native_support_boundary(samples, components[0])
    assert boundary.provenance["source"] == "native_tsdf_cell_adjacency"
    fit = fit_boundary_first_region_representative(samples, components[0], boundary)
    assert fit.status == MATERIALIZED_REPRESENTATIVE, (fit.reason, fit.tsdf_to_representative, boundary.provenance)
    assert fit.tsdf_to_representative["count"] == len(cells)

    small = _samples([[0, 0, 0]])
    small_support = RegionOwnedTSDFSupport(
        nearest_region_id=torch.zeros((1,), dtype=torch.int64),
        nearest_membership_status=(MEMBERSHIP_CORE,),
        owned_region_id=torch.zeros((1,), dtype=torch.int64),
        membership_status=(MEMBERSHIP_CORE,),
        accepted_mask=torch.ones((1,), dtype=torch.bool),
        accounting={},
    )
    small_components, _ = build_native_tsdf_support_components(small, small_support)
    small_boundary = derive_native_support_boundary(small, small_components[0])
    small_fit = fit_boundary_first_region_representative(small, small_components[0], small_boundary)
    assert small_fit.status == ABSTAIN_REPRESENTATIVE
