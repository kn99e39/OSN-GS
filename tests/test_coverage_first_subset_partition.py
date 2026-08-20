from __future__ import annotations

import ast
import inspect

import torch

from osn_gs.surface.torch_coverage_first_subset_partition import (
    OWNERSHIP_FALLBACK_NO_SPATIAL_NEIGHBOR,
    OWNERSHIP_FALLBACK_NORMAL_INCOMPATIBLE,
    OWNERSHIP_KINDS,
    OWNERSHIP_NORMAL_COHERENT,
    CoverageFirstPartitionConfig,
    count_spatially_disconnected_subsets,
    partition_accounting,
    partition_gaussian_subsets,
)
from osn_gs.surface.torch_gaussian_surface_orientation import (
    GaussianSurfaceOrientation,
    derive_surface_orientation_from_scale_rotation,
)


def _grid(rows: int, columns: int, pitch: float) -> torch.Tensor:
    u = torch.arange(rows, dtype=torch.float32) * pitch
    v = torch.arange(columns, dtype=torch.float32) * pitch
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    return torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)


def _planar_orientation(positions: torch.Tensor, normals: torch.Tensor) -> GaussianSurfaceOrientation:
    """Hand-built orientation so a test can control the normal field EXACTLY
    (including sign), independently of any covariance canonicalization."""

    normals = torch.nn.functional.normalize(normals, dim=-1)
    reference = torch.zeros_like(normals)
    reference.scatter_(-1, normals.abs().argmin(dim=-1, keepdim=True), 1.0)
    tangent_u = torch.nn.functional.normalize(
        reference - (reference * normals).sum(dim=-1, keepdim=True) * normals, dim=-1
    )
    count = int(positions.shape[0])
    return GaussianSurfaceOrientation(
        gaussian_ids=torch.arange(count),
        positions=positions,
        tangent_axis_u=tangent_u,
        tangent_axis_v=torch.cross(normals, tangent_u, dim=-1),
        surface_normal=normals,
        eigenvalues=torch.tensor([[0.09, 0.04, 0.0001]]).repeat(count, 1),
        tangent_major_scale=torch.full((count,), 0.3),
        tangent_minor_scale=torch.full((count,), 0.2),
        normal_thickness=torch.full((count,), 0.01),
        axis_separability=torch.zeros((count,), dtype=torch.int8),
        source="test_fixture",
    )


def _flat_sheet(rows: int = 12, columns: int = 12, pitch: float = 0.1, z: float = 0.0) -> torch.Tensor:
    plane = _grid(rows, columns, pitch)
    return torch.cat([plane, torch.full((plane.shape[0], 1), z)], dim=1)


# --------------------------------------------------------------------------
# Coverage contract
# --------------------------------------------------------------------------


def test_every_gaussian_receives_exactly_one_subset_owner():
    torch.manual_seed(1)
    positions = torch.randn(500, 3)
    normals = torch.nn.functional.normalize(torch.randn(500, 3), dim=-1)
    partition = partition_gaussian_subsets(_planar_orientation(positions, normals))

    assert int(partition.subset_ids.shape[0]) == 500
    assert int(partition.subset_ids.min()) >= 0
    assert int(partition.subset_ids.max()) < partition.subset_count
    occupancy = torch.bincount(partition.subset_ids, minlength=partition.subset_count)
    assert torch.equal(occupancy, partition.subset_sizes)
    assert int(partition.subset_sizes.sum()) == 500


def test_accounting_reports_no_dropped_or_multiply_owned_gaussian():
    positions = _flat_sheet()
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(positions.shape[0], 1)
    accounting = partition_accounting(partition_gaussian_subsets(_planar_orientation(positions, normals)))

    assert accounting["input_gaussian_count"] == int(positions.shape[0])
    assert accounting["assigned_gaussian_count"] == accounting["input_gaussian_count"]
    assert accounting["unassigned_gaussian_count"] == 0
    assert accounting["multiply_owned_gaussian_count"] == 0
    assert accounting["subset_size_sum"] == accounting["input_gaussian_count"]
    assert accounting["subset_sizes_match_ownership_map"] is True
    assert accounting["coverage_identity_holds"] is True
    assert sum(accounting["ownership_kind_counts"].values()) == accounting["input_gaussian_count"]


def test_isolated_gaussian_keeps_ownership_through_the_reported_fallback():
    """No UNASSIGNED population: a Gaussian nothing can attach to still owns a
    subset, and the fallback is named in the accounting."""

    sheet = _flat_sheet()
    far_away = torch.tensor([[50.0, 50.0, 50.0]])
    positions = torch.cat([sheet, far_away], dim=0)
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(positions.shape[0], 1)
    partition = partition_gaussian_subsets(_planar_orientation(positions, normals))
    accounting = partition_accounting(partition)

    outlier = int(partition.subset_ids[-1])
    assert int((partition.subset_ids == outlier).sum()) == 1
    assert OWNERSHIP_KINDS[int(partition.ownership_kind[-1])] == OWNERSHIP_FALLBACK_NO_SPATIAL_NEIGHBOR
    assert accounting["fallback_ownership_count"] == 1
    assert accounting["unassigned_gaussian_count"] == 0
    assert accounting["coverage_identity_holds"] is True


def test_gaussian_with_only_normal_incompatible_neighbours_keeps_ownership():
    """Low normal agreement must not revoke subset ownership -- SUBSET
    OWNERSHIP != TRUSTABILITY."""

    sheet = _flat_sheet(rows=8, columns=8, pitch=0.1)
    intruder = torch.tensor([[0.35, 0.35, 0.0]])
    positions = torch.cat([sheet, intruder], dim=0)
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(positions.shape[0], 1)
    normals[-1] = torch.tensor([1.0, 0.0, 0.0])  # orthogonal to every neighbour
    partition = partition_gaussian_subsets(_planar_orientation(positions, normals))
    accounting = partition_accounting(partition)

    assert OWNERSHIP_KINDS[int(partition.ownership_kind[-1])] == OWNERSHIP_FALLBACK_NORMAL_INCOMPATIBLE
    assert int((partition.subset_ids == int(partition.subset_ids[-1])).sum()) == 1
    assert accounting["ownership_kind_counts"][OWNERSHIP_FALLBACK_NORMAL_INCOMPATIBLE] == 1
    assert accounting["coverage_identity_holds"] is True
    assert accounting["normal_compatibility_cut_edge_count"] > 0


# --------------------------------------------------------------------------
# Partition semantics
# --------------------------------------------------------------------------


def test_sign_flipped_equivalent_normals_do_not_split_one_surface():
    """n and -n are the same local surface orientation; alternating the stored
    sign across one flat sheet must leave a single subset."""

    positions = _flat_sheet()
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(positions.shape[0], 1)
    flipped = normals.clone()
    flipped[::2] *= -1.0

    consistent = partition_gaussian_subsets(_planar_orientation(positions, normals))
    alternating = partition_gaussian_subsets(_planar_orientation(positions, flipped))

    assert consistent.subset_count == 1
    assert alternating.subset_count == 1
    assert torch.equal(consistent.subset_ids, alternating.subset_ids)


def test_distant_parallel_surfaces_are_not_merged():
    """Global normal clustering would fuse these two walls; local spatial
    connectivity must keep them apart."""

    near = _flat_sheet(z=0.0)
    far = _flat_sheet(z=5.0)
    positions = torch.cat([near, far], dim=0)
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(positions.shape[0], 1)
    partition = partition_gaussian_subsets(_planar_orientation(positions, normals))

    assert partition.subset_count == 2
    assert int(partition.subset_ids[: near.shape[0]].unique().numel()) == 1
    assert int(partition.subset_ids[near.shape[0] :].unique().numel()) == 1
    assert int(partition.subset_ids[0]) != int(partition.subset_ids[-1])


def test_strong_local_normal_discontinuity_creates_a_partition_boundary():
    """A right-angle crease: both faces are spatially adjacent along the seam,
    so only the normal test can separate them."""

    pitch = 0.1
    plane = _grid(10, 10, pitch)
    floor = torch.cat([plane, torch.zeros((plane.shape[0], 1))], dim=1)
    wall = torch.cat([torch.zeros((plane.shape[0], 1)), plane[:, :1], plane[:, 1:] + pitch], dim=1)
    positions = torch.cat([floor, wall], dim=0)
    normals = torch.cat(
        [
            torch.tensor([[0.0, 0.0, 1.0]]).repeat(floor.shape[0], 1),
            torch.tensor([[1.0, 0.0, 0.0]]).repeat(wall.shape[0], 1),
        ],
        dim=0,
    )
    partition = partition_gaussian_subsets(_planar_orientation(positions, normals))
    accounting = partition_accounting(partition)

    assert int(partition.subset_ids[: floor.shape[0]].unique().numel()) == 1
    assert int(partition.subset_ids[floor.shape[0] :].unique().numel()) == 1
    assert int(partition.subset_ids[0]) != int(partition.subset_ids[-1])
    # The seam is genuinely adjacent, so the split came from normals, not distance.
    assert accounting["normal_compatibility_cut_edge_count"] > 0
    assert int(partition.normal_cut_edges.shape[0]) == accounting["normal_compatibility_cut_edge_count"]


def test_every_subset_is_connected_under_the_partition_graph():
    torch.manual_seed(7)
    positions = torch.cat([_flat_sheet(), _flat_sheet(z=5.0), torch.randn(60, 3) * 4.0], dim=0)
    normals = torch.nn.functional.normalize(torch.randn(positions.shape[0], 3), dim=-1)
    normals[: 288] = torch.tensor([0.0, 0.0, 1.0])
    partition = partition_gaussian_subsets(_planar_orientation(positions, normals))

    assert count_spatially_disconnected_subsets(partition) == 0
    assert partition_accounting(partition)["spatially_disconnected_subset_count"] == 0


def test_long_thin_surface_resolves_to_one_subset():
    """Regression: a first implementation propagated component labels one graph
    hop per round, so a long strip needed O(diameter) rounds and blew the
    solver budget on the real scene. A single connected strip must come back as
    exactly one subset, whatever its diameter."""

    from osn_gs.surface.torch_coverage_first_subset_partition import _connected_component_roots

    positions = _flat_sheet(rows=400, columns=3, pitch=0.1)
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(positions.shape[0], 1)
    partition = partition_gaussian_subsets(_planar_orientation(positions, normals))
    assert partition.subset_count == 1
    assert count_spatially_disconnected_subsets(partition) == 0

    # Direct solver check on a chain whose node indices are shuffled, so the
    # label minimum has to travel the whole graph rather than one end to the other.
    torch.manual_seed(4)
    node_count = 5000
    order = torch.randperm(node_count)
    chain = torch.stack([order[:-1], order[1:]], dim=1)
    chain = torch.stack([chain.min(dim=1).values, chain.max(dim=1).values], dim=1)
    roots = _connected_component_roots(node_count, chain, CoverageFirstPartitionConfig())
    assert int(roots.unique().numel()) == 1


def test_subset_ids_are_deterministic_and_size_ordered():
    torch.manual_seed(2)
    positions = torch.cat([_flat_sheet(rows=14, columns=14), _flat_sheet(rows=5, columns=5, z=5.0)], dim=0)
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(positions.shape[0], 1)
    orientation = _planar_orientation(positions, normals)

    first = partition_gaussian_subsets(orientation)
    second = partition_gaussian_subsets(orientation)
    assert torch.equal(first.subset_ids, second.subset_ids)
    assert torch.equal(first.subset_sizes, second.subset_sizes)
    # Subset 0 is always the largest; sizes descend.
    assert torch.all(first.subset_sizes[:-1] >= first.subset_sizes[1:])
    assert int(first.subset_sizes[0]) == 14 * 14


def test_partition_does_not_mutate_the_source_tensors():
    positions = _flat_sheet()
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(positions.shape[0], 1)
    orientation = _planar_orientation(positions, normals)
    position_snapshot = orientation.positions.clone()
    normal_snapshot = orientation.surface_normal.clone()
    id_snapshot = orientation.gaussian_ids.clone()

    partition = partition_gaussian_subsets(orientation)

    assert torch.equal(orientation.positions, position_snapshot)
    assert torch.equal(orientation.surface_normal, normal_snapshot)
    assert torch.equal(orientation.gaussian_ids, id_snapshot)
    assert torch.equal(partition.gaussian_ids, id_snapshot)


def test_derived_orientation_feeds_the_partition_end_to_end():
    """The production entry point (scale/rotation) must drive the partition
    without any intermediate hand-built fixture."""

    positions = torch.cat([_flat_sheet(), _flat_sheet(z=5.0)], dim=0)
    count = int(positions.shape[0])
    scaling = torch.tensor([[0.05, 0.04, 0.001]]).repeat(count, 1)
    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1)
    orientation = derive_surface_orientation_from_scale_rotation(positions, scaling, rotation)
    accounting = partition_accounting(partition_gaussian_subsets(orientation))

    assert accounting["subset_count"] == 2
    assert accounting["coverage_identity_holds"] is True
    assert accounting["spatially_disconnected_subset_count"] == 0


def test_subset_size_histogram_accounts_for_every_subset_and_gaussian():
    positions = torch.cat([_flat_sheet(rows=14, columns=14), _flat_sheet(rows=3, columns=3, z=5.0)], dim=0)
    positions = torch.cat([positions, torch.tensor([[40.0, 40.0, 40.0]])], dim=0)
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(positions.shape[0], 1)
    accounting = partition_accounting(partition_gaussian_subsets(_planar_orientation(positions, normals)))

    histogram = accounting["subset_size_histogram"]
    assert sum(bucket["subset_count"] for bucket in histogram) == accounting["subset_count"]
    assert sum(bucket["gaussian_count"] for bucket in histogram) == accounting["input_gaussian_count"]
    assert accounting["largest_subset_size"] == 14 * 14
    assert abs(accounting["largest_subset_gaussian_fraction"] - (14 * 14) / int(positions.shape[0])) < 1e-9


def test_partition_parameters_are_reported_and_centralized():
    config = CoverageFirstPartitionConfig()
    payload = config.payload()
    assert payload["neighbor_count"] == config.neighbor_count
    assert payload["spatial_connect_spacing_multiplier"] == config.spatial_connect_spacing_multiplier
    assert payload["normal_compatibility_min_alignment"] == config.normal_compatibility_min_alignment
    assert abs(payload["normal_compatibility_angle_degrees"] - 31.788) < 0.01

    positions = _flat_sheet()
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(positions.shape[0], 1)
    accounting = partition_accounting(partition_gaussian_subsets(_planar_orientation(positions, normals), config))
    assert accounting["partition_parameters"] == payload


def test_empty_and_single_gaussian_inputs_stay_coverage_exact():
    empty = _planar_orientation(torch.zeros((0, 3)), torch.zeros((0, 3)))
    empty_partition = partition_gaussian_subsets(empty)
    assert empty_partition.subset_count == 0
    assert partition_accounting(empty_partition)["coverage_identity_holds"] is True

    single = _planar_orientation(torch.zeros((1, 3)), torch.tensor([[0.0, 0.0, 1.0]]))
    single_partition = partition_gaussian_subsets(single)
    assert single_partition.subset_count == 1
    assert OWNERSHIP_KINDS[int(single_partition.ownership_kind[0])] == OWNERSHIP_FALLBACK_NO_SPATIAL_NEIGHBOR
    assert partition_accounting(single_partition)["coverage_identity_holds"] is True


# --------------------------------------------------------------------------
# Architectural isolation from the Worklog 95-104 selection-first pipeline
# --------------------------------------------------------------------------


def test_partition_does_not_depend_on_latent_support_chart_or_nurbs_validity():
    """Static proof that the coverage-first partition is upstream of every
    Worklog 95-104 acceptance stage."""

    from osn_gs.surface import torch_coverage_first_subset_partition, torch_gaussian_surface_orientation

    forbidden = (
        "latent_surface",
        "intrinsic_chart_atlas",
        "patch_identifiability",
        "adaptive_patch_capacity",
        "parametric_domain_validity",
        "torch_nurbs",
        "curve_network",
        "curve_lattice",
        "boundary",
        "held_out",
    )
    for module in (torch_coverage_first_subset_partition, torch_gaussian_surface_orientation):
        tree = ast.parse(inspect.getsource(module))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
                imported.extend(f"{node.module}.{alias.name}" for alias in node.names)
        for name in imported:
            for banned in forbidden:
                assert banned not in name, f"{module.__name__} must not import {name}"


def test_ownership_kinds_cover_every_gaussian_exactly_once():
    assert OWNERSHIP_KINDS == (
        OWNERSHIP_NORMAL_COHERENT,
        OWNERSHIP_FALLBACK_NORMAL_INCOMPATIBLE,
        OWNERSHIP_FALLBACK_NO_SPATIAL_NEIGHBOR,
    )
    torch.manual_seed(9)
    positions = torch.randn(300, 3) * 2.0
    normals = torch.nn.functional.normalize(torch.randn(300, 3), dim=-1)
    partition = partition_gaussian_subsets(_planar_orientation(positions, normals))
    counts = partition_accounting(partition)["ownership_kind_counts"]
    assert sum(counts.values()) == 300
    assert set(counts) == set(OWNERSHIP_KINDS)
