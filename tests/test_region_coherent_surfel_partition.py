from __future__ import annotations

import ast
import inspect
import math

import torch

from osn_gs.surface.torch_coverage_first_subset_partition import (
    CoverageFirstPartitionConfig,
    partition_accounting,
    partition_gaussian_subsets,
)
from osn_gs.surface.torch_region_coherent_surfel_partition import (
    PARTITION_ROLES,
    ROLE_ISOLATED_FALLBACK,
    ROLE_OWNERSHIP_PROPAGATED,
    ROLE_STRUCTURAL_CORE,
    RegionCoherenceConfig,
    _lambda_max_over_trace_batch,
    count_spatially_disconnected_structural_regions,
    partition_surfels_region_coherent,
    region_coherent_accounting,
)


class _Orientation:
    def __init__(self, positions: torch.Tensor, normals: torch.Tensor):
        self.positions = positions
        self.surface_normal = normals
        self.gaussian_ids = torch.arange(int(positions.shape[0]))


def _flat_sheet(rows: int = 12, columns: int = 12, pitch: float = 0.1, z: float = 0.0) -> torch.Tensor:
    u = torch.arange(rows, dtype=torch.float32) * pitch
    v = torch.arange(columns, dtype=torch.float32) * pitch
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    return torch.stack([uu.reshape(-1), vv.reshape(-1), torch.full((rows * columns,), z)], dim=1)


def _rotation_chain(n: int, angle_step_rad: float, pitch: float = 0.1) -> _Orientation:
    """A line of surfels each locally normal-compatible with its immediate
    neighbour (angle_step_rad well under the local threshold's ~31.79 degrees)
    but accumulating unbounded orientation drift end to end."""

    positions = torch.stack([torch.arange(n, dtype=torch.float32) * pitch, torch.zeros(n), torch.zeros(n)], dim=1)
    normals = torch.stack(
        [torch.tensor([math.sin(i * angle_step_rad), 0.0, math.cos(i * angle_step_rad)]) for i in range(n)]
    )
    return _Orientation(positions, normals)


# --------------------------------------------------------------------------
# concentration-floor derivation
# --------------------------------------------------------------------------


def test_concentration_floor_matches_closed_form_two_normal_formula():
    config = RegionCoherenceConfig(local=CoverageFirstPartitionConfig(normal_compatibility_min_alignment=0.85))
    assert abs(config.concentration_floor() - 0.925) < 1e-9

    # Direct construction: two unit normals at exactly the alignment floor.
    a = 0.85
    theta = math.acos(a)
    n1 = torch.tensor([1.0, 0.0, 0.0])
    n2 = torch.tensor([math.cos(theta), math.sin(theta), 0.0])
    scatter = (n1.outer(n1) + n2.outer(n2)).unsqueeze(0)
    concentration = float(_lambda_max_over_trace_batch(scatter)[0])
    assert abs(concentration - config.concentration_floor()) < 1e-5


def test_lambda_max_over_trace_matches_numeric_eigensolver():
    torch.manual_seed(0)
    vectors = torch.randn(6, 4, 3)  # 6 batches of 4 vectors each in R^3
    scatter = torch.einsum("bik,bil->bkl", vectors, vectors)  # sum_i v_i v_i^T, (6, 3, 3)
    closed_form = _lambda_max_over_trace_batch(scatter)
    eigenvalues = torch.linalg.eigvalsh(scatter)
    expected = eigenvalues.max(dim=-1).values / eigenvalues.sum(dim=-1)
    assert torch.allclose(closed_form, expected, atol=1e-4)


# --------------------------------------------------------------------------
# anti-chaining core behaviour
# --------------------------------------------------------------------------


def test_accumulated_drift_chain_does_not_become_one_region():
    """Each pairwise step passes the LOCAL 0.85 threshold (15 degrees << 31.79
    degrees), but the chain sweeps 270 degrees end to end -- Worklog 96's
    plain connected-component partition keeps this as ONE subset (reproduced
    here as the negative control); region coherence must not."""

    orientation = _rotation_chain(n=19, angle_step_rad=math.radians(15.0))

    plain_cc = partition_accounting(partition_gaussian_subsets(orientation, CoverageFirstPartitionConfig()))
    assert plain_cc["subset_count"] == 1  # negative control: WL96 mechanics alone DO chain this

    region = region_coherent_accounting(partition_surfels_region_coherent(orientation))
    assert region["subset_count"] > 1
    assert region["region_coherence_rejected_merge_count"] > 0
    assert region["coverage_identity_holds"] is True


def test_bounded_smooth_curvature_sheet_remains_one_region():
    """A gently curved sheet (small per-step angle, bounded total sweep) is
    valid single-patch evidence and must NOT be shattered merely for curving."""

    orientation = _rotation_chain(n=8, angle_step_rad=math.radians(3.0))  # ~21 degrees total sweep
    region = region_coherent_accounting(partition_surfels_region_coherent(orientation))
    assert region["subset_count"] == 1
    assert region["region_coherence_rejected_merge_count"] == 0


def test_flat_sheet_is_one_fully_structural_region():
    orientation = _Orientation(_flat_sheet(), torch.tensor([[0.0, 0.0, 1.0]]).repeat(144, 1))
    partition = partition_surfels_region_coherent(orientation)
    accounting = region_coherent_accounting(partition)
    assert accounting["subset_count"] == 1
    assert accounting["partition_role_counts"][ROLE_STRUCTURAL_CORE] == 144
    assert accounting["partition_role_counts"][ROLE_OWNERSHIP_PROPAGATED] == 0
    assert accounting["partition_role_counts"][ROLE_ISOLATED_FALLBACK] == 0
    assert accounting["region_orientation"]["concentration_median"] > 0.99


# --------------------------------------------------------------------------
# sign invariance
# --------------------------------------------------------------------------


def test_sign_flipped_normals_produce_identical_partition():
    orientation = _Orientation(_flat_sheet(), torch.tensor([[0.0, 0.0, 1.0]]).repeat(144, 1))
    flipped_normals = orientation.surface_normal.clone()
    flipped_normals[::3] *= -1.0
    flipped = _Orientation(orientation.positions, flipped_normals)

    a = partition_surfels_region_coherent(orientation)
    b = partition_surfels_region_coherent(flipped)
    assert torch.equal(a.subset_ids, b.subset_ids)
    assert torch.allclose(a.region_concentration, b.region_concentration, atol=1e-5)


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_partition_is_deterministic_across_repeated_runs():
    orientation = _rotation_chain(n=19, angle_step_rad=math.radians(15.0))
    first = partition_surfels_region_coherent(orientation)
    second = partition_surfels_region_coherent(orientation)
    assert torch.equal(first.subset_ids, second.subset_ids)
    assert torch.equal(first.partition_role, second.partition_role)
    assert torch.equal(first.rejected_merge_mask, second.rejected_merge_mask)


# --------------------------------------------------------------------------
# anti-chaining bridge / propagation containment
# --------------------------------------------------------------------------


def _two_regions_with_bridge() -> _Orientation:
    """Two flat, mutually INCOMPATIBLE (perpendicular) sheets placed close
    enough that a local candidate edge exists directly between them (so the
    region-coherence test, not spatial distance, is what has to reject the
    merge), plus one extra surfel whose own normal is compatible with BOTH
    sheets locally (a 45-degree bridge) but which must not fuse them."""

    sheet_a = _flat_sheet(rows=6, columns=6, pitch=0.1, z=0.0)
    normals_a = torch.tensor([[0.0, 0.0, 1.0]]).repeat(sheet_a.shape[0], 1)

    sheet_b = _flat_sheet(rows=6, columns=6, pitch=0.1, z=0.0)
    sheet_b[:, 0] += 0.55  # spatially adjacent to sheet_a's right edge
    normals_b = torch.tensor([[1.0, 0.0, 0.0]]).repeat(sheet_b.shape[0], 1)  # perpendicular to sheet_a

    bridge_position = torch.tensor([[0.525, 0.25, 0.0]])
    bridge_normal = torch.nn.functional.normalize(torch.tensor([[0.5, 0.0, 0.5]]), dim=-1)

    positions = torch.cat([sheet_a, sheet_b, bridge_position], dim=0)
    normals = torch.cat([normals_a, normals_b, bridge_normal], dim=0)
    return _Orientation(positions, normals)


def test_rejected_region_merge_is_not_reconnected_through_ownership_only_surfel():
    orientation = _two_regions_with_bridge()
    accounting = region_coherent_accounting(partition_surfels_region_coherent(orientation))
    # The two perpendicular sheets must remain in DIFFERENT final subsets --
    # a bridge surfel locally compatible with both sides must not fuse them.
    assert accounting["subset_count"] >= 2
    assert accounting["coverage_identity_holds"] is True


def test_ownership_propagated_surfel_cannot_merge_two_structural_regions():
    orientation = _two_regions_with_bridge()
    partition = partition_surfels_region_coherent(orientation)
    core_mask = partition.partition_role == PARTITION_ROLES.index(ROLE_STRUCTURAL_CORE)
    core_subset_ids = torch.unique(partition.subset_ids[core_mask])
    # At least two DISTINCT structural regions exist among the core members
    # (the two perpendicular sheets), regardless of where the bridge landed.
    assert int(core_subset_ids.numel()) >= 2

    propagated_mask = partition.partition_role == PARTITION_ROLES.index(ROLE_OWNERSHIP_PROPAGATED)
    if int(propagated_mask.sum()) > 0:
        # A propagated surfel's subset id must be ONE of the existing
        # structural region ids -- it never creates a third, merged id.
        propagated_ids = torch.unique(partition.subset_ids[propagated_mask])
        assert bool(torch.isin(propagated_ids, core_subset_ids).all())


# --------------------------------------------------------------------------
# coverage contract
# --------------------------------------------------------------------------


def test_every_surfel_receives_exactly_one_owner_and_none_is_dropped():
    torch.manual_seed(3)
    positions = torch.cat([_flat_sheet(), _flat_sheet(z=5.0), torch.randn(40, 3) * 3.0], dim=0)
    normals = torch.nn.functional.normalize(torch.randn(positions.shape[0], 3), dim=-1)
    normals[:144] = torch.tensor([0.0, 0.0, 1.0])
    orientation = _Orientation(positions, normals)

    accounting = region_coherent_accounting(partition_surfels_region_coherent(orientation))
    assert accounting["input_surfel_count"] == int(positions.shape[0])
    assert accounting["assigned_surfel_count"] == accounting["input_surfel_count"]
    assert accounting["unassigned_surfel_count"] == 0
    assert accounting["multiply_owned_surfel_count"] == 0
    assert accounting["coverage_identity_holds"] is True
    role_total = sum(accounting["partition_role_counts"].values())
    assert role_total == accounting["input_surfel_count"]


def test_structural_regions_remain_spatially_connected():
    torch.manual_seed(4)
    positions = torch.cat([_flat_sheet(), _flat_sheet(z=5.0)], dim=0)
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(positions.shape[0], 1)
    orientation = _Orientation(positions, normals)

    partition = partition_surfels_region_coherent(orientation)
    assert count_spatially_disconnected_structural_regions(partition) == 0


def test_original_tensors_are_unchanged_after_partitioning():
    positions = _flat_sheet()
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(positions.shape[0], 1)
    orientation = _Orientation(positions, normals)
    position_snapshot = positions.clone()
    normal_snapshot = normals.clone()

    partition_surfels_region_coherent(orientation)

    assert torch.equal(orientation.positions, position_snapshot)
    assert torch.equal(orientation.surface_normal, normal_snapshot)


def test_empty_and_singleton_input_stay_coverage_exact():
    empty = _Orientation(torch.zeros((0, 3)), torch.zeros((0, 3)))
    empty_accounting = region_coherent_accounting(partition_surfels_region_coherent(empty))
    assert empty_accounting["coverage_identity_holds"] is True

    single = _Orientation(torch.zeros((1, 3)), torch.tensor([[0.0, 0.0, 1.0]]))
    single_partition = partition_surfels_region_coherent(single)
    single_accounting = region_coherent_accounting(single_partition)
    assert single_accounting["subset_count"] == 1
    assert single_accounting["coverage_identity_holds"] is True
    assert PARTITION_ROLES[int(single_partition.partition_role[0])] == ROLE_ISOLATED_FALLBACK


# --------------------------------------------------------------------------
# architectural isolation: no per-primitive normal derivation in this module
# --------------------------------------------------------------------------


def test_module_does_not_derive_per_surfel_normals():
    """This module's own eigenvalue math is for REGION-scatter diagnostics
    only -- it must never import a per-primitive normal-deriving function
    (covariance eigendecomposition or otherwise); orientation always arrives
    from the caller already computed."""

    from osn_gs.surface import torch_region_coherent_surfel_partition

    tree = ast.parse(inspect.getsource(torch_region_coherent_surfel_partition))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
            imported.extend(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    forbidden = ("torch_gaussian_surface_orientation.derive", "torch_surfel_surface_orientation.derive", "covariance_frame")
    for name in imported:
        for banned in forbidden:
            assert banned not in name, f"must not import {name}"


def test_only_one_new_free_parameter_is_introduced():
    """structural_weight is the only new independent field; the coherence
    floor is a pure function of the existing local alignment threshold."""

    default = RegionCoherenceConfig()
    fields = {f for f in default.__dataclass_fields__ if f != "local"}
    assert fields == {"structural_weight"}
    assert default.structural_weight == 1.0
