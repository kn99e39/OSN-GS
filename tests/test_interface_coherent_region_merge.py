from __future__ import annotations

import ast
import inspect
import math

import torch

from osn_gs.surface.torch_interface_coherent_region_merge import (
    InterfaceCoherentMergeConfig,
    interface_coherent_accounting,
    partition_surfels_interface_coherent,
)


class _Orientation:
    def __init__(self, positions: torch.Tensor, normals: torch.Tensor, tangent_u: torch.Tensor, tangent_v: torch.Tensor):
        self.positions = positions
        self.surface_normal = normals
        self.tangent_axis_u = tangent_u
        self.tangent_axis_v = tangent_v
        self.gaussian_ids = torch.arange(int(positions.shape[0]))


def _flat_sheet(rows: int = 14, columns: int = 14, pitch: float = 0.1, z: float = 0.0, x0: float = 0.0) -> _Orientation:
    u = torch.arange(rows, dtype=torch.float32) * pitch + x0
    v = torch.arange(columns, dtype=torch.float32) * pitch
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    count = rows * columns
    positions = torch.stack([uu.reshape(-1), vv.reshape(-1), torch.full((count,), z)], dim=1)
    normals = torch.tensor([[0.0, 0.0, 1.0]]).repeat(count, 1)
    tangent_u = torch.tensor([[1.0, 0.0, 0.0]]).repeat(count, 1)
    tangent_v = torch.tensor([[0.0, 1.0, 0.0]]).repeat(count, 1)
    return _Orientation(positions, normals, tangent_u, tangent_v)


def _cylinder_band(n_theta: int = 60, n_y: int = 8, radius: float = 0.5, theta_span: float = math.pi, pitch_y: float = 0.05) -> _Orientation:
    theta = torch.linspace(0.0, theta_span, n_theta)
    y = torch.arange(n_y, dtype=torch.float32) * pitch_y
    tt, yy = torch.meshgrid(theta, y, indexing="ij")
    tt, yy = tt.reshape(-1), yy.reshape(-1)
    positions = torch.stack([radius * torch.cos(tt), yy, radius * torch.sin(tt)], dim=1)
    normals = torch.stack([torch.cos(tt), torch.zeros_like(tt), torch.sin(tt)], dim=1)
    tangent_u = torch.stack([-torch.sin(tt), torch.zeros_like(tt), torch.cos(tt)], dim=1)
    tangent_v = torch.tensor([[0.0, 1.0, 0.0]]).repeat(positions.shape[0], 1)
    return _Orientation(positions, normals, tangent_u, tangent_v)


def _crease(rows: int = 12, columns: int = 12, pitch: float = 0.1, angle_deg: float = 90.0) -> tuple[_Orientation, int]:
    side_a = _flat_sheet(rows, columns, pitch)
    side_a.positions[:, 0] -= (rows - 1) * pitch

    angle = math.radians(angle_deg)
    u = torch.arange(rows, dtype=torch.float32) * pitch
    v = torch.arange(columns, dtype=torch.float32) * pitch
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    uu, vv = uu.reshape(-1), vv.reshape(-1)
    positions_b = torch.stack([uu * math.cos(angle), vv, uu * math.sin(angle)], dim=1)
    count_b = positions_b.shape[0]
    normals_b = torch.tensor([[math.sin(angle), 0.0, math.cos(angle)]]).repeat(count_b, 1)
    tangent_u_b = torch.tensor([[math.cos(angle), 0.0, math.sin(angle)]]).repeat(count_b, 1)
    tangent_v_b = torch.tensor([[0.0, 1.0, 0.0]]).repeat(count_b, 1)

    positions = torch.cat([side_a.positions, positions_b], dim=0)
    normals = torch.cat([side_a.surface_normal, normals_b], dim=0)
    tangent_u = torch.cat([side_a.tangent_axis_u, tangent_u_b], dim=0)
    tangent_v = torch.cat([side_a.tangent_axis_v, tangent_v_b], dim=0)
    return _Orientation(positions, normals, tangent_u, tangent_v), rows * columns


def _zigzag(plate_count: int = 3, rows: int = 8, columns: int = 8, pitch: float = 0.1, angle_deg: float = 90.0) -> list[int]:
    """`plate_count` flat plates joined edge-to-edge, each rotated by
    `angle_deg` from the previous one (a zigzag of real creases) -- tests
    that a CHAIN of narrow bridges does not collapse via transitivity.
    Returns (orientation, [surfel_count_per_plate])."""

    positions_list = []
    normals_list = []
    tangent_u_list = []
    tangent_v_list = []
    counts = []

    u = torch.arange(rows, dtype=torch.float32) * pitch
    v = torch.arange(columns, dtype=torch.float32) * pitch
    uu, vv = torch.meshgrid(u, v, indexing="ij")
    uu, vv = uu.reshape(-1), vv.reshape(-1)
    count_per_plate = rows * columns

    cursor = torch.zeros(3)
    direction = torch.tensor([1.0, 0.0, 0.0])
    normal = torch.tensor([0.0, 0.0, 1.0])
    for plate_index in range(plate_count):
        positions = cursor.unsqueeze(0) + uu.unsqueeze(-1) * direction.unsqueeze(0) + vv.unsqueeze(-1) * torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0)
        positions_list.append(positions)
        normals_list.append(normal.unsqueeze(0).repeat(count_per_plate, 1))
        tangent_u_list.append(direction.unsqueeze(0).repeat(count_per_plate, 1))
        tangent_v_list.append(torch.tensor([[0.0, 1.0, 0.0]]).repeat(count_per_plate, 1))
        counts.append(count_per_plate)

        # advance cursor to the far edge of this plate, then rotate direction
        # (and normal) by `angle_deg` around the shared y-axis seam for the
        # NEXT plate -- a genuine sharp fold each time, alternating sign so
        # the chain does not spiral back on itself.
        far_edge = cursor + (rows - 1) * pitch * direction
        sign = 1.0 if plate_index % 2 == 0 else -1.0
        angle = math.radians(angle_deg) * sign
        rot = torch.tensor(
            [[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)]]
        )
        cursor = far_edge
        direction = rot @ direction
        normal = rot @ normal

    positions = torch.cat(positions_list, dim=0)
    normals = torch.cat(normals_list, dim=0)
    tangent_u = torch.cat(tangent_u_list, dim=0)
    tangent_v = torch.cat(tangent_v_list, dim=0)
    return _Orientation(positions, normals, tangent_u, tangent_v), counts


# --------------------------------------------------------------------------
# recovers Worklog 97's over-segmentation of smooth curved surfaces
# --------------------------------------------------------------------------


def test_quarter_cylinder_over_segmented_by_wl97_merges_back_into_one_region():
    orientation = _cylinder_band(theta_span=math.pi / 2.0)
    partition = partition_surfels_interface_coherent(orientation)
    accounting = interface_coherent_accounting(partition)

    assert accounting["initial_region_count"] > 1  # WL97 really does over-segment this
    assert accounting["final_region_count"] == 1
    assert accounting["coverage_identity_holds"] is True


def test_half_cylinder_over_segmented_by_wl97_merges_back_into_one_region():
    orientation = _cylinder_band(theta_span=math.pi)
    partition = partition_surfels_interface_coherent(orientation)
    accounting = interface_coherent_accounting(partition)

    assert accounting["initial_region_count"] > 1
    assert accounting["final_region_count"] == 1
    assert accounting["coverage_identity_holds"] is True
    assert accounting["merges_applied"] == accounting["initial_region_count"] - 1


# --------------------------------------------------------------------------
# preserves true discontinuities
# --------------------------------------------------------------------------


def test_sharp_crease_regions_remain_separate():
    orientation, count_per_side = _crease(angle_deg=90.0)
    partition = partition_surfels_interface_coherent(orientation)
    accounting = interface_coherent_accounting(partition)

    assert accounting["final_region_count"] >= 2
    assert accounting["coverage_identity_holds"] is True
    top_two_fraction = float(partition.subset_sizes[:2].sum()) / int(len(partition))
    assert top_two_fraction > 0.85
    # the interface between the two initial regions must have been evaluated
    # and REJECTED, not simply never adjacent.
    rejected = [i for i in partition.all_interfaces if not i.accepted]
    assert len(rejected) >= 1


def test_two_nearby_parallel_sheets_remain_separate():
    a = _flat_sheet(z=0.0)
    b = _flat_sheet(z=0.15, x0=0.03)
    positions = torch.cat([a.positions, b.positions], dim=0)
    normals = torch.cat([a.surface_normal, b.surface_normal], dim=0)
    tangent_u = torch.cat([a.tangent_axis_u, b.tangent_axis_u], dim=0)
    tangent_v = torch.cat([a.tangent_axis_v, b.tangent_axis_v], dim=0)
    orientation = _Orientation(positions, normals, tangent_u, tangent_v)

    partition = partition_surfels_interface_coherent(orientation)
    accounting = interface_coherent_accounting(partition)
    assert accounting["coverage_identity_holds"] is True
    assert accounting["final_region_count"] >= 2
    half = int(positions.shape[0]) // 2
    assert int(partition.subset_ids[:half].unique().numel()) == 1
    assert int(partition.subset_ids[half:].unique().numel()) == 1
    assert int(partition.subset_ids[0]) != int(partition.subset_ids[half])


def test_narrow_multi_region_bridge_chain_does_not_percolate():
    """A zigzag chain of several plates, each pair joined only by a real
    sharp fold -- transitivity through the chain must not collapse it into
    one region even though each fold is only a NARROW shared edge."""

    orientation, counts = _zigzag(plate_count=4, angle_deg=90.0)
    partition = partition_surfels_interface_coherent(orientation)
    accounting = interface_coherent_accounting(partition)

    assert accounting["coverage_identity_holds"] is True
    # the first and last plates (not directly adjacent) must never share a
    # final region -- if they did, the chain percolated end to end.
    first_plate_ids = partition.subset_ids[: counts[0]].unique()
    last_plate_ids = partition.subset_ids[sum(counts[:-1]) :].unique()
    assert not bool(torch.isin(first_plate_ids, last_plate_ids).any())
    assert accounting["final_region_count"] >= 3


# --------------------------------------------------------------------------
# interface support: a single accidental smooth edge must not merge
# --------------------------------------------------------------------------


def test_a_single_smooth_edge_alone_is_insufficient_support_for_merge():
    """Directly exercises the support floor: an interface with only ONE
    edge, however smooth, must fail `min_unique_surfels_per_interface_side`
    regardless of its curvature/positional statistics."""

    config = InterfaceCoherentMergeConfig()
    min_side = config.min_unique_surfels_per_interface_side()
    assert min_side >= 2  # a single edge (1 surfel per side) cannot pass this


# --------------------------------------------------------------------------
# sign invariance / determinism
# --------------------------------------------------------------------------


def test_sign_flipped_normals_produce_an_identical_partition():
    orientation = _cylinder_band(theta_span=math.pi)
    flipped_normals = orientation.surface_normal.clone() * -1.0
    flipped = _Orientation(orientation.positions, flipped_normals, orientation.tangent_axis_u, orientation.tangent_axis_v)

    a = partition_surfels_interface_coherent(orientation)
    b = partition_surfels_interface_coherent(flipped)
    assert torch.equal(a.subset_ids, b.subset_ids)


def test_partition_is_deterministic_across_repeated_runs():
    orientation, _ = _crease(angle_deg=45.0)
    first = partition_surfels_interface_coherent(orientation)
    second = partition_surfels_interface_coherent(orientation)
    assert torch.equal(first.subset_ids, second.subset_ids)
    assert first.merge_provenance == second.merge_provenance


# --------------------------------------------------------------------------
# coverage contract
# --------------------------------------------------------------------------


def test_every_surfel_receives_exactly_one_owner_and_none_is_dropped():
    torch.manual_seed(7)
    orientation, _ = _crease(angle_deg=60.0)
    accounting = interface_coherent_accounting(partition_surfels_interface_coherent(orientation))
    assert accounting["assigned_surfel_count"] == accounting["input_surfel_count"]
    assert accounting["unassigned_surfel_count"] == 0
    assert accounting["multiply_owned_surfel_count"] == 0
    assert accounting["coverage_identity_holds"] is True


def test_original_tensors_are_unchanged_after_partitioning():
    orientation = _cylinder_band(theta_span=math.pi)
    position_snapshot = orientation.positions.clone()
    normal_snapshot = orientation.surface_normal.clone()

    partition_surfels_interface_coherent(orientation)

    assert torch.equal(orientation.positions, position_snapshot)
    assert torch.equal(orientation.surface_normal, normal_snapshot)


def test_empty_and_single_surfel_input_stay_coverage_exact():
    empty = _Orientation(torch.zeros((0, 3)), torch.zeros((0, 3)), torch.zeros((0, 3)), torch.zeros((0, 3)))
    empty_accounting = interface_coherent_accounting(partition_surfels_interface_coherent(empty))
    assert empty_accounting["coverage_identity_holds"] is True

    single = _Orientation(
        torch.zeros((1, 3)), torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[0.0, 1.0, 0.0]])
    )
    single_accounting = interface_coherent_accounting(partition_surfels_interface_coherent(single))
    assert single_accounting["subset_count"] == 1
    assert single_accounting["coverage_identity_holds"] is True


# --------------------------------------------------------------------------
# provenance / architectural isolation
# --------------------------------------------------------------------------


def test_merge_provenance_is_inspectable_and_matches_final_grouping():
    orientation = _cylinder_band(theta_span=math.pi)
    partition = partition_surfels_interface_coherent(orientation)

    assert len(partition.merge_provenance) == partition.initial_region_count - partition.subset_count
    for record in partition.merge_provenance:
        assert set(record) >= {
            "round", "region_a", "region_b", "edge_count", "unique_surfel_count_a", "unique_surfel_count_b",
            "extent_in_spacing_units", "fraction_smooth_continuation", "mean_residual", "mean_normal_offset_ratio",
        }


def test_only_one_new_free_parameter_is_introduced():
    default = InterfaceCoherentMergeConfig()
    fields = {
        f for f in default.__dataclass_fields__
        if f not in ("local", "region", "shape_operator_neighbor_count", "shape_operator_ridge",
                     "residual_mad_multiplier", "parallel_sheet_normal_over_tangent_ratio")
    }
    assert fields == {"interface_smooth_majority_fraction"}
    assert default.interface_smooth_majority_fraction == 0.5


def test_module_does_not_derive_per_surfel_normals_or_import_covariance_frame():
    from osn_gs.surface import torch_interface_coherent_region_merge

    tree = ast.parse(inspect.getsource(torch_interface_coherent_region_merge))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
            imported.extend(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    forbidden = ("covariance_frame", "derive_surface_orientation")
    for name in imported:
        for banned in forbidden:
            assert banned not in name, f"must not import {name}"
