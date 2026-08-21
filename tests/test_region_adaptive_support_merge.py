from __future__ import annotations

import math

import torch

from osn_gs.surface.torch_bilateral_interface_region_merge import (
    partition_surfels_bilateral_interface,
)
from osn_gs.surface.torch_coverage_first_subset_partition import CoverageFirstPartitionConfig, build_candidate_graph
from osn_gs.surface.torch_discontinuity_first_surfel_partition import _auto_chunk_size, _knn
from osn_gs.surface.torch_region_adaptive_support_merge import (
    AdaptiveSupportConfig,
    SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL,
    SUPPORT_MODE_FIXED_MASKED_KNN,
    _evaluate_interface_evidence,
    _support_neighbor_pool,
    partition_surfels_region_adaptive_support,
    region_adaptive_support_accounting,
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


def _cylinder_band(n_theta: int = 90, n_y: int = 8, radius: float = 0.5, theta_span: float = math.pi, pitch_y: float = 0.05) -> _Orientation:
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


def _parallel_sheets(rows: int = 12, columns: int = 12, pitch: float = 0.1, gap: float = 0.15, lateral_offset: float = 0.03) -> _Orientation:
    a = _flat_sheet(rows, columns, pitch, z=0.0)
    b = _flat_sheet(rows, columns, pitch, z=gap, x0=lateral_offset)
    positions = torch.cat([a.positions, b.positions], dim=0)
    normals = torch.cat([a.surface_normal, b.surface_normal], dim=0)
    tangent_u = torch.cat([a.tangent_axis_u, b.tangent_axis_u], dim=0)
    tangent_v = torch.cat([a.tangent_axis_v, b.tangent_axis_v], dim=0)
    return _Orientation(positions, normals, tangent_u, tangent_v)


def _zigzag(plate_count: int = 4, rows: int = 8, columns: int = 8, pitch: float = 0.1, angle_deg: float = 90.0):
    positions_list, normals_list, tangent_u_list, tangent_v_list, counts = [], [], [], [], []
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
        far_edge = cursor + (rows - 1) * pitch * direction
        sign = 1.0 if plate_index % 2 == 0 else -1.0
        angle = math.radians(angle_deg) * sign
        rot = torch.tensor([[math.cos(angle), 0.0, math.sin(angle)], [0.0, 1.0, 0.0], [-math.sin(angle), 0.0, math.cos(angle)]])
        cursor = far_edge
        direction = rot @ direction
        normal = rot @ normal
    positions = torch.cat(positions_list, dim=0)
    normals = torch.cat(normals_list, dim=0)
    tangent_u = torch.cat(tangent_u_list, dim=0)
    tangent_v = torch.cat(tangent_v_list, dim=0)
    return _Orientation(positions, normals, tangent_u, tangent_v), counts


def _finger_embedded_in_denser_opposite_region(finger_pitch: float = 0.1, block_pitch: float = 0.05, n_finger: int = 20, n_block_x: int = 14, n_block_y: int = 70):
    """A thin 1-wide 'finger' of one orientation flanked on both sides by a
    DENSER block of an incompatible orientation. At the finger's own
    (uniform) internal pitch, its immediate same-strip neighbours are its
    two closest possible points by construction -- but the denser
    surrounding block still crowds most of a FIXED, global k=8 window,
    verified below to push support under 2 while a locality-bounded
    same-region search (still using the SAME distance contract) recovers
    it. Returns (orientation, n_finger)."""

    finger_y = torch.arange(n_finger, dtype=torch.float32) * finger_pitch
    finger_pos = torch.stack([torch.zeros_like(finger_y), finger_y, torch.zeros_like(finger_y)], dim=1)
    finger_normal = torch.tensor([[0.0, 0.0, 1.0]]).repeat(n_finger, 1)

    def _block(x_offsets: torch.Tensor) -> torch.Tensor:
        by = torch.arange(0, n_block_y, dtype=torch.float32) * block_pitch
        bxx, byy = torch.meshgrid(x_offsets, by, indexing="ij")
        return torch.stack([bxx.reshape(-1), byy.reshape(-1), torch.zeros_like(bxx.reshape(-1))], dim=1)

    right = _block(torch.arange(1, n_block_x + 1, dtype=torch.float32) * block_pitch)
    left = _block(-torch.arange(1, n_block_x + 1, dtype=torch.float32) * block_pitch)
    block_pos = torch.cat([right, left], dim=0)
    block_normal = torch.tensor([[1.0, 0.0, 0.0]]).repeat(block_pos.shape[0], 1)

    positions = torch.cat([finger_pos, block_pos], dim=0)
    normals = torch.cat([finger_normal, block_normal], dim=0)
    tangent_u = torch.tensor([[1.0, 0.0, 0.0]]).repeat(positions.shape[0], 1)
    tangent_v = torch.tensor([[0.0, 1.0, 0.0]]).repeat(positions.shape[0], 1)
    return _Orientation(positions, normals, tangent_u, tangent_v), n_finger


# --------------------------------------------------------------------------
# FIXED mode reproduces Worklog 100 byte-for-byte (A/B isolation contract)
# --------------------------------------------------------------------------


def test_fixed_masked_knn_mode_reproduces_worklog_100_exactly():
    for orientation in (_cylinder_band(theta_span=math.pi), _parallel_sheets()):
        baseline = partition_surfels_bilateral_interface(orientation)
        config = AdaptiveSupportConfig(support_mode=SUPPORT_MODE_FIXED_MASKED_KNN)
        reproduced = partition_surfels_region_adaptive_support(orientation, config)
        assert torch.equal(baseline.subset_ids, reproduced.subset_ids)


# --------------------------------------------------------------------------
# the central support-starvation mechanism, isolated and deterministic
# --------------------------------------------------------------------------


def test_dense_opposite_region_starves_fixed_support_but_adaptive_recovers_it():
    orientation, n_finger = _finger_embedded_in_denser_opposite_region()
    local_config = CoverageFirstPartitionConfig()
    graph = build_candidate_graph(orientation, local_config)
    node_root = torch.cat([
        torch.zeros(n_finger, dtype=torch.int64),
        torch.ones(int(orientation.positions.shape[0]) - n_finger, dtype=torch.int64),
    ])
    query = torch.tensor([n_finger // 2])  # a middle finger node, safely away from either end

    pool_fixed_index, pool_fixed_distance = _knn(orientation.positions, 8, _auto_chunk_size(int(orientation.positions.shape[0]), orientation.positions.device), None)
    _, fixed_mask = _support_neighbor_pool(
        SUPPORT_MODE_FIXED_MASKED_KNN, query, node_root, pool_fixed_index, pool_fixed_distance,
        graph.local_spacing, local_config.spatial_connect_spacing_multiplier, 8,
    )
    assert int(fixed_mask.sum().item()) < 2  # UNSUPPORTED under the fixed, global-8 mask

    pool_adaptive_index, pool_adaptive_distance = _knn(orientation.positions, 32, _auto_chunk_size(int(orientation.positions.shape[0]), orientation.positions.device), None)
    _, adaptive_mask = _support_neighbor_pool(
        SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL, query, node_root, pool_adaptive_index, pool_adaptive_distance,
        graph.local_spacing, local_config.spatial_connect_spacing_multiplier, 8,
    )
    assert int(adaptive_mask.sum().item()) >= 2  # SUPPORTED under the locality-bounded same-region search


def test_adaptive_support_never_reaches_beyond_the_locality_bound():
    """Directive section 8: adaptive support must not turn a tiny fragment
    into a model estimated from a geometrically distant part of the same
    region. Every kept neighbour's distance must respect the SAME locality
    contract (spacing_multiplier * local_spacing) as the base candidate
    graph -- verified directly against the returned mask."""

    orientation, n_finger = _finger_embedded_in_denser_opposite_region()
    local_config = CoverageFirstPartitionConfig()
    graph = build_candidate_graph(orientation, local_config)
    node_root = torch.cat([
        torch.zeros(n_finger, dtype=torch.int64),
        torch.ones(int(orientation.positions.shape[0]) - n_finger, dtype=torch.int64),
    ])
    query = torch.arange(n_finger)
    pool_index, pool_distance = _knn(orientation.positions, 32, _auto_chunk_size(int(orientation.positions.shape[0]), orientation.positions.device), None)
    _, mask = _support_neighbor_pool(
        SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL, query, node_root, pool_index, pool_distance,
        graph.local_spacing, local_config.spatial_connect_spacing_multiplier, 8,
    )
    kept_distance = pool_distance[query]
    bound = (local_config.spatial_connect_spacing_multiplier * graph.local_spacing[query]).unsqueeze(1)
    violating = mask & (kept_distance > bound)
    assert int(violating.sum().item()) == 0


def test_adaptive_support_recovers_more_evaluable_interface_edges_than_fixed():
    orientation, n_finger = _finger_embedded_in_denser_opposite_region()
    local_config = CoverageFirstPartitionConfig()
    graph = build_candidate_graph(orientation, local_config)
    node_root = torch.cat([
        torch.zeros(n_finger, dtype=torch.int64),
        torch.ones(int(orientation.positions.shape[0]) - n_finger, dtype=torch.int64),
    ])
    left, right = graph.candidate_edges[:, 0], graph.candidate_edges[:, 1]
    cross_mask = graph.spatial_edge_mask & (node_root[left] != node_root[right])
    cross_left, cross_right = left[cross_mask], right[cross_mask]

    pool_fixed_index, pool_fixed_distance = _knn(orientation.positions, 8, _auto_chunk_size(int(orientation.positions.shape[0]), orientation.positions.device), None)
    fixed_config = AdaptiveSupportConfig(support_mode=SUPPORT_MODE_FIXED_MASKED_KNN)
    fixed_evidence = _evaluate_interface_evidence(
        orientation, pool_fixed_index, pool_fixed_distance, graph.local_spacing, node_root, cross_left, cross_right, fixed_config,
    )

    pool_adaptive_index, pool_adaptive_distance = _knn(orientation.positions, 32, _auto_chunk_size(int(orientation.positions.shape[0]), orientation.positions.device), None)
    adaptive_config = AdaptiveSupportConfig(support_mode=SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL)
    adaptive_evidence = _evaluate_interface_evidence(
        orientation, pool_adaptive_index, pool_adaptive_distance, graph.local_spacing, node_root, cross_left, cross_right, adaptive_config,
    )

    fixed_supported = int((~fixed_evidence["unsupported_left"]).sum() + (~fixed_evidence["unsupported_right"]).sum())
    adaptive_supported = int((~adaptive_evidence["unsupported_left"]).sum() + (~adaptive_evidence["unsupported_right"]).sum())
    assert adaptive_supported > fixed_supported


# --------------------------------------------------------------------------
# negative fixtures: adaptive support must not cause unwanted merges
# --------------------------------------------------------------------------


def test_sharp_crease_remains_separate_under_adaptive_support():
    orientation, _ = _crease(angle_deg=90.0)
    config = AdaptiveSupportConfig(support_mode=SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL)
    partition = partition_surfels_region_adaptive_support(orientation, config)
    accounting = region_adaptive_support_accounting(partition)
    assert accounting["final_region_count"] >= 2
    assert accounting["coverage_identity_holds"] is True
    top_two_fraction = float(partition.subset_sizes[:2].sum()) / int(len(partition))
    assert top_two_fraction > 0.85


def test_parallel_sheets_remain_separate_under_adaptive_support():
    orientation = _parallel_sheets()
    config = AdaptiveSupportConfig(support_mode=SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL)
    partition = partition_surfels_region_adaptive_support(orientation, config)
    accounting = region_adaptive_support_accounting(partition)
    assert accounting["final_region_count"] >= 2
    assert accounting["coverage_identity_holds"] is True
    half = int(orientation.positions.shape[0]) // 2
    assert int(partition.subset_ids[:half].unique().numel()) == 1
    assert int(partition.subset_ids[half:].unique().numel()) == 1
    assert int(partition.subset_ids[0]) != int(partition.subset_ids[half])


def test_narrow_bridge_chain_does_not_percolate_under_adaptive_support():
    orientation, counts = _zigzag(plate_count=4, angle_deg=90.0)
    config = AdaptiveSupportConfig(support_mode=SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL)
    partition = partition_surfels_region_adaptive_support(orientation, config)
    accounting = region_adaptive_support_accounting(partition)
    assert accounting["coverage_identity_holds"] is True
    first_plate_ids = partition.subset_ids[: counts[0]].unique()
    last_plate_ids = partition.subset_ids[sum(counts[:-1]) :].unique()
    assert not bool(torch.isin(first_plate_ids, last_plate_ids).any())


def test_unrelated_surface_with_incompatible_normal_does_not_merge_via_adaptive_support():
    """The dense-block/finger fixture itself: block and finger have
    INCOMPATIBLE normals (a genuine discontinuity), so adaptive support
    acquiring more same-region evidence for the finger must still never
    cause it to merge with the perpendicular block."""

    orientation, n_finger = _finger_embedded_in_denser_opposite_region()
    config = AdaptiveSupportConfig(support_mode=SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL)
    partition = partition_surfels_region_adaptive_support(orientation, config)
    accounting = region_adaptive_support_accounting(partition)
    assert accounting["coverage_identity_holds"] is True
    finger_ids = partition.subset_ids[:n_finger].unique()
    block_ids = partition.subset_ids[n_finger:].unique()
    assert not bool(torch.isin(finger_ids, block_ids).any())


# --------------------------------------------------------------------------
# curvature recovery preserved (Worklog 100's own positive results)
# --------------------------------------------------------------------------


def test_quarter_and_half_cylinder_still_merge_back_under_adaptive_support():
    for span in (math.pi / 2.0, math.pi):
        orientation = _cylinder_band(theta_span=span)
        config = AdaptiveSupportConfig(support_mode=SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL)
        partition = partition_surfels_region_adaptive_support(orientation, config)
        accounting = region_adaptive_support_accounting(partition)
        assert accounting["initial_region_count"] > 1
        assert accounting["final_region_count"] == 1
        assert accounting["coverage_identity_holds"] is True


# --------------------------------------------------------------------------
# coverage / determinism / architectural isolation
# --------------------------------------------------------------------------


def test_every_surfel_receives_exactly_one_owner_and_none_is_dropped():
    orientation, _ = _crease(angle_deg=60.0)
    for mode in (SUPPORT_MODE_FIXED_MASKED_KNN, SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL):
        config = AdaptiveSupportConfig(support_mode=mode)
        accounting = region_adaptive_support_accounting(partition_surfels_region_adaptive_support(orientation, config))
        assert accounting["assigned_surfel_count"] == accounting["input_surfel_count"]
        assert accounting["unassigned_surfel_count"] == 0
        assert accounting["multiply_owned_surfel_count"] == 0
        assert accounting["coverage_identity_holds"] is True


def test_partition_is_deterministic_across_repeated_runs():
    orientation, _ = _crease(angle_deg=45.0)
    config = AdaptiveSupportConfig(support_mode=SUPPORT_MODE_ADAPTIVE_SAME_REGION_LOCAL)
    first = partition_surfels_region_adaptive_support(orientation, config)
    second = partition_surfels_region_adaptive_support(orientation, config)
    assert torch.equal(first.subset_ids, second.subset_ids)
    assert first.merge_provenance == second.merge_provenance


def test_no_merge_threshold_field_differs_from_worklog_100_defaults():
    """Directive section 13: residual_mad_multiplier,
    parallel_sheet_normal_over_tangent_ratio, interface_smooth_majority_fraction,
    and the interface support floors must be untouched."""

    from osn_gs.surface.torch_bilateral_interface_region_merge import BilateralInterfaceMergeConfig

    wl100_default = BilateralInterfaceMergeConfig()
    new_default = AdaptiveSupportConfig()
    assert new_default.residual_mad_multiplier == wl100_default.residual_mad_multiplier
    assert new_default.parallel_sheet_normal_over_tangent_ratio == wl100_default.parallel_sheet_normal_over_tangent_ratio
    assert new_default.interface_smooth_majority_fraction == wl100_default.interface_smooth_majority_fraction
    assert new_default.min_unique_surfels_per_interface_side() == wl100_default.min_unique_surfels_per_interface_side()
    assert new_default.min_interface_extent_in_spacing_units() == wl100_default.min_interface_extent_in_spacing_units()
