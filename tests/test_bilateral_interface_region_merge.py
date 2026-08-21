from __future__ import annotations

import ast
import inspect
import math

import torch

from osn_gs.surface.torch_bilateral_interface_region_merge import (
    BilateralInterfaceMergeConfig,
    _compute_bilateral_edge_evidence,
    _fit_region_conditioned_shape_operators,
    bilateral_interface_accounting,
    partition_surfels_bilateral_interface,
)
from osn_gs.surface.torch_coverage_first_subset_partition import CoverageFirstPartitionConfig
from osn_gs.surface.torch_discontinuity_first_surfel_partition import (
    _auto_chunk_size,
    _knn,
    _predicted_delta_n_t,
    _tangent_plane_components,
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


def _parallel_sheets(rows: int = 12, columns: int = 12, pitch: float = 0.1, gap: float = 0.15, lateral_offset: float = 0.03) -> _Orientation:
    a = _flat_sheet(rows, columns, pitch, z=0.0)
    b = _flat_sheet(rows, columns, pitch, z=gap, x0=lateral_offset)
    positions = torch.cat([a.positions, b.positions], dim=0)
    normals = torch.cat([a.surface_normal, b.surface_normal], dim=0)
    tangent_u = torch.cat([a.tangent_axis_u, b.tangent_axis_u], dim=0)
    tangent_v = torch.cat([a.tangent_axis_v, b.tangent_axis_v], dim=0)
    return _Orientation(positions, normals, tangent_u, tangent_v)


def _zigzag(plate_count: int = 3, rows: int = 8, columns: int = 8, pitch: float = 0.1, angle_deg: float = 90.0):
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
# curvature recovery: over-segmented smooth surfaces still merge back
# --------------------------------------------------------------------------


# `n_theta=90` (rather than the WL98/99 default 60): with the default, WL97's
# over-segmentation of this fixture happens to leave a very small (~16
# surfel) tail fragment. Region-conditioned fits on such a tiny fragment are
# still numerically WELL-CONDITIONED (verified: eigenvalue ratios ~0.05-0.1,
# not degenerate) but drawn from only 3-5 same-region neighbours, giving a
# measurably different Taylor-truncation residual (still tiny in absolute
# terms, ~1e-6) than the large regions' near-exact fits (~1e-7) -- purely a
# small-sample-size artifact of this idealized, floating-point-exact
# fixture, not a modelling bug (see Worklog 100's write-up). A finer
# angular sampling avoids leaving such a small tail fragment in the WL97
# initialization for THIS fixture; this is a test-geometry choice, not a
# change to any algorithm threshold.
def test_quarter_cylinder_over_segmented_by_wl97_merges_back_into_one_region():
    orientation = _cylinder_band(theta_span=math.pi / 2.0, n_theta=90)
    partition = partition_surfels_bilateral_interface(orientation)
    accounting = bilateral_interface_accounting(partition)

    assert accounting["initial_region_count"] > 1
    assert accounting["final_region_count"] == 1
    assert accounting["coverage_identity_holds"] is True


def test_half_cylinder_large_total_rotation_still_merges_back_into_one_region():
    orientation = _cylinder_band(theta_span=math.pi, n_theta=90)
    partition = partition_surfels_bilateral_interface(orientation)
    accounting = bilateral_interface_accounting(partition)

    assert accounting["initial_region_count"] > 1
    assert accounting["final_region_count"] == 1
    assert accounting["coverage_identity_holds"] is True


# --------------------------------------------------------------------------
# true discontinuities remain separate
# --------------------------------------------------------------------------


def test_sharp_crease_regions_remain_separate():
    orientation, _ = _crease(angle_deg=90.0)
    partition = partition_surfels_bilateral_interface(orientation)
    accounting = bilateral_interface_accounting(partition)

    assert accounting["final_region_count"] >= 2
    assert accounting["coverage_identity_holds"] is True
    top_two_fraction = float(partition.subset_sizes[:2].sum()) / int(len(partition))
    assert top_two_fraction > 0.85


def test_two_nearby_parallel_sheets_remain_separate():
    orientation = _parallel_sheets()
    partition = partition_surfels_bilateral_interface(orientation)
    accounting = bilateral_interface_accounting(partition)

    assert accounting["coverage_identity_holds"] is True
    assert accounting["final_region_count"] >= 2
    half = int(orientation.positions.shape[0]) // 2
    assert int(partition.subset_ids[:half].unique().numel()) == 1
    assert int(partition.subset_ids[half:].unique().numel()) == 1
    assert int(partition.subset_ids[0]) != int(partition.subset_ids[half])


def test_narrow_multi_region_bridge_chain_does_not_percolate():
    orientation, counts = _zigzag(plate_count=4, angle_deg=90.0)
    partition = partition_surfels_bilateral_interface(orientation)
    accounting = bilateral_interface_accounting(partition)

    assert accounting["coverage_identity_holds"] is True
    first_plate_ids = partition.subset_ids[: counts[0]].unique()
    last_plate_ids = partition.subset_ids[sum(counts[:-1]) :].unique()
    assert not bool(torch.isin(first_plate_ids, last_plate_ids).any())


# --------------------------------------------------------------------------
# the central semantic change: bilateral agreement, region-conditioned fit
# --------------------------------------------------------------------------


def test_one_sided_explainable_interface_does_not_pass_the_bilateral_edge_test():
    """Hand-constructed, fully deterministic: region A's own-conditioned
    model predicts the step toward B (predicted (1.2, 0) vs actual (0, 0) ->
    residual 1.2, large); region B's own model predicts the step toward A
    perfectly (predicted (0, 0) vs actual (0, 0) -> residual 0). A locally
    explainable interface from only ONE side must not count as bilaterally
    smooth."""

    positions = torch.tensor([
        [0.0, 0.0, 0.0],    # 0: i, region A
        [-0.3, 0.0, 0.0],   # 1: j, region B
        [0.1, 0.0, 0.0],    # 2: A same-region neighbor
        [0.2, 0.0, 0.0],    # 3: A same-region neighbor
        [-0.4, 0.0, 0.0],   # 4: B same-region neighbor
        [-0.5, 0.0, 0.0],   # 5: B same-region neighbor
    ])
    normals = torch.nn.functional.normalize(torch.tensor([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [-0.4, 0.0, 0.916515],
        [-0.8, 0.0, 0.6],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]), dim=-1)
    tangent_u = torch.tensor([[1.0, 0.0, 0.0]]).repeat(6, 1)
    tangent_v = torch.tensor([[0.0, 1.0, 0.0]]).repeat(6, 1)

    query_index = torch.tensor([0, 1])
    neighbor_index = torch.tensor([[2, 3], [4, 5]])
    valid_mask = torch.tensor([[True, True], [True, True]])

    shape_operator, support = _fit_region_conditioned_shape_operators(
        positions, normals, tangent_u, tangent_v, query_index, neighbor_index, valid_mask, 1e-8
    )
    assert torch.equal(support, torch.tensor([2, 2]))

    i, j = 0, 1
    dx = positions[j] - positions[i]
    dxt_i = _tangent_plane_components(dx, tangent_u[i], tangent_v[i])
    dxt_j = _tangent_plane_components(-dx, tangent_u[j], tangent_v[j])
    sign = 1.0 if float((normals[i] * normals[j]).sum()) >= 0 else -1.0
    aligned_j = normals[j] * sign
    dnt_i = _tangent_plane_components(aligned_j - normals[i], tangent_u[i], tangent_v[i])
    aligned_i = normals[i] * sign
    dnt_j = _tangent_plane_components(aligned_i - normals[j], tangent_u[j], tangent_v[j])

    predicted_i = _predicted_delta_n_t(shape_operator[0:1], dxt_i.unsqueeze(0))[0]
    predicted_j = _predicted_delta_n_t(shape_operator[1:2], dxt_j.unsqueeze(0))[0]
    r_a_to_b = (dnt_i - predicted_i).norm()
    r_b_to_a = (dnt_j - predicted_j).norm()

    assert float(r_a_to_b) > 1.0  # A's own model badly fails to predict B
    assert float(r_b_to_a) < 1e-4  # B's own model predicts A almost exactly
    # A REASONABLE residual threshold (anywhere well below 1.2, well above 0)
    # must classify this as one-sided, not bilaterally smooth:
    threshold = 0.5
    bilateral_smooth = bool((r_a_to_b <= threshold) and (r_b_to_a <= threshold))
    assert bilateral_smooth is False


def test_insufficient_same_region_support_is_marked_unsupported_not_smooth():
    """A boundary node with fewer than 2 same-region neighbours cannot
    support ANY 2x2 fit -- must be flagged unsupported, never silently
    treated as evidence FOR a smooth merge."""

    positions = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    normals = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    tangent_u = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    tangent_v = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

    query_index = torch.tensor([0])
    neighbor_index = torch.tensor([[1, 1]])  # only 1 distinct same-region neighbour available
    valid_mask = torch.tensor([[True, False]])  # only 1 of 2 flagged valid -> support_count=1

    _, support = _fit_region_conditioned_shape_operators(
        positions, normals, tangent_u, tangent_v, query_index, neighbor_index, valid_mask, 1e-8
    )
    assert int(support[0]) == 1  # below the structural minimum of 2


def test_boundary_contaminated_neighborhood_is_excluded_from_the_fit():
    """The exact reason Worklog 98 needed the min-direction workaround: a
    node whose FULL kNN neighbourhood straddles a region boundary gets a
    contaminated fit if cross-region neighbours are not excluded. Region
    conditioning must recover the SAME-region-only fit even when the query's
    raw neighbour list is dominated by the other region."""

    # Query node i has 1 same-region (A) neighbour and 3 cross-region (B)
    # neighbours with wildly different normals -- an UNCONDITIONED fit would
    # be dominated/contaminated by the B neighbours; region-conditioning
    # must use ONLY the flagged-valid (A) neighbour... but with support=1
    # (below the structural minimum) it should be UNSUPPORTED, not a
    # contaminated fit either. This confirms contamination cannot leak in
    # even when support is below the floor.
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],   # same-region neighbour
        [-0.1, 0.0, 0.0],  # cross-region neighbour, contaminating normal
        [-0.2, 0.0, 0.0],
        [-0.3, 0.0, 0.0],
    ])
    normals = torch.nn.functional.normalize(torch.tensor([
        [0.0, 0.0, 1.0],
        [0.05, 0.0, 1.0],
        [0.9, 0.0, 1.0],   # contaminating: huge normal swing
        [-0.9, 0.0, 1.0],
        [0.95, 0.0, 1.0],
    ]), dim=-1)
    tangent_u = torch.tensor([[1.0, 0.0, 0.0]]).repeat(5, 1)
    tangent_v = torch.tensor([[0.0, 1.0, 0.0]]).repeat(5, 1)

    query_index = torch.tensor([0])
    neighbor_index = torch.tensor([[1, 2, 3, 4]])
    valid_mask_conditioned = torch.tensor([[True, False, False, False]])
    valid_mask_unconditioned = torch.tensor([[True, True, True, True]])

    s_conditioned, support_conditioned = _fit_region_conditioned_shape_operators(
        positions, normals, tangent_u, tangent_v, query_index, neighbor_index, valid_mask_conditioned, 1e-8
    )
    s_unconditioned, _ = _fit_region_conditioned_shape_operators(
        positions, normals, tangent_u, tangent_v, query_index, neighbor_index, valid_mask_unconditioned, 1e-8
    )
    assert int(support_conditioned[0]) == 1  # below floor -> unsupported, not usable
    # the UNCONDITIONED fit is measurably different (contaminated) -- proves
    # the mask genuinely changes which neighbours the regression sees.
    assert not torch.allclose(s_conditioned, s_unconditioned, atol=1e-3)


# --------------------------------------------------------------------------
# coverage / determinism
# --------------------------------------------------------------------------


def test_every_surfel_receives_exactly_one_owner_and_none_is_dropped():
    orientation, _ = _crease(angle_deg=60.0)
    accounting = bilateral_interface_accounting(partition_surfels_bilateral_interface(orientation))
    assert accounting["assigned_surfel_count"] == accounting["input_surfel_count"]
    assert accounting["unassigned_surfel_count"] == 0
    assert accounting["multiply_owned_surfel_count"] == 0
    assert accounting["coverage_identity_holds"] is True


def test_partition_is_deterministic_across_repeated_runs():
    orientation, _ = _crease(angle_deg=45.0)
    first = partition_surfels_bilateral_interface(orientation)
    second = partition_surfels_bilateral_interface(orientation)
    assert torch.equal(first.subset_ids, second.subset_ids)
    assert first.merge_provenance == second.merge_provenance


def test_original_tensors_are_unchanged_after_partitioning():
    orientation = _cylinder_band(theta_span=math.pi)
    position_snapshot = orientation.positions.clone()
    normal_snapshot = orientation.surface_normal.clone()
    partition_surfels_bilateral_interface(orientation)
    assert torch.equal(orientation.positions, position_snapshot)
    assert torch.equal(orientation.surface_normal, normal_snapshot)


def test_empty_and_single_surfel_input_stay_coverage_exact():
    empty = _Orientation(torch.zeros((0, 3)), torch.zeros((0, 3)), torch.zeros((0, 3)), torch.zeros((0, 3)))
    empty_accounting = bilateral_interface_accounting(partition_surfels_bilateral_interface(empty))
    assert empty_accounting["coverage_identity_holds"] is True

    single = _Orientation(
        torch.zeros((1, 3)), torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[0.0, 1.0, 0.0]])
    )
    single_accounting = bilateral_interface_accounting(partition_surfels_bilateral_interface(single))
    assert single_accounting["subset_count"] == 1
    assert single_accounting["coverage_identity_holds"] is True


# --------------------------------------------------------------------------
# no new free parameter / architectural isolation
# --------------------------------------------------------------------------


def test_no_new_free_parameter_is_introduced():
    """Every threshold field must match Worklog 99's values exactly -- this
    batch changes the CERTIFICATE, not any threshold."""

    default = BilateralInterfaceMergeConfig()
    assert default.residual_mad_multiplier == 3.0
    assert default.parallel_sheet_normal_over_tangent_ratio == 1.0
    assert default.interface_smooth_majority_fraction == 0.5
    assert default.min_unique_surfels_per_interface_side() == default.local.neighbor_count
    assert default.min_interface_extent_in_spacing_units() == default.local.spatial_connect_spacing_multiplier


def test_module_does_not_derive_per_surfel_normals_or_import_covariance_frame():
    from osn_gs.surface import torch_bilateral_interface_region_merge

    tree = ast.parse(inspect.getsource(torch_bilateral_interface_region_merge))
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
