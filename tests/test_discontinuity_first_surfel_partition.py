from __future__ import annotations

import ast
import inspect
import math

import torch

from osn_gs.surface.torch_discontinuity_first_surfel_partition import (
    CUT_REASON_PARALLEL_SHEET,
    CUT_REASON_RESIDUAL,
    DiscontinuityFirstConfig,
    discontinuity_first_accounting,
    partition_surfels_discontinuity_first,
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
    """A smoothly curved strip -- normals rotate by up to `theta_span`
    radians end to end, but the local curvature is exactly what a shape
    operator predicts. Used for the curved-surface contract (section 7)."""

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
    """Two flat half-planes sharing an edge at x=0, meeting at `angle_deg` --
    a genuine, non-smooth surface transition. Returns (orientation, count_per_side)."""

    side_a = _flat_sheet(rows, columns, pitch)
    side_a.positions[:, 0] -= (rows - 1) * pitch  # occupies x in [-(rows-1)*pitch, 0]

    angle = math.radians(angle_deg)
    u = torch.arange(rows, dtype=torch.float32) * pitch  # starts exactly at the shared x=0 seam
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
    """Two flat sheets with IDENTICAL normals, close enough to be spatial
    candidates, separated mainly along their shared normal direction."""

    a = _flat_sheet(rows, columns, pitch, z=0.0)
    b = _flat_sheet(rows, columns, pitch, z=gap, x0=lateral_offset)
    positions = torch.cat([a.positions, b.positions], dim=0)
    normals = torch.cat([a.surface_normal, b.surface_normal], dim=0)
    tangent_u = torch.cat([a.tangent_axis_u, b.tangent_axis_u], dim=0)
    tangent_v = torch.cat([a.tangent_axis_v, b.tangent_axis_v], dim=0)
    return _Orientation(positions, normals, tangent_u, tangent_v)


# --------------------------------------------------------------------------
# curved-surface contract (section 7)
# --------------------------------------------------------------------------


def test_flat_sheet_stays_one_subset_with_zero_cuts():
    accounting = discontinuity_first_accounting(partition_surfels_discontinuity_first(_flat_sheet()))
    assert accounting["subset_count"] == 1
    assert accounting["boundary_cut_edge_count"] == 0
    assert accounting["coverage_identity_holds"] is True


def test_cylindrical_band_with_180_degree_rotation_remains_one_region():
    """The central architectural distinction: large normal ROTATION alone
    must not create a boundary when it is exactly what the local shape
    operator predicts (smooth curvature)."""

    orientation = _cylinder_band(theta_span=math.pi)
    partition = partition_surfels_discontinuity_first(orientation)
    accounting = discontinuity_first_accounting(partition)

    assert accounting["subset_count"] == 1
    assert accounting["boundary_cut_edge_count"] == 0
    # The normal really does rotate substantially -- confirm the gradient
    # magnitude is large so "no cuts" is not simply because there was no
    # curvature to test against.
    assert float(partition.normal_gradient_magnitude.median()) > 1.0


def test_rounded_quarter_cylinder_does_not_fragment_from_accumulated_rotation():
    orientation = _cylinder_band(theta_span=math.pi / 2.0)
    accounting = discontinuity_first_accounting(partition_surfels_discontinuity_first(orientation))
    assert accounting["subset_count"] == 1
    assert accounting["boundary_cut_edge_count"] == 0


# --------------------------------------------------------------------------
# discontinuity contracts (section 8)
# --------------------------------------------------------------------------


def test_sharp_crease_is_cut_and_separates_into_two_dominant_components():
    orientation, count_per_side = _crease(angle_deg=90.0)
    partition = partition_surfels_discontinuity_first(orientation)
    accounting = discontinuity_first_accounting(partition)

    assert accounting["boundary_cut_edge_count"] > 0
    assert accounting["coverage_identity_holds"] is True
    # The two largest final subsets must overwhelmingly reconstruct the two
    # original flat halves -- a small fringe of boundary-adjacent fragments
    # is acceptable (finite kNN neighbourhood scale near a true
    # discontinuity), but the crease must not be missed or over-merged.
    top_two_fraction = float(partition.subset_sizes[:2].sum()) / int(len(partition))
    assert top_two_fraction > 0.85
    assert int(partition.subset_sizes[0].item()) <= count_per_side
    assert int(partition.subset_sizes[1].item()) <= count_per_side


def test_crease_boundary_evidence_is_dominated_by_true_crossing_edges():
    orientation, _ = _crease(angle_deg=90.0)
    partition = partition_surfels_discontinuity_first(orientation)
    boundary = partition.boundary_edges
    left_x = orientation.positions[boundary[:, 0], 0]
    right_x = orientation.positions[boundary[:, 1], 0]
    crosses_seam = ((left_x <= 0) & (right_x >= 0)) | ((right_x <= 0) & (left_x >= 0))
    assert float(crosses_seam.float().mean()) > 0.7


def test_two_nearby_parallel_sheets_separate_despite_matching_normals():
    orientation = _parallel_sheets(gap=0.15)
    partition = partition_surfels_discontinuity_first(orientation)
    accounting = discontinuity_first_accounting(partition)

    assert accounting["subset_count"] >= 2
    assert accounting["boundary_cut_reason_counts"][CUT_REASON_PARALLEL_SHEET] > 0
    assert accounting["coverage_identity_holds"] is True
    # Confirm it split along the two sheets, not arbitrarily.
    half = int(orientation.positions.shape[0]) // 2
    assert int(partition.subset_ids[:half].unique().numel()) == 1
    assert int(partition.subset_ids[half:].unique().numel()) == 1
    assert int(partition.subset_ids[0]) != int(partition.subset_ids[half])


def test_parallel_sheet_criterion_is_not_degenerate_against_the_candidate_gate():
    """Regression for a real bug found during implementation: comparing the
    normal-direction offset against `spatial_connect_spacing_multiplier *
    spacing` can never fire, because the candidate gate itself already
    bounds total displacement at that same value. The fixed criterion
    (normal offset vs. the edge's OWN tangential offset) must actually cut."""

    config = DiscontinuityFirstConfig()
    orientation = _parallel_sheets(gap=0.15)
    partition = partition_surfels_discontinuity_first(orientation, config)
    assert int(partition.cut_reason_parallel_sheet.sum()) > 0


# --------------------------------------------------------------------------
# sign invariance / determinism
# --------------------------------------------------------------------------


def test_sign_flipped_normals_produce_an_identical_partition():
    orientation = _flat_sheet()
    flipped_normals = orientation.surface_normal.clone()
    flipped_normals[::3] *= -1.0
    flipped = _Orientation(orientation.positions, flipped_normals, orientation.tangent_axis_u, orientation.tangent_axis_v)

    a = partition_surfels_discontinuity_first(orientation)
    b = partition_surfels_discontinuity_first(flipped)
    assert torch.equal(a.subset_ids, b.subset_ids)
    assert torch.equal(a.cut_mask, b.cut_mask)


def test_partition_is_deterministic_across_repeated_runs():
    orientation, _ = _crease(angle_deg=45.0)
    first = partition_surfels_discontinuity_first(orientation)
    second = partition_surfels_discontinuity_first(orientation)
    assert torch.equal(first.subset_ids, second.subset_ids)
    assert torch.equal(first.cut_mask, second.cut_mask)
    assert first.residual_threshold == second.residual_threshold


# --------------------------------------------------------------------------
# threshold is data-derived, not a swept constant
# --------------------------------------------------------------------------


def test_residual_threshold_matches_the_documented_median_plus_mad_formula():
    orientation, _ = _crease(angle_deg=60.0)
    config = DiscontinuityFirstConfig()
    partition = partition_surfels_discontinuity_first(orientation, config)

    spatial_mask = partition.graph.spatial_edge_mask
    residual = partition.edge_residual[spatial_mask]
    median = torch.median(residual)
    mad = torch.median((residual - median).abs())
    expected = float(median + config.residual_mad_multiplier * 1.4826 * mad)
    assert abs(partition.residual_threshold - expected) < 1e-4


# --------------------------------------------------------------------------
# coverage contract
# --------------------------------------------------------------------------


def test_every_surfel_receives_exactly_one_owner_and_none_is_dropped():
    torch.manual_seed(5)
    positions = torch.cat([_flat_sheet().positions, torch.randn(40, 3) * 3.0], dim=0)
    normals = torch.nn.functional.normalize(torch.randn(positions.shape[0], 3), dim=-1)
    normals[:196] = torch.tensor([0.0, 0.0, 1.0])
    tangent_u = torch.nn.functional.normalize(torch.randn(positions.shape[0], 3), dim=-1)
    tangent_u = tangent_u - (tangent_u * normals).sum(-1, keepdim=True) * normals
    tangent_u = torch.nn.functional.normalize(tangent_u, dim=-1)
    tangent_v = torch.cross(normals, tangent_u, dim=-1)
    orientation = _Orientation(positions, normals, tangent_u, tangent_v)

    accounting = discontinuity_first_accounting(partition_surfels_discontinuity_first(orientation))
    assert accounting["input_surfel_count"] == int(positions.shape[0])
    assert accounting["assigned_surfel_count"] == accounting["input_surfel_count"]
    assert accounting["unassigned_surfel_count"] == 0
    assert accounting["multiply_owned_surfel_count"] == 0
    assert accounting["coverage_identity_holds"] is True


def test_original_tensors_are_unchanged_after_partitioning():
    orientation = _flat_sheet()
    position_snapshot = orientation.positions.clone()
    normal_snapshot = orientation.surface_normal.clone()

    partition_surfels_discontinuity_first(orientation)

    assert torch.equal(orientation.positions, position_snapshot)
    assert torch.equal(orientation.surface_normal, normal_snapshot)


def test_empty_and_single_surfel_input_stay_coverage_exact():
    empty = _Orientation(torch.zeros((0, 3)), torch.zeros((0, 3)), torch.zeros((0, 3)), torch.zeros((0, 3)))
    empty_accounting = discontinuity_first_accounting(partition_surfels_discontinuity_first(empty))
    assert empty_accounting["coverage_identity_holds"] is True

    single = _Orientation(
        torch.zeros((1, 3)), torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[0.0, 1.0, 0.0]])
    )
    single_accounting = discontinuity_first_accounting(partition_surfels_discontinuity_first(single))
    assert single_accounting["subset_count"] == 1
    assert single_accounting["coverage_identity_holds"] is True


# --------------------------------------------------------------------------
# architectural isolation
# --------------------------------------------------------------------------


def test_only_two_new_free_parameters_documented():
    default = DiscontinuityFirstConfig()
    fields = {
        f for f in default.__dataclass_fields__
        if f not in ("local", "shape_operator_neighbor_count", "shape_operator_ridge")
    }
    assert fields == {"residual_mad_multiplier", "parallel_sheet_normal_over_tangent_ratio"}
    assert default.residual_mad_multiplier == 3.0
    assert default.parallel_sheet_normal_over_tangent_ratio == 1.0


def test_module_does_not_derive_per_surfel_normals_or_import_covariance_frame():
    from osn_gs.surface import torch_discontinuity_first_surfel_partition

    tree = ast.parse(inspect.getsource(torch_discontinuity_first_surfel_partition))
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


def test_boundary_cut_reasons_are_reported_separately():
    orientation, _ = _crease(angle_deg=90.0)
    accounting = discontinuity_first_accounting(partition_surfels_discontinuity_first(orientation))
    reasons = accounting["boundary_cut_reason_counts"]
    assert set(reasons) == {CUT_REASON_RESIDUAL, CUT_REASON_PARALLEL_SHEET, "both"}
    assert reasons[CUT_REASON_RESIDUAL] > 0
