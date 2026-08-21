from __future__ import annotations

import dataclasses
import math

import torch

from osn_gs.surface.torch_observation_evidence import CameraViewEvidence, ObservationEvidence
from osn_gs.surface.torch_positive_visible_adjacency import (
    RELATION_STATES,
    STATE_CUT_KNOWN_FREE_SPACE,
    STATE_CUT_OCCLUDED_DOMAIN,
    STATE_CUT_POSITIONAL_SHEET_SEPARATION,
    STATE_CUT_VISIBLE_DISCONTINUITY,
    STATE_POSITIVE_VISIBLE_CONTINUATION,
    STATE_UNKNOWN_NO_POSITIVE_OBSERVATION,
    STATE_UNRESOLVED_CONFLICT,
    PositiveVisibleAdjacencyConfig,
    partition_positive_visible_adjacency,
    positive_visible_adjacency_accounting,
)
from osn_gs.surface.torch_maximal_visible_connectivity import _project_to_camera
from tests.test_maximal_visible_connectivity import (
    _LOCAL_CONFIG,
    _Orientation,
    _build_camera,
    _cylinder_band,
    _flat_orientation,
    _grid,
    _lookat_world_view,
    _procedural_evidence,
    _wall_gap_wall_camera_evidence,
    _wall_pair_with_gap,
)


def _state_counts(result) -> dict[str, int]:
    accounting = positive_visible_adjacency_accounting(result)
    return accounting["relation_state_counts"]


def _splatted_evidence(camera, height, width, positions, depth_epsilon=0.03, radius=6):
    """Exact procedural z-buffer built by splatting each point's OWN projected
    depth into its own (dilated) pixel footprint. Same "exact deterministic
    z-buffer" philosophy as Worklog 102's own `_procedural_evidence` (reused/
    imported above for the wall-gap fixtures), generalized from manual column
    segments to an arbitrary point cloud so it can also stand in for a real
    render for the fully-visible flat/curved/crease fixtures (A/B/G) without
    the CPU fallback renderer's documented blending imprecision.

    Two passes, deliberately in this order: (1) every point writes its OWN
    exact depth to its OWN exact rounded pixel first (nearest-wins only
    resolves genuine same-pixel collisions between two different points, it
    never lets a point's dilated neighbor overwrite that point's own pixel --
    otherwise a fine sample spacing below the camera's pixel resolution would
    let one point's disc silently overwrite an adjacent point's own true
    depth with a slightly different value, corrupting that point's own
    on-surface self-consistency check for no reason connected to any cut
    semantics being tested here); (2) only the pixels still unfilled after
    pass 1 are then filled by the nearest dilated point, purely to give the
    range-based screen-walk a continuous corridor between two already-
    self-consistent points.
    """

    dummy_view = CameraViewEvidence(
        camera_index=0, image_height=height, image_width=width,
        world_view_transform=camera.world_view_transform, full_proj_transform=camera.full_proj_transform,
        view_depth=torch.zeros(height, width), valid_depth_mask=torch.zeros(height, width, dtype=torch.bool),
        coverage_alpha=None, backend_source="fallback", coverage_kind="x", depth_kind="x", depth_is_approximate=False,
    )
    proj = _project_to_camera(positions, dummy_view)
    depth_buffer = torch.full((height, width), float("inf"))
    valid_buffer = torch.zeros((height, width), dtype=torch.bool)
    rows = proj["pixel_row"].round().long()
    cols = proj["pixel_col"].round().long()
    depths = proj["view_depth"]
    in_bounds = proj["in_bounds"]

    # Pass 1: exact self-writes only (no dilation) -- nearest-wins resolves
    # only genuine exact-pixel collisions between distinct points.
    for i in range(int(positions.shape[0])):
        if not bool(in_bounds[i]):
            continue
        r, c, d = int(rows[i]), int(cols[i]), float(depths[i])
        if d < float(depth_buffer[r, c]):
            depth_buffer[r, c] = d
        valid_buffer[r, c] = True

    # Pass 2: dilate into still-empty neighboring pixels only, for corridor
    # continuity between two already self-consistent points.
    if radius > 0:
        filled_depth = depth_buffer.clone()
        filled_valid = valid_buffer.clone()
        for i in range(int(positions.shape[0])):
            if not bool(in_bounds[i]):
                continue
            r, c, d = int(rows[i]), int(cols[i]), float(depths[i])
            r0, r1 = max(0, r - radius), min(height, r + radius + 1)
            c0, c1 = max(0, c - radius), min(width, c + radius + 1)
            empty = ~valid_buffer[r0:r1, c0:c1]
            patch = filled_depth[r0:r1, c0:c1]
            better = empty & (d < patch)
            filled_depth[r0:r1, c0:c1] = torch.where(better, torch.full_like(patch, d), patch)
            filled_valid[r0:r1, c0:c1] = filled_valid[r0:r1, c0:c1] | empty
        depth_buffer, valid_buffer = filled_depth, filled_valid
    view = CameraViewEvidence(
        camera_index=0, image_height=height, image_width=width,
        world_view_transform=camera.world_view_transform, full_proj_transform=camera.full_proj_transform,
        view_depth=depth_buffer, valid_depth_mask=valid_buffer, coverage_alpha=None,
        backend_source="fallback", coverage_kind="binary_contribution_mask", depth_kind="direct_linear",
        depth_is_approximate=False,
    )
    return ObservationEvidence(views=[view], near=1e-3, far=1e6, depth_epsilon=depth_epsilon, topology_version="test", camera_set_version="test")


_TEST_RESOLUTION = 256


# --------------------------------------------------------------------------
# A. flat fully visible surface, positively observed -> one component
# --------------------------------------------------------------------------


def test_flat_surface_with_positive_observation_is_one_component():
    positions = _grid((-0.5, 0.5), (-0.5, 0.5), 2.0, 0.1)
    orientation = _flat_orientation(positions)
    camera = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), height=_TEST_RESOLUTION, width=_TEST_RESOLUTION)
    evidence = _splatted_evidence(camera, _TEST_RESOLUTION, _TEST_RESOLUTION, positions)
    config = PositiveVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_positive_visible_adjacency(orientation, evidence, config)
    accounting = positive_visible_adjacency_accounting(result)
    assert accounting["visible_component_count"] == 1
    assert accounting["coverage_identity_holds"] is True
    assert accounting["relation_state_counts"][STATE_POSITIVE_VISIBLE_CONTINUATION] > 0


# --------------------------------------------------------------------------
# B. smooth curved fully visible surface, positively observed -> one component
# --------------------------------------------------------------------------


def test_curved_surface_with_positive_observation_remains_one_component():
    """Directive section 4/5 for this batch: the range-based screen-walk test
    (reused from Worklog 102) must not mistake ordinary curvature for a
    contradiction, so a fully, positively observed curved band stays one
    component -- not merely "not cut", but POSITIVELY connected."""

    orientation = _cylinder_band(n_theta=20, n_y=4, theta_span=math.pi / 2.0, pitch_y=0.05)
    resolution = 512
    # fovx tight / fovy wide: keeps the theta-direction pixel spacing above
    # the splat's exact-self-write resolution while keeping the y-direction
    # spacing small enough for a modest dilation radius to bridge -- both
    # axes' spacing are geometric consequences of this camera choice, not
    # independently tuned per axis.
    camera = _build_camera(_lookat_world_view((0.0, 0.075, -1.5), (0.0, 0.075, 1.5), (0.0, 1.0, 0.0)), fovx=0.5, fovy=1.8, height=resolution, width=resolution)
    evidence = _splatted_evidence(camera, resolution, resolution, orientation.positions, depth_epsilon=0.05, radius=4)
    config = PositiveVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_positive_visible_adjacency(orientation, evidence, config)
    accounting = positive_visible_adjacency_accounting(result)
    assert accounting["relation_state_counts"][STATE_CUT_OCCLUDED_DOMAIN] == 0
    assert accounting["relation_state_counts"][STATE_CUT_KNOWN_FREE_SPACE] == 0
    assert accounting["relation_state_counts"][STATE_POSITIVE_VISIBLE_CONTINUATION] > 0
    assert accounting["visible_component_count"] == 1


# --------------------------------------------------------------------------
# C. wall + occluder + wall -> two components, positive CUT_OCCLUDED_DOMAIN
# --------------------------------------------------------------------------


def test_wall_with_central_occluder_yields_two_components():
    orientation, count_a = _wall_pair_with_gap()
    evidence, _ = _wall_gap_wall_camera_evidence(count_a, orientation.positions, gap_depth=1.0)
    config = PositiveVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_positive_visible_adjacency(orientation, evidence, config)
    accounting = positive_visible_adjacency_accounting(result)
    assert accounting["visible_component_count"] >= 2
    assert accounting["relation_state_counts"][STATE_CUT_OCCLUDED_DOMAIN] > 0
    assert accounting["coverage_identity_holds"] is True
    wall_a_ids = result.subset_ids[:count_a].unique()
    wall_b_ids = result.subset_ids[count_a:].unique()
    assert not bool(torch.isin(wall_a_ids, wall_b_ids).any())


# --------------------------------------------------------------------------
# D. curved surface with an occluded gap -> stays two components
# --------------------------------------------------------------------------


def test_curved_surface_with_occluded_gap_remains_two_components():
    full = _cylinder_band(n_theta=40, theta_span=math.pi / 2.0)
    theta_values = torch.linspace(0.0, math.pi / 2.0, 40)
    gap_mask = (theta_values > math.pi / 2.0 * 0.42) & (theta_values < math.pi / 2.0 * 0.58)
    keep_theta = ~gap_mask
    keep_mask = keep_theta.unsqueeze(1).repeat(1, 10).reshape(-1)
    positions = full.positions[keep_mask]
    normals = full.surface_normal[keep_mask]
    tangent_u = full.tangent_axis_u[keep_mask]
    tangent_v = full.tangent_axis_v[keep_mask]
    orientation = _Orientation(positions, normals, tangent_u, tangent_v)
    count_before_gap = int((keep_theta[: gap_mask.nonzero()[0].item()]).sum().item()) * 10

    occluder = _grid((-0.05, 0.05), (-0.1, 0.6), 0.6, 0.04)
    camera = _build_camera(_lookat_world_view((0.0, 0.25, -1.5), (0.0, 0.25, 1.5), (0.0, 1.0, 0.0)), fovx=1.8, fovy=1.8, height=_TEST_RESOLUTION, width=_TEST_RESOLUTION)
    all_positions = torch.cat([orientation.positions, occluder], dim=0)
    evidence = _splatted_evidence(camera, _TEST_RESOLUTION, _TEST_RESOLUTION, all_positions, depth_epsilon=0.03)

    config = PositiveVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_positive_visible_adjacency(orientation, evidence, config)
    accounting = positive_visible_adjacency_accounting(result)
    assert accounting["visible_component_count"] >= 2
    assert accounting["coverage_identity_holds"] is True
    left_ids = result.subset_ids[:count_before_gap].unique()
    right_ids = result.subset_ids[-count_before_gap:].unique()
    assert not bool(torch.isin(left_ids, right_ids).any())


# --------------------------------------------------------------------------
# E. known free-space gap -> two components, positive CUT_KNOWN_FREE_SPACE
# --------------------------------------------------------------------------


def test_known_free_space_gap_separates_two_components():
    orientation, count_a = _wall_pair_with_gap()
    evidence, _ = _wall_gap_wall_camera_evidence(count_a, orientation.positions, gap_depth=5.0)
    config = PositiveVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_positive_visible_adjacency(orientation, evidence, config)
    accounting = positive_visible_adjacency_accounting(result)
    assert accounting["visible_component_count"] >= 2
    assert accounting["relation_state_counts"][STATE_CUT_KNOWN_FREE_SPACE] > 0
    wall_a_ids = result.subset_ids[:count_a].unique()
    wall_b_ids = result.subset_ids[count_a:].unique()
    assert not bool(torch.isin(wall_a_ids, wall_b_ids).any())


# --------------------------------------------------------------------------
# F. positively co-observed pair dominated by a normal-direction offset ->
#    demoted by the secondary positional-sheet gate (directive section 7)
# --------------------------------------------------------------------------


def test_positively_observed_pair_with_normal_offset_is_cut_by_positional_gate():
    a = _grid((-0.3, 0.3), (-0.3, 0.3), 2.0, 0.1)
    b = a.clone()
    b[:, 2] += 0.2
    b[:, 0] += 0.03
    positions = torch.cat([a, b], dim=0)
    orientation = _flat_orientation(positions)
    camera = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), height=_TEST_RESOLUTION, width=_TEST_RESOLUTION)
    # Wide splat radius: this fixture deliberately probes the SECONDARY
    # geometric gate in isolation, so the camera evidence is built generously
    # (not adversarially thin) to guarantee at least some cross-sheet
    # candidate edges receive positive observation support before the
    # positional-offset test has a chance to demote them.
    evidence = _splatted_evidence(camera, _TEST_RESOLUTION, _TEST_RESOLUTION, positions, radius=4)
    config = PositiveVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_positive_visible_adjacency(orientation, evidence, config)
    accounting = positive_visible_adjacency_accounting(result)
    assert accounting["relation_state_counts"][STATE_CUT_POSITIONAL_SHEET_SEPARATION] > 0
    half = int(a.shape[0])
    a_ids = result.subset_ids[:half].unique()
    b_ids = result.subset_ids[half:].unique()
    assert not bool(torch.isin(a_ids, b_ids).any())


# --------------------------------------------------------------------------
# G. true sharp discontinuity, positively observed on both sides -> preserved
# --------------------------------------------------------------------------


def test_sharp_crease_is_preserved_when_positively_observed():
    side_a = _grid((-1.2, 0.0), (-0.5, 0.5), 2.0, 0.1)
    angle = math.radians(60.0)
    xs = torch.arange(0.0, 1.2 + 1e-6, 0.1)
    ys = torch.arange(-0.5, 0.5 + 1e-6, 0.1)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    xx, yy = xx.flatten(), yy.flatten()
    side_b = torch.stack([xx * math.cos(angle), yy, 2.0 - xx * math.sin(angle)], dim=1)
    positions = torch.cat([side_a, side_b], dim=0)
    normals = torch.cat([
        torch.tensor([[0.0, 0.0, -1.0]]).repeat(side_a.shape[0], 1),
        torch.tensor([[math.sin(angle), 0.0, math.cos(angle)]]).repeat(side_b.shape[0], 1),
    ])
    tangent_u = torch.cat([
        torch.tensor([[1.0, 0.0, 0.0]]).repeat(side_a.shape[0], 1),
        torch.tensor([[math.cos(angle), 0.0, -math.sin(angle)]]).repeat(side_b.shape[0], 1),
    ])
    tangent_v = torch.tensor([[0.0, 1.0, 0.0]]).repeat(positions.shape[0], 1)
    orientation = _Orientation(positions, normals, tangent_u, tangent_v)

    camera = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), fovx=1.8, fovy=1.8, height=_TEST_RESOLUTION, width=_TEST_RESOLUTION)
    evidence = _splatted_evidence(camera, _TEST_RESOLUTION, _TEST_RESOLUTION, positions, radius=3)
    config = PositiveVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_positive_visible_adjacency(orientation, evidence, config)
    accounting = positive_visible_adjacency_accounting(result)
    assert accounting["relation_state_counts"][STATE_POSITIVE_VISIBLE_CONTINUATION] > 0
    assert accounting["relation_state_counts"][STATE_CUT_VISIBLE_DISCONTINUITY] > 0
    top_two = float(result.subset_sizes[:2].sum()) / int(len(result))
    assert top_two > 0.75


# --------------------------------------------------------------------------
# H. spatially close pair, NO camera provides a positive relationship ->
#    must NOT become adjacent merely because nothing contradicts it
# --------------------------------------------------------------------------


def test_no_positive_observation_never_creates_adjacency():
    """Central inversion of this batch versus Worklog 102: with
    observation_evidence=None (no camera data at all), NOT CONTRADICTED must
    NOT silently become VISIBLE CONNECTED -- every spatially close pair stays
    UNKNOWN_NO_POSITIVE_OBSERVATION and every surfel is its own singleton
    component, even though the WL102 (connectivity-by-default) module would
    keep this exact same input as one component (see
    test_flat_surface_geometric_only_is_one_component)."""

    positions = _grid((-0.5, 0.5), (-0.5, 0.5), 2.0, 0.1)
    orientation = _flat_orientation(positions)
    config = PositiveVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_positive_visible_adjacency(orientation, None, config)
    accounting = positive_visible_adjacency_accounting(result)
    assert accounting["relation_state_counts"][STATE_POSITIVE_VISIBLE_CONTINUATION] == 0
    assert accounting["visible_component_count"] == accounting["input_surfel_count"]
    assert accounting["coverage_identity_holds"] is True


# --------------------------------------------------------------------------
# I. one camera positively observes, a second camera simply cannot observe
#    the relation at all -> absence of observation is not a contradiction,
#    the edge still connects on the one positive view alone
# --------------------------------------------------------------------------


def test_one_positive_view_plus_one_non_observing_view_still_connects():
    positions = _grid((-0.5, 0.5), (-0.5, 0.5), 2.0, 0.1)
    orientation = _flat_orientation(positions)
    camera_sees = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), height=_TEST_RESOLUTION, width=_TEST_RESOLUTION)
    evidence_sees = _splatted_evidence(camera_sees, _TEST_RESOLUTION, _TEST_RESOLUTION, positions)

    # A camera pointed the opposite way: the whole grid sits behind it, so
    # every sample is outside_valid_view -- genuinely no data, not a claim of
    # free space or occlusion.
    camera_blind = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)), height=_TEST_RESOLUTION, width=_TEST_RESOLUTION)
    view_blind = CameraViewEvidence(
        camera_index=1, image_height=_TEST_RESOLUTION, image_width=_TEST_RESOLUTION,
        world_view_transform=camera_blind.world_view_transform, full_proj_transform=camera_blind.full_proj_transform,
        view_depth=torch.zeros(_TEST_RESOLUTION, _TEST_RESOLUTION), valid_depth_mask=torch.zeros(_TEST_RESOLUTION, _TEST_RESOLUTION, dtype=torch.bool),
        coverage_alpha=None, backend_source="fallback", coverage_kind="x", depth_kind="x", depth_is_approximate=False,
    )
    combined = ObservationEvidence(
        views=[evidence_sees.views[0], view_blind], near=evidence_sees.near, far=evidence_sees.far,
        depth_epsilon=evidence_sees.depth_epsilon, topology_version="test", camera_set_version="test",
    )
    config = PositiveVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_positive_visible_adjacency(orientation, combined, config)
    accounting = positive_visible_adjacency_accounting(result)
    assert accounting["visible_component_count"] == 1
    assert accounting["relation_state_counts"][STATE_UNRESOLVED_CONFLICT] == 0


# --------------------------------------------------------------------------
# J. genuine multi-view conflict -> unresolved, never silently connected
# --------------------------------------------------------------------------


def test_conflicting_multi_view_evidence_is_unresolved_not_connected():
    orientation, count_a = _wall_pair_with_gap()
    # View 1: the gap reads a depth continuous with both walls (positive).
    evidence_positive, _ = _wall_gap_wall_camera_evidence(count_a, orientation.positions, gap_depth=2.05)
    # View 2 (SAME camera pose, different scene content): an occluder now
    # sits in the gap -- a genuine contradiction for the very same edges.
    evidence_negative, _ = _wall_gap_wall_camera_evidence(count_a, orientation.positions, gap_depth=1.0)
    view_negative = dataclasses.replace(evidence_negative.views[0], camera_index=1)
    combined = ObservationEvidence(
        views=[evidence_positive.views[0], view_negative], near=evidence_positive.near, far=evidence_positive.far,
        depth_epsilon=evidence_positive.depth_epsilon, topology_version="test", camera_set_version="test",
    )
    config = PositiveVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_positive_visible_adjacency(orientation, combined, config)
    accounting = positive_visible_adjacency_accounting(result)
    assert accounting["relation_state_counts"][STATE_UNRESOLVED_CONFLICT] > 0
    wall_a_ids = result.subset_ids[:count_a].unique()
    wall_b_ids = result.subset_ids[count_a:].unique()
    assert not bool(torch.isin(wall_a_ids, wall_b_ids).any())


# --------------------------------------------------------------------------
# coverage / determinism
# --------------------------------------------------------------------------


def test_every_surfel_receives_exactly_one_owner_and_none_is_dropped():
    orientation = _cylinder_band(theta_span=math.pi / 2.0)
    config = PositiveVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    accounting = positive_visible_adjacency_accounting(partition_positive_visible_adjacency(orientation, None, config))
    assert accounting["assigned_surfel_count"] == accounting["input_surfel_count"]
    assert accounting["unassigned_surfel_count"] == 0
    assert accounting["multiply_owned_surfel_count"] == 0
    assert accounting["coverage_identity_holds"] is True


def test_partition_is_deterministic_across_repeated_runs():
    orientation, count_a = _wall_pair_with_gap()
    evidence, _ = _wall_gap_wall_camera_evidence(count_a, orientation.positions, gap_depth=1.0)
    config = PositiveVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    first = partition_positive_visible_adjacency(orientation, evidence, config)
    second = partition_positive_visible_adjacency(orientation, evidence, config)
    assert torch.equal(first.subset_ids, second.subset_ids)
    assert torch.equal(first.relation_state, second.relation_state)


def test_empty_and_single_surfel_input_stay_coverage_exact():
    empty = _Orientation(torch.zeros((0, 3)), torch.zeros((0, 3)), torch.zeros((0, 3)), torch.zeros((0, 3)))
    config = PositiveVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    empty_accounting = positive_visible_adjacency_accounting(partition_positive_visible_adjacency(empty, None, config))
    assert empty_accounting["coverage_identity_holds"] is True

    single = _Orientation(
        torch.zeros((1, 3)), torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[0.0, 1.0, 0.0]])
    )
    single_accounting = positive_visible_adjacency_accounting(partition_positive_visible_adjacency(single, None, config))
    assert single_accounting["visible_component_count"] == 1
    assert single_accounting["coverage_identity_holds"] is True


def test_relation_state_set_matches_directive_contract():
    assert set(RELATION_STATES) == {
        STATE_POSITIVE_VISIBLE_CONTINUATION, STATE_CUT_KNOWN_FREE_SPACE, STATE_CUT_OCCLUDED_DOMAIN,
        STATE_CUT_VISIBLE_DISCONTINUITY, STATE_CUT_POSITIONAL_SHEET_SEPARATION, STATE_UNRESOLVED_CONFLICT,
        STATE_UNKNOWN_NO_POSITIVE_OBSERVATION,
    }
