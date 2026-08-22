from __future__ import annotations

import ast
import math
from pathlib import Path

import torch

from osn_gs.surface.torch_renderer_grounded_visible_adjacency import (
    RELATION_STATES,
    STATE_CUT_KNOWN_FREE_SPACE,
    STATE_CUT_OCCLUDED_DOMAIN,
    STATE_CUT_POSITIONAL_SHEET_SEPARATION,
    STATE_CUT_VISIBLE_DISCONTINUITY,
    STATE_POSITIVE_RENDERER_VISIBLE_CONTINUATION,
    STATE_UNKNOWN_NO_RENDERER_SUPPORTED_RELATION,
    STATE_UNRESOLVED_CONFLICT,
    RendererGroundedVisibleAdjacencyConfig,
    partition_renderer_grounded_visible_adjacency,
    renderer_grounded_visible_adjacency_accounting,
)
from osn_gs.surface.torch_observation_evidence import CameraViewEvidence, ObservationEvidence
from tests.test_maximal_visible_connectivity import (
    _LOCAL_CONFIG,
    _Orientation,
    _build_camera,
    _cylinder_band,
    _flat_orientation,
    _grid,
    _lookat_world_view,
    _wall_pair_with_gap,
)
from tests.test_positive_visible_adjacency import _splatted_evidence, _wall_gap_wall_camera_evidence


def _all_true(count: int) -> torch.Tensor:
    return torch.ones((count,), dtype=torch.bool)


def _all_false(count: int) -> torch.Tensor:
    return torch.zeros((count,), dtype=torch.bool)


# --------------------------------------------------------------------------
# module isolation: never re-derives Phase-C center classification
# --------------------------------------------------------------------------


def test_module_does_not_depend_on_phase_c_center_classification():
    path = Path(__file__).resolve().parent.parent / "osn_gs" / "surface" / "torch_renderer_grounded_visible_adjacency.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    forbidden = {"osn_gs.surface.torch_observation_evidence.classify_world_samples", "osn_gs.surface.torch_maximal_visible_connectivity._per_view_status_codes"}
    assert not (names & forbidden), f"must not import Phase-C center classification: {names & forbidden}"


# --------------------------------------------------------------------------
# central contract: renderer contribution, not Phase-C center, is the
# endpoint eligibility source
# --------------------------------------------------------------------------


def test_center_negative_but_contributing_endpoints_form_positive_adjacency():
    """This module has no notion of Phase-C center status at all -- a fully
    contributing mask (as Worklog 105 would report for surfels Worklog 103/
    104 called center-negative) is sufficient on its own to form adjacency."""

    positions = _grid((-0.5, 0.5), (-0.5, 0.5), 2.0, 0.1)
    orientation = _flat_orientation(positions)
    camera = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), height=256, width=256)
    evidence = _splatted_evidence(camera, 256, 256, positions)
    count = int(positions.shape[0])
    config = RendererGroundedVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_renderer_grounded_visible_adjacency(orientation, evidence, [_all_true(count)], config)
    accounting = renderer_grounded_visible_adjacency_accounting(result)
    assert accounting["visible_component_count"] == 1
    assert accounting["relation_state_counts"][STATE_POSITIVE_RENDERER_VISIBLE_CONTINUATION] > 0
    assert accounting["coverage_identity_holds"] is True


def test_projected_only_noncontributing_surfel_cannot_form_adjacency():
    """A surfel that never contributes (per the supplied mask) in any view
    cannot become an endpoint of a positive edge, even if the corridor
    between it and a contributing neighbor would otherwise read clean."""

    a = _grid((-0.3, 0.0), (-0.3, 0.3), 2.0, 0.1)
    b = _grid((0.02, 0.32), (-0.3, 0.3), 2.0, 0.1)
    positions = torch.cat([a, b], dim=0)
    orientation = _flat_orientation(positions)
    camera = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), height=256, width=256)
    evidence = _splatted_evidence(camera, 256, 256, positions)
    count = int(positions.shape[0])
    mask = _all_true(count)
    mask[int(a.shape[0]):] = False  # group b never contributes
    config = RendererGroundedVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_renderer_grounded_visible_adjacency(orientation, evidence, [mask], config)
    a_ids = result.subset_ids[: int(a.shape[0])].unique()
    b_ids = result.subset_ids[int(a.shape[0]):].unique()
    assert not bool(torch.isin(a_ids, b_ids).any())
    # b never even reaches the corridor test -- every incident edge stays UNKNOWN
    accounting = renderer_grounded_visible_adjacency_accounting(result)
    assert accounting["relation_state_counts"][STATE_UNKNOWN_NO_RENDERER_SUPPORTED_RELATION] > 0


def test_co_contributing_endpoints_separated_by_occluder_do_not_connect():
    """Directive section 9's explicit contract: positive contribution never
    bridges an occluded gap, even though both wall sides fully contribute."""

    orientation, count_a = _wall_pair_with_gap()
    evidence, _ = _wall_gap_wall_camera_evidence(count_a, orientation.positions, gap_depth=1.0)
    count = int(orientation.positions.shape[0])
    config = RendererGroundedVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_renderer_grounded_visible_adjacency(orientation, evidence, [_all_true(count)], config)
    accounting = renderer_grounded_visible_adjacency_accounting(result)
    assert accounting["visible_component_count"] >= 2
    assert accounting["relation_state_counts"][STATE_CUT_OCCLUDED_DOMAIN] > 0
    wall_a_ids = result.subset_ids[:count_a].unique()
    wall_b_ids = result.subset_ids[count_a:].unique()
    assert not bool(torch.isin(wall_a_ids, wall_b_ids).any())


def test_known_free_space_still_cuts_a_co_contributing_pair():
    orientation, count_a = _wall_pair_with_gap()
    evidence, _ = _wall_gap_wall_camera_evidence(count_a, orientation.positions, gap_depth=5.0)
    count = int(orientation.positions.shape[0])
    config = RendererGroundedVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_renderer_grounded_visible_adjacency(orientation, evidence, [_all_true(count)], config)
    accounting = renderer_grounded_visible_adjacency_accounting(result)
    assert accounting["relation_state_counts"][STATE_CUT_KNOWN_FREE_SPACE] > 0
    wall_a_ids = result.subset_ids[:count_a].unique()
    wall_b_ids = result.subset_ids[count_a:].unique()
    assert not bool(torch.isin(wall_a_ids, wall_b_ids).any())


def test_same_view_absence_is_not_negative_evidence():
    """One view has both endpoints co-contributing cleanly; a second view has
    NEITHER contributing at all (absence, not a camera claiming occlusion) --
    the pair must still connect on the strength of the one positive view."""

    positions = _grid((-0.5, 0.5), (-0.5, 0.5), 2.0, 0.1)
    orientation = _flat_orientation(positions)
    camera = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), height=256, width=256)
    evidence_positive = _splatted_evidence(camera, 256, 256, positions)
    count = int(positions.shape[0])

    blind_view = CameraViewEvidence(
        camera_index=1, image_height=256, image_width=256,
        world_view_transform=evidence_positive.views[0].world_view_transform,
        full_proj_transform=evidence_positive.views[0].full_proj_transform,
        view_depth=torch.zeros(256, 256), valid_depth_mask=torch.zeros(256, 256, dtype=torch.bool),
        coverage_alpha=None, backend_source="fallback", coverage_kind="x", depth_kind="x", depth_is_approximate=False,
    )
    combined = ObservationEvidence(
        views=[evidence_positive.views[0], blind_view], near=evidence_positive.near, far=evidence_positive.far,
        depth_epsilon=evidence_positive.depth_epsilon, topology_version="test", camera_set_version="test",
    )
    masks = [_all_true(count), _all_false(count)]  # second view: nobody contributes (absence)
    config = RendererGroundedVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_renderer_grounded_visible_adjacency(orientation, combined, masks, config)
    accounting = renderer_grounded_visible_adjacency_accounting(result)
    assert accounting["visible_component_count"] == 1
    assert accounting["relation_state_counts"][STATE_UNRESOLVED_CONFLICT] == 0


def test_true_conflict_stays_unresolved_not_connected():
    orientation, count_a = _wall_pair_with_gap()
    evidence_positive, _ = _wall_gap_wall_camera_evidence(count_a, orientation.positions, gap_depth=2.05)
    evidence_negative, _ = _wall_gap_wall_camera_evidence(count_a, orientation.positions, gap_depth=1.0)
    import dataclasses

    view_negative = dataclasses.replace(evidence_negative.views[0], camera_index=1)
    combined = ObservationEvidence(
        views=[evidence_positive.views[0], view_negative], near=evidence_positive.near, far=evidence_positive.far,
        depth_epsilon=evidence_positive.depth_epsilon, topology_version="test", camera_set_version="test",
    )
    count = int(orientation.positions.shape[0])
    masks = [_all_true(count), _all_true(count)]
    config = RendererGroundedVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_renderer_grounded_visible_adjacency(orientation, combined, masks, config)
    accounting = renderer_grounded_visible_adjacency_accounting(result)
    assert accounting["relation_state_counts"][STATE_UNRESOLVED_CONFLICT] > 0
    wall_a_ids = result.subset_ids[:count_a].unique()
    wall_b_ids = result.subset_ids[count_a:].unique()
    assert not bool(torch.isin(wall_a_ids, wall_b_ids).any())


def test_curved_co_contributing_surface_remains_connected():
    orientation = _cylinder_band(n_theta=20, n_y=4, theta_span=math.pi / 2.0, pitch_y=0.05)
    resolution = 512
    camera = _build_camera(_lookat_world_view((0.0, 0.075, -1.5), (0.0, 0.075, 1.5), (0.0, 1.0, 0.0)), fovx=0.5, fovy=1.8, height=resolution, width=resolution)
    evidence = _splatted_evidence(camera, resolution, resolution, orientation.positions, depth_epsilon=0.05, radius=4)
    count = int(orientation.positions.shape[0])
    config = RendererGroundedVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_renderer_grounded_visible_adjacency(orientation, evidence, [_all_true(count)], config)
    accounting = renderer_grounded_visible_adjacency_accounting(result)
    assert accounting["relation_state_counts"][STATE_CUT_OCCLUDED_DOMAIN] == 0
    assert accounting["relation_state_counts"][STATE_CUT_KNOWN_FREE_SPACE] == 0
    assert accounting["visible_component_count"] == 1


def test_nearby_separate_sheets_remain_separated_by_positional_gate():
    a = _grid((-0.3, 0.3), (-0.3, 0.3), 2.0, 0.1)
    b = a.clone()
    b[:, 2] += 0.2
    b[:, 0] += 0.03
    positions = torch.cat([a, b], dim=0)
    orientation = _flat_orientation(positions)
    camera = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), height=256, width=256)
    evidence = _splatted_evidence(camera, 256, 256, positions, radius=4)
    count = int(positions.shape[0])
    config = RendererGroundedVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_renderer_grounded_visible_adjacency(orientation, evidence, [_all_true(count)], config)
    accounting = renderer_grounded_visible_adjacency_accounting(result)
    assert accounting["relation_state_counts"][STATE_CUT_POSITIONAL_SHEET_SEPARATION] > 0
    half = int(a.shape[0])
    a_ids = result.subset_ids[:half].unique()
    b_ids = result.subset_ids[half:].unique()
    assert not bool(torch.isin(a_ids, b_ids).any())


# --------------------------------------------------------------------------
# primitive visibility vs graph degree independence (directive section 15)
# --------------------------------------------------------------------------


def test_primitive_visibility_and_graph_degree_remain_independent():
    """This module exposes no per-node visibility flag at all -- component
    size is purely an ADJACENCY result. A far-isolated point (no spatial
    neighbor at all) still receives exactly one owner (itself), and the
    module never claims anything about whether that point is or is not
    'renderer visible' -- that judgement belongs to Worklog 105's own
    per-node accounting, not to this module."""

    positions = _grid((-0.2, 0.2), (-0.2, 0.2), 2.0, 0.1)
    isolated = torch.tensor([[500.0, 500.0, 2.0]])
    all_positions = torch.cat([positions, isolated], dim=0)
    orientation = _flat_orientation(all_positions)
    camera = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), height=256, width=256)
    evidence = _splatted_evidence(camera, 256, 256, positions)  # only the main cluster is ever mentioned
    count = int(all_positions.shape[0])
    config = RendererGroundedVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_renderer_grounded_visible_adjacency(orientation, evidence, [_all_true(count)], config)
    accounting = renderer_grounded_visible_adjacency_accounting(result)
    assert accounting["coverage_identity_holds"] is True
    isolated_subset = int(result.subset_ids[-1])
    assert int(result.subset_sizes[isolated_subset]) == 1


# --------------------------------------------------------------------------
# coverage / determinism
# --------------------------------------------------------------------------


def test_every_surfel_receives_exactly_one_owner_and_none_is_dropped():
    orientation = _cylinder_band(theta_span=math.pi / 2.0)
    config = RendererGroundedVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    accounting = renderer_grounded_visible_adjacency_accounting(partition_renderer_grounded_visible_adjacency(orientation, None, None, config))
    assert accounting["assigned_surfel_count"] == accounting["input_surfel_count"]
    assert accounting["unassigned_surfel_count"] == 0
    assert accounting["coverage_identity_holds"] is True


def test_no_evidence_at_all_never_creates_adjacency():
    positions = _grid((-0.5, 0.5), (-0.5, 0.5), 2.0, 0.1)
    orientation = _flat_orientation(positions)
    config = RendererGroundedVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_renderer_grounded_visible_adjacency(orientation, None, None, config)
    accounting = renderer_grounded_visible_adjacency_accounting(result)
    assert accounting["relation_state_counts"][STATE_POSITIVE_RENDERER_VISIBLE_CONTINUATION] == 0
    assert accounting["visible_component_count"] == accounting["input_surfel_count"]


def test_partition_is_deterministic_across_repeated_runs():
    orientation, count_a = _wall_pair_with_gap()
    evidence, _ = _wall_gap_wall_camera_evidence(count_a, orientation.positions, gap_depth=1.0)
    count = int(orientation.positions.shape[0])
    masks = [_all_true(count)]
    config = RendererGroundedVisibleAdjacencyConfig(local=_LOCAL_CONFIG)
    first = partition_renderer_grounded_visible_adjacency(orientation, evidence, masks, config)
    second = partition_renderer_grounded_visible_adjacency(orientation, evidence, masks, config)
    assert torch.equal(first.subset_ids, second.subset_ids)
    assert torch.equal(first.relation_state, second.relation_state)


def test_relation_state_set_matches_directive_contract():
    assert set(RELATION_STATES) == {
        STATE_POSITIVE_RENDERER_VISIBLE_CONTINUATION, STATE_CUT_KNOWN_FREE_SPACE, STATE_CUT_OCCLUDED_DOMAIN,
        STATE_CUT_VISIBLE_DISCONTINUITY, STATE_CUT_POSITIONAL_SHEET_SEPARATION, STATE_UNRESOLVED_CONFLICT,
        STATE_UNKNOWN_NO_RENDERER_SUPPORTED_RELATION,
    }
