from __future__ import annotations

import math

import torch

from osn_gs.surface.torch_camera_induced_visible_adjacency import (
    REASON_GEOMETRIC_DISCONTINUITY,
    REASON_POSITIONAL_SHEET_SEPARATION,
    CameraInducedAdjacencyConfig,
    camera_induced_visible_adjacency_accounting,
    partition_camera_induced_visible_adjacency,
)
from tests.test_maximal_visible_connectivity import _LOCAL_CONFIG, _Orientation, _grid


def _flat_two_group_orientation(gap: float = 0.05, pitch: float = 0.1):
    """Two small flat groups of points, close enough in 3D to be spatial
    candidates, sharing one normal field."""

    a = _grid((-0.3, 0.0), (-0.3, 0.3), 2.0, pitch)
    b = _grid((gap, gap + 0.3), (-0.3, 0.3), 2.0, pitch)
    positions = torch.cat([a, b], dim=0)
    count = int(positions.shape[0])
    normals = torch.tensor([[0.0, 0.0, -1.0]]).repeat(count, 1)
    tangent_u = torch.tensor([[1.0, 0.0, 0.0]]).repeat(count, 1)
    tangent_v = torch.tensor([[0.0, 1.0, 0.0]]).repeat(count, 1)
    return _Orientation(positions, normals, tangent_u, tangent_v), int(a.shape[0])


def _rep_map(height: int, width: int, cell_ids: list[list[int]]) -> torch.Tensor:
    return torch.tensor(cell_ids, dtype=torch.int64).reshape(height, width)


def test_image_space_adjacent_representatives_generate_edge():
    orientation, count_a = _flat_two_group_orientation()
    a_id, b_id = count_a - 7, count_a  # A's gap-facing column, B's gap-facing column, same row
    rep = _rep_map(1, 2, [[a_id, b_id]])
    config = CameraInducedAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_camera_induced_visible_adjacency(orientation, [rep], config)
    accounting = camera_induced_visible_adjacency_accounting(result)
    assert accounting["final_positive_edge_count"] > 0
    assert bool(result.subset_ids[a_id] == result.subset_ids[b_id])


def test_view_dependent_occlusion_does_not_veto_other_view_positive():
    """Directive's central correction B: one view positively observes A--B
    adjacency; a second view simply never mentions A or B at all (absence,
    not a camera claiming a contradiction). The pair must still connect --
    there is no conflict/veto mechanism in this architecture at all."""

    orientation, count_a = _flat_two_group_orientation()
    a_id, b_id = count_a - 7, count_a
    view_positive = _rep_map(1, 2, [[a_id, b_id]])
    view_absent = _rep_map(1, 2, [[-1, -1]])  # nothing observed at all
    config = CameraInducedAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_camera_induced_visible_adjacency(orientation, [view_positive, view_absent], config)
    assert bool(result.subset_ids[a_id] == result.subset_ids[b_id])


def test_globally_occluded_gap_remains_disconnected():
    """No view ever shows A and B as image-adjacent representatives (every
    view has an occluder/gap between them) -> they must remain separate
    components, even though they are 3D-local candidates."""

    orientation, count_a = _flat_two_group_orientation()
    a_id, b_id = count_a - 7, count_a
    occluder_id = 1  # a third, unrelated point from group A acting as a "gap" filler
    view1 = _rep_map(1, 3, [[a_id, occluder_id, b_id]])  # A and B never directly adjacent
    view2 = _rep_map(1, 2, [[-1, -1]])
    config = CameraInducedAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_camera_induced_visible_adjacency(orientation, [view1, view2], config)
    assert bool(result.subset_ids[a_id] != result.subset_ids[b_id])


def test_locality_filter_rejects_non_local_pairs():
    """Two surfels far apart in 3D (never a spatial candidate edge) but
    hand-constructed to appear image-adjacent -- the 3D-locality restriction
    (directive section 6) must reject this, never generating topology from
    image adjacency alone."""

    near = _grid((-0.1, 0.1), (-0.1, 0.1), 2.0, 0.1)
    far = torch.tensor([[500.0, 500.0, 2.0]])
    positions = torch.cat([near, far], dim=0)
    count = int(positions.shape[0])
    orientation = _Orientation(
        positions, torch.tensor([[0.0, 0.0, -1.0]]).repeat(count, 1),
        torch.tensor([[1.0, 0.0, 0.0]]).repeat(count, 1), torch.tensor([[0.0, 1.0, 0.0]]).repeat(count, 1),
    )
    far_id = count - 1
    rep = _rep_map(1, 2, [[0, far_id]])
    config = CameraInducedAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_camera_induced_visible_adjacency(orientation, [rep], config)
    accounting = camera_induced_visible_adjacency_accounting(result)
    assert accounting["locality_rejected_pair_count"] > 0
    assert accounting["final_positive_edge_count"] == 0
    assert bool(result.subset_ids[0] != result.subset_ids[far_id])


def test_curved_visible_surface_remains_connected():
    """A smooth curved band (large total normal rotation, no true
    discontinuity) with image-space adjacency generated along its whole
    length must stay one component -- normal rotation alone is never a cut."""

    n_theta = 12
    theta = torch.linspace(0.0, math.pi / 2.0, n_theta)
    radius = 0.3
    positions = torch.stack([radius * torch.cos(theta), torch.zeros(n_theta), radius * torch.sin(theta) + 1.5], dim=1)
    normals = torch.stack([torch.cos(theta), torch.zeros(n_theta), torch.sin(theta)], dim=1)
    tangent_u = torch.stack([-torch.sin(theta), torch.zeros(n_theta), torch.cos(theta)], dim=1)
    tangent_v = torch.tensor([[0.0, 1.0, 0.0]]).repeat(n_theta, 1)
    orientation = _Orientation(positions, normals, tangent_u, tangent_v)
    local_config = _LOCAL_CONFIG.__class__(neighbor_count=4, spatial_connect_spacing_multiplier=3.0)
    rep = _rep_map(1, n_theta, [list(range(n_theta))])  # sequential representative along the band
    config = CameraInducedAdjacencyConfig(local=local_config)
    result = partition_camera_induced_visible_adjacency(orientation, [rep], config)
    accounting = camera_induced_visible_adjacency_accounting(result)
    assert accounting["visible_component_count"] == 1


def test_nearby_separate_sheets_remain_separated_by_existing_geometry():
    """Two parallel sheets close in 3D but offset mostly along the shared
    normal direction: even if image-space adjacency is (hypothetically)
    generated between them, the reused secondary positional-sheet gate must
    still separate them."""

    a = _grid((-0.3, 0.3), (-0.3, 0.3), 2.0, 0.1)
    b = a.clone()
    b[:, 2] += 0.2
    b[:, 0] += 0.03
    positions = torch.cat([a, b], dim=0)
    count = int(positions.shape[0])
    orientation = _Orientation(
        positions, torch.tensor([[0.0, 0.0, -1.0]]).repeat(count, 1),
        torch.tensor([[1.0, 0.0, 0.0]]).repeat(count, 1), torch.tensor([[0.0, 1.0, 0.0]]).repeat(count, 1),
    )
    half = int(a.shape[0])
    rep = _rep_map(1, 2, [[0, half]])
    config = CameraInducedAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_camera_induced_visible_adjacency(orientation, [rep], config)
    accounting = camera_induced_visible_adjacency_accounting(result)
    assert accounting["geometric_rejection_reason_counts"][REASON_POSITIONAL_SHEET_SEPARATION] > 0
    assert bool(result.subset_ids[0] != result.subset_ids[half])


def test_deterministic_multi_view_union():
    orientation, count_a = _flat_two_group_orientation()
    a_id, b_id = count_a - 7, count_a
    views = [_rep_map(1, 2, [[a_id, b_id]]), _rep_map(1, 2, [[-1, -1]])]
    config = CameraInducedAdjacencyConfig(local=_LOCAL_CONFIG)
    first = partition_camera_induced_visible_adjacency(orientation, views, config)
    second = partition_camera_induced_visible_adjacency(orientation, views, config)
    assert torch.equal(first.subset_ids, second.subset_ids)
    assert torch.equal(first.positive_visible_edges, second.positive_visible_edges)


def test_no_views_never_creates_adjacency():
    orientation, count_a = _flat_two_group_orientation()
    config = CameraInducedAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_camera_induced_visible_adjacency(orientation, None, config)
    accounting = camera_induced_visible_adjacency_accounting(result)
    assert accounting["final_positive_edge_count"] == 0
    assert accounting["coverage_identity_holds"] is True


def test_coverage_identity_holds_with_real_pairs():
    orientation, count_a = _flat_two_group_orientation()
    a_id, b_id = count_a - 7, count_a
    rep = _rep_map(1, 2, [[a_id, b_id]])
    config = CameraInducedAdjacencyConfig(local=_LOCAL_CONFIG)
    result = partition_camera_induced_visible_adjacency(orientation, [rep], config)
    accounting = camera_induced_visible_adjacency_accounting(result)
    assert accounting["coverage_identity_holds"] is True
    assert accounting["assigned_surfel_count"] == accounting["input_surfel_count"]
