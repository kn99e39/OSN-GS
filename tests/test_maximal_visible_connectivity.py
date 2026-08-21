from __future__ import annotations

import math

import torch

from osn_gs.data.colmap_scene import projection_matrix
from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.render.gaussian_rasterizer import GaussianRasterizerConfig, OSNGaussianRasterizer
from osn_gs.render.torch_fallback import TorchCamera
from osn_gs.surface.torch_coverage_first_subset_partition import CoverageFirstPartitionConfig
from osn_gs.surface.torch_observation_evidence import (
    VIEW_STATUS_BEHIND_FIRST_OBSERVED_SURFACE,
    VIEW_STATUS_KNOWN_FREE_SPACE,
    VIEW_STATUS_ON_OBSERVED_SURFACE,
    VIEW_STATUS_OUTSIDE_VALID_VIEW,
    VIEW_STATUS_UNOBSERVED,
    CameraViewEvidence,
    ObservationEvidence,
    build_observation_evidence,
    classify_world_samples,
)
from osn_gs.surface.torch_maximal_visible_connectivity import (
    CUT_KNOWN_FREE_SPACE,
    CUT_OCCLUDED_DOMAIN,
    CUT_POSITIONAL_SHEET_SEPARATION,
    CUT_VISIBLE_GEOMETRIC_DISCONTINUITY,
    MaximalVisibleConnectivityConfig,
    _per_view_status_codes,
    _project_to_camera,
    maximal_visible_connectivity_accounting,
    partition_maximal_visible_components,
)


class _Orientation:
    def __init__(self, positions: torch.Tensor, normals: torch.Tensor, tangent_u: torch.Tensor, tangent_v: torch.Tensor):
        self.positions = positions
        self.surface_normal = normals
        self.tangent_axis_u = tangent_u
        self.tangent_axis_v = tangent_v
        self.gaussian_ids = torch.arange(int(positions.shape[0]))


def _lookat_world_view(eye, target, up):
    eye_t = torch.tensor(eye, dtype=torch.float32)
    target_t = torch.tensor(target, dtype=torch.float32)
    up_t = torch.tensor(up, dtype=torch.float32)
    forward = torch.nn.functional.normalize(target_t - eye_t, dim=0, eps=1e-8)
    right = torch.nn.functional.normalize(torch.cross(forward, up_t, dim=0), dim=0, eps=1e-8)
    true_up = torch.cross(right, forward, dim=0)
    rotation = torch.stack([right, true_up, forward], dim=0)
    translation = -(rotation @ eye_t)
    world_view = torch.eye(4, dtype=torch.float32)
    world_view[:3, :3] = rotation
    world_view[:3, 3] = translation
    return world_view.transpose(0, 1).contiguous(), eye_t


def _build_camera(world_view_and_center, fovx=1.4, fovy=1.4, height=128, width=128):
    world_view, camera_center = world_view_and_center
    projection = projection_matrix(0.01, 100.0, fovx, fovy, device="cpu").transpose(0, 1).contiguous()
    full_proj = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    return TorchCamera(
        image_height=height, image_width=width, world_view_transform=world_view, full_proj_transform=full_proj,
        camera_center=camera_center, FoVx=fovx, FoVy=fovy,
    )


def _grid(x_range, y_range, z, pitch):
    xs = torch.arange(x_range[0], x_range[1] + 1e-6, pitch)
    ys = torch.arange(y_range[0], y_range[1] + 1e-6, pitch)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    zz = torch.full_like(xx, z)
    return torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)


def _flat_orientation(positions: torch.Tensor, normal=(0.0, 0.0, -1.0)) -> _Orientation:
    count = int(positions.shape[0])
    normals = torch.tensor([list(normal)], dtype=torch.float32).repeat(count, 1)
    tangent_u = torch.tensor([[1.0, 0.0, 0.0]]).repeat(count, 1)
    tangent_v = torch.tensor([[0.0, 1.0, 0.0]]).repeat(count, 1)
    return _Orientation(positions, normals, tangent_u, tangent_v)


def _procedural_evidence(camera, height: int, width: int, column_depth_segments, depth_epsilon=0.03) -> ObservationEvidence:
    """Builds an exact, deterministic per-column depth map for one camera --
    used ONLY for the occlusion/free-space gap fixtures (C, E), where the CPU
    fallback renderer's blended-weight Gaussian approximation (documented in
    tests/test_observation_evidence.py as having "no real depth-ordered
    occlusion") is not precise enough to place a gap's observed evidence at
    an exact pixel column without also bleeding into the neighbouring wall's
    own boundary points (verified empirically while writing these fixtures).
    `column_depth_segments` is `[(col_min, col_max, depth), ...]` in
    NEAREST-WINS order semantics (a column covered by more than one segment
    takes the smallest depth) -- this is a literal, exact z-buffer, not an
    approximation, and still produces the SAME `CameraViewEvidence` /
    `ObservationEvidence` dataclasses the canonical Phase-C query consumes.
    """

    depth_row = torch.full((width,), float("inf"))
    valid_row = torch.zeros((width,), dtype=torch.bool)
    for col_min, col_max, depth in column_depth_segments:
        # col_min/col_max are treated as INCLUSIVE integer pixel columns
        # (callers pass already-rounded values); +1 on the upper bound only
        # to convert to Python's exclusive slice-end convention.
        lo = max(0, int(round(col_min)))
        hi = min(width, int(round(col_max)) + 1)
        if lo >= hi:
            continue
        segment_depth = torch.full((hi - lo,), float(depth))
        better = segment_depth < depth_row[lo:hi]
        depth_row[lo:hi] = torch.where(better, segment_depth, depth_row[lo:hi])
        valid_row[lo:hi] = valid_row[lo:hi] | True
    view_depth = depth_row.unsqueeze(0).repeat(height, 1)
    valid_depth_mask = valid_row.unsqueeze(0).repeat(height, 1)
    view = CameraViewEvidence(
        camera_index=0, image_height=height, image_width=width,
        world_view_transform=camera.world_view_transform, full_proj_transform=camera.full_proj_transform,
        view_depth=view_depth, valid_depth_mask=valid_depth_mask, coverage_alpha=None,
        backend_source="fallback", coverage_kind="binary_contribution_mask", depth_kind="direct_linear",
        depth_is_approximate=False,
    )
    return ObservationEvidence(views=[view], near=1e-3, far=1e6, depth_epsilon=depth_epsilon, topology_version="test", camera_set_version="test")


def _evidence_for(positions: torch.Tensor, cameras: list, opacities=None, scales=None, depth_epsilon=0.02):
    model = TorchGaussianModel(sh_degree=0, device="cpu")
    if scales is None:
        scales = torch.full((positions.shape[0], 3), 0.02)
    model.initialize(positions=positions, colors=torch.full((positions.shape[0], 3), 0.5), scales=scales, opacities=opacities)
    rasterizer = OSNGaussianRasterizer(GaussianRasterizerConfig(prefer_cuda=False, allow_fallback=True))
    return build_observation_evidence(cameras, model, rasterizer, depth_epsilon=depth_epsilon)


_LOCAL_CONFIG = CoverageFirstPartitionConfig(neighbor_count=8, spatial_connect_spacing_multiplier=2.0)


# --------------------------------------------------------------------------
# canonical Phase-C equivalence (directive: do not create a second system)
# --------------------------------------------------------------------------


def test_vectorized_per_view_classification_matches_canonical_classify_world_samples():
    wall = _grid((-0.5, 0.5), (-0.5, 0.5), 2.0, 0.1)
    occluder = _grid((2.4, 2.6), (-0.1, 0.1), 1.0, 0.1)
    positions = torch.cat([wall, occluder], dim=0)
    opacities = torch.cat([torch.full((wall.shape[0], 1), 0.3), torch.full((occluder.shape[0], 1), 0.95)])
    camera = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
    evidence = _evidence_for(positions, [camera], opacities=opacities)

    samples = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0], [50.0, 0.0, 1.0]])
    canonical = classify_world_samples(evidence, samples)
    canonical_statuses = [record.per_view[0].status for record in canonical]

    view = evidence.views[0]
    proj = _project_to_camera(samples, view)
    codes = _per_view_status_codes(
        proj["view_depth"], proj["observed_depth"], proj["valid_at_pixel"], proj["in_bounds"], evidence.near, evidence.far, evidence.depth_epsilon
    )
    name_map = {
        0: VIEW_STATUS_OUTSIDE_VALID_VIEW, 1: VIEW_STATUS_UNOBSERVED, 2: VIEW_STATUS_KNOWN_FREE_SPACE,
        3: VIEW_STATUS_ON_OBSERVED_SURFACE, 4: VIEW_STATUS_BEHIND_FIRST_OBSERVED_SURFACE,
    }
    vectorized_statuses = [name_map[int(code)] for code in codes.tolist()]
    assert vectorized_statuses == canonical_statuses


# --------------------------------------------------------------------------
# A. flat fully visible surface -> one component
# --------------------------------------------------------------------------


def test_flat_surface_geometric_only_is_one_component():
    orientation = _flat_orientation(_grid((-0.5, 0.5), (-0.5, 0.5), 2.0, 0.1))
    config = MaximalVisibleConnectivityConfig(local=_LOCAL_CONFIG)
    result = partition_maximal_visible_components(orientation, None, config)
    accounting = maximal_visible_connectivity_accounting(result)
    assert accounting["visible_component_count"] == 1
    assert accounting["coverage_identity_holds"] is True


def test_flat_surface_with_clean_unoccluded_camera_stays_one_component():
    positions = _grid((-0.5, 0.5), (-0.5, 0.5), 2.0, 0.1)
    orientation = _flat_orientation(positions)
    camera = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
    evidence = _evidence_for(positions, [camera])
    config = MaximalVisibleConnectivityConfig(local=_LOCAL_CONFIG)
    result = partition_maximal_visible_components(orientation, evidence, config)
    accounting = maximal_visible_connectivity_accounting(result)
    assert accounting["visible_component_count"] == 1
    assert accounting["observation_evaluated_edge_count"] > 0
    assert accounting["boundary_cut_reason_counts"][CUT_OCCLUDED_DOMAIN] == 0
    assert accounting["boundary_cut_reason_counts"][CUT_KNOWN_FREE_SPACE] == 0


# --------------------------------------------------------------------------
# B. smooth curved fully visible surface -> one component (directive section 5)
# --------------------------------------------------------------------------


def _cylinder_band(n_theta=40, n_y=10, radius=0.3, theta_span=math.pi, pitch_y=0.05):
    theta = torch.linspace(0.0, theta_span, n_theta)
    y = torch.arange(n_y, dtype=torch.float32) * pitch_y
    tt, yy = torch.meshgrid(theta, y, indexing="ij")
    tt, yy = tt.reshape(-1), yy.reshape(-1)
    positions = torch.stack([radius * torch.cos(tt), yy, radius * torch.sin(tt) + 1.5], dim=1)
    normals = torch.stack([torch.cos(tt), torch.zeros_like(tt), torch.sin(tt)], dim=1)
    tangent_u = torch.stack([-torch.sin(tt), torch.zeros_like(tt), torch.cos(tt)], dim=1)
    tangent_v = torch.tensor([[0.0, 1.0, 0.0]]).repeat(positions.shape[0], 1)
    return _Orientation(positions, normals, tangent_u, tangent_v)


def test_curved_surface_geometric_only_remains_one_component():
    orientation = _cylinder_band()
    config = MaximalVisibleConnectivityConfig(local=_LOCAL_CONFIG)
    result = partition_maximal_visible_components(orientation, None, config)
    accounting = maximal_visible_connectivity_accounting(result)
    assert accounting["visible_component_count"] == 1
    assert float(result.normal_gradient_magnitude.median()) > 1.0  # real curvature, not a trivial flat case


def test_curved_surface_with_camera_does_not_falsely_trigger_occlusion_from_curvature():
    """The central directive-section-5 contract: a camera observing the
    WHOLE curved band unoccluded must never classify its own local candidate
    edges as CUT_OCCLUDED_DOMAIN / CUT_KNOWN_FREE_SPACE purely because a
    straight screen-space/chord depth guess would disagree with the true
    curve -- this module's range-based test (not a point guess) must stay
    silent here."""

    orientation = _cylinder_band(theta_span=math.pi / 2.0)
    camera = _build_camera(_lookat_world_view((0.0, 0.25, -1.5), (0.0, 0.25, 1.5), (0.0, 1.0, 0.0)), fovx=1.8, fovy=1.8)
    evidence = _evidence_for(orientation.positions, [camera], depth_epsilon=0.03)
    config = MaximalVisibleConnectivityConfig(local=_LOCAL_CONFIG)
    result = partition_maximal_visible_components(orientation, evidence, config)
    accounting = maximal_visible_connectivity_accounting(result)
    assert accounting["observation_evaluated_edge_count"] > 0
    assert accounting["boundary_cut_reason_counts"][CUT_OCCLUDED_DOMAIN] == 0
    assert accounting["boundary_cut_reason_counts"][CUT_KNOWN_FREE_SPACE] == 0
    assert accounting["visible_component_count"] == 1


# --------------------------------------------------------------------------
# C. visible wall with central occluder -> TWO visible components
# --------------------------------------------------------------------------


def _wall_pair_with_gap(gap_half_width=0.05, pitch=0.05):
    wall_a = _grid((-1.0, -gap_half_width), (-0.5, 0.5), 2.0, pitch)
    wall_b = _grid((gap_half_width, 1.0), (-0.5, 0.5), 2.0, pitch)
    positions = torch.cat([wall_a, wall_b], dim=0)
    return _flat_orientation(positions), int(wall_a.shape[0])


_TEST_RESOLUTION = 256


def _wall_gap_wall_camera_evidence(count_a: int, wall_positions: torch.Tensor, gap_depth: float):
    """A z-buffer-exact `ObservationEvidence` for one camera looking straight
    at `wall_positions` (two flat wall groups either side of a gap, both at
    z=2): wall_a's own column range and wall_b's own column range each read
    z=2 (their true depth); the GAP's own column range (computed from the
    same wall plane, so it exactly spans the visual gap, no bleed) reads
    `gap_depth` -- 1.0 for a nearer occluder (fixture C), 5.0 for a farther
    background confirming free space (fixture E)."""

    camera = _build_camera(
        _lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        fovx=1.6, fovy=1.6, height=_TEST_RESOLUTION, width=_TEST_RESOLUTION,
    )
    dummy_view = CameraViewEvidence(
        camera_index=0, image_height=_TEST_RESOLUTION, image_width=_TEST_RESOLUTION,
        world_view_transform=camera.world_view_transform, full_proj_transform=camera.full_proj_transform,
        view_depth=torch.zeros(_TEST_RESOLUTION, _TEST_RESOLUTION),
        valid_depth_mask=torch.zeros(_TEST_RESOLUTION, _TEST_RESOLUTION, dtype=torch.bool),
        coverage_alpha=None, backend_source="fallback", coverage_kind="x", depth_kind="x", depth_is_approximate=False,
    )
    wall_a_x = wall_positions[:count_a, 0]
    wall_b_x = wall_positions[count_a:, 0]
    # Project the ACTUAL boundary points (not synthetic corners) so the
    # segment boundaries land on exactly the same rounded pixel columns the
    # real per-surfel classification will use.
    corners = torch.tensor([
        [wall_a_x.min().item(), 0.0, 2.0], [wall_a_x.max().item(), 0.0, 2.0],
        [wall_b_x.min().item(), 0.0, 2.0], [wall_b_x.max().item(), 0.0, 2.0],
    ])
    proj = _project_to_camera(corners, dummy_view)
    # corners = [wall_a.min, wall_a.max, wall_b.min, wall_b.max] in WORLD x;
    # wall_a.max and wall_b.min are each wall's own GAP-FACING edge (the gap
    # sits between them, whichever pixel-column direction that maps to).
    rounded_cols = [int(round(c)) for c in proj["pixel_col"].tolist()]
    wall_a_col_min, wall_a_col_max = min(rounded_cols[0], rounded_cols[1]), max(rounded_cols[0], rounded_cols[1])
    wall_b_col_min, wall_b_col_max = min(rounded_cols[2], rounded_cols[3]), max(rounded_cols[2], rounded_cols[3])
    # The gap sits strictly BETWEEN the two walls' own gap-facing rounded
    # columns (rounded_cols[1], rounded_cols[2]) -- never touching either,
    # so a wall's own boundary point's rounded pixel is unambiguously part
    # of ITS wall segment, never contested by the gap segment's smaller
    # depth (both segments now use INCLUSIVE integer column ranges, matching
    # the same `.round()` convention `_per_view_status_codes` uses).
    gap_edge_a, gap_edge_b = sorted((rounded_cols[1], rounded_cols[2]))
    gap_col_min, gap_col_max = gap_edge_a + 1, gap_edge_b - 1
    segments = [
        (wall_a_col_min, wall_a_col_max, 2.0),
        (wall_b_col_min, wall_b_col_max, 2.0),
        (gap_col_min, gap_col_max, gap_depth),
    ]
    return _procedural_evidence(camera, _TEST_RESOLUTION, _TEST_RESOLUTION, segments, depth_epsilon=0.03), camera


def test_wall_with_central_occluder_yields_two_visible_components():
    orientation, count_a = _wall_pair_with_gap()
    evidence, _ = _wall_gap_wall_camera_evidence(count_a, orientation.positions, gap_depth=1.0)

    config = MaximalVisibleConnectivityConfig(local=_LOCAL_CONFIG)
    result = partition_maximal_visible_components(orientation, evidence, config)
    accounting = maximal_visible_connectivity_accounting(result)

    assert accounting["visible_component_count"] >= 2
    assert accounting["boundary_cut_reason_counts"][CUT_OCCLUDED_DOMAIN] > 0
    assert accounting["coverage_identity_holds"] is True
    wall_a_ids = result.subset_ids[:count_a].unique()
    wall_b_ids = result.subset_ids[count_a:].unique()
    assert not bool(torch.isin(wall_a_ids, wall_b_ids).any())


# --------------------------------------------------------------------------
# D. same underlying curved surface with occluded gap -> stays separate
# --------------------------------------------------------------------------


def test_curved_surface_with_occluded_gap_remains_two_visible_components():
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

    # occluder placed between the camera and the gap region only
    occluder = _grid((-0.05, 0.05), (-0.1, 0.6), 0.6, 0.04)
    all_positions = torch.cat([orientation.positions, occluder], dim=0)
    opacities = torch.cat([
        torch.full((orientation.positions.shape[0], 1), 0.3), torch.full((occluder.shape[0], 1), 0.97),
    ])
    camera = _build_camera(_lookat_world_view((0.0, 0.25, -1.5), (0.0, 0.25, 1.5), (0.0, 1.0, 0.0)), fovx=1.8, fovy=1.8)
    evidence = _evidence_for(all_positions, [camera], opacities=opacities, depth_epsilon=0.03)

    config = MaximalVisibleConnectivityConfig(local=_LOCAL_CONFIG)
    result = partition_maximal_visible_components(orientation, evidence, config)
    accounting = maximal_visible_connectivity_accounting(result)
    assert accounting["visible_component_count"] >= 2
    assert accounting["coverage_identity_holds"] is True
    left_ids = result.subset_ids[:count_before_gap].unique()
    right_ids = result.subset_ids[-count_before_gap:].unique()
    assert not bool(torch.isin(left_ids, right_ids).any())


# --------------------------------------------------------------------------
# E. known free-space gap -> no surface connection
# --------------------------------------------------------------------------


def test_known_free_space_gap_separates_two_visible_components():
    orientation, count_a = _wall_pair_with_gap()
    evidence, _ = _wall_gap_wall_camera_evidence(count_a, orientation.positions, gap_depth=5.0)

    config = MaximalVisibleConnectivityConfig(local=_LOCAL_CONFIG)
    result = partition_maximal_visible_components(orientation, evidence, config)
    accounting = maximal_visible_connectivity_accounting(result)
    assert accounting["visible_component_count"] >= 2
    assert accounting["boundary_cut_reason_counts"][CUT_KNOWN_FREE_SPACE] > 0
    wall_a_ids = result.subset_ids[:count_a].unique()
    wall_b_ids = result.subset_ids[count_a:].unique()
    assert not bool(torch.isin(wall_a_ids, wall_b_ids).any())


# --------------------------------------------------------------------------
# F. nearby parallel visible sheets -> remain separate
# --------------------------------------------------------------------------


def test_parallel_sheets_remain_separate():
    a = _grid((-0.5, 0.5), (-0.5, 0.5), 2.0, 0.1)
    b = a.clone()
    b[:, 2] += 0.2
    b[:, 0] += 0.03
    positions = torch.cat([a, b], dim=0)
    orientation = _flat_orientation(positions)
    config = MaximalVisibleConnectivityConfig(local=_LOCAL_CONFIG)
    result = partition_maximal_visible_components(orientation, None, config)
    accounting = maximal_visible_connectivity_accounting(result)
    assert accounting["visible_component_count"] >= 2
    assert accounting["boundary_cut_reason_counts"][CUT_POSITIONAL_SHEET_SEPARATION] > 0
    half = int(a.shape[0])
    a_ids = result.subset_ids[:half].unique()
    b_ids = result.subset_ids[half:].unique()
    assert not bool(torch.isin(a_ids, b_ids).any())


# --------------------------------------------------------------------------
# G. true visible sharp discontinuity -> preserved
# --------------------------------------------------------------------------


def test_sharp_crease_is_preserved_as_a_visible_boundary():
    side_a = _grid((-1.2, 0.0), (-0.5, 0.5), 2.0, 0.1)
    angle = math.radians(90.0)
    xs = torch.arange(0.0, 1.2 + 1e-6, 0.1)
    ys = torch.arange(-0.5, 0.5 + 1e-6, 0.1)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    xx, yy = xx.flatten(), yy.flatten()
    side_b = torch.stack([xx * math.cos(angle), yy, 2.0 + xx * math.sin(angle)], dim=1)
    positions = torch.cat([side_a, side_b], dim=0)
    normals = torch.cat([
        torch.tensor([[0.0, 0.0, -1.0]]).repeat(side_a.shape[0], 1),
        torch.tensor([[math.sin(angle), 0.0, -math.cos(angle)]]).repeat(side_b.shape[0], 1),
    ])
    tangent_u = torch.cat([
        torch.tensor([[1.0, 0.0, 0.0]]).repeat(side_a.shape[0], 1),
        torch.tensor([[math.cos(angle), 0.0, math.sin(angle)]]).repeat(side_b.shape[0], 1),
    ])
    tangent_v = torch.tensor([[0.0, 1.0, 0.0]]).repeat(positions.shape[0], 1)
    orientation = _Orientation(positions, normals, tangent_u, tangent_v)

    config = MaximalVisibleConnectivityConfig(local=_LOCAL_CONFIG)
    result = partition_maximal_visible_components(orientation, None, config)
    accounting = maximal_visible_connectivity_accounting(result)
    assert accounting["visible_component_count"] >= 2
    assert accounting["boundary_cut_reason_counts"][CUT_VISIBLE_GEOMETRIC_DISCONTINUITY] > 0
    top_two = float(result.subset_sizes[:2].sum()) / int(len(result))
    assert top_two > 0.85


# --------------------------------------------------------------------------
# H. unobserved/unknown gap WITHOUT positive occlusion evidence -> must NOT
#    be treated as a positive occluded surface (directive section 4)
# --------------------------------------------------------------------------


def test_unobserved_gap_without_positive_evidence_does_not_force_a_cut():
    """No background object behind the gap, no occluder in front: the camera
    simply has no data for the gap region (never renders anything there).
    Absence of evidence must not manufacture a cut -- geometric connectivity
    (both walls flat, same normal, within local-spacing range) is retained."""

    orientation, count_a = _wall_pair_with_gap(gap_half_width=0.05, pitch=0.05)
    camera = _build_camera(_lookat_world_view((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), fovx=1.6, fovy=1.6)
    evidence = _evidence_for(orientation.positions, [camera], depth_epsilon=0.03)

    config = MaximalVisibleConnectivityConfig(local=_LOCAL_CONFIG)
    result = partition_maximal_visible_components(orientation, evidence, config)
    accounting = maximal_visible_connectivity_accounting(result)
    assert accounting["boundary_cut_reason_counts"][CUT_KNOWN_FREE_SPACE] == 0
    assert accounting["boundary_cut_reason_counts"][CUT_OCCLUDED_DOMAIN] == 0
    assert accounting["visible_component_count"] == 1
    assert accounting["coverage_identity_holds"] is True


# --------------------------------------------------------------------------
# coverage / determinism
# --------------------------------------------------------------------------


def test_every_surfel_receives_exactly_one_owner_and_none_is_dropped():
    orientation = _cylinder_band(theta_span=math.pi / 2.0)
    config = MaximalVisibleConnectivityConfig(local=_LOCAL_CONFIG)
    accounting = maximal_visible_connectivity_accounting(partition_maximal_visible_components(orientation, None, config))
    assert accounting["assigned_surfel_count"] == accounting["input_surfel_count"]
    assert accounting["unassigned_surfel_count"] == 0
    assert accounting["multiply_owned_surfel_count"] == 0
    assert accounting["coverage_identity_holds"] is True


def test_partition_is_deterministic_across_repeated_runs():
    orientation, _ = _wall_pair_with_gap()
    config = MaximalVisibleConnectivityConfig(local=_LOCAL_CONFIG)
    first = partition_maximal_visible_components(orientation, None, config)
    second = partition_maximal_visible_components(orientation, None, config)
    assert torch.equal(first.subset_ids, second.subset_ids)
    assert torch.equal(first.cut_mask, second.cut_mask)


def test_empty_and_single_surfel_input_stay_coverage_exact():
    empty = _Orientation(torch.zeros((0, 3)), torch.zeros((0, 3)), torch.zeros((0, 3)), torch.zeros((0, 3)))
    config = MaximalVisibleConnectivityConfig(local=_LOCAL_CONFIG)
    empty_accounting = maximal_visible_connectivity_accounting(partition_maximal_visible_components(empty, None, config))
    assert empty_accounting["coverage_identity_holds"] is True

    single = _Orientation(
        torch.zeros((1, 3)), torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[0.0, 1.0, 0.0]])
    )
    single_accounting = maximal_visible_connectivity_accounting(partition_maximal_visible_components(single, None, config))
    assert single_accounting["visible_component_count"] == 1
    assert single_accounting["coverage_identity_holds"] is True
