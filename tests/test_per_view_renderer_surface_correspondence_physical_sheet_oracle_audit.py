from __future__ import annotations

import numpy as np

import devtools.demo.per_view_renderer_surface_correspondence_physical_sheet_oracle_audit as audit
from devtools.demo.oracle_single_surface_support_appearance_evidence import ManualCameraMask


class _Camera:
    def __init__(self, name: str = "camera") -> None:
        self.image_name = name
        self.image_width = 8
        self.image_height = 6
        self.world_view_transform = np.eye(4, dtype=np.float64)
        self.full_proj_transform = np.eye(4, dtype=np.float64)
        self.camera_center = np.zeros(3, dtype=np.float64)


def _full_mask(name: str = "camera") -> ManualCameraMask:
    return ManualCameraMask(
        name,
        ((-10.0, -10.0), (658.0, -10.0), (658.0, 430.0), (-10.0, 430.0)),
        "synthetic full-frame polygon",
    )


def _cloud(name: str, points: np.ndarray) -> audit.PerViewEventCloud:
    points = np.asarray(points, dtype=np.float64)
    normals = np.tile(np.asarray([[0.0, 0.0, 1.0]]), (len(points), 1))
    return audit.PerViewEventCloud(
        name,
        _full_mask(name),
        np.arange(len(points), dtype=np.int64),
        np.zeros(len(points), dtype=np.int64),
        points[:, 2].copy(),
        points.copy(),
        normals,
    )


def test_per_view_event_cloud_contains_only_valid_frozen_polygon_pixels() -> None:
    camera = _Camera()
    mask = _full_mask()
    depth = np.ones((6, 8), dtype=np.float64)
    depth[0, 0] = 0.0
    depth[5, 7] = np.nan

    cloud, summary = audit._build_per_view_event_cloud(camera, mask, depth)

    assert summary["valid_polygon_pixel_count"] == 48
    assert summary["valid_median_event_count"] == 46
    assert np.all((cloud.pixel_x >= 0) & (cloud.pixel_x < camera.image_width))
    assert np.all((cloud.pixel_y >= 0) & (cloud.pixel_y < camera.image_height))
    projection = audit._renderer_projected_pixels(cloud.points, camera)
    assert np.allclose(projection["x"], cloud.pixel_x)
    assert np.allclose(projection["y"], cloud.pixel_y)
    assert np.allclose(projection["depth"], cloud.median_depth)


def test_renderer_depth_reconstruction_round_trips_exactly() -> None:
    camera = _Camera()
    pixel_x = np.asarray([0.0, 3.0, 7.0])
    pixel_y = np.asarray([0.0, 2.0, 5.0])
    depth = np.asarray([0.8, 1.0, 2.0])

    points = audit._reconstruct_world_from_renderer_pixel_depth(pixel_x, pixel_y, depth, camera)
    projection = audit._renderer_projected_pixels(points, camera)

    assert np.allclose(projection["x"], pixel_x)
    assert np.allclose(projection["y"], pixel_y)
    assert np.allclose(projection["depth"], depth)
    assert np.all(projection["valid"])


def test_pairwise_metrics_are_continuous_and_do_not_select_or_mutate_clouds() -> None:
    first = _cloud("c1", [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]])
    second = _cloud("c2", [[0.0, 0.0, 1.1], [1.0, 0.0, 1.1], [2.0, 0.0, 1.1]])
    first_before = first.points.copy()
    second_before = second.points.copy()

    report, arrays = audit._pairwise_surface_agreement(first, second, h=0.1, mu=0.3)

    assert report["threshold_or_membership_use"] is False
    assert report["reciprocal"]["world_distance"]["status"] == "MEASURED"
    assert np.isclose(report["reciprocal"]["world_distance"]["median"], 0.1)
    assert arrays["first_to_second_distance"].shape == (3,)
    assert np.array_equal(first.points, first_before)
    assert np.array_equal(second.points, second_before)


def test_wl127_attribution_is_diagnostic_and_preserves_three_ranked_distances() -> None:
    row_ids = np.asarray([17, 23], dtype=np.int64)
    points = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    clouds = [
        _cloud("c1", points),
        _cloud("c2", points + np.asarray([0.0, 0.0, 0.1])),
        _cloud("c3", points + np.asarray([0.0, 0.0, 0.2])),
    ]

    report, arrays = audit._wl127_mask_only_attribution(row_ids, points, clouds, h=0.1, mu=0.3)

    assert report["diagnostic_only"] is True
    assert report["exact_point_identity_required"] is False
    assert report["selection_or_membership_use"] is False
    assert report["wl127_mask_only_point_count"] == 2
    assert arrays["sorted_distance_to_camera_clouds"].shape == (2, 3)
    assert report["nearest_camera_cloud"]["world_distance"]["median"] == 0.0


def test_overlap_accounting_uses_only_frozen_h_and_mu_reference_radii() -> None:
    points = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    clouds = [
        _cloud("c1", points),
        _cloud("c2", points + np.asarray([0.0, 0.0, 0.05])),
        _cloud("c3", points + np.asarray([0.0, 0.0, 0.25])),
    ]

    report, _arrays = audit._overlap_accounting(clouds, h=0.1, mu=0.3)

    assert report["c1"]["h"]["descriptive_reference_only"] is True
    assert report["c1"]["mu"]["descriptive_reference_only"] is True
    assert report["c1"]["continuous_distribution_precedes_reference_accounting"] is True
    assert report["c1"]["h"]["shared_by_another_count"] == 2
    assert report["c1"]["h"]["shared_by_both_other_cameras_count"] == 0
