from __future__ import annotations

import numpy as np
import torch

from devtools.demo.worklog_165_historical_candidate_b_arbitrary_point_occlusion_sufficiency_audit import (
    IMAGE,
    _camera_ray_grid,
    _categorical_map,
    _fixture_accounting,
    _plane_surface,
    _rotation_to_quaternion,
    _sphere_surface,
    _surface_samples,
)
from scripts.devtools.observed_occluded import candidate_b_median_depth as candidate_b
from scripts.devtools.observed_occluded.shared import (
    RELEVANCE_OK,
    STATE_OBSERVED,
    STATE_OCCLUDED,
    ViewGeometry,
)


def _ray_grid(direction: tuple[float, float, float]) -> object:
    direction_array = np.asarray(direction, dtype=np.float64).reshape(1, 1, 3)
    return type("RayGridTest", (), {
        "origin": np.asarray((0.0, 0.0, -4.0), dtype=np.float64),
        "direction": direction_array,
        "camera_depth": np.ones((1, 1), dtype=np.float64),
        "pixel_x": np.zeros((1, 1), dtype=np.float64),
        "pixel_y": np.zeros((1, 1), dtype=np.float64),
    })()


def test_front_parallel_plane_exact_intersection() -> None:
    hit = _plane_surface("plane") .intersect(_ray_grid((0.0, 0.0, 1.0)))
    assert bool(hit.hit[0, 0])
    assert float(hit.depth[0, 0]) == 4.0
    assert float(hit.local_u[0, 0]) == 0.0
    assert float(hit.local_v[0, 0]) == 0.0


def test_oblique_plane_intersection_is_camera_space_depth() -> None:
    hit = _plane_surface("oblique", oblique=True).intersect(_ray_grid((0.0, 0.0, 1.0)))
    assert bool(hit.hit[0, 0])
    assert np.isfinite(hit.depth[0, 0])
    assert float(hit.depth[0, 0]) > 0.0


def test_sphere_uses_first_positive_root() -> None:
    hit = _sphere_surface().intersect(_ray_grid((0.0, 0.0, 1.0)))
    assert bool(hit.hit[0, 0])
    assert np.isclose(float(hit.depth[0, 0]), 3.35, atol=1e-6)


def test_surface_sampling_is_deterministic_and_density_is_explicit() -> None:
    surface = _plane_surface("plane")
    coarse_a = _surface_samples(surface, "coarse")
    coarse_b = _surface_samples(surface, "coarse")
    dense = _surface_samples(surface, "dense")
    np.testing.assert_array_equal(coarse_a[0], coarse_b[0])
    np.testing.assert_array_equal(coarse_a[1], coarse_b[1])
    assert coarse_a[0].shape[0] == 81
    assert dense[0].shape[0] == 289
    assert dense[0].shape[0] > coarse_a[0].shape[0]


def test_frame_quaternion_round_trip_identity_and_oblique() -> None:
    identity = _rotation_to_quaternion(np.eye(3, dtype=np.float64))
    assert np.isclose(np.linalg.norm(identity), 1.0)
    oblique = _plane_surface("oblique", oblique=True)
    frame = np.column_stack((oblique.tangent_u, oblique.tangent_v, oblique.normal))
    quaternion = _rotation_to_quaternion(frame)
    assert np.isclose(np.linalg.norm(quaternion), 1.0)


def test_candidate_b_frozen_ordering_is_reused_without_epsilon() -> None:
    geometry = ViewGeometry(
        pixel_x=torch.tensor([0.0]), pixel_y=torch.tensor([0.0]),
        pixel_col=torch.tensor([0]), pixel_row=torch.tensor([0]),
        pixel_index=torch.tensor([0]), depth=torch.tensor([3.0]),
        relevant=torch.tensor([True]), relevance_code=torch.tensor([RELEVANCE_OK], dtype=torch.int8),
    )
    result = candidate_b.classify_view(geometry, torch.tensor([2.0]))
    assert int(result["states"][0]) == STATE_OCCLUDED
    geometry.depth[:] = 2.0
    result = candidate_b.classify_view(geometry, torch.tensor([2.0]))
    assert int(result["states"][0]) == STATE_OBSERVED


def test_accounting_keeps_no_hit_and_exact_equality_separate() -> None:
    class Hit:
        depth = np.asarray([[2.0, 3.0]])
        hit = np.asarray([[True, False]])
        silhouette = np.asarray([[False, False]])

    accounting = _fixture_accounting(Hit(), np.asarray([[2.0, 2.5]]))
    assert accounting["blocker_hit_ordering"]["m_eq_z_star_exact_float_equality"] == 1
    assert accounting["no_hit_valid_median_count"] == 1
    assert accounting["z_star_minus_m_distribution_for_m_lt_z_star"]["count"] == 0


def test_categorical_map_does_not_hide_invalid_rays() -> None:
    class Hit:
        depth = np.full((IMAGE, IMAGE), np.nan)
        hit = np.zeros((IMAGE, IMAGE), dtype=bool)
        silhouette = np.zeros((IMAGE, IMAGE), dtype=bool)

    image, masks = _categorical_map(Hit(), np.zeros((IMAGE, IMAGE), dtype=np.float64))
    assert image.shape == (IMAGE, IMAGE, 3)
    assert int(masks["other_invalid"].sum()) == IMAGE * IMAGE
