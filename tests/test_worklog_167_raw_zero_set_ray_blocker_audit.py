from __future__ import annotations

import numpy as np
import torch

from devtools.demo.worklog_167_raw_zero_set_ray_blocker_audit import (
    REL_BEHIND_ZEROSET_SURFACE,
    REL_IN_FRONT_OF_ZEROSET_SURFACE,
    REL_NO_DECISION,
    REL_ZEROSET_FIRST_SURFACE,
    STATUS_AMBIGUOUS,
    STATUS_HIT,
    STATUS_NO_HIT,
    AnalyticSurface,
    RayBundle,
    _component_labels,
    _plane_surface,
    _query_ladder,
    _ray_triangle_batch,
    _sphere_surface,
    intersect_first_hit_bruteforce,
    intersect_first_hit_screen_index,
    ScreenTileIndex,
    _real_camera_rays,
    _roi_pixels,
    _query_relation,
)


def _rays(origin=(0.0, 0.0, -2.0), directions=((0.0, 0.0, 1.0),)) -> RayBundle:
    directions = np.asarray(directions, dtype=np.float64)
    origins = np.repeat(np.asarray(origin, dtype=np.float64).reshape(1, 3), len(directions), axis=0)
    return RayBundle(origins, directions, np.ones((len(directions),), dtype=np.float64))


def test_two_sided_ray_triangle_returns_positive_hit_and_barycentric() -> None:
    vertices = np.asarray([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    result = intersect_first_hit_bruteforce(_rays(), vertices, faces)
    assert result.status.tolist() == [STATUS_HIT]
    assert result.depth[0] == 2.0
    np.testing.assert_allclose(result.world_xyz[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(result.barycentric[0], [0.25, 0.25, 0.5])
    assert result.valid_positive_depth_intersections.tolist() == [1]


def test_first_hit_orders_by_depth_then_triangle_id_deterministically() -> None:
    vertices = np.asarray(
        [[-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0],
         [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    result_a = intersect_first_hit_bruteforce(_rays(), vertices, faces)
    result_b = intersect_first_hit_bruteforce(_rays(), vertices, faces)
    assert result_a.triangle_id.tolist() == [1]
    np.testing.assert_array_equal(result_a.triangle_id, result_b.triangle_id)
    assert result_a.valid_positive_depth_intersections.tolist() == [2]


def test_degenerate_triangle_is_not_a_valid_hit() -> None:
    vertices = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    result = intersect_first_hit_bruteforce(_rays(), vertices, faces)
    assert result.status.tolist() == [STATUS_NO_HIT]
    assert result.valid_positive_depth_intersections.tolist() == [0]


def test_exact_coplanar_ray_is_ambiguous_fail_closed() -> None:
    vertices = np.asarray([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    result = intersect_first_hit_bruteforce(_rays(origin=(0.0, 0.0, 0.0), directions=((1.0, 0.0, 0.0),)), vertices, faces)
    assert result.status.tolist() == [STATUS_AMBIGUOUS]
    assert result.coplanar_ambiguity_count.tolist() == [1]


def test_analytic_plane_and_sphere_controls_have_expected_first_hits() -> None:
    plane = _plane_surface("plane")
    sphere = _sphere_surface()
    plane_depth, plane_hit, _ = plane.intersect(_rays())
    sphere_depth, sphere_hit, _ = sphere.intersect(_rays())
    assert plane_hit.tolist() == [True]
    assert plane_depth.tolist() == [2.0]
    assert sphere_hit.tolist() == [True]
    assert np.isclose(sphere_depth[0], 1.35)


def test_query_ladder_uses_exact_surface_point_and_fail_closed_relation() -> None:
    vertices = np.asarray([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    rays = _rays()
    result = intersect_first_hit_bruteforce(rays, vertices, faces, np.asarray([7], dtype=np.int64))
    ladder = _query_ladder(rays, result, 0.5)
    assert [row["label"] for row in ladder] == ["Q_before", "Q_surface", "Q_behind"]
    assert [row["relation"] for row in ladder] == [REL_IN_FRONT_OF_ZEROSET_SURFACE, REL_ZEROSET_FIRST_SURFACE, REL_BEHIND_ZEROSET_SURFACE]
    np.testing.assert_allclose(ladder[1]["query_world_xyz"], result.world_xyz[0])
    assert all(not row["surface_membership_threshold_used"] for row in ladder)


def test_no_hit_relation_is_no_decision() -> None:
    rays = _rays()
    result = intersect_first_hit_bruteforce(
        rays,
        np.asarray([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        np.asarray([[0, 1, 2]], dtype=np.int64),
    )
    no_hit = intersect_first_hit_bruteforce(
        _rays(directions=((3.0, 3.0, 1.0),)),
        np.asarray([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        np.asarray([[0, 1, 2]], dtype=np.int64),
    )
    assert result.status.tolist() == [STATUS_HIT]
    assert no_hit.status.tolist() == [STATUS_NO_HIT]
    assert np.isnan(no_hit.depth[0])
    assert _query_relation(2.0, no_hit, 0) == REL_NO_DECISION


def test_frozen_roi_sampler_returns_aligned_region_masks() -> None:
    pixels, masks, bounds = _roi_pixels("DSC08003.JPG", 648, 420, 4)
    assert pixels.ndim == 2 and pixels.shape[1] == 2
    assert all(mask.shape == (len(pixels),) for mask in masks.values())
    assert bounds[0] <= bounds[2] and bounds[1] <= bounds[3]


def test_component_labels_preserve_disconnected_fragments() -> None:
    vertices = np.zeros((6, 3), dtype=np.float64)
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    labels, face_labels, sizes = _component_labels(faces, len(vertices))
    assert len(sizes) == 2
    assert len(np.unique(face_labels)) == 2
    assert sorted(sizes[:, 0].tolist()) == [3, 3]
    assert labels.shape == (6,)


def test_raw_batch_matches_first_hit_contract_for_multiple_rays() -> None:
    triangles = np.asarray(
        [[[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 1.0, 1.0]],
         [[-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0]]],
        dtype=np.float64,
    )
    rays = _rays(directions=((0.0, 0.0, 1.0), (3.0, 3.0, 1.0)))
    count, best_t, best_face, bary, coplanar = _ray_triangle_batch(rays.origins, rays.directions, triangles, np.asarray([10, 11], dtype=np.int64))
    assert count.tolist() == [2, 0]
    assert best_t.tolist() == [3.0, np.inf]
    assert best_face.tolist() == [10, -1]
    assert np.all(np.isfinite(bary[0]))
    assert coplanar.tolist() == [0, 0]


def test_screen_broad_phase_projects_world_points_once() -> None:
    """Regression for the row-vector full-projection convention."""
    camera = type("Camera", (), {
        "image_width": 4,
        "image_height": 4,
        "FoVx": 0.7,
        "FoVy": 0.7,
        "world_view_transform": torch.eye(4),
        "full_proj_transform": torch.eye(4),
        "camera_center": torch.zeros(3),
    })()
    vertices = np.asarray([[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    pixels = np.asarray([[1, 1]], dtype=np.int64)
    rays = _real_camera_rays(camera, pixels)
    index = ScreenTileIndex.build(vertices, faces, camera, (0, 0, 3, 3), tile_size=16)
    result = intersect_first_hit_screen_index(rays, vertices, faces, camera, index)
    assert result.status.tolist() == [STATUS_HIT]
    assert result.triangle_id.tolist() == [0]
