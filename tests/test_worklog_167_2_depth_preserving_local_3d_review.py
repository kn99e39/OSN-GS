from __future__ import annotations

import numpy as np

from devtools.demo.worklog_167_2_depth_preserving_local_3d_review import (
    _project,
    derive_inspection_bounds,
    select_ray_indices,
    triangles_in_aabb,
)


def test_select_ray_indices_is_fixed_row_major_and_bounded() -> None:
    indices = np.arange(100, dtype=np.int64)
    selected = select_ray_indices(indices, maximum=32)
    assert len(selected) == 25
    assert np.array_equal(selected, indices[::4])
    assert np.array_equal(select_ray_indices(np.arange(5), maximum=32), np.arange(5))


def test_inspection_bounds_cover_camera_and_all_query_points() -> None:
    camera = np.array([0.0, 0.0, 0.0])
    hits = np.array([[2.0, 1.0, 3.0], [2.5, 1.5, 3.5]])
    behind = hits + np.array([0.1, 0.0, 0.0])
    lower, upper, padding = derive_inspection_bounds(camera, hits, (behind,))
    population = np.vstack((camera, hits, behind))
    assert np.all(lower <= population.min(axis=0))
    assert np.all(upper >= population.max(axis=0))
    assert np.all(padding >= 0.05)


def test_triangles_in_aabb_keeps_every_intersecting_raw_face() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [2.0, 2.0, 2.0], [3.0, 2.0, 2.0], [2.0, 3.0, 2.0],
            [-2.0, -2.0, -2.0], [-1.0, -2.0, -2.0], [-2.0, -1.0, -2.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.int64)
    ids, triangles = triangles_in_aabb(vertices, faces, np.array([-0.1, -0.1, -0.1]), np.array([1.1, 1.1, 1.1]), chunk_size=1)
    assert np.array_equal(ids, np.array([0]))
    assert triangles.shape == (1, 3, 3)


def test_depth_ordered_side_projection_preserves_ray_depth_order() -> None:
    camera = np.zeros(3)
    target = np.array([0.0, 0.0, 5.0])
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])
    coords, _, label = _project(points, "side", camera, target, np.vstack((camera, points)))
    assert "depth" in label
    assert np.all(np.diff(coords[:, 0]) > 0)
