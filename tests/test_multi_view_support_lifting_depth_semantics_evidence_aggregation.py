from __future__ import annotations

import numpy as np

import devtools.demo.multi_view_support_lifting_depth_semantics_evidence_aggregation as demo
from devtools.demo.meeting_occluded_surface_feasibility import Box
from devtools.demo.oracle_single_surface_support_appearance_evidence import (
    ManualCameraMask,
    OracleSurfaceControl,
)
from devtools.demo.physical_chart_surface_representative import RepresentativeCaseConfig


class _Camera:
    def __init__(self, name: str = "cam") -> None:
        self.image_name = name
        self.image_width = 8
        self.image_height = 6
        self.world_view_transform = np.eye(4, dtype=np.float64)
        self.full_proj_transform = np.eye(4, dtype=np.float64)
        self.camera_center = np.zeros(3, dtype=np.float64)


def _control() -> OracleSurfaceControl:
    config = RepresentativeCaseConfig(
        name="synthetic",
        semantic_label="synthetic",
        roi_box=Box((-1.0, -1.0, 0.5), (1.0, 1.0, 2.0)),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
        n_axis=(0.0, 0.0, 1.0),
        u_bounds=(-1.0, 1.0),
        v_bounds=(-1.0, 1.0),
        n_bounds=(0.5, 2.0),
        u_cut=1.0,
        frontier_source="synthetic test",
    )
    masks = tuple(
        ManualCameraMask(
            name,
            ((0.0, 0.0), (648.0, 0.0), (648.0, 420.0), (0.0, 420.0)),
            "synthetic full-frame mask",
        )
        for name in ("c0", "c1", "c2")
    )
    return OracleSurfaceControl(
        "synthetic",
        "SYNTHETIC",
        config,
        masks,
        "synthetic test only",
    )


def test_renderer_event_roundtrip_recomputes_same_pixel_and_depth() -> None:
    camera = _Camera()
    pixel_x = np.asarray([0.0, 3.0, 7.0], dtype=np.float64)
    pixel_y = np.asarray([0.0, 2.0, 5.0], dtype=np.float64)
    depth = np.asarray([0.8, 1.0, 2.0], dtype=np.float64)

    points = demo._reconstruct_world_from_renderer_pixel_depth(
        pixel_x, pixel_y, depth, camera
    )
    projection = demo._renderer_projected_pixels(points, camera)

    assert np.allclose(projection["x"], pixel_x)
    assert np.allclose(projection["y"], pixel_y)
    assert np.allclose(projection["depth"], depth)
    assert np.all(projection["valid"])


def test_depth_semantics_are_explicitly_camera_z_event_not_center_or_ray_length() -> None:
    semantics = demo.renderer_depth_median_semantics()

    assert semantics["event_depth_formula"] == "depth=(s.x*Tw.x+s.y*Tw.y)+Tw.z"
    assert "Gaussian center view-space z in general" in semantics["not_equal_to"]
    assert "Euclidean camera-to-event ray length" in semantics["not_equal_to"]
    assert "normalized or inverse depth" in semantics["not_equal_to"]


def test_per_view_states_and_diagnostic_populations_share_one_frozen_matrix() -> None:
    control = _control()
    cameras = {name: _Camera(name) for name in ("c0", "c1", "c2")}
    points = np.asarray(
        [
            [-0.5, 0.0, 1.00],  # 3 near -> D3
            [0.0, 0.0, 1.20],  # 2 near, 1 after -> D2
            [0.5, 0.0, 1.30],  # 1 near, 2 after -> D1
            [0.8, 0.0, 0.70],  # 0 near, 3 before -> no D
        ],
        dtype=np.float64,
    )
    depths = {
        "c0": np.ones((6, 8), dtype=np.float64),
        "c1": np.ones((6, 8), dtype=np.float64),
        "c2": np.ones((6, 8), dtype=np.float64),
    }
    depths["c0"][3, 4] = 1.20
    depths["c1"][3, 4] = 1.20
    depths["c2"][3, 4] = 1.00
    depths["c0"][3, 5] = 1.30

    states, _ = demo._classify_states(points, control, cameras, depths, mu=0.05)
    counts = demo._state_counts(states)
    mask_only = counts["mask_match"] >= demo.MASK_VOTE_MIN
    populations = {
        "D1": mask_only & (counts["near"] >= 1),
        "D2": mask_only & (counts["near"] >= 2),
        "D3": mask_only & (counts["near"] >= 3),
    }

    assert np.array_equal(counts["near"], [3, 2, 1, 0])
    assert np.array_equal(counts["before"], [0, 0, 0, 3])
    assert np.array_equal(counts["after"], [0, 1, 2, 0])
    assert np.array_equal(populations["D1"], [True, True, True, False])
    assert np.array_equal(populations["D2"], [True, True, False, False])
    assert np.array_equal(populations["D3"], [True, False, False, False])
    assert np.array_equal(states[0], [demo.NEAR_MEDIAN, demo.NEAR_MEDIAN, demo.NEAR_MEDIAN])
    assert np.array_equal(states[1], [demo.NEAR_MEDIAN, demo.NEAR_MEDIAN, demo.AFTER_MEDIAN])
    assert np.array_equal(states[2], [demo.NEAR_MEDIAN, demo.AFTER_MEDIAN, demo.AFTER_MEDIAN])
    assert np.array_equal(states[3], [demo.BEFORE_MEDIAN, demo.BEFORE_MEDIAN, demo.BEFORE_MEDIAN])


def test_zero_attribution_separates_near_requirement_from_hard_veto() -> None:
    counts = {
        "near": np.asarray([3, 2, 1, 0], dtype=np.int16),
        "before": np.asarray([0, 0, 0, 3], dtype=np.int16),
        "after": np.asarray([0, 1, 2, 0], dtype=np.int16),
        "mask_match": np.asarray([3, 3, 3, 3], dtype=np.int16),
        "no_valid_depth": np.zeros(4, dtype=np.int16),
    }
    attribution = demo._wl142_zero_attribution(
        counts,
        np.ones(4, dtype=bool),
        {"depth_accounting": {"mask_plus_depth_count": 0}},
    )

    assert attribution["near_requirement_before_zero_before_after_veto"] == 2
    assert attribution["removed_because_at_least_one_after"] == 1
    assert attribution["hard_veto_survivors_recomputed"] == 1
    assert attribution["all_near_requirement_points_vetoed"] is False
