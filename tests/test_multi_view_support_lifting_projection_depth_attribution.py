from __future__ import annotations

import numpy as np

import devtools.demo.multi_view_support_lifting_projection_depth_attribution as demo
from devtools.demo.meeting_occluded_surface_feasibility import Box
from devtools.demo.oracle_single_surface_support_appearance_evidence import ManualCameraMask


class _Camera:
    def __init__(self, name: str = "cam") -> None:
        self.image_name = name
        self.image_width = 8
        self.image_height = 6
        self.world_view_transform = np.eye(4, dtype=np.float64)
        self.full_proj_transform = np.eye(4, dtype=np.float64)
        self.camera_center = np.zeros(3, dtype=np.float64)


def _control() -> demo.OracleSurfaceControl:
    config = demo.RepresentativeCaseConfig(
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
    mask = ManualCameraMask(
        "cam",
        ((0.0, 0.0), (648.0, 0.0), (648.0, 420.0), (0.0, 420.0)),
        "synthetic full-frame mask",
    )
    return demo.OracleSurfaceControl(
        "synthetic",
        "SYNTHETIC",
        config,
        (mask, mask, mask),
        "synthetic test only",
    )


def test_projection_contract_uses_row_vector_and_image_y_down() -> None:
    camera = _Camera()
    points = np.asarray(
        [[0.0, 0.0, 1.0], [1.5, -1.0, 1.0], [-1.0, 1.0, 1.0]],
        dtype=np.float64,
    )

    projection = demo.project_world_points(points, camera)

    assert np.allclose(projection["x"], [3.5, 8.75, 0.0])
    assert np.allclose(projection["y"], [2.5, 5.0, 0.0])
    assert np.array_equal(projection["valid"], np.asarray([True, False, True]))
    audit = demo.projection_contract_audit([camera])
    assert audit["status"] == "PROJECTION_CONTRACT_PASS"
    assert audit["camera_reports"][0]["camera_center_l2_gap"] == 0.0


def test_depth_relation_distinguishes_consistent_behind_front_and_no_depth() -> None:
    camera = _Camera()
    mask = _control().masks[0]
    points = np.asarray(
        [
            [0.0, 0.0, 1.00],
            [0.2, 0.0, 1.02],
            [0.4, 0.0, 1.30],
            [0.6, 0.0, 0.70],
            [2.0, 0.0, 1.00],
        ],
        dtype=np.float64,
    )
    depth = np.ones((6, 8), dtype=np.float64)
    depth[2, 5] = 0.0

    relation = demo._depth_relation_for_camera(points, camera, mask, depth, mu=0.05)

    assert np.array_equal(
        relation["mask_match"],
        np.asarray([True, True, True, True, False]),
    )
    assert bool(relation["depth_consistent"][0])
    assert bool(relation["depth_consistent"][1])
    assert bool(relation["behind_visible_frontier"][2])
    assert bool(relation["in_front_of_visible_frontier"][3])
    assert bool(relation["no_relevant_renderer_depth"][4])


def test_mask_plus_depth_is_fixed_filter_over_replayed_mask_only_support() -> None:
    camera = _Camera()
    control = _control()
    points = np.asarray(
        [
            [0.0, 0.0, 1.00],
            [0.2, 0.0, 1.30],
            [0.4, 0.0, 0.70],
        ],
        dtype=np.float64,
    )
    depth = np.ones((6, 8), dtype=np.float64)

    accounting = demo.depth_layer_accounting(
        points,
        control,
        {"cam": camera},
        {"cam": depth},
        mu=0.05,
        h=0.1,
    )

    assert accounting["mask_only_count"] == 3
    assert accounting["mask_plus_depth_count"] == 1
    assert accounting["removed_by_depth_count"] == 2
    assert accounting["rule"]["depth_tolerance_derivation"] == "frozen WL139 mu; not selected from WL142 labels or output quality"
    assert np.array_equal(
        accounting["arrays"]["mask_plus_depth"],
        np.asarray([True, False, False]),
    )


def test_depth_accounting_is_deterministic_for_same_inputs() -> None:
    camera = _Camera()
    control = _control()
    points = np.asarray([[0.0, 0.0, 1.0], [0.2, 0.0, 1.1]], dtype=np.float64)
    depth = np.ones((6, 8), dtype=np.float64)

    first = demo.depth_layer_accounting(points, control, {"cam": camera}, {"cam": depth}, 0.05, 0.1)
    second = demo.depth_layer_accounting(points, control, {"cam": camera}, {"cam": depth}, 0.05, 0.1)

    assert np.array_equal(first["arrays"]["mask_only"], second["arrays"]["mask_only"])
    assert np.array_equal(first["arrays"]["mask_plus_depth"], second["arrays"]["mask_plus_depth"])
    assert first["mask_plus_depth_count"] == second["mask_plus_depth_count"]
