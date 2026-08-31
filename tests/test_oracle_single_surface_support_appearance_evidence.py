from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

import devtools.demo.oracle_single_surface_support_appearance_evidence as demo


class _Camera:
    def __init__(self, name: str) -> None:
        self.image_name = name
        self.image_width = 648
        self.image_height = 420
        self.world_view_transform = np.eye(4, dtype=np.float64)
        self.full_proj_transform = np.eye(4, dtype=np.float64)
        self.camera_center = np.zeros(3, dtype=np.float64)


def _camera_mask(name: str) -> demo.ManualCameraMask:
    return demo.ManualCameraMask(name, ((100, 100), (500, 100), (500, 320), (100, 320)), "test")


def test_manual_control_set_has_three_fixed_surfaces_and_no_fit_dependency() -> None:
    controls = demo._manual_controls()
    assert len(controls) >= 3
    assert {control.name for control in controls} == {
        "tabletop_top_oracle",
        "curved_table_rim_oracle",
        "paver_ground_oracle",
    }
    source = inspect.getsource(demo._manual_controls)
    assert "fit_" not in source
    assert "graphness" not in source.lower()


def test_polygon_support_is_deterministic_and_uses_only_projection_masks() -> None:
    camera = _Camera("cam")
    config = demo.RepresentativeCaseConfig(
        name="synthetic",
        semantic_label="synthetic",
        roi_box=demo.Box((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
        n_axis=(0.0, 0.0, 1.0),
        u_bounds=(-1.0, 2.0),
        v_bounds=(-1.0, 2.0),
        n_bounds=(-1.0, 2.0),
        u_cut=2.0,
        frontier_source="test",
    )
    control = demo.OracleSurfaceControl(
        "synthetic",
        "TEST",
        config,
        (_camera_mask("cam"), _camera_mask("cam"), _camera_mask("cam")),
        "test",
    )
    points = np.asarray([
        [-0.5, -0.5, 0.5],
        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.5],
        [1.5, 1.5, 0.5],
    ])
    first = demo.build_oracle_support(points, control, [camera])
    second = demo.build_oracle_support(points, control, [camera])
    assert np.array_equal(first.oracle_row_ids, second.oracle_row_ids)
    assert np.array_equal(first.mask_vote_counts, second.mask_vote_counts)
    assert np.array_equal(first.oracle_row_ids, np.asarray([0, 1, 2], dtype=np.int64))


def test_oracle_membership_api_has_no_representative_or_metric_input() -> None:
    signature = inspect.signature(demo.build_oracle_support)
    names = set(signature.parameters)
    assert "representative" not in names
    assert "graphness" not in names
    assert "metric" not in names


def test_support_domain_annotation_is_geometry_preserving() -> None:
    class _Representative:
        domain_u = (0.0, 1.0)
        domain_v = (0.0, 1.0)
        sampled_points = np.zeros((demo.SAMPLE_COUNT_U * demo.SAMPLE_COUNT_V, 3), dtype=np.float64)

    representative = _Representative()
    representative.sampled_points[:, 0] = np.repeat(np.linspace(0.0, 1.0, demo.SAMPLE_COUNT_U), demo.SAMPLE_COUNT_V)
    representative.sampled_points[:, 1] = np.tile(np.linspace(0.0, 1.0, demo.SAMPLE_COUNT_V), demo.SAMPLE_COUNT_U)
    config = demo.RepresentativeCaseConfig(
        name="synthetic",
        semantic_label="synthetic",
        roi_box=demo.Box((0.0, 0.0, -1.0), (1.0, 1.0, 1.0)),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
        n_axis=(0.0, 0.0, 1.0),
        u_bounds=(0.0, 1.0),
        v_bounds=(0.0, 1.0),
        n_bounds=(-1.0, 1.0),
        u_cut=1.0,
        frontier_source="test",
    )
    points = np.asarray([[0.1, 0.1, 0.0], [0.9, 0.9, 0.0]], dtype=np.float64)
    before = representative.sampled_points.copy()
    result = demo.support_domain_diagnostic(points, representative, config)
    assert np.array_equal(representative.sampled_points, before)
    assert result["fitted_geometry_unchanged"] is True
    assert result["support_annotation_only"] is True
    assert 0.0 < result["supported_chart_fraction"] < 1.0


def test_provenance_audit_rejects_ply_without_primitive_identity(tmp_path: Path) -> None:
    raw = tmp_path / "raw.ply"
    raw.write_bytes(
        b"ply\nformat ascii 1.0\nelement vertex 1\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"property float f_dc_0\nend_header\n0 0 0 0\n"
    )
    report = demo.audit_gaussian_provenance(raw, tmp_path / "checkpoint.pt")
    assert report["status"] == "NO_VALID_PRIMITIVE_PROVENANCE"
    assert report["valid_for_sh_evaluation"] is False
    assert report["nearest_primitive_proxy_used"] is False


def test_stage_b_contains_no_automatic_segmentation_or_threshold_path() -> None:
    source = inspect.getsource(demo)
    assert "region_grow" not in source.lower()
    assert "nearest" in source.lower()
    assert "nearest_primitive_proxy_used" in source
    assert "appearance_signals" in source
