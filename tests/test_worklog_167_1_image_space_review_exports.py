from __future__ import annotations

from pathlib import Path

import numpy as np

from devtools.demo.worklog_167_1_image_space_review_exports import (
    TARGETS,
    _layout_report,
    _polygon_mask,
    derive_crop_box,
    derive_spotlight_box,
    project_world_points,
    reconstruct_query_ladder,
    target_mask,
)
from devtools.demo.worklog_167_raw_zero_set_ray_blocker_audit import REVIEW_POLYGONS


class _TensorLike:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def detach(self) -> "_TensorLike":
        return self

    def cpu(self) -> "_TensorLike":
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class _DummyCamera:
    image_width = 100
    image_height = 80
    FoVx = 0.7
    FoVy = 0.7
    full_proj_transform = _TensorLike(np.eye(4, dtype=np.float64))
    world_view_transform = _TensorLike(np.eye(4, dtype=np.float64))
    camera_center = _TensorLike(np.zeros((3,), dtype=np.float64))


def test_projection_uses_full_projection_once_and_reports_bounds() -> None:
    camera = _DummyCamera()
    projected, valid = project_world_points(camera, np.asarray([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0]]))
    assert np.allclose(projected[0], [49.5, 39.5])
    assert valid.tolist() == [True, False]


def test_crop_box_is_deterministic_and_has_readable_minimum_canvas() -> None:
    support = np.asarray([[20.0, 20.0], [32.0, 20.0], [32.0, 28.0], [20.0, 28.0]])
    hits = np.asarray([[24.0, 24.0], [28.0, 24.0]])
    first = derive_crop_box((100, 80), support, hits)
    second = derive_crop_box((100, 80), support, hits)
    assert first == second
    assert first[2] - first[0] >= 96
    assert first[3] - first[1] >= 64
    assert first[0] <= support[:, 0].min() <= support[:, 0].max() < first[2]
    assert first[1] <= support[:, 1].min() <= support[:, 1].max() < first[3]


def test_spotlight_box_is_deterministic_and_contains_suspicious_hits() -> None:
    suspicious = np.asarray([[40.0, 50.0], [44.0, 54.0]])
    first = derive_spotlight_box((100, 80), suspicious)
    second = derive_spotlight_box((100, 80), suspicious)
    assert first == second
    assert first[0] <= suspicious[:, 0].min() <= suspicious[:, 0].max() < first[2]
    assert first[1] <= suspicious[:, 1].min() <= suspicious[:, 1].max() < first[3]


def test_contact_mask_is_the_frozen_union_of_tabletop_and_vase() -> None:
    camera_name = "DSC07960.JPG"
    tabletop = np.asarray(REVIEW_POLYGONS["tabletop"][camera_name], dtype=np.float64)
    vase = np.asarray(REVIEW_POLYGONS["vase_foreground_structure"][camera_name], dtype=np.float64)
    pixels = np.asarray(
        [[int(round(tabletop[:, 1].mean())), int(round(tabletop[:, 0].mean()))], [int(round(vase[:, 1].mean())), int(round(vase[:, 0].mean()))]],
        dtype=np.int64,
    )
    contact = target_mask(camera_name, pixels, "tabletop_vase_contact")
    assert contact.tolist() == [True, True]
    assert set(TARGETS) == {"tabletop", "tabletop_vase_contact", "table_side_lower_geometry", "vase_foreground_structure"}


def test_reconstructed_ladder_preserves_w167_relations_and_offset() -> None:
    camera = _DummyCamera()
    record = {
        "pixel": np.asarray([[39, 49]], dtype=np.int64),
        "status": np.asarray(["HIT"]),
        "depth": np.asarray([2.0], dtype=np.float64),
        "world_xyz": np.asarray([[0.0, 0.0, 2.0]], dtype=np.float64),
        "triangle_id": np.asarray([7], dtype=np.int64),
        "component_id": np.asarray([3], dtype=np.int64),
    }
    ladder = reconstruct_query_ladder(camera, record, offset=0.5)
    assert [row["label"] for row in ladder["rows"]] == ["Q_before", "Q_surface", "Q_behind"]
    assert [row["relation"] for row in ladder["rows"]] == ["IN_FRONT_OF_ZEROSET_SURFACE", "ZEROSET_FIRST_SURFACE", "BEHIND_ZEROSET_SURFACE"]
    assert [row["query_depth"] for row in ladder["rows"]] == [1.5, 2.0, 2.5]
    assert all(row["candidate_b_median_used"] is False for row in ladder["rows"])


def test_layout_requires_shared_readme_and_direct_camera_pngs(tmp_path: Path) -> None:
    camera_names = ("DSC07960.JPG", "DSC08003.JPG", "DSC08043.JPG")
    first = tmp_path / "first_hit_overlay_full"
    first.mkdir()
    (first / "README.md").write_text("shared", encoding="utf-8")
    for camera_name in camera_names:
        (first / f"{Path(camera_name).stem}.png").write_bytes(b"png")
    report = _layout_report(tmp_path, [first], camera_names)
    assert report["shared_readme_plus_direct_camera_png_rule"] is True

    (first / "DSC07960").mkdir()
    broken = _layout_report(tmp_path, [first], camera_names)
    assert broken["shared_readme_plus_direct_camera_png_rule"] is False
    assert any("nested camera directory" in item for item in broken["violations"])
