import json
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from devtools.demo import worklog_172_render_view_projection_review as review


def test_renderer_half_pixel_and_downward_y_convention():
    camera = SimpleNamespace(image_width=648, image_height=420,
                             world_view_transform=torch.eye(4), full_proj_transform=torch.eye(4))
    pixels, valid = review.project(np.array([[0, 0, 1], [.5, .5, 1], [3, 0, 1], [0, 0, .1]]), camera)
    np.testing.assert_array_equal(pixels[:2], [[323.5, 209.5], [485.5, 314.5]])
    np.testing.assert_array_equal(valid, [True, True, False, False])


def test_source_artifacts_and_render_pair_preserved():
    report = json.loads((review.OUT / "projection_report.json").read_text(encoding="utf-8"))
    assert report["source_files_unchanged"]
    for mapping in ("source_w172_files", "source_renders"):
        for path, digest in report[mapping].items():
            assert review.sha(review.ROOT / path) == digest
    assert len(report["source_renders"]) == 6
    assert not report["gpu_rendering"]


def test_camera_exports_and_complete_projection_accounting():
    report = json.loads((review.OUT / "projection_report.json").read_text(encoding="utf-8"))
    for name, row in report["camera_accounting"].items():
        path = review.OUT / (name.removesuffix(".JPG") + ".png")
        with Image.open(path) as image:
            assert image.format == "PNG" and image.width > 1900
            image.verify()
        for case in row["cases"].values():
            assert case["complete_count"] == case["selected_projected"] + case["excluded_projected"] + case["outside_frustum_or_near"]
            assert not case["display_subsampling"] and not case["depth_visibility_culling"]
    assert len(list(review.OUT.glob("*.png"))) == 3
    assert (review.OUT / "README.md").read_text(encoding="utf-8")
