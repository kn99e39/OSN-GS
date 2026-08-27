"""Focused contract checks for the fixed Worklog 125 visualization exporter."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


_SCRIPT = Path(__file__).parents[1] / "scripts" / "devtools" / "export_wl123_observed_occluded_gaussian_visualization.py"
_SPEC = importlib.util.spec_from_file_location("wl123_fixed_visualization", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(MODULE)


def test_streamed_global_state_preserves_frozen_any_observed_rule():
    observed_any = torch.tensor([True, False, False, False])
    has_relevant = torch.tensor([True, True, False, True])
    has_unresolved = torch.tensor([False, False, False, True])
    actual = MODULE.global_state_from_accumulators(observed_any, has_relevant, has_unresolved)
    assert actual.tolist() == [MODULE.STATE_OBSERVED, MODULE.STATE_OCCLUDED, MODULE.STATE_UNRESOLVED, MODULE.STATE_UNRESOLVED]


def test_state_colours_are_one_for_one_existing_gaussians():
    states = torch.tensor([MODULE.STATE_OBSERVED, MODULE.STATE_OCCLUDED, MODULE.STATE_UNRESOLVED], dtype=torch.int8)
    colours = MODULE.state_colours(states)
    assert tuple(colours.shape) == (3, 3)
    assert torch.allclose(colours[0], torch.tensor(MODULE._OBSERVED_RGB))
    assert torch.allclose(colours[1], torch.tensor(MODULE._OCCLUDED_RGB))
    assert torch.allclose(colours[2], torch.tensor(MODULE._UNRESOLVED_RGB))


def test_exporter_explicitly_forbids_marker_and_lighting_semantics():
    source = _SCRIPT.read_text(encoding="utf-8")
    assert '"marker_gaussians_added": 0' in source
    assert '"lighting_added": False' in source
    assert "renderer_event_provenance\": \"absent for every centre" in source


def test_novel_inspection_camera_is_separate_from_query_cameras():
    from osn_gs.render.torch_fallback import TorchCamera

    cameras = [
        TorchCamera(16, 16, torch.eye(4), torch.eye(4), torch.tensor((3.0, 0.0, 0.0)), 0.7, 0.7, "QUERY_A"),
        TorchCamera(16, 16, torch.eye(4), torch.eye(4), torch.tensor((-3.0, 0.0, 0.0)), 0.7, 0.7, "QUERY_B"),
    ]
    positions = torch.tensor(((0.0, 0.0, 0.0), (0.3, 0.1, 0.2), (-0.2, -0.1, 0.1)))
    candidates = MODULE.novel_inspection_candidates(cameras, positions)
    assert len(candidates) == 16
    assert all(candidate.image_name.startswith("NOVEL_OUTER_ORBIT_") for candidate in candidates)
    query_centres = torch.stack([camera.camera_center for camera in cameras])
    assert all(float(torch.cdist(candidate.camera_center.reshape(1, 3), query_centres).min()) > 1e-3 for candidate in candidates)