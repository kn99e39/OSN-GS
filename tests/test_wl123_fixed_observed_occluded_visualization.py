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
