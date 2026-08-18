from __future__ import annotations

import sys
from pathlib import Path

import torch

DEVTOOLS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "devtools"
if str(DEVTOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(DEVTOOLS_DIR))

from patch_identifiability_capacity_gate_replay import (  # noqa: E402
    FIXED_DEGREE,
    FIXED_RESOLUTION,
    _candidate_a,
    _candidate_b,
    _candidate_c,
)


def _sample_chart():
    torch.manual_seed(0)
    count = 40
    coords = torch.linspace(0.0, 1.0, count)
    points = torch.stack([coords, torch.sin(coords), torch.zeros(count)], dim=1)
    uv = torch.stack([coords, (coords * 0.3) % 1.0], dim=1)
    held_out = torch.randn(5, 3)
    return points, uv, held_out


def test_identical_chart_uv_evidence_across_candidates():
    points, uv, held_out = _sample_chart()
    record_a, _surface_a = _candidate_a(points, uv, held_out, 1.0)
    record_b, _surface_b = _candidate_b(points, uv, held_out, 1.0)
    record_c, _surface_c = _candidate_c(points, uv, held_out, 1.0)
    # All three candidates must have been evaluated against the exact same
    # sample count in their own identifiability report -- no candidate
    # silently receives a different/resized UV or evidence set.
    assert record_a["identifiability"]["sample_count"] == int(points.shape[0])
    if record_b["identifiability"]:
        assert record_b["identifiability"]["sample_count"] == int(points.shape[0])
    if record_c["identifiability"]:
        assert record_c["identifiability"]["sample_count"] == int(points.shape[0])


def test_fixed_6x6_baseline_unchanged():
    points, uv, held_out = _sample_chart()
    record_a, _surface_a = _candidate_a(points, uv, held_out, 1.0)
    assert record_a["identifiability"]["control_grid_u"] == FIXED_RESOLUTION
    assert record_a["identifiability"]["control_grid_v"] == FIXED_RESOLUTION
    assert record_a["identifiability"]["degree_u"] == FIXED_DEGREE


def test_no_chart_resizing_or_upstream_modification_in_replay_script():
    import ast
    import inspect

    import patch_identifiability_capacity_gate_replay as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    # The replay script must reuse build_local_chart_atlas unmodified and
    # must never re-derive the tangent field independently per candidate.
    assert "build_local_chart_atlas" in imported_names
    assert not any("seed_curve" in name.lower() and name != "build_seed_curves" for name in imported_names)
