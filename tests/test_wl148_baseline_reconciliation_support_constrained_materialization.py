from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from devtools.demo import wl145_baseline_reconciliation_support_constrained_materialization as audit


def test_exact_wl145_event_and_support_replay():
    baseline = audit._load_frozen_baseline()
    assert len(baseline["oracle_points"]) == 1586
    assert audit._sha256_rows(baseline["oracle_points"]) == audit.EVENT_UNION_SHA256
    assert baseline["baseline"]["event_camera_counts"] == {
        "DSC08043.JPG": 754,
        "DSC07960.JPG": 330,
        "DSC08003.JPG": 502,
    }
    assert int(np.sum(baseline["support_vertices"])) == 314
    assert audit._sha256_array(baseline["support_vertices"].astype(np.uint8)) == audit.SUPPORT_MASK_SHA256
    assert int(np.sum(baseline["cell_mask"])) == 211


def test_all_four_support_relation_is_the_frozen_wl145_rule():
    baseline = audit._load_frozen_baseline()
    expected = (
        baseline["support_vertices"][:-1, :-1]
        & baseline["support_vertices"][1:, :-1]
        & baseline["support_vertices"][:-1, 1:]
        & baseline["support_vertices"][1:, 1:]
    )
    np.testing.assert_array_equal(baseline["cell_mask"], expected)
    assert audit._cell_mask_from_support(baseline["support_vertices"]).shape == (95, 39)
    assert audit._topology_accounting(baseline["cell_mask"])["holes"] == 0


def test_arms_share_exact_frozen_xyz_and_normals():
    baseline = audit._load_frozen_baseline()
    arm_a = audit._materialized_vertex_indices(np.ones((95, 39), dtype=bool))
    arm_b = audit._materialized_vertex_indices(baseline["cell_mask"])
    assert np.array_equal(baseline["representative_points"][arm_a], baseline["representative_points"])
    assert np.array_equal(baseline["representative_normals"][arm_a], baseline["representative_normals"])
    assert set(arm_b.tolist()).issubset(set(arm_a.tolist()))
    assert audit._sha256_array(baseline["representative_points"][arm_b]) == audit._sha256_array(
        baseline["representative_points"][audit._materialized_vertex_indices(baseline["cell_mask"])]
    )


def test_frozen_load_has_no_fit_or_continuation_call():
    source_path = Path(audit.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden = {"fit_physical_chart_surface", "build_self_continuation"}
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert calls.isdisjoint(forbidden)
    frozen_loader = ast.get_source_segment(source_path.read_text(encoding="utf-8"), next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_load_frozen_baseline"
    ))
    assert "fit_physical_chart_surface(" not in frozen_loader
    assert "build_self_continuation(" not in frozen_loader
