from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from devtools.demo import physical_sheet_evidence_vs_chart_extent_failure_attribution as audit


def test_wl149_reconciles_the_committed_wl148_baseline():
    baseline = audit._load_frozen_baseline(
        audit.WL145_REPORT_PATH,
        audit.WL145_REPRESENTATIVE,
        audit.WL139_REPORT_PATH,
        audit.WL145_EVENT_ROOT,
    )
    replay = audit._load_wl148_artifact(baseline)
    assert replay["exact_replay"] is True
    assert len(baseline["oracle_points"]) == 1586
    assert audit._sha256_rows(baseline["oracle_points"]) == "79855ad840164a923f8c4bb1c6935ce22cff8030bfedebf7a0dc4cd141026c78"
    assert baseline["baseline"]["event_camera_counts"] == {
        "DSC08043.JPG": 754,
        "DSC07960.JPG": 330,
        "DSC08003.JPG": 502,
    }
    assert int(np.sum(baseline["support_vertices"])) == 314
    assert int(np.sum(baseline["cell_mask"])) == 211


def test_pca_replay_and_extrema_are_deterministic():
    baseline = audit._load_frozen_baseline(
        audit.WL145_REPORT_PATH,
        audit.WL145_REPRESENTATIVE,
        audit.WL139_REPORT_PATH,
        audit.WL145_EVENT_ROOT,
    )
    frame_a = audit._pca_frame(baseline["oracle_points"])
    frame_b = audit._pca_frame(baseline["oracle_points"])
    np.testing.assert_array_equal(frame_a["axes"], frame_b["axes"])
    np.testing.assert_array_equal(frame_a["projected"], frame_b["projected"])
    extrema = {
        "u_min": audit._owners(frame_a["projected"][:, 0], int(np.argmin(frame_a["projected"][:, 0]))),
        "u_max": audit._owners(frame_a["projected"][:, 0], int(np.argmax(frame_a["projected"][:, 0]))),
        "v_min": audit._owners(frame_a["projected"][:, 1], int(np.argmin(frame_a["projected"][:, 1]))),
        "v_max": audit._owners(frame_a["projected"][:, 1], int(np.argmax(frame_a["projected"][:, 1]))),
    }
    assert extrema == {"u_min": [947], "u_max": [795], "v_min": [1527], "v_max": [1104]}


def test_leave_one_out_and_provenance_cover_every_event_without_filtering():
    baseline = audit._load_frozen_baseline(
        audit.WL145_REPORT_PATH,
        audit.WL145_REPRESENTATIVE,
        audit.WL139_REPORT_PATH,
        audit.WL145_EVENT_ROOT,
    )
    provenance = audit._load_provenance(baseline)
    frame = audit._pca_frame(baseline["oracle_points"])
    fixed_a, ranking_a = audit._fixed_axis_loo(frame["projected"])
    fixed_b, ranking_b = audit._fixed_axis_loo(frame["projected"])
    pca_a, pca_ranking_a = audit._full_pca_loo(baseline["oracle_points"], frame)
    pca_b, pca_ranking_b = audit._full_pca_loo(baseline["oracle_points"], frame)
    assert fixed_a["u_span_influence"]["samples"] == 1586
    assert pca_a["joint_axis_rotation_distribution_degrees"]["samples"] == 1586
    assert [row["event_id"] for row in ranking_a] == [row["event_id"] for row in ranking_b]
    assert [row["event_id"] for row in pca_ranking_a] == [row["event_id"] for row in pca_ranking_b]
    assert len(provenance["event_id"]) == 1586
    assert len(np.unique(provenance["event_id"])) == 1586
    assert set(provenance) == {"event_id", "source_camera", "source_pixel_x", "source_pixel_y", "depth", "world_xyz", "normal"}
    assert np.array_equal(provenance["world_xyz"], baseline["oracle_points"])
    assert np.array_equal(ranking_a[0].keys(), ranking_b[0].keys())
    assert pca_a["ranking_is_diagnostic_only"] is True
    assert pca_b["no_keep_reject_threshold"] is True


def test_synthetic_contracts_pass_and_no_completion_logic_is_present():
    result = audit._synthetic_contracts()
    assert result["all_synthetic_contracts_pass"] is True
    assert result["fixture_A_compact_planar_plus_far_same_plane"]["point_not_automatically_rejected"] is True
    assert result["fixture_C_no_dominant_individual_extrema"]["automatic_rejection_decision"] is False

    source = Path(audit.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert calls.isdisjoint({"fit_physical_chart_surface", "build_self_continuation", "build_occluded_surface"})
    assert "withheld_reference_geometry_evaluation_only" in source.lower()
