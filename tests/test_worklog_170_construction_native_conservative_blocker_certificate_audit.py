from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from devtools.demo import worklog_170_construction_native_conservative_blocker_certificate_audit as w170


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_stored_schema_does_not_retain_per_view_provenance_or_extrema() -> None:
    evidence = w170.available_evidence_contract()
    assert evidence["sparse_field_stored_fields"] == ["keys", "value", "support_count", "h", "mu"]
    assert evidence["per_view_phi_stored"] is False
    assert evidence["source_view_or_camera_id_stored"] is False
    assert evidence["projective_depth_sample_stored"] is False
    control = evidence["fusion_noninjectivity_control"]
    assert control["same_count"] is True
    assert control["same_fused_mean"] is True
    assert control["different_per_view_extrema_and_signs"] is True


def test_no_available_quantity_has_a_construction_native_one_sided_meaning() -> None:
    evidence = w170.available_evidence_contract()
    semantic = w170.one_sided_semantic_audit()
    assert semantic
    assert not any(record.get("one_sided_physical_relation", False) for record in semantic.values())
    decision = w170.derive_certificate_decision(evidence, semantic)
    assert decision["verdict"] == w170.VERDICT_NO_CERTIFICATE
    assert decision["certificate_defined"] is False
    assert decision["usable_one_sided_quantities"] == []
    assert decision["no_fallback_margin"] is True


def test_trilinear_corner_evaluation_is_exact_for_simple_controls() -> None:
    corners = np.arange(8, dtype=np.float64).reshape(1, 8)
    offsets = w170.cell_corner_offsets()
    for corner, coordinate in enumerate(offsets):
        value = w170._trilinear(corners, coordinate.astype(np.float64).reshape(1, 3))
        assert value[0] == corners[0, corner]


def test_full_frozen_fixture_audit_reports_no_certificate_and_preserves_inputs() -> None:
    output_parent = REPO_ROOT / "output"
    output_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="w170_test_", dir=output_parent) as directory:
        report = w170.run(
            argparse.Namespace(
                w168_root=w170.DEFAULT_W168_ROOT,
                w169_root=w170.DEFAULT_W169_ROOT,
                out=Path(directory),
            )
        )
        assert report["7_architecture_result"]["verdict"] == w170.VERDICT_NO_CERTIFICATE
        assert report["4_certificate_definition"]["mathematical_predicate"] is None
        assert report["5_synthetic_soundness"]["certificate_evaluated"] is False
        assert report["6_coverage_abstention"]["certification_coverage"] is None
        available = report["2_available_construction_native_evidence"]
        assert available["hashes_unchanged"] is True
        assert all(item["w168_fixture"]["all_exact"] for item in available["reproduction"].values())
        assert all(item["field_to_extraction"]["all_exact"] for item in available["reproduction"].values())
        fixture_evidence = available["fixture_first_hit_evidence"]
        assert set(fixture_evidence) == set(w170.w168.FIXTURE_NAMES)
        for item in fixture_evidence.values():
            assert item["all_hit_cells_eight_corner_authoritative"] is True
            assert item["all_hit_cells_sign_changing"] is True
            assert item["cell_ray_extent"]["all_rays_intersect_recovered_owning_cell"] is True
            assert item["interpolation"]["edge_interpolation_count"] == item["interpolation"]["triangle_vertex_count"]
            assert item["interpolation"]["not_single_cell_edge_count"] == 0
            assert item["projective_depth_observation_provenance"].startswith("UNAVAILABLE")
        assert fixture_evidence["layered_two_sheet"]["interior_boundary"]["premature_interior"] == 970
        assert fixture_evidence["layered_two_sheet"]["interior_boundary"]["premature_boundary"] == 68
