from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from devtools.demo import worklog_171_zero_set_derived_visible_structural_nurbs_audit as w171
from osn_gs.surface.torch_gaussian_region_owned_tsdf import ABSTAIN_REPRESENTATIVE, MATERIALIZED_REPRESENTATIVE


@pytest.fixture(scope="module")
def synthetic_results() -> dict[str, tuple[w171.CaseRuntime, dict]]:
    rows: dict[str, tuple[w171.CaseRuntime, dict]] = {}
    for name in w171.SYNTHETIC_CASES:
        runtime = w171._synthetic_runtime(name)
        rows[name] = (runtime, w171.analyze_case(runtime))
    return rows


def test_single_sheet_materialization_and_mixed_abstention(synthetic_results) -> None:
    planar = synthetic_results["synthetic_planar_single_sheet"][1]
    curved = synthetic_results["synthetic_curved_single_sheet"][1]
    layered = synthetic_results["synthetic_layered_non_single_chart"][1]
    assert planar["result"] == MATERIALIZED_REPRESENTATIVE
    assert curved["result"] == MATERIALIZED_REPRESENTATIVE
    assert layered["result"] == ABSTAIN_REPRESENTATIVE
    assert layered["reason"] == "multiple_native_tsdf_components"
    assert layered["components"]["count"] == 2


def test_support_preservation_and_boundary_fitter_contract(synthetic_results) -> None:
    for name in ("synthetic_planar_single_sheet", "synthetic_curved_single_sheet"):
        result = synthetic_results[name][1]
        assert result["support"]["preserved_exactly"]
        assert result["components"]["count"] == 1
        assert result["boundary"]["loop_count"] == 1
        assert result["boundary"]["closed"]
        assert result["boundary"]["ordered_boundary_valid"]
        assert result["solvability"]["design_matrix_rank"] == 32
        assert result["solvability"]["required_rank"] == 32
        assert result["fit_residual"]["world_units"]["count"] == result["support"]["count"]
        assert result["fit_contract"] == w171.FIT_CONTRACT


def test_real_case_selection_is_deterministic_and_pre_fit() -> None:
    first = w171.prior_real_case_specs(w171.DEFAULT_W154, w171.DEFAULT_W145)
    second = w171.prior_real_case_specs(w171.DEFAULT_W154, w171.DEFAULT_W145)
    assert first.keys() == second.keys()
    for name in first:
        assert first[name]["camera"] == "DSC07960.JPG"
        assert first[name]["prior_region_id"] == second[name]["prior_region_id"]
        assert np.array_equal(first[name]["aabb_min"], second[name]["aabb_min"])
        assert np.array_equal(first[name]["aabb_max"], second[name]["aabb_max"])
        assert first[name]["selection_frozen_before_w171_fit"]
        assert not first[name]["selection_uses_w171_visual_success"]


def test_real_complete_support_abstains_without_component_filter() -> None:
    report_path = w171.DEFAULT_OUT / "worklog_171_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for name in w171.REAL_CASE_CONTRACT:
        result = report["cases"][name]
        assert result["result"] == ABSTAIN_REPRESENTATIVE
        assert result["components"]["count"] > 1
        assert result["components"]["all_components_retained"]
        assert not result["components"]["largest_component_selection"]
        assert not result["no_rescue"]["component_deletion"]


def test_required_visualizations_and_local_readmes_exist() -> None:
    validation = w171.validate_visual_artifacts(w171.DEFAULT_OUT)
    assert validation["complete"]
    assert validation["png_count"] == 30
    assert validation["ppm_files"] == []
    for case in w171.CASE_ORDER:
        assert (w171.DEFAULT_OUT / "case_artifacts" / case / "README.md").is_file()
