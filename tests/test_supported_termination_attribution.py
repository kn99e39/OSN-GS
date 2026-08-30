"""Focused Worklog 132 attribution-contract tests."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from devtools.demo.explicit_geometric_termination_continuation import TerminationCurve
from devtools.demo.parametric_surface_continuation import PRIMARY_ROI
from devtools.demo.supported_termination_attribution import (
    ROOT_DIAGNOSTIC_TOLERANCE,
    ROOT_RESIDUAL_TOLERANCE,
    _curvature_gate,
    _metric_close,
    _supported_gamma_contract,
    _thin_root_classification,
    build_second_order_candidate,
    correspondence_restricted_metrics,
    directional_terms_from_derivatives,
    nearest_gamma_assignment,
    partition_by_gamma_support,
)


def _small_curve(support_distances: np.ndarray | None = None) -> TerminationCurve:
    count = 2
    if support_distances is None:
        support_distances = np.zeros(count, dtype=np.float64)
    return TerminationCurve(
        gamma_uv=np.asarray([[0.58, 0.0], [0.58, 1.0]], dtype=np.float64),
        gamma_points=np.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        derivatives_uv=np.zeros((count, 2), dtype=np.float64),
        physical_direction=np.tile(np.asarray([[1.0, 0.0, 0.0]]), (count, 1)),
        local_u_derivative=np.ones(count, dtype=np.float64),
        support_fit_distance=np.asarray(support_distances, dtype=np.float64),
        support_mesh_distance=np.asarray(support_distances, dtype=np.float64),
        root_rows=[],
        valid_mask=np.ones(count, dtype=bool),
    )


def test_frozen_worklog131_arm_b_report_contract_is_available_and_strict():
    report_path = Path("output/demo_explicit_geometric_termination_continuation/explicit_geometric_termination_report.json")
    if not report_path.exists():
        pytest.skip("generated frozen Worklog 131 output is not present in this checkout")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    primary = next(item for item in report["cases"] if item["roi"]["name"] == "curved_table_rim")
    frozen = primary["arm_B_explicit_termination_first_order"]["full_population_metrics"]
    assert frozen["samples"] == 12000
    assert _metric_close(frozen, frozen)
    assert primary["root_contract"]["root_coverage_over_v_domain"] == 1.0


def test_every_target_row_receives_exactly_one_nearest_gamma_index():
    target_v = np.asarray([0.0, 0.2, 0.5, 0.8, 1.0])
    assigned = nearest_gamma_assignment(target_v, np.asarray([0.0, 0.5, 1.0]))
    assert assigned.shape == target_v.shape
    assert np.all((assigned >= 0) & (assigned < 3))
    assert len(assigned) == len(target_v)


def test_supported_and_unsupported_target_sets_are_disjoint_and_exhaustive():
    assigned = np.asarray([0, 1, 1, 2, 0])
    supported, unsupported = partition_by_gamma_support(assigned, np.asarray([True, False, True]))
    assert not np.any(supported & unsupported)
    assert np.all(supported | unsupported)
    assert int(supported.sum() + unsupported.sum()) == len(assigned)


def test_isolated_unsupported_gamma_stays_unsupported_in_its_voronoi_cell():
    assigned = nearest_gamma_assignment(np.asarray([0.49, 0.50, 0.51]), np.asarray([0.0, 0.5, 1.0]))
    supported, unsupported = partition_by_gamma_support(assigned, np.asarray([True, False, True]))
    assert assigned.tolist() == [1, 1, 1]
    assert unsupported.tolist() == [True, True, True]
    assert not np.any(supported)


def test_correspondence_restricted_metric_cannot_use_another_gamma_column():
    points = np.asarray([[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], [[1.0, 0.0, 0.0], [101.0, 0.0, 0.0]]])
    normals = np.tile(np.asarray([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]), (2, 1, 1))
    restricted = correspondence_restricted_metrics(
        np.asarray([[100.0, 0.0, 0.0]]),
        np.asarray([0]),
        points,
        normals,
        np.asarray([0, 1]),
        1.0,
    )
    assert restricted["median_over_h"] == 99.0


def test_support_threshold_remains_exactly_two_h():
    curve = _small_curve(np.asarray([2.0, 2.000001]))
    contract = _supported_gamma_contract(curve, np.asarray([0, 1]), 0.01)
    assert contract["support_threshold_h"] == 2.0
    assert contract["supported_gamma_mask"].tolist() == [True, False]
    assert contract["supported_target_mask"].tolist() == [True, False]


def test_thin_root_diagnostic_detects_even_multiplicity_contact():
    u = np.linspace(0.0, 1.0, 257)
    values = (u - 0.5) ** 2
    assert _thin_root_classification(values, False, ROOT_RESIDUAL_TOLERANCE, ROOT_DIAGNOSTIC_TOLERANCE) == "possible_tangential_near_contact"


def test_curvature_code_is_gated_after_supported_first_order_failure():
    source = Path("devtools/demo/supported_termination_attribution.py").read_text(encoding="utf-8")
    complete = source.index("def _run_attribution_case_complete")
    gate = source.index("materially_poor_for_attribution", complete)
    curvature = source.index("_evaluate_directional_curvature", gate)
    assert gate < curvature
    assert '"second_order_candidate_executed": False' in source


def test_directional_curvature_uses_full_T_and_A_contractions():
    a = np.asarray([2.0])
    b = np.asarray([-1.0])
    su = np.asarray([[1.0, 0.0, 0.0]])
    sv = np.asarray([[0.0, 3.0, 0.0]])
    suu = np.asarray([[1.0, 2.0, 3.0]])
    suv = np.asarray([[4.0, 5.0, 6.0]])
    svv = np.asarray([[7.0, 8.0, 9.0]])
    tangent, curvature = directional_terms_from_derivatives(a, b, su, sv, suu, suv, svv)
    assert np.allclose(tangent, np.asarray([[2.0, -3.0, 0.0]]))
    assert np.allclose(curvature, 4.0 * suu - 4.0 * suv + svv)


def test_second_order_candidate_has_fixed_contract_and_no_q_scale():
    curve = _small_curve()
    tangent = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    curvature = np.asarray([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]])
    points, _normals, l_values = build_second_order_candidate(curve, tangent, curvature, PRIMARY_ROI, l_samples=3)
    grid = points.reshape(3, 2, 3)
    expected = curve.gamma_points[0] + l_values[-1] * tangent[0] + 0.5 * l_values[-1] ** 2 * curvature[0]
    assert np.allclose(grid[-1, 0], expected)
    assert "q *" not in inspect.getsource(build_second_order_candidate)


def test_curvature_gate_is_false_when_alignment_is_not_positive():
    report = {
        "residual_direction_summary": {"median_cosine": -0.1, "fraction_dot_positive": 0.2},
        "valid_gamma_fraction": 0.9,
        "fixed_bins": [{"median_residual_over_h": 1.0}, {"median_residual_over_h": 2.0}],
    }
    gate = _curvature_gate(report)
    assert gate["passed"] is False
