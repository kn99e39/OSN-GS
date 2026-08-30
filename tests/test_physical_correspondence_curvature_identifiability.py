"""Focused Worklog 133 physical-correspondence closure tests."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from devtools.demo.parametric_continuation_attribution import _roi_vertex_contract
from devtools.demo.parametric_surface_continuation import PRIMARY_ROI
from devtools.demo.physical_correspondence_curvature_identifiability import (
    _assignment_contract,
    boundary_bias_contract,
    curvature_identifiability_report,
    gamma_v_audit,
    physical_normalized_v,
    residual_bin_report,
)


MODULE_SOURCE = Path("devtools/demo/physical_correspondence_curvature_identifiability.py").read_text(encoding="utf-8")


def test_physical_v_uses_the_frozen_roi_coordinate_definition():
    config = PRIMARY_ROI
    origin = np.asarray(config.origin, dtype=np.float64)
    axis_v = np.asarray(config.axis_v, dtype=np.float64)
    points = np.stack(
        [
            origin + axis_v * float(config.v_bounds[0]),
            origin + axis_v * float(config.v_bounds[1]),
        ],
        axis=0,
    )
    assert np.allclose(physical_normalized_v(points, config), [0.0, 1.0])
    audit = gamma_v_audit(points, np.asarray([[0.0, 0.0], [0.0, 1.0]]), config)
    assert np.allclose(audit["gamma_v_physical"], audit["gamma_v_parametric"])
    assert audit["median_absolute_difference"] == 0.0


def test_roi_holdout_is_deterministic_and_boundary_attached():
    config = PRIMARY_ROI
    origin = np.asarray(config.origin, dtype=np.float64)
    axis_u = np.asarray(config.axis_u, dtype=np.float64)
    axis_v = np.asarray(config.axis_v, dtype=np.float64)
    axis_n = np.asarray(config.axis_n, dtype=np.float64)
    v = sum(config.v_bounds) / 2.0
    n = sum(config.n_bounds) / 2.0
    points = np.stack(
        [
            origin + axis_u * (config.u_bounds[0] + 0.25 * (config.u_bounds[1] - config.u_bounds[0])) + axis_v * v + axis_n * n,
            origin + axis_u * (config.u_bounds[0] + 0.80 * (config.u_bounds[1] - config.u_bounds[0])) + axis_v * v + axis_n * n,
        ],
        axis=0,
    )
    first = _roi_vertex_contract(points, config)
    second = _roi_vertex_contract(points, config)
    assert np.array_equal(first.observed_mask, second.observed_mask)
    assert np.array_equal(first.withheld_mask, second.withheld_mask)
    assert first.observed_mask.tolist() == [True, False]
    assert first.withheld_mask.tolist() == [False, True]


def test_physical_assignment_is_unique_and_ties_choose_lowest_gamma():
    assignment = _assignment_contract(
        target_v=np.asarray([0.25, 0.50, 0.75]),
        gamma_v_parametric=np.asarray([0.0, 0.60, 1.0]),
        gamma_v_physical=np.asarray([0.0, 0.50, 1.0]),
        valid_gamma_indices=np.asarray([0, 1, 2]),
        supported_gamma=np.asarray([True, False, True]),
    )
    assert assignment["old_assignment"].tolist() == [0, 1, 1]
    assert assignment["physical_assignment"].tolist() == [0, 1, 1]
    assert assignment["unique_old_assignment"] is True
    assert assignment["unique_physical_assignment"] is True
    tie = _assignment_contract(
        target_v=np.asarray([0.25]),
        gamma_v_parametric=np.asarray([0.0, 0.50]),
        gamma_v_physical=np.asarray([0.0, 0.50]),
        valid_gamma_indices=np.asarray([4, 9]),
        supported_gamma=np.asarray([True] * 10),
    )
    assert tie["physical_assignment"].tolist() == [4]


def test_assignment_and_bias_selection_do_not_read_withheld_xyz_or_error():
    assignment_source = inspect.getsource(_assignment_contract)
    bias_source = inspect.getsource(boundary_bias_contract)
    assert "reference_eval_points" not in assignment_source
    assert "prediction" not in assignment_source
    assert "reference_eval_points" not in bias_source
    assert '"selection_uses_target_error": False' in bias_source


def test_fixed_bins_and_curvature_signal_are_deterministic_and_evaluation_only():
    h = 1.0
    target = np.zeros((3, 3), dtype=np.float64)
    l = np.asarray([0.5, 1.0, 3.0], dtype=np.float64)
    raw = np.column_stack([np.zeros(3), np.zeros(3), l * l])
    delta = raw * 0.5
    curvature = np.tile(np.asarray([[0.0, 0.0, 2.0]]), (3, 1))
    first = residual_bin_report(target, l, raw, delta, curvature, h)
    second = residual_bin_report(target, l, raw, delta, curvature, h)
    assert first == second
    report = curvature_identifiability_report(target, l, raw, delta, curvature, 0.5, h)
    first_bin = report["fixed_bins"][0]
    assert first_bin["target_count"] == 2
    assert np.isclose(first_bin["median_curvature_signal_over_h"], 0.625)
    assert report["classification_basis"].endswith("no fitted threshold or q scale")


def test_meeting_demo_replays_frozen_fit_without_a_new_fitter_or_candidate():
    assert "fit_torch_visible_surface_lsq" not in MODULE_SOURCE
    assert "build_second_order_candidate" not in MODULE_SOURCE
    assert "X2" not in MODULE_SOURCE
    assert '"second_order_candidate_constructed": False' in MODULE_SOURCE
    assert '"withheld_geometry_used_for_bias_selection": False' in MODULE_SOURCE
    assert '"target_xyz_used_for_fit_or_prediction": False' in MODULE_SOURCE


def test_generated_primary_report_closes_frozen_identity_and_withheld_accounting():
    report_path = Path("output/confirmed/demo_physical_correspondence_curvature_identifiability/physical_correspondence_curvature_identifiability_report.json")
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    primary = next(case for case in report["cases"] if case["roi"]["name"] == PRIMARY_ROI.name)
    assert report["architecture_verdict"] in {"A", "B", "C", "D"}
    assert primary["wl132_identity"]["status"] == "PASS"
    assert all(primary["wl132_identity"]["checks"].values())
    assert primary["target_population_contract"]["withheld_geometry_used_for_bias_selection"] is False
    assert primary["physical_correspondence"]["populations"]["SUPPORTED_TARGET"]["correspondence_restricted"]["samples"] == primary["assignment_contract"]["supported_target_count_physical"]
