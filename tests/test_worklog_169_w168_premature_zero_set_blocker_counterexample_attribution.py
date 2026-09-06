from __future__ import annotations

import tempfile
from decimal import Decimal
from pathlib import Path

import numpy as np

from devtools.demo import worklog_167_raw_zero_set_ray_blocker_audit as w167
from devtools.demo import worklog_168_raw_zero_set_first_hit_positive_occlusion_evidence_audit as w168
from devtools.demo.worklog_169_w168_premature_zero_set_blocker_counterexample_attribution import (
    ATTR_GEOMETRIC,
    ATTR_MIXED,
    ATTR_NUMERICAL,
    analytic_surface_residual,
    attribute_fixture,
    classify_attribution,
    high_precision_analytic_intersection,
    high_precision_surface_residual,
    high_precision_triangle_intersection,
    reproduce_w168_fixture,
)


def test_analytic_residual_evaluation_for_plane_and_sphere() -> None:
    plane = w167._plane_surface("plane")
    points = np.asarray([[0.0, 0.0, -0.25], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])
    np.testing.assert_allclose(analytic_surface_residual(plane, points), [-0.25, 0.0, 0.5])
    sphere = w167._sphere_surface()
    surface_point = sphere.center + np.asarray([sphere.radius, 0.0, 0.0])
    assert analytic_surface_residual(sphere, surface_point)[0] == 0.0
    assert high_precision_surface_residual(plane, (Decimal(0), Decimal(0), Decimal(0))) == 0


def test_high_precision_reference_preserves_exact_and_front_shifted_geometry() -> None:
    plane = w167._plane_surface("plane")
    origin = np.asarray([0.0, 0.0, -2.0])
    direction = np.asarray([0.0, 0.0, 1.0])
    exact_triangle = np.asarray([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    front_triangle = exact_triangle.copy()
    front_triangle[:, 2] = -1.0e-9
    analytic_depth = high_precision_analytic_intersection(plane, origin, direction)
    exact = high_precision_triangle_intersection(origin, direction, exact_triangle)
    front = high_precision_triangle_intersection(origin, direction, front_triangle)
    assert exact["valid"] and front["valid"] and analytic_depth is not None
    assert exact["t"] - analytic_depth == 0
    assert front["t"] - analytic_depth < 0


def test_premature_attribution_classifies_geometry_numerical_and_mixed_without_epsilon() -> None:
    assert classify_attribution([Decimal("-1e-30")])["verdict"] == ATTR_GEOMETRIC
    assert classify_attribution([Decimal(0)])["verdict"] == ATTR_NUMERICAL
    result = classify_attribution([Decimal("-1e-30"), Decimal(0), Decimal("1e-30")])
    assert result["verdict"] == ATTR_MIXED
    assert result["geometric_front_bias_count"] == 1
    assert result["numerical_ordering_only_count"] == 2


def test_layered_fixture_reproduction_and_magnitude_accounting() -> None:
    fixture = w168._build_fixture("layered_two_sheet")
    frozen = w168.DEFAULT_OUT / "synthetic/layered_two_sheet/fixture_audit.npz"
    reproduction = reproduce_w168_fixture(fixture, frozen)
    assert reproduction["all_exact"] is True
    with tempfile.TemporaryDirectory(prefix="wl169_", dir=w168.REPO_ROOT / "output") as directory:
        report = attribute_fixture(fixture, frozen, Path(directory))
    assert report["premature_ray_count"] == fixture["intervals"]["premature_blocker_ray_count"]
    assert report["float64_premature_interval_length"]["count"] == report["premature_ray_count"]
    attributed = report["attribution"]
    assert attributed["geometric_front_bias_count"] + attributed["numerical_ordering_only_count"] + attributed["higher_precision_unresolved_count"] == report["premature_ray_count"]
