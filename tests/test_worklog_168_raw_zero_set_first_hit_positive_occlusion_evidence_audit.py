from __future__ import annotations

import numpy as np
import pytest

from devtools.demo import worklog_167_raw_zero_set_ray_blocker_audit as w167
from devtools.demo.worklog_168_raw_zero_set_first_hit_positive_occlusion_evidence_audit import (
    FIXTURE_NAMES,
    _build_fixture,
    _discrete_support_boundary,
    _multi_surface_result,
    analytic_blocked,
    disagreement_interval_accounting,
    zero_set_blocked,
)


@pytest.fixture(scope="module")
def analytic_fixtures() -> dict[str, dict]:
    return {name: _build_fixture(name) for name in FIXTURE_NAMES}


def test_strict_first_hit_ordering_and_exact_boundary() -> None:
    z_s = 2.0
    assert bool(zero_set_blocked(z_s, np.nextafter(z_s, np.inf))) is True
    assert bool(zero_set_blocked(z_s, z_s)) is False
    assert bool(zero_set_blocked(z_s, np.nextafter(z_s, -np.inf))) is False
    assert bool(zero_set_blocked(z_s, 3.0, False)) is False
    assert bool(analytic_blocked(z_s, z_s)) is False


def test_complete_depth_interval_accounting_covers_every_disagreement_type() -> None:
    analytic_depth = np.asarray([2.0, 2.0, 2.0, np.nan, 2.0])
    analytic_hit = np.asarray([True, True, True, False, True])
    zero_depth = np.asarray([1.5, 2.5, 2.0, 1.0, np.nan])
    zero_status = np.asarray([w167.STATUS_HIT, w167.STATUS_HIT, w167.STATUS_HIT, w167.STATUS_HIT, w167.STATUS_NO_HIT])
    result = disagreement_interval_accounting(analytic_depth, analytic_hit, zero_depth, zero_status, h=0.5)
    assert result["premature_blocker_ray_count"] == 1
    assert result["delayed_blocker_ray_count"] == 1
    assert result["exact_first_depth_rays"] == 1
    assert result["false_zero_set_hit_rays"] == 1
    assert result["missed_zero_set_hit_rays"] == 1
    assert result["strict_positive_evidence_counterexample_count"] == 2
    assert result["premature_blocker_interval_length_over_h"]["max"] == 1.0
    assert result["delayed_blocker_interval_length_over_h"]["max"] == 1.0


def test_fixed_raster_boundary_split_uses_exact_adjacency() -> None:
    hit = np.ones((3, 3), dtype=bool)
    interior, boundary, exterior = _discrete_support_boundary(hit.reshape(-1), image=3)
    assert interior.reshape(3, 3)[1, 1]
    assert int(interior.sum()) == 1
    assert int(boundary.sum()) == 8
    assert int(exterior.sum()) == 0


def test_single_surface_fixtures_use_historical_extraction_and_complete_accounting(analytic_fixtures: dict[str, dict]) -> None:
    for name in ("fronto_parallel_plane", "oblique_plane", "curved_sphere"):
        fixture = analytic_fixtures[name]
        result = fixture["intervals"]
        assert len(fixture["vertices"]) > 0
        assert len(fixture["faces"]) > 0
        assert result["analytic_hit_rays"] > 0
        assert result["false_zero_set_hit_rays"] == 0
        assert result["both_hit_rays"] + result["missed_zero_set_hit_rays"] == result["analytic_hit_rays"]


def test_layered_fixture_uses_first_surface_not_target_identity(analytic_fixtures: dict[str, dict]) -> None:
    result = _multi_surface_result(analytic_fixtures["layered_two_sheet"])
    assert result["analytic_both_surface_hit_rays"] > 0
    assert result["auditable_zero_set_first_hit_rays"] > 0
    assert result["zero_set_first_hit_nearer_rear_sheet"] == 0
    assert result["zero_set_first_hit_nearer_front_sheet"] == result["auditable_zero_set_first_hit_rays"]
    assert result["rear_surface_associated_query"]["candidate_blocked_by_first_zero_set_count"] == result["auditable_zero_set_first_hit_rays"]
    assert result["target_identity_used"] is False
