from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from devtools.demo import worklog_172_fragmentation_gate_vs_structural_fit_audit as w172
from osn_gs.surface import torch_gaussian_region_owned_tsdf as native

w171 = w172.w171


@pytest.fixture(scope="module")
def real_cases():
    return w171._load_real_runtimes(w171.DEFAULT_W154, w171.DEFAULT_W145)


@pytest.fixture(scope="module")
def report():
    return json.loads((w172.OUT / "worklog_172_report.json").read_text(encoding="utf-8"))


def test_frozen_w171_baseline_and_fitter_files_unchanged(report):
    assert report["baseline_files_unchanged"]
    assert report["baseline_and_fitter_manifest"] == w172.frozen_manifest()
    assert set(report["cases"]) == set(w172.CASES)
    assert not report["mixed_contact_fit_attempted"]


def test_largest_component_tie_break_is_independent_of_enumeration():
    samples = SimpleNamespace(source_cell_keys=torch.tensor([10, 20, 30, 40, 50]))
    first = SimpleNamespace(sample_indices=torch.tensor([0, 1]), region_id=2)
    second = SimpleNamespace(sample_indices=torch.tensor([2, 3]), region_id=1)
    tiny = SimpleNamespace(sample_indices=torch.tensor([4]), region_id=0)
    assert w172.largest_component(samples, (tiny, second, first)) is first
    assert w172.largest_component(samples, (first, second, tiny)) is first


@pytest.mark.parametrize("name", w172.CASES)
def test_real_baseline_identity_rows_and_unchanged_boundary_fitter(real_cases, report, name, monkeypatch):
    complete = real_cases[name]
    calls = []
    assert w171.derive_native_support_boundary is native.derive_native_support_boundary
    assert w171.fit_boundary_first_region_representative is native.fit_boundary_first_region_representative
    for function_name in ("derive_native_support_boundary", "fit_boundary_first_region_representative"):
        original = getattr(native, function_name)

        def wrapper(*args, _original=original, _name=function_name, **kwargs):
            calls.append((_name, kwargs))
            return _original(*args, **kwargs)

        monkeypatch.setattr(w171, function_name, wrapper)
    # No fit-parameter override: the W154 implementation defaults are frozen.
    signature = inspect.signature(native.fit_boundary_first_region_representative)
    for key, value in w171.FIT_CONTRACT.items():
        if key in signature.parameters:
            assert signature.parameters[key].default == value
    stored = report["cases"][name]
    diagnostic, indices, result = w172.diagnose(complete, stored["baseline"])
    assert w171._jsonable(result) == stored
    chosen = w172.largest_component(complete.samples, complete.components)
    reversed_choice = w172.largest_component(complete.samples, tuple(reversed(complete.components)))
    assert chosen.component_id == reversed_choice.component_id == stored["diagnostic_component_id"]
    assert int(chosen.sample_indices.numel()) == stored["baseline"]["components"]["size"]["max"]
    assert calls == [("derive_native_support_boundary", {}), ("fit_boundary_first_region_representative", {})]
    with np.load(w172.OUT / "case_artifacts" / name / "selected_support.npz") as archive:
        assert np.array_equal(archive["complete_row_indices"], indices)
        for key in w172.ROW_FIELDS:
            assert np.array_equal(archive[key], getattr(diagnostic.samples, key).numpy())
    assert result["diagnostic"]["reason"] == "native_support_boundary_has_multiple_loops"
    assert result["diagnostic"]["solvability"]["design_matrix_rank"] is None
    assert not result["least_squares_attempted"]
    for units in ("world_units", "normalized_by_h"):
        stats = result["diagnostic"]["fit_residual"][units]
        assert stats["count"] == 0
        assert all(stats[key] is None for key in ("min", "median", "mean", "p95", "max"))


def test_generated_review_artifacts_are_valid_and_complete(report):
    assert w172.validate_artifacts(w172.OUT) == report["visualization"]
    assert report["visualization"]["png_count"] == 8
    for name in w172.CASES:
        assert not (w172.OUT / "case_artifacts" / name / "bounded_nurbs.npz").exists()
        with np.load(w172.OUT / "case_artifacts" / name / "boundary_chart.npz") as archive:
            assert len(archive["loop_offsets"]) - 1 == report["cases"][name]["diagnostic"]["boundary"]["loop_count"]
