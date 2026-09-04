import numpy as np

from devtools.demo.worklog_157_same_region_tsdf_component_separation_topology_spatial_provenance_audit import (
    INTERVENING_CATEGORIES,
    POSITIVE_OFFSETS,
    _classify_intervening_scalar,
    _connectivity_accounting,
    _encode_cells,
    _separation_category,
    _synthetic_contracts,
)


def test_exact_lattice_displacement_categories_are_deterministic() -> None:
    assert _separation_category(np.asarray([1, 0, 0]), 1) == "FACE_TOUCH"
    assert _separation_category(np.asarray([1, 1, 0]), 1) == "EDGE_TOUCH"
    assert _separation_category(np.asarray([1, 1, 1]), 1) == "CORNER_TOUCH"
    assert _separation_category(np.asarray([2, 0, 0]), 2) == "ONE_CELL_AXIAL_GAP"
    assert _separation_category(np.asarray([2, 1, 0]), 2) == "OTHER_NEAR_GAP"
    assert _separation_category(np.asarray([17, 0, 0]), 17) == "REMOTE"


def test_intervening_state_classifier_has_no_connectivity_side_effect() -> None:
    assert _classify_intervening_scalar(authoritative=False, zero_surface=False, sample_exists=False, accepted=False, sample_region=-1, target_region=0) == "NOT_AUTHORITATIVE"
    assert _classify_intervening_scalar(authoritative=True, zero_surface=False, sample_exists=True, accepted=False, sample_region=-1, target_region=0) == "AUTHORITATIVE_NOT_ZERO_SURFACE"
    assert _classify_intervening_scalar(authoritative=True, zero_surface=True, sample_exists=True, accepted=True, sample_region=2, target_region=0) == "ZERO_SURFACE_DIFFERENT_REGION"
    assert _classify_intervening_scalar(authoritative=True, zero_surface=True, sample_exists=True, accepted=False, sample_region=0, target_region=0) == "ZERO_SURFACE_UNOWNED_OR_AMBIGUOUS"
    assert _classify_intervening_scalar(authoritative=True, zero_surface=True, sample_exists=True, accepted=True, sample_region=0, target_region=0) == "OTHER_EXISTING_CONTRACT_STATE"
    assert set(INTERVENING_CATEGORIES) == {
        "NOT_AUTHORITATIVE",
        "AUTHORITATIVE_NOT_ZERO_SURFACE",
        "ZERO_SURFACE_DIFFERENT_REGION",
        "ZERO_SURFACE_UNOWNED_OR_AMBIGUOUS",
        "OTHER_EXISTING_CONTRACT_STATE",
    }


def test_synthetic_connectivity_contracts_pass() -> None:
    result = _synthetic_contracts()
    assert result["all_pass"] is True
    assert result["diagnostic_mechanics_only"] is True


def test_diagnostic_18_and_26_connectivity_only_merge_lattice_contacts() -> None:
    cells = np.asarray([[0, 0, 0], [1, 1, 0], [1, 1, 1], [4, 0, 0]], dtype=np.int64)
    order = np.argsort(_encode_cells(cells), kind="stable")
    cells = cells[order]
    region = {
        "keys": _encode_cells(cells),
        "cells": cells,
        "native_components": np.asarray([0, 1, 2, 3], dtype=np.int64)[order],
        "population": {"component_ids": np.asarray([0, 1, 2, 3]), "component_sizes": np.asarray([1, 1, 1, 1]), "summary": {"owned_sample_count": 4, "native_component_count": 4}},
    }
    metrics, labels, audit = _connectivity_accounting(region)
    assert metrics["6"]["component_count"] == 4
    assert metrics["18"]["component_count"] == 2
    assert metrics["26"]["component_count"] == 2
    assert len(labels[18]) == len(cells)
    assert audit["edge_touch"]["pair_count"] == 1
    assert audit["corner_touch"]["pair_count"] == 1
