import numpy as np

from devtools.demo.worklog_159_partial_zero_set_topology_ambiguity_contract_audit import (
    AMBIGUOUS_CODES,
    CATEGORY_TO_CODE,
    _ambiguity_leverage_fast,
    _build_candidate_h_fast,
    _classify_cell,
    _guaranteed_relation,
    _local_patch_accounting,
    _synthetic_contracts_w159,
)


def test_local_patch_accounting_separates_same_cell_patches() -> None:
    result = _local_patch_accounting(
        np.asarray([10, 10, 10]),
        np.asarray([[1, 2, 3], [3, 4, 5], [6, 7, 8]], dtype=np.int64),
    )
    assert result["cell_keys"].tolist() == [10]
    assert result["triangle_counts"].tolist() == [3]
    assert result["patch_counts"].tolist() == [2]


def test_exact_zero_and_decider_tie_are_ambiguous_contracts() -> None:
    assert _classify_cell(np.asarray([0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]), False)["category"] == "EXACT_ZERO_VERTEX_DEGENERACY"
    assert _classify_cell(np.asarray([0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]), False)["category"] == "EXACT_ZERO_EDGE_FACE_DEGENERACY"
    assert _classify_cell(np.asarray([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0]), False)["category"] == "LEWINER_DETERMINISTIC_BUT_FIELD_UNDERDETERMINED"


def test_guaranteed_states_keep_ambiguous_separate_from_hard_disconnect() -> None:
    assert _guaranteed_relation(same_region=True, source_state="DETERMINISTIC", neighbor_state="DETERMINISTIC", shared_entity=True) == "GUARANTEED_CONNECT"
    assert _guaranteed_relation(same_region=True, source_state="DETERMINISTIC", neighbor_state="DETERMINISTIC", shared_entity=False) == "GUARANTEED_DISCONNECT"
    assert _guaranteed_relation(same_region=True, source_state="AMBIGUOUS", neighbor_state="DETERMINISTIC", shared_entity=True) == "TOPOLOGY_AMBIGUOUS"
    assert _guaranteed_relation(same_region=False, source_state="DETERMINISTIC", neighbor_state="DETERMINISTIC", shared_entity=True) == "GUARANTEED_DISCONNECT"


def test_candidate_h_omits_ambiguous_triangle_and_keeps_deterministic_component() -> None:
    region = {"keys": np.asarray([10, 11]), "cells": np.asarray([[0, 0, 0], [1, 0, 0]]), "native_components": np.asarray([0, 1])}
    classified = {"category_codes": np.asarray([CATEGORY_TO_CODE["DETERMINISTIC_SINGLE_PATCH"], CATEGORY_TO_CODE["EXACT_ZERO_VERTEX_DEGENERACY"]], dtype=np.int8)}
    incidence = {"triangle_cells": np.asarray([10, 10, 11]), "triangle_entities": np.asarray([[1, 2, 3], [3, 4, 1], [4, 5, 6]], dtype=np.int64), "ambiguous_cells": np.asarray([11])}
    graph = _build_candidate_h_fast(region, incidence, classified)
    assert graph["deterministic_triangle_count"] == 2
    assert graph["component_count"] == 1
    assert graph["cell_component"].tolist() == [0, -1]


def test_ambiguity_leverage_reports_macro_identity_change_without_promoting_it() -> None:
    region = {"keys": np.asarray([10]), "cells": np.asarray([[0, 0, 0]])}
    classified = {"category_codes": np.asarray([CATEGORY_TO_CODE["EXACT_ZERO_VERTEX_DEGENERACY"]], dtype=np.int8), "genuine_ambiguity_mask": np.asarray([True])}
    graph = {"entity_keys": np.asarray([1, 2]), "entity_labels": np.asarray([0, 1], dtype=np.int32), "component_count": 2, "component_cell_sizes": np.asarray([10, 10])}
    incidence = {"triangle_cells": np.asarray([10]), "triangle_entities": np.asarray([[1, 2, 1]], dtype=np.int64)}
    leverage = _ambiguity_leverage_fast(region, incidence, classified, graph)
    assert leverage["ambiguous_cells_touching_multiple_components"] == 1
    assert leverage["macro_identity_change_possible"] is True
    assert leverage["hypothetical_component_count_reduction"] == 1


def test_synthetic_w159_a_to_h_contracts_pass() -> None:
    result = _synthetic_contracts_w159()
    assert result["all_pass"] is True
    assert result["multi_patch_is_not_ambiguity"] is True
    assert all(case["pass"] for case in result["cases"])