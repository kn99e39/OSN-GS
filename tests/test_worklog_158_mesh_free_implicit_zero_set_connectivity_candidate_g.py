import numpy as np

from devtools.demo.worklog_158_mesh_free_implicit_zero_set_connectivity_candidate_g import (
    _candidate_g_bridge_allowed,
    _edge_entity_ids,
    _shared_edge_key,
    _synthetic_contracts,
)


def test_lattice_edge_identity_is_origin_invariant() -> None:
    vertices_a = np.asarray([[4.0, 0.5, 4.0], [4.0, 1.5, 4.0]], dtype=np.float64)
    vertices_b = np.asarray([[8.0, 4.5, 8.0], [8.0, 5.5, 8.0]], dtype=np.float64)
    ids_a, valid_a = _edge_entity_ids(vertices_a, np.asarray([4, 4, 4]), 16)
    ids_b, valid_b = _edge_entity_ids(vertices_b, np.asarray([0, 0, 0]), 16)
    assert valid_a.tolist() == [True, True]
    assert valid_b.tolist() == [True, True]
    assert ids_a.tolist() == ids_b.tolist()


def test_corner_vertex_is_not_silently_promoted_to_an_edge() -> None:
    ids, valid = _edge_entity_ids(np.asarray([[1.0, 1.0, 1.0]]), np.zeros(3, dtype=np.int64), 4)
    assert valid.tolist() == [False]
    assert ids.shape == (1,)


def test_shared_edge_key_is_canonical_for_diagonal_cell_pair() -> None:
    assert _shared_edge_key(np.asarray([0, 0, 0]), np.asarray([1, 1, 0])) == _shared_edge_key(
        np.asarray([1, 1, 0]), np.asarray([0, 0, 0])
    )


def test_ownership_and_gap_guards_forbid_non_topological_bridges() -> None:
    assert _candidate_g_bridge_allowed(
        same_region=True,
        source_zero_surface=True,
        neighbor_zero_surface=True,
        intervening_state="NONE",
        shared_entity=True,
    )
    for kwargs in (
        {"same_region": True, "source_zero_surface": True, "neighbor_zero_surface": True, "intervening_state": "AUTHORITATIVE_NOT_ZERO_SURFACE", "shared_entity": True},
        {"same_region": True, "source_zero_surface": True, "neighbor_zero_surface": True, "intervening_state": "NOT_AUTHORITATIVE", "shared_entity": True},
        {"same_region": False, "source_zero_surface": True, "neighbor_zero_surface": True, "intervening_state": "NONE", "shared_entity": True},
        {"same_region": True, "source_zero_surface": False, "neighbor_zero_surface": True, "intervening_state": "NONE", "shared_entity": True},
        {"same_region": True, "source_zero_surface": True, "neighbor_zero_surface": True, "intervening_state": "NONE", "shared_entity": False},
    ):
        assert not _candidate_g_bridge_allowed(**kwargs)


def test_synthetic_connectivity_contracts_pass() -> None:
    result = _synthetic_contracts()
    assert result["all_pass"] is True
    assert result["diagnostic_mechanics_only"] is True
    assert all(case["pass"] for case in result["cases"])
