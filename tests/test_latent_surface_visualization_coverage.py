from __future__ import annotations

from dataclasses import replace

import torch

from osn_gs.surface.torch_latent_surface_coverage_audit import LatentSupportUnit
from osn_gs.surface.torch_latent_surface_visualization_coverage import (
    MIN_SUBDIVIDABLE_SIZE,
    materialize_unit_with_subdivision,
)


def _line_unit(count: int = 12) -> LatentSupportUnit:
    """A degenerate collinear unit -- a chain graph (each node connected
    only to its immediate neighbor), forcing subdivision to actually be
    exercised structurally even where the fitter itself is forgiving."""

    coords = torch.linspace(0.0, 1.0, count)
    positions = torch.stack([coords, torch.zeros(count), torch.zeros(count)], dim=1)
    edges = tuple((i, i + 1) for i in range(count - 1))
    return LatentSupportUnit(
        unit_id=0, node_indices=tuple(range(100, 100 + count)),
        raw_positions=positions, latent_positions=positions,
        projection_displacement=torch.zeros_like(positions), normals=torch.zeros_like(positions), edges=edges,
    )


def _disconnected_pair_edges_unit() -> LatentSupportUnit:
    # Two internally-connected triangles with NO edge between them --
    # simulates a unit whose OWN edge list already contains two disjoint
    # pieces (subdivision must never merge them back together).
    positions = torch.tensor([
        [0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0],
        [5.0, 5.0, 0.0], [5.1, 5.0, 0.0], [5.0, 5.1, 0.0],
    ])
    edges = ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3))
    return LatentSupportUnit(
        unit_id=1, node_indices=(10, 11, 12, 20, 21, 22),
        raw_positions=positions, latent_positions=positions,
        projection_displacement=torch.zeros_like(positions), normals=torch.zeros_like(positions), edges=edges,
    )


def _every_node_accounted(unit: LatentSupportUnit, materialized, unrepresented) -> bool:
    covered: set[int] = set()
    for fragment in materialized:
        covered.update(fragment.node_indices)
    for fragment in unrepresented:
        covered.update(fragment.node_indices)
    return covered == set(unit.node_indices)


def test_every_node_represented_or_explicitly_unrepresented():
    unit = _line_unit()
    materialized, unrepresented = materialize_unit_with_subdivision(unit)
    assert _every_node_accounted(unit, materialized, unrepresented)


def test_no_node_appears_in_both_materialized_and_unrepresented():
    unit = _line_unit()
    materialized, unrepresented = materialize_unit_with_subdivision(unit)
    materialized_nodes = {node for fragment in materialized for node in fragment.node_indices}
    unrepresented_nodes = {node for fragment in unrepresented for node in fragment.node_indices}
    assert not (materialized_nodes & unrepresented_nodes)


def test_subdivision_never_introduces_unsupported_source_nodes():
    unit = _line_unit()
    materialized, unrepresented = materialize_unit_with_subdivision(unit)
    all_reported = {node for fragment in materialized for node in fragment.node_indices}
    all_reported |= {node for fragment in unrepresented for node in fragment.node_indices}
    assert all_reported.issubset(set(unit.node_indices))


def test_subdivision_never_joins_disconnected_components():
    unit = _disconnected_pair_edges_unit()
    materialized, unrepresented = materialize_unit_with_subdivision(unit)
    # No single fragment may span both the {10,11,12} group and the
    # {20,21,22} group -- they were never connected in the unit's own edges.
    group_a = {10, 11, 12}
    group_b = {20, 21, 22}
    for fragment in materialized:
        nodes = set(fragment.node_indices)
        assert not (nodes & group_a and nodes & group_b)
    for fragment in unrepresented:
        nodes = set(fragment.node_indices)
        assert not (nodes & group_a and nodes & group_b)


def test_fragment_below_minimum_size_reports_unrepresented_not_infinite_recursion():
    # A single below-floor fragment with degenerate geometry that cannot
    # even reach 3 points must terminate as unrepresented, not recurse
    # forever.
    tiny = LatentSupportUnit(
        unit_id=2, node_indices=(1, 2),
        raw_positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        latent_positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        projection_displacement=torch.zeros((2, 3)), normals=torch.zeros((2, 3)), edges=((0, 1),),
    )
    materialized, unrepresented = materialize_unit_with_subdivision(tiny)
    assert _every_node_accounted(tiny, materialized, unrepresented)
    assert len(materialized) == 0  # 2 points can never materialize (MIN sample floor is 3)
    assert len(unrepresented) == 1
    assert unrepresented[0].node_indices == (1, 2)


def test_visualization_quality_labels_never_filter_geometry():
    import ast
    import inspect

    from osn_gs.surface import torch_latent_surface_visualization_coverage as module

    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            node.value = ast.Constant(value="")
    code_only = ast.unparse(tree)
    assert "unsafe" not in code_only.lower()
    assert "extrapolat" not in code_only.lower()
    assert "identifiab" not in code_only.lower()
    assert "chart" not in code_only.lower()


def test_provenance_preserved_through_subdivision():
    unit = _line_unit()
    materialized, _unrepresented = materialize_unit_with_subdivision(unit)
    for fragment in materialized:
        # fragment.result.surface exists and node_indices map back into the
        # ORIGINAL unit's own global node_indices, never a fabricated range.
        assert set(fragment.node_indices).issubset(set(unit.node_indices))


def test_estimator_and_support_results_unchanged_by_this_module():
    import ast
    import inspect

    from osn_gs.surface import torch_latent_surface_visualization_coverage as module

    tree = ast.parse(inspect.getsource(module))
    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("support" in name.lower() and "latent_surface_support" in name.lower() for name in imported_names)
    assert "LatentSupportUnit" in imported_names  # consumes, never reconstructs, the Worklog 103 unit
