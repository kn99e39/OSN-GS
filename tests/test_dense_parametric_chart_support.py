"""Worklog 80: dense parametric chart support (topology/geometry separation)."""

from __future__ import annotations

import math
import unittest

import torch

from osn_gs.surface.torch_dense_parametric_chart_support import (
    STATE_COVERAGE_FAILED,
    STATE_MATERIALIZED,
    STATE_NO_DENSE_SUPPORT,
    STATE_UNRESOLVED_TOPOLOGY,
    build_dense_chart_support,
    independent_chart_components,
)

AXIS_U = torch.tensor([1.0, 0.0, 0.0])
AXIS_V = torch.tensor([0.0, 1.0, 0.0])
ORIGIN = torch.zeros(3)


def _ring(count: int, radius: float) -> torch.Tensor:
    angles = torch.arange(count, dtype=torch.float32) / count * 2 * math.pi
    return torch.stack((radius * torch.cos(angles), radius * torch.sin(angles), torch.zeros(count)), dim=1)


def _disc(count_per_axis: int, radius: float) -> torch.Tensor:
    axis = torch.linspace(-radius, radius, count_per_axis)
    u, v = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack((u.reshape(-1), v.reshape(-1), torch.zeros(count_per_axis ** 2)), dim=1)
    return points[points[:, :2].norm(dim=1) <= radius * 0.95]


def _build(sparse, kinds, dense, evidence, spacing=0.25, **kw):
    return build_dense_chart_support(
        0, sparse, kinds, list(range(int(dense.shape[0]))), dense, evidence,
        axis_u=AXIS_U, axis_v=AXIS_V, origin=ORIGIN, full_evidence_spacing=spacing, **kw,
    )


class TopologyGeometrySeparationTest(unittest.TestCase):
    """The redesign's central claim: the sparse cycle orders and types the
    perimeter; the dense observed support defines where it actually is."""

    def _small_sparse_triangle(self) -> torch.Tensor:
        # Deliberately tiny relative to the evidence -- the worklog 79 shape.
        return _ring(3, 0.15)

    def test_chart_geometry_comes_from_dense_support_not_representatives(self):
        sparse = self._small_sparse_triangle()
        dense = _ring(24, 1.0)
        evidence = _disc(12, 1.0)
        result = _build(sparse, ["crease"] * 3, dense, evidence)
        self.assertEqual(result.state, STATE_MATERIALIZED)
        # None of the sparse representative positions may appear as vertices.
        for vertex in result.ordered_positions:
            self.assertGreater(float((sparse - vertex[None, :]).norm(dim=1).min()), 1e-3)
        # And the chart must now reach the evidence scale, not the sparse scale.
        extent = float((result.ordered_positions - result.ordered_positions.mean(0)).norm(dim=1).max())
        self.assertGreater(extent, 0.9)

    def test_coverage_contract_is_satisfied_by_the_dense_domain(self):
        result = _build(self._small_sparse_triangle(), ["crease"] * 3, _ring(24, 1.0), _disc(12, 1.0))
        self.assertEqual(result.state, STATE_MATERIALIZED)
        self.assertIsNotNone(result.evidence_outside_domain_fraction)
        self.assertLess(result.evidence_outside_domain_fraction, 0.5)

    def test_the_sparse_polygon_alone_would_have_failed_coverage(self):
        # Same evidence, but the OLD representation (sparse triangle as the
        # chart) leaves nearly all evidence outside -- this is the worklog 79
        # failure, reproduced here as the control for the redesign.
        from osn_gs.surface.torch_region_owned_full_evidence import evidence_outside_chart_domain_fraction
        outside = evidence_outside_chart_domain_fraction(self._small_sparse_triangle(), _disc(12, 1.0))
        self.assertGreater(outside, 0.9)

    def test_typed_frontier_provenance_is_preserved_per_arc(self):
        sparse = self._small_sparse_triangle()
        kinds = ["physical_termination", "crease", "observation_frontier"]
        result = _build(sparse, kinds, _ring(24, 1.0), _disc(12, 1.0))
        self.assertEqual(result.state, STATE_MATERIALIZED)
        seen = {segment.segment_kind for segment in result.segments}
        self.assertTrue(seen.issubset(set(kinds)))
        self.assertEqual(len(seen), 3, "every sparse arc's type must survive into the dense chart")

    def test_sparse_topology_node_count_is_recorded_separately_from_geometry(self):
        result = _build(self._small_sparse_triangle(), ["crease"] * 3, _ring(24, 1.0), _disc(12, 1.0))
        self.assertEqual(result.sparse_topology_node_count, 3)
        self.assertGreater(len(result.ordered_ids), 3)


class FailClosedTest(unittest.TestCase):
    def test_no_dense_support_never_falls_back_to_the_sparse_polygon(self):
        sparse = _ring(3, 0.15)
        result = _build(sparse, ["crease"] * 3, sparse[:0], _disc(12, 1.0))
        self.assertEqual(result.state, STATE_NO_DENSE_SUPPORT)
        self.assertEqual(result.ordered_ids, ())

    def test_sparse_cycle_below_three_nodes_is_unresolved(self):
        sparse = _ring(2, 0.15)
        result = _build(sparse, ["crease"] * 2, _ring(24, 1.0), _disc(12, 1.0))
        self.assertEqual(result.state, STATE_UNRESOLVED_TOPOLOGY)

    def test_dense_support_that_does_not_contain_the_evidence_fails_coverage(self):
        # Dense support clustered on one side only; evidence spread across the
        # whole disc -> the domain cannot claim that evidence.
        angles = torch.linspace(0.0, 0.6, 12)
        dense = torch.stack((torch.cos(angles), torch.sin(angles), torch.zeros(12)), dim=1)
        result = _build(_ring(3, 0.15), ["crease"] * 3, dense, _disc(12, 1.0))
        self.assertIn(result.state, (STATE_COVERAGE_FAILED, STATE_NO_DENSE_SUPPORT))

    def test_geometry_report_is_always_produced_for_a_built_loop(self):
        result = _build(_ring(3, 0.15), ["crease"] * 3, _ring(24, 1.0), _disc(12, 1.0))
        self.assertIsNotNone(result.geometry)
        self.assertEqual(result.geometry.proper_crossing_count, 0)

    def test_arc_without_dense_support_is_disclosed_not_invented(self):
        # Observed support covers only two of the three sparse arcs. The third
        # must be reported as unsupported, never filled in with invented
        # geometry -- and the loop must not silently claim the missing span.
        sparse = _ring(3, 0.5)
        angles = torch.linspace(math.radians(10), math.radians(230), 14)
        dense = torch.stack((1.2 * torch.cos(angles), 1.2 * torch.sin(angles), torch.zeros(14)), dim=1)
        result = _build(sparse, ["crease"] * 3, dense, dense)
        self.assertEqual(result.arc_support_counts, (7, 6, 0))
        self.assertEqual(result.unsupported_arc_count, 1)
        self.assertEqual(sum(result.arc_support_counts), len(result.ordered_ids))


class IndependentChartComponentsTest(unittest.TestCase):
    def test_two_disjoint_cycles_are_two_charts(self):
        members = ["a", "b", "c", "x", "y", "z"]
        edges = [("a", "b"), ("b", "c"), ("c", "a"), ("x", "y"), ("y", "z"), ("z", "x")]
        self.assertEqual(len(independent_chart_components(members, edges)), 2)

    def test_a_single_interwoven_two_core_is_not_split(self):
        # Two triangles sharing an edge: one 2-core component, several cycles.
        # Ambiguous branching -- must NOT be reported as two charts.
        members = ["a", "b", "c", "d"]
        edges = [("a", "b"), ("b", "c"), ("c", "a"), ("b", "d"), ("d", "c")]
        self.assertEqual(len(independent_chart_components(members, edges)), 1)

    def test_pendant_nodes_are_stripped_and_cannot_form_a_chart(self):
        members = ["a", "b", "c", "tail"]
        edges = [("a", "b"), ("b", "c"), ("c", "a"), ("a", "tail")]
        components = independent_chart_components(members, edges)
        self.assertEqual(len(components), 1)
        self.assertNotIn("tail", components[0])

    def test_open_chain_yields_no_chart_component(self):
        members = ["a", "b", "c"]
        edges = [("a", "b"), ("b", "c")]
        self.assertEqual(independent_chart_components(members, edges), ())


if __name__ == "__main__":
    unittest.main()
