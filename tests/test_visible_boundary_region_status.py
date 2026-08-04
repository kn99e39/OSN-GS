"""Worklog 54: region-level visible-boundary eligibility production contract.

Every region is classified into exactly one of five states
(`eligible_closed_boundary`, `open_observed_fragment`,
`insufficient_observation`, `ambiguous_boundary`, `rejected_unsafe`), and only
`eligible_closed_boundary` regions ever reach NURBS materialization. This
file tests the classifier directly against synthetic fixtures for each state,
plus an end-to-end check that materialization is actually restricted.
"""

from __future__ import annotations

import math
import unittest

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_directed_boundary_ordering import recover_directed_boundary_components
from osn_gs.surface.torch_visible_boundary_region_status import (
    STATUS_AMBIGUOUS,
    STATUS_ELIGIBLE_CLOSED,
    STATUS_INSUFFICIENT,
    STATUS_OPEN_FRAGMENT,
    STATUS_REJECTED_UNSAFE,
    classify_all_region_boundary_statuses,
)
from osn_gs.surface.torch_world_space_boundary_halfedges import WorldSpaceBoundaryHalfEdgeCandidate


def _ring_candidates(n: int, radius: float = 1.0, region_id: int = 0):
    candidates = []
    for i in range(n):
        angle = 2 * math.pi * i / n
        x, y = radius * math.cos(angle), radius * math.sin(angle)
        tangent = (-math.sin(angle), math.cos(angle), 0.0)
        candidates.append(WorldSpaceBoundaryHalfEdgeCandidate(
            half_edge_id=f"h{i}", source_region_id=region_id, source_gaussian_id=i, adjacent_gaussian_id=None,
            world_position=(x, y, 0.0), local_normal=(0.0, 0.0, 1.0),
            local_tangent_direction=tangent, boundary_direction=tangent,
            boundary_reason="observed_support_termination", source_pair_ids=None, confidence=0.7,
            ordering_state="locally_chainable", review_reasons=(),
        ))
    return candidates


def _ring_accepted_pairs(n: int):
    return [(i, (i + 1) % n) for i in range(n)]


class RegionStatusFiveWayClassificationTest(unittest.TestCase):
    def test_closed_ring_is_eligible_closed_boundary(self):
        n = 12
        candidates = _ring_candidates(n)
        accepted = _ring_accepted_pairs(n)
        _, components = recover_directed_boundary_components(candidates, accepted)
        statuses = classify_all_region_boundary_statuses((0,), components, candidates)
        self.assertEqual(len(statuses), 1)
        status = statuses[0]
        self.assertEqual(status.status, STATUS_ELIGIBLE_CLOSED)
        self.assertEqual(status.reason, "validated_simple_closed_loop")
        self.assertEqual(len(status.eligible_component_ids), 1)
        self.assertEqual(status.candidate_count, n)

    def test_open_arc_is_open_observed_fragment(self):
        # Same ring geometry but with the closing accepted-topology pair
        # removed -- the chain cannot close, but the rest genuinely links.
        n = 12
        candidates = _ring_candidates(n)
        accepted = _ring_accepted_pairs(n)[:-1]
        _, components = recover_directed_boundary_components(candidates, accepted)
        statuses = classify_all_region_boundary_statuses((0,), components, candidates)
        self.assertEqual(len(statuses), 1)
        status = statuses[0]
        self.assertEqual(status.status, STATUS_OPEN_FRAGMENT)
        self.assertEqual(status.reason, "best_available_downstream_valid_open_path")
        self.assertEqual(status.eligible_component_ids, ())

    def test_no_candidates_in_region_is_insufficient_observation(self):
        statuses = classify_all_region_boundary_statuses((0,), (), ())
        self.assertEqual(len(statuses), 1)
        status = statuses[0]
        self.assertEqual(status.status, STATUS_INSUFFICIENT)
        self.assertEqual(status.reason, "no_boundary_evidence_in_region")
        self.assertEqual(status.candidate_count, 0)

    def test_isolated_single_candidate_is_insufficient_observation(self):
        candidates = _ring_candidates(1)
        _, components = recover_directed_boundary_components(candidates, ())
        statuses = classify_all_region_boundary_statuses((0,), components, candidates)
        self.assertEqual(len(statuses), 1)
        status = statuses[0]
        self.assertEqual(status.status, STATUS_INSUFFICIENT)
        self.assertEqual(status.reason, "only_isolated_physical_candidates_no_linked_fragment")

    def test_only_typed_nonphysical_evidence_is_ambiguous_boundary(self):
        candidate = WorldSpaceBoundaryHalfEdgeCandidate(
            half_edge_id="h0", source_region_id=0, source_gaussian_id=0, adjacent_gaussian_id=None,
            world_position=(0.0, 0.0, 0.0), local_normal=(0.0, 0.0, 1.0),
            local_tangent_direction=(1.0, 0.0, 0.0), boundary_direction=(1.0, 0.0, 0.0),
            boundary_reason="ambiguous_continuation", source_pair_ids=None, confidence=0.3,
            ordering_state="ambiguous_ordering", review_reasons=(),
        )
        statuses = classify_all_region_boundary_statuses((0,), (), (candidate,))
        self.assertEqual(len(statuses), 1)
        status = statuses[0]
        self.assertEqual(status.status, STATUS_AMBIGUOUS)
        self.assertEqual(status.reason, "no_physical_termination_candidates_only_typed_nonphysical_evidence")
        self.assertEqual(status.candidate_count, 0)

    def test_self_intersecting_closed_loop_is_rejected_unsafe(self):
        # Bypass the solver entirely and hand-construct an
        # `OrderedBoundaryComponent` whose ordering is a bowtie (0-2-1-3) --
        # this deterministically exercises the classifier's own safety gate
        # regardless of which ordering the Hungarian solver would pick for
        # any particular compatibility graph.
        from osn_gs.surface.torch_ordered_world_boundary_graph import OrderedBoundaryComponent

        # Corners of a unit square in diagonal (crossing) order: h0=(0,0) ->
        # h1=(1,1) -> h2=(1,0) -> h3=(0,1) -> h0 draws both diagonals, which
        # cross at the square's center.
        positions = {0: (0.0, 0.0, 0.0), 1: (1.0, 1.0, 0.0), 2: (1.0, 0.0, 0.0), 3: (0.0, 1.0, 0.0)}
        order = ["h0", "h1", "h2", "h3"]
        candidates = [
            WorldSpaceBoundaryHalfEdgeCandidate(
                half_edge_id=f"h{node_id}", source_region_id=0, source_gaussian_id=node_id, adjacent_gaussian_id=None,
                world_position=position, local_normal=(0.0, 0.0, 1.0),
                local_tangent_direction=(1.0, 0.0, 0.0), boundary_direction=(1.0, 0.0, 0.0),
                boundary_reason="observed_support_termination", source_pair_ids=None, confidence=0.7,
                ordering_state="locally_chainable", review_reasons=(),
            )
            for node_id, position in positions.items()
        ]
        bowtie_component = OrderedBoundaryComponent(
            "region:0:directed:h0", 0, tuple(order), (0, 1, 2, 3), "ordered_closed_loop", True, (),
            {"observed_support_termination": 4}, 0.7, "outer_boundary_candidate", "reliable_core_only", False, (),
        )
        statuses = classify_all_region_boundary_statuses((0,), (bowtie_component,), candidates)
        self.assertEqual(len(statuses), 1)
        status = statuses[0]
        self.assertEqual(status.status, STATUS_REJECTED_UNSAFE)
        self.assertEqual(status.reason, "closed_loop_failed_self_intersection_check")
        self.assertEqual(status.eligible_component_ids, ())


class MaterializationRestrictedToEligibleRegionsTest(unittest.TestCase):
    def test_only_eligible_closed_boundary_regions_reach_materialization(self):
        scene = make_gaussian_reliability_scene("thin_slab")
        positions = torch.as_tensor(scene.positions, dtype=torch.float32)
        covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
        opacity = torch.ones(positions.shape[0])
        stable_ids = list(range(positions.shape[0]))
        pipeline = TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=64), device="cpu")
        bundle = pipeline._construct_canonical_with_full_evidence(positions, covariance, opacity, stable_ids)
        construction = bundle.construction
        eligible_region_ids = {
            status["region_id"] for status in construction.diagnostic_summary["region_boundary_statuses"]
            if status["status"] == STATUS_ELIGIBLE_CLOSED
        }
        for attempt in construction.materialization_attempts:
            self.assertIn(attempt.input.source_region_id, eligible_region_ids)
        # Every eligible region's approved component is exactly the one
        # attempted -- materialization never sees a component this contract
        # did not approve.
        eligible_component_ids = {
            component_id
            for status in construction.diagnostic_summary["region_boundary_statuses"]
            if status["status"] == STATUS_ELIGIBLE_CLOSED
            for component_id in status["eligible_component_ids"]
        }
        attempted_component_ids = {attempt.input.source_boundary_component_id for attempt in construction.materialization_attempts}
        self.assertEqual(attempted_component_ids, eligible_component_ids)

    def test_materialized_attempts_carry_region_status_provenance(self):
        scene = make_gaussian_reliability_scene("thin_slab")
        positions = torch.as_tensor(scene.positions, dtype=torch.float32)
        covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
        opacity = torch.ones(positions.shape[0])
        stable_ids = list(range(positions.shape[0]))
        pipeline = TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=64), device="cpu")
        bundle = pipeline._construct_canonical_with_full_evidence(positions, covariance, opacity, stable_ids)
        construction = bundle.construction
        # Worklog 55: thin_slab's front+back share one region with two
        # independently-validated closed loops -- no inner/hole detection
        # exists, so BOTH stay eligible and materialize (excluding either
        # regressed real materialization 3->2); the ambiguity is disclosed
        # via the reason string, never silently hidden, and materialization
        # count must not regress relative to worklog 54's baseline (3).
        self.assertEqual(construction.diagnostic_summary["materialized_surface_count"], 3)
        self.assertEqual(construction.diagnostic_summary["region_boundary_multiple_closed_loops_count"], 1)
        self.assertGreaterEqual(construction.diagnostic_summary["region_boundary_status_inconsistency_count"], 0)
        self.assertEqual(construction.diagnostic_summary["region_boundary_status_inconsistency_count"], 0)
        for attempt in construction.materialization_attempts:
            self.assertEqual(attempt.input.region_status, STATUS_ELIGIBLE_CLOSED)
            self.assertEqual(attempt.input.boundary_role_scope, "outer_boundary_only")
            self.assertTrue(attempt.input.region_status_reason)
            self.assertTrue(attempt.input.supporting_source_ids)
        # `eligible_materialized_surfaces()` is the sanctioned single entry
        # point and must be identical to `materialized_visible_nurbs_surfaces`.
        self.assertEqual(construction.eligible_materialized_surfaces(), construction.materialized_visible_nurbs_surfaces)


if __name__ == "__main__":
    unittest.main()
