from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.core.torch_pipeline import _representative_knn_spacing
from scripts.devtools.replay_termination_neighborhood_scale import replay_fixture


class TerminationNeighborhoodScaleReplayTest(unittest.TestCase):
    def test_dual_scale_replay_emits_required_lineage_and_counts(self):
        report = replay_fixture(make_gaussian_reliability_scene("box_face"))
        self.assertIn("stable_id_diff", report)
        for branch_name in ("footprint", "candidate"):
            branch = report[branch_name]
            for key in (
                "representative_count",
                "raw_termination_count",
                "normalized_termination_count",
                "physical_assertion_count",
                "duplicate_count",
                "directed_compatibility_edge_count",
                "ordered_component_count",
                "closed_loop_count",
                "open_chain_count",
                "branch_component_count",
                "seed_admission_count",
                "nurbs_materialization_count",
                "rejection_reason_histogram",
                "candidate_lineage",
                "false_support",
            ):
                self.assertIn(key, branch)
            self.assertGreater(branch["representative_count"], 0)
            if branch["candidate_lineage"]:
                record = branch["candidate_lineage"][0]
                for key in (
                    "source_representative_stable_id",
                    "supporting_representative_stable_ids",
                    "region_id",
                    "extraction_scale",
                    "numeric_radius",
                    "raw_candidate_id",
                    "normalized_candidate_id",
                    "typed_reason",
                    "sector_angular_evidence",
                    "directed_ordering_input_id",
                    "compatibility_edge_ids",
                    "ordered_component_id",
                    "component_state",
                    "seed_admission_result",
                    "nurbs_materialization_result",
                    "rejection_reason",
                ):
                    self.assertIn(key, record)

    def test_representative_knn_spacing_singleton_and_pair_are_finite(self):
        singleton = _representative_knn_spacing(torch.tensor([[1.0, 2.0, 3.0]]))
        self.assertEqual(tuple(singleton.shape), (1,))
        self.assertTrue(bool(torch.isfinite(singleton).all()))
        torch.testing.assert_close(singleton, torch.tensor([1e-9]))

        pair = _representative_knn_spacing(torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]))
        self.assertTrue(bool(torch.isfinite(pair).all()))
        torch.testing.assert_close(pair, torch.tensor([5.0, 5.0]))

    def test_physical_gate_trace_counts_production_candidate_before_sector_replay_gates(self):
        from osn_gs.surface.torch_world_space_boundary_halfedges import WorldSpaceBoundaryHalfEdgeCandidate
        from scripts.devtools.trace_physical_termination_gates import trace

        candidate = WorldSpaceBoundaryHalfEdgeCandidate(
            half_edge_id="region:0:gaussian:42:continuation:observed_support_termination",
            source_region_id=0,
            source_gaussian_id=42,
            adjacent_gaussian_id=None,
            world_position=(0.0, 0.0, 0.0),
            local_normal=(0.0, 0.0, 1.0),
            local_tangent_direction=(1.0, 0.0, 0.0),
            boundary_direction=(1.0, 0.0, 0.0),
            boundary_reason="observed_support_termination",
            source_pair_ids=None,
            confidence=0.7,
            ordering_state="locally_chainable",
            review_reasons=("full_cloud_continuation_shell_gap",),
            support_radius=1.0,
        )
        regions = SimpleNamespace(
            node_region_id=[0],
            node_membership_state=["core_member"],
            regions=[SimpleNamespace(region_id=0, member_ids=(42,), internal_accepted_edge_ids=())],
        )
        construction = SimpleNamespace(
            surface_regions=regions,
            boundary_halfedge_candidates=(candidate,),
            ordered_boundary_components=(),
        )
        frame = SimpleNamespace(
            oriented_normal=torch.tensor([0.0, 0.0, 1.0]),
            tangent_axis_0=torch.tensor([1.0, 0.0, 0.0]),
            tangent_axis_1=torch.tensor([0.0, 1.0, 0.0]),
        )
        report = trace(construction, torch.zeros((1, 3)), (42,), (frame,), torch.tensor([1.0]))
        self.assertEqual(report["first_failure_counts"].get("generated_physical_candidate"), 1)
        self.assertNotIn("no_neighbor_support", report["first_failure_counts"])


if __name__ == "__main__":
    unittest.main()
