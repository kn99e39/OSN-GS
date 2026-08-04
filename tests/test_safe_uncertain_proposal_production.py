from __future__ import annotations

import unittest

import torch

from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
from osn_gs.core.torch_pipeline import TorchOSNGSPipeline, TorchPipelineConfig
from osn_gs.surface.torch_eligible_boundary_continuation_bridge import (
    EligibleBoundaryContinuationBridgeResult,
    build_eligible_boundary_continuation_bridge,
)
from osn_gs.surface.torch_occluded_region_candidate import build_geometric_region_candidates
from osn_gs.surface.torch_patch_boundary import PatchBoundarySegment
from osn_gs.surface.torch_safe_uncertain_proposal_production import (
    STATUS_DOMAIN_NOT_CANDIDATE_READY,
    STATUS_PROPOSED,
    build_safe_uncertain_proposals_from_bridge,
    run_safe_uncertain_proposals_from_gaussians,
)
from tests.test_occluded_region_candidate import _facing_planar


def _boundary(domain):
    world = domain.world[:, 0]
    n = int(world.shape[0])
    uv = torch.stack([torch.linspace(0, 1, n, dtype=world.dtype), torch.zeros(n, dtype=world.dtype)], 1)
    tangent = torch.nn.functional.normalize(world[1] - world[0], dim=0).expand(n, 3).clone()
    normal = torch.tensor([0.0, 0.0, 1.0], dtype=world.dtype).expand(n, 3).clone()
    return PatchBoundarySegment(
        domain.source_boundary_id, domain.source_patch_id, "eligible_visible_boundary", uv, world,
        uv, world, tangent, torch.zeros_like(tangent), normal, False, "ccw",
        provenance={"region_id": domain.source_patch_id, "supporting_source_ids": [domain.source_patch_id]},
    )


def _manual_bridge(*, state_a="valid", state_b="valid"):
    a, b = _facing_planar(gap=0.2)
    a.state, b.state = state_a, state_b
    candidates = build_geometric_region_candidates([a, b], {})
    return EligibleBoundaryContinuationBridgeResult(
        attempts=(), boundaries_by_id={a.source_boundary_id: _boundary(a), b.source_boundary_id: _boundary(b)},
        continuation_domains=(a, b), occluded_region_candidates=tuple(candidates),
    )


def _construction(scene_name):
    scene = make_gaussian_reliability_scene(scene_name)
    positions = torch.as_tensor(scene.positions, dtype=torch.float32)
    covariance = torch.as_tensor(scene.covariances, dtype=torch.float32)
    return TorchOSNGSPipeline(TorchPipelineConfig(canonical_construction_max_points=64), device="cpu")._construct_canonical_with_full_evidence(
        positions, covariance, torch.ones(positions.shape[0]), list(range(positions.shape[0])),
    ).construction


class SafeUncertainProposalProductionTest(unittest.TestCase):
    def test_candidate_to_fitted_safe_chart_to_proposal_preserves_source_chain(self):
        bridge = _manual_bridge()
        result = build_safe_uncertain_proposals_from_bridge(bridge, surfaces_by_patch_id={})
        self.assertTrue(result.diagnostic_summary()["all_candidates_accounted"])
        self.assertEqual(len(result.attempts), 1)
        attempt = result.attempts[0]
        self.assertEqual(attempt.status, STATUS_PROPOSED)
        self.assertIsNotNone(attempt.chart)
        self.assertIsNotNone(attempt.safety)
        self.assertIsNotNone(attempt.proposal)
        self.assertGreater(int(attempt.proposal.valid_mask.sum()), 0)
        self.assertEqual(tuple(attempt.chart.supporting_domain_ids), attempt.supporting_domain_ids)
        self.assertEqual(tuple(attempt.chart.supporting_boundary_ids), attempt.supporting_boundary_ids)
        self.assertEqual(tuple(attempt.proposal.metadata["supporting_domain_ids"]), attempt.supporting_domain_ids)
        self.assertEqual(attempt.proposal.append_state, "not_appended")
        self.assertEqual(attempt.proposal.appearance_state, "unset")
        self.assertEqual(attempt.proposal.opacity_state, "unset")

    def test_degenerate_domain_is_typed_rejection_not_phase_f_input(self):
        bridge = _manual_bridge(state_a="degenerate")
        result = build_safe_uncertain_proposals_from_bridge(bridge, surfaces_by_patch_id={})
        self.assertEqual(len(result.attempts), 1)
        attempt = result.attempts[0]
        self.assertEqual(attempt.status, STATUS_DOMAIN_NOT_CANDIDATE_READY)
        self.assertIn("continuation_domain_state:degenerate", attempt.reasons)
        self.assertIsNone(attempt.chart)
        self.assertIsNone(attempt.proposal)

    def test_box_and_thin_slab_candidates_run_through_real_production_phases(self):
        for name, expected_candidates in (("box", 7), ("thin_slab", 3)):
            with self.subTest(scene=name):
                construction = _construction(name)
                bridge = build_eligible_boundary_continuation_bridge(construction)
                self.assertEqual(len(bridge.occluded_region_candidates), expected_candidates)
                surfaces = {
                    int(item.input.source_region_id): item.surface
                    for item in construction.eligible_materialized_surfaces()
                    if item.surface is not None
                }
                result = build_safe_uncertain_proposals_from_bridge(bridge, surfaces_by_patch_id=surfaces)
                self.assertTrue(result.diagnostic_summary()["all_candidates_accounted"])
                self.assertEqual(len(result.attempts), expected_candidates)
                self.assertTrue(all(item.status != STATUS_PROPOSED for item in result.attempts))

    def test_sphere_has_no_candidate_or_proposal_and_no_synthetic_fallback(self):
        construction = _construction("sphere")
        bridge = build_eligible_boundary_continuation_bridge(construction)
        result = build_safe_uncertain_proposals_from_bridge(bridge, surfaces_by_patch_id={})
        self.assertEqual(result.attempts, ())
        self.assertEqual(result.diagnostic_summary()["proposal_sample_count"], 0)
    def test_single_gaussian_evidence_entry_point_preserves_zero_candidate_sphere(self):
        scene = make_gaussian_reliability_scene("sphere")
        result = run_safe_uncertain_proposals_from_gaussians(
            scene.positions, covariance=scene.covariances, stable_ids=tuple(range(len(scene.positions))),
        )
        self.assertEqual(result.candidate_bridge.bridge.occluded_region_candidates, ())
        self.assertEqual(result.production.attempts, ())


if __name__ == "__main__":
    unittest.main()
