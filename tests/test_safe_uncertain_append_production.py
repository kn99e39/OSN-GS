from __future__ import annotations

from unittest import mock
import unittest

import torch

from osn_gs.gaussian.torch_model import TorchGaussianModel
from osn_gs.gaussian.torch_safe_uncertain_append_production import (
    APPENDED, DUPLICATE, REJECTED, ROLLED_BACK, append_safe_uncertain_proposals,
    run_safe_uncertain_proposals_and_append_from_gaussians,
)
from osn_gs.gaussian.torch_uncertain_append_adapter import (
    UncertainAppendInitialization, UncertainGaussianAppendAdapter,
)
from osn_gs.surface.torch_safe_uncertain_proposal_production import build_safe_uncertain_proposals_from_bridge
from tests.test_safe_uncertain_proposal_production import _construction, _manual_bridge
from osn_gs.surface.torch_eligible_boundary_continuation_bridge import build_eligible_boundary_continuation_bridge


def _initialization(batch, _attempt):
    n = len(batch.sample_ids)
    return UncertainAppendInitialization(
        torch.zeros((n, 1, 3)), torch.zeros((n, 3, 3)),
        torch.zeros((n, 1)), torch.full((n, 1), -1.0),
    )


def _safe_result():
    return build_safe_uncertain_proposals_from_bridge(_manual_bridge(), surfaces_by_patch_id={})


def _snapshot(model, adapter):
    tensors = tuple(x.detach().clone() for x in (
        model._xyz, model._features_dc, model._features_rest, model._opacity, model._scaling,
        model._rotation, model._uncertain_confidence, model.is_uncertain, model.surface_uv,
        model.cluster_ids, model.surface_owner_kind, model.surface_owner_id,
    ))
    return tensors, dict(model.occluded_chart_owner_registry), frozenset(model.appended_uncertain_batch_ids), adapter.provenance_sidecar


class SafeUncertainAppendProductionTest(unittest.TestCase):
    def test_candidate_ready_proposal_appends_with_full_adapter_provenance(self):
        safe = _safe_result()
        model, adapter = TorchGaussianModel(sh_degree=1, device="cpu"), UncertainGaussianAppendAdapter()
        result = append_safe_uncertain_proposals(safe, model=model, initialization_provider=_initialization, adapter=adapter)
        attempt = result.attempts[0]
        self.assertEqual(attempt.status, APPENDED)
        self.assertEqual(len(model), attempt.receipt.appended_sample_count)
        sidecar = adapter.provenance_sidecar[attempt.proposal_batch_id]
        self.assertEqual(sidecar["source_candidate_id"], attempt.candidate_id)
        self.assertEqual(sidecar["source_chart_id"], attempt.chart_id)
        self.assertIn(attempt.proposal_batch_id, model.appended_uncertain_batch_ids)
        self.assertTrue(result.diagnostic_summary()["all_candidates_accounted"])

    def test_second_identical_execution_is_duplicate_without_tensor_growth(self):
        safe = _safe_result()
        model, adapter = TorchGaussianModel(sh_degree=1, device="cpu"), UncertainGaussianAppendAdapter()
        first = append_safe_uncertain_proposals(safe, model=model, initialization_provider=_initialization, adapter=adapter)
        count = len(model)
        second = append_safe_uncertain_proposals(safe, model=model, initialization_provider=_initialization, adapter=adapter)
        self.assertEqual(first.attempts[0].status, APPENDED)
        self.assertEqual(second.attempts[0].status, DUPLICATE)
        self.assertEqual(len(model), count)

    def test_missing_initialization_is_typed_rejection_without_synthesis(self):
        safe = _safe_result()
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        result = append_safe_uncertain_proposals(safe, model=model, initialization_provider=None)
        self.assertEqual(result.attempts[0].status, REJECTED)
        self.assertIn("appearance_initialization_required", result.attempts[0].reasons)
        self.assertEqual(len(model), 0)

    def test_injected_transaction_failure_is_rolled_back_everywhere(self):
        safe = _safe_result()
        model, adapter = TorchGaussianModel(sh_degree=1, device="cpu"), UncertainGaussianAppendAdapter()
        before = _snapshot(model, adapter)
        with mock.patch.object(UncertainGaussianAppendAdapter, "_commit_ledger", side_effect=RuntimeError("injected")):
            result = append_safe_uncertain_proposals(safe, model=model, initialization_provider=_initialization, adapter=adapter)
        self.assertEqual(result.attempts[0].status, ROLLED_BACK)
        after = _snapshot(model, adapter)
        for old, new in zip(before[0], after[0]): torch.testing.assert_close(old, new)
        self.assertEqual(before[1:], after[1:])

    def test_rejected_safe_attempt_never_mutates_model(self):
        safe = _safe_result()
        rejected = safe.attempts[0].__class__(
            **{**safe.attempts[0].__dict__, "status": "domain_not_candidate_ready", "proposal": None}
        )
        safe = safe.__class__(safe.bridge, (rejected,))
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        result = append_safe_uncertain_proposals(safe, model=model, initialization_provider=_initialization)
        self.assertEqual(result.attempts[0].status, REJECTED)
        self.assertEqual(len(model), 0)

    def test_box_and_thin_slab_degenerate_candidates_append_nothing(self):
        for name in ("box", "thin_slab"):
            with self.subTest(scene=name):
                construction = _construction(name)
                bridge = build_eligible_boundary_continuation_bridge(construction)
                surfaces = {int(item.input.source_region_id): item.surface for item in construction.eligible_materialized_surfaces() if item.surface is not None}
                safe = build_safe_uncertain_proposals_from_bridge(bridge, surfaces_by_patch_id=surfaces)
                model = TorchGaussianModel(sh_degree=1, device="cpu")
                result = append_safe_uncertain_proposals(safe, model=model, initialization_provider=_initialization)
                self.assertTrue(result.diagnostic_summary()["all_candidates_accounted"])
                self.assertEqual(len(model), 0)
                self.assertTrue(all(item.status == REJECTED for item in result.attempts))

    def test_raw_gaussian_single_entry_point_keeps_sphere_at_zero_append(self):
        from nurbs_constructor_benchmark.gaussian_reliability_scenes import make_gaussian_reliability_scene
        scene = make_gaussian_reliability_scene("sphere")
        model = TorchGaussianModel(sh_degree=1, device="cpu")
        result = run_safe_uncertain_proposals_and_append_from_gaussians(
            scene.positions, covariance=scene.covariances, stable_ids=tuple(range(len(scene.positions))),
            model=model, initialization_provider=_initialization,
        )
        self.assertEqual(result.append_result.attempts, ())
        self.assertEqual(len(model), 0)

if __name__ == "__main__":
    unittest.main()